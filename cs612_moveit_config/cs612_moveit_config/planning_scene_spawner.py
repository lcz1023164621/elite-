"""
将 Gazebo 中的 rect_pickup 与 carton_box 注入 MoveIt PlanningScene（/apply_planning_scene），
使 RViz MotionPlanning 插件中「Scene Geometry」可见——与 moveit2_tutorials 中
CollisionObject + PlanningScene 做法一致（参见 moveit2_tutorials 及 pymoveit2 示例）。

参考思路（社区常见做法）：
- ros-planning/moveit2_tutorials: PlanningScene / CollisionObject
- Gazebo Sim + ros_gz_bridge：位姿来自 /model/*/pose，与 URDF base_link 对齐时可直接用 base_link
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import rclpy
import tf2_ros
import yaml
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from shape_msgs.msg import SolidPrimitive


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quaternion:
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return Quaternion(
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
        w=cr * cp * cy + sr * sp * sy,
    )


def _quat_rotate_vec(q: Quaternion, vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    x, y, z = vx, vy, vz
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    fx = x + qw * tx + qy * tz - qz * ty
    fy = y + qw * ty + qz * tx - qx * tz
    fz = z + qw * tz + qx * ty - qy * tx
    return fx, fy, fz


def _load_yaml() -> dict[str, Any]:
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("cs612_moveit_config"))
        p = share / "config" / "scene_objects.yaml"
        if p.is_file():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _make_box(
    object_id: str,
    center: Point,
    orientation: Quaternion,
    size_xyz: Sequence[float],
) -> CollisionObject:
    co = CollisionObject()
    co.id = object_id
    co.header.frame_id = "base_link"
    prim = SolidPrimitive()
    prim.type = SolidPrimitive.BOX
    prim.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]
    pose = Pose(position=center, orientation=orientation)
    co.primitives = [prim]
    co.primitive_poses = [pose]
    co.operation = CollisionObject.ADD
    return co


def _point_off(c: Point, q: Quaternion, ox: float, oy: float, oz: float) -> Point:
    dx, dy, dz = _quat_rotate_vec(q, ox, oy, oz)
    return Point(x=c.x + dx, y=c.y + dy, z=c.z + dz)


class PlanningSceneSpawner(Node):
    """订阅 Gazebo 位姿（或 scene_objects.yaml 回退），向 MoveIt 注入矩形块 + 纸箱碰撞体。"""

    def __init__(self) -> None:
        super().__init__("cs612_planning_scene_spawner")
        self._cfg = _load_yaml()
        r = self._cfg.get("rect_pickup") or {}
        cbox = self._cfg.get("carton_box") or {}
        self._rect_size: list[float] = list(r.get("size_xyz", [0.20, 0.14, 0.08]))
        self._carton_outer = [0.28, 0.22, 0.13]
        self._carton_wall_t = 0.008
        self._carton_floor_t = 0.006
        self._ground_size = [4.0, 4.0]
        self._ground_thickness = 0.02
        cp = cbox.get("model_pose_xyz", [0.82, -0.32, 0.0])
        self._carton_fallback = PoseStamped()
        self._carton_fallback.header.frame_id = "base_link"
        self._carton_fallback.pose.position = Point(
            x=float(cp[0]), y=float(cp[1]), z=float(cp[2])
        )
        self._carton_fallback.pose.orientation = _quat_from_rpy(0.0, 0.0, 0.0)

        rc = r.get("center_xyz", [0.68, 0.16, 0.03])
        self._rect_fallback = PoseStamped()
        self._rect_fallback.header.frame_id = "base_link"
        self._rect_fallback.pose.position = Point(
            x=float(rc[0]), y=float(rc[1]), z=float(rc[2])
        )
        self._rect_fallback.pose.orientation = _quat_from_rpy(0.0, 0.0, 0.0)

        self._rect: PoseStamped | None = None
        self._carton: PoseStamped | None = None
        self._applied_revision: int | None = None
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            PoseStamped,
            "/model/rect_pickup/pose",
            self._on_rect,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            "/model/carton_box/pose",
            self._on_carton,
            qos_profile_sensor_data,
        )
        self._client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._pending = None
        self._pending_rev: int | None = None
        self.declare_parameter("scene_update_quantization_m", 0.02)
        # Gazebo 中物体会因接触/推挤产生小位移；规划场景默认持续同步，避免 RViz 与 Gazebo 位置漂移。
        self.declare_parameter("continuous_scene_sync", True)
        self.create_timer(1.5, self._tick)

    def _on_rect(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._rect = msg

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._carton = msg

    def _effective_rect(self) -> PoseStamped:
        ps = self._rect if self._rect is not None else self._rect_fallback
        return self._pose_to_base(ps)

    def _effective_carton(self) -> PoseStamped:
        ps = self._carton if self._carton is not None else self._carton_fallback
        return self._pose_to_base(ps)

    def _pose_to_base(self, ps: PoseStamped) -> PoseStamped:
        raw = (ps.header.frame_id or "").strip()
        if raw in ("", "world", "arm_world", "map", "base_link", "rect_pickup", "carton_box"):
            out = PoseStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = ps.header.stamp
            out.pose = ps.pose
            return out
        try:
            from rclpy.duration import Duration as RclDuration

            tf = self._tf_buffer.lookup_transform(
                "base_link",
                raw,
                rclpy.time.Time(),
                timeout=RclDuration(seconds=0.2),
            )
            return do_transform_pose(ps, tf)
        except Exception:
            out = PoseStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = ps.header.stamp
            out.pose = ps.pose
            return out

    def _build_objects(self) -> list[CollisionObject]:
        rect_ps = self._effective_rect()
        carton_ps = self._effective_carton()
        rq = rect_ps.pose.orientation
        rp = rect_ps.pose.position
        cq = carton_ps.pose.orientation
        cp = carton_ps.pose.position

        sx, sy, sz = [float(v) for v in self._rect_size]
        objects: list[CollisionObject] = [
            _make_box(
                "scene_ground",
                Point(x=0.0, y=0.0, z=-0.5 * self._ground_thickness),
                Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                [self._ground_size[0], self._ground_size[1], self._ground_thickness],
            ),
            _make_box("scene_rect_pickup", Point(x=rp.x, y=rp.y, z=rp.z), rq, [sx, sy, sz]),
        ]

        bx, by, bz = self._carton_outer
        wt = self._carton_wall_t
        ft = self._carton_floor_t
        half_x = 0.5 * bx
        half_y = 0.5 * by
        half_h = 0.5 * bz
        wcx = half_x - 0.5 * wt
        wcy = half_y - 0.5 * wt
        objects.extend(
            [
                _make_box(
                    "scene_carton_floor",
                    _point_off(cp, cq, 0.0, 0.0, 0.5 * ft),
                    cq,
                    [bx, by, ft],
                ),
                _make_box(
                    "scene_carton_wall_px",
                    _point_off(cp, cq, wcx, 0.0, half_h),
                    cq,
                    [wt, by, bz],
                ),
                _make_box(
                    "scene_carton_wall_nx",
                    _point_off(cp, cq, -wcx, 0.0, half_h),
                    cq,
                    [wt, by, bz],
                ),
                _make_box(
                    "scene_carton_wall_py",
                    _point_off(cp, cq, 0.0, wcy, half_h),
                    cq,
                    [bx, wt, bz],
                ),
                _make_box(
                    "scene_carton_wall_ny",
                    _point_off(cp, cq, 0.0, -wcy, half_h),
                    cq,
                    [bx, wt, bz],
                ),
            ]
        )
        return objects

    def _stable_rev(self) -> int:
        """仿真位姿有微小抖动，用厘米级量化避免无意义重复 apply。"""
        r = self._effective_rect().pose.position
        c = self._effective_carton().pose.position
        q = max(0.005, float(self.get_parameter("scene_update_quantization_m").value))
        return hash(
            (
                round(r.x / q),
                round(r.y / q),
                round(r.z / q),
                round(c.x / q),
                round(c.y / q),
                round(c.z / q),
            )
        )

    def _tick(self) -> None:
        if self._pending is not None:
            if self._pending.done():
                try:
                    res = self._pending.result()
                    if res is not None and res.success and self._pending_rev is not None:
                        self._applied_revision = self._pending_rev
                        self.get_logger().info(
                            "已将 rect_pickup + carton_box 写入 MoveIt PlanningScene（RViz 打开 MotionPlanning → Scene Geometry）。"
                        )
                    else:
                        self.get_logger().warn("apply_planning_scene 返回失败，将重试。")
                except Exception as e:
                    self.get_logger().warn(f"apply_planning_scene 异常: {e}")
                self._pending = None
                self._pending_rev = None
            return

        if (
            self._applied_revision is not None
            and not bool(self.get_parameter("continuous_scene_sync").value)
        ):
            return

        if not self._client.wait_for_service(timeout_sec=0.0):
            return
        rev = self._stable_rev()
        if self._applied_revision is not None and self._applied_revision == rev:
            return

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = self._build_objects()
        req = ApplyPlanningScene.Request()
        req.scene = scene
        self._pending_rev = rev
        self._pending = self._client.call_async(req)


def main() -> None:
    rclpy.init()
    node = PlanningSceneSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
