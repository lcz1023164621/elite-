"""
将 Gazebo 中的 rect_pickup 与 carton_box 注入 MoveIt PlanningScene（/apply_planning_scene），
使 RViz MotionPlanning 插件中「Scene Geometry」可见——与 moveit2_tutorials 中
CollisionObject + PlanningScene 做法一致（参见 moveit2_tutorials 及 pymoveit2 示例）。

参考思路（社区常见做法）：
- ros-planning/moveit2_tutorials: PlanningScene / CollisionObject
- Gazebo Sim + ros_gz_bridge：位姿来自 /model/*/pose，与 URDF base_link 对齐时可直接用 base_link
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import rclpy
import tf2_ros
import yaml
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool
from tf2_msgs.msg import TFMessage

from .gazebo_pose_sync import extract_model_pose

_DEBUG_LOG_PATH = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-cfd510.log")
_DEBUG_SESSION_ID = "cfd510"


def _debug_log(location: str, message: str, hypothesis_id: str, data: dict) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        _DEBUG_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception:
        pass


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
    co.header.frame_id = "world"
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
        box2 = self._cfg.get("box2") or {}
        self._rect_size: list[float] = list(r.get("size_xyz", [0.20, 0.14, 0.08]))
        self._carton_outer = list(cbox.get("outer_size_xyz", [0.42, 0.30, 0.22]))
        self._carton_wall_t = float(cbox.get("wall_thickness", 0.008))
        self._carton_floor_t = float(cbox.get("floor_thickness", 0.006))
        self._box2_outer = list(box2.get("outer_size_xyz", self._carton_outer))
        self._box2_wall_t = float(box2.get("wall_thickness", self._carton_wall_t))
        self._box2_floor_t = float(box2.get("floor_thickness", self._carton_floor_t))
        self._ground_size = [4.0, 4.0]
        self._ground_thickness = 0.02
        mconv = self._cfg.get("middle_conveyor") or {}
        self._conveyor_size: list[float] = list(mconv.get("size_xyz", [1.50, 0.30, 0.20]))
        self._conveyor_pose: list[float] = list(mconv.get("model_pose_xyz", [1.00825, -0.35547, 0.0]))
        self._conveyor_rpy: list[float] = list(mconv.get("model_pose_rpy", [0.0, 0.0, -0.338955]))
        cp = cbox.get("model_pose_xyz", [-0.82, 0.30, 0.0])
        self._carton_fallback = PoseStamped()
        self._carton_fallback.header.frame_id = "base_link"
        self._carton_fallback.pose.position = Point(
            x=float(cp[0]), y=float(cp[1]), z=float(cp[2])
        )
        self._carton_fallback.pose.orientation = _quat_from_rpy(0.0, 0.0, 0.0)

        bp = box2.get("model_pose_xyz", [2.90000, 0.00000, 0.0])
        self._box2_fallback = PoseStamped()
        self._box2_fallback.header.frame_id = "base_link"
        self._box2_fallback.pose.position = Point(
            x=float(bp[0]), y=float(bp[1]), z=float(bp[2])
        )
        self._box2_fallback.pose.orientation = _quat_from_rpy(0.0, 0.0, 0.0)

        rc = r.get("center_xyz", [-0.82, 0.30, 0.046])
        self._rect_fallback = PoseStamped()
        self._rect_fallback.header.frame_id = "base_link"
        self._rect_fallback.pose.position = Point(
            x=float(rc[0]), y=float(rc[1]), z=float(rc[2])
        )
        self._rect_fallback.pose.orientation = _quat_from_rpy(0.0, 0.0, 0.0)

        self._rect: PoseStamped | None = None
        self._carton: PoseStamped | None = None
        self._box2: PoseStamped | None = None
        self._gz_suction_attached = False
        self._assumed_suction_attached = False
        self._prev_suction_attached = False
        self._visual_attached = False
        self._attach_offset: Pose | None = None
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
        self.create_subscription(
            PoseStamped,
            "/model/box2/pose",
            self._on_box2,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TFMessage,
            "/world/arm_world/pose/info",
            self._on_world_pose_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TFMessage,
            "/world/arm_world/dynamic_pose/info",
            self._on_world_pose_info,
            qos_profile_sensor_data,
        )
        self.create_subscription(Bool, "/cs612/suction/state", self._on_suction_state, qos_profile_sensor_data)
        self.create_subscription(Bool, "/cs612/suction/assumed_state", self._on_assumed_state, qos_profile_sensor_data)
        self.create_subscription(
            Bool,
            "/cs612/suction/attached_visual_state",
            self._on_visual_attached,
            qos_profile_sensor_data,
        )
        # cs612_2 独立吸附状态
        self._gz_cs612_2_suction_attached = False
        self._assumed_cs612_2_suction_attached = False
        self._cs612_2_attach_offset: Pose | None = None
        self.create_subscription(Bool, "/cs612_2/suction/state", self._on_cs612_2_suction_state, qos_profile_sensor_data)
        self.create_subscription(Bool, "/cs612_2/suction/assumed_state", self._on_cs612_2_assumed_state, qos_profile_sensor_data)
        self._client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._client_cs612_2 = self.create_client(ApplyPlanningScene, "/cs612_2/apply_planning_scene")
        self._pending = None
        self._pending_cs612_2 = None
        self._pending_rev: int | None = None
        self._dbg_service_wait_logged = False
        self.declare_parameter("scene_update_quantization_m", 0.02)
        # Gazebo 中物体会因接触/推挤产生小位移；规划场景默认持续同步，避免 RViz 与 Gazebo 位置漂移。
        self.declare_parameter("continuous_scene_sync", True)
        self.create_timer(1.5, self._tick)

    @property
    def _suction_attached(self) -> bool:
        return self._gz_suction_attached or self._assumed_suction_attached

    @property
    def _cs612_2_suction_attached(self) -> bool:
        return self._gz_cs612_2_suction_attached or self._assumed_cs612_2_suction_attached

    def _on_rect(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._rect = msg
        if not hasattr(self, "_dbg_rect_logged"):
            self._dbg_rect_logged = True
            # #region agent log
            _debug_log(
                "planning_scene_spawner.py:_on_rect",
                "rect_pose_received",
                "H1",
                {"frame": msg.header.frame_id or "", "x": float(p.x), "y": float(p.y), "z": float(p.z)},
            )
            # #endregion

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._carton = msg
        if not hasattr(self, "_dbg_carton_logged"):
            self._dbg_carton_logged = True
            # #region agent log
            _debug_log(
                "planning_scene_spawner.py:_on_carton",
                "carton_pose_received",
                "H1",
                {"frame": msg.header.frame_id or "", "x": float(p.x), "y": float(p.y), "z": float(p.z)},
            )
            # #endregion

    def _on_box2(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._box2 = msg

    def _on_world_pose_info(self, msg: TFMessage) -> None:
        rect = extract_model_pose(msg, "rect_pickup")
        if rect is not None:
            self._on_rect(rect)
        carton = extract_model_pose(msg, "carton_box")
        if carton is not None:
            self._on_carton(carton)
        box2 = extract_model_pose(msg, "box2")
        if box2 is not None:
            self._on_box2(box2)

    def _on_suction_state(self, msg: Bool) -> None:
        was = self._suction_attached
        self._gz_suction_attached = bool(msg.data)
        if self._suction_attached and not was:
            self._record_attach_offset()

    def _on_assumed_state(self, msg: Bool) -> None:
        # auto_pick_place.py 在 Gazebo state 不可靠时通过 assumed_state 主动声明 attach/detach
        was = self._suction_attached
        self._assumed_suction_attached = bool(msg.data)
        self.get_logger().info(
            f"收到 assumed_state: {self._assumed_suction_attached} (suction_attached 变为 {self._suction_attached})"
        )
        if self._suction_attached and not was:
            self._record_attach_offset()

    def _on_visual_attached(self, msg: Bool) -> None:
        self._visual_attached = bool(msg.data)

    def _on_cs612_2_suction_state(self, msg: Bool) -> None:
        was = self._cs612_2_suction_attached
        self._gz_cs612_2_suction_attached = bool(msg.data)
        if self._cs612_2_suction_attached and not was:
            self._record_cs612_2_attach_offset()

    def _on_cs612_2_assumed_state(self, msg: Bool) -> None:
        was = self._cs612_2_suction_attached
        self._assumed_cs612_2_suction_attached = bool(msg.data)
        self.get_logger().info(
            f"收到 cs612_2 assumed_state: {self._assumed_cs612_2_suction_attached} (suction_attached 变为 {self._cs612_2_suction_attached})"
        )
        if self._cs612_2_suction_attached and not was:
            self._record_cs612_2_attach_offset()

    def _record_cs612_2_attach_offset(self) -> None:
        if self._rect is None:
            self.get_logger().warn("无法记录 cs612_2 attach 偏移: rect_pose 尚未收到")
            return
        try:
            from rclpy.duration import Duration as RclDuration

            tf = self._tf_buffer.lookup_transform(
                "cs612_2_suction_tcp_link",
                "base_link",
                rclpy.time.Time(),
                timeout=RclDuration(seconds=0.5),
            )
            rect_ps = self._effective_rect()
            self._cs612_2_attach_offset = do_transform_pose(rect_ps.pose, tf)
            self.get_logger().info(
                f"记录 cs612_2 attach 偏移: cs612_2_suction_tcp_link 下 ({self._cs612_2_attach_offset.position.x:.4f}, "
                f"{self._cs612_2_attach_offset.position.y:.4f}, {self._cs612_2_attach_offset.position.z:.4f})"
            )
        except Exception as e:
            self.get_logger().warn(f"无法记录 cs612_2 attach 偏移: {e}")

    def _record_attach_offset(self) -> None:
        """记录 attach 瞬间 rect_pickup 相对于 suction_tcp_link 的位姿。"""
        if self._rect is None:
            self.get_logger().warn("无法记录 attach 偏移: rect_pose 尚未收到")
            return
        try:
            from rclpy.duration import Duration as RclDuration

            # 获取 base_link -> suction_tcp_link（即 suction_tcp_link 在 base_link 下的位姿）
            tf = self._tf_buffer.lookup_transform(
                "suction_tcp_link",
                "base_link",
                rclpy.time.Time(),
                timeout=RclDuration(seconds=0.5),
            )
            rect_ps = self._effective_rect()
            self._attach_offset = do_transform_pose(rect_ps.pose, tf)
            self.get_logger().info(
                f"记录 attach 偏移: suction_tcp_link 下 ({self._attach_offset.position.x:.4f}, "
                f"{self._attach_offset.position.y:.4f}, {self._attach_offset.position.z:.4f})"
            )
        except Exception as e:
            self.get_logger().warn(f"无法记录 attach 偏移: {e}")

    def _effective_rect(self) -> PoseStamped:
        ps = self._rect if self._rect is not None else self._rect_fallback
        return self._pose_to_base(ps)

    def _effective_carton(self) -> PoseStamped:
        ps = self._carton if self._carton is not None else self._carton_fallback
        return self._pose_to_base(ps)

    def _effective_box2(self) -> PoseStamped:
        ps = self._box2 if self._box2 is not None else self._box2_fallback
        return self._pose_to_base(ps)

    def _pose_to_base(self, ps: PoseStamped) -> PoseStamped:
        raw = (ps.header.frame_id or "").strip()
        if raw in ("", "world", "arm_world", "map", "base_link", "rect_pickup", "carton_box", "box2"):
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
            out = PoseStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = ps.header.stamp
            out.pose = do_transform_pose(ps.pose, tf)
            return out
        except Exception:
            out = PoseStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = ps.header.stamp
            out.pose = ps.pose
            return out

    def _build_objects(self, for_cs612_2: bool = False) -> list[CollisionObject]:
        rect_ps = self._effective_rect()
        carton_ps = self._effective_carton()
        box2_ps = self._effective_box2()
        rq = rect_ps.pose.orientation
        rp = rect_ps.pose.position
        cq = carton_ps.pose.orientation
        cp = carton_ps.pose.position
        b2q = box2_ps.pose.orientation
        b2p = box2_ps.pose.position

        suction_holding = self._cs612_2_suction_attached if for_cs612_2 else (self._suction_attached or self._visual_attached)

        sx, sy, sz = [float(v) for v in self._rect_size]
        objects: list[CollisionObject] = [
            _make_box(
                "scene_ground",
                Point(x=0.0, y=0.0, z=-0.5 * self._ground_thickness),
                Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                [self._ground_size[0], self._ground_size[1], self._ground_thickness],
            ),
        ]
        if not suction_holding:
            objects.append(_make_box("scene_rect_pickup", Point(x=rp.x, y=rp.y, z=rp.z), rq, [sx, sy, sz]))

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
        b2x, b2y, b2z = self._box2_outer
        b2_wt = self._box2_wall_t
        b2_ft = self._box2_floor_t
        b2_half_x = 0.5 * b2x
        b2_half_y = 0.5 * b2y
        b2_half_h = 0.5 * b2z
        b2_wcx = b2_half_x - 0.5 * b2_wt
        b2_wcy = b2_half_y - 0.5 * b2_wt
        objects.extend(
            [
                _make_box(
                    "scene_box2_floor",
                    _point_off(b2p, b2q, 0.0, 0.0, 0.5 * b2_ft),
                    b2q,
                    [b2x, b2y, b2_ft],
                ),
                _make_box(
                    "scene_box2_wall_px",
                    _point_off(b2p, b2q, b2_wcx, 0.0, b2_half_h),
                    b2q,
                    [b2_wt, b2y, b2z],
                ),
                _make_box(
                    "scene_box2_wall_nx",
                    _point_off(b2p, b2q, -b2_wcx, 0.0, b2_half_h),
                    b2q,
                    [b2_wt, b2y, b2z],
                ),
                _make_box(
                    "scene_box2_wall_py",
                    _point_off(b2p, b2q, 0.0, b2_wcy, b2_half_h),
                    b2q,
                    [b2x, b2_wt, b2z],
                ),
                _make_box(
                    "scene_box2_wall_ny",
                    _point_off(b2p, b2q, 0.0, -b2_wcy, b2_half_h),
                    b2q,
                    [b2x, b2_wt, b2z],
                ),
            ]
        )
        # middle_conveyor 静态碰撞体。视觉 mesh 由 Gazebo/RViz marker 显示；
        # MoveIt 这里只保留薄顶面和较小底座，避免 RViz 中一个实心大盒遮住 IFRA 传送带外观。
        cv_q = _quat_from_rpy(
            float(self._conveyor_rpy[0]),
            float(self._conveyor_rpy[1]),
            float(self._conveyor_rpy[2]),
        )
        cv_pose_z = float(self._conveyor_pose[2])
        cv_sx = float(self._conveyor_size[0])
        cv_sy = float(self._conveyor_size[1])
        cv_sz = float(self._conveyor_size[2])
        top_thickness = min(0.025, max(0.01, 0.10 * cv_sz))
        body_height = max(0.02, cv_sz - top_thickness)
        top_c = Point(
            x=float(self._conveyor_pose[0]),
            y=float(self._conveyor_pose[1]),
            z=cv_pose_z + cv_sz - 0.5 * top_thickness,
        )
        body_c = Point(
            x=float(self._conveyor_pose[0]),
            y=float(self._conveyor_pose[1]),
            z=cv_pose_z + 0.5 * body_height,
        )
        objects.extend(
            [
                _make_box(
                    "scene_middle_conveyor_top",
                    top_c,
                    cv_q,
                    [cv_sx, cv_sy, top_thickness],
                ),
                _make_box(
                    "scene_middle_conveyor_body",
                    body_c,
                    cv_q,
                    [cv_sx, max(0.08, cv_sy * 0.45), body_height],
                ),
            ]
        )
        return objects

    def _build_attached_object(self) -> AttachedCollisionObject | None:
        """构造随机械臂同步的 AttachedCollisionObject（物理吸住时使用）。"""
        if not self._suction_attached:
            return None
        sx, sy, sz = [float(v) for v in self._rect_size]
        aco = AttachedCollisionObject()
        aco.link_name = "suction_tcp_link"
        aco.object = CollisionObject()
        aco.object.id = "scene_rect_pickup"
        aco.object.header.frame_id = "suction_tcp_link"
        aco.object.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [sx, sy, sz]
        aco.object.primitives = [prim]
        if self._attach_offset is not None:
            aco.object.primitive_poses = [self._attach_offset]
        else:
            # 默认近似偏移（suction_tcp_link -> suction_cup_link 为 -0.0095，
            # rect_pickup 半高 0.04，底部贴住 cup）
            aco.object.primitive_poses = [
                Pose(
                    position=Point(x=0.0, y=0.0, z=-0.0495),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ]
        aco.touch_links = ["suction_tcp_link", "suction_cup_link", "ee_link", "tool0"]
        return aco

    def _build_cs612_2_attached_object(self) -> AttachedCollisionObject | None:
        if not self._cs612_2_suction_attached:
            return None
        sx, sy, sz = [float(v) for v in self._rect_size]
        aco = AttachedCollisionObject()
        aco.link_name = "cs612_2_suction_tcp_link"
        aco.object = CollisionObject()
        aco.object.id = "scene_rect_pickup"
        aco.object.header.frame_id = "cs612_2_suction_tcp_link"
        aco.object.operation = CollisionObject.ADD
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [sx, sy, sz]
        aco.object.primitives = [prim]
        if self._cs612_2_attach_offset is not None:
            aco.object.primitive_poses = [self._cs612_2_attach_offset]
        else:
            aco.object.primitive_poses = [
                Pose(
                    position=Point(x=0.0, y=0.0, z=-0.0495),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                )
            ]
        aco.touch_links = ["cs612_2_suction_tcp_link", "cs612_2_suction_cup_link", "cs612_2_ee_link", "cs612_2_tool0"]
        return aco

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
                self._suction_attached,
                self._visual_attached,
                self._cs612_2_suction_attached,
            )
        )

    def _tick(self) -> None:
        # 处理主 move_group pending
        if self._pending is not None:
            if self._pending.done():
                try:
                    res = self._pending.result()
                    if res is not None and res.success and self._pending_rev is not None:
                        self._applied_revision = self._pending_rev
                        self.get_logger().info(
                            "已将 rect_pickup + carton_box + box2 写入 MoveIt PlanningScene（RViz 打开 MotionPlanning → Scene Geometry）。"
                        )
                    else:
                        self.get_logger().warn("apply_planning_scene (主) 返回失败，将重试。")
                except Exception as e:
                    self.get_logger().warn(f"apply_planning_scene (主) 异常: {e}")
                self._pending = None
            # 主 pending 未完成时不阻塞 cs612_2 的发送，fall through

        # 处理 cs612_2 move_group pending
        if self._pending_cs612_2 is not None:
            if self._pending_cs612_2.done():
                try:
                    res = self._pending_cs612_2.result()
                    if res is not None and res.success:
                        self.get_logger().info(
                            "已将 rect_pickup + carton_box + box2 写入 cs612_2 MoveIt PlanningScene。"
                        )
                    else:
                        self.get_logger().warn("apply_planning_scene (cs612_2) 返回失败，将重试。")
                except Exception as e:
                    self.get_logger().warn(f"apply_planning_scene (cs612_2) 异常: {e}")
                self._pending_cs612_2 = None
            # cs612_2 pending 未完成时继续尝试主发送，fall through

        # 如果主 pending 还在进行中，不重复发送（用 revision 去重）
        if self._pending is not None and self._pending_cs612_2 is not None:
            return

        if (
            self._applied_revision is not None
            and not bool(self.get_parameter("continuous_scene_sync").value)
        ):
            return

        rev = self._stable_rev()
        if self._applied_revision is not None and self._applied_revision == rev:
            return

        # 检查主 move_group 服务
        main_ready = self._client.wait_for_service(timeout_sec=0.0)
        if not main_ready:
            if not self._dbg_service_wait_logged:
                self._dbg_service_wait_logged = True
                _debug_log(
                    "planning_scene_spawner.py:_tick",
                    "apply_service_not_ready",
                    "H3",
                    {},
                )
        else:
            self._dbg_service_wait_logged = False

        cs612_2_ready = self._client_cs612_2.wait_for_service(timeout_sec=0.0)

        if not main_ready and not cs612_2_ready:
            return

        # ---- 主 move_group (/apply_planning_scene) ----
        if main_ready and self._pending is None:
            scene = PlanningScene()
            scene.is_diff = True
            scene.world.collision_objects = self._build_objects(for_cs612_2=False)
            attached = self._build_attached_object()
            if attached is not None:
                scene.robot_state.is_diff = True
                scene.robot_state.attached_collision_objects = [attached]
            else:
                scene.robot_state.is_diff = True
                aco_remove = AttachedCollisionObject()
                aco_remove.link_name = "suction_tcp_link"
                aco_remove.object = CollisionObject()
                aco_remove.object.id = "scene_rect_pickup"
                aco_remove.object.operation = CollisionObject.REMOVE
                scene.robot_state.attached_collision_objects = [aco_remove]
            req = ApplyPlanningScene.Request()
            req.scene = scene
            self._pending_rev = rev
            self._pending = self._client.call_async(req)
            _debug_log(
                "planning_scene_spawner.py:_tick",
                "apply_scene_sent",
                "H1",
                {
                    "rev": int(rev),
                    "target": "/apply_planning_scene",
                    "object_count": len(scene.world.collision_objects),
                    "attached": bool(self._suction_attached or self._visual_attached),
                },
            )

        # ---- cs612_2 move_group (/cs612_2/apply_planning_scene) ----
        if cs612_2_ready and self._pending_cs612_2 is None:
            scene2 = PlanningScene()
            scene2.is_diff = True
            scene2.world.collision_objects = self._build_objects(for_cs612_2=True)
            attached2 = self._build_cs612_2_attached_object()
            if attached2 is not None:
                scene2.robot_state.is_diff = True
                scene2.robot_state.attached_collision_objects = [attached2]
            else:
                scene2.robot_state.is_diff = True
                aco_remove2 = AttachedCollisionObject()
                aco_remove2.link_name = "cs612_2_suction_tcp_link"
                aco_remove2.object = CollisionObject()
                aco_remove2.object.id = "scene_rect_pickup"
                aco_remove2.object.operation = CollisionObject.REMOVE
                scene2.robot_state.attached_collision_objects = [aco_remove2]
            req2 = ApplyPlanningScene.Request()
            req2.scene = scene2
            self._pending_cs612_2 = self._client_cs612_2.call_async(req2)
            _debug_log(
                "planning_scene_spawner.py:_tick",
                "apply_scene_sent",
                "H1",
                {
                    "rev": int(rev),
                    "target": "/cs612_2/apply_planning_scene",
                    "object_count": len(scene2.world.collision_objects),
                    "attached": bool(self._cs612_2_suction_attached),
                },
            )


def main() -> None:
    rclpy.init()
    node = PlanningSceneSpawner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # During Ctrl-C shutdown, the Gazebo Pose_V bridge may invalidate an in-flight
        # TFMessage conversion. Do not report that as a node failure.
        if rclpy.ok() and "Unable to convert call argument" not in str(exc):
            raise
    finally:
        try:
            node.destroy_node()
        except BaseException:
            pass
        try:
            rclpy.shutdown()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
