"""在 RViz 中显示抓取物体和纸箱（来自 Gazebo pose 话题）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ament_index_python.packages import get_package_share_directory
import rclpy
import yaml
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from .gazebo_pose_sync import extract_model_pose


def _load_rect_fallback_from_share() -> tuple[list[float], list[float]] | None:
    """与 worlds/my_world.sdf 同步的默认位姿（config/scene_objects.yaml）。"""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("cs612_moveit_config"))
        cfg_path = share / "config" / "scene_objects.yaml"
        if not cfg_path.is_file():
            return None
        cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        r = cfg.get("rect_pickup") or {}
        c = r.get("center_xyz")
        s = r.get("size_xyz")
        if isinstance(c, list) and len(c) == 3 and isinstance(s, list) and len(s) == 3:
            return [float(c[0]), float(c[1]), float(c[2])], [float(s[0]), float(s[1]), float(s[2])]
    except Exception:
        pass
    return None


def _load_carton_fallback_from_share() -> list[float] | None:
    """与 worlds/my_world.sdf 同步的 carton_box 默认位姿。"""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("cs612_moveit_config"))
        cfg_path = share / "config" / "scene_objects.yaml"
        if not cfg_path.is_file():
            return None
        cfg: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cbox = cfg.get("carton_box") or {}
        pose = cbox.get("model_pose_xyz")
        if isinstance(pose, list) and len(pose) == 3:
            return [float(pose[0]), float(pose[1]), float(pose[2])]
    except Exception:
        pass
    return None


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


class WorldMarkersNode(Node):
    def __init__(self) -> None:
        super().__init__("cs612_world_markers")
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("rect_size_xyz", [0.20, 0.14, 0.08])
        self.declare_parameter("approach_clearance", 0.06)
        self.declare_parameter("suction_contact_offset_z", 0.214)
        self.declare_parameter("carton_outer_size_xyz", [0.42, 0.30, 0.22])
        self.declare_parameter("carton_wall_thickness", 0.008)
        self.declare_parameter("carton_floor_thickness", 0.006)
        self.declare_parameter("carton_floor_top_z", 0.006)
        self.declare_parameter("place_height_above_floor", 0.18)
        self.declare_parameter("carton_fallback_pose_xyz", _load_carton_fallback_from_share() or [-0.82, 0.30, 0.0])
        self.declare_parameter("use_scene_yaml_fallback", True)
        self.declare_parameter("middle_conveyor_size_xyz", [1.50, 0.30, 0.20])
        self.declare_parameter("middle_conveyor_pose_xyz", [1.00825, -0.35547, 0.0])
        self.declare_parameter("middle_conveyor_pose_rpy", [0.0, 0.0, -0.338955])
        self.declare_parameter("conveyor_place_inset_margin_m", 0.04)
        self.declare_parameter("box_half_size_xyz", [0.10, 0.07, 0.04])

        self._rect: PoseStamped | None = None
        self._carton: PoseStamped | None = None

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
        # depth 过小 + DDS 缓冲不足时 RViz 可能收不到 Marker（表现为物体/纸箱“消失”）
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(MarkerArray, "/cs612/world_markers", marker_qos)
        self._rect_pub = self.create_publisher(Marker, "/cs612/rect_pickup_marker", marker_qos)

        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        self.create_timer(1.0 / hz, self._on_timer)

        self._rect_fallback_center: list[float] | None = None
        self._rect_fallback_size: list[float] | None = None
        fb = _load_rect_fallback_from_share()
        if fb is not None:
            self._rect_fallback_center, self._rect_fallback_size = fb
            self.get_logger().info(
                f"rect_pickup 回退位姿已加载（无 Gazebo 桥接时仍显示）: center={self._rect_fallback_center}"
            )

    def _on_rect(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        # 仅丢弃桥接首帧/异常全零位姿；勿用「近原点」过滤，否则会误伤合法工作区靠近基座的物体。
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._rect = msg
        if not hasattr(self, "_rect_logged"):
            self._rect_logged = True
            self.get_logger().info(
                f"WorldMarkers 已收到 rect_pickup: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._carton = msg

    def _on_world_pose_info(self, msg: TFMessage) -> None:
        rect = extract_model_pose(msg, "rect_pickup")
        if rect is not None:
            self._on_rect(rect)
        carton = extract_model_pose(msg, "carton_box")
        if carton is not None:
            self._on_carton(carton)

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

    def _point_with_local_offset(
        self, origin: Point, q: Quaternion, ox: float, oy: float, oz: float
    ) -> Point:
        dx, dy, dz = _quat_rotate_vec(q, ox, oy, oz)
        return Point(x=origin.x + dx, y=origin.y + dy, z=origin.z + dz)

    def _fallback_carton_pose(self) -> PoseStamped:
        xyz = list(self.get_parameter("carton_fallback_pose_xyz").value)
        if len(xyz) != 3:
            xyz = [-0.82, 0.30, 0.0]
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return ps

    def _marker_cube(
        self,
        marker_id: int,
        ns: str,
        stamp,
        center: Point,
        orientation: Quaternion,
        size_xyz: Sequence[float],
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.CUBE
        m.action = Marker.ADD
        m.pose = Pose(position=center, orientation=orientation)
        m.scale.x = float(size_xyz[0])
        m.scale.y = float(size_xyz[1])
        m.scale.z = float(size_xyz[2])
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        return m

    def _marker_delete(self, marker_id: int, ns: str, stamp) -> Marker:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.action = Marker.DELETE
        return m

    def _marker_mesh(
        self,
        marker_id: int,
        ns: str,
        stamp,
        origin: Point,
        orientation: Quaternion,
        scale_xyz: Sequence[float],
        mesh_resource: str,
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.MESH_RESOURCE
        m.action = Marker.ADD
        m.pose = Pose(position=origin, orientation=orientation)
        m.scale.x = float(scale_xyz[0])
        m.scale.y = float(scale_xyz[1])
        m.scale.z = float(scale_xyz[2])
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        m.mesh_resource = mesh_resource
        m.mesh_use_embedded_materials = True
        return m

    def _marker_sphere(
        self,
        marker_id: int,
        ns: str,
        stamp,
        center: Point,
        diameter: float,
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = center
        m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        m.scale.x = float(diameter)
        m.scale.y = float(diameter)
        m.scale.z = float(diameter)
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        return m

    def _marker_text(
        self,
        marker_id: int,
        ns: str,
        stamp,
        center: Point,
        text: str,
        height: float,
        rgba: tuple[float, float, float, float],
    ) -> Marker:
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.ns = ns
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position = center
        m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        m.scale.z = float(height)
        m.color.r = float(rgba[0])
        m.color.g = float(rgba[1])
        m.color.b = float(rgba[2])
        m.color.a = float(rgba[3])
        m.text = text
        return m

    def _on_timer(self) -> None:
        now = self.get_clock().now().to_msg()
        out = MarkerArray()
        rect_size = list(self.get_parameter("rect_size_xyz").value)
        if len(rect_size) != 3:
            rect_size = [0.20, 0.14, 0.08]
        use_fb = bool(self.get_parameter("use_scene_yaml_fallback").value)
        rect_src: PoseStamped | None = self._rect
        if (
            use_fb
            and rect_src is None
            and self._rect_fallback_center is not None
            and self._rect_fallback_size is not None
        ):
            rect_size = list(self._rect_fallback_size)
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = now
            c = self._rect_fallback_center
            ps.pose.position = Point(x=float(c[0]), y=float(c[1]), z=float(c[2]))
            ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            rect_src = ps
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        approach_clearance = float(self.get_parameter("approach_clearance").value)
        floor_top_z = float(self.get_parameter("carton_floor_top_z").value)
        place_height = float(self.get_parameter("place_height_above_floor").value)

        # 物体（优先 Gazebo 位姿；无桥接时用 scene_objects.yaml 回退，不写入 self._rect 以免挡住后续订阅）
        if rect_src is None:
            out.markers.append(self._marker_delete(1, "rect_pickup", now))
            out.markers.append(self._marker_delete(2, "rect_pickup", now))
            out.markers.append(self._marker_delete(3, "rect_pickup", now))
            out.markers.append(self._marker_delete(4, "rect_pickup", now))
            self._rect_pub.publish(self._marker_delete(100, "rect_pickup_single", now))
        else:
            rect = self._pose_to_base(rect_src)
            out.markers.append(
                self._marker_cube(
                    1,
                    "rect_pickup",
                    now,
                    rect.pose.position,
                    rect.pose.orientation,
                    rect_size,
                    (0.95, 0.52, 0.22, 0.95),
                )
            )
            self._rect_pub.publish(
                self._marker_cube(
                    100,
                    "rect_pickup_single",
                    now,
                    rect.pose.position,
                    rect.pose.orientation,
                    rect_size,
                    (1.0, 0.40, 0.05, 0.98),
                )
            )
            rect_top = self._point_with_local_offset(
                rect.pose.position,
                rect.pose.orientation,
                0.0,
                0.0,
                0.5 * float(rect_size[2]),
            )
            pick_touch = Point(
                x=rect_top.x,
                y=rect_top.y,
                z=rect_top.z + suction_contact_offset,
            )
            approach = Point(
                x=pick_touch.x,
                y=pick_touch.y,
                z=pick_touch.z + approach_clearance,
            )
            out.markers.append(
                self._marker_sphere(2, "rect_pickup", now, rect_top, 0.035, (1.0, 0.9, 0.1, 0.95))
            )
            out.markers.append(
                self._marker_sphere(3, "rect_pickup", now, approach, 0.03, (0.1, 0.8, 1.0, 0.90))
            )
            out.markers.append(
                self._marker_text(
                    4,
                    "rect_pickup",
                    now,
                    Point(x=rect.pose.position.x, y=rect.pose.position.y, z=rect_top.z + 0.08),
                    "rect_pickup / grasp center",
                    0.05,
                    (0.1, 0.1, 0.1, 0.95),
                )
            )

        # 纸箱（底板+四壁）
        carton = self._pose_to_base(self._carton) if self._carton is not None else self._fallback_carton_pose()
        c = carton.pose.position
        q = carton.pose.orientation
        sx, sy, sz = [float(v) for v in self.get_parameter("carton_outer_size_xyz").value]
        wall_t = float(self.get_parameter("carton_wall_thickness").value)
        floor_t = float(self.get_parameter("carton_floor_thickness").value)
        if sx <= 0.0 or sy <= 0.0 or sz <= 0.0 or wall_t <= 0.0 or floor_t <= 0.0:
            sx, sy, sz, wall_t, floor_t = 0.42, 0.30, 0.22, 0.008, 0.006

        half_x = 0.5 * sx
        half_y = 0.5 * sy
        half_h = 0.5 * sz
        wall_cx = half_x - 0.5 * wall_t
        wall_cy = half_y - 0.5 * wall_t
        color = (0.25, 0.85, 0.85, 0.52)

        out.markers.append(
            self._marker_cube(
                10,
                "carton",
                now,
                self._point_with_local_offset(c, q, 0.0, 0.0, 0.5 * floor_t),
                q,
                [sx, sy, floor_t],
                color,
            )
        )
        out.markers.append(
            self._marker_cube(
                11,
                "carton",
                now,
                self._point_with_local_offset(c, q, wall_cx, 0.0, half_h),
                q,
                [wall_t, sy, sz],
                color,
            )
        )
        out.markers.append(
            self._marker_cube(
                12,
                "carton",
                now,
                self._point_with_local_offset(c, q, -wall_cx, 0.0, half_h),
                q,
                [wall_t, sy, sz],
                color,
            )
        )
        out.markers.append(
            self._marker_cube(
                13,
                "carton",
                now,
                self._point_with_local_offset(c, q, 0.0, wall_cy, half_h),
                q,
                [sx, wall_t, sz],
                color,
            )
        )
        out.markers.append(
            self._marker_cube(
                14,
                "carton",
                now,
                self._point_with_local_offset(c, q, 0.0, -wall_cy, half_h),
                q,
                [sx, wall_t, sz],
                color,
            )
        )
        place_center = self._point_with_local_offset(c, q, 0.0, 0.0, floor_top_z + place_height)
        out.markers.append(
            self._marker_sphere(15, "carton", now, place_center, 0.04, (0.15, 0.95, 0.25, 0.95))
        )
        out.markers.append(
            self._marker_text(
                16,
                "carton",
                now,
                Point(x=place_center.x, y=place_center.y, z=place_center.z + 0.07),
                "carton center / place target",
                0.05,
                (0.05, 0.2, 0.05, 0.95),
            )
        )

        # middle_conveyor marker
        try:
            cv_size = list(self.get_parameter("middle_conveyor_size_xyz").value)
            cv_pose = list(self.get_parameter("middle_conveyor_pose_xyz").value)
            cv_rpy = list(self.get_parameter("middle_conveyor_pose_rpy").value)
            if len(cv_size) == 3 and len(cv_pose) == 3 and len(cv_rpy) == 3:
                import math
                cy, sy = math.cos(cv_rpy[2] * 0.5), math.sin(cv_rpy[2] * 0.5)
                cp, sp = math.cos(cv_rpy[1] * 0.5), math.sin(cv_rpy[1] * 0.5)
                cr, sr = math.cos(cv_rpy[0] * 0.5), math.sin(cv_rpy[0] * 0.5)
                cq = Quaternion(
                    x=sr * cp * cy - cr * sp * sy,
                    y=cr * sp * cy + sr * cp * sy,
                    z=cr * cp * sy - sr * sp * cy,
                    w=cr * cp * cy + sr * sp * sy,
                )
                cc = Point(
                    x=float(cv_pose[0]),
                    y=float(cv_pose[1]),
                    z=float(cv_pose[2]) + 0.5 * float(cv_size[2]),
                )
                try:
                    share = Path(get_package_share_directory("cs612_moveit_config"))
                    package_mesh = share / "models" / "middle_conveyor" / "meshes" / "conveyor_belt.dae"
                    source_mesh = Path.cwd() / "models" / "middle_conveyor" / "meshes" / "conveyor_belt.dae"
                    if package_mesh.is_file():
                        mesh_resource = "package://cs612_moveit_config/models/middle_conveyor/meshes/conveyor_belt.dae"
                    elif source_mesh.is_file():
                        mesh_resource = "file://" + str(source_mesh.resolve())
                    else:
                        raise FileNotFoundError(str(package_mesh))
                    if not hasattr(self, "_middle_conveyor_mesh_logged"):
                        self._middle_conveyor_mesh_logged = True
                        self.get_logger().info(f"middle_conveyor RViz mesh: {mesh_resource}")
                    mesh_yaw = float(cv_rpy[2]) - 1.57079632679
                    cy_m, sy_m = math.cos(mesh_yaw * 0.5), math.sin(mesh_yaw * 0.5)
                    mesh_q = Quaternion(x=0.0, y=0.0, z=sy_m, w=cy_m)
                    out.markers.append(
                        self._marker_mesh(
                            20,
                            "middle_conveyor",
                            now,
                            Point(x=float(cv_pose[0]), y=float(cv_pose[1]), z=float(cv_pose[2])),
                            mesh_q,
                            [0.64794816415, 1.25, 0.26917900404],
                            mesh_resource,
                            (1.0, 1.0, 1.0, 1.0),
                        )
                    )
                except Exception as exc:
                    if not hasattr(self, "_middle_conveyor_mesh_warned"):
                        self._middle_conveyor_mesh_warned = True
                        self.get_logger().warn(f"middle_conveyor mesh 不可用，回退为盒体 marker: {exc}")
                    out.markers.append(
                        self._marker_cube(
                            20, "middle_conveyor", now, cc, cq, cv_size, (0.35, 0.38, 0.42, 0.35)
                        )
                    )
                # 传送带放置中心：物体前缘对齐传送带起点，中心向内偏移半个物体长度。
                start_offset = 0.5 * float(cv_size[0])
                cs = math.cos(cv_rpy[2])
                sn = math.sin(cv_rpy[2])
                edge_pt = Point(
                    x=cc.x - cs * start_offset,
                    y=cc.y - sn * start_offset,
                    z=float(cv_pose[2]) + float(cv_size[2]) + 0.03,
                )
                box_half = list(self.get_parameter("box_half_size_xyz").value)
                inset = (float(box_half[0]) if len(box_half) > 0 else 0.10) + max(
                    0.0, float(self.get_parameter("conveyor_place_inset_margin_m").value)
                )
                start_pt = Point(
                    x=edge_pt.x + cs * inset,
                    y=edge_pt.y + sn * inset,
                    z=edge_pt.z,
                )
                end_center = Point(
                    x=cc.x + cs * (start_offset - inset),
                    y=cc.y + sn * (start_offset - inset),
                    z=edge_pt.z,
                )
                out.markers.append(
                    self._marker_sphere(21, "middle_conveyor", now, start_pt, 0.04, (0.95, 0.25, 0.25, 0.95))
                )
                out.markers.append(
                    self._marker_text(
                        22,
                        "middle_conveyor",
                        now,
                        Point(x=start_pt.x, y=start_pt.y, z=start_pt.z + 0.06),
                        "conveyor start / place target",
                        0.05,
                        (0.2, 0.05, 0.05, 0.95),
                    )
                )
                out.markers.append(
                    self._marker_sphere(23, "middle_conveyor", now, end_center, 0.04, (0.20, 0.55, 0.95, 0.95))
                )
                out.markers.append(
                    self._marker_text(
                        24,
                        "middle_conveyor",
                        now,
                        Point(x=end_center.x, y=end_center.y, z=end_center.z + 0.06),
                        "conveyor end",
                        0.05,
                        (0.05, 0.12, 0.28, 0.95),
                    )
                )
        except Exception:
            pass

        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = WorldMarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        # ros_gz_bridge Pose_V -> TFMessage can race with shutdown and throw from rclpy
        # while converting a message. Treat that as a clean shutdown path.
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
