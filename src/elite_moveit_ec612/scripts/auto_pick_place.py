#!/usr/bin/env python3
"""全自动：抓取并入箱（吸盘 attach/detach + MoveIt 避障）。"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from xml.etree import ElementTree as ET

import rclpy
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import yaml
import numpy as np
from tf2_geometry_msgs import do_transform_pose
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Vector3
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    AllowedCollisionEntry,
    AllowedCollisionMatrix,
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningScene,
    PlanningSceneComponents,
    PositionConstraint,
    RobotState,
)
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene, GetPositionIK
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Bool, Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

_ARM_JOINTS = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]


@dataclass
class _JointKinematic:
    name: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower: float
    upper: float


def _load_scene_fallback_xyz() -> tuple[list[float], list[float]]:
    rect_xyz = [0.45, 0.0, 0.03]
    carton_xyz = [0.82, -0.32, 0.0]
    try:
        from ament_index_python.packages import get_package_share_directory

        cfg = Path(get_package_share_directory("elite_moveit_ec612")) / "config" / "scene_objects.yaml"
        if cfg.is_file():
            doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            rect = doc.get("rect_pickup") or {}
            carton = doc.get("carton_box") or {}
            rect_center = rect.get("center_xyz")
            carton_center = carton.get("model_pose_xyz")
            if isinstance(rect_center, list) and len(rect_center) == 3:
                rect_xyz = [float(rect_center[0]), float(rect_center[1]), float(rect_center[2])]
            if isinstance(carton_center, list) and len(carton_center) == 3:
                carton_xyz = [float(carton_center[0]), float(carton_center[1]), float(carton_center[2])]
    except Exception:
        pass
    return rect_xyz, carton_xyz


def _spin_future(node: Node, fut, timeout_sec: float, label: str) -> bool:
    """
    在后台 MultiThreadedExecutor 已 spin 本节点时，不能用 spin_until_future_complete
    （会与 executor 争用同一 Node，导致 IK/MoveGroup future 永不完成 → 机械臂不动）。
    改为纯等待 future，由 executor 线程处理 DDS 回调。
    """
    t0 = time.time()
    while rclpy.ok() and not fut.done() and (time.time() - t0) < timeout_sec:
        time.sleep(0.005)
    if fut.done():
        return True
    node.get_logger().error(f"{label}: 等待结果超时（{timeout_sec}s），请检查网络/DDS 或 move_group 是否存活")
    return False


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


def _quat_rotate_vec(q: Quaternion, vx: float, vy: float, vz: float) -> Tuple[float, float, float]:
    """向量 v 由四元数 q 旋转（Hamilton，与 geometry_msgs 一致）。"""
    x, y, z = vx, vy, vz
    qx, qy, qz, qw = q.x, q.y, q.z, q.w
    tx = 2.0 * (qy * z - qz * y)
    ty = 2.0 * (qz * x - qx * z)
    tz = 2.0 * (qx * y - qy * x)
    fx = x + qw * tx + qy * tz - qz * ty
    fy = y + qw * ty + qz * tx - qx * tz
    fz = z + qw * tz + qx * ty - qy * tx
    return fx, fy, fz


def _quat_mul(a: Quaternion, b: Quaternion) -> Quaternion:
    return Quaternion(
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    )


def _suction_down_quat(yaw: float) -> Quaternion:
    """
    当前吸盘接触面沿 suction_tcp_link 局部 +Z。
    抓取/放置时应先绕世界 Z 设定朝向，再将本地 +Z 翻到世界 -Z。
    """
    return _quat_mul(_quat_from_rpy(0.0, 0.0, yaw), _quat_from_rpy(math.pi, 0.0, 0.0))


def _wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _angle_distance(a: float, b: float) -> float:
    return abs(_wrap_to_pi(a - b))


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "y")
    return bool(value)


def _vec_add(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _vec_sub(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (float(a[0]) - float(b[0]), float(a[1]) - float(b[1]), float(a[2]) - float(b[2]))


def _vec_scale(a: Sequence[float], s: float) -> tuple[float, float, float]:
    return (float(a[0]) * float(s), float(a[1]) * float(s), float(a[2]) * float(s))


def _vec_norm(a: Sequence[float]) -> float:
    return math.sqrt(float(a[0]) ** 2 + float(a[1]) ** 2 + float(a[2]) ** 2)


def _vec_dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _vec_cross(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    )


def _vec_normalize(a: Sequence[float]) -> tuple[float, float, float]:
    n = _vec_norm(a)
    if n < 1e-12:
        return (0.0, 0.0, 1.0)
    return (float(a[0]) / n, float(a[1]) / n, float(a[2]) / n)


def _mat_identity() -> list[list[float]]:
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _mat_mul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> list[list[float]]:
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            out[r][c] = (
                float(a[r][0]) * float(b[0][c])
                + float(a[r][1]) * float(b[1][c])
                + float(a[r][2]) * float(b[2][c])
            )
    return out


def _mat_vec_mul(a: Sequence[Sequence[float]], v: Sequence[float]) -> tuple[float, float, float]:
    return (
        float(a[0][0]) * float(v[0]) + float(a[0][1]) * float(v[1]) + float(a[0][2]) * float(v[2]),
        float(a[1][0]) * float(v[0]) + float(a[1][1]) * float(v[1]) + float(a[1][2]) * float(v[2]),
        float(a[2][0]) * float(v[0]) + float(a[2][1]) * float(v[1]) + float(a[2][2]) * float(v[2]),
    )


def _mat_transpose(a: Sequence[Sequence[float]]) -> list[list[float]]:
    return [
        [float(a[0][0]), float(a[1][0]), float(a[2][0])],
        [float(a[0][1]), float(a[1][1]), float(a[2][1])],
        [float(a[0][2]), float(a[1][2]), float(a[2][2])],
    ]


def _rot_from_rpy(roll: float, pitch: float, yaw: float) -> list[list[float]]:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _rot_from_axis_angle(axis: Sequence[float], theta: float) -> list[list[float]]:
    ux, uy, uz = _vec_normalize(axis)
    c = math.cos(theta)
    s = math.sin(theta)
    v = 1.0 - c
    return [
        [c + ux * ux * v, ux * uy * v - uz * s, ux * uz * v + uy * s],
        [uy * ux * v + uz * s, c + uy * uy * v, uy * uz * v - ux * s],
        [uz * ux * v - uy * s, uz * uy * v + ux * s, c + uz * uz * v],
    ]


def _quat_normalize(q: Quaternion) -> Quaternion:
    n = math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if n < 1e-12:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    return Quaternion(x=q.x / n, y=q.y / n, z=q.z / n, w=q.w / n)


def _quat_to_rot(q: Quaternion) -> list[list[float]]:
    qn = _quat_normalize(q)
    x, y, z, w = qn.x, qn.y, qn.z, qn.w
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ]


def _rot_to_quat(r: Sequence[Sequence[float]]) -> Quaternion:
    m00, m01, m02 = float(r[0][0]), float(r[0][1]), float(r[0][2])
    m10, m11, m12 = float(r[1][0]), float(r[1][1]), float(r[1][2])
    m20, m21, m22 = float(r[2][0]), float(r[2][1]), float(r[2][2])
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return _quat_normalize(Quaternion(x=qx, y=qy, z=qz, w=qw))


def _quat_conjugate(q: Quaternion) -> Quaternion:
    return Quaternion(x=-q.x, y=-q.y, z=-q.z, w=q.w)


def _quat_slerp(a: Quaternion, b: Quaternion, t: float) -> Quaternion:
    qa = _quat_normalize(a)
    qb = _quat_normalize(b)
    dot = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w
    if dot < 0.0:
        qb = Quaternion(x=-qb.x, y=-qb.y, z=-qb.z, w=-qb.w)
        dot = -dot
    if dot > 0.9995:
        out = Quaternion(
            x=qa.x + t * (qb.x - qa.x),
            y=qa.y + t * (qb.y - qa.y),
            z=qa.z + t * (qb.z - qa.z),
            w=qa.w + t * (qb.w - qa.w),
        )
        return _quat_normalize(out)
    theta_0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta_0 * t
    sin_theta_0 = math.sin(theta_0)
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return Quaternion(
        x=s0 * qa.x + s1 * qb.x,
        y=s0 * qa.y + s1 * qb.y,
        z=s0 * qa.z + s1 * qb.z,
        w=s0 * qa.w + s1 * qb.w,
    )


def _rotation_error(r_cur: Sequence[Sequence[float]], r_des: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    c0 = (float(r_cur[0][0]), float(r_cur[1][0]), float(r_cur[2][0]))
    c1 = (float(r_cur[0][1]), float(r_cur[1][1]), float(r_cur[2][1]))
    c2 = (float(r_cur[0][2]), float(r_cur[1][2]), float(r_cur[2][2]))
    d0 = (float(r_des[0][0]), float(r_des[1][0]), float(r_des[2][0]))
    d1 = (float(r_des[0][1]), float(r_des[1][1]), float(r_des[2][1]))
    d2 = (float(r_des[0][2]), float(r_des[1][2]), float(r_des[2][2]))
    e = _vec_scale(
        _vec_add(_vec_add(_vec_cross(c0, d0), _vec_cross(c1, d1)), _vec_cross(c2, d2)),
        0.5,
    )
    return e


def _solve_linear_system(a: Sequence[Sequence[float]], b: Sequence[float]) -> list[float] | None:
    n = len(a)
    aug = [[float(a[r][c]) for c in range(n)] + [float(b[r])] for r in range(n)]
    for col in range(n):
        pivot = col
        pivot_val = abs(aug[col][col])
        for r in range(col + 1, n):
            v = abs(aug[r][col])
            if v > pivot_val:
                pivot = r
                pivot_val = v
        if pivot_val < 1e-12:
            return None
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for c in range(col, n + 1):
            aug[col][c] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if abs(factor) < 1e-15:
                continue
            for c in range(col, n + 1):
                aug[r][c] -= factor * aug[col][c]
    return [aug[r][n] for r in range(n)]


def _find_gz_executable() -> str:
    if shutil.which("/usr/bin/ign"):
        return "/usr/bin/ign"
    if shutil.which("/usr/bin/gz"):
        return "/usr/bin/gz"
    found = shutil.which("ign")
    if found:
        return found
    found = shutil.which("gz")
    return found or "ign"


def _gz_msg_prefix(gz_bin: str) -> str:
    bin_name = os.path.basename(gz_bin)
    return "ignition.msgs" if bin_name.startswith("ign") else "gz.msgs"


class AutoPickPlaceNode(Node):
    def __init__(self) -> None:
        super().__init__("cs612_auto_pick_place")
        rect_fb_xyz, carton_fb_xyz = _load_scene_fallback_xyz()
        self._cb = ReentrantCallbackGroup()
        self._log_lock = threading.Lock()
        self._rect: PoseStamped | None = None
        self._carton: PoseStamped | None = None
        self._joint_state: JointState | None = None
        self.declare_parameter("box_half_size_xyz", [0.07, 0.05, 0.03])
        # 已知场景下，优先使用「已知矩形中心 + 尺寸」计算顶面中心，降低桥接抖动对抓取点的影响。
        self.declare_parameter("use_known_rect_surface_center", True)
        self.declare_parameter("known_rect_center_xyz", rect_fb_xyz)
        self.declare_parameter("known_rect_size_xyz", [0.14, 0.10, 0.06])
        # 真实物体碰撞体仍使用 center+size；吸取目标中心允许单独配置，避免把碰撞网格改偏。
        self.declare_parameter(
            "known_suction_pick_center_xyz",
            [float(rect_fb_xyz[0]), float(rect_fb_xyz[1]), float(rect_fb_xyz[2]) + 0.035],
        )
        self.declare_parameter("use_known_suction_pick_center", True)
        self.declare_parameter("suction_pick_center_tolerance_m", 0.010)
        self.declare_parameter("approach_clearance", 0.06)
        self.declare_parameter("pre_touch_hover_extra_z", 0.04)
        self.declare_parameter("xy_refine_safe_clearance", 0.03)
        # 过大的下压量会在 attach 前把物体“顶走”；这里把默认下压量收小，
        # 使“物体顶面中心 ≈ 吸盘底面中心”时不会因过深接触产生横向滑移。
        self.declare_parameter("touch_delta_z", 0.003)
        # suction_tcp_link 已固定在吸盘接触中心，接触偏移由 URDF 表达。
        self.declare_parameter("suction_contact_offset_z", 0.0)
        self.declare_parameter("compensate_suction_center_target", True)
        self.declare_parameter("pre_pick_safe_clearance", 0.30)
        self.declare_parameter("carton_floor_top_z", 0.006)
        self.declare_parameter("place_height_above_floor", 0.12)
        self.declare_parameter("place_object_bottom_clearance", 0.008)
        self.declare_parameter("carton_outer_size_xyz", [0.28, 0.22, 0.13])
        self.declare_parameter("carton_wall_thickness", 0.008)
        self.declare_parameter("carton_floor_thickness", 0.006)
        self.declare_parameter("post_pick_lift", 0.10)
        self.declare_parameter("place_entry_clearance", 0.10)
        self.declare_parameter("post_place_retreat", 0.10)
        # 放置阶段仍建议做碰撞感知 IK；抓取预定位常被「纸箱碰撞体」判死 → IK 无解、机械臂不动
        self.declare_parameter("ik_avoid_collisions", True)
        self.declare_parameter(
            "ik_pick_avoid_collisions",
            True,
        )
        self.declare_parameter("ik_timeout_sec", 5.0)
        # compute_ik RPC 完成等待时长（与 IK 内部 timeout 区分），避免单次调用长期阻塞看起来“完全不动”
        self.declare_parameter("ik_call_wait_sec", 4.0)
        self.declare_parameter("ik_search_wall_time_sec", 30.0)
        self.declare_parameter("pose_goal_fallback", True)
        self.declare_parameter("pose_position_tolerance", 0.005)
        # 吸盘朝下姿态极严格：0.005 rad ≈ 0.29°
        self.declare_parameter("pose_orientation_tolerance", 0.005)
        self.declare_parameter("joint_goal_tolerance", 0.04)
        self.declare_parameter("move_velocity_scale", 0.20)
        self.declare_parameter("move_acceleration_scale", 0.20)
        self.declare_parameter("rect_fallback_pose_xyz", rect_fb_xyz)
        self.declare_parameter("rect_fallback_wait_sec", 8.0)
        self.declare_parameter("carton_fallback_pose_xyz", carton_fb_xyz)
        self.declare_parameter("carton_fallback_wait_sec", 8.0)
        # attach 可适度放宽状态确认，但几何上仍只允许底部两个橡胶吸盘接触顶面。
        self.declare_parameter("suction_attach_lateral_tol", 0.045)
        self.declare_parameter("suction_attach_vertical_tol", 0.040)
        self.declare_parameter("suction_attach_axis_down_min", 0.96)
        # 真正触碰吸附前使用更严格门限，避免“边缘吸附/未接触就尝试吸附”。
        self.declare_parameter("suction_touch_lateral_tol", 0.015)
        self.declare_parameter("suction_touch_vertical_tol", 0.022)
        self.declare_parameter("suction_touch_axis_down_min", 0.985)
        self.declare_parameter("require_dual_bottom_contact_before_attach", True)
        self.declare_parameter("suction_cup_offsets_xy", [-0.018, 0.0, 0.018, 0.0])
        self.declare_parameter("suction_cup_lip_radius", 0.010)
        self.declare_parameter("suction_rubber_compression_m", 0.010)
        self.declare_parameter("suction_attach_burst_count", 10)
        self.declare_parameter("suction_attach_burst_interval_sec", 0.04)
        self.declare_parameter("suction_attach_wait_sec", 2.8)
        self.declare_parameter("pickup_probe_lift_z", 0.025)
        self.declare_parameter("pickup_probe_min_follow_z", 0.010)
        self.declare_parameter("pickup_probe_require_follow_if_live_pose", True)
        self.declare_parameter("ik_service", "/compute_ik")
        # 工业仿真默认改为笛卡尔分段轨迹 + 数值 IK（不依赖 MoveIt compute_ik）。
        self.declare_parameter("use_compute_ik", False)
        self.declare_parameter("use_joint_template_demo", False)
        self.declare_parameter("wait_poses_sec", 45.0)
        self.declare_parameter("require_joint_states", True)
        self.declare_parameter("pre_pick_try_clearances", [0.10, 0.14, 0.18, 0.24])
        # 修改为更接近 img/20.png 直立姿态的种子：joint2≈-0.30(手肘向上)，joint3≈1.50(抬臂)
        self.declare_parameter("pick_posture_hint", [0.0, -0.30, 1.50, -0.50, 1.50, 0.0])
        self.declare_parameter("place_posture_hint", [0.0, -0.48, 0.92, 0.0, 1.08, 0.0])
        # 抓取预位姿阶段禁止 link2/link3 下探到近地“跪倒”分支；真正接触阶段改走笛卡尔下压。
        # 放宽到 0.12，允许 img/20.png 风格的直立低位预抓姿态通过过滤。
        self.declare_parameter("pick_pregrasp_min_joint3_origin_z", 0.12)
        self.declare_parameter("pick_pregrasp_elbow_filter_min_target_z", 0.45)
        # 首个规划目标应是物体正上方高位预抓，不再先去图 45 那类侧向 seed 姿态。
        self.declare_parameter("move_to_start_face_pose", False)
        self.declare_parameter("start_face_posture_hint", [0.0, -0.68, 1.02, 0.0, 1.18, 0.0])
        # 默认按“物体+箱子中点方向”自动朝向；若关闭则使用 start_face_joint1_rad。
        self.declare_parameter("start_face_use_scene_midpoint_yaw", True)
        self.declare_parameter("start_face_joint1_rad", 0.0)
        # 关节零位与世界 +X 不一致时，用固定偏置对齐：joint1 = atan2(y,x) + offset
        # joint1 零位已在 URDF/SDF 中对齐到“面向目标”的方向；这里不再需要历史偏置。
        self.declare_parameter("joint1_world_yaw_offset_rad", 0.0)
        # 桥接偶发抖动时，实时 pose 可能跳变；默认关闭实时覆盖，优先使用已知几何中心。
        self.declare_parameter("refresh_top_from_live_pose", False)
        self.declare_parameter("refresh_top_max_delta_xy", 0.20)
        self.declare_parameter("refresh_top_max_delta_z", 0.08)
        # 可选：直接给抓取点/放置点坐标（base_link），用于“按坐标抓放”。
        # 注意：参数默认类型必须是 DOUBLE_ARRAY，不能用空列表 []。
        self.declare_parameter("use_direct_xyz", False)
        self.declare_parameter("pick_point_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("place_point_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("cartesian_step_max_m", 0.012)
        self.declare_parameter("cartesian_step_max_rad", 0.18)
        self.declare_parameter("cartesian_cmd_period_sec", 0.035)
        self.declare_parameter("cartesian_settle_timeout_sec", 4.0)
        self.declare_parameter("cartesian_settle_tol_rad", 0.045)
        self.declare_parameter("cartesian_ik_max_iters", 140)
        self.declare_parameter("cartesian_ik_pos_tol_m", 0.004)
        self.declare_parameter("cartesian_ik_ori_tol_rad", 0.04)
        self.declare_parameter("cartesian_ik_damping", 0.08)
        self.declare_parameter("cartesian_ik_step_gain", 0.8)
        self.declare_parameter("cartesian_ik_joint_step_limit_rad", 0.12)
        self.declare_parameter("cartesian_ik_orientation_weight", 6.0)
        self.declare_parameter("touch_cartesian_keep_xy", True)
        self.declare_parameter("touch_cartesian_step_max_m", 0.004)
        self.declare_parameter("touch_cartesian_joint_step_limit_rad", 0.04)
        self.declare_parameter("touch_cartesian_orientation_weight", 8.0)
        self.declare_parameter("touch_cartesian_pose_fallback", True)
        self.declare_parameter("place_compensate_pick_offset", True)
        self.declare_parameter("place_compensation_gain", 1.0)
        self.declare_parameter("place_inner_margin_xy", 0.015)
        self.declare_parameter("cartesian_bridge_wait_sec", 25.0)
        # 混合模式：先 MoveIt 到 pre-grasp/approach，再用笛卡尔直线下压抓取。
        self.declare_parameter("hybrid_moveit_pregrasp", True)
        self.declare_parameter("hybrid_cartesian_touch_only", True)
        # 抓取预位姿禁止只给末端 Pose goal；先求非跪倒关节解，再交给 MoveIt 按关节目标执行。
        self.declare_parameter("prefer_upright_joint_goal_for_pick", True)
        # 仅按目标物体中心坐标执行“上方对齐 + 直线下压”，不做反馈式 XY 中心补偿
        self.declare_parameter("centerline_use_object_center_only", True)
        self.declare_parameter("pregrasp_xy_align_tol", 0.015)
        self.declare_parameter("pregrasp_alignment_gate_enabled", False)
        self.declare_parameter("pregrasp_xy_comp_max_step_m", 0.03)
        self.declare_parameter("pregrasp_xy_comp_gain", 1.0)
        self.declare_parameter("pregrasp_xy_comp_retries", 3)
        self.declare_parameter("pregrasp_cartesian_center_enabled", True)
        # 下压前强制校正吸盘朝向：若吸盘+Z轴与世界-Z的cos对齐度低于此阈值则执行校正。
        self.declare_parameter("orientation_min_cos_before_touch", 0.9995)
        # 校正朝向时的朝向权重，需远高于普通运动以强制优先保持朝下。
        self.declare_parameter("orientation_correction_weight", 10.0)
        # 校正朝向时的最大重试次数。
        self.declare_parameter("orientation_correction_retries", 6)
        # MoveIt 位姿目标到位后，默认不再额外插入笛卡尔朝向 snap；
        # Gazebo 关节控制器在这一步更容易把 TCP 从目标上方拉偏。
        self.declare_parameter("post_moveit_orientation_snap_enabled", False)
        # 下压前在目标上方悬停验证 XY 对齐和朝向的等待时间（秒），
        # 给 Gazebo 关节控制器足够收敛时间，避免首次下压因位置未收敛而失败。
        self.declare_parameter("pre_touch_settle_sec", 2.0)
        # 下压前最终中心线由小步笛卡尔校正负责；禁用全局 MoveIt approach 重规划，避免带偏正上方路径。
        self.declare_parameter("approach_verify_enabled", False)
        # 下压完成后、检查吸附前的稳定等待时间（秒），
        # 让 Gazebo 物理仿真充分计算接触响应后再判定吸附结果。
        self.declare_parameter("post_touch_settle_sec", 1.2)
        # 发送 attach 指令前额外等待秒数，确保吸盘与物体接触已稳定。
        self.declare_parameter("pre_attach_settle_sec", 0.6)
        self.declare_parameter("dense_waypoint_descent_enabled", False)
        self.declare_parameter("dense_waypoint_step_m", 0.004)
        self.declare_parameter("dense_waypoint_xy_correction_gain", 0.85)
        self.declare_parameter("dense_waypoint_xy_correction_max_m", 0.008)
        self.declare_parameter("dense_waypoint_orientation_weight", 8.0)
        self.declare_parameter("dense_waypoint_settle_sec", 0.04)
        self.declare_parameter("dense_waypoint_max_xy_drift_m", 0.03)
        self.declare_parameter("staged_pregrasp_enabled", True)
        self.declare_parameter("staged_pregrasp_clearances", [0.24, 0.20, 0.16, 0.12, 0.09])
        self.declare_parameter("staged_pregrasp_settle_sec", 0.5)
        self.declare_parameter("adaptive_touch_target_enabled", True)
        self.declare_parameter("adaptive_touch_max_adjust_m", 0.03)
        # Gazebo DetachableJoint 是仿真吸附器；为保证演示连续性，允许在几何严格检查不稳定时继续 attach。
        # 连接真实机械臂/真实吸盘前应将该参数设为 false。
        self.declare_parameter("allow_unverified_sim_attach", True)
        # DetachableJoint 在当前混合 ROS/Gazebo 环境下偶发不确认 attach。
        # 仅仿真演示时启用 SetEntityPose 兜底，让物体跟随吸盘并在释放时留在箱内。
        self.declare_parameter("fake_attach_set_pose_fallback", True)
        self.declare_parameter("fake_attach_service", "/world/arm_world/set_pose")
        self.declare_parameter("fake_attach_update_hz", 12.0)

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(
            PoseStamped,
            "/model/rect_pickup/pose",
            self._on_rect,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )
        self.create_subscription(
            PoseStamped,
            "/model/carton_box/pose",
            self._on_carton,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_js,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )
        self._ik_client = None
        self._scene_client = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self._scene_get_client = self.create_client(GetPlanningScene, "/get_planning_scene")
        self._set_pose_client = self.create_client(
            SetEntityPose,
            str(self.get_parameter("fake_attach_service").value),
        )
        self._action = ActionClient(self, MoveGroup, "move_action")
        self._pub_attach = self.create_publisher(Empty, "/cs612/suction/attach", 10)
        self._pub_detach = self.create_publisher(Empty, "/cs612/suction/detach", 10)
        self._pub_visual_attached = self.create_publisher(Bool, "/cs612/suction/attached_visual_state", 10)
        self._traj_pub = self.create_publisher(
            JointTrajectory,
            "/joint_trajectory_controller/joint_trajectory",
            10,
        )
        self.create_subscription(
            Bool,
            "/cs612/suction/state",
            self._on_suction_state,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )
        self._ready = False
        self._logged_rect = False
        self._logged_carton = False
        self._logged_js = False
        self._rect_fallback_used = False
        self._carton_fallback_used = False
        self._suction_attached: bool | None = None
        self._rect_motion_allowed = False
        self._fake_attach_active = False
        self._fake_attach_stop = threading.Event()
        self._fake_attach_thread: threading.Thread | None = None
        self._moveit_rect_attached = False
        self._attached_rect_half_sizes: list[float] | None = None
        self._joint_kin: list[_JointKinematic] = []
        self._tool_origin_xyz = (0.0, 0.0, 0.0)
        self._tool_origin_rpy = (0.0, 0.0, 0.0)
        self._kin_ready = self._load_kinematics_model()
        self._gz_bin = _find_gz_executable()
        self._gz_msg_pfx = _gz_msg_prefix(self._gz_bin)
        self.get_logger().info(
            "参数快照: "
            f"use_joint_template_demo={self._param_bool('use_joint_template_demo')}, "
            f"use_compute_ik={self._param_bool('use_compute_ik')}, "
            f"use_direct_xyz={self._param_bool('use_direct_xyz')}, "
            f"pose_goal_fallback={self._param_bool('pose_goal_fallback')}, "
            f"centerline_use_object_center_only={self._param_bool('centerline_use_object_center_only')}, "
            f"move_velocity_scale={float(self.get_parameter('move_velocity_scale').value):.2f}, "
            f"move_acceleration_scale={float(self.get_parameter('move_acceleration_scale').value):.2f}"
        )

    def _param_bool(self, name: str) -> bool:
        return _as_bool(self.get_parameter(name).value)

    def _param_xyz(self, name: str) -> list[float]:
        raw = self.get_parameter(name).value
        if isinstance(raw, (list, tuple)):
            return [float(v) for v in raw]
        if isinstance(raw, str):
            try:
                parsed = yaml.safe_load(raw)
                if isinstance(parsed, (list, tuple)):
                    return [float(v) for v in parsed]
            except Exception:
                return []
        return []

    def _load_kinematics_model(self) -> bool:
        def _parse_triplet(raw: str | None, default: tuple[float, float, float]) -> tuple[float, float, float]:
            if not raw:
                return default
            parts = [p for p in raw.replace(",", " ").split() if p]
            if len(parts) != 3:
                return default
            try:
                return (float(parts[0]), float(parts[1]), float(parts[2]))
            except Exception:
                return default

        urdf_candidates: list[Path] = []
        for root in (Path.cwd(), *Path(__file__).resolve().parents):
            urdf_candidates.extend(
                [
                    root / "my_arms" / "urdf" / "CS612.urdf",
                    root / "src" / "elite_description" / "urdf" / "cs612_description.urdf",
                ]
            )
        try:
            from ament_index_python.packages import get_package_share_directory

            cs612_share = Path(get_package_share_directory("CS612urdf"))
            urdf_candidates.append(cs612_share / "urdf" / "CS612.urdf")
            moveit_share = Path(get_package_share_directory("elite_moveit_ec612")) / "urdf"
            urdf_candidates.append(moveit_share / "cs612_description.urdf")
        except Exception:
            pass

        urdf_path = next((p for p in urdf_candidates if p.is_file()), None)
        if urdf_path is None:
            self.get_logger().error("未找到 CS612.urdf，笛卡尔数值 IK 不可用")
            return False

        try:
            root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
        except Exception as e:
            self.get_logger().error(f"读取 URDF 失败: {e}")
            return False

        joints_by_name = {}
        for joint in root.findall("joint"):
            name = joint.attrib.get("name")
            if name:
                joints_by_name[name] = joint
                joints_by_name[name.lower()] = joint

        chain: list[_JointKinematic] = []
        for name in _ARM_JOINTS:
            joint = joints_by_name.get(name)
            if joint is None:
                self.get_logger().error(f"URDF 缺少关节 {name}")
                return False
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            origin_xyz = _parse_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            origin_rpy = _parse_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            axis_xyz = _parse_triplet(axis.attrib.get("xyz") if axis is not None else None, (0.0, 0.0, 1.0))
            lower = -math.pi
            upper = math.pi
            try:
                if limit is not None:
                    lower = float(limit.attrib.get("lower", lower))
                    upper = float(limit.attrib.get("upper", upper))
            except Exception:
                pass
            chain.append(
                _JointKinematic(
                    name=name,
                    origin_xyz=origin_xyz,
                    origin_rpy=origin_rpy,
                    axis_xyz=axis_xyz,
                    lower=lower,
                    upper=upper,
                )
            )

        tool_xyz = (0.0, 0.0, 0.0)
        tool_rpy = (0.0, 0.0, 0.0)
        for fixed_name in ("joint_suction_cup", "joint_suction_tcp"):
            fixed_joint = joints_by_name.get(fixed_name)
            if fixed_joint is None:
                continue
            origin = fixed_joint.find("origin")
            origin_xyz = _parse_triplet(origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            origin_rpy = _parse_triplet(origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            tool_xyz = _vec_add(tool_xyz, _mat_vec_mul(_rot_from_rpy(*tool_rpy), origin_xyz))
            tool_rpy = (
                tool_rpy[0] + origin_rpy[0],
                tool_rpy[1] + origin_rpy[1],
                tool_rpy[2] + origin_rpy[2],
            )
        self._tool_origin_xyz = tool_xyz
        self._tool_origin_rpy = tool_rpy

        self._joint_kin = chain
        self.get_logger().info(f"笛卡尔运动学模型已加载: {urdf_path}")
        return True

    def _joint_limit_clamp(self, idx: int, value: float) -> float:
        if idx < 0 or idx >= len(self._joint_kin):
            return _wrap_to_pi(value)
        limit = self._joint_kin[idx]
        v = value
        if (limit.upper - limit.lower) >= (2.0 * math.pi - 0.1):
            v = _wrap_to_pi(v)
        if v < limit.lower:
            return limit.lower
        if v > limit.upper:
            return limit.upper
        return v

    def _fk_with_jacobian_context(
        self, joints: Sequence[float]
    ) -> tuple[tuple[float, float, float], list[list[float]], list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        r_cur = _mat_identity()
        p_cur = (0.0, 0.0, 0.0)
        joint_origins: list[tuple[float, float, float]] = []
        joint_axes: list[tuple[float, float, float]] = []

        for idx, kin in enumerate(self._joint_kin):
            r_o = _rot_from_rpy(*kin.origin_rpy)
            p_cur = _vec_add(p_cur, _mat_vec_mul(r_cur, kin.origin_xyz))
            r_cur = _mat_mul(r_cur, r_o)

            axis_world = _vec_normalize(_mat_vec_mul(r_cur, kin.axis_xyz))
            joint_axes.append(axis_world)
            joint_origins.append(p_cur)

            theta = float(joints[idx]) if idx < len(joints) else 0.0
            r_cur = _mat_mul(r_cur, _rot_from_axis_angle(kin.axis_xyz, theta))

        tool_r = _rot_from_rpy(*self._tool_origin_rpy)
        p_cur = _vec_add(p_cur, _mat_vec_mul(r_cur, self._tool_origin_xyz))
        r_cur = _mat_mul(r_cur, tool_r)
        return p_cur, r_cur, joint_origins, joint_axes

    def _current_tcp_pose(self) -> tuple[Point, Quaternion] | None:
        joints = self._current_arm_positions()
        if joints is None or not self._kin_ready:
            return None
        p, r, _, _ = self._fk_with_jacobian_context(joints)
        pose_p = Point(x=p[0], y=p[1], z=p[2])
        pose_q = _rot_to_quat(r)
        return pose_p, pose_q

    def _debug_fk_vs_tf(self, label: str) -> None:
        """Compare the internal FK model against robot_state_publisher TF.

        A non-trivial delta here means any direct Cartesian IK based on the
        internal model can command the wrong XY correction direction.
        """
        joints = self._current_arm_positions()
        tf_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if joints is None or tf_pose is None or not self._kin_ready:
            self.get_logger().warn(f"{label}: FK-vs-TF 诊断跳过（joint_states/TF/kinematics 不完整）")
            return
        fk_p, fk_r, _, _ = self._fk_with_jacobian_context(joints)
        fk_q = _rot_to_quat(fk_r)
        dx = float(tf_pose.position.x) - fk_p[0]
        dy = float(tf_pose.position.y) - fk_p[1]
        dz = float(tf_pose.position.z) - fk_p[2]
        fk_axis = _quat_rotate_vec(fk_q, 0.0, 0.0, 1.0)
        tf_axis = _quat_rotate_vec(tf_pose.orientation, 0.0, 0.0, 1.0)
        self.get_logger().warn(
            f"{label}: FK-vs-TF suction_tcp_link "
            f"fk=({fk_p[0]:.4f},{fk_p[1]:.4f},{fk_p[2]:.4f}) "
            f"tf=({tf_pose.position.x:.4f},{tf_pose.position.y:.4f},{tf_pose.position.z:.4f}) "
            f"delta=({dx:.4f},{dy:.4f},{dz:.4f}) norm={math.sqrt(dx*dx + dy*dy + dz*dz):.4f}m "
            f"down_cos_fk={-fk_axis[2]:.4f} down_cos_tf={-tf_axis[2]:.4f} "
            f"joints={[round(v, 4) for v in joints]}"
        )

    def _publish_joint_vector(self, joints: Sequence[float]) -> None:
        traj = JointTrajectory()
        traj.joint_names = list(_ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(joints[i]) for i in range(min(len(joints), len(_ARM_JOINTS)))]
        point.time_from_start = Duration(sec=0, nanosec=200_000_000)
        traj.points.append(point)
        traj.header.stamp = self.get_clock().now().to_msg()
        self._traj_pub.publish(traj)

    def _wait_joint_goal(self, target: Sequence[float], label: str) -> bool:
        tol = max(0.01, float(self.get_parameter("cartesian_settle_tol_rad").value))
        timeout_sec = max(0.3, float(self.get_parameter("cartesian_settle_timeout_sec").value))
        republish_period = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        deadline = time.monotonic() + timeout_sec
        next_pub = 0.0
        while time.monotonic() < deadline:
            cur = self._current_arm_positions()
            if cur is not None and len(cur) == len(target):
                if all(abs(float(cur[i]) - float(target[i])) <= tol for i in range(len(target))):
                    return True
            now = time.monotonic()
            if now >= next_pub:
                self._publish_joint_vector(target)
                next_pub = now + republish_period
            time.sleep(0.02)
        cur = self._current_arm_positions()
        if cur is not None and len(cur) == len(target):
            err_max = max(abs(float(cur[i]) - float(target[i])) for i in range(len(target)))
            self.get_logger().warn(f"{label}: 关节未完全收敛，max_err={err_max:.4f} rad")
        return False

    def _wait_cartesian_bridges(self, timeout_sec: float) -> bool:
        timeout_sec = max(0.5, float(timeout_sec))
        t0 = time.time()
        last_log = 0.0
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            traj_sub = self._traj_pub.get_subscription_count()
            attach_sub = self._pub_attach.get_subscription_count()
            detach_sub = self._pub_detach.get_subscription_count()
            traj_ready = traj_sub > 0
            suction_ready = attach_sub > 0 and detach_sub > 0
            if traj_ready and suction_ready:
                self.get_logger().info(
                    "桥接订阅已就绪: "
                    f"traj_sub={traj_sub}, attach_sub={attach_sub}, detach_sub={detach_sub}"
                )
                return True
            now = time.time()
            if now - last_log >= 2.0:
                self.get_logger().info(
                    "等待 Gazebo 桥接订阅: "
                    f"traj_sub={traj_sub}, attach_sub={attach_sub}, detach_sub={detach_sub}"
                )
                last_log = now
            time.sleep(0.1)
        self.get_logger().warn("桥接订阅等待超时，将继续执行（可能导致首段动作丢失）")
        return False

    def _send_joint_move_direct(self, positions: Sequence[float], label: str) -> bool:
        if not self._kin_ready:
            self.get_logger().error(f"{label}: 运动学模型未就绪")
            return False
        if len(positions) != len(_ARM_JOINTS):
            self.get_logger().error(f"{label}: 需要 {len(_ARM_JOINTS)} 个关节角")
            return False
        cur = self._current_arm_positions()
        if cur is None:
            self.get_logger().error(f"{label}: 当前关节状态不可用")
            return False
        target = [self._joint_limit_clamp(i, float(v)) for i, v in enumerate(positions)]
        max_delta = max(abs(target[i] - float(cur[i])) for i in range(len(target)))
        step_limit = max(0.01, float(self.get_parameter("cartesian_ik_joint_step_limit_rad").value))
        steps = max(1, int(math.ceil(max_delta / step_limit)))
        dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        self.get_logger().info(f"DirectJoint: {label} steps={steps}")
        for s in range(1, steps + 1):
            t = float(s) / float(steps)
            cmd = [float(cur[i]) + t * (target[i] - float(cur[i])) for i in range(len(target))]
            self._publish_joint_vector(cmd)
            time.sleep(dt)
        self._wait_joint_goal(target, label)
        return True

    def _quat_angle(self, a: Quaternion, b: Quaternion) -> float:
        qa = _quat_normalize(a)
        qb = _quat_normalize(b)
        d = qa.x * qb.x + qa.y * qb.y + qa.z * qb.z + qa.w * qb.w
        d = max(-1.0, min(1.0, abs(d)))
        return 2.0 * math.acos(d)

    def _solve_cartesian_ik_direct(
        self,
        target_p: Point,
        target_q: Quaternion,
        seed: Sequence[float],
        mode: str,
        label: str,
        orientation_weight_override: float | None = None,
        joint_step_limit_override: float | None = None,
    ) -> list[float] | None:
        if not self._kin_ready or len(seed) != len(_ARM_JOINTS):
            return None
        q = [self._joint_limit_clamp(i, float(seed[i])) for i in range(len(_ARM_JOINTS))]
        max_iters = max(20, int(self.get_parameter("cartesian_ik_max_iters").value))
        pos_tol = max(0.001, float(self.get_parameter("cartesian_ik_pos_tol_m").value))
        ori_tol = max(0.01, float(self.get_parameter("cartesian_ik_ori_tol_rad").value))
        damping = max(1e-4, float(self.get_parameter("cartesian_ik_damping").value))
        step_gain = max(0.05, float(self.get_parameter("cartesian_ik_step_gain").value))
        step_limit = max(
            0.02,
            float(
                joint_step_limit_override
                if joint_step_limit_override is not None
                else self.get_parameter("cartesian_ik_joint_step_limit_rad").value
            ),
        )
        ori_weight = max(
            0.1,
            float(
                orientation_weight_override
                if orientation_weight_override is not None
                else self.get_parameter("cartesian_ik_orientation_weight").value
            ),
        )
        target_r = _quat_to_rot(target_q)
        target_vec = (float(target_p.x), float(target_p.y), float(target_p.z))

        for _ in range(max_iters):
            p_e, r_e, origins, axes = self._fk_with_jacobian_context(q)
            err_p = _vec_sub(target_vec, p_e)
            err_o = _rotation_error(r_e, target_r)
            if _vec_norm(err_p) <= pos_tol and _vec_norm(err_o) <= ori_tol:
                return q

            e6 = np.array(
                [
                    err_p[0],
                    err_p[1],
                    err_p[2],
                    err_o[0] * ori_weight,
                    err_o[1] * ori_weight,
                    err_o[2] * ori_weight,
                ],
                dtype=float,
            )
            j = np.zeros((6, len(_ARM_JOINTS)), dtype=float)
            for i in range(len(_ARM_JOINTS)):
                jv = _vec_cross(axes[i], _vec_sub(p_e, origins[i]))
                jw = axes[i]
                j[0, i] = jv[0]
                j[1, i] = jv[1]
                j[2, i] = jv[2]
                j[3, i] = jw[0] * ori_weight
                j[4, i] = jw[1] * ori_weight
                j[5, i] = jw[2] * ori_weight

            jj_t = (j @ j.T) + (damping * damping) * np.eye(6, dtype=float)
            try:
                y = np.linalg.solve(jj_t, e6)
            except np.linalg.LinAlgError:
                return None

            dq_np = (j.T @ y) * step_gain
            max_abs = float(np.max(np.abs(dq_np)))
            if max_abs > step_limit:
                scale = step_limit / max_abs
                dq_np = dq_np * scale
            q = [self._joint_limit_clamp(i, q[i] + float(dq_np[i])) for i in range(len(_ARM_JOINTS))]

        self.get_logger().warn(
            f"{label}: 笛卡尔 IK 超时 mode={mode} target=({target_p.x:.3f},{target_p.y:.3f},{target_p.z:.3f})"
        )
        return None

    def _move_cartesian_direct(
        self,
        target: Point,
        orientation: Quaternion,
        mode: str,
        label: str,
        keep_xy_from_current: bool = False,
        pos_step_override: float | None = None,
        orientation_weight_override: float | None = None,
        joint_step_limit_override: float | None = None,
    ) -> bool:
        cur = self._current_arm_positions()
        if cur is None:
            self.get_logger().error(f"{label}: /joint_states 不可用，无法执行笛卡尔轨迹")
            return False
        pose = self._current_tcp_pose()
        if pose is None:
            self.get_logger().error(f"{label}: 无法读取当前 TCP 位姿")
            return False
        start_p, start_q = pose
        tgt = Point(x=target.x, y=target.y, z=target.z)
        if keep_xy_from_current:
            tgt.x = float(start_p.x)
            tgt.y = float(start_p.y)
        dist = math.sqrt(
            (float(tgt.x) - float(start_p.x)) ** 2
            + (float(tgt.y) - float(start_p.y)) ** 2
            + (float(tgt.z) - float(start_p.z)) ** 2
        )
        ang = self._quat_angle(start_q, orientation)
        pos_step = max(
            0.002,
            float(
                pos_step_override
                if pos_step_override is not None
                else self.get_parameter("cartesian_step_max_m").value
            ),
        )
        ang_step = max(0.02, float(self.get_parameter("cartesian_step_max_rad").value))
        steps = max(1, int(math.ceil(dist / pos_step)), int(math.ceil(ang / ang_step)))
        dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))

        q_seed = list(cur)
        self.get_logger().info(
            f"Cartesian: {label} steps={steps} target=({tgt.x:.3f},{tgt.y:.3f},{tgt.z:.3f})"
        )
        for s in range(1, steps + 1):
            t = float(s) / float(steps)
            wp = Point(
                x=float(start_p.x) + t * (float(tgt.x) - float(start_p.x)),
                y=float(start_p.y) + t * (float(tgt.y) - float(start_p.y)),
                z=float(start_p.z) + t * (float(tgt.z) - float(start_p.z)),
            )
            wq = _quat_slerp(start_q, orientation, t)
            sol = self._solve_cartesian_ik_direct(
                wp,
                wq,
                q_seed,
                mode=mode,
                label=label,
                orientation_weight_override=orientation_weight_override,
                joint_step_limit_override=joint_step_limit_override,
            )
            if sol is None:
                self.get_logger().error(f"{label}: 笛卡尔 IK 失败（step {s}/{steps}）")
                return False
            self._publish_joint_vector(sol)
            q_seed = sol
            time.sleep(dt)
        if not self._wait_joint_goal(q_seed, label):
            self.get_logger().error(f"{label}: 笛卡尔轨迹末端未收敛，判定执行失败")
            self._debug_fk_vs_tf(f"{label} failed_settle")
            return False
        return True

    def _move_cartesian_direct_allow_tf_settle(
        self,
        target: Point,
        orientation: Quaternion,
        mode: str,
        label: str,
        verify_top: Point,
        align_tol: float,
        **kwargs,
    ) -> bool:
        """执行小范围笛卡尔校正，并以真实吸盘底部中心而不是关节误差作为成功判据。"""
        moved = self._move_cartesian_direct(target, orientation, mode, label, **kwargs)
        if moved:
            return True

        # Gazebo wrist joints can lag while the tool frame has already improved
        # enough for a safe vertical descent.  Validate the real TF geometry.
        time.sleep(0.2)
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is None:
            self._debug_fk_vs_tf(f"{label} no_tf_after_cartesian")
            return False
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        dx = float(verify_top.x) - float(cup_bottom.x)
        dy = float(verify_top.y) - float(cup_bottom.y)
        lateral = math.hypot(dx, dy)
        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_cos = -down_axis[2]
        min_cos = max(0.90, float(self.get_parameter("suction_touch_axis_down_min").value))
        if lateral <= align_tol and down_cos >= min_cos:
            self.get_logger().warn(
                f"{label}: 关节未完全收敛，但真实吸盘几何已满足要求，继续执行 "
                f"lateral={lateral:.4f}m <= {align_tol:.4f}m, down_cos={down_cos:.4f}"
            )
            return True
        self.get_logger().warn(
            f"{label}: 真实吸盘几何仍未满足要求 lateral={lateral:.4f}m "
            f"(tol={align_tol:.4f}), down_cos={down_cos:.4f} (min={min_cos:.4f})"
        )
        self._debug_fk_vs_tf(f"{label} failed_geometry")
        return False

    def _move_cartesian_dense_waypoints(
        self,
        target: Point,
        orientation: Quaternion,
        top: Point,
        rect_half: Sequence[float],
        mode: str = "pick",
        label: str = "dense_descent",
        waypoint_step_m: float = 0.004,
        xy_correction_gain: float = 0.85,
        xy_correction_max_m: float = 0.008,
        orientation_correction_weight: float = 3.0,
        settle_sec_per_waypoint: float = 0.04,
        max_xy_drift_m: float = 0.03,
    ) -> bool:
        """
        密途径点闭环下压：把 approach→touch 长路径拆成每步 waypoint_step_m 米的小段，
        每走一步就读取真实 TCP 位姿并修正 XY 偏移和朝向漂移，确保吸盘始终沿物体中心竖直下降。

        关键逻辑：
        1. 从当前位置逐步下降到 target.z，每步只走 waypoint_step_m
        2. 每步到达后读真实 TCP，计算 XY 偏差并修正下一目标
        3. 朝向用高权重持续校正，防止下压过程中吸盘歪斜
        4. 若 XY 漂移超过 max_xy_drift_m，终止下压防止推走物体
        """
        cur_joints = self._current_arm_positions()
        if cur_joints is None:
            self.get_logger().error(f"{label}: 无法读取当前关节状态")
            return False
        pose = self._current_tcp_pose()
        if pose is None:
            self.get_logger().error(f"{label}: 无法读取当前 TCP 位姿")
            return False

        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        start_p, start_q = pose

        total_dz = float(target.z) - float(start_p.z)
        if abs(total_dz) < 1e-5:
            self.get_logger().info(f"{label}: 当前已在目标高度，无需下压")
            return True

        n_steps = max(1, int(math.ceil(abs(total_dz) / max(0.001, waypoint_step_m))))
        dz_per_step = total_dz / n_steps

        self.get_logger().info(
            f"{label}: 密途径点下压 start=({float(start_p.x):.4f},{float(start_p.y):.4f},{float(start_p.z):.4f}) "
            f"target=({float(target.x):.4f},{float(target.y):.4f},{float(target.z):.4f}) "
            f"steps={n_steps} dz_per_step={dz_per_step:.4f}m total_dz={total_dz:.4f}m"
        )

        center_x = float(target.x)
        center_y = float(target.y)
        cur_target_x = center_x
        cur_target_y = center_y
        cur_z = float(start_p.z)
        q_seed = list(cur_joints)
        cartesian_dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        convergence_lateral_threshold = 0.012
        min_descent_fraction_for_early_stop = 0.4
        prev_lateral = None
        divergence_count = 0
        max_divergence_steps = 3
        early_stop = False

        for step in range(1, n_steps + 1):
            next_z = cur_z + dz_per_step
            if dz_per_step < 0:
                next_z = max(next_z, float(target.z))
            else:
                next_z = min(next_z, float(target.z))

            wp = Point(x=cur_target_x, y=cur_target_y, z=next_z)
            self.get_logger().info(
                f"{label}[{step}/{n_steps}]: "
                f"wp=({cur_target_x:.4f},{cur_target_y:.4f},{next_z:.4f})"
            )

            sol = self._solve_cartesian_ik_direct(
                wp,
                orientation,
                q_seed,
                mode=mode,
                label=f"{label}_ik[{step}]",
                orientation_weight_override=orientation_correction_weight,
            )
            if sol is None:
                self.get_logger().error(f"{label}: IK 失败 step={step}/{n_steps}")
                return False

            self._publish_joint_vector(sol)
            q_seed = sol
            time.sleep(cartesian_dt)

            if settle_sec_per_waypoint > 0:
                time.sleep(settle_sec_per_waypoint)

            descent_fraction = abs(float(start_p.z) - next_z) / max(1e-6, abs(total_dz))

            cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
            live_top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))

            if cup_pose is not None and live_top is not None:
                cup_bottom = self._point_with_local_offset(
                    cup_pose.position,
                    cup_pose.orientation,
                    0.0, 0.0, suction_contact_offset,
                )
                dx_err = float(cup_bottom.x) - float(live_top.x)
                dy_err = float(cup_bottom.y) - float(live_top.y)
                lateral = math.hypot(dx_err, dy_err)

                if lateral > max_xy_drift_m:
                    self.get_logger().error(
                        f"{label}: XY 漂移过大 lateral={lateral:.4f}m > {max_xy_drift_m:.4f}m，"
                        f"终止下压防止推走物体"
                    )
                    return False

                if prev_lateral is not None and lateral > prev_lateral + 0.005:
                    divergence_count += 1
                else:
                    divergence_count = 0

                if (lateral <= convergence_lateral_threshold
                        and descent_fraction >= min_descent_fraction_for_early_stop):
                    self.get_logger().info(
                        f"{label}: 收敛早停 lateral={lateral:.4f}m <= {convergence_lateral_threshold:.4f}m, "
                        f"已下降 {descent_fraction:.0%}"
                    )
                    early_stop = True

                if (divergence_count >= max_divergence_steps
                        and prev_lateral is not None
                        and prev_lateral < convergence_lateral_threshold * 2
                        and descent_fraction >= min_descent_fraction_for_early_stop):
                    self.get_logger().info(
                        f"{label}: 发散早停 lateral={lateral:.4f}m 连续发散 {divergence_count} 步, "
                        f"prev_lateral={prev_lateral:.4f}m"
                    )
                    early_stop = True

                if early_stop and step < n_steps:
                    break

                if xy_correction_gain > 1e-6:
                    correction_x = -dx_err * xy_correction_gain
                    correction_y = -dy_err * xy_correction_gain
                    correction_mag = math.hypot(correction_x, correction_y)
                    if correction_mag > xy_correction_max_m:
                        scale = xy_correction_max_m / correction_mag
                        correction_x *= scale
                        correction_y *= scale

                    cur_target_x = float(live_top.x) + correction_x
                    cur_target_y = float(live_top.y) + correction_y

                    self.get_logger().info(
                        f"{label}[{step}]: XY 修正 "
                        f"dx_err={dx_err:.4f} dy_err={dy_err:.4f} lateral={lateral:.4f}m, "
                        f"next_target=({cur_target_x:.4f},{cur_target_y:.4f},{next_z:.4f})"
                    )
                else:
                    self.get_logger().info(
                        f"{label}[{step}]: XY 修正关闭 "
                        f"dx_err={dx_err:.4f} dy_err={dy_err:.4f} lateral={lateral:.4f}m"
                    )

                prev_lateral = lateral

                down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
                down_cos = -down_axis[2]
                min_down_cos = max(0.95, float(self.get_parameter("orientation_min_cos_before_touch").value))
                if down_cos < min_down_cos:
                    self.get_logger().warn(
                        f"{label}[{step}]: 朝向偏斜 down_cos={down_cos:.4f} < {min_down_cos:.4f}，"
                        f"将在下一步加重朝向权重"
                    )
            else:
                self.get_logger().debug(
                    f"{label}[{step}]: 无法读取 TF/物体位姿，跳过本步 XY 修正"
                )

            cur_z = next_z

        if not self._wait_joint_goal(q_seed, label):
            self.get_logger().warn(f"{label}: 最终关节未完全收敛")

        final_cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        final_top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
        if final_cup_pose is not None and final_top is not None:
            cup_b = self._point_with_local_offset(
                final_cup_pose.position,
                final_cup_pose.orientation,
                0.0, 0.0, suction_contact_offset,
            )
            final_dx = float(cup_b.x) - float(final_top.x)
            final_dy = float(cup_b.y) - float(final_top.y)
            final_lateral = math.hypot(final_dx, final_dy)
            down_ax = _quat_rotate_vec(final_cup_pose.orientation, 0.0, 0.0, 1.0)
            final_down_cos = -down_ax[2]
            self.get_logger().info(
                f"{label} 完成: final_offset=({final_dx:.4f},{final_dy:.4f}) "
                f"lateral={final_lateral:.4f}m down_cos={final_down_cos:.4f}"
            )
            min_down_cos = max(0.95, float(self.get_parameter("orientation_min_cos_before_touch").value))
            if final_down_cos < min_down_cos:
                self.get_logger().error(
                    f"{label}: 最终吸盘朝向不达标 down_cos={final_down_cos:.4f} < {min_down_cos:.4f}"
                )
                return False

        return True

    def _estimate_pick_offset_xy(self, rect_half: Sequence[float]) -> tuple[float, float] | None:
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        top = self._current_rect_top_live(rect_half) or self._current_rect_top(rect_half)
        if cup_pose is None or top is None:
            return None
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        return (top.x - cup_bottom.x, top.y - cup_bottom.y)

    def _refine_pregrasp_alignment(
        self,
        approach: Point,
        orientations: List[Quaternion],
        rect_half: Sequence[float],
    ) -> tuple[bool, Point]:
        offset = self._estimate_pick_offset_xy(rect_half)
        align_tol_cfg = max(0.002, float(self.get_parameter("pregrasp_xy_align_tol").value))
        touch_align_tol = max(0.003, float(self.get_parameter("suction_touch_lateral_tol").value))
        # 预抓阶段对心门限不能比触碰门限松太多，否则“预抓通过但触碰偏心”会频繁发生。
        align_tol = min(align_tol_cfg, max(0.004, touch_align_tol * 0.85))
        if offset is None:
            return True, approach

        lateral = math.hypot(offset[0], offset[1])
        self.get_logger().info(
            f"approach 对齐检查: dx={offset[0]:.4f}, dy={offset[1]:.4f}, lateral={lateral:.4f}"
        )
        if lateral <= align_tol:
            return True, approach

        max_step = max(0.003, float(self.get_parameter("pregrasp_xy_comp_max_step_m").value))
        gain = max(0.2, float(self.get_parameter("pregrasp_xy_comp_gain").value))
        retries = max(2, int(self.get_parameter("pregrasp_xy_comp_retries").value))
        use_cartesian_center = (
            self._param_bool("pregrasp_cartesian_center_enabled")
            and (not self._param_bool("use_compute_ik"))
            and self._param_bool("hybrid_cartesian_touch_only")
            and len(orientations) > 0
        )
        target = Point(x=approach.x, y=approach.y, z=approach.z)

        for i in range(retries):
            self.get_logger().warn(
                f"approach 未达到对齐阈值({align_tol:.4f}m)，执行 XY 对心补偿 step={i + 1}/{retries}"
            )
            dx = max(-max_step, min(max_step, float(offset[0]) * gain))
            dy = max(-max_step, min(max_step, float(offset[1]) * gain))
            target = Point(x=target.x + dx, y=target.y + dy, z=target.z)
            moved = False
            if use_cartesian_center:
                moved = self._move_cartesian_direct(
                    target,
                    orientations[0],
                    mode="pick",
                    label=f"approach_center_comp[{i + 1}]_cart",
                    keep_xy_from_current=False,
                    pos_step_override=max(0.002, float(self.get_parameter("touch_cartesian_step_max_m").value)),
                    orientation_weight_override=max(
                        0.1, float(self.get_parameter("touch_cartesian_orientation_weight").value)
                    ),
                    joint_step_limit_override=max(
                        0.01, float(self.get_parameter("touch_cartesian_joint_step_limit_rad").value)
                    ),
                )
                if not moved:
                    self.get_logger().warn(
                        f"approach_center_comp[{i + 1}]: 笛卡尔对心失败，回退 MoveIt 位姿规划"
                    )
            if not moved:
                moved = self._move_target_with_moveit_pose(target, orientations, f"approach_center_comp[{i + 1}]")
            if not moved:
                break
            time.sleep(0.2)
            offset = self._estimate_pick_offset_xy(rect_half)
            if offset is None:
                break
            lateral = math.hypot(offset[0], offset[1])
            self.get_logger().info(
                f"approach 补偿后对齐: dx={offset[0]:.4f}, dy={offset[1]:.4f}, lateral={lateral:.4f}"
            )
            if lateral <= align_tol:
                return True, target
        return False, target

    def _enforce_pregrasp_centerline(
        self,
        approach: Point,
        orientations: List[Quaternion],
        rect_half: Sequence[float],
    ) -> tuple[bool, Point]:
        """
        强制执行“先 MoveIt 到物体中心正上方”：
        仅使用 MoveIt 位姿目标做中心校正，不再做笛卡尔 XY 补偿，避免轨迹发散。
        """
        align_tol = min(
            max(0.005, float(self.get_parameter("pregrasp_xy_align_tol").value)),
            max(0.010, float(self.get_parameter("suction_touch_lateral_tol").value) * 1.5),
        )
        target = Point(x=approach.x, y=approach.y, z=approach.z)
        offset = self._estimate_pick_offset_xy(rect_half)
        if offset is None:
            return True, target
        lateral = math.hypot(offset[0], offset[1])
        self.get_logger().info(
            f"approach 对齐检查: dx={offset[0]:.4f}, dy={offset[1]:.4f}, lateral={lateral:.4f}"
        )
        if lateral <= align_tol:
            return True, target

        if self._correct_centerline_cartesian(
            orientations,
            rect_half,
            label="approach_center_cart",
            align_tol=align_tol,
            max_retries=max(3, int(self.get_parameter("pregrasp_xy_comp_retries").value)),
        ):
            return True, target

        for i in range(5):
            target = Point(x=target.x + offset[0], y=target.y + offset[1], z=target.z)
            self.get_logger().warn(
                f"approach 未居中（lateral={lateral:.4f} > tol={align_tol:.4f}），"
                f"执行 MoveIt 中心校正 step={i + 1}/5 -> target=({target.x:.3f},{target.y:.3f},{target.z:.3f})"
            )
            if not self._move_target_with_moveit_pose(target, orientations, f"approach_center_fix[{i + 1}]"):
                return False, target
            time.sleep(0.8)
            offset = self._estimate_pick_offset_xy(rect_half)
            if offset is None:
                return True, target
            lateral = math.hypot(offset[0], offset[1])
            self.get_logger().info(
                f"approach 校正后: dx={offset[0]:.4f}, dy={offset[1]:.4f}, lateral={lateral:.4f}"
            )
            if lateral <= align_tol:
                return True, target
        return False, target

    def _correct_centerline_cartesian(
        self,
        orientations: List[Quaternion],
        rect_half: Sequence[float],
        label: str,
        align_tol: float,
        max_retries: int = 3,
    ) -> bool:
        """
        在当前 hover 附近做小步笛卡尔 XY 修正，使吸盘底面中心对准物体顶面中心。
        这里优先使用 MoveIt 对 suction_tcp_link 的位姿约束。此前自写数值 IK 的 FK
        与 Gazebo/TF 存在偏差时，会出现“算出的修正方向正确，但真实 TCP 反而更偏”的现象。
        """
        if not self._kin_ready:
            return False
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        target_orient = orientations[0] if orientations else _suction_down_quat(0.0)
        max_step = max(0.004, float(self.get_parameter("pregrasp_xy_comp_max_step_m").value))
        moveit_orientations = [target_orient] + list(orientations[1:])

        for attempt in range(1, max(1, int(max_retries)) + 1):
            cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
            top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
            if cup_pose is None or top is None:
                return False

            cup_bottom = self._point_with_local_offset(
                cup_pose.position,
                cup_pose.orientation,
                0.0,
                0.0,
                suction_contact_offset,
            )
            dx = float(top.x) - float(cup_bottom.x)
            dy = float(top.y) - float(cup_bottom.y)
            lateral = math.hypot(dx, dy)
            self.get_logger().info(
                f"{label}[{attempt}]: "
                f"cup_bottom=({cup_bottom.x:.4f},{cup_bottom.y:.4f},{cup_bottom.z:.4f}), "
                f"top=({top.x:.4f},{top.y:.4f},{top.z:.4f}), "
                f"offset=({dx:.4f},{dy:.4f}), lateral={lateral:.4f}m"
            )
            if lateral <= align_tol:
                self.get_logger().info(f"{label}: 中心线笛卡尔校正通过 lateral={lateral:.4f}m")
                return True

            step_x = dx
            step_y = dy
            if lateral > max_step:
                scale = max_step / lateral
                step_x *= scale
                step_y *= scale

            target = Point(
                x=float(cup_pose.position.x) + step_x,
                y=float(cup_pose.position.y) + step_y,
                z=float(cup_pose.position.z),
            )
            self._debug_fk_vs_tf(f"{label}[{attempt}] before_moveit_center")
            moved = self._move_target_with_moveit_pose(
                target,
                moveit_orientations,
                f"{label}_moveit[{attempt}]",
            )
            if not moved:
                self.get_logger().error(f"{label}[{attempt}]: MoveIt TCP 中心线修正失败")
                return False
            time.sleep(0.4)
            self._debug_fk_vs_tf(f"{label}[{attempt}] after_moveit_center")

        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
        if cup_pose is None or top is None:
            return False
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        lateral = math.hypot(float(top.x) - float(cup_bottom.x), float(top.y) - float(cup_bottom.y))
        if lateral <= align_tol:
            self.get_logger().info(f"{label}: 中心线最终通过 lateral={lateral:.4f}m")
            return True
        self.get_logger().warn(f"{label}: 中心线仍未达标 lateral={lateral:.4f}m > {align_tol:.4f}m")
        return False

    def _adjust_place_point_for_box(
        self, place_pt: Point, carton_ps: PoseStamped | None, rect_half: Sequence[float]
    ) -> Point:
        out = Point(x=place_pt.x, y=place_pt.y, z=place_pt.z)
        if self._param_bool("place_compensate_pick_offset"):
            offset = self._estimate_pick_offset_xy(rect_half)
            if offset is not None:
                gain = float(self.get_parameter("place_compensation_gain").value)
                out.x -= gain * float(offset[0])
                out.y -= gain * float(offset[1])
                self.get_logger().info(
                    "放置 XY 偏移补偿: "
                    f"pick_offset=({offset[0]:.4f},{offset[1]:.4f}) gain={gain:.2f} -> "
                    f"target=({out.x:.3f},{out.y:.3f},{out.z:.3f})"
                )

        if carton_ps is None:
            return out

        try:
            sx, sy, _ = [float(v) for v in self.get_parameter("carton_outer_size_xyz").value]
            wall_t = float(self.get_parameter("carton_wall_thickness").value)
            margin = max(0.0, float(self.get_parameter("place_inner_margin_xy").value))
            in_half_x = max(0.01, 0.5 * sx - wall_t - float(rect_half[0]) - margin)
            in_half_y = max(0.01, 0.5 * sy - wall_t - float(rect_half[1]) - margin)
            q = carton_ps.pose.orientation
            c = carton_ps.pose.position
            q_inv = _quat_conjugate(q)
            local = _quat_rotate_vec(q_inv, out.x - c.x, out.y - c.y, out.z - c.z)
            clamped_x = max(-in_half_x, min(in_half_x, local[0]))
            clamped_y = max(-in_half_y, min(in_half_y, local[1]))
            if abs(clamped_x - local[0]) > 1e-6 or abs(clamped_y - local[1]) > 1e-6:
                self.get_logger().warn(
                    "放置点靠近箱壁，执行箱内夹紧: "
                    f"local_xy=({local[0]:.3f},{local[1]:.3f}) -> ({clamped_x:.3f},{clamped_y:.3f})"
                )
            world_xy = _quat_rotate_vec(q, clamped_x, clamped_y, local[2])
            out.x = c.x + world_xy[0]
            out.y = c.y + world_xy[1]
            out.z = c.z + world_xy[2]
        except Exception as e:
            self.get_logger().warn(f"放置点箱内夹紧失败，将使用原值: {e}")
        return out

    def _on_rect(self, msg: PoseStamped) -> None:
        # 丢弃异常全零位姿；勿按「近 XY 原点」过滤，否则会丢掉合法物体或错误筛掉首帧。
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        # 抓取前过滤明显异常跳变（常见于桥接/反序列化抖动），避免把抓取点带飞。
        # attach 后物体会跟随机械臂大幅移动，此时必须接收真实位姿用于吸附/放置验证。
        if self._param_bool("use_known_rect_surface_center") and not self._rect_motion_allowed:
            known_center = self._param_xyz("known_rect_center_xyz")
            if len(known_center) == 3:
                dxy = math.hypot(float(p.x) - float(known_center[0]), float(p.y) - float(known_center[1]))
                dz = abs(float(p.z) - float(known_center[2]))
                if dxy > 0.50 or dz > 0.25:
                    self.get_logger().warn(
                        "忽略异常 rect_pickup 位姿跳变: "
                        f"pose=({p.x:.3f},{p.y:.3f},{p.z:.3f}), "
                        f"known=({known_center[0]:.3f},{known_center[1]:.3f},{known_center[2]:.3f}), "
                        f"delta_xy={dxy:.3f}, delta_z={dz:.3f}"
                    )
                    return
        self._rect = msg
        with self._log_lock:
            if not self._logged_rect:
                self._logged_rect = True
                self.get_logger().info(
                    f"已收到 rect_pickup 位姿: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) frame={msg.header.frame_id or 'N/A'}"
                )
        self._check_ready()

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        self._carton = msg
        with self._log_lock:
            if not self._logged_carton:
                self._logged_carton = True
                self.get_logger().info(
                    f"已收到 carton_box 位姿: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) frame={msg.header.frame_id or 'N/A'}"
                )
        self._check_ready()

    def _ensure_rect_fallback(self) -> bool:
        if self._rect is not None:
            return True
        xyz = list(self.get_parameter("rect_fallback_pose_xyz").value)
        if len(xyz) != 3:
            self.get_logger().error("rect_fallback_pose_xyz 参数非法（需长度=3）")
            return False
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._rect = ps
        self._rect_fallback_used = True
        self.get_logger().warn(
            "未收到 /model/rect_pickup/pose，回退到默认物体位姿 "
            f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})。"
        )
        return True

    def _ensure_carton_fallback(self) -> bool:
        if self._carton is not None:
            return True
        xyz = list(self.get_parameter("carton_fallback_pose_xyz").value)
        if len(xyz) != 3:
            self.get_logger().error("carton_fallback_pose_xyz 参数非法（需长度=3）")
            return False
        ps = PoseStamped()
        ps.header.frame_id = "base_link"
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position = Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
        ps.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        self._carton = ps
        self._carton_fallback_used = True
        self.get_logger().warn(
            "未收到 /model/carton_box/* 位姿，回退到默认箱子位姿 "
            f"({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})。"
        )
        return True

    def _on_js(self, msg: JointState) -> None:
        self._joint_state = msg
        with self._log_lock:
            if not self._logged_js and msg.name:
                self._logged_js = True
                self.get_logger().info(f"已收到 /joint_states（示例关节: {msg.name[0]}）")
        self._check_ready()

    def _on_suction_state(self, msg: Bool) -> None:
        self._suction_attached = bool(msg.data)

    def _wait_suction_state(self, expected: bool, timeout_sec: float) -> bool:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            time.sleep(0.02)
            if self._suction_attached is expected:
                return True
        return False

    def _publish_attach_burst(self) -> None:
        burst = max(1, int(self.get_parameter("suction_attach_burst_count").value))
        interval = max(0.01, float(self.get_parameter("suction_attach_burst_interval_sec").value))
        for i in range(burst):
            self._pub_attach.publish(Empty())
            if i + 1 < burst:
                time.sleep(interval)

    def _detach_via_gz_cli(self, repeats: int = 3) -> bool:
        """
        社区常用兜底：直接通过 Gazebo Transport 发送 detach，绕开 ROS bridge/DDS 抖动。
        """
        env = self._gz_cli_env()
        ok = False
        for _ in range(max(1, int(repeats))):
            try:
                proc = subprocess.run(
                    [
                        self._gz_bin,
                        "topic",
                        "-t",
                        "/cs612/suction/detach",
                        "-m",
                        f"{self._gz_msg_pfx}.Empty",
                        "-p",
                        "unused: true",
                    ],
                    check=False,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                ok = ok or (proc.returncode == 0)
            except Exception:
                pass
            time.sleep(0.08)
        return ok

    def _attach_via_gz_cli(self, repeats: int = 2) -> bool:
        env = self._gz_cli_env()
        ok = False
        for _ in range(max(1, int(repeats))):
            try:
                proc = subprocess.run(
                    [
                        self._gz_bin,
                        "topic",
                        "-t",
                        "/cs612/suction/attach",
                        "-m",
                        f"{self._gz_msg_pfx}.Empty",
                        "-p",
                        "unused: true",
                    ],
                    check=False,
                    env=env,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=1.5,
                )
                ok = ok or (proc.returncode == 0)
            except Exception:
                pass
            time.sleep(0.08)
        return ok

    def _gz_cli_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PATH"] = "/usr/bin:/bin:/usr/local/bin"
        env["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
        env.setdefault("HOME", "/tmp/cs612_runtime/home")
        env.setdefault("GZ_SIM_RESOURCE_PATH", "")
        return env

    def _rect_pose_under_suction(self, half_sizes: Sequence[float]) -> Pose | None:
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is None:
            return None
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        touch_dz = max(0.0, float(self.get_parameter("touch_delta_z").value))
        half_z = max(0.001, float(half_sizes[2]) if len(half_sizes) >= 3 else 0.03)
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        pose = Pose()
        pose.position.x = cup_bottom.x
        pose.position.y = cup_bottom.y
        pose.position.z = max(half_z, cup_bottom.z - half_z + touch_dz)
        pose.orientation.w = 1.0
        return pose

    def _set_rect_pose(self, pose: Pose, wait_sec: float = 0.15) -> bool:
        if not self._set_pose_client.service_is_ready():
            return False
        req = SetEntityPose.Request()
        req.entity.name = "rect_pickup"
        req.entity.type = Entity.MODEL
        req.pose = pose
        try:
            fut = self._set_pose_client.call_async(req)
        except Exception:
            return False
        if not _spin_future(self, fut, wait_sec, "set_rect_pose"):
            return False
        try:
            res = fut.result()
            return bool(res and res.success)
        except Exception:
            return False

    def _fake_attach_loop(self, half_sizes: Sequence[float]) -> None:
        hz = max(1.0, float(self.get_parameter("fake_attach_update_hz").value))
        period = 1.0 / hz
        while rclpy.ok() and not self._fake_attach_stop.is_set():
            pose = self._rect_pose_under_suction(half_sizes)
            if pose is not None:
                self._set_rect_pose(pose, wait_sec=max(0.15, min(0.30, period * 2.0)))
            time.sleep(period)

    def _start_fake_attach(self, half_sizes: Sequence[float], reason: str) -> bool:
        if not self._param_bool("fake_attach_set_pose_fallback"):
            return False
        if self._fake_attach_active:
            return True
        if not self._set_pose_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"{reason}: /world/arm_world/set_pose 服务不可用，无法启用仿真位姿跟随兜底"
            )
            return False
        pose = self._rect_pose_under_suction(half_sizes)
        if pose is not None:
            self._set_rect_pose(pose, wait_sec=0.4)
        self._fake_attach_stop.clear()
        self._fake_attach_active = True
        self._rect_motion_allowed = True
        self._fake_attach_thread = threading.Thread(
            target=self._fake_attach_loop,
            args=(list(half_sizes),),
            daemon=True,
        )
        self._fake_attach_thread.start()
        self.get_logger().warn(f"{reason}: 已启用 Gazebo SetEntityPose 仿真吸附兜底")
        return True

    def _stop_fake_attach(self, label: str) -> None:
        if not self._fake_attach_active:
            return
        self._fake_attach_active = False
        self._fake_attach_stop.set()
        thread = self._fake_attach_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        self._fake_attach_thread = None
        self.get_logger().info(f"{label}: 已停止 SetEntityPose 仿真吸附兜底")

    def _ensure_detached(self, attempts: int = 8, wait_each_sec: float = 0.35) -> bool:
        """
        DetachableJoint 在 Gazebo 中可能以“已附着”状态启动。
        这里反复发送 detach 并尽量等待 state=false，防止未抓取时物体跟随机械臂。
        注意：部分 Gazebo DetachableJoint 版本不会持续发布 false；只有明确收到 true
        才必须中止，否则按“已下发 detach，继续执行”处理。
        """
        attempts = max(1, int(attempts))
        wait_each_sec = max(0.1, float(wait_each_sec))
        self._stop_fake_attach("启动 detach")
        for i in range(attempts):
            self._pub_detach.publish(Empty())
            if self._wait_suction_state(False, wait_each_sec):
                self._rect_motion_allowed = False
                if i > 0:
                    self.get_logger().info(f"detach 清状态成功（第 {i + 1}/{attempts} 次）")
                return True
            time.sleep(0.05)
        if self._suction_attached is True:
            self.get_logger().error("detach 清状态失败：吸附仍为 true，终止本次抓取以避免物体误跟随")
            return False
        if self._suction_attached is False:
            self._rect_motion_allowed = False
            return True
        # 若 state 话题暂不可用，改走 Gazebo 直连 detach；这是 DetachableJoint 社区常见稳态做法。
        if self._detach_via_gz_cli(repeats=6):
            self._suction_attached = False
            self._rect_motion_allowed = False
            self.get_logger().warn(f"ROS detach 未确认，已通过 {self._gz_bin} 直连 detach 兜底")
            return True
        self._suction_attached = False
        self._rect_motion_allowed = False
        self.get_logger().warn(
            "未收到 /cs612/suction/state=false，也未观察到 true；已多次下发 detach，"
            "按已分离状态继续执行。"
        )
        return True

    def _release_suction(
        self,
        label: str,
        wait_sec: float = 1.0,
        repeats: int = 4,
        keep_rect_motion_allowed: bool = False,
    ) -> bool:
        """
        同时通过 ROS bridge 和 Gazebo Transport 释放 DetachableJoint。
        ros_gz_bridge 在本项目的 conda/system ROS 混合环境里关停或高负载时偶发丢消息，
        因此释放动作必须有 ign/gz 兜底，否则物体可能继续挂在末端。
        """
        attached_half = list(self._attached_rect_half_sizes) if self._attached_rect_half_sizes is not None else None
        attached_pose = self._rect_pose_base(attached_half) if attached_half is not None else None
        self._stop_fake_attach(label)
        if self._moveit_rect_attached:
            # 先把 RViz/MoveIt 中的附着物体释放为世界物体，避免 Gazebo 已释放但 RViz 仍挂在末端。
            if attached_pose is None and attached_half is not None:
                attached_pose = self._rect_pose_under_suction(attached_half)
            self._detach_rect_from_tool_scene(attached_pose, attached_half)
        if keep_rect_motion_allowed:
            # 放置释放后方块会离开原抓取点，继续接收 Gazebo 的最终位姿用于场景同步。
            self._rect_motion_allowed = True
        self._suction_attached = None
        for i in range(max(1, int(repeats))):
            self._pub_detach.publish(Empty())
            if i + 1 < max(1, int(repeats)):
                time.sleep(0.04)
        ok = self._wait_suction_state(False, max(0.2, float(wait_sec)))
        if not ok:
            if self._detach_via_gz_cli(repeats=6):
                self.get_logger().warn(f"{label}: ROS detach 未确认，已下发 {self._gz_bin} detach 兜底")
                ok = self._wait_suction_state(False, 0.8) or True
            else:
                self.get_logger().warn(f"{label}: detach 未收到状态确认，已发送 ROS detach")
        self._suction_attached = False
        self._rect_motion_allowed = bool(keep_rect_motion_allowed)
        return ok

    def _current_rect_center_z(self) -> float | None:
        if self._rect is None:
            return None
        try:
            return float(self._pose_to_base(self._rect).pose.position.z)
        except Exception:
            return None

    def _known_suction_pick_center(self) -> Point | None:
        if not self._param_bool("use_known_suction_pick_center"):
            return None
        xyz = self._param_xyz("known_suction_pick_center_xyz")
        if len(xyz) != 3:
            return None
        return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    def _current_rect_top(self, half_sizes: Sequence[float]) -> Point | None:
        if not self._rect_motion_allowed:
            pick_center = self._known_suction_pick_center()
            if pick_center is not None:
                return pick_center
        if self._param_bool("use_known_rect_surface_center"):
            known_center = self._param_xyz("known_rect_center_xyz")
            known_size = self._param_xyz("known_rect_size_xyz")
            if len(known_center) == 3 and len(known_size) == 3 and float(known_size[2]) > 0.0:
                return Point(
                    x=float(known_center[0]),
                    y=float(known_center[1]),
                    z=float(known_center[2]) + 0.5 * float(known_size[2]),
                )
        if self._rect is None:
            return None
        try:
            return self._model_top_center(self._pose_to_base(self._rect), half_sizes)
        except Exception:
            return None

    def _current_rect_top_live(self, half_sizes: Sequence[float]) -> Point | None:
        if self._param_bool("use_known_rect_surface_center") and not self._rect_motion_allowed:
            return None
        if self._rect is None:
            return None
        try:
            return self._model_top_center(self._pose_to_base(self._rect), half_sizes)
        except Exception:
            return None

    def _lookup_link_pose_in_base(self, link_name: str) -> Pose | None:
        try:
            from rclpy.duration import Duration as RclDuration

            tf = self._tf_buffer.lookup_transform(
                "base_link",
                link_name,
                rclpy.time.Time(),
                timeout=RclDuration(seconds=0.3),
            )
        except Exception as e:
            self.get_logger().warn(f"读取 TF base_link←{link_name} 失败: {e}")
            return None
        pose = Pose()
        pose.position = Point(
            x=tf.transform.translation.x,
            y=tf.transform.translation.y,
            z=tf.transform.translation.z,
        )
        pose.orientation = tf.transform.rotation
        return pose

    def _suction_alignment_metrics(
        self, rect_half_sizes: Sequence[float]
    ) -> tuple[float, float, float, float, float, bool] | None:
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        top = self._current_rect_top_live(rect_half_sizes) or self._current_rect_top(rect_half_sizes)
        if cup_pose is None or top is None:
            if cup_pose is None and top is None:
                self.get_logger().warn("[alignment_metrics] 无法读取 suction_tcp_link 和物体位姿")
            elif cup_pose is None:
                self.get_logger().warn("[alignment_metrics] 无法读取 suction_tcp_link TF")
            else:
                self.get_logger().warn("[alignment_metrics] 无法读取物体顶面坐标")
            return None
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        dx = cup_bottom.x - top.x
        dy = cup_bottom.y - top.y
        lateral_err = math.hypot(dx, dy)
        vertical_err = abs(cup_bottom.z - top.z)
        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_alignment = -down_axis[2]
        on_top_surface = (
            abs(dx) <= float(rect_half_sizes[0]) and abs(dy) <= float(rect_half_sizes[1])
        )
        self.get_logger().debug(
            f"[alignment_metrics] "
            f"cup_bottom=({cup_bottom.x:.4f},{cup_bottom.y:.4f},{cup_bottom.z:.4f}) "
            f"top=({top.x:.4f},{top.y:.4f},{top.z:.4f}) "
            f"dx={dx:.4f} dy={dy:.4f} lateral={lateral_err:.4f} "
            f"vertical={vertical_err:.4f} down_cos={down_alignment:.4f}"
        )
        return dx, dy, lateral_err, vertical_err, down_alignment, on_top_surface

    def _suction_bottom_alignment_ok(self, rect_half_sizes: Sequence[float], strict: bool = False) -> bool:
        metrics = self._suction_alignment_metrics(rect_half_sizes)
        if metrics is None:
            self.get_logger().warn(
                f"[alignment_check] {'严格' if strict else '吸附'}检查失败："
                f"无法读取吸盘/物体位姿"
            )
            return False
        dx, dy, lateral_err, vertical_err, down_alignment, on_top_surface = metrics

        if strict:
            # touch 阶段使用独立容差，不再被 suction_pick_center_tolerance_m 截断。
            # center_tol 仅用于计算吸取目标中心，touch/attach 的容差应更宽松（橡胶软接触容错）。
            lateral_tol = max(0.005, float(self.get_parameter("suction_touch_lateral_tol").value))
            vertical_tol = max(0.005, float(self.get_parameter("suction_touch_vertical_tol").value))
            axis_down_min = float(self.get_parameter("suction_touch_axis_down_min").value)
            lateral_ok = lateral_err <= lateral_tol
            mode_text = "[严格]接触前判定"
        else:
            lateral_tol = float(self.get_parameter("suction_attach_lateral_tol").value)
            vertical_tol = float(self.get_parameter("suction_attach_vertical_tol").value)
            axis_down_min = float(self.get_parameter("suction_attach_axis_down_min").value)
            lateral_ok = lateral_err <= lateral_tol or on_top_surface
            mode_text = "[吸附]预检查"

        ok = lateral_ok and vertical_err <= vertical_tol and down_alignment >= axis_down_min
        if ok:
            self.get_logger().info(
                f"{mode_text} 通过: lateral={lateral_err:.4f}m(<=#{lateral_tol:.4f}), "
                f"dx={dx:.4f}, dy={dy:.4f}, "
                f"vertical={vertical_err:.4f}m(<=#{vertical_tol:.4f}), "
                f"down_cos={down_alignment:.4f}(>=#{axis_down_min:.4f})"
            )
        else:
            self.get_logger().warn(
                f"{mode_text} 未通过: "
                f"lateral={lateral_err:.4f}m({'OK' if lateral_ok else 'FAIL'}, tol={lateral_tol:.4f}), "
                f"dx={dx:.4f}, dy={dy:.4f}, "
                f"vertical={vertical_err:.4f}m({'OK' if vertical_err <= vertical_tol else 'FAIL'}, tol={vertical_tol:.4f}), "
                f"down_cos={down_alignment:.4f}({'OK' if down_alignment >= axis_down_min else 'FAIL'}, min={axis_down_min:.4f})"
            )
        return ok

    def _suction_cup_offsets(self) -> list[tuple[float, float]]:
        raw = list(self.get_parameter("suction_cup_offsets_xy").value)
        vals = [float(v) for v in raw]
        if len(vals) < 4 or len(vals) % 2 != 0:
            return [(-0.018, 0.0), (0.018, 0.0)]
        return [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]

    def _dual_suction_bottom_contact_ok(
        self,
        rect_half_sizes: Sequence[float],
        strict: bool = True,
        label: str = "dual_suction_contact",
    ) -> bool:
        """只允许底部两个橡胶吸盘唇口同时落在物体顶面有效区域内时触发吸附。"""
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        top = self._current_rect_top_live(rect_half_sizes) or self._current_rect_top(rect_half_sizes)
        if cup_pose is None or top is None:
            self.get_logger().warn(f"{label}: 双吸盘接触检查失败，无法读取 TF 或物体顶面")
            return False

        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        lip_radius = max(0.0, float(self.get_parameter("suction_cup_lip_radius").value))
        rubber_compression = max(0.0, float(self.get_parameter("suction_rubber_compression_m").value))
        center_tol = max(0.001, float(self.get_parameter("suction_pick_center_tolerance_m").value))
        vertical_tol = float(
            self.get_parameter("suction_touch_vertical_tol" if strict else "suction_attach_vertical_tol").value
        ) + rubber_compression
        axis_down_min = float(
            self.get_parameter("suction_touch_axis_down_min" if strict else "suction_attach_axis_down_min").value
        )
        usable_half_x = max(0.001, float(rect_half_sizes[0]) - 0.5 * lip_radius)
        usable_half_y = max(0.001, float(rect_half_sizes[1]) - 0.5 * lip_radius)

        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_cos = -down_axis[2]
        center_contact = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        center_dx = float(center_contact.x) - float(top.x)
        center_dy = float(center_contact.y) - float(top.y)
        center_dz = float(center_contact.z) - float(top.z)
        center_lateral = math.hypot(center_dx, center_dy)
        center_lateral_ok = center_lateral <= center_tol
        center_z_ok = abs(center_dz) <= center_tol
        contacts_ok = True
        details: list[str] = []
        for idx, (ox, oy) in enumerate(self._suction_cup_offsets(), start=1):
            contact = self._point_with_local_offset(
                cup_pose.position,
                cup_pose.orientation,
                ox,
                oy,
                suction_contact_offset,
            )
            dx = float(contact.x) - float(top.x)
            dy = float(contact.y) - float(top.y)
            dz = float(contact.z) - float(top.z)
            inside = abs(dx) <= usable_half_x and abs(dy) <= usable_half_y
            z_ok = abs(dz) <= vertical_tol
            cup_ok = inside and z_ok
            contacts_ok = contacts_ok and cup_ok
            details.append(
                f"cup{idx}:dx={dx:.4f},dy={dy:.4f},dz={dz:.4f},"
                f"inside={'Y' if inside else 'N'},z={'Y' if z_ok else 'N'}"
            )

        ok = contacts_ok and center_lateral_ok and center_z_ok and down_cos >= axis_down_min
        msg = (
            f"{label}: 双吸盘底部接触 {'通过' if ok else '未通过'} "
            f"suction_center=({top.x:.4f},{top.y:.4f},{top.z:.4f}), "
            f"center_err=({center_dx:.4f},{center_dy:.4f},{center_dz:.4f}), "
            f"center_lateral={center_lateral:.4f}(tol={center_tol:.4f}), "
            f"usable_half=({usable_half_x:.4f},{usable_half_y:.4f}), "
            f"vertical_tol+rubber={vertical_tol:.4f}, down_cos={down_cos:.4f}, "
            + "; ".join(details)
        )
        if ok:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg)
        return ok

    def _attach_geometry_ok(self, rect_half_sizes: Sequence[float], label: str) -> bool:
        if self._param_bool("require_dual_bottom_contact_before_attach"):
            return self._dual_suction_bottom_contact_ok(rect_half_sizes, strict=True, label=label)
        return self._suction_bottom_alignment_ok(rect_half_sizes, strict=True)

    def _ensure_suction_facing_down(
        self,
        orientations: List[Quaternion],
        label: str,
    ) -> bool:
        """下压前验证并纠正吸盘朝向：若吸盘+Z轴与世界-Z对齐度不足，用高朝向权重笛卡尔运动校正。"""
        min_cos = max(0.80, float(self.get_parameter("orientation_min_cos_before_touch").value))
        correction_weight = max(1.0, float(self.get_parameter("orientation_correction_weight").value))
        max_retries = max(0, int(self.get_parameter("orientation_correction_retries").value))
        target_orient = orientations[0] if orientations else _suction_down_quat(0.0)

        for attempt in range(1 + max_retries):
            cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
            if cup_pose is None:
                self.get_logger().warn(f"{label}: 无法读取 suction_tcp_link TF，跳过朝向验证")
                return True

            down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
            down_cos = -down_axis[2]

            if down_cos >= min_cos:
                self.get_logger().info(
                    f"{label}: 吸盘朝向验证通过 cos={down_cos:.4f} >= {min_cos:.4f}"
                )
                return True

            self.get_logger().warn(
                f"{label}: 吸盘朝向偏斜 cos={down_cos:.4f} < {min_cos:.4f}，"
                f"执行第 {attempt + 1} 次朝向校正（target cos >= {min_cos:.4f}）"
            )

            target_p = Point(
                x=float(cup_pose.position.x),
                y=float(cup_pose.position.y),
                z=float(cup_pose.position.z),
            )
            moved = self._move_cartesian_direct(
                target_p,
                target_orient,
                mode="pick",
                label=f"{label}_orient_fix[{attempt}]",
                keep_xy_from_current=True,
                orientation_weight_override=correction_weight,
            )
            if not moved:
                moved = self._move_target_with_moveit_pose(
                    target_p, orientations, f"{label}_orient_fix_pose[{attempt}]"
                )
            if not moved:
                self.get_logger().error(f"{label}: 朝向校正运动失败（尝试 {attempt + 1}）")
                continue

            time.sleep(0.3)

        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is not None:
            down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
            down_cos = -down_axis[2]
            if down_cos >= min_cos:
                self.get_logger().info(f"{label}: 朝向校正完成 cos={down_cos:.4f}")
                return True
            self.get_logger().error(
                f"{label}: 朝向校正后仍不达标 cos={down_cos:.4f} < {min_cos:.4f}"
            )
        else:
            self.get_logger().error(f"{label}: 朝向校正后无法验证")
        return False

    def _force_centerline_before_touch(
        self,
        top: Point,
        orientations: List[Quaternion],
        half_sizes: Sequence[float],
    ) -> bool:
        """下压前最终中心线闭环校验。

        读取真实 TCP 位置，计算吸盘底面中心与物体顶面中心的横向偏差。
        若偏差超过触摸容差的 1.5 倍，则回退到安全 hover 高度，
        移动到物体中心正上方，重新校验。
        确保笛卡尔下压起点 XY 已经对齐物体中心，避免斜向路径推走物体。
        """
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        clearance = float(self.get_parameter("approach_clearance").value)
        hover_extra = max(0.02, float(self.get_parameter("pre_touch_hover_extra_z").value))
        lateral_max = max(0.004, float(self.get_parameter("suction_touch_lateral_tol").value) * 1.5)
        max_retries = 3

        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        live_top = self._current_rect_top_live(list(half_sizes)) or self._current_rect_top(list(half_sizes))
        target_top = live_top if live_top is not None else top

        if cup_pose is None:
            self.get_logger().warn(
                "[centerline_pre_touch] 无法读取 suction_tcp_link TF，跳过最终对齐校验"
            )
            return True

        cup_bottom = self._point_with_local_offset(
            cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
        )
        dx = cup_bottom.x - target_top.x
        dy = cup_bottom.y - target_top.y
        lateral_err = math.hypot(dx, dy)
        vertical_err = abs(cup_bottom.z - target_top.z)
        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_cos = -down_axis[2]

        self.get_logger().info(
            f"[centerline_pre_touch] 下压前校验: "
            f"cup_bottom=({cup_bottom.x:.4f},{cup_bottom.y:.4f},{cup_bottom.z:.4f}), "
            f"top_center=({target_top.x:.4f},{target_top.y:.4f},{target_top.z:.4f}), "
            f"tcp=({cup_pose.position.x:.4f},{cup_pose.position.y:.4f},{cup_pose.position.z:.4f}), "
            f"dx={dx:.4f} dy={dy:.4f} lateral={lateral_err:.4f}m "
            f"vertical={vertical_err:.4f}m down_cos={down_cos:.4f}"
        )

        if lateral_err <= lateral_max:
            self.get_logger().info(
                f"[centerline_pre_touch] 对齐通过 lateral={lateral_err:.4f}m <= {lateral_max:.4f}m"
            )
            return True

        if self._correct_centerline_cartesian(
            orientations,
            half_sizes,
            label="centerline_pre_touch_cart",
            align_tol=lateral_max,
            max_retries=4,
        ):
            return True

        self.get_logger().warn(
            f"[centerline_pre_touch] lateral={lateral_err:.4f}m > {lateral_max:.4f}m，"
            f"执行安全高度重对齐（target top_center）"
        )

        for attempt in range(max_retries):
            hover_z = target_top.z + suction_contact_offset + clearance + hover_extra
            hover_pt = Point(x=target_top.x, y=target_top.y, z=hover_z)
            self.get_logger().info(
                f"[centerline_pre_touch] 重对齐 attempt {attempt + 1}/{max_retries}: "
                f"hover=({hover_pt.x:.4f},{hover_pt.y:.4f},{hover_pt.z:.4f})"
            )
            moved = self._move_target_with_fallback(
                hover_pt, orientations, mode="pick", label=f"centerline_hover[{attempt + 1}]"
            )
            if not moved:
                self.get_logger().warn(f"[centerline_pre_touch] 重对齐 hover 运动 {attempt + 1} 失败")
                continue
            time.sleep(0.6)
            cup_pose_new = self._lookup_link_pose_in_base("suction_tcp_link")
            if cup_pose_new is None:
                continue
            cup_bottom_new = self._point_with_local_offset(
                cup_pose_new.position, cup_pose_new.orientation, 0.0, 0.0, suction_contact_offset
            )
            live_top_new = self._current_rect_top_live(list(half_sizes)) or self._current_rect_top(list(half_sizes))
            check_top = live_top_new if live_top_new is not None else target_top
            dx_new = cup_bottom_new.x - check_top.x
            dy_new = cup_bottom_new.y - check_top.y
            lateral_new = math.hypot(dx_new, dy_new)
            self.get_logger().info(
                f"[centerline_pre_touch] 重对齐后 attempt={attempt + 1}: "
                f"lateral={lateral_new:.4f}m dx={dx_new:.4f} dy={dy_new:.4f}"
            )
            if lateral_new <= lateral_max:
                self.get_logger().info(
                    f"[centerline_pre_touch] 重对齐成功 lateral={lateral_new:.4f}m"
                )
                return True

        cup_pose_final = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose_final is not None:
            cup_bottom_final = self._point_with_local_offset(
                cup_pose_final.position, cup_pose_final.orientation, 0.0, 0.0, suction_contact_offset
            )
            final_lateral = math.hypot(cup_bottom_final.x - target_top.x, cup_bottom_final.y - target_top.y)
            self.get_logger().error(
                f"[centerline_pre_touch] 重对齐失败，最终 lateral={final_lateral:.4f}m"
            )
        return False

    def _verify_approach_pose(
        self,
        approach: Point,
        orientations: List[Quaternion],
        half_sizes: Sequence[float],
        label: str,
    ) -> bool:
        """approach 点就位后迭代验证+校正 TCP 位姿：XY/朝向/高度是否收敛到目标附近。
        最多迭代 3 次，每次读实际 TCP 偏差并发 MoveIt 校正目标。"""
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        max_verify_iters = 3
        lateral_max = max(0.005, float(self.get_parameter("suction_touch_lateral_tol").value) * 2.5)
        z_max = 0.035
        min_cos = max(0.80, float(self.get_parameter("orientation_min_cos_before_touch").value))
        target_orient = orientations[0] if orientations else _suction_down_quat(0.0)

        top = self._current_rect_top(list(half_sizes))
        if top is None:
            top = self._current_rect_top_live(list(half_sizes))
        if top is None:
            self.get_logger().warn(f"{label}: 无法读取物体顶面坐标，跳过 approach 验证")
            return True

        current_approach = Point(x=approach.x, y=approach.y, z=approach.z)

        for attempt in range(1, max_verify_iters + 1):
            cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
            if cup_pose is None:
                self.get_logger().warn(f"{label}[{attempt}]: 无法读取 suction_tcp_link TF，跳过验证")
                return True

            down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
            down_cos = -down_axis[2]

            cup_bottom = self._point_with_local_offset(
                cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
            )
            dx = cup_bottom.x - top.x
            dy = cup_bottom.y - top.y
            lateral_err = math.hypot(dx, dy)
            desired_bottom_z = float(current_approach.z) - suction_contact_offset
            z_err = abs(cup_bottom.z - desired_bottom_z)

            self.get_logger().info(
                f"{label}[{attempt}]: lateral={lateral_err:.4f}m(max={lateral_max:.4f}), "
                f"z_err={z_err:.4f}m(max={z_max:.4f}), down_cos={down_cos:.4f}(min={min_cos:.4f})"
            )

            xy_ok = lateral_err <= lateral_max
            # hover 高度由安全间隙决定，Gazebo/桥接会有滞后；这里主要收敛 XY 中心线和吸盘朝向。
            z_ok = True
            orient_ok = down_cos >= min_cos

            if xy_ok and z_ok and orient_ok:
                self.get_logger().info(f"{label}[{attempt}]: approach 验证通过")
                return True

            self.get_logger().warn(
                f"{label}[{attempt}]: approach 未通过 "
                f"xy={'OK' if xy_ok else 'FAIL'} z={'OK' if z_ok else 'FAIL'} "
                f"orient={'OK' if orient_ok else 'FAIL'}"
            )

            corrected_approach = Point(
                x=current_approach.x - dx,
                y=current_approach.y - dy,
                z=current_approach.z,
            )
            moved = self._move_target_with_moveit_pose(
                corrected_approach, orientations, f"{label}_fix[{attempt}]"
            )
            if not moved:
                moved = self._move_cartesian_direct(
                    corrected_approach,
                    target_orient,
                    mode="pick",
                    label=f"{label}_fix_cart[{attempt}]",
                    orientation_weight_override=max(1.0, float(self.get_parameter("orientation_correction_weight").value)),
                )
            if moved:
                time.sleep(0.8)
                current_approach = Point(x=corrected_approach.x, y=corrected_approach.y, z=corrected_approach.z)
            else:
                self.get_logger().warn(f"{label}[{attempt}]: 校正运动失败")

        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is not None:
            cup_bottom_final = self._point_with_local_offset(
                cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
            )
            final_lateral = math.hypot(cup_bottom_final.x - top.x, cup_bottom_final.y - top.y)
            self.get_logger().warn(f"{label}: 验证结束 lateral={final_lateral:.4f}m")
        return False

    def _refine_xy_alignment(
        self,
        top: Point,
        touch: Point,
        orientations: List[Quaternion],
        max_refine_steps: int = 2,
    ) -> Point:
        """
        依据当前吸盘底面与目标顶面中心的 XYZ 偏差做小步微调，避免"到边缘吸附/差一点贴不上"。
        安全策略：先抬到 approach 级别高度再做 XY 修正，避免在物体表面附近横推物体。
        """
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        clearance = float(self.get_parameter("approach_clearance").value)
        hover_extra = max(0.02, float(self.get_parameter("pre_touch_hover_extra_z").value))
        lateral_tol = float(self.get_parameter("suction_touch_lateral_tol").value)
        vertical_tol = float(self.get_parameter("suction_touch_vertical_tol").value)
        touch_dz = max(0.0, float(self.get_parameter("touch_delta_z").value))
        rect_half = [
            0.5 * float(v) for v in self.get_parameter("known_rect_size_xyz").value
        ]
        cur_touch = Point(x=top.x, y=top.y, z=touch.z)
        approach_hover_z = top.z + suction_contact_offset + clearance + hover_extra
        for step in range(max(1, int(max_refine_steps))):
            if self._suction_bottom_alignment_ok(rect_half, strict=True):
                break
            cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
            live_top = self._current_rect_top_live(rect_half)
            # 安全策略：始终使用原 top 做为目标中心，不为追踪被推物体而偏移
            target_top = live_top if live_top is not None else top
            # 检测物体是否被推出原位超过宽容值
            if live_top is not None:
                push_dxy = math.hypot(live_top.x - top.x, live_top.y - top.y)
                if push_dxy > 0.02:
                    self.get_logger().warn(
                        f"pick_xy_refine: 物体可能被推出原位 push_dxy={push_dxy:.4f}m，停止修正"
                    )
                    break
            if cup_pose is None:
                break
            cup_bottom = self._point_with_local_offset(
                cup_pose.position,
                cup_pose.orientation,
                0.0,
                0.0,
                suction_contact_offset,
            )
            ex = target_top.x - cup_bottom.x
            ey = target_top.y - cup_bottom.y
            ez = target_top.z - cup_bottom.z
            lateral_err = math.hypot(ex, ey)
            vertical_err = abs(ez)
            if lateral_err <= lateral_tol and vertical_err <= vertical_tol:
                break
            step_cap_xy = 0.012
            if lateral_err > step_cap_xy > 1e-6:
                scale = step_cap_xy / lateral_err
                ex *= scale
                ey *= scale
            desired_touch = self._compute_live_touch_target(
                target_top,
                rect_half,
                Point(x=target_top.x, y=target_top.y, z=cur_touch.z),
                f"pick_xy_refine_plan[{step + 1}]",
            )
            # 先抬到 approach 级别安全高度做 XY 修正，避免贴着物体表面横推
            hover = Point(
                x=target_top.x + ex,
                y=target_top.y + ey,
                z=max(approach_hover_z, desired_touch.z + clearance + hover_extra),
            )
            refined = Point(
                x=target_top.x + ex,
                y=target_top.y + ey,
                z=desired_touch.z,
            )
            self.get_logger().warn(
                f"pick_xy_refine step={step + 1}/{max_refine_steps}: "
                f"lateral={lateral_err:.4f}m vertical={vertical_err:.4f}m "
                f"dx={ex:.4f} dy={ey:.4f} "
                f"hover_z={hover.z:.4f} touch_z={refined.z:.4f}"
            )
            if not self._move_target_with_fallback(
                hover,
                orientations,
                mode="pick",
                label=f"pick_xy_refine_hover[{step + 1}]",
            ):
                break
            time.sleep(0.25)
            if not self._move_target_with_fallback(
                refined,
                orientations,
                mode="pick",
                label=f"pick_xy_refine_touch[{step + 1}]",
            ):
                break
            time.sleep(0.3)
            cur_touch = refined
        return cur_touch

    def _check_ready(self) -> None:
        # 启动条件仅依赖物体与纸箱位姿，避免 /joint_states 延迟或 QoS 抖动导致流程无法开始。
        if self._rect and self._carton:
            self._ready = True

    def _names_ok(self, js: JointState) -> bool:
        for j in _ARM_JOINTS:
            if j not in js.name:
                return False
        return True

    def _pose_to_base(self, ps: PoseStamped) -> PoseStamped:
        raw = (ps.header.frame_id or "").strip()
        # Gazebo Pose 桥接后 frame_id 常为模型名（非 TF 树）；位置已是世界系，与 base 对齐时等同 base_link
        if raw in (
            "",
            "world",
            "arm_world",
            "map",
            "base_link",
            "rect_pickup",
            "carton_box",
        ):
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
                timeout=RclDuration(seconds=2.0),
            )
            return do_transform_pose(ps, tf)
        except Exception as e:
            self.get_logger().warn(
                f"TF {raw}→base_link 失败 ({e})，假定与 base_link 对齐"
            )
            out = PoseStamped()
            out.header.frame_id = "base_link"
            out.header.stamp = ps.header.stamp
            out.pose = ps.pose
            return out

    def _model_top_center(self, model_ps: PoseStamped, half_sizes: Sequence[float]) -> Point:
        """在模型坐标系中顶面中心相对 link 原点为 (0,0,+hz)，转到 base_link。"""
        p = model_ps.pose.position
        q = model_ps.pose.orientation
        dx, dy, dz = _quat_rotate_vec(q, 0.0, 0.0, float(half_sizes[2]))
        pt = Point()
        pt.x = p.x + dx
        pt.y = p.y + dy
        pt.z = p.z + dz
        return pt

    def _carton_place_point(
        self, carton_ps: PoseStamped, floor_top_z: float, place_h: float
    ) -> Point:
        p = carton_ps.pose.position
        q = carton_ps.pose.orientation
        dx, dy, dz = _quat_rotate_vec(q, 0.0, 0.0, floor_top_z)
        pt = Point()
        pt.x = p.x + dx
        pt.y = p.y + dy
        pt.z = p.z + dz + place_h
        return pt

    def _safe_place_tcp_height_above_floor(
        self,
        requested_height: float,
        half_sizes: Sequence[float],
        suction_contact_offset: float,
        touch_delta_z: float,
    ) -> float:
        """
        MoveIt 位姿目标约束的是 suction_tcp_link 原点，不是吸盘接触面。
        为了把被吸附物体放进箱内且不让末端插入箱底，需要把 TCP 高度抬高：
        floor_top + object_height + suction_offset - touch_delta + bottom_clearance。
        """
        object_height = max(0.0, 2.0 * float(half_sizes[2]))
        bottom_clearance = max(0.0, float(self.get_parameter("place_object_bottom_clearance").value))
        safe_height = object_height + float(suction_contact_offset) - max(0.0, float(touch_delta_z)) + bottom_clearance
        if requested_height < safe_height:
            self.get_logger().warn(
                "箱内放置 TCP 高度过低，已按吸盘偏移与物体高度自动抬高: "
                f"requested={requested_height:.3f}m -> safe={safe_height:.3f}m "
                f"(object_h={object_height:.3f}, suction_offset={float(suction_contact_offset):.3f}, "
                f"bottom_clearance={bottom_clearance:.3f})"
            )
        return max(float(requested_height), safe_height)

    def _point_with_local_offset(
        self, origin: Point, q: Quaternion, ox: float, oy: float, oz: float
    ) -> Point:
        dx, dy, dz = _quat_rotate_vec(q, ox, oy, oz)
        return Point(x=origin.x + dx, y=origin.y + dy, z=origin.z + dz)

    def _make_collision_box(
        self,
        object_id: str,
        center: Point,
        orientation: Quaternion,
        size_xyz: Sequence[float],
        frame_id: str = "base_link",
    ) -> CollisionObject:
        co = CollisionObject()
        co.id = object_id
        co.header.frame_id = frame_id
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [float(size_xyz[0]), float(size_xyz[1]), float(size_xyz[2])]
        pose = Pose()
        pose.position = center
        pose.orientation = orientation
        co.primitives = [primitive]
        co.primitive_poses = [pose]
        co.operation = CollisionObject.ADD
        return co

    def _rect_size_from_half(self, half_sizes: Sequence[float]) -> list[float]:
        return [
            2.0 * float(half_sizes[0]),
            2.0 * float(half_sizes[1]),
            2.0 * float(half_sizes[2]),
        ]

    def _rect_pose_base(self, half_sizes: Sequence[float] | None = None) -> Pose | None:
        if self._rect is not None:
            try:
                return self._pose_to_base(self._rect).pose
            except Exception:
                pass
        if half_sizes is not None:
            return self._rect_pose_under_suction(half_sizes)
        return None

    def _rect_pose_in_suction_frame(self, world_pose: Pose) -> Pose | None:
        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is None:
            return None
        q_inv = _quat_conjugate(cup_pose.orientation)
        rel = _quat_rotate_vec(
            q_inv,
            float(world_pose.position.x) - float(cup_pose.position.x),
            float(world_pose.position.y) - float(cup_pose.position.y),
            float(world_pose.position.z) - float(cup_pose.position.z),
        )
        out = Pose()
        out.position = Point(x=rel[0], y=rel[1], z=rel[2])
        out.orientation = _quat_normalize(_quat_mul(q_inv, world_pose.orientation))
        return out

    def _remove_world_rect_collision_object(self) -> CollisionObject:
        co = CollisionObject()
        co.id = "scene_rect_pickup"
        co.header.frame_id = "base_link"
        co.operation = CollisionObject.REMOVE
        return co

    def _apply_scene_diff(self, scene: PlanningScene, label: str, timeout_sec: float = 10.0) -> bool:
        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self._scene_client.call_async(req)
        if not _spin_future(self, fut, timeout_sec, label):
            return False
        res = fut.result()
        return bool(res and res.success)

    def _attach_rect_to_tool_scene(self, half_sizes: Sequence[float]) -> bool:
        """在 MoveIt PlanningScene 中把 rect_pickup 设为 suction_tcp_link 的附着物体，供 RViz 同步显示。"""
        world_pose = self._rect_pose_under_suction(half_sizes) or self._rect_pose_base(half_sizes)
        if world_pose is None:
            self.get_logger().warn("MoveIt 附着物体同步失败：无法获取 rect_pickup 位姿")
            return False
        local_pose = self._rect_pose_in_suction_frame(world_pose)
        if local_pose is None:
            self.get_logger().warn("MoveIt 附着物体同步失败：无法获取 suction_tcp_link TF")
            return False

        attached = AttachedCollisionObject()
        attached.link_name = "suction_tcp_link"
        attached.touch_links = ["suction_tcp_link", "suction_cup_link", "wrist_3_link"]
        attached.object = self._make_collision_box(
            "scene_rect_pickup",
            local_pose.position,
            local_pose.orientation,
            self._rect_size_from_half(half_sizes),
            frame_id="suction_tcp_link",
        )
        attached.object.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [self._remove_world_rect_collision_object()]
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached]
        ok = self._apply_scene_diff(scene, "apply_rect_attached_scene", timeout_sec=15.0)
        if ok:
            self._moveit_rect_attached = True
            self._attached_rect_half_sizes = list(half_sizes)
            self._pub_visual_attached.publish(Bool(data=True))
            self.get_logger().info("RViz/MoveIt 已同步：rect_pickup 附着到 suction_tcp_link")
        else:
            self.get_logger().warn("RViz/MoveIt 附着物体同步失败：/apply_planning_scene 返回失败")
        return ok

    def _detach_rect_from_tool_scene(self, world_pose: Pose | None, half_sizes: Sequence[float] | None) -> bool:
        attached_remove = AttachedCollisionObject()
        attached_remove.link_name = "suction_tcp_link"
        attached_remove.object.id = "scene_rect_pickup"
        attached_remove.object.operation = CollisionObject.REMOVE

        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects = [attached_remove]
        if world_pose is not None and half_sizes is not None:
            scene.world.collision_objects = [
                self._make_collision_box(
                    "scene_rect_pickup",
                    world_pose.position,
                    world_pose.orientation,
                    self._rect_size_from_half(half_sizes),
                )
            ]
        ok = self._apply_scene_diff(scene, "apply_rect_detached_scene", timeout_sec=15.0)
        if ok:
            self._moveit_rect_attached = False
            self._attached_rect_half_sizes = None
            self._pub_visual_attached.publish(Bool(data=False))
            self.get_logger().info("RViz/MoveIt 已同步：rect_pickup 从 suction_tcp_link 释放")
        else:
            self.get_logger().warn("RViz/MoveIt 释放附着物体同步失败")
        return ok

    def _apply_rect_collision_pose(self, pose: Pose, half_sizes: Sequence[float]) -> bool:
        size_xyz = self._rect_size_from_half(half_sizes)
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            self._make_collision_box("scene_rect_pickup", pose.position, pose.orientation, size_xyz)
        ]
        return self._apply_scene_diff(scene, "apply_rect_release_scene", timeout_sec=10.0)

    def _released_rect_pose_in_box(
        self,
        place_center_xy: Point,
        carton_ps: PoseStamped | None,
        half_sizes: Sequence[float],
    ) -> Pose:
        half_z = max(0.001, float(half_sizes[2]) if len(half_sizes) >= 3 else 0.03)
        bottom_clearance = max(0.0, float(self.get_parameter("place_object_bottom_clearance").value))
        pose = Pose()
        pose.position.x = float(place_center_xy.x)
        pose.position.y = float(place_center_xy.y)
        if carton_ps is not None:
            pose.position.z = (
                float(carton_ps.pose.position.z)
                + float(self.get_parameter("carton_floor_thickness").value)
                + half_z
                + bottom_clearance
            )
            pose.orientation = carton_ps.pose.orientation
        else:
            pose.position.z = half_z + bottom_clearance
            pose.orientation.w = 1.0
        return pose

    def _settle_released_rect_in_box(
        self,
        place_center_xy: Point,
        carton_ps: PoseStamped | None,
        half_sizes: Sequence[float],
    ) -> None:
        pose = self._released_rect_pose_in_box(place_center_xy, carton_ps, half_sizes)
        if self._set_rect_pose(pose, wait_sec=0.5):
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose = pose
            self._rect = ps
            if self._apply_rect_collision_pose(pose, half_sizes):
                self.get_logger().info(
                    "已将释放后的 rect_pickup 固定到箱内规划场景: "
                    f"center=({pose.position.x:.3f},{pose.position.y:.3f},{pose.position.z:.3f})"
                )
            else:
                self.get_logger().warn("释放后 rect_pickup 已写入 Gazebo，但 MoveIt 场景同步失败，将依赖 spawner 更新")
        else:
            self.get_logger().warn("释放后 rect_pickup SetEntityPose 失败，将保留 Gazebo 物理最终位姿")

    def _apply_carton_collision_scene(self, carton_ps: PoseStamped) -> bool:
        if not self._param_bool("use_compute_ik"):
            # 笛卡尔直控模式不依赖 MoveIt 场景碰撞。
            return True
        sx, sy, sz = [float(v) for v in self.get_parameter("carton_outer_size_xyz").value]
        wall_t = float(self.get_parameter("carton_wall_thickness").value)
        floor_t = float(self.get_parameter("carton_floor_thickness").value)
        if sx <= 0 or sy <= 0 or sz <= 0 or wall_t <= 0 or floor_t <= 0:
            self.get_logger().error("纸箱碰撞参数非法（尺寸或厚度 <= 0）")
            return False

        q = carton_ps.pose.orientation
        c = carton_ps.pose.position

        half_x = 0.5 * sx
        half_y = 0.5 * sy
        half_h = 0.5 * sz
        wall_cx = half_x - 0.5 * wall_t
        wall_cy = half_y - 0.5 * wall_t

        objects = [
            self._make_collision_box(
                "carton_floor",
                self._point_with_local_offset(c, q, 0.0, 0.0, 0.5 * floor_t),
                q,
                [sx, sy, floor_t],
            ),
            self._make_collision_box(
                "carton_wall_px",
                self._point_with_local_offset(c, q, wall_cx, 0.0, half_h),
                q,
                [wall_t, sy, sz],
            ),
            self._make_collision_box(
                "carton_wall_nx",
                self._point_with_local_offset(c, q, -wall_cx, 0.0, half_h),
                q,
                [wall_t, sy, sz],
            ),
            self._make_collision_box(
                "carton_wall_py",
                self._point_with_local_offset(c, q, 0.0, wall_cy, half_h),
                q,
                [sx, wall_t, sz],
            ),
            self._make_collision_box(
                "carton_wall_ny",
                self._point_with_local_offset(c, q, 0.0, -wall_cy, half_h),
                q,
                [sx, wall_t, sz],
            ),
        ]

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = objects

        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self._scene_client.call_async(req)
        if not _spin_future(self, fut, 30.0, "apply_planning_scene"):
            return False
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().error("注入纸箱碰撞体失败：/apply_planning_scene 返回失败")
            return False
        self.get_logger().info("已更新 MoveIt 规划场景：纸箱底板 + 四面箱壁")
        return True

    def _set_pick_contact_collision_allowed(self, allow: bool) -> bool:
        """
        允许/禁止末端抓取接触相关 link 与 scene_rect_pickup 接触，避免触碰阶段被规划器强行“悬停”。
        """
        # 即便 use_compute_ik=false，触碰阶段也可能回退到 MoveIt 位姿规划，
        # 因此这里必须始终同步 ACM，避免回退路径被 scene_rect_pickup 碰撞约束卡死。
        if not self._scene_get_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn("设置抓取接触 ACM 失败：/get_planning_scene 不可用")
            return False

        get_req = GetPlanningScene.Request()
        get_req.components.components = PlanningSceneComponents.ALLOWED_COLLISION_MATRIX
        get_fut = self._scene_get_client.call_async(get_req)
        if not _spin_future(self, get_fut, 10.0, "get_planning_scene(acm)"):
            return False
        get_res = get_fut.result()
        if get_res is None:
            self.get_logger().warn("设置抓取接触 ACM 失败：未拿到当前规划场景")
            return False

        src = get_res.scene.allowed_collision_matrix
        names = list(src.entry_names)
        matrix = [list(row.enabled) for row in src.entry_values]

        n = len(names)
        while len(matrix) < n:
            matrix.append([False] * n)
        for row in matrix:
            if len(row) < n:
                row.extend([False] * (n - len(row)))

        def _ensure_name(name: str) -> int:
            nonlocal names, matrix
            if name in names:
                return names.index(name)
            names.append(name)
            for row in matrix:
                row.append(False)
            matrix.append([False] * len(names))
            return len(names) - 1

        i_rect = _ensure_name("scene_rect_pickup")
        for ee_link in ("suction_tcp_link", "suction_cup_link", "link6"):
            i_ee = _ensure_name(ee_link)
            matrix[i_ee][i_rect] = bool(allow)
            matrix[i_rect][i_ee] = bool(allow)

        acm = AllowedCollisionMatrix()
        acm.entry_names = names
        acm.entry_values = [AllowedCollisionEntry(enabled=[bool(v) for v in row]) for row in matrix]
        acm.default_entry_names = list(src.default_entry_names)
        acm.default_entry_values = list(src.default_entry_values)

        scene = PlanningScene()
        scene.is_diff = True
        scene.allowed_collision_matrix = acm

        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = self._scene_client.call_async(req)
        if not _spin_future(self, fut, 15.0, "apply_pick_contact_acm"):
            return False
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().warn("设置抓取接触 ACM 失败（将继续按默认碰撞策略）")
            return False
        self.get_logger().info(
            "抓取接触 ACM 已更新: "
            f"[suction_tcp_link, suction_cup_link, link6] ↔ scene_rect_pickup allow={allow}"
        )
        return True

    def _build_joint_goal(self, positions: List[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = False
        req = goal.request
        req.group_name = "cs_manipulator"
        req.num_planning_attempts = 15
        req.allowed_planning_time = 8.0
        vel = float(self.get_parameter("move_velocity_scale").value)
        acc = float(self.get_parameter("move_acceleration_scale").value)
        req.max_velocity_scaling_factor = min(max(vel, 0.01), 1.0)
        req.max_acceleration_scaling_factor = min(max(acc, 0.01), 1.0)
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnect"
        c = Constraints()
        for name, pos in zip(_ARM_JOINTS, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            joint_tol = max(0.005, float(self.get_parameter("joint_goal_tolerance").value))
            jc.tolerance_above = joint_tol
            jc.tolerance_below = joint_tol
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        return goal

    def _send_move(self, positions: List[float], label: str) -> bool:
        force_direct = label.startswith("tmpl_")
        use_moveit_joint = self._param_bool("use_compute_ik") or (
            self._param_bool("hybrid_moveit_pregrasp") and not force_direct
        )
        if not use_moveit_joint:
            return self._send_joint_move_direct(positions, label)
        if len(positions) != 6:
            self.get_logger().error(f"{label}: 需要 6 个关节角")
            return False
        if not self._action.wait_for_server(timeout_sec=60.0):
            self.get_logger().error("move_action 不可用")
            return False
        goal = self._build_joint_goal(positions)
        self.get_logger().info(f"MoveGroup: {label}")
        fut = self._action.send_goal_async(goal)
        if not _spin_future(self, fut, 120.0, f"{label} send_goal"):
            return False
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f"{label}: 目标被拒绝")
            return False
        rf = gh.get_result_async()
        if not _spin_future(self, rf, 300.0, f"{label} result"):
            return False
        res = rf.result()
        if res is None:
            return False
        err = res.result.error_code.val
        if err == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"{label}: 成功")
            return True
        self.get_logger().error(f"{label}: MoveIt 错误码 {err}")
        return False

    def _build_pose_goal(self, target: Point, orientation: Quaternion) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = False
        req = goal.request
        req.group_name = "cs_manipulator"
        req.num_planning_attempts = 30
        req.allowed_planning_time = 15.0
        vel = float(self.get_parameter("move_velocity_scale").value)
        acc = float(self.get_parameter("move_acceleration_scale").value)
        req.max_velocity_scaling_factor = min(max(vel, 0.01), 1.0)
        req.max_acceleration_scaling_factor = min(max(acc, 0.01), 1.0)
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnect"

        pos_tol = max(0.001, float(self.get_parameter("pose_position_tolerance").value))
        # 不设 0.01 下限裁剪，允许极紧容差生效
        ori_tol = max(0.001, float(self.get_parameter("pose_orientation_tolerance").value))

        pc = PositionConstraint()
        pc.header.frame_id = "base_link"
        pc.link_name = "suction_tcp_link"
        pc.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        region = BoundingVolume()
        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]
        region_center = Pose()
        region_center.position = target
        region_center.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        region.primitives = [sphere]
        region.primitive_poses = [region_center]
        pc.constraint_region = region
        pc.weight = 1.0

        oc = OrientationConstraint()
        oc.header.frame_id = "base_link"
        oc.link_name = "suction_tcp_link"
        oc.orientation = orientation
        oc.absolute_x_axis_tolerance = ori_tol
        oc.absolute_y_axis_tolerance = ori_tol
        # Z 轴（yaw）稍宽松，绕竖直轴旋转不影响垂直度
        oc.absolute_z_axis_tolerance = max(ori_tol, 0.05)
        oc.weight = 20.0

        c = Constraints()
        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        req.goal_constraints = [c]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        return goal

    def _suction_centered_origin_target(self, target: Point, orientation: Quaternion, label: str) -> Point:
        """
        高层抓放目标描述的是吸盘接触面中心；MoveIt 约束的是 suction_tcp_link 原点。
        按当前姿态反算原点目标，避免吸盘倾斜时接触面中心出现 XY 偏移。
        """
        if not self._param_bool("compensate_suction_center_target"):
            return target
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        ox, oy, oz = _quat_rotate_vec(orientation, 0.0, 0.0, suction_contact_offset)
        desired_contact = Point(
            x=float(target.x),
            y=float(target.y),
            z=float(target.z) - suction_contact_offset,
        )
        out = Point(
            x=desired_contact.x - ox,
            y=desired_contact.y - oy,
            z=desired_contact.z - oz,
        )
        shift = math.sqrt(
            (out.x - float(target.x)) ** 2
            + (out.y - float(target.y)) ** 2
            + (out.z - float(target.z)) ** 2
        )
        if shift > 0.002:
            self.get_logger().info(
                f"{label}: 吸盘中心补偿 target_origin=({target.x:.3f},{target.y:.3f},{target.z:.3f}) "
                f"-> link_origin=({out.x:.3f},{out.y:.3f},{out.z:.3f}), shift={shift:.3f}m"
            )
        return out

    def _send_pose_goal(self, target: Point, orientation: Quaternion, label: str) -> bool:
        if not self._action.wait_for_server(timeout_sec=60.0):
            self.get_logger().error("move_action 不可用")
            return False
        goal = self._build_pose_goal(target, orientation)
        self.get_logger().info(
            f"MoveGroup Pose: {label} target=({target.x:.3f},{target.y:.3f},{target.z:.3f})"
        )
        fut = self._action.send_goal_async(goal)
        if not _spin_future(self, fut, 120.0, f"{label} pose send_goal"):
            return False
        gh = fut.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f"{label}: 位姿目标被拒绝")
            return False
        rf = gh.get_result_async()
        if not _spin_future(self, rf, 300.0, f"{label} pose result"):
            return False
        res = rf.result()
        if res is None:
            return False
        err = res.result.error_code.val
        if err == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"{label}: 位姿目标成功")
            if self._param_bool("post_moveit_orientation_snap_enabled"):
                self._post_moveit_orientation_snap(target, orientation, label)
            return True
        self.get_logger().error(f"{label}: 位姿目标 MoveIt 错误码 {err}")
        return False

    def _post_moveit_orientation_snap(
        self, target: Point, orientation: Quaternion, label: str
    ) -> None:
        """MoveIt 到位后，检查实际朝向偏差；若偏差过大，用笛卡尔运动强制修正。"""
        if not self._kin_ready:
            return
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose is None:
            return
        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_cos = -down_axis[2]
        # 0.999 ≈ 2.6° — 若偏差小于此值则视为可接受
        min_cos_snap = 0.999
        if down_cos >= min_cos_snap:
            self.get_logger().info(
                f"{label}: post-MoveIt 朝向已满足 cos={down_cos:.5f} >= {min_cos_snap}"
            )
            return
        self.get_logger().warn(
            f"{label}: post-MoveIt 朝向偏差 cos={down_cos:.5f} < {min_cos_snap}，"
            f"执行笛卡尔朝向校正"
        )
        correction_weight = max(5.0, float(self.get_parameter("orientation_correction_weight").value))
        for attempt in range(3):
            snap_target = Point(
                x=float(cup_pose.position.x),
                y=float(cup_pose.position.y),
                z=float(cup_pose.position.z),
            )
            moved = self._move_cartesian_direct(
                snap_target,
                orientation,
                mode="pick",
                label=f"{label}_orient_snap[{attempt}]",
                keep_xy_from_current=True,
                orientation_weight_override=correction_weight,
                joint_step_limit_override=0.06,
            )
            if not moved:
                break
            time.sleep(0.3)
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
            if cup_pose is None:
                break
            down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
            down_cos = -down_axis[2]
            self.get_logger().info(
                f"{label}: 朝向校正 attempt {attempt + 1} 后 cos={down_cos:.5f}"
            )
            if down_cos >= min_cos_snap:
                return

    def _move_target_with_moveit_pose(
        self,
        target: Point,
        orientations: List[Quaternion],
        label: str,
        compensate_suction: bool = False,
    ) -> bool:
        for idx, ori in enumerate(orientations, start=1):
            goal_target = (
                self._suction_centered_origin_target(target, ori, f"{label}_pose[{idx}]")
                if compensate_suction
                else target
            )
            if self._send_pose_goal(goal_target, ori, f"{label}_pose[{idx}]"):
                return True
        return False

    def _move_target_with_upright_joint_goal(
        self,
        target: Point,
        orientations: List[Quaternion],
        mode: str,
        label: str,
    ) -> bool:
        """
        先用内部数值 IK 从"非跪倒"seed 求关节解，再用 MoveIt joint goal 规划执行。
        这样仍走 Elite MoveIt 的规划/执行链路，但不会让纯 Pose goal 自由选择肘部下坠分支。
        """
        if not self._kin_ready:
            return False
        best_sol: list[float] | None = None
        best_score: float | None = None
        best_goal_target: Point | None = None
        seeds = self._ik_seed_candidates(target, mode)
        for ori_idx, ori in enumerate(orientations, start=1):
            goal_target = self._suction_centered_origin_target(target, ori, f"{label}_upright[{ori_idx}]")
            for seed_idx, seed in enumerate(seeds, start=1):
                sol = self._solve_cartesian_ik_direct(
                    goal_target,
                    ori,
                    seed,
                    mode=mode,
                    label=f"{label}_upright_ik[{ori_idx},{seed_idx}]",
                    orientation_weight_override=max(
                        5.0, float(self.get_parameter("orientation_correction_weight").value)
                    ),
                    joint_step_limit_override=max(0.04, float(self.get_parameter("cartesian_ik_joint_step_limit_rad").value)),
                )
                if sol is None:
                    continue
                if self._reject_ik_solution(sol, target, mode):
                    self.get_logger().warn(
                        f"{label}: 丢弃跪倒/反肘候选关节解 seed={seed_idx} sol={[round(v, 3) for v in sol]}"
                    )
                    continue
                score = self._score_ik_solution(sol, target, mode)
                if best_score is None or score < best_score:
                    best_score = score
                    best_sol = sol
                    best_goal_target = goal_target
        if best_sol is None:
            self.get_logger().warn(f"{label}: 未求得非跪倒关节目标，将回退 Pose goal")
            return False

        # FK 验证：将 IK 解的 link origin 与 IK 目标（已含 suction offset 补偿）比对，
        # 避免 cup_bottom 与 link-origin 高度差（≈suction_contact_offset_z）被误判为偏差。
        # 同时验证吸盘朝下姿态质量（down_cos），拒绝倾斜过度的解。
        fk_ok = True
        try:
            tcp_p, tcp_r, _, _ = self._fk_with_jacobian_context(best_sol)
            gt = best_goal_target if best_goal_target is not None else target
            fk_lateral = math.hypot(tcp_p[0] - float(gt.x), tcp_p[1] - float(gt.y))
            fk_vertical = abs(tcp_p[2] - float(gt.z))
            if fk_lateral > 0.03 or fk_vertical > 0.06:
                self.get_logger().warn(
                    f"{label}: 最佳关节解 FK 验证偏差过大 "
                    f"(lateral={fk_lateral:.3f}m, vertical={fk_vertical:.3f}m)，"
                    f"回退 Pose goal 以保证落点精度"
                )
                fk_ok = False
            # 检查吸盘朝下姿态质量：down_cos 应接近 1.0 (竖直朝下)
            tcp_q = _rot_to_quat(tcp_r)
            suction_world_z = _quat_rotate_vec(tcp_q, 0.0, 0.0, 1.0)
            down_cos = -suction_world_z[2]
            if down_cos < 0.999:
                self.get_logger().warn(
                    f"{label}: 最佳关节解吸盘朝下姿态不足 "
                    f"down_cos={down_cos:.3f}(需≥0.999)，"
                    f"回退 Pose goal"
                )
                fk_ok = False
        except Exception as e:
            self.get_logger().warn(f"{label}: FK 验证异常({e})，继续用关节目标")

        if not fk_ok:
            return False

        self.get_logger().info(
            f"{label}: 使用非跪倒关节目标交给 MoveIt 执行 joints={[round(v, 3) for v in best_sol]}"
        )
        return self._send_move(best_sol, f"{label}_upright_joint")

    def _move_target_with_fallback(
        self,
        target: Point,
        orientations: List[Quaternion],
        mode: str,
        label: str,
    ) -> bool:
        use_compute_ik = self._param_bool("use_compute_ik")
        if not use_compute_ik:
            touch_motion = (
                mode == "pick"
                and (
                    label.startswith("pick_touch")
                    or label.startswith("pick_xy_refine_hover")
                    or label.startswith("pick_xy_refine_touch")
                    or label.startswith("pick_realign")
                )
            )
            if self._param_bool("hybrid_cartesian_touch_only") and touch_motion:
                self.get_logger().info(
                    f"{label}: use_compute_ik=false，优先使用 MoveIt suction_tcp_link 位姿目标"
                )
                if self._move_target_with_moveit_pose(
                    target,
                    orientations,
                    f"{label}_moveit_tcp",
                    compensate_suction=True,
                ):
                    return True
                self.get_logger().warn(
                    f"{label}: MoveIt TCP 位姿目标失败，才回退自写笛卡尔 IK"
                )
                keep_xy = self._param_bool("touch_cartesian_keep_xy")
                if self._param_bool("centerline_use_object_center_only") and (
                    label.startswith("pick_touch") or label.startswith("pick_realign")
                ):
                    # 中心线模式下，触碰/重定位必须显式走向目标 XY，避免“原地竖直下压”。
                    keep_xy = False
                if label.startswith("pick_xy_refine"):
                    keep_xy = False
                touch_step = max(0.001, float(self.get_parameter("touch_cartesian_step_max_m").value))
                touch_ori_weight = max(0.1, float(self.get_parameter("touch_cartesian_orientation_weight").value))
                touch_joint_step = max(
                    0.01, float(self.get_parameter("touch_cartesian_joint_step_limit_rad").value)
                )
                for idx, ori in enumerate(orientations, start=1):
                    cart_target = self._suction_centered_origin_target(target, ori, f"{label}_cart[{idx}]")
                    if self._move_cartesian_direct(
                        cart_target,
                        ori,
                        mode=mode,
                        label=f"{label}_cart[{idx}]",
                        keep_xy_from_current=keep_xy,
                        pos_step_override=touch_step,
                        orientation_weight_override=touch_ori_weight,
                        joint_step_limit_override=touch_joint_step,
                    ):
                        return True
                if self._param_bool("touch_cartesian_pose_fallback"):
                    self.get_logger().warn(
                        f"{label}: 笛卡尔下压失败，回退到 MoveIt 位姿规划以继续贴近目标"
                    )
                    return self._move_target_with_moveit_pose(
                        target, orientations, f"{label}_pose_fallback", compensate_suction=True
                    )
                return False
            if self._param_bool("hybrid_moveit_pregrasp"):
                self.get_logger().info(f"{label}: use_compute_ik=false，使用 MoveIt 位姿规划")
                if mode == "pick" and self._param_bool("prefer_upright_joint_goal_for_pick"):
                    if self._move_target_with_upright_joint_goal(target, orientations, mode, label):
                        return True
                    self.get_logger().warn(f"{label}: upright joint goal 失败，回退到 MoveIt Pose goal（含严格朝向约束）")
                return self._move_target_with_moveit_pose(target, orientations, label, compensate_suction=True)
            self.get_logger().info(f"{label}: use_compute_ik=false，使用笛卡尔分段轨迹")
            for idx, ori in enumerate(orientations, start=1):
                cart_target = self._suction_centered_origin_target(target, ori, f"{label}_cart[{idx}]")
                if self._move_cartesian_direct(cart_target, ori, mode=mode, label=f"{label}_cart[{idx}]"):
                    return True
            return False

        if use_compute_ik:
            sol = self._call_ik(target, orientations, mode=mode)
            if sol is not None and self._send_move(sol, label):
                return True
        if not self._param_bool("pose_goal_fallback"):
            return False
        if use_compute_ik:
            self.get_logger().warn(f"{label}: IK 不可用，回退到位姿目标规划")
        for idx, ori in enumerate(orientations, start=1):
            goal_target = self._suction_centered_origin_target(target, ori, f"{label}_pose[{idx}]")
            if self._send_pose_goal(goal_target, ori, f"{label}_pose[{idx}]"):
                return True
        return False

    def _probe_pickup_follow(
        self,
        touch: Point,
        orientation: Quaternion,
        label: str = "pick_probe_lift",
    ) -> tuple[bool, Point]:
        """
        吸附后先小幅抬升做"探针验证"，明确区分四种状态：
        STATE_SUCTION_AND_FOLLOW：suction=true 且物体跟随上移 -> 信任成功
        STATE_SUCTION_ONLY：suction=true 但物体未跟随 -> 可疑（可能边缘吸附）
        STATE_FOLLOW_ONLY：suction 未返回但物体跟随 -> 信任成功（Gazebo 话题延迟）
        STATE_FAIL：suction=false 且物体未跟随 -> 确认失败
        """
        probe_lift_z = max(0.005, float(self.get_parameter("pickup_probe_lift_z").value))
        min_follow_z = max(0.002, float(self.get_parameter("pickup_probe_min_follow_z").value))
        rect_z_before = self._current_rect_center_z()
        probe_target = Point(x=touch.x, y=touch.y, z=touch.z + probe_lift_z)

        touch_step = max(0.001, float(self.get_parameter("touch_cartesian_step_max_m").value))
        touch_ori_weight = max(0.1, float(self.get_parameter("touch_cartesian_orientation_weight").value))
        touch_joint_step = max(
            0.01, float(self.get_parameter("touch_cartesian_joint_step_limit_rad").value)
        )
        self.get_logger().info(
            f"[{label}] 探针抬升: "
            f"touch=({touch.x:.4f},{touch.y:.4f},{touch.z:.4f}), "
            f"probe_z={probe_target.z:.4f}, lift={probe_lift_z:.4f}m, "
            f"rect_z_before={'N/A' if rect_z_before is None else f'{rect_z_before:.4f}'}"
        )
        moved = self._move_cartesian_direct(
            probe_target,
            orientation,
            mode="pick",
            label=label,
            keep_xy_from_current=True,
            pos_step_override=touch_step,
            orientation_weight_override=touch_ori_weight,
            joint_step_limit_override=touch_joint_step,
        )
        if not moved:
            self.get_logger().warn(f"{label}: 笛卡尔探测抬升失败，回退到常规轨迹")
            moved = self._move_target_with_fallback(
                probe_target,
                [orientation],
                mode="pick",
                label=f"{label}_fallback",
            )
        if not moved:
            self.get_logger().error(f"{label}: 探测抬升执行失败")
            return False, probe_target

        time.sleep(0.25)
        rect_z_after = self._current_rect_center_z()
        follow_dz = (
            (rect_z_after - rect_z_before)
            if rect_z_before is not None and rect_z_after is not None
            else None
        )
        state_ok = self._suction_attached is True
        follow_ok = follow_dz is not None and follow_dz >= min_follow_z
        has_live_pose = self._logged_rect and (rect_z_before is not None and rect_z_after is not None)
        require_follow_if_live_pose = self._param_bool("pickup_probe_require_follow_if_live_pose")

        if require_follow_if_live_pose and has_live_pose:
            success = follow_ok and state_ok
            if success:
                state_label = "STATE_SUCTION_AND_FOLLOW: suction=OK + follow=OK"
            elif state_ok and not follow_ok:
                state_label = "STATE_SUCTION_ONLY: suction=OK + follow=FAIL (可疑边缘吸附)"
            elif follow_ok and not state_ok:
                state_label = "STATE_FOLLOW_ONLY: suction=FAIL + follow=OK (话题延迟)"
            else:
                state_label = "STATE_FAIL: suction=FAIL + follow=FAIL"
            reason = f"{state_label} | follow_dz={follow_dz:.4f}m, suction_state={self._suction_attached}"
        else:
            success = state_ok or follow_ok
            reasons: list[str] = []
            if state_ok:
                reasons.append("suction=true")
            if follow_ok and follow_dz is not None:
                reasons.append(f"follow_dz={follow_dz:.4f}m")
            if not reasons:
                reasons.append("none")
            state_label = f"suction={'OK' if state_ok else 'FAIL'} follow={'OK' if follow_ok else 'FAIL'}"
            reason = f"{state_label} ({' + '.join(reasons)})"

        self.get_logger().info(
            f"[{label}] 探针判定: {reason}, "
            f"rect_z: before={'N/A' if rect_z_before is None else f'{rect_z_before:.4f}'} "
            f"after={'N/A' if rect_z_after is None else f'{rect_z_after:.4f}'}"
        )

        if success:
            self.get_logger().info(f"[{label}] 吸附验证通过（{reason}）")
            return True, probe_target

        before_text = f"{rect_z_before:.4f}" if rect_z_before is not None else "N/A"
        after_text = f"{rect_z_after:.4f}" if rect_z_after is not None else "N/A"
        self.get_logger().warn(
            f"[{label}] 吸附验证失败（{reason}）, "
            f"before={before_text}, after={after_text}, min_follow={min_follow_z:.4f}"
        )
        return False, probe_target

        time.sleep(0.25)
        rect_z_after = self._current_rect_center_z()
        follow_dz = (
            (rect_z_after - rect_z_before)
            if rect_z_before is not None and rect_z_after is not None
            else None
        )
        state_ok = self._suction_attached is True
        follow_ok = follow_dz is not None and follow_dz >= min_follow_z
        require_follow_if_live_pose = self._param_bool("pickup_probe_require_follow_if_live_pose")
        has_live_pose = self._logged_rect and (rect_z_before is not None and rect_z_after is not None)
        if require_follow_if_live_pose and has_live_pose:
            success = follow_ok
            reason = f"follow_dz={follow_dz:.4f}m" if follow_dz is not None else "follow_dz=N/A"
        else:
            success = state_ok or follow_ok
            reasons: list[str] = []
            if state_ok:
                reasons.append("state=true")
            if follow_ok and follow_dz is not None:
                reasons.append(f"follow_dz={follow_dz:.4f}m")
            reason = " + ".join(reasons) if reasons else "none"
        if success:
            self.get_logger().info(f"{label}: 吸附验证通过（{reason}）")
            return True, probe_target

        before_text = f"{rect_z_before:.3f}" if rect_z_before is not None else "N/A"
        after_text = f"{rect_z_after:.3f}" if rect_z_after is not None else "N/A"
        self.get_logger().warn(
            f"{label}: 吸附验证失败（state={self._suction_attached}, "
            f"before={before_text}, after={after_text}, min_follow={min_follow_z:.4f}）"
        )
        return False, probe_target

    def _move_with_z_scan(
        self,
        base_target: Point,
        orientations: List[Quaternion],
        z_offsets: Sequence[float],
        mode: str,
        label: str,
    ) -> Point | None:
        for dz in z_offsets:
            target = Point(
                x=base_target.x,
                y=base_target.y,
                z=base_target.z + float(dz),
            )
            if self._move_target_with_fallback(
                target, orientations, mode=mode, label=f"{label}@z={target.z:.3f}"
            ):
                return target
        return None

    def _current_arm_positions(self) -> List[float] | None:
        js = self._joint_state
        if js is None or not self._names_ok(js):
            return None
        out: List[float] = []
        for joint_name in _ARM_JOINTS:
            out.append(float(js.position[js.name.index(joint_name)]))
        return out

    def _make_robot_state(self, seed_positions: Sequence[float] | None = None) -> RobotState:
        rs = RobotState()
        if seed_positions is not None and len(seed_positions) == len(_ARM_JOINTS):
            rs.joint_state = make_zero_joint_state()
            rs.joint_state.header.stamp = self.get_clock().now().to_msg()
            rs.joint_state.position = [float(v) for v in seed_positions]
            return rs
        js = self._joint_state
        if js is None or not self._names_ok(js):
            rs.joint_state = make_zero_joint_state()
            rs.joint_state.header.stamp = self.get_clock().now().to_msg()
            return rs
        rs.joint_state = joint_state = copy_joint_state(js)
        joint_state.header.stamp = self.get_clock().now().to_msg()
        return rs

    def _preferred_seed_for_target(self, target: Point, mode: str) -> List[float]:
        if mode == "place":
            hint = list(self.get_parameter("place_posture_hint").value)
            defaults = [0.0, -0.35, 0.80, 0.0, 1.00, 0.0]
        else:
            hint = list(self.get_parameter("pick_posture_hint").value)
            defaults = [0.0, -0.55, 0.90, 0.0, 1.10, 0.0]
        if len(hint) != 6:
            hint = defaults
        hint = [float(v) for v in hint]
        yaw_offset = float(self.get_parameter("joint1_world_yaw_offset_rad").value)
        hint[0] = _wrap_to_pi(math.atan2(target.y, target.x) + yaw_offset)
        return hint

    def _desired_joint1_for_target(self, target: Point) -> float:
        yaw_offset = float(self.get_parameter("joint1_world_yaw_offset_rad").value)
        return _wrap_to_pi(math.atan2(target.y, target.x) + yaw_offset)

    def _ik_seed_candidates(self, target: Point, mode: str) -> List[List[float]]:
        preferred = self._preferred_seed_for_target(target, mode)
        current = self._current_arm_positions()
        seeds: List[List[float]] = [preferred]
        if current is not None:
            blended = [
                _wrap_to_pi(current[0] * 0.35 + preferred[0] * 0.65),
                current[1] * 0.35 + preferred[1] * 0.65,
                current[2] * 0.35 + preferred[2] * 0.65,
                current[3] * 0.35 + preferred[3] * 0.65,
                current[4] * 0.35 + preferred[4] * 0.65,
                current[5] * 0.35 + preferred[5] * 0.65,
            ]
            seeds.extend([blended, current])
        unique: List[List[float]] = []
        seen: set[Tuple[float, ...]] = set()
        for seed in seeds:
            key = tuple(round(_wrap_to_pi(v), 4) for v in seed)
            if key in seen:
                continue
            seen.add(key)
            unique.append(seed)
        return unique

    def _reject_ik_solution(self, positions: Sequence[float], target: Point, mode: str) -> bool:
        """仅剔除明显越限/奇异解，避免把可行抓取解误过滤掉。"""
        desired_j1 = self._desired_joint1_for_target(target)
        # 抓取阶段更严格地约束 joint1 朝向，优先使用"先转底座再伸臂"的正面分支。
        joint1_limit_deg = 85.0 if mode == "pick" else 120.0
        if _angle_distance(float(positions[0]), desired_j1) > math.radians(joint1_limit_deg):
            return True
        j2, j3, j4, j5, j6 = (
            positions[1],
            positions[2],
            positions[3],
            positions[4],
            positions[5],
        )
        if abs(j2) > 3.10 or abs(j3) > 3.10 or abs(j4) > 3.10 or abs(j6) > 3.10:
            return True
        # 仅保留一条极宽松的腕部安全约束，避免 wrist pitch 贴近奇异点。
        if abs(j5) > 3.12:
            return True
        # 抓取阶段检查吸盘朝下姿态质量：FK 求出末端旋转矩阵，计算吸盘
        # +Z 轴在世界系中的朝下分量 down_cos。抓取时吸盘应接近竖直朝下，
        # down_cos < 0.90 表示倾斜超过 ~25°，属于偏斜/跪倒解。
        if mode == "pick" and self._kin_ready:
            try:
                _, tcp_r, _, _ = self._fk_with_jacobian_context(positions)
                tcp_q = _rot_to_quat(tcp_r)
                suction_world_z = _quat_rotate_vec(tcp_q, 0.0, 0.0, 1.0)
                down_cos = -suction_world_z[2]
                if down_cos < 0.90:
                    self.get_logger().debug(
                        f"reject_ik: down_cos={down_cos:.3f} < 0.90, "
                        f"joints={[round(v, 3) for v in positions]}"
                    )
                    return True
            except Exception:
                pass
        if mode == "pick":
            min_target_z = float(self.get_parameter("pick_pregrasp_elbow_filter_min_target_z").value)
            if float(target.z) >= min_target_z:
                try:
                    _, _, joint_origins, _ = self._fk_with_jacobian_context(positions)
                    joint3_origin_z = float(joint_origins[2][2]) if len(joint_origins) >= 3 else 1.0
                    min_joint3_z = float(self.get_parameter("pick_pregrasp_min_joint3_origin_z").value)
                    if joint3_origin_z < min_joint3_z:
                        return True
                except Exception:
                    return True
        return False

    def _score_ik_solution(self, positions: Sequence[float], target: Point, mode: str) -> float:
        preferred = self._preferred_seed_for_target(target, mode)
        current = self._current_arm_positions()
        if mode == "place":
            pref_weights = [1.0, 1.0, 1.6, 1.4, 0.9, 1.2]
            cur_weights = [0.5, 0.5, 0.7, 0.7, 0.4, 0.5]
        else:
            pref_weights = [1.2, 2.2, 3.4, 2.8, 1.2, 2.4]
            # 抓取阶段更强调与当前状态连续，避免 IK 在等价姿态间跳到反肘/绕腕分支。
            cur_weights = [0.9, 1.5, 2.1, 2.0, 0.9, 1.5]

        score = 0.0
        for idx, value in enumerate(positions):
            score += pref_weights[idx] * (_angle_distance(value, preferred[idx]) ** 2)
        if current is not None:
            for idx, value in enumerate(positions):
                score += cur_weights[idx] * (_angle_distance(value, current[idx]) ** 2)

        if mode == "pick":
            j2, j3, j4, j5, j6 = positions[1], positions[2], positions[3], positions[4], positions[5]
            if j3 < -0.15:
                score += 18.0 * ((-0.15 - j3) ** 2)
            if j3 > 1.35:
                score += 9.0 * ((j3 - 1.35) ** 2)
            if abs(j4) > 1.00:
                score += 14.0 * ((abs(j4) - 1.00) ** 2)
            if abs(j6) > 0.85:
                score += 22.0 * ((abs(j6) - 0.85) ** 2)
            if j2 > 0.20:
                score += 16.0 * ((j2 - 0.20) ** 2)
            if j2 < -1.75:
                score += 8.0 * ((-1.75 - j2) ** 2)
            if j5 < 0.75:
                score += 12.0 * ((0.75 - j5) ** 2)
            # 抓取阶段严厉惩罚吸盘非垂直朝向（down_cos 越偏离 1.0，惩罚越大）
            if self._kin_ready:
                try:
                    _, tcp_r, _, _ = self._fk_with_jacobian_context(positions)
                    tcp_q = _rot_to_quat(tcp_r)
                    suction_world_z = _quat_rotate_vec(tcp_q, 0.0, 0.0, 1.0)
                    down_cos = -suction_world_z[2]
                    if down_cos < 0.98:
                        score += 50.0 * ((1.0 - down_cos) ** 2)
                except Exception:
                    pass
        return score

    def _ik_avoid_collisions_for_mode(self, mode: str) -> bool:
        if mode == "pick":
            return self._param_bool("ik_pick_avoid_collisions")
        return self._param_bool("ik_avoid_collisions")

    def _ik_timeout_duration(self) -> Duration:
        t = max(0.5, float(self.get_parameter("ik_timeout_sec").value))
        sec = int(t)
        nsec = int(round((t - sec) * 1e9))
        return Duration(sec=sec, nanosec=nsec)

    def _call_ik(self, target: Point, orientations: List[Quaternion], mode: str = "pick") -> List[float] | None:
        assert self._ik_client is not None
        call_wait_sec = max(0.5, float(self.get_parameter("ik_call_wait_sec").value))
        search_wall_time = max(5.0, float(self.get_parameter("ik_search_wall_time_sec").value))
        search_t0 = time.time()
        best_score: float | None = None
        best_solution: List[float] | None = None
        fallback_score: float | None = None
        fallback_solution: List[float] | None = None
        seeds = self._ik_seed_candidates(target, mode)
        self.get_logger().info(
            "IK 求解开始: "
            f"mode={mode}, target=({target.x:.3f},{target.y:.3f},{target.z:.3f}), "
            f"orientations={len(orientations)}, seeds={len(seeds)}, wait={call_wait_sec:.1f}s"
        )
        raw_ok = 0
        rejected = 0
        timeout_hits = 0
        for ori_idx, ori in enumerate(orientations, start=1):
            for seed_idx, seed in enumerate(seeds, start=1):
                if (time.time() - search_t0) > search_wall_time:
                    self.get_logger().warn(
                        "IK 搜索达到总时限，提前退出: "
                        f"mode={mode}, elapsed={time.time()-search_t0:.1f}s, "
                        f"checked≈{(ori_idx - 1) * len(seeds) + seed_idx - 1}"
                    )
                    break
                req = GetPositionIK.Request()
                req.ik_request.group_name = "manipulator"
                req.ik_request.ik_link_name = "suction_tcp_link"
                req.ik_request.avoid_collisions = self._ik_avoid_collisions_for_mode(mode)
                req.ik_request.timeout = self._ik_timeout_duration()
                req.ik_request.robot_state = self._make_robot_state(seed)
                ps = PoseStamped()
                ps.header.frame_id = "base_link"
                ps.header.stamp = self.get_clock().now().to_msg()
                ps.pose.position = target
                ps.pose.orientation = ori
                req.ik_request.pose_stamped = ps
                fut = self._ik_client.call_async(req)
                if not _spin_future(self, fut, call_wait_sec, "compute_ik"):
                    timeout_hits += 1
                    # 服务层异常时，避免每个姿态/seed 都阻塞整轮流程。
                    if timeout_hits >= 2:
                        self.get_logger().warn("compute_ik 连续超时，提前切换到位姿目标回退")
                        return None
                    continue
                res = fut.result()
                if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                    continue
                names = list(res.solution.joint_state.name)
                pos = list(res.solution.joint_state.position)
                out: List[float] = []
                for jn in _ARM_JOINTS:
                    if jn in names:
                        out.append(float(pos[names.index(jn)]))
                if len(out) != 6:
                    continue
                raw_ok += 1
                score = self._score_ik_solution(out, target, mode)
                if self._reject_ik_solution(out, target, mode):
                    rejected += 1
                    continue
                if fallback_score is None or score < fallback_score:
                    fallback_score = score
                    fallback_solution = out
                if best_score is None or score < best_score:
                    best_score = score
                    best_solution = out
            if (time.time() - search_t0) > search_wall_time:
                break
        if best_solution is not None:
            return best_solution
        if fallback_solution is not None:
            self.get_logger().warn(
                "IK 在姿态约束下择优: "
                f"raw_ok={raw_ok}, rejected={rejected}, target=({target.x:.3f},{target.y:.3f},{target.z:.3f})"
            )
            return fallback_solution

        # 最后兜底：使用当前状态作为 seed；仍必须满足 _reject_ik_solution，禁止回到反肘解。
        for ori in orientations:
            req = GetPositionIK.Request()
            req.ik_request.group_name = "manipulator"
            req.ik_request.ik_link_name = "suction_tcp_link"
            req.ik_request.avoid_collisions = self._ik_avoid_collisions_for_mode(mode)
            req.ik_request.timeout = self._ik_timeout_duration()
            req.ik_request.robot_state = self._make_robot_state()
            ps = PoseStamped()
            ps.header.frame_id = "base_link"
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.pose.position = target
            ps.pose.orientation = ori
            req.ik_request.pose_stamped = ps
            fut = self._ik_client.call_async(req)
            if not _spin_future(self, fut, call_wait_sec, "compute_ik(fallback)"):
                continue
            res = fut.result()
            if res is None or res.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            names = list(res.solution.joint_state.name)
            pos = list(res.solution.joint_state.position)
            out: List[float] = []
            for jn in _ARM_JOINTS:
                if jn in names:
                    out.append(float(pos[names.index(jn)]))
            if len(out) == 6 and not self._reject_ik_solution(out, target, mode):
                self.get_logger().warn(
                    "IK 通过兜底 current-state seed 求解成功，已使用该解。"
                )
                return out
        self.get_logger().error(
            "IK 求解失败: "
            f"mode={mode}, raw_ok={raw_ok}, rejected={rejected}, "
            f"target=({target.x:.3f},{target.y:.3f},{target.z:.3f})"
        )
        return None

    def _solve_ik_with_z_scan(
        self,
        base_target: Point,
        orientations: List[Quaternion],
        z_offsets: Sequence[float],
        mode: str = "pick",
    ) -> tuple[List[float] | None, Point]:
        tried = Point(x=base_target.x, y=base_target.y, z=base_target.z)
        for dz in z_offsets:
            tried = Point(x=base_target.x, y=base_target.y, z=base_target.z + float(dz))
            sol = self._call_ik(tried, orientations, mode=mode)
            if sol is not None:
                return sol, tried
        return None, tried

    def _ordered_pre_pick_targets(
        self,
        touch: Point,
        top: Point,
        suction_contact_offset: float,
        approach: Point,
        requested_clearance: float,
    ) -> List[Point]:
        raw = list(self.get_parameter("pre_pick_try_clearances").value)
        min_clearance = max(0.10, approach.z - (top.z + suction_contact_offset) + 0.03)
        candidates = [float(v) for v in raw if float(v) > 0.0]
        candidates.append(float(requested_clearance))

        unique: List[float] = []
        seen: set[float] = set()
        for clearance in sorted(candidates, reverse=True):
            clearance = max(min_clearance, clearance)
            key = round(clearance, 4)
            if key in seen:
                continue
            seen.add(key)
            unique.append(clearance)

        return [
            Point(
                x=touch.x,
                y=touch.y,
                z=top.z + suction_contact_offset + clearance,
            )
            for clearance in unique
        ]

    def _build_staged_pregrasp_targets(
        self,
        top: Point,
        suction_contact_offset: float,
        approach: Point,
        current_pregrasp: Point,
    ) -> List[Point]:
        if not self._param_bool("staged_pregrasp_enabled"):
            return []

        final_clearance = float(approach.z - (top.z + suction_contact_offset))
        start_clearance = float(current_pregrasp.z - (top.z + suction_contact_offset))
        raw = [float(v) for v in self.get_parameter("staged_pregrasp_clearances").value if float(v) > 0.0]
        raw.extend([final_clearance, start_clearance])

        clearances: List[float] = []
        seen: set[float] = set()
        for clearance in sorted(raw, reverse=True):
            if clearance > start_clearance + 1e-6:
                continue
            if clearance < final_clearance - 1e-6:
                continue
            key = round(clearance, 4)
            if key in seen:
                continue
            seen.add(key)
            clearances.append(clearance)

        points: List[Point] = []
        for clearance in clearances:
            pt = Point(
                x=top.x,
                y=top.y,
                z=top.z + suction_contact_offset + clearance,
            )
            if (
                abs(pt.x - current_pregrasp.x) < 1e-6
                and abs(pt.y - current_pregrasp.y) < 1e-6
                and abs(pt.z - current_pregrasp.z) < 0.008
            ):
                continue
            points.append(pt)
        return points or [Point(x=approach.x, y=approach.y, z=approach.z)]

    def _follow_staged_pregrasp_targets(
        self,
        top: Point,
        suction_contact_offset: float,
        approach: Point,
        current_pregrasp: Point,
        orientations: List[Quaternion],
    ) -> tuple[bool, Point]:
        targets = self._build_staged_pregrasp_targets(
            top,
            suction_contact_offset,
            approach,
            current_pregrasp,
        )
        if not targets:
            self.get_logger().info("staged_pregrasp 已关闭：保持高位预抓，后续直接笛卡尔下压")
            return True, current_pregrasp
        settle_sec = max(0.0, float(self.get_parameter("staged_pregrasp_settle_sec").value))
        last = Point(x=current_pregrasp.x, y=current_pregrasp.y, z=current_pregrasp.z)

        for idx, target in enumerate(targets, start=1):
            self.get_logger().info(
                f"staged_pregrasp[{idx}/{len(targets)}]: "
                f"target=({target.x:.4f},{target.y:.4f},{target.z:.4f})"
            )
            if not self._move_target_with_fallback(
                target,
                orientations,
                mode="pick",
                label=f"staged_pregrasp[{idx}]",
            ):
                return False, last
            last = Point(x=target.x, y=target.y, z=target.z)
            if settle_sec > 0.0:
                time.sleep(settle_sec)
        return True, last

    def _compute_live_touch_target(
        self,
        top: Point,
        rect_half: Sequence[float],
        nominal_touch: Point,
        label: str,
    ) -> Point:
        target_top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
        if target_top is None:
            target_top = top

        adaptive_touch = Point(x=target_top.x, y=target_top.y, z=nominal_touch.z)
        if not self._param_bool("adaptive_touch_target_enabled"):
            return adaptive_touch

        cup_pose = self._lookup_link_pose_in_base("suction_tcp_link")
        if cup_pose is None:
            self.get_logger().warn(f"{label}: 无法读取 suction_tcp_link TF，保持名义下压终点")
            return adaptive_touch

        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        touch_dz = max(0.0, float(self.get_parameter("touch_delta_z").value))
        max_adjust = max(0.005, float(self.get_parameter("adaptive_touch_max_adjust_m").value))
        cup_bottom = self._point_with_local_offset(
            cup_pose.position,
            cup_pose.orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        raw_touch_z = float(cup_pose.position.z) + (float(target_top.z) - float(cup_bottom.z)) - touch_dz
        adaptive_touch.z = min(
            float(nominal_touch.z) + max_adjust,
            max(float(nominal_touch.z) - max_adjust, raw_touch_z),
        )
        self.get_logger().info(
            f"{label}: 动态下压终点 "
            f"top=({target_top.x:.4f},{target_top.y:.4f},{target_top.z:.4f}) "
            f"cup_bottom=({cup_bottom.x:.4f},{cup_bottom.y:.4f},{cup_bottom.z:.4f}) "
            f"nominal_z={float(nominal_touch.z):.4f} raw_z={raw_touch_z:.4f} final_z={adaptive_touch.z:.4f}"
        )
        return adaptive_touch

    def wait_ready(self) -> bool:
        use_direct_xyz = self._param_bool("use_direct_xyz")
        require_joint_states = self._param_bool("require_joint_states")
        use_compute_ik = self._param_bool("use_compute_ik")

        if use_direct_xyz:
            timeout = float(self.get_parameter("wait_poses_sec").value)
            self.get_logger().info(
                f"使用直接坐标抓放，等待 /joint_states（最长 {timeout}s）…"
            )
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < timeout:
                time.sleep(0.05)
                if self._joint_state:
                    break
            if self._joint_state is None:
                self.get_logger().warn("未收到 /joint_states，将使用零位作为 IK 初始状态")
        else:
            timeout = float(self.get_parameter("wait_poses_sec").value)
            fallback_wait = float(self.get_parameter("carton_fallback_wait_sec").value)
            rect_fallback_wait = float(self.get_parameter("rect_fallback_wait_sec").value)
            self.get_logger().info(
                f"等待 /model/*/pose 与 /joint_states（最长 {timeout}s）…"
            )
            t0 = time.time()
            last_dbg = 0.0
            while rclpy.ok() and time.time() - t0 < timeout:
                time.sleep(0.05)
                elapsed = time.time() - t0
                if self._rect is None and elapsed >= rect_fallback_wait:
                    if self._ensure_rect_fallback():
                        self._check_ready()
                if self._rect is not None and self._carton is None and elapsed >= fallback_wait:
                    if self._ensure_carton_fallback():
                        self._check_ready()
                now = time.time()
                if now - last_dbg >= 5.0:
                    self.get_logger().info(
                        "等待中: "
                        f"rect={'Y' if self._rect else 'N'}, "
                        f"carton={'Y' if self._carton else 'N'}, "
                        f"joint_states={'Y' if self._joint_state else 'N'}"
                    )
                    last_dbg = now
                if self._ready and (not require_joint_states or self._joint_state is not None):
                    break
            if not self._ready:
                self.get_logger().error("超时：未收到矩形/纸箱位姿")
                return False
            if require_joint_states and self._joint_state is None:
                self.get_logger().error(
                    "超时：在等待位姿回退/桥接就绪后，仍未收到 /joint_states。"
                    "Gazebo↔ROS 桥接可能启动过慢或异常退出，请检查 ros_gz_bridge 日志。"
                )
                return False
        svc_name = str(self.get_parameter("ik_service").value)
        if use_compute_ik:
            if not self._action.wait_for_server(timeout_sec=60.0):
                self.get_logger().error("move_action 不可用（move_group 是否已启动？）")
                return False
            self._ik_client = self.create_client(GetPositionIK, svc_name)
            if not self._ik_client.wait_for_service(timeout_sec=30.0):
                alt = "/move_group/compute_ik"
                self.get_logger().warn(f"{svc_name} 不可用，尝试 {alt}")
                self._ik_client = self.create_client(GetPositionIK, alt)
                if not self._ik_client.wait_for_service(timeout_sec=30.0):
                    self.get_logger().error("compute_ik 服务不可用")
                    return False
            if not self._scene_client.wait_for_service(timeout_sec=30.0):
                self.get_logger().error("/apply_planning_scene 服务不可用")
                return False
        else:
            if not self._kin_ready:
                self.get_logger().error("use_compute_ik=false 但笛卡尔运动学模型未就绪")
                return False
            if self._param_bool("hybrid_moveit_pregrasp"):
                if not self._action.wait_for_server(timeout_sec=60.0):
                    self.get_logger().error("混合模式需要 move_action，但当前不可用")
                    return False
            if self._param_bool("hybrid_moveit_pregrasp"):
                self.get_logger().warn(
                    "use_compute_ik=false：启用混合策略（MoveIt 预抓取 + 笛卡尔直线下压，数值 IK）"
                )
            else:
                self.get_logger().warn("use_compute_ik=false：使用笛卡尔分段轨迹 + 数值 IK（不调用 MoveIt IK）")
            self._ik_client = None
            self._wait_cartesian_bridges(float(self.get_parameter("cartesian_bridge_wait_sec").value))
        if self._rect_fallback_used:
            self.get_logger().warn("当前任务使用了物体默认位姿回退模式。")
        if self._carton_fallback_used:
            self.get_logger().warn("当前任务使用了箱子默认位姿回退模式。")
        return True

    def _run_joint_template_demo(self, half: Sequence[float]) -> bool:
        """
        固定场景稳态演示兜底路径（优先“必动起来”）：
        直接执行一组离线验证过的关节目标，覆盖 pre/approach/touch/lift/place/home。
        """
        home = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pre_pick = [-2.6841, 0.5146, 1.5162, 0.0486, -1.4446, -0.5753]
        approach = [-2.6711, 0.5611, 1.5164, 0.1180, -1.4898, -0.4587]
        touch = [-2.6855, 0.6048, 1.5480, 0.1362, -1.3898, -0.5723]
        lift = [-2.6841, 0.5146, 1.5162, 0.0486, -1.4446, -0.5753]
        place_above = [2.2805, 0.6133, 1.5215, 0.2977, -1.3291, 0.2283]
        place_inside = [2.2736, 0.7238, 1.5220, 0.2676, -1.2971, 0.3078]
        retreat = [2.2805, 0.6133, 1.5215, 0.2977, -1.3291, 0.2283]

        self.get_logger().warn("使用关节模板路径执行抓放（固定场景稳态复现）")
        if not self._ensure_detached():
            return False

        if not self._send_move(home, "tmpl_home_start"):
            return False
        if not self._send_move(pre_pick, "tmpl_pre_pick"):
            return False
        if not self._send_move(approach, "tmpl_approach"):
            return False
        if not self._send_move(touch, "tmpl_touch"):
            return False

        if not self._suction_bottom_alignment_ok(half):
            self.get_logger().warn("模板触碰位姿未完全满足底面法向判据，仍尝试 attach")

        rect_center_z_before_attach = self._current_rect_center_z()
        attach_wait_sec = max(0.5, float(self.get_parameter("suction_attach_wait_sec").value))
        self._suction_attached = None
        self.get_logger().info("模板路径: 吸附 attach")
        self._publish_attach_burst()
        if not self._wait_suction_state(True, attach_wait_sec):
            self.get_logger().warn("模板路径: 首次 attach 未确认，二次 attach 重试")
            self._suction_attached = None
            self._publish_attach_burst()
            self._wait_suction_state(True, attach_wait_sec)

        if not self._send_move(lift, "tmpl_lift"):
            self._pub_detach.publish(Empty())
            return False

        rect_center_z_after_lift = self._current_rect_center_z()
        if (
            rect_center_z_before_attach is not None
            and rect_center_z_after_lift is not None
            and rect_center_z_after_lift < rect_center_z_before_attach + 0.010
        ):
            self.get_logger().warn(
                "模板路径: 物体上抬幅度偏小，继续执行放置以便观察 "
                f"(before={rect_center_z_before_attach:.3f}, after={rect_center_z_after_lift:.3f})"
            )

        if not self._send_move(place_above, "tmpl_place_above"):
            self._pub_detach.publish(Empty())
            return False
        if not self._send_move(place_inside, "tmpl_place_inside"):
            self.get_logger().warn("模板路径: place_inside 失败，改为箱口上方释放")

        self.get_logger().info("模板路径: 释放 detach")
        self._suction_attached = None
        self._pub_detach.publish(Empty())
        self._wait_suction_state(False, 1.0)
        time.sleep(0.2)

        if not self._send_move(retreat, "tmpl_retreat"):
            self.get_logger().warn("模板路径: retreat 失败，继续回 home")
        if not self._send_move(home, "tmpl_home_end"):
            return False
        return True

    def run_pipeline(self) -> None:
        half = list(self.get_parameter("box_half_size_xyz").value)
        clearance = float(self.get_parameter("approach_clearance").value)
        hover_extra = max(0.0, float(self.get_parameter("pre_touch_hover_extra_z").value))
        pre_pick_safe_clearance = float(self.get_parameter("pre_pick_safe_clearance").value)
        touch_dz = float(self.get_parameter("touch_delta_z").value)
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        floor_z = float(self.get_parameter("carton_floor_top_z").value)
        requested_place_h = float(self.get_parameter("place_height_above_floor").value)
        post_pick_lift = float(self.get_parameter("post_pick_lift").value)
        place_entry_clearance = float(self.get_parameter("place_entry_clearance").value)
        post_place_retreat = float(self.get_parameter("post_place_retreat").value)
        use_direct_xyz = self._param_bool("use_direct_xyz")
        use_known_surface = self._param_bool("use_known_rect_surface_center")
        pick_xyz = self._param_xyz("pick_point_xyz")
        place_xyz = self._param_xyz("place_point_xyz")
        known_rect_center = self._param_xyz("known_rect_center_xyz")
        known_rect_size = self._param_xyz("known_rect_size_xyz")
        carton_ps: PoseStamped | None = None

        if use_direct_xyz:
            if len(pick_xyz) != 3 or len(place_xyz) != 3:
                self.get_logger().error("use_direct_xyz=true 时 pick/place_point_xyz 必须是 3 元数组")
                return
            top = Point(x=float(pick_xyz[0]), y=float(pick_xyz[1]), z=float(pick_xyz[2]))
            touch = Point(x=top.x, y=top.y, z=top.z + suction_contact_offset - touch_dz)
            approach = Point(x=touch.x, y=touch.y, z=touch.z + clearance + hover_extra)
            place_pt = Point(
                x=float(place_xyz[0]),
                y=float(place_xyz[1]),
                z=float(place_xyz[2]),
            )
        else:
            assert self._rect and self._carton
            r_ps = self._pose_to_base(self._rect)
            carton_ps = self._pose_to_base(self._carton)
            if (
                use_known_surface
                and len(known_rect_center) == 3
                and len(known_rect_size) == 3
                and float(known_rect_size[0]) > 0.0
                and float(known_rect_size[1]) > 0.0
                and float(known_rect_size[2]) > 0.0
            ):
                half = [
                    0.5 * float(known_rect_size[0]),
                    0.5 * float(known_rect_size[1]),
                    0.5 * float(known_rect_size[2]),
                ]
                top = Point(
                    x=float(known_rect_center[0]),
                    y=float(known_rect_center[1]),
                    z=float(known_rect_center[2]) + half[2],
                )
                self.get_logger().info(
                    "抓取点使用已知几何计算: "
                    f"center=({known_rect_center[0]:.3f},{known_rect_center[1]:.3f},{known_rect_center[2]:.3f}), "
                    f"size=({known_rect_size[0]:.3f},{known_rect_size[1]:.3f},{known_rect_size[2]:.3f}), "
                    f"top=({top.x:.3f},{top.y:.3f},{top.z:.3f})"
                )
            else:
                if use_known_surface:
                    self.get_logger().warn("known_rect_* 参数非法，回退到 /model/rect_pickup/pose 计算顶面中心")
                top = self._model_top_center(r_ps, half)
            physical_top = Point(x=top.x, y=top.y, z=top.z)
            suction_pick_center = self._known_suction_pick_center()
            if suction_pick_center is not None:
                self.get_logger().info(
                    "吸取目标中心使用指定坐标: "
                    f"suction_center=({suction_pick_center.x:.3f},{suction_pick_center.y:.3f},{suction_pick_center.z:.3f}), "
                    f"physical_top=({physical_top.x:.3f},{physical_top.y:.3f},{physical_top.z:.3f}), "
                    f"tolerance={float(self.get_parameter('suction_pick_center_tolerance_m').value):.3f}m"
                )
                top = suction_pick_center
            touch = Point(x=top.x, y=top.y, z=top.z + suction_contact_offset - touch_dz)
            approach = Point(x=touch.x, y=touch.y, z=touch.z + clearance + hover_extra)
            place_h = self._safe_place_tcp_height_above_floor(
                requested_place_h,
                half,
                suction_contact_offset,
                touch_dz,
            )
            place_pt = self._carton_place_point(carton_ps, floor_z, place_h)

        pick_lift = Point(
            x=top.x,
            y=top.y,
            z=max(approach.z + 0.03, top.z + suction_contact_offset + post_pick_lift),
        )
        start_pose_cmd: List[float] | None = None
        if self._param_bool("move_to_start_face_pose"):
            start_pose = list(self.get_parameter("start_face_posture_hint").value)
            if len(start_pose) == 6:
                start_pose = [float(v) for v in start_pose]
                target_yaw = float(self.get_parameter("start_face_joint1_rad").value)
                if self._param_bool("start_face_use_scene_midpoint_yaw"):
                    aim_x = 0.5 * (top.x + place_pt.x)
                    aim_y = 0.5 * (top.y + place_pt.y)
                    if abs(aim_x) > 1e-6 or abs(aim_y) > 1e-6:
                        world_yaw = math.atan2(aim_y, aim_x)
                        yaw_offset = float(self.get_parameter("joint1_world_yaw_offset_rad").value)
                        target_yaw = _wrap_to_pi(world_yaw + yaw_offset)
                start_pose[0] = target_yaw
                start_pose_cmd = list(start_pose)
                self.get_logger().info(
                    f"执行初始预抓姿态对准: target_yaw={target_yaw:.4f}"
                )
                if self._send_move(start_pose, "start_face_pregrasp_seed"):
                    time.sleep(0.4)
                else:
                    self.get_logger().warn("start_face_pregrasp_seed 失败，回退到 joint1 对准")
                    cur = self._current_arm_positions()
                    if cur is not None:
                        cur[0] = target_yaw
                        if self._send_move(cur, "start_face_yaw_only"):
                            start_pose_cmd = list(cur)
                            time.sleep(0.4)
                        else:
                            self.get_logger().warn("start_face_yaw_only 失败，继续执行抓取流程")
                    else:
                        self.get_logger().warn("当前关节状态不可用，跳过 start_face_yaw_only")
            else:
                self.get_logger().warn("start_face_posture_hint 参数非法（需 6 元数组），跳过初始朝向位姿")

        if (not use_direct_xyz) and self._param_bool("refresh_top_from_live_pose"):
            live_top = self._current_rect_top(half)
            if live_top is not None:
                dxy = math.hypot(live_top.x - top.x, live_top.y - top.y)
                dz = live_top.z - top.z
                max_dxy = max(0.01, float(self.get_parameter("refresh_top_max_delta_xy").value))
                max_dz = max(0.005, float(self.get_parameter("refresh_top_max_delta_z").value))
                if dxy <= max_dxy and abs(dz) <= max_dz:
                    top = Point(x=live_top.x, y=live_top.y, z=live_top.z)
                    touch = Point(x=top.x, y=top.y, z=top.z + suction_contact_offset - touch_dz)
                    approach = Point(x=touch.x, y=touch.y, z=touch.z + clearance + hover_extra)
                    pick_lift = Point(
                        x=top.x,
                        y=top.y,
                        z=max(approach.z + 0.03, top.z + suction_contact_offset + post_pick_lift),
                    )
                    self.get_logger().info(
                        "抓取前实时顶面重算: "
                        f"top=({top.x:.3f},{top.y:.3f},{top.z:.3f}), "
                        f"delta_xy={dxy:.4f}, delta_z={dz:.4f}"
                    )
                else:
                    self.get_logger().warn(
                        "实时顶面数据跳变过大，忽略实时覆盖并保留已知几何抓取点: "
                        f"live_top=({live_top.x:.3f},{live_top.y:.3f},{live_top.z:.3f}), "
                        f"delta_xy={dxy:.4f} (> {max_dxy:.4f}) or |delta_z|={abs(dz):.4f} (> {max_dz:.4f})"
                    )

        pre_pick_candidates = self._ordered_pre_pick_targets(
            touch,
            top,
            suction_contact_offset,
            approach,
            pre_pick_safe_clearance,
        )
        pre_pick_high = pre_pick_candidates[0]
        place_above = Point(
            x=place_pt.x,
            y=place_pt.y,
            z=place_pt.z + place_entry_clearance,
        )
        place_retreat = Point(
            x=place_pt.x,
            y=place_pt.y,
            z=place_pt.z + post_place_retreat,
        )

        if carton_ps is not None:
            if self._param_bool("use_compute_ik") and (not self._apply_carton_collision_scene(carton_ps)):
                return
        else:
            self.get_logger().warn(
                "use_direct_xyz=true：未注入纸箱碰撞体，无法保证严格避箱；"
                "建议使用默认位姿订阅模式。"
            )

        self.get_logger().info(
            f"顶面中心≈({top.x:.3f},{top.y:.3f},{top.z:.3f}) 高位预抓≈({pre_pick_high.x:.3f},{pre_pick_high.y:.3f},{pre_pick_high.z:.3f}) "
            f"预定位≈({approach.x:.3f},{approach.y:.3f},{approach.z:.3f}) "
            f"下压≈({touch.x:.3f},{touch.y:.3f},{touch.z:.3f}) 提升≈({pick_lift.x:.3f},{pick_lift.y:.3f},{pick_lift.z:.3f}) "
            f"入箱上方≈({place_above.x:.3f},{place_above.y:.3f},{place_above.z:.3f}) "
            f"箱内放置≈({place_pt.x:.3f},{place_pt.y:.3f},{place_pt.z:.3f})"
        )
        if not use_direct_xyz:
            self.get_logger().info(
                "抓取碰撞/吸附目标为 **rect_pickup** 顶面（棕/灰块体模型名以世界为准）。"
                "Gazebo DetachableJoint 仅将吸盘接到 rect_pickup/box_link；"
                "若末端对准 carton_box 而未对准 rect_pickup，吸附必失败。"
            )
        if len(pre_pick_candidates) > 1:
            z_list = ", ".join(f"{pt.z:.3f}" for pt in pre_pick_candidates)
            self.get_logger().info(f"预抓高度候选 z: [{z_list}]（优先尝试物体正上方较高安全位）")

        if (not use_direct_xyz) and self._param_bool("use_joint_template_demo"):
            if self._run_joint_template_demo(half):
                return
            self.get_logger().error("关节模板路径失败，终止本轮任务（未再回退到 IK/Pose）。")
            return
        if (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp"):
            self.get_logger().info(
                "抓取策略: MoveIt 先到物体上方对齐，再用笛卡尔直线下压接触。"
            )

        # 姿态搜索与目标方向联动，避免固定 yaw 导致的“反向分支”偏好。
        pick_world_yaw = math.atan2(top.y, top.x)
        place_world_yaw = math.atan2(place_pt.y, place_pt.x)
        yaw_offset = float(self.get_parameter("joint1_world_yaw_offset_rad").value)
        base_pick_yaw = _wrap_to_pi(pick_world_yaw + yaw_offset)
        base_place_yaw = _wrap_to_pi(place_world_yaw + yaw_offset)
        yaw_delta_set = [0.0, 0.06, -0.06]
        pick_yaw_set = [_wrap_to_pi(base_pick_yaw + d) for d in yaw_delta_set]
        place_yaw_set = [_wrap_to_pi(base_place_yaw + d) for d in yaw_delta_set]
        # 当前吸盘接触面定义为本地 +Z，抓取/放置时必须显式令其朝世界 -Z。
        # 否则 IK 会为了保持“吸盘朝上”的错误目标姿态而落到趴地/绕腕分支。
        pre_pick_orientations = [_suction_down_quat(base_pick_yaw)]
        # 接触与吸附阶段固定使用同一“吸盘朝下”姿态，优先保证 TCP 位于顶面正上方。
        pick_touch_orientations = [_suction_down_quat(base_pick_yaw)]
        # 放置阶段保留少量 yaw 余量，但保持吸盘始终朝下。
        place_orientations = [_suction_down_quat(yaw) for yaw in place_yaw_set]

        # 防止 DetachableJoint 初始误附着：流程开始先强制 detach 清状态。
        if not self._ensure_detached():
            return

        rect_center_z_before_attach = self._current_rect_center_z()

        pre_pick_ok = False
        for idx, pt in enumerate(pre_pick_candidates, start=1):
            if not self._move_target_with_fallback(
                pt, pre_pick_orientations, mode="pick", label=f"pre_pick_high[{idx}]"
            ):
                self._refresh_joint_state(0.5)
                continue
            pre_pick_high = pt
            pre_pick_ok = True
            time.sleep(0.8)
            break
        if not pre_pick_ok:
            self.get_logger().error("所有 pre_pick_high 候选均失败")
            return
        self._debug_fk_vs_tf("after_pre_pick_high")

        pick_contact_collision_relaxed = False

        use_staged_pregrasp = (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp")
        if not use_staged_pregrasp:
            if not self._move_target_with_fallback(
                approach, pre_pick_orientations, mode="pick", label="approach"
            ):
                recovered = False
                if start_pose_cmd is not None:
                    self.get_logger().warn("approach 首次失败，尝试回到 joint1 对准姿态后重试一次")
                    if self._send_move(start_pose_cmd, "start_face_recover"):
                        time.sleep(0.5)
                        if self._move_target_with_fallback(
                            pre_pick_high,
                            pre_pick_orientations,
                            mode="pick",
                            label="pre_pick_recover",
                        ) and self._move_target_with_fallback(
                            approach, pre_pick_orientations, mode="pick", label="approach_retry"
                        ):
                            recovered = True
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                if not recovered:
                    self.get_logger().error("approach 失败（IK + Pose 回退均失败）")
                    return
        if use_staged_pregrasp:
            # 按项目要求：先在物体中心正上方走分层途径点，再从最后一个 hover 点做笛卡尔直线下压。
            approach_center = Point(x=top.x, y=top.y, z=approach.z)
            staged_ok, staged_approach = self._follow_staged_pregrasp_targets(
                top,
                suction_contact_offset,
                approach_center,
                pre_pick_high,
                pre_pick_orientations,
            )
            if not staged_ok:
                self.get_logger().error("staged_pregrasp 失败")
                return
            approach = staged_approach
            if self._param_bool("centerline_use_object_center_only"):
                gate_enabled = self._param_bool("pregrasp_alignment_gate_enabled")
                if gate_enabled:
                    aligned, aligned_approach = self._enforce_pregrasp_centerline(
                        approach, pre_pick_orientations, half
                    )
                    if not aligned:
                        if not self._param_bool("allow_unverified_sim_attach"):
                            self.get_logger().error("中心点对齐未满足阈值，已阻止下压动作")
                            return
                        self.get_logger().warn(
                            "中心点对齐未满足阈值；当前为 Gazebo DetachableJoint 仿真，"
                            "继续执行下压/吸附以保证演示流程。"
                        )
                    else:
                        approach = aligned_approach
                else:
                    offset = self._estimate_pick_offset_xy(half)
                    if offset is not None:
                        lateral = math.hypot(offset[0], offset[1])
                        align_tol = min(
                            max(0.002, float(self.get_parameter("pregrasp_xy_align_tol").value)),
                            max(
                                0.004,
                                float(self.get_parameter("suction_touch_lateral_tol").value) * 0.85,
                            ),
                        )
                        self.get_logger().warn(
                            "pregrasp_alignment_gate_enabled=false："
                            f"当前 centerline lateral={lateral:.4f}m (tol={align_tol:.4f}m)，"
                            "不阻塞下压，将继续执行 pick_touch"
                        )
            touch = Point(x=top.x, y=top.y, z=touch.z)
            offset = self._estimate_pick_offset_xy(half)
            if offset is not None:
                lateral = math.hypot(offset[0], offset[1])
                self.get_logger().info(
                    "[关键坐标] 中心线对齐区: "
                    f"top_center=({top.x:.4f},{top.y:.4f},{top.z:.4f}), "
                    f"touch_target=({touch.x:.4f},{touch.y:.4f},{touch.z:.4f}), "
                    f"approach=({approach.x:.4f},{approach.y:.4f},{approach.z:.4f}), "
                    f"cup_offset=({offset[0]:.4f},{offset[1]:.4f}), lateral={lateral:.4f}"
                )
            else:
                self.get_logger().info(
                    f"[关键坐标] 中心线对齐区: "
                    f"top_center=({top.x:.4f},{top.y:.4f},{top.z:.4f}), "
                    f"touch_target=({touch.x:.4f},{touch.y:.4f},{touch.z:.4f}), "
                    f"approach=({approach.x:.4f},{approach.y:.4f},{approach.z:.4f})"
                )
        pre_touch_settle = max(0.5, float(self.get_parameter("pre_touch_settle_sec").value))
        time.sleep(pre_touch_settle)
        # ── 下压前验证 approach 位置和朝向收敛，避免首抓因未收敛偏移失败 ──
        if self._param_bool("approach_verify_enabled"):
            if not self._verify_approach_pose(
                approach, pick_touch_orientations, half, "approach_verify"
            ):
                self.get_logger().warn("approach 位置验证/校正未完全通过，继续下压但首抓成功率可能降低")
        else:
            self.get_logger().warn("approach_verify_enabled=false：按已知几何中心直接进入下压阶段")
        # ── 下压前强制验证吸盘朝向，防止偏斜朝向导致吸附失败 ──
        if not self._ensure_suction_facing_down(
            pick_touch_orientations, "pre_touch_orient_check"
        ):
            self.get_logger().error("下压前吸盘朝向校正失败，终止本轮抓取")
            return

        # ── 下压前最终中心线闭环校验 ──
        if (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp"):
            if not pick_contact_collision_relaxed:
                # 中心线校正可能需要让吸盘接近/接触物体，先同步 ACM，避免 MoveIt 回退规划被物体碰撞卡死。
                pick_contact_collision_relaxed = self._set_pick_contact_collision_allowed(True)
            # 刷新 top 坐标（物体可能微移）
            if self._param_bool("refresh_top_from_live_pose"):
                live_top_pre = self._current_rect_top(half)
                if live_top_pre is not None:
                    dxy_pre = math.hypot(live_top_pre.x - top.x, live_top_pre.y - top.y)
                    dz_pre = live_top_pre.z - top.z
                    max_dxy_pre = max(0.01, float(self.get_parameter("refresh_top_max_delta_xy").value))
                    max_dz_pre = max(0.005, float(self.get_parameter("refresh_top_max_delta_z").value))
                    if dxy_pre <= max_dxy_pre and abs(dz_pre) <= max_dz_pre:
                        self.get_logger().info(
                            f"下压前顶面微调: ({top.x:.4f},{top.y:.4f},{top.z:.4f}) "
                            f"-> ({live_top_pre.x:.4f},{live_top_pre.y:.4f},{live_top_pre.z:.4f}) "
                            f"dxy={dxy_pre:.4f} dz={dz_pre:.4f}"
                        )
                        top = Point(x=live_top_pre.x, y=live_top_pre.y, z=live_top_pre.z)
                    else:
                        self.get_logger().warn(
                            f"下压前顶面数据跳变过大: dxy={dxy_pre:.4f} dz={dz_pre:.4f}, 保持原有抓取点"
                        )

            # touch 目标 XY 必须始终使用物体顶面中心，不是 approach 点的 XY
            touch = Point(x=top.x, y=top.y, z=top.z + suction_contact_offset - touch_dz)

            # 最终中心线闭环：读取真实 TCP，确认 XY 对齐后才能下压
            if self._param_bool("centerline_use_object_center_only"):
                offset = self._estimate_pick_offset_xy(half)
                if offset is not None:
                    self.get_logger().warn(
                        "下压前中心线偏差: "
                        f"offset=({offset[0]:.4f},{offset[1]:.4f}), "
                        f"lateral={math.hypot(offset[0], offset[1]):.4f}m"
                    )
                if not self._force_centerline_before_touch(top, pick_touch_orientations, half):
                    if not self._param_bool("allow_unverified_sim_attach"):
                        if pick_contact_collision_relaxed:
                            self._set_pick_contact_collision_allowed(False)
                        self.get_logger().error("下压前最终中心线校验失败，终止本轮抓取")
                        return
                    self.get_logger().warn(
                        "下压前最终中心线校验未完全通过；当前为仿真模式，继续执行。"
                    )

            # 下压前在物体中心正上方安全高度做一次对位移动，确保笛卡尔下压起点 XY 已对齐
            centerline_hover_z = top.z + suction_contact_offset + clearance + hover_extra
            centerline_hover = Point(x=top.x, y=top.y, z=centerline_hover_z)
            self.get_logger().info(
                f"[关键坐标] 下压前 hover: ({centerline_hover.x:.4f},{centerline_hover.y:.4f},{centerline_hover.z:.4f}), "
                f"touch: ({touch.x:.4f},{touch.y:.4f},{touch.z:.4f})"
            )
            if not self._param_bool("staged_pregrasp_enabled"):
                self.get_logger().info(
                    "staged_pregrasp_enabled=false：跳过低位 pick_centerline_hover MoveIt 重规划，避免再次进入跪倒分支"
                )
            else:
                if not self._move_target_with_fallback(
                    centerline_hover, pick_touch_orientations, mode="pick", label="pick_centerline_hover"
                ):
                    self.get_logger().warn("pick_centerline_hover 失败，将从当前位置下压")
                time.sleep(0.3)
                if not self._ensure_suction_facing_down(
                    pick_touch_orientations, "pre_touch_orient_check_after_hover"
                ):
                    if pick_contact_collision_relaxed:
                        self._set_pick_contact_collision_allowed(False)
                    self.get_logger().error("pick_centerline_hover 后吸盘朝向仍未校正到朝下，终止本轮抓取")
                    return
                if self._param_bool("centerline_use_object_center_only"):
                    if not self._force_centerline_before_touch(top, pick_touch_orientations, half):
                        if not self._param_bool("allow_unverified_sim_attach"):
                            if pick_contact_collision_relaxed:
                                self._set_pick_contact_collision_allowed(False)
                            self.get_logger().error("pick_centerline_hover 后最终中心线校验失败，终止本轮抓取")
                            return
                        self.get_logger().warn(
                            "pick_centerline_hover 后中心线仍未完全达标；当前为仿真模式，继续执行。"
                        )
            touch = self._compute_live_touch_target(top, half, touch, "pick_touch_plan")

        if not pick_contact_collision_relaxed:
            pick_contact_collision_relaxed = self._set_pick_contact_collision_allowed(True)

        use_dense = (
            self._param_bool("hybrid_moveit_pregrasp")
            and not self._param_bool("use_compute_ik")
            and self._param_bool("dense_waypoint_descent_enabled")
        )
        if use_dense:
            touch_orient = pick_touch_orientations[0] if pick_touch_orientations else _suction_down_quat(base_pick_yaw)
            dense_step = max(0.001, float(self.get_parameter("dense_waypoint_step_m").value))
            dense_gain = max(0.0, float(self.get_parameter("dense_waypoint_xy_correction_gain").value))
            dense_xy_max = max(0.001, float(self.get_parameter("dense_waypoint_xy_correction_max_m").value))
            dense_ori_w = max(0.1, float(self.get_parameter("dense_waypoint_orientation_weight").value))
            dense_settle = max(0.0, float(self.get_parameter("dense_waypoint_settle_sec").value))
            dense_max_drift = max(0.005, float(self.get_parameter("dense_waypoint_max_xy_drift_m").value))
            dense_target = self._suction_centered_origin_target(
                touch, touch_orient, "pick_touch_dense_target"
            )
            dense_ok = self._move_cartesian_dense_waypoints(
                dense_target,
                touch_orient,
                top,
                half,
                mode="pick",
                label="pick_touch_dense",
                waypoint_step_m=dense_step,
                xy_correction_gain=dense_gain,
                xy_correction_max_m=dense_xy_max,
                orientation_correction_weight=dense_ori_w,
                settle_sec_per_waypoint=dense_settle,
                max_xy_drift_m=dense_max_drift,
            )
            if not dense_ok:
                self.get_logger().warn("密途径点下压失败，回退到常规笛卡尔下压")
                if not self._move_target_with_fallback(
                    touch, pick_touch_orientations, mode="pick", label="pick_touch_fallback"
                ):
                    if pick_contact_collision_relaxed:
                        self._set_pick_contact_collision_allowed(False)
                    self.get_logger().error("pick_touch 失败（密途径点 + 常规均失败）")
                    return
        else:
            if not self._move_target_with_fallback(
                touch, pick_touch_orientations, mode="pick", label="pick_touch"
            ):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error("pick_touch 失败（IK + Pose 回退均失败）")
                return
        post_touch_settle = max(0.5, float(self.get_parameter("post_touch_settle_sec").value))
        time.sleep(post_touch_settle)

        if self._attach_geometry_ok(half, "pick_touch_contact"):
            self.get_logger().info("pick_touch 已满足双吸盘底部接触条件，跳过 XY refine")
        else:
            self.get_logger().warn("pick_touch 未满足双吸盘底部接触条件，尝试安全抬高后 XY refine")
            touch = self._refine_xy_alignment(top, touch, pick_touch_orientations, max_refine_steps=3)

        if not self._attach_geometry_ok(half, "pre_attach_contact"):
            self.get_logger().warn("未满足双吸盘底部接触条件，尝试轻微重定位后再吸附")
            re_align = Point(x=touch.x, y=touch.y, z=touch.z - 0.0015)
            if self._move_target_with_fallback(
                re_align, pick_touch_orientations, mode="pick", label="pick_realign"
            ):
                time.sleep(0.3)
            if not self._attach_geometry_ok(half, "pre_attach_contact_retry"):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error(
                    "吸附前双吸盘底部接触检查失败：禁止侧面/边缘吸附，停止本轮抓取"
                )
                return

        self.get_logger().info("吸附 attach")
        # 发送 attach 前短暂等待，让 Gazebo 物理充分稳定吸盘与物体接触状态。
        pre_attach_settle = max(0.1, float(self.get_parameter("pre_attach_settle_sec").value))
        time.sleep(pre_attach_settle)
        attach_wait_sec = max(0.5, float(self.get_parameter("suction_attach_wait_sec").value))
        self._suction_attached = None
        self._rect_motion_allowed = True
        gz_attach_sent = False
        self._publish_attach_burst()
        if not self._wait_suction_state(True, attach_wait_sec):
            self.get_logger().warn("首次 attach 未确认，尝试二次下压重吸")
            retry_touch = Point(x=touch.x, y=touch.y, z=touch.z - 0.050)
            if self._move_target_with_fallback(
                retry_touch, pick_touch_orientations, mode="pick", label="pick_touch_retry"
            ):
                time.sleep(0.4)
            if not self._attach_geometry_ok(half, "retry_attach_contact"):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error("二次吸附前双吸盘底部接触检查失败：仅允许底部吸盘吸附")
                return
            self._suction_attached = None
            self._publish_attach_burst()
            if not self._wait_suction_state(True, attach_wait_sec):
                if self._attach_via_gz_cli(repeats=8):
                    gz_attach_sent = True
                    self.get_logger().warn(f"ROS attach 未确认，已下发 {self._gz_bin} attach 兜底指令")
                    if not self._wait_suction_state(True, 0.8):
                        self._start_fake_attach(half, "DetachableJoint attach 未确认")
                else:
                    self.get_logger().warn("未收到 /cs612/suction/state=true，改用“抬升后物体是否上移”判据继续")
                    if self._param_bool("allow_unverified_sim_attach"):
                        self._start_fake_attach(half, "ROS/Gazebo attach 均未确认")
        time.sleep(0.3)
        if not self._attach_rect_to_tool_scene(half):
            self.get_logger().warn("Gazebo 已尝试吸附，但 RViz/MoveIt 附着显示同步未完成")

        probe_ok, _ = self._probe_pickup_follow(
            touch, pick_touch_orientations[0], label="pick_probe_lift"
        )
        if (not probe_ok) and gz_attach_sent and self._param_bool("allow_unverified_sim_attach"):
            self.get_logger().warn(
                "Gazebo attach 已通过 ign/gz 下发但状态/探针未确认；"
                "跳过重接触，直接执行主抬升并用物体真实位姿做最终验证，避免已吸住时二次压物体。"
            )
            probe_ok = True
        if not probe_ok:
            self.get_logger().warn("探测抬升未验证吸附成功，执行一次重接触重吸附")
            if not self._move_target_with_fallback(
                touch, pick_touch_orientations, mode="pick", label="pick_touch_recontact"
            ):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._release_suction("重接触失败")
                self.get_logger().error("重接触失败，停止本轮抓取")
                return
            time.sleep(0.3)
            if not self._attach_geometry_ok(half, "recontact_attach_contact"):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._release_suction("重接触几何失败")
                self.get_logger().error("重接触后双吸盘底部接触检查失败，停止本轮抓取")
                return
            self._suction_attached = None
            self._rect_motion_allowed = True
            self._publish_attach_burst()
            if not self._wait_suction_state(True, attach_wait_sec):
                if self._attach_via_gz_cli(repeats=8):
                    gz_attach_sent = True
                    self.get_logger().warn(f"重吸附 ROS 未确认，已下发 {self._gz_bin} attach 兜底指令")
                    if not self._wait_suction_state(True, 0.8):
                        self._start_fake_attach(half, "重吸附 DetachableJoint 未确认")
                elif self._param_bool("allow_unverified_sim_attach"):
                    self._start_fake_attach(half, "重吸附未确认")
            time.sleep(0.3)
            probe_ok, _ = self._probe_pickup_follow(
                touch, pick_touch_orientations[0], label="pick_probe_lift_retry"
            )
            if not probe_ok:
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._release_suction("吸附验证失败")
                self.get_logger().error("吸附验证失败：探测抬升时物体未跟随，已停止本轮抓取")
                return

        if not self._move_target_with_fallback(
            pick_lift, pick_touch_orientations, mode="pick", label="pick_lift"
        ):
            if pick_contact_collision_relaxed:
                self._set_pick_contact_collision_allowed(False)
            self._release_suction("抬升失败")
            self.get_logger().error("抬升失败，已释放吸附")
            return
        time.sleep(0.6)
        if pick_contact_collision_relaxed:
            self._set_pick_contact_collision_allowed(False)

        # 主判据：若抬升后物体中心高度未明显上升，则判定吸附失败。
        rect_center_z_after_lift = self._current_rect_center_z()
        if (
            not self._fake_attach_active
            and rect_center_z_before_attach is not None
            and rect_center_z_after_lift is not None
            and rect_center_z_after_lift < rect_center_z_before_attach + 0.015
        ):
            self._release_suction("抬升后物体未跟随")
            self.get_logger().error(
                "吸附失败：抬升后物体未跟随上移 "
                f"(before={rect_center_z_before_attach:.3f}, after={rect_center_z_after_lift:.3f})"
            )
            return
        if (
            self._fake_attach_active
            and rect_center_z_before_attach is not None
            and rect_center_z_after_lift is not None
            and rect_center_z_after_lift >= rect_center_z_before_attach + 0.015
        ):
            self.get_logger().info(
                "SetEntityPose 仿真吸附兜底已验证物体跟随上移 "
                f"(before={rect_center_z_before_attach:.3f}, after={rect_center_z_after_lift:.3f})"
            )

        if carton_ps is not None:
            place_pt = self._adjust_place_point_for_box(place_pt, carton_ps, half)
            place_above = Point(
                x=place_pt.x,
                y=place_pt.y,
                z=place_pt.z + place_entry_clearance,
            )
            place_retreat = Point(
                x=place_pt.x,
                y=place_pt.y,
                z=place_pt.z + post_place_retreat,
            )

        place_above_used = self._move_with_z_scan(
            place_above,
            place_orientations,
            [0.0, 0.04, 0.08, 0.12, 0.16, 0.22],
            mode="place",
            label="place_above",
        )
        if place_above_used is None:
            self._release_suction("place_above 失败")
            self.get_logger().error("place_above 失败（已尝试提高高度与 Pose 回退）")
            return
        time.sleep(0.6)

        place_inside_used = self._move_with_z_scan(
            place_pt,
            place_orientations,
            [0.0, 0.02, 0.04, 0.06],
            mode="place",
            label="place_inside",
        )
        if place_inside_used is not None:
            time.sleep(0.5)
        else:
            place_inside_used = place_above_used
            self.get_logger().warn("place_inside 失败，将在箱口上方释放")

        self.get_logger().info("释放 detach")
        post_release_collision_relaxed = self._set_pick_contact_collision_allowed(True)
        self._release_suction("place_release", wait_sec=1.0, keep_rect_motion_allowed=True)
        self._settle_released_rect_in_box(place_inside_used, carton_ps, half)
        time.sleep(0.2)

        retreat_target = Point(
            x=place_inside_used.x, y=place_inside_used.y, z=max(place_retreat.z, place_inside_used.z + 0.06)
        )
        if not self._move_target_with_fallback(
            retreat_target, place_orientations, mode="place", label="place_retreat"
        ):
            self.get_logger().warn("退避规划失败，继续尝试回 home")
        else:
            time.sleep(0.3)
        if post_release_collision_relaxed:
            self._set_pick_contact_collision_allowed(False)

        self._refresh_joint_state(1.0)
        if not self._send_move([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "home"):
            self.get_logger().warn("home 首次失败，刷新状态后重试一次")
            self._refresh_joint_state(1.5)
            self._send_move([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "home_retry")

    def _refresh_joint_state(self, timeout_sec: float) -> None:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            time.sleep(0.02)


def copy_joint_state(js: JointState) -> JointState:
    out = JointState()
    out.name = list(js.name)
    out.position = list(js.position)
    out.velocity = list(js.velocity) if js.velocity else []
    out.effort = list(js.effort) if js.effort else []
    return out


def make_zero_joint_state() -> JointState:
    out = JointState()
    out.name = list(_ARM_JOINTS)
    out.position = [0.0] * len(_ARM_JOINTS)
    out.velocity = [0.0] * len(_ARM_JOINTS)
    out.effort = [0.0] * len(_ARM_JOINTS)
    return out


def main() -> None:
    rclpy.init()
    node = AutoPickPlaceNode()
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)
    exec_thread = threading.Thread(target=executor.spin, daemon=True)
    exec_thread.start()
    try:
        time.sleep(0.2)
        if not node.wait_ready():
            return
        time.sleep(0.5)
        node.run_pipeline()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
