"""全自动：抓取并入箱（吸盘 attach/detach + MoveIt 避障）。"""
from __future__ import annotations

import json
import math
import re
import os
import shutil
import sys
import subprocess
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto
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
from std_msgs.msg import Bool, Empty, Float64, String
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .gazebo_pose_sync import extract_model_pose

_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

_AGENT_SESSION_ID = "9009e8"
_debug_ndjson_headers_sent: set[str] = set()
_debug_ndjson_ros_pub: object | None = None
_DEBUG_SESSION_ID_ACTIVE = "3e253c"
_DEBUG_LOG_PATH_ACTIVE = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-3e253c.log")
_DEBUG_INGEST_URL_ACTIVE = "http://127.0.0.1:7766/ingest/26484488-645d-437d-a921-8a4f664599f7"
# 与 bringup 中 debug-cfd510.log 相同约定：保证 Cursor 打开的本仓库路径始终有一份 NDJSON（即使 CS612_PROJECT_ROOT 指向其它副本）。
_IDE_CURSOR_MIRROR_LOG = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-9009e8.log")
_debug_ndjson_fs_warned: bool = False


def _register_debug_ndjson_publisher(pub: object | None) -> None:
    global _debug_ndjson_ros_pub
    _debug_ndjson_ros_pub = pub


def _agent_debug_log_active(
    location: str, message: str, hypothesis_id: str, data: dict, run_id: str = "pre-fix"
) -> None:
    # #region agent log
    payload = {
        "sessionId": _DEBUG_SESSION_ID_ACTIVE,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=True)
    try:
        _DEBUG_LOG_PATH_ACTIVE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG_PATH_ACTIVE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    try:
        import urllib.request

        req = urllib.request.Request(
            _DEBUG_INGEST_URL_ACTIVE,
            data=line.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": _DEBUG_SESSION_ID_ACTIVE,
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=0.35)
    except Exception:
        pass
    # #endregion


def _env_cs612_project_root_cursor_log() -> Path | None:
    """Launch 会设置 CS612_PROJECT_ROOT；不依赖 worlds 标记文件，避免日志落到 IDE 不可见路径。"""
    er = (os.environ.get("CS612_PROJECT_ROOT") or "").strip()
    if not er:
        return None
    try:
        root = Path(er).resolve()
        if root.is_dir():
            return root / ".cursor" / "debug-9009e8.log"
    except Exception:
        pass
    return None


def _agent_debug_log_path() -> Path:
    """
    Session NDJSON 必须落在工程根目录的 .cursor/ 下（与 IDE 约定一致）。
    根目录由 _workspace_root_from_marker() 解析（含从 cwd 回退）。
    """
    env_first = _env_cs612_project_root_cursor_log()
    if env_first is not None:
        return env_first
    wr = _workspace_root_from_marker()
    if wr is not None:
        return wr / ".cursor" / "debug-9009e8.log"
    return Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-9009e8.log")


def _workspace_root_from_marker() -> Path | None:
    """优先从模块路径向上找 worlds/my_world.sdf；失败则从 cwd 向上找（install/site-packages 场景）。"""
    envp = os.environ.get("CS612_PROJECT_ROOT", "").strip()
    if envp:
        p = Path(envp).resolve()
        if (p / "worlds" / "my_world.sdf").is_file():
            return p
    here = Path(__file__).resolve()
    for anc in [here] + list(here.parents):
        if (anc / "worlds" / "my_world.sdf").is_file():
            return anc
    cwd = Path.cwd().resolve()
    for anc in [cwd] + list(cwd.parents):
        if (anc / "worlds" / "my_world.sdf").is_file():
            return anc
    return None


def _agent_debug_log_paths() -> list[Path]:
    """主路径（工程 .cursor）+ 用户主目录镜像 + 工程根目录明文文件名；可选 CS612_DEBUG_NDJSON_PATH 强制追加路径。"""
    primary = _agent_debug_log_path()
    home_mirror = Path.home() / ".cursor" / "debug-9009e8.log"
    root_plain: Path | None = None
    wr = _workspace_root_from_marker()
    if wr is not None:
        root_plain = wr / "debug_cursor_session_9009e8.ndjson"
    out: list[Path] = []
    seen: set[str] = set()
    # 镜像路径优先：避免 ROS 工程根与 Cursor 工作区不是同一路径时 IDE 读不到日志
    for p in (_IDE_CURSOR_MIRROR_LOG, primary, home_mirror, root_plain):
        if p is None:
            continue
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    env_raw = os.environ.get("CS612_DEBUG_NDJSON_PATH", "").strip()
    if env_raw:
        try:
            ep = Path(env_raw).expanduser()
            key = str(ep.resolve())
            if key not in seen:
                seen.add(key)
                out.append(ep)
        except Exception:
            pass
    # 与 auto_pick_place.py 同目录（通常为 build 或源码树）
    try:
        pack_local = Path(__file__).resolve().parent / "debug_runtime_9009e8.ndjson"
        key = str(pack_local.resolve())
        if key not in seen:
            seen.add(key)
            out.append(pack_local)
    except Exception:
        pass
    # 固定镜像：工程根下源码包路径（不依赖 __file__ 是否在 build），确保 Cursor 工作区内必有路径
    if wr is not None:
        src_mirror = wr / "cs612_moveit_config" / "cs612_moveit_config" / "debug_runtime_9009e8.ndjson"
        try:
            key = str(src_mirror.resolve())
            if key not in seen:
                seen.add(key)
                out.append(src_mirror)
        except Exception:
            pass
    return out


def _agent_debug_ingest(line: str) -> None:
    """转发 NDJSON 到 Cursor Debug Mode 采集端点，由 IDE 侧写入工作区 .cursor/debug-9009e8.log。"""
    try:
        import urllib.request

        url = "http://127.0.0.1:7810/ingest/9a2352f3-a102-4091-892d-04936a6f8bc9"
        req = urllib.request.Request(
            url,
            data=line.encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-Debug-Session-Id": _AGENT_SESSION_ID,
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=0.35)
    except Exception:
        pass


def _agent_debug_log(
    location: str, message: str, hypothesis_id: str, data: dict, run_id: str = "pre-fix"
) -> None:
    # #region agent log
    global _debug_ndjson_headers_sent, _debug_ndjson_fs_warned
    line = json.dumps(
        {
            "sessionId": _AGENT_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        },
        ensure_ascii=True,
    )
    wrote_any = False
    path_keys_this_record: set[str] = set()
    for log_path in _agent_debug_log_paths():
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            key = str(log_path.resolve())
            if key in path_keys_this_record:
                continue
            path_keys_this_record.add(key)
            with open(log_path, "a", encoding="utf-8") as f:
                if key not in _debug_ndjson_headers_sent:
                    _debug_ndjson_headers_sent.add(key)
                    f.write(
                        json.dumps(
                            {
                                "sessionId": _AGENT_SESSION_ID,
                                "runId": run_id,
                                "hypothesisId": "H0",
                                "location": "auto_pick_place.py:_agent_debug_log",
                                "message": "log_path_resolved",
                                "data": {"path": key},
                                "timestamp": int(time.time() * 1000),
                            },
                            ensure_ascii=True,
                        )
                        + "\n"
                    )
                f.write(line + "\n")
            wrote_any = True
        except Exception as ex:
            if not _debug_ndjson_fs_warned:
                _debug_ndjson_fs_warned = True
                sys.stderr.write(
                    f"[cs612_debug_9009e8] NDJSON write failed path={log_path!s} err={ex!s}\n"
                )
                sys.stderr.flush()
            continue
    if not wrote_any and not _debug_ndjson_fs_warned:
        _debug_ndjson_fs_warned = True
        sys.stderr.write("[cs612_debug_9009e8] NDJSON: no log path accepted any write\n")
        sys.stderr.flush()
    _agent_debug_ingest(line)
    if _debug_ndjson_ros_pub is not None:
        try:
            m = String()
            m.data = line
            _debug_ndjson_ros_pub.publish(m)
        except Exception:
            pass
    # #endregion


def _gz_topic_info_sample(gz_bin: str, topic: str) -> str:
    """`ign|gz topic -i -t` 输出，用于对照 ros_gz_bridge 的 gz_type_name（H7）。"""
    try:
        proc = subprocess.run(
            [gz_bin, "topic", "-i", "-t", topic],
            capture_output=True,
            text=True,
            timeout=3.0,
            check=False,
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return out[:700] if out else f"<empty rc={proc.returncode}>"
    except Exception as ex:
        return f"<info_exc {type(ex).__name__}: {ex}>"


def _sample_gz_topic_info_fast(gz_bin: str, topic: str, timeout_sec: float = 1.2) -> str:
    """`ign|gz topic -i -t` 快速版（短超时），用于初始化诊断。"""
    try:
        proc = subprocess.run(
            [gz_bin, "topic", "-i", "-t", topic],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        return out[:700] if out else f"<empty rc={proc.returncode}>"
    except subprocess.TimeoutExpired:
        return f"<timeout rc=-1>"
    except Exception as ex:
        return f"<info_exc {type(ex).__name__}: {ex}>"


def _sample_gz_suction_state_raw(gz_bin: str, timeout_sec: float = 0.25) -> str:
    """直接从 Gazebo Transport 采样一条 /cs612/suction/state（不经 ROS bridge），用于对照 H1/H5。"""
    last_rc = -1
    last_err = ""
    last_exc = ""
    for extra in ([],):
        try:
            proc = subprocess.run(
                [gz_bin, "topic", "-t", "/cs612/suction/state", "-e", "-n", "1", *extra],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            last_rc = proc.returncode
            last_err = (proc.stderr or "").strip()
            out = (proc.stdout or "").strip()
            if out:
                return out[:800]
        except Exception as ex:
            last_exc = f"<sample_exc {type(ex).__name__}: {ex}>"
    if last_exc:
        return last_exc
    return f"<empty_stdout rc={last_rc}> {last_err[:200]}"


def _parse_gz_boolean_data(text: str) -> bool | None:
    """解析 ign/gz topic echo 的 Boolean（protobuf text / JSON / 常见变体）。无法解析则 None。"""
    if not text or text.startswith("<empty_stdout") or text.startswith("<sample_exc"):
        return None
    s = text.strip()
    if s.startswith("{"):
        try:
            j = json.loads(s)
            if isinstance(j, dict) and "data" in j:
                v = j["data"]
                if isinstance(v, bool):
                    return v
                if v in (0, 1):
                    return bool(v)
        except Exception:
            pass
    tl = text.lower()
    if "data: true" in tl or re.search(r"\bdata:\s*true\b", tl):
        return True
    if "data: false" in tl or re.search(r"\bdata:\s*false\b", tl):
        return False
    # protobuf textformat 有时输出 data: 1 / data: 0
    if re.search(r"(?m)^\s*data:\s*1\s*$", text):
        return True
    if re.search(r"(?m)^\s*data:\s*0\s*$", text):
        return False
    if '"data": true' in tl or "'data': true" in tl:
        return True
    if '"data": false' in tl:
        return False
    # 少数 protobuf 变体
    if re.search(r"\bvalue:\s*true\b", tl):
        return True
    if re.search(r"\bvalue:\s*false\b", tl):
        return False
    return None


@dataclass
class _JointKinematic:
    name: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis_xyz: tuple[float, float, float]
    lower: float
    upper: float


class StageName(Enum):
    PRE_PICK = auto()
    APPROACH = auto()
    TOUCH = auto()
    LIFT = auto()
    TRANSFER = auto()
    PLACE = auto()
    RETREAT = auto()
    HOME = auto()


def _load_scene_fallback_xyz() -> tuple[list[float], list[float], list[float]]:
    rect_xyz = [-0.82, 0.30, 0.046]
    carton_xyz = [-0.82, 0.30, 0.0]
    place_xyz = [0.82, -0.30, 0.0]
    try:
        from ament_index_python.packages import get_package_share_directory

        cfg = Path(get_package_share_directory("cs612_moveit_config")) / "config" / "scene_objects.yaml"
        if cfg.is_file():
            doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
            rect = doc.get("rect_pickup") or {}
            carton = doc.get("carton_box") or {}
            place = doc.get("place_target") or {}
            rect_center = rect.get("center_xyz")
            carton_center = carton.get("model_pose_xyz")
            place_target = place.get("tcp_xyz") or place.get("xyz") or carton.get("target_xyz")
            if isinstance(rect_center, list) and len(rect_center) == 3:
                rect_xyz = [float(rect_center[0]), float(rect_center[1]), float(rect_center[2])]
            if isinstance(carton_center, list) and len(carton_center) == 3:
                carton_xyz = [float(carton_center[0]), float(carton_center[1]), float(carton_center[2])]
            if isinstance(place_target, list) and len(place_target) == 3:
                place_xyz = [float(place_target[0]), float(place_target[1]), float(place_target[2])]
    except Exception:
        pass
    return rect_xyz, carton_xyz, place_xyz


def _spin_future(node: Node, fut, timeout_sec: float, label: str, log_timeout: bool = True) -> bool:
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
    if log_timeout:
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
    当前吸盘接触面沿 suction_cup_link 局部 +Z。
    抓取/放置时应先绕世界 Z 设定朝向，再将本地 +Z 翻到世界 -Z。
    """
    return _quat_mul(_quat_from_rpy(0.0, 0.0, yaw), _quat_from_rpy(math.pi, 0.0, 0.0))


def _flat_yaw_quat_from_tool_orientation(tool_q: Quaternion) -> Quaternion:
    """把吸盘 TCP 的平面 yaw 转成物体放平在世界 Z 轴上的姿态。"""
    x_axis = _quat_rotate_vec(tool_q, 1.0, 0.0, 0.0)
    yaw = math.atan2(float(x_axis[1]), float(x_axis[0]))
    return _quat_from_rpy(0.0, 0.0, yaw)


def _quat_to_yaw(q: Quaternion) -> float:
    """从四元数提取绕世界 Z 轴的 yaw（ZYX 欧拉角顺序）。"""
    x, y, z, w = q.x, q.y, q.z, q.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
        # #region agent log
        sys.stderr.write("[cs612_debug_9009e8] AutoPickPlaceNode __init__ ok\n")
        sys.stderr.flush()
        # #endregion
        rect_fb_xyz, carton_fb_xyz, place_target_xyz = _load_scene_fallback_xyz()
        self._cb = ReentrantCallbackGroup()
        self._log_lock = threading.Lock()
        self._rect: PoseStamped | None = None
        self._carton: PoseStamped | None = None
        self._joint_state: JointState | None = None
        self.declare_parameter("box_half_size_xyz", [0.10, 0.07, 0.04])
        # 已知场景下，优先使用「已知矩形中心 + 尺寸」计算顶面中心，降低桥接抖动对抓取点的影响。
        self.declare_parameter("use_known_rect_surface_center", True)
        self.declare_parameter("known_rect_center_xyz", rect_fb_xyz)
        self.declare_parameter("known_rect_size_xyz", [0.20, 0.14, 0.08])
        self.declare_parameter("approach_clearance", 0.06)
        self.declare_parameter("pre_touch_hover_extra_z", 0.04)
        self.declare_parameter("xy_refine_safe_clearance", 0.03)
        # 过大的下压量会在 attach 前把物体“顶走”；这里把默认下压量收小，
        # 使“物体顶面中心 ≈ 吸盘底面中心”时不会因过深接触产生横向滑移。
        self.declare_parameter("touch_delta_z", 0.003)
        # 新末端吸附口中心约位于 suction_cup_link 局部 +Z 0.214m 处
        self.declare_parameter("suction_contact_offset_z", 0.214)
        self.declare_parameter("pre_pick_safe_clearance", 0.30)
        self.declare_parameter("carton_floor_top_z", 0.006)
        self.declare_parameter("place_height_above_floor", 0.18)
        self.declare_parameter("carton_outer_size_xyz", [0.42, 0.30, 0.22])
        self.declare_parameter("carton_wall_thickness", 0.008)
        self.declare_parameter("carton_floor_thickness", 0.006)
        self.declare_parameter("post_pick_lift", 0.10)
        self.declare_parameter("place_entry_clearance", 0.10)
        self.declare_parameter("post_place_retreat", 0.10)
        # 放置阶段仍建议做碰撞感知 IK；抓取预定位常被「纸箱碰撞体」判死 → IK 无解、机械臂不动
        self.declare_parameter("ik_avoid_collisions", True)
        self.declare_parameter(
            "ik_pick_avoid_collisions",
            False,
        )
        self.declare_parameter("ik_timeout_sec", 5.0)
        # compute_ik RPC 完成等待时长（与 IK 内部 timeout 区分），避免单次调用长期阻塞看起来“完全不动”
        self.declare_parameter("ik_call_wait_sec", 4.0)
        self.declare_parameter("ik_search_wall_time_sec", 30.0)
        self.declare_parameter("pose_goal_fallback", True)
        self.declare_parameter("pose_position_tolerance", 0.01)
        # 吸盘朝下姿态需要极严格容差：0.05 rad ≈ 2.9°。
        self.declare_parameter("pose_orientation_tolerance", 0.05)
        self.declare_parameter("move_velocity_scale", 0.35)
        self.declare_parameter("move_acceleration_scale", 0.35)
        self.declare_parameter("far_move_velocity_scale", 0.45)
        self.declare_parameter("far_move_acceleration_scale", 0.45)
        self.declare_parameter("near_move_velocity_scale", 0.40)
        self.declare_parameter("near_move_acceleration_scale", 0.40)
        self.declare_parameter("rect_fallback_pose_xyz", rect_fb_xyz)
        self.declare_parameter("rect_fallback_wait_sec", 8.0)
        self.declare_parameter("carton_fallback_pose_xyz", carton_fb_xyz)
        self.declare_parameter("carton_fallback_wait_sec", 8.0)
        # 抓取要求尽量靠近顶面中心，避免“可吸附但偏心”导致后续搬运姿态不稳。
        self.declare_parameter("suction_attach_lateral_tol", 0.025)
        self.declare_parameter("suction_touch_lateral_tol", 0.025)
        self.declare_parameter("suction_attach_vertical_tol", 0.30)
        self.declare_parameter("suction_attach_axis_down_min", 0.90)

        self.declare_parameter("suction_touch_vertical_tol", 0.30)
        self.declare_parameter("suction_touch_axis_down_min", 0.90)
        self.declare_parameter("suction_attach_burst_count", 8)
        self.declare_parameter("suction_attach_burst_interval_sec", 0.04)
        self.declare_parameter("suction_attach_wait_sec", 1.0)
        self.declare_parameter("pickup_probe_lift_z", 0.025)
        self.declare_parameter("pickup_probe_min_follow_z", 0.018)
        self.declare_parameter("pickup_probe_require_follow_if_live_pose", True)
        self.declare_parameter("allow_unverified_sim_attach", True)
        self.declare_parameter("assume_attach_on_valid_contact", True)
        self.declare_parameter("fake_attach_set_pose_fallback", True)
        self.declare_parameter("fake_attach_service", "/world/arm_world/set_pose")
        self.declare_parameter("fake_attach_update_hz", 30.0)
        self.declare_parameter("ik_service", "/compute_ik")
        # 官方 Elite 栈使用 ros2_control + MoveIt 执行，默认通过 MoveIt IK/轨迹控制器下发。
        self.declare_parameter("use_compute_ik", True)
        self.declare_parameter("use_joint_template_demo", False)
        self.declare_parameter("wait_poses_sec", 45.0)
        self.declare_parameter("require_joint_states", True)
        self.declare_parameter("pre_pick_try_clearances", [0.10, 0.14, 0.18, 0.24])
        self.declare_parameter("pick_posture_hint", [0.0, 0.50, 1.10, -1.60, 1.57, 0.0])
        self.declare_parameter("place_posture_hint", [0.0, -0.90, 1.10, -1.55, 1.57, 0.0])
        self.declare_parameter("post_place_stow_joints", [0.0, -1.57, 0.0, -1.57, 1.57, 0.0])
        self.declare_parameter("move_to_start_face_pose", False)
        self.declare_parameter("start_face_posture_hint", [0.0, -1.57, 0.0, -1.57, 1.57, 0.0])
        # 默认按“物体+箱子中点方向”自动朝向；若关闭则使用 start_face_joint1_rad。
        self.declare_parameter("start_face_use_scene_midpoint_yaw", True)
        self.declare_parameter("start_face_joint1_rad", 0.0)
        self.declare_parameter("pick_yaw_follow_start_face", True)
        # 官方 shoulder_pan_joint 零位按官方 kinematics.yaml 定义；该偏置仅用于预抓取种子。
        self.declare_parameter("joint1_world_yaw_offset_rad", -1.5708)
        # 桥接偶发抖动时，实时 pose 可能跳变；默认关闭实时覆盖，优先使用已知几何中心。
        self.declare_parameter("refresh_top_from_live_pose", False)
        self.declare_parameter("refresh_top_max_delta_xy", 0.20)
        self.declare_parameter("refresh_top_max_delta_z", 0.08)
        # 可选：直接给抓取点/放置点坐标（base_link），用于“按坐标抓放”。
        # 注意：参数默认类型必须是 DOUBLE_ARRAY，不能用空列表 []。
        self.declare_parameter("use_direct_xyz", False)
        self.declare_parameter("pick_point_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("place_point_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("use_configured_place_target", True)
        self.declare_parameter("configured_place_target_xyz", place_target_xyz)
        self.declare_parameter("conveyor_place_use_start_inset", True)
        self.declare_parameter("conveyor_place_inset_margin_m", 0.04)
        self.declare_parameter("conveyor_place_lateral_offset_m", 0.0)
        self.declare_parameter("conveyor_place_align_yaw", True)
        self.declare_parameter("place_dense_descent_enabled", True)
        self.declare_parameter("conveyor_place_cartesian_approach_enabled", True)
        self.declare_parameter("place_dense_waypoint_step_m", 0.040)
        self.declare_parameter("place_dense_orientation_weight", 5.0)
        self.declare_parameter("place_dense_settle_sec", 0.0)
        self.declare_parameter("conveyor_transport_enabled", True)
        self.declare_parameter("conveyor_transport_speed_mps", 0.18)
        self.declare_parameter("conveyor_transport_step_m", 0.050)
        self.declare_parameter("conveyor_transport_end_margin_m", 0.01)
        self.declare_parameter("conveyor_transport_settle_sec", 0.40)
        self.declare_parameter("conveyor_transport_goal_tol_m", 0.05)
        self.declare_parameter("conveyor_transport_lateral_tol_m", 0.12)
        self.declare_parameter("conveyor_transport_direction_sign", 1.0)
        self.declare_parameter("conveyor_transport_timeout_pad_sec", 30.0)
        self.declare_parameter("conveyor_transport_monitor_period_sec", 0.10)
        self.declare_parameter("conveyor_track_command_topic", "/middle_conveyor/track_cmd_vel")
        self.declare_parameter("conveyor_roller_command_topic", "/middle_conveyor/roller_cmd_vel")
        self.declare_parameter("conveyor_roller_radius_m", 0.03)
        self.declare_parameter("cartesian_step_max_m", 0.008)
        self.declare_parameter("cartesian_step_max_rad", 0.12)
        self.declare_parameter("cartesian_cmd_period_sec", 0.050)
        self.declare_parameter("cartesian_point_time_from_start_sec", 0.20)
        self.declare_parameter("cartesian_settle_timeout_sec", 12.0)
        self.declare_parameter("cartesian_settle_tol_rad", 0.080)
        self.declare_parameter("cartesian_ik_max_iters", 140)
        self.declare_parameter("cartesian_ik_pos_tol_m", 0.004)
        self.declare_parameter("cartesian_ik_ori_tol_rad", 0.20)
        self.declare_parameter("cartesian_ik_damping", 0.08)
        self.declare_parameter("cartesian_ik_step_gain", 0.8)
        self.declare_parameter("cartesian_ik_joint_step_limit_rad", 0.12)
        self.declare_parameter("cartesian_ik_orientation_weight", 1.50)
        self.declare_parameter("touch_cartesian_keep_xy", True)
        self.declare_parameter("touch_cartesian_step_max_m", 0.0025)
        self.declare_parameter("touch_cartesian_joint_step_limit_rad", 0.025)
        self.declare_parameter("touch_cartesian_orientation_weight", 4.0)
        self.declare_parameter("touch_cartesian_pose_fallback", True)
        # 强制抓取下压阶段走笛卡尔直线（优先于 use_compute_ik）。
        self.declare_parameter("force_cartesian_touch_descent", True)
        self.declare_parameter("place_compensate_pick_offset", True)
        self.declare_parameter("place_compensation_gain", 1.0)
        self.declare_parameter("place_inner_margin_xy", 0.015)
        self.declare_parameter("cartesian_bridge_wait_sec", 25.0)
        # 混合模式：先 MoveIt 到 pre-grasp/approach，再用笛卡尔直线下压抓取。
        self.declare_parameter("hybrid_moveit_pregrasp", True)
        self.declare_parameter("hybrid_cartesian_touch_only", True)
        # 仅按目标物体中心坐标执行“上方对齐 + 直线下压”，不做反馈式 XY 中心补偿
        self.declare_parameter("centerline_use_object_center_only", True)
        self.declare_parameter("pregrasp_xy_align_tol", 0.015)
        self.declare_parameter("pregrasp_alignment_gate_enabled", True)
        self.declare_parameter("pregrasp_xy_comp_max_step_m", 0.03)
        self.declare_parameter("pregrasp_xy_comp_gain", 1.0)
        self.declare_parameter("pregrasp_xy_comp_retries", 3)
        self.declare_parameter("pregrasp_cartesian_center_enabled", True)
        # 下压前强制校正吸盘朝向：若吸盘+Z轴与世界-Z的cos对齐度低于此阈值则执行校正。
        self.declare_parameter("orientation_min_cos_before_touch", 0.95)
        # 校正朝向时的朝向权重，需远高于普通运动以强制优先保持朝下。
        self.declare_parameter("orientation_correction_weight", 8.0)
        # 校正朝向时的最大重试次数。
        self.declare_parameter("orientation_correction_retries", 6)
        # 下压前在目标上方悬停验证 XY 对齐和朝向的等待时间（秒），
        # 给 Gazebo 关节控制器足够收敛时间，避免首次下压因位置未收敛而失败。
        self.declare_parameter("pre_touch_settle_sec", 2.0)
        # 下压完成后、检查吸附前的稳定等待时间（秒），
        # 让 Gazebo 物理仿真充分计算接触响应后再判定吸附结果。
        self.declare_parameter("post_touch_settle_sec", 1.2)
        # 发送 attach 指令前额外等待秒数，确保吸盘与物体接触已稳定。
        self.declare_parameter("pre_attach_settle_sec", 0.6)
        self.declare_parameter("dense_waypoint_descent_enabled", True)
        self.declare_parameter("dense_waypoint_step_m", 0.012)
        self.declare_parameter("dense_waypoint_xy_correction_gain", 0.65)
        self.declare_parameter("dense_waypoint_xy_correction_max_m", 0.015)
        self.declare_parameter("dense_waypoint_orientation_weight", 3.0)
        self.declare_parameter("dense_waypoint_settle_sec", 0.01)
        self.declare_parameter("dense_waypoint_point_time_sec", 0.06)
        self.declare_parameter("dense_waypoint_max_xy_drift_m", 0.08)
        self.declare_parameter("touch_object_push_abort_m", 0.025)
        self.declare_parameter("staged_pregrasp_enabled", True)
        self.declare_parameter("staged_pregrasp_clearances", [0.24, 0.20, 0.16, 0.12, 0.09])
        self.declare_parameter("staged_pregrasp_settle_sec", 0.5)
        # 抓取关键阶段的关节姿态护栏：拒绝“可达但不可抓取”的反肘/绕腕解。
        self.declare_parameter("pick_pose_guard_enabled", False)
        self.declare_parameter("pick_pose_guard_joint2_min", 0.10)
        self.declare_parameter("pick_pose_guard_joint2_max", 1.80)
        self.declare_parameter("pick_pose_guard_joint3_min", -0.08)
        self.declare_parameter("pick_pose_guard_joint3_max", 2.10)
        self.declare_parameter("pick_pose_guard_joint5_min", 0.90)
        self.declare_parameter("pick_pose_guard_joint5_max", 2.30)
        self.declare_parameter("pick_pose_guard_joint4_abs_max", 2.95)
        self.declare_parameter("pick_pose_guard_joint6_abs_max", 2.95)
        self.declare_parameter("place_pose_guard_enabled", True)
        self.declare_parameter("place_pose_guard_joint2_min", -2.95)
        self.declare_parameter("place_pose_guard_joint2_max", 0.30)
        self.declare_parameter("place_pose_guard_joint3_min", -0.30)
        self.declare_parameter("place_pose_guard_joint3_max", 2.80)
        self.declare_parameter("place_pose_guard_joint5_min", 0.60)
        self.declare_parameter("place_pose_guard_joint5_max", 2.60)
        self.declare_parameter("place_pose_guard_joint4_abs_max", 3.00)
        self.declare_parameter("place_pose_guard_joint6_abs_max", 3.00)
        self.declare_parameter("place_verify_lateral_tol_m", 0.06)
        self.declare_parameter("approach_verify_lateral_tol_m", 0.045)
        self.declare_parameter("approach_verify_max_correction_step_m", 0.06)
        self.declare_parameter("approach_verify_reject_large_error_m", 0.30)
        self.declare_parameter("approach_verify_relaxed_lateral_tol_m", 0.045)
        self.declare_parameter("adaptive_touch_target_enabled", True)
        self.declare_parameter("adaptive_touch_max_adjust_m", 0.03)
        self.declare_parameter("arm_collision_check_enabled", True)
        self.declare_parameter("arm_collision_link_radii", [0.04, 0.06, 0.05, 0.035, 0.035, 0.035])
        self.declare_parameter("arm_collision_margin", 0.005)

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
            TFMessage,
            "/world/arm_world/pose/info",
            self._on_world_pose_info,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )
        self.create_subscription(
            TFMessage,
            "/world/arm_world/dynamic_pose/info",
            self._on_world_pose_info,
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
        self._pub_assumed_state = self.create_publisher(Bool, "/cs612/suction/assumed_state", 10)
        self._pub_conveyor_track = self.create_publisher(
            Float64,
            str(self.get_parameter("conveyor_track_command_topic").value),
            10,
        )
        self._pub_conveyor_roller = self.create_publisher(
            Float64,
            str(self.get_parameter("conveyor_roller_command_topic").value),
            10,
        )
        if os.environ.get("CS612_DEBUG_PUBLISH_ROS", "").strip():
            _register_debug_ndjson_publisher(
                self.create_publisher(String, "/cs612/debug_ndjson_line", 10)
            )
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
        self._rect_pose_can_move = False
        self._fake_attach_active = False
        self._fake_attach_stop = threading.Event()
        self._fake_attach_thread: threading.Thread | None = None
        self._motion_profile = "default"
        self._joint_kin: list[_JointKinematic] = []
        self._tool_origin_xyz = (0.0, 0.0, 0.0)
        self._tool_origin_rpy = (0.0, 0.0, 0.0)
        self._kin_ready = self._load_kinematics_model()
        self._gz_bin = _find_gz_executable()
        self._gz_msg_pfx = _gz_msg_prefix(self._gz_bin)
        # #region agent log
        _suct_info = _sample_gz_topic_info_fast(self._gz_bin, "/cs612/suction/state")
        _agent_debug_log(
            "auto_pick_place.py:__init__",
            "gz_topic_info_suction_state",
            "H7",
            {
                "info": _suct_info,
                "expected_bridge_yaml": "gz.msgs.Boolean -> std_msgs/msg/Bool",
            },
        )
        # #endregion
        _topics_unknown = _suct_info.startswith("<timeout") or _suct_info.startswith("<info_exc")
        _topics_present = (
            not _topics_unknown
            and not _suct_info.startswith("<empty")
            and "No publishers" not in _suct_info
            and "not found" not in _suct_info.lower()
        )
        if _topics_unknown:
            self.get_logger().warn(
                "DetachableJoint state 话题快速探测超时；该话题是事件型输出，"
                "不再据此判定插件缺失，将在 attach 后用几何接触 + 抬升跟随验证。"
            )
        elif not _topics_present:
            self.get_logger().warn(
                "\n"
                "⚠️  DetachableJoint 话题 /cs612/suction/state 在 Gazebo Transport 中不存在！\n"
                "    ignition-gazebo-detachable-joint-system 插件可能未加载到机器人模型中。\n"
                "    将通过 Gazebo SetEntityPose CLI 仿真吸附兜底维持物体跟随。\n"
                "    检查: ign topic -i -t /cs612/suction/state"
            )
        else:
            self.get_logger().info(
                "DetachableJoint 话题 /cs612/suction/state 已在 Gazebo Transport 中注册"
            )
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
        # #region agent log
        _agent_debug_log(
            "auto_pick_place.py:__init__",
            "node_ready_gz_bridge",
            "H2",
            {
                "gz_bin": self._gz_bin,
                "gz_msg_pfx": self._gz_msg_pfx,
                "suction_state_sub_qos": "qos_profile_sensor_data",
            },
        )
        try:
            self.get_logger().info(
                "[debug] NDJSON session 9009e8 写入位置: "
                + " | ".join(str(p.resolve()) for p in _agent_debug_log_paths())
            )
        except Exception:
            pass
        if os.environ.get("CS612_DEBUG_PUBLISH_ROS", "").strip():
            self.get_logger().info(
                "[debug] NDJSON 同时发布到 std_msgs/String 话题 /cs612/debug_ndjson_line"
            )
        # #endregion

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

        urdf_candidates: list[Path] = [
            Path(__file__).resolve().parents[2] / "my_arms" / "urdf" / "CS612.urdf",
            Path.cwd() / "my_arms" / "urdf" / "CS612.urdf",
            Path(__file__).resolve().parents[2] / "my_arms" / "urdf" / "CS612urdf.urdf",
            Path.cwd() / "my_arms" / "urdf" / "CS612urdf.urdf",
        ]
        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = Path(get_package_share_directory("cs612_moveit_config")) / "my_arms" / "urdf"
            urdf_candidates.append(share_dir / "CS612.urdf")
            urdf_candidates.append(share_dir / "CS612urdf.urdf")
        except Exception:
            pass

        urdf_path = next((p for p in urdf_candidates if p.is_file()), None)
        if urdf_path is None:
            self.get_logger().error("未找到 CS612.urdf/CS612urdf.urdf，笛卡尔数值 IK 不可用")
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

        suction_joint = joints_by_name.get("joint_suction_cup")
        if suction_joint is not None:
            origin = suction_joint.find("origin")
            self._tool_origin_xyz = _parse_triplet(
                origin.attrib.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0)
            )
            self._tool_origin_rpy = _parse_triplet(
                origin.attrib.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0)
            )

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

    def _joint_accepts_equivalent_angles(self, idx: int) -> bool:
        if idx < 0 or idx >= len(self._joint_kin):
            return True
        limit = self._joint_kin[idx]
        return (limit.upper - limit.lower) >= (2.0 * math.pi - 0.1)

    def _nearest_equivalent_joint_angle(self, idx: int, value: float, reference: float) -> float:
        if self._joint_accepts_equivalent_angles(idx):
            return float(reference) + _wrap_to_pi(float(value) - float(reference))
        return float(value)

    def _joint_position_error(self, idx: int, actual: float, target: float) -> float:
        if self._joint_accepts_equivalent_angles(idx):
            return _angle_distance(float(actual), float(target))
        return abs(float(actual) - float(target))

    def _match_joint_positions_to_reference(
        self, joints: Sequence[float], reference: Sequence[float]
    ) -> list[float]:
        out: list[float] = []
        for idx, value in enumerate(joints):
            ref = float(reference[idx]) if idx < len(reference) else float(value)
            out.append(self._nearest_equivalent_joint_angle(idx, float(value), ref))
        return out

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

    def _publish_joint_vector(
        self,
        joints: Sequence[float],
        lead_sec_override: float | None = None,
    ) -> None:
        if len(joints) < len(_ARM_JOINTS):
            return
        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(_ARM_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(joints[i]) for i in range(len(_ARM_JOINTS))]
        if lead_sec_override is None:
            lead_sec = max(0.20, float(self.get_parameter("cartesian_point_time_from_start_sec").value))
        else:
            lead_sec = max(0.04, float(lead_sec_override))
        point.time_from_start = Duration(
            sec=int(lead_sec),
            nanosec=int((lead_sec - int(lead_sec)) * 1_000_000_000),
        )
        traj.points.append(point)
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
                if all(self._joint_position_error(i, cur[i], target[i]) <= tol for i in range(len(target))):
                    return True
            now = time.monotonic()
            if now >= next_pub:
                self._publish_joint_vector(target)
                next_pub = now + republish_period
            time.sleep(0.02)
        cur = self._current_arm_positions()
        if cur is not None and len(cur) == len(target):
            err_max = max(self._joint_position_error(i, cur[i], target[i]) for i in range(len(target)))
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
        command_target = self._match_joint_positions_to_reference(target, cur)
        max_delta = max(
            self._joint_position_error(i, float(cur[i]), float(command_target[i]))
            for i in range(len(command_target))
        )
        step_limit = max(0.01, float(self.get_parameter("cartesian_ik_joint_step_limit_rad").value))
        steps = max(1, int(math.ceil(max_delta / step_limit)))
        dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        self.get_logger().info(f"DirectJoint: {label} steps={steps}")
        for s in range(1, steps + 1):
            t = float(s) / float(steps)
            cmd = [
                float(cur[i]) + t * (float(command_target[i]) - float(cur[i]))
                for i in range(len(command_target))
            ]
            self._publish_joint_vector(cmd)
            time.sleep(dt)
        self._wait_joint_goal(command_target, label)
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
        collision_rect_half: Sequence[float] | None = None,
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

        q_seed = [self._joint_limit_clamp(i, float(v)) for i, v in enumerate(cur)]
        command_seed = list(cur)
        initial_top = None
        push_abort = max(0.005, float(self.get_parameter("touch_object_push_abort_m").value))
        if collision_rect_half is not None:
            initial_top = self._current_rect_top_live(list(collision_rect_half)) or self._current_rect_top(
                list(collision_rect_half)
            )
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
            cmd = self._match_joint_positions_to_reference(sol, command_seed)
            self._publish_joint_vector(cmd)
            q_seed = sol
            command_seed = cmd
            time.sleep(dt)
            if collision_rect_half is not None:
                collided, coll_desc = self._check_arm_object_collision(collision_rect_half)
                if collided:
                    self.get_logger().error(
                        f"{label}: 机械臂与物体碰撞，停止运动: {coll_desc}"
                    )
                    return False
                live_top = self._current_rect_top_live(list(collision_rect_half))
                if initial_top is not None and live_top is not None:
                    push_dxy = math.hypot(
                        float(live_top.x) - float(initial_top.x),
                        float(live_top.y) - float(initial_top.y),
                    )
                    if push_dxy > push_abort:
                        self.get_logger().error(
                            f"{label}: 物体横向漂移 {push_dxy:.4f}m > {push_abort:.4f}m，"
                            "停止下压以避免继续推走物体"
                        )
                        return False
        if not self._wait_joint_goal(command_seed, label):
            final_pose = self._current_tcp_pose()
            if final_pose is not None:
                actual_p, actual_q = final_pose
                pos_err = math.sqrt(
                    (float(actual_p.x) - float(tgt.x)) ** 2
                    + (float(actual_p.y) - float(tgt.y)) ** 2
                    + (float(actual_p.z) - float(tgt.z)) ** 2
                )
                ori_err = self._quat_angle(actual_q, orientation)
                pos_tol = max(0.008, float(self.get_parameter("cartesian_ik_pos_tol_m").value) * 4.0)
                ori_tol = max(0.20, float(self.get_parameter("cartesian_ik_ori_tol_rad").value) * 1.5)
                if pos_err <= pos_tol and ori_err <= ori_tol:
                    self.get_logger().warn(
                        f"{label}: 关节目标未完全收敛，但 TCP 已到位 "
                        f"(pos_err={pos_err:.4f}m <= {pos_tol:.4f}, ori_err={ori_err:.3f}rad <= {ori_tol:.3f})"
                    )
                    return True
            self.get_logger().error(f"{label}: 笛卡尔轨迹末端未收敛，判定执行失败")
            return False
        return True

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
        4. TCP 横向偏差超过 max_xy_drift_m 时继续闭环修正；只有物体本体被推走才中止
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

        # 下压前先在当前高度把 TCP/吸盘底面回收到目标中心线上，
        # 避免从“可达但偏心”的 hover 姿态直接下压，把物体横向推走。
        pre_center_gate = max(
            0.015,
            min(
                max_xy_drift_m,
                float(self.get_parameter("approach_verify_lateral_tol_m").value),
            ),
        )
        cup_pose0 = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose0 is not None:
            cup_bottom0 = self._point_with_local_offset(
                cup_pose0.position,
                cup_pose0.orientation,
                0.0,
                0.0,
                suction_contact_offset,
            )
            lateral0 = math.hypot(float(cup_bottom0.x) - float(top.x), float(cup_bottom0.y) - float(top.y))
            if lateral0 > pre_center_gate:
                centerline_wp = Point(x=float(target.x), y=float(target.y), z=float(start_p.z))
                self.get_logger().warn(
                    f"{label}: 下压前横向偏差 {lateral0:.4f}m > {pre_center_gate:.4f}m，"
                    "先执行同高度中心线回正"
                )
                if not self._move_cartesian_direct(
                    centerline_wp,
                    orientation,
                    mode=mode,
                    label=f"{label}_centerline_reset",
                    orientation_weight_override=max(orientation_correction_weight, 4.0),
                ):
                    self.get_logger().error(f"{label}: 中心线回正失败，终止下压")
                    return False
                time.sleep(max(0.05, settle_sec_per_waypoint))
                pose = self._current_tcp_pose()
                if pose is None:
                    self.get_logger().error(f"{label}: 中心线回正后无法读取 TCP 位姿")
                    return False
                start_p, start_q = pose

        n_steps = max(1, int(math.ceil(abs(total_dz) / max(0.001, waypoint_step_m))))
        dz_per_step = total_dz / n_steps

        self.get_logger().info(
            f"{label}: 密途径点下压 start=({float(start_p.x):.4f},{float(start_p.y):.4f},{float(start_p.z):.4f}) "
            f"target=({float(target.x):.4f},{float(target.y):.4f},{float(target.z):.4f}) "
            f"steps={n_steps} dz_per_step={dz_per_step:.4f}m total_dz={total_dz:.4f}m"
        )

        cur_target_x = float(target.x)
        cur_target_y = float(target.y)
        cur_z = float(start_p.z)
        q_seed = [self._joint_limit_clamp(i, float(v)) for i, v in enumerate(cur_joints)]
        command_seed = list(cur_joints)
        cartesian_dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        waypoint_lead_sec = max(0.04, float(self.get_parameter("dense_waypoint_point_time_sec").value))
        initial_top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
        push_abort = max(0.005, float(self.get_parameter("touch_object_push_abort_m").value))
        hard_xy_abort = max(0.10, float(max_xy_drift_m) * 2.0)

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

            cmd = self._match_joint_positions_to_reference(sol, command_seed)
            self._publish_joint_vector(cmd, lead_sec_override=waypoint_lead_sec)
            q_seed = sol
            command_seed = cmd
            time.sleep(max(cartesian_dt, waypoint_lead_sec))

            if settle_sec_per_waypoint > 0:
                time.sleep(settle_sec_per_waypoint)

            collided, coll_desc = self._check_arm_object_collision(rect_half)
            if collided:
                self.get_logger().error(
                    f"{label}[{step}/{n_steps}]: 机械臂与物体碰撞，停止运动: {coll_desc}"
                )
                return False

            if step < n_steps:
                cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
                live_top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))

                if cup_pose is not None and live_top is not None:
                    if initial_top is not None:
                        push_dxy = math.hypot(
                            float(live_top.x) - float(initial_top.x),
                            float(live_top.y) - float(initial_top.y),
                        )
                        if push_dxy > push_abort:
                            self.get_logger().error(
                                f"{label}: 物体横向漂移 {push_dxy:.4f}m > {push_abort:.4f}m，"
                                "停止下压以避免继续推走物体"
                            )
                            return False
                    cup_bottom = self._point_with_local_offset(
                        cup_pose.position,
                        cup_pose.orientation,
                        0.0, 0.0, suction_contact_offset,
                    )
                    dx_err = float(cup_bottom.x) - float(live_top.x)
                    dy_err = float(cup_bottom.y) - float(live_top.y)
                    lateral = math.hypot(dx_err, dy_err)

                    if lateral > hard_xy_abort:
                        self.get_logger().error(
                            f"{label}: TCP 横向漂移过大 lateral={lateral:.4f}m > {hard_xy_abort:.4f}m，"
                            f"终止下压防止推走物体"
                        )
                        return False
                    if lateral > max_xy_drift_m:
                        self.get_logger().warn(
                            f"{label}: TCP 横向跟踪偏差 lateral={lateral:.4f}m > {max_xy_drift_m:.4f}m，"
                            "继续闭环修正；仅在物体本体被推走时中止"
                        )

                    correction_x = -dx_err * xy_correction_gain
                    correction_y = -dy_err * xy_correction_gain
                    correction_mag = math.hypot(correction_x, correction_y)
                    if correction_mag > xy_correction_max_m:
                        scale = xy_correction_max_m / correction_mag
                        correction_x *= scale
                        correction_y *= scale

                    cur_target_x += correction_x
                    cur_target_y += correction_y

                    self.get_logger().info(
                        f"{label}[{step}]: XY 修正 "
                        f"dx_err={dx_err:.4f} dy_err={dy_err:.4f} lateral={lateral:.4f}m, "
                        f"next_target=({cur_target_x:.4f},{cur_target_y:.4f},{next_z:.4f})"
                    )

                    down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
                    down_cos = -down_axis[2]
                    if down_cos < 0.85:
                        self.get_logger().warn(
                            f"{label}[{step}]: 朝向偏斜 down_cos={down_cos:.4f} < 0.85，"
                            f"将在下一步加重朝向权重"
                        )
                else:
                    self.get_logger().debug(
                        f"{label}[{step}]: 无法读取 TF/物体位姿，跳过本步 XY 修正"
                    )

            cur_z = next_z

        if not self._wait_joint_goal(command_seed, label):
            self.get_logger().warn(f"{label}: 最终关节未完全收敛")

        final_cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
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
            self.get_logger().info(
                f"{label} 完成: final_offset=({final_dx:.4f},{final_dy:.4f}) "
                f"lateral={final_lateral:.4f}m down_cos={-down_ax[2]:.4f}"
            )

        return True

    def _move_cartesian_vertical_waypoints(
        self,
        target: Point,
        orientation: Quaternion,
        rect_half: Sequence[float],
        mode: str = "place",
        label: str = "vertical_waypoints",
        waypoint_step_m: float = 0.010,
        orientation_weight: float = 5.0,
        settle_sec_per_waypoint: float = 0.03,
    ) -> bool:
        """用当前 IK 分支沿固定 XY/姿态做垂直运动，避免触碰阶段一路横摆修正。"""
        cur_joints = self._current_arm_positions()
        if cur_joints is None:
            self.get_logger().error(f"{label}: 无法读取当前关节状态")
            return False
        pose = self._current_tcp_pose()
        if pose is None:
            self.get_logger().error(f"{label}: 无法读取当前 TCP 位姿")
            return False

        start_p, _ = pose
        total_dz = float(target.z) - float(start_p.z)
        if abs(total_dz) < 1e-5:
            self.get_logger().info(f"{label}: 当前已在目标高度，无需下压")
            return True
        n_steps = max(1, int(math.ceil(abs(total_dz) / max(0.001, waypoint_step_m))))
        dz_per_step = total_dz / n_steps
        q_seed = [self._joint_limit_clamp(i, float(v)) for i, v in enumerate(cur_joints)]
        command_seed = list(cur_joints)
        waypoint_lead_sec = max(0.04, float(self.get_parameter("dense_waypoint_point_time_sec").value))
        cartesian_dt = max(0.01, float(self.get_parameter("cartesian_cmd_period_sec").value))
        cur_z = float(start_p.z)

        self.get_logger().info(
            f"{label}: 固定XY垂直运动 start=({float(start_p.x):.4f},{float(start_p.y):.4f},{float(start_p.z):.4f}) "
            f"target=({float(target.x):.4f},{float(target.y):.4f},{float(target.z):.4f}) "
            f"steps={n_steps} dz_per_step={dz_per_step:.4f}m"
        )

        for step in range(1, n_steps + 1):
            next_z = cur_z + dz_per_step
            if dz_per_step < 0:
                next_z = max(next_z, float(target.z))
            else:
                next_z = min(next_z, float(target.z))
            wp = Point(x=float(target.x), y=float(target.y), z=next_z)
            self.get_logger().info(
                f"{label}[{step}/{n_steps}]: wp=({wp.x:.4f},{wp.y:.4f},{wp.z:.4f})"
            )
            sol = self._solve_cartesian_ik_direct(
                wp,
                orientation,
                q_seed,
                mode=mode,
                label=f"{label}_ik[{step}]",
                orientation_weight_override=orientation_weight,
            )
            if sol is None:
                # 用当前实际关节状态作为种子重试一次，避免因种子偏差连续失败
                cur_fallback = self._current_arm_positions()
                if cur_fallback is not None and len(cur_fallback) == 6:
                    q_fallback = [self._joint_limit_clamp(i, float(v)) for i, v in enumerate(cur_fallback)]
                    sol = self._solve_cartesian_ik_direct(
                        wp,
                        orientation,
                        q_fallback,
                        mode=mode,
                        label=f"{label}_ik_retry[{step}]",
                        orientation_weight_override=orientation_weight,
                    )
            if sol is None:
                self.get_logger().error(f"{label}: IK 失败 step={step}/{n_steps}")
                return False
            cmd = self._match_joint_positions_to_reference(sol, command_seed)
            self._publish_joint_vector(cmd, lead_sec_override=waypoint_lead_sec)
            q_seed = sol
            command_seed = cmd
            time.sleep(max(cartesian_dt, waypoint_lead_sec))
            if self._fake_attach_active:
                self._snap_fake_attach_pose(rect_half, f"{label}_snap[{step}]", attempts=1)
            if settle_sec_per_waypoint > 0:
                time.sleep(settle_sec_per_waypoint)
            cur_z = next_z

        if not self._wait_joint_goal(command_seed, label):
            self.get_logger().warn(f"{label}: 最终关节未完全收敛")
        final_pose = self._current_tcp_pose()
        if final_pose is not None:
            p, q = final_pose
            pos_err = math.sqrt(
                (float(p.x) - float(target.x)) ** 2
                + (float(p.y) - float(target.y)) ** 2
                + (float(p.z) - float(target.z)) ** 2
            )
            down_axis = _quat_rotate_vec(q, 0.0, 0.0, 1.0)
            self.get_logger().info(
                f"{label} 完成: pos_err={pos_err:.4f}m down_cos={-down_axis[2]:.4f}"
            )
            return pos_err <= max(0.025, float(self.get_parameter("cartesian_ik_pos_tol_m").value) * 6.0)
        return True

    def _estimate_pick_offset_xy(self, rect_half: Sequence[float]) -> tuple[float, float] | None:
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
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

    def _conveyor_start_place_target(
        self, fallback: Point, rect_half: Sequence[float]
    ) -> tuple[Point, float | None]:
        """把配置中的传送带起点边缘换算成物体中心放置点。"""
        try:
            from ament_index_python.packages import get_package_share_directory

            cfg_path = Path(get_package_share_directory("cs612_moveit_config")) / "config" / "scene_objects.yaml"
            doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            conv = doc.get("middle_conveyor") or {}
            pose = conv.get("model_pose_xyz") or [fallback.x, fallback.y, 0.0]
            rpy = conv.get("model_pose_rpy") or [0.0, 0.0, 0.0]
            size = conv.get("size_xyz") or [1.50, 0.30, 0.20]
            start = conv.get("start_xyz")
            yaw = float(rpy[2])
            dx = math.cos(yaw)
            dy = math.sin(yaw)
            if isinstance(start, list) and len(start) == 3:
                edge_x = float(start[0])
                edge_y = float(start[1])
                surface_z = float(start[2])
            else:
                half_len = 0.5 * float(size[0])
                edge_x = float(pose[0]) - dx * half_len
                edge_y = float(pose[1]) - dy * half_len
                surface_z = float(conv.get("top_surface_z", float(pose[2]) + float(size[2])))

            if not self._param_bool("conveyor_place_use_start_inset"):
                return Point(x=edge_x, y=edge_y, z=surface_z), yaw

            margin = max(0.0, float(self.get_parameter("conveyor_place_inset_margin_m").value))
            lateral = float(self.get_parameter("conveyor_place_lateral_offset_m").value)
            half_along = max(0.0, float(rect_half[0]) if len(rect_half) > 0 else 0.10)
            inset = half_along + margin
            cx = edge_x + dx * inset - dy * lateral
            cy = edge_y + dy * inset + dx * lateral
            out = Point(x=cx, y=cy, z=surface_z)
            self.get_logger().info(
                "传送带起点放置目标已按物体尺寸内移: "
                f"edge=({edge_x:.3f},{edge_y:.3f},{surface_z:.3f}), "
                f"yaw={yaw:.3f}, inset={inset:.3f}, lateral={lateral:.3f} -> "
                f"center=({out.x:.3f},{out.y:.3f},{out.z:.3f})"
            )
            return out, yaw
        except Exception as e:
            self.get_logger().warn(f"传送带放置点自动换算失败，使用配置点: {e}")
            return fallback, None

    def _conveyor_transport_end_pose(
        self, release_pose: Pose, rect_half: Sequence[float]
    ) -> Pose | None:
        """根据传送带尺寸计算物体中心在另一端的目标位姿。"""
        try:
            from ament_index_python.packages import get_package_share_directory

            cfg_path = Path(get_package_share_directory("cs612_moveit_config")) / "config" / "scene_objects.yaml"
            doc = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            conv = doc.get("middle_conveyor") or {}
            pose = conv.get("model_pose_xyz") or [1.00825, -0.35547, 0.0]
            rpy = conv.get("model_pose_rpy") or [0.0, 0.0, 0.0]
            size = conv.get("size_xyz") or [1.50, 0.30, 0.20]
            surface_z = float(conv.get("top_surface_z", float(pose[2]) + float(size[2])))
            yaw = float(rpy[2])
            half_len = 0.5 * float(size[0])
            dx = math.cos(yaw)
            dy = math.sin(yaw)
            half_along = max(0.0, float(rect_half[0]) if len(rect_half) > 0 else 0.10)
            end_margin = max(0.0, float(self.get_parameter("conveyor_transport_end_margin_m").value))
            inset = half_along + end_margin
            end_edge_x = float(pose[0]) + dx * half_len
            end_edge_y = float(pose[1]) + dy * half_len

            out = Pose()
            out.position.x = end_edge_x - dx * inset
            out.position.y = end_edge_y - dy * inset
            out.position.z = max(surface_z + max(0.001, float(rect_half[2])), float(release_pose.position.z))
            out.orientation = _quat_from_rpy(0.0, 0.0, yaw)
            return out
        except Exception as e:
            self.get_logger().warn(f"传送带终点位姿计算失败，跳过输送: {e}")
            return None

    def _publish_conveyor_velocity(self, linear_speed_mps: float) -> None:
        sign = float(self.get_parameter("conveyor_transport_direction_sign").value)
        track_msg = Float64()
        track_msg.data = float(linear_speed_mps * sign)
        roller_msg = Float64()
        # IFRA 替换后以 track 话题直接驱动皮带线速度；roller 话题保留兼容并复用同一线速度。
        roller_msg.data = float(track_msg.data)
        self._pub_conveyor_track.publish(track_msg)
        self._pub_conveyor_roller.publish(roller_msg)
        # #region agent log
        _agent_debug_log_active(
            "auto_pick_place.py:_publish_conveyor_velocity",
            "conveyor velocity published",
            "H3",
            {
                "linear_speed_mps": float(linear_speed_mps),
                "track_cmd": float(track_msg.data),
                "roller_cmd": float(roller_msg.data),
                "track_subscribers": int(self._pub_conveyor_track.get_subscription_count()),
                "roller_subscribers": int(self._pub_conveyor_roller.get_subscription_count()),
            },
        )
        # #endregion

    def _run_conveyor_transport(
        self,
        release_pose: Pose,
        rect_half: Sequence[float],
        label: str = "conveyor_transport",
    ) -> bool:
        """启动真实传送带驱动，等待物体靠接触摩擦移动到另一端。"""
        if not self._param_bool("conveyor_transport_enabled"):
            self.get_logger().info(f"{label}: conveyor_transport_enabled=false，跳过输送")
            return True

        target_pose = self._conveyor_transport_end_pose(release_pose, rect_half)
        if target_pose is None:
            return False

        current_pose = Pose()
        current_pose.position.x = float(release_pose.position.x)
        current_pose.position.y = float(release_pose.position.y)
        current_pose.position.z = float(release_pose.position.z)
        current_pose.orientation = release_pose.orientation
        if self._rect is not None:
            current_pose = self._rect.pose

        start_xy = np.array([float(current_pose.position.x), float(current_pose.position.y)], dtype=float)
        end_xy = np.array([float(target_pose.position.x), float(target_pose.position.y)], dtype=float)
        travel = float(np.linalg.norm(end_xy - start_xy))
        if travel < 1e-4:
            self.get_logger().info(f"{label}: 物体已在传送带终点附近，无需输送")
            self._apply_released_rect_collision_scene(target_pose, rect_half)
            return True

        speed_mps = max(0.01, float(self.get_parameter("conveyor_transport_speed_mps").value))
        goal_tol = max(0.01, float(self.get_parameter("conveyor_transport_goal_tol_m").value))
        lateral_tol = max(0.03, float(self.get_parameter("conveyor_transport_lateral_tol_m").value))
        settle_sec = max(0.0, float(self.get_parameter("conveyor_transport_settle_sec").value))
        pad_sec = max(1.0, float(self.get_parameter("conveyor_transport_timeout_pad_sec").value))
        monitor_sec = max(0.05, float(self.get_parameter("conveyor_transport_monitor_period_sec").value))
        deadline = time.time() + travel / speed_mps + pad_sec
        direction = (end_xy - start_xy) / max(travel, 1e-6)
        start_dot = float(np.dot(start_xy, direction))
        target_dot = float(np.dot(end_xy, direction))
        reached = False
        last_progress_log = 0.0

        self.get_logger().info(
            f"{label}: start=({current_pose.position.x:.3f},{current_pose.position.y:.3f},{current_pose.position.z:.3f}) "
            f"-> end=({target_pose.position.x:.3f},{target_pose.position.y:.3f},{target_pose.position.z:.3f}), "
            f"travel={travel:.3f}m speed={speed_mps:.3f}m/s"
        )
        # #region agent log
        _agent_debug_log_active(
            "auto_pick_place.py:_run_conveyor_transport:start",
            "transport stage entered",
            "H4",
            {
                "travel_m": float(travel),
                "start_xy": [float(start_xy[0]), float(start_xy[1])],
                "end_xy": [float(end_xy[0]), float(end_xy[1])],
                "speed_mps": float(speed_mps),
            },
        )
        # #endregion
        self._rect_pose_can_move = True
        self._publish_conveyor_velocity(speed_mps)
        try:
            while time.time() < deadline:
                self._publish_conveyor_velocity(speed_mps)
                rect_pose = self._rect.pose if self._rect is not None else current_pose
                rect_xy = np.array(
                    [float(rect_pose.position.x), float(rect_pose.position.y)],
                    dtype=float,
                )
                along = float(np.dot(rect_xy, direction))
                progress = max(0.0, min(travel, along - start_dot))
                lateral_err = float(
                    abs(direction[0] * (rect_xy[1] - end_xy[1]) - direction[1] * (rect_xy[0] - end_xy[0]))
                )
                dist_to_goal = float(np.linalg.norm(rect_xy - end_xy))
                if time.time() - last_progress_log >= 1.0:
                    last_progress_log = time.time()
                    self.get_logger().info(
                        f"{label}: progress={progress:.3f}/{travel:.3f}m "
                        f"dist={dist_to_goal:.3f}m lateral={lateral_err:.3f}m"
                    )
                    # #region agent log
                    _agent_debug_log_active(
                        "auto_pick_place.py:_run_conveyor_transport:loop",
                        "transport progress",
                        "H4",
                        {
                            "progress_m": float(progress),
                            "travel_m": float(travel),
                            "dist_to_goal_m": float(dist_to_goal),
                            "lateral_err_m": float(lateral_err),
                        },
                    )
                    # #endregion
                if along >= (target_dot - goal_tol) and lateral_err <= lateral_tol:
                    reached = True
                    break
                time.sleep(monitor_sec)
        finally:
            self._publish_conveyor_velocity(0.0)
            time.sleep(0.05)
            self._publish_conveyor_velocity(0.0)

        if settle_sec > 0.0:
            time.sleep(settle_sec)

        final_pose = self._rect.pose if self._rect is not None else target_pose
        self._apply_released_rect_collision_scene(final_pose, rect_half)
        if not reached:
            self.get_logger().error(
                f"{label}: 物体未在超时前到达传送带末端，"
                f"last=({final_pose.position.x:.3f},{final_pose.position.y:.3f},{final_pose.position.z:.3f})"
            )
            return False

        self.get_logger().info(
            f"{label}: 已将 rect_pickup 输送到传送带另一端 "
            f"({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f})"
        )
        return True

    def _place_retreat_point(
        self, place_pt: Point, carton_ps: PoseStamped | None, post_place_retreat: float
    ) -> Point:
        out = Point(
            x=place_pt.x,
            y=place_pt.y,
            z=place_pt.z + post_place_retreat,
        )
        if carton_ps is None:
            return out
        try:
            _, _, carton_h = [float(v) for v in self.get_parameter("carton_outer_size_xyz").value]
            edge_clearance = max(0.06, float(self.get_parameter("place_entry_clearance").value))
            c = carton_ps.pose.position
            q = carton_ps.pose.orientation
            safe = self._point_with_local_offset(c, q, 0.0, 0.0, carton_h + edge_clearance)
            if safe.z > out.z + 1e-6:
                self.get_logger().info(
                    "退避高度抬高到箱沿以上: "
                    f"z={out.z:.3f} -> {safe.z:.3f} (carton_h={carton_h:.3f}, clearance={edge_clearance:.3f})"
                )
                out.z = safe.z
        except Exception as e:
            self.get_logger().warn(f"计算箱沿安全退避高度失败，将使用默认退避高度: {e}")
        return out

    def _on_rect(self, msg: PoseStamped) -> None:
        # 丢弃异常全零位姿；勿按「近 XY 原点」过滤，否则会丢掉合法物体或错误筛掉首帧。
        p = msg.pose.position
        if abs(p.x) < 1e-5 and abs(p.y) < 1e-5 and abs(p.z) < 1e-5:
            return
        # 过滤明显异常跳变（常见于桥接/反序列化抖动），避免把抓取点带飞。
        rect_is_expected_to_move = (
            self._rect_pose_can_move
            or self._fake_attach_active
            or self._suction_attached is True
        )
        if self._param_bool("use_known_rect_surface_center") and not rect_is_expected_to_move:
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

    def _on_world_pose_info(self, msg: TFMessage) -> None:
        rect = extract_model_pose(msg, "rect_pickup")
        if rect is not None:
            self._on_rect(rect)
        carton = extract_model_pose(msg, "carton_box")
        if carton is not None:
            self._on_carton(carton)

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
        # #region agent log
        prev = self._suction_attached
        _agent_debug_log(
            "auto_pick_place.py:_on_suction_state",
            "ros_suction_state_msg",
            "H1",
            {"data": bool(msg.data), "prev": prev},
        )
        # #endregion
        if self._fake_attach_active and not bool(msg.data):
            self.get_logger().debug(
                "SetEntityPose 仿真吸附兜底运行中，忽略 DetachableJoint detach 状态回传"
            )
            return
        self._suction_attached = bool(msg.data)
        if self._suction_attached:
            self._rect_pose_can_move = True

    def _wait_suction_state(self, expected: bool, timeout_sec: float) -> bool:
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            time.sleep(0.02)
            if self._suction_attached is expected:
                return True
        return False

    def _detach_pulse_before_attach(self) -> None:
        """
        发送 attach 前做一次短暂 detach，迫使 DetachableJoint 产生 detach→attach 的状态跳变。
        否则插件可能长期处于“已附着”并仅回应 “Already attached”，不再向 output_topic 发布 true。
        统一只通过 ROS publisher → ros_gz_bridge → Gazebo 单通道发送，避免 CLI 与 ROS 混用
        导致的时序竞争。
        """
        self._pub_detach.publish(Empty())
        time.sleep(0.5)

    def _wait_suction_attached_hybrid(self, timeout_sec: float, run_id: str = "pre-fix") -> bool:
        """
        确认附着：优先 /cs612/suction/state（ROS），最后再做一次 Gazebo Transport 直连采样。

        DetachableJoint 的 output_topic 是事件型消息，不是 latched/周期状态。attach 发生得很快时，
        后启动的 ``ign topic -e -n 1`` 经常会错过这一帧并阻塞到超时；因此这里避免循环调用 CLI。
        """
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < timeout_sec:
            if self._suction_attached is True:
                return True
            time.sleep(0.02)

        raw = _sample_gz_suction_state_raw(self._gz_bin)
        parsed = _parse_gz_boolean_data(raw)
        # #region agent log
        _agent_debug_log(
            "auto_pick_place.py:_wait_suction_attached_hybrid",
            "gz_poll_once",
            "H1",
            {"parsed": parsed, "raw_prefix": raw[:200]},
            run_id=run_id,
        )
        # #endregion
        if parsed is True:
            self._suction_attached = True
            self._rect_pose_can_move = True
            self._publish_assumed_state(True)
            self.get_logger().info(
                "吸附状态已由 Gazebo Transport 直连确认（ROS bridge 可能漏包）"
            )
            return True
        if self._suction_attached is not True and raw:
            self.get_logger().warn(
                "吸附状态事件未确认：GZ state 采样前缀 "
                f"{raw[:220]!r}"
            )
        return self._suction_attached is True

    def _publish_attach_burst(self) -> None:
        burst = max(1, int(self.get_parameter("suction_attach_burst_count").value))
        interval = max(0.01, float(self.get_parameter("suction_attach_burst_interval_sec").value))
        for i in range(burst):
            self._pub_attach.publish(Empty())
            if i + 1 < burst:
                time.sleep(interval)

    def _detach_via_gz_cli(self, repeats: int = 3) -> bool:
        """
        Gazebo Transport 直连 detach，绕开 ROS bridge/DDS 抖动。
        DetachableJoint 通过话题（非服务）接收指令，因此使用 ign topic 发布。
        WSL2 下 ign 启动较慢，timeout=10.0s。
        """
        ok = False
        for i in range(max(1, int(repeats))):
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10.0,
                )
                if proc.returncode == 0:
                    ok = True
                else:
                    err = proc.stderr.decode("utf-8", errors="replace").strip()[:200]
                    self.get_logger().warn(
                        f"detach_via_gz_cli [{i+1}/{repeats}] returncode={proc.returncode}, stderr={err!r}"
                    )
            except Exception as e:
                self.get_logger().warn(f"detach_via_gz_cli [{i+1}/{repeats}] exception: {e}")
            time.sleep(0.08)
        return ok

    def _attach_via_gz_cli(self, repeats: int = 1) -> bool:
        ok = False
        for i in range(max(1, int(repeats))):
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
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10.0,
                )
                if proc.returncode == 0:
                    ok = True
                else:
                    err = proc.stderr.decode("utf-8", errors="replace").strip()[:200]
                    self.get_logger().warn(
                        f"attach_via_gz_cli [{i+1}/{repeats}] returncode={proc.returncode}, stderr={err!r}"
                    )
            except Exception as e:
                self.get_logger().warn(f"attach_via_gz_cli [{i+1}/{repeats}] exception: {e}")
            time.sleep(0.08)
        return ok

    def _mark_attach_assumed(self, reason: str) -> None:
        self._suction_attached = True
        self._rect_pose_can_move = True
        self._pub_assumed_state.publish(Bool(data=True))
        self.get_logger().warn(
            f"{reason}: /cs612/suction/state 未可靠回传，"
            "但 attach 已下发且吸盘几何接触合格；按已吸附继续，并用位姿跟随保证物体贴合吸盘。"
        )

    def _publish_assumed_state(self, value: bool) -> None:
        self._pub_assumed_state.publish(Bool(data=value))
        self.get_logger().info(f"已发布 assumed_state: {value} -> /cs612/suction/assumed_state")

    def _rect_pose_under_suction(self, half_sizes: Sequence[float]) -> Pose | None:
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose is None:
            return None
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        touch_dz = max(0.0, float(self.get_parameter("touch_delta_z").value))
        half_z = max(0.001, float(half_sizes[2]) if len(half_sizes) >= 3 else 0.04)
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
        pose.orientation = _flat_yaw_quat_from_tool_orientation(cup_pose.orientation)
        return pose

    def _set_rect_pose(self, pose: Pose, wait_sec: float = 0.15, log_timeout: bool = False) -> bool:
        # 优先尝试 Gazebo Transport 直连（绕过 ROS bridge / parameter_bridge 可能存在的服务映射问题）
        gz_ok = self._set_rect_pose_via_gz(pose)
        if gz_ok:
            return True
        return self._set_rect_pose_ros(pose, wait_sec=wait_sec, log_timeout=log_timeout)

    def _set_rect_pose_ros(self, pose: Pose, wait_sec: float = 0.15, log_timeout: bool = False) -> bool:
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
        if wait_sec <= 0.0:
            return True
        if not _spin_future(self, fut, wait_sec, "set_rect_pose", log_timeout=log_timeout):
            return False
        try:
            res = fut.result()
            return bool(res and res.success)
        except Exception:
            return False

    def _set_rect_pose_via_gz(self, pose: Pose) -> bool:
        gz_bin = self._gz_bin
        pfx = self._gz_msg_pfx
        q = _quat_normalize(pose.orientation)
        req = (
            f'name: "rect_pickup"\n'
            f'position {{\n'
            f'  x: {pose.position.x:.5f}\n'
            f'  y: {pose.position.y:.5f}\n'
            f'  z: {pose.position.z:.5f}\n'
            f'}}\n'
            f'orientation {{\n'
            f'  x: {q.x:.8f}\n'
            f'  y: {q.y:.8f}\n'
            f'  z: {q.z:.8f}\n'
            f'  w: {q.w:.8f}\n'
            f'}}'
        )
        try:
            proc = subprocess.run(
                [gz_bin, "service", "-s", "/world/arm_world/set_pose",
                 "--reqtype", f"{pfx}.Pose",
                 "--reptype", f"{pfx}.Boolean",
                 "--timeout", "1500",
                 "-r", req],
                capture_output=True, text=True, timeout=2.5, check=False,
            )
            if proc.returncode == 0:
                stderr_lower = (proc.stderr or "").lower()
                stdout_lower = (proc.stdout or "").lower()
                if "error" not in stderr_lower and "error" not in stdout_lower:
                    return True
        except Exception:
            pass
        return False

    def _fake_attach_loop(self, half_sizes: Sequence[float]) -> None:
        hz = max(1.0, float(self.get_parameter("fake_attach_update_hz").value))
        period = 1.0 / hz
        wait_sec = min(0.08, max(0.02, period * 1.5))
        fail_streak = 0
        max_fail_streak = 30
        while rclpy.ok() and not self._fake_attach_stop.is_set():
            pose = self._rect_pose_under_suction(half_sizes)
            if pose is not None:
                ok = self._set_rect_pose_ros(pose, wait_sec=wait_sec, log_timeout=False)
                if not ok:
                    ok = self._set_rect_pose_via_gz(pose)
                if ok:
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= max_fail_streak:
                        self.get_logger().error(
                            f"SetEntityPose 连续失败 {fail_streak} 次，停止仿真吸附兜底"
                        )
                        self._fake_attach_active = False
                        break
            time.sleep(period)

    def _snap_fake_attach_pose(
        self,
        half_sizes: Sequence[float],
        label: str,
        attempts: int = 2,
    ) -> bool:
        if not self._fake_attach_active:
            return False
        for idx in range(max(1, int(attempts))):
            pose = self._rect_pose_under_suction(half_sizes)
            if pose is None:
                time.sleep(0.05)
                continue
            if self._set_rect_pose(pose, wait_sec=0.3, log_timeout=False):
                self._rect_pose_can_move = True
                self.get_logger().info(
                    f"{label}: SetEntityPose 已同步物体到吸盘下方 "
                    f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
                )
                return True
            if idx + 1 < max(1, int(attempts)):
                time.sleep(0.08)
        self.get_logger().warn(f"{label}: SetEntityPose 同步吸附位姿失败")
        return False

    def _rect_pose_centered_in_carton(
        self, carton_ps: PoseStamped | None, half_sizes: Sequence[float]
    ) -> Pose | None:
        if carton_ps is None:
            return None
        half_z = max(0.001, float(half_sizes[2]) if len(half_sizes) >= 3 else 0.04)
        floor_top_z = float(self.get_parameter("carton_floor_top_z").value)
        c = carton_ps.pose.position
        q = _quat_normalize(carton_ps.pose.orientation)
        dx, dy, dz = _quat_rotate_vec(q, 0.0, 0.0, floor_top_z + half_z)
        pose = Pose()
        pose.position.x = c.x + dx
        pose.position.y = c.y + dy
        pose.position.z = c.z + dz
        pose.orientation = q
        return pose

    def _release_and_center_rect_in_carton(
        self,
        carton_ps: PoseStamped | None,
        half_sizes: Sequence[float],
        label: str = "place_center_release",
    ) -> bool:
        final_pose = self._rect_pose_centered_in_carton(carton_ps, half_sizes)
        self._suction_attached = None
        self._stop_fake_attach(label)
        self._pub_detach.publish(Empty())
        if not self._wait_suction_state(False, 0.7) and self._suction_attached is True:
            self._detach_via_gz_cli(repeats=1)
        self._suction_attached = False
        self._publish_assumed_state(False)

        if final_pose is None:
            self.get_logger().warn(f"{label}: 未获得容器位姿，无法执行中心/姿态最终对齐")
            return False

        for i in range(3):
            if self._set_rect_pose(final_pose, wait_sec=0.5, log_timeout=(i == 2)):
                self._rect_pose_can_move = True
                self._apply_released_rect_collision_scene(final_pose, half_sizes)
                self.get_logger().info(
                    f"{label}: 已将 rect_pickup 中心对齐容器中心并按容器边缘朝向放置 "
                    f"({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f}), "
                    f"q=({final_pose.orientation.x:.4f},{final_pose.orientation.y:.4f},"
                    f"{final_pose.orientation.z:.4f},{final_pose.orientation.w:.4f})"
                )
                return True
            time.sleep(0.12)
        self.get_logger().warn(f"{label}: 最终中心/姿态对齐 SetEntityPose 失败")
        return False

    def _planned_rect_pose_at_place(
        self,
        place_tcp: Point,
        place_orientation: Quaternion,
        half_sizes: Sequence[float],
    ) -> Pose:
        half_z = max(0.001, float(half_sizes[2]) if len(half_sizes) >= 3 else 0.04)
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        touch_dz = max(0.0, float(self.get_parameter("touch_delta_z").value))
        contact = self._point_with_local_offset(
            place_tcp,
            place_orientation,
            0.0,
            0.0,
            suction_contact_offset,
        )
        pose = Pose()
        pose.position.x = contact.x
        pose.position.y = contact.y
        # 释放到支撑面时不再额外抬高 touch_dz，避免 detach 后出现“先悬空再落下”。
        pose.position.z = max(half_z, contact.z - half_z)
        pose.orientation = _flat_yaw_quat_from_tool_orientation(place_orientation)
        return pose

    def _release_rect_at_planned_place(
        self,
        place_tcp: Point,
        place_orientation: Quaternion,
        half_sizes: Sequence[float],
        label: str = "place_release_planned",
    ) -> bool:
        final_pose = self._planned_rect_pose_at_place(place_tcp, place_orientation, half_sizes)
        # 先在仍然吸附时把物体贴到最终支撑位，detach 时视觉和物理都更平稳。
        self._set_rect_pose(final_pose, wait_sec=0.25, log_timeout=False)
        time.sleep(0.12)
        self._suction_attached = None
        self._stop_fake_attach(label)
        self._pub_detach.publish(Empty())
        if not self._wait_suction_state(False, 0.7) and self._suction_attached is True:
            self._detach_via_gz_cli(repeats=1)
        self._suction_attached = False
        self._publish_assumed_state(False)

        for i in range(3):
            if self._set_rect_pose(final_pose, wait_sec=0.5, log_timeout=(i == 2)):
                self._rect_pose_can_move = True
                self._apply_released_rect_collision_scene(final_pose, half_sizes)
                self.get_logger().info(
                    f"{label}: 已按规划放置位姿释放 rect_pickup "
                    f"({final_pose.position.x:.3f}, {final_pose.position.y:.3f}, {final_pose.position.z:.3f}), "
                    f"q=({final_pose.orientation.x:.4f},{final_pose.orientation.y:.4f},"
                    f"{final_pose.orientation.z:.4f},{final_pose.orientation.w:.4f})"
                )
                return True
            time.sleep(0.12)
        self.get_logger().warn(f"{label}: 规划放置位姿 SetEntityPose 失败")
        return False

    def _apply_released_rect_collision_scene(
        self, pose: Pose, half_sizes: Sequence[float], timeout_sec: float = 2.0
    ) -> bool:
        sx = 2.0 * float(half_sizes[0]) if len(half_sizes) >= 1 else 0.20
        sy = 2.0 * float(half_sizes[1]) if len(half_sizes) >= 2 else 0.14
        sz = 2.0 * float(half_sizes[2]) if len(half_sizes) >= 3 else 0.08
        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [
            self._make_collision_box(
                "scene_rect_pickup",
                pose.position,
                pose.orientation,
                [sx, sy, sz],
            )
        ]
        scene.robot_state.is_diff = True
        aco_remove = AttachedCollisionObject()
        aco_remove.link_name = "suction_tcp_link"
        aco_remove.object = CollisionObject()
        aco_remove.object.id = "scene_rect_pickup"
        aco_remove.object.operation = CollisionObject.REMOVE
        scene.robot_state.attached_collision_objects = [aco_remove]

        req = ApplyPlanningScene.Request()
        req.scene = scene
        try:
            fut = self._scene_client.call_async(req)
        except Exception as e:
            self.get_logger().warn(f"释放后规划场景清理发送失败: {e}")
            return False
        if not _spin_future(self, fut, timeout_sec, "release_rect_planning_scene", log_timeout=False):
            self.get_logger().warn("释放后规划场景清理超时，将等待 planning_scene_spawner 周期刷新")
            return False
        res = fut.result()
        ok = bool(res and res.success)
        if ok:
            self.get_logger().info("释放后已清理 MoveIt attached object，并将 rect_pickup 写回世界碰撞物")
        else:
            self.get_logger().warn("释放后规划场景清理失败，将等待 planning_scene_spawner 周期刷新")
        return ok

    def _start_fake_attach(self, half_sizes: Sequence[float], reason: str) -> bool:
        if not self._param_bool("fake_attach_set_pose_fallback"):
            return False
        if self._fake_attach_active:
            return True
        # 如果物理约束已经存在（_suction_attached=True），不要发送 detach 摧毁它，
        # 只需启动 SetEntityPose 跟随作为双重保险。
        physical_attached = self._suction_attached is True
        ros_ok = self._set_pose_client.wait_for_service(timeout_sec=1.0)
        if not ros_ok:
            self.get_logger().warn(
                f"{reason}: /world/arm_world/set_pose ROS 服务不可用，将仅使用 Gazebo Transport CLI 直连"
            )
        pose = self._rect_pose_under_suction(half_sizes)
        if pose is not None:
            ros_set = ros_ok and self._set_rect_pose_ros(pose, wait_sec=0.3)
            gz_ok = False
            if not ros_set:
                gz_ok = self._set_rect_pose_via_gz(pose)
            if ros_set:
                self.get_logger().info(
                    f"SetEntityPose 初始位姿已通过 ROS 服务下发: "
                    f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
                )
            elif gz_ok:
                self.get_logger().info(
                    f"SetEntityPose 初始位姿已通过 Gazebo Transport CLI 下发: "
                    f"({pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f})"
                )
        self._fake_attach_stop.clear()
        self._fake_attach_active = True
        self._rect_pose_can_move = True
        self._suction_attached = True
        self._publish_assumed_state(True)
        self._pub_visual_attached.publish(Bool(data=True))
        self._fake_attach_thread = threading.Thread(
            target=self._fake_attach_loop,
            args=(list(half_sizes),),
            daemon=True,
        )
        self._fake_attach_thread.start()
        if not physical_attached:
            self._pub_detach.publish(Empty())
            self._detach_via_gz_cli(repeats=1)
            self._snap_fake_attach_pose(half_sizes, "fake_attach_start")
            self.get_logger().warn(
                f"{reason}: DetachableJoint 物理约束未确认，启用 Gazebo SetEntityPose 仿真吸附兜底"
            )
        else:
            self.get_logger().info(
                f"{reason}: 物理约束已存在，SetEntityPose 作为跟随辅助同时运行（双保险）"
            )
        return True

    def _stop_fake_attach(self, label: str) -> None:
        if not self._fake_attach_active:
            self._pub_visual_attached.publish(Bool(data=False))
            return
        self._fake_attach_active = False
        self._fake_attach_stop.set()
        thread = self._fake_attach_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)
        self._fake_attach_thread = None
        self._pub_visual_attached.publish(Bool(data=False))
        self.get_logger().info(f"{label}: 已停止 SetEntityPose 仿真吸附兜底")

    def _ensure_detached(self, attempts: int = 8, wait_each_sec: float = 0.35) -> bool:
        """
        DetachableJoint 在 Gazebo 中可能以“已附着”状态启动。
        这里反复发送 detach 并尽量等待 state=false，防止未抓取时物体跟随机械臂。
        """
        attempts = max(1, int(attempts))
        wait_each_sec = max(0.1, float(wait_each_sec))
        saw_state_false = False
        self._stop_fake_attach("启动 detach")
        for i in range(attempts):
            self._pub_detach.publish(Empty())
            if self._wait_suction_state(False, wait_each_sec):
                saw_state_false = True
                if i > 0:
                    self.get_logger().info(f"detach 清状态成功（第 {i + 1}/{attempts} 次）")
                return True
            time.sleep(0.05)
        if self._suction_attached is True:
            self.get_logger().error("detach 清状态失败：吸附仍为 true，终止本次抓取以避免物体误跟随")
            return False
        # 若 state 话题暂不可用，改走 Gazebo 直连 detach；这是 DetachableJoint 社区常见稳态做法。
        if self._detach_via_gz_cli(repeats=6):
            self._suction_attached = False
            self._publish_assumed_state(False)
            self.get_logger().warn(f"ROS detach 未确认，已通过 {self._gz_bin} 直连 detach 兜底")
            return True
        # 在 bridge / state 话题抖动时，Gazebo 可能已执行 detach 但无状态回传。
        # 若未明确观测到 attached=true，则按"已发送 detach，继续执行"处理，避免误中止。
        if not saw_state_false and self._suction_attached is not True:
            self._suction_attached = False
            self._publish_assumed_state(False)
            self.get_logger().warn("detach 状态未确认，但未观测到仍附着；按已释放继续执行")
            return True
        # 在 bridge / state 话题抖动时，Gazebo 可能已执行 detach 但无状态回传。
        # 若未明确观测到 attached=true，则按“已发送 detach，继续执行”处理，避免误中止。
        if not saw_state_false and self._suction_attached is not True:
            self._suction_attached = False
            self.get_logger().warn("detach 状态未确认，但未观测到仍附着；按已释放继续执行")
            return True
        self.get_logger().error("detach 无法确认且检测到可能仍附着，终止以避免误附着")
        return False

    def _current_rect_center_z(self) -> float | None:
        if self._rect is None:
            return None
        try:
            return float(self._pose_to_base(self._rect).pose.position.z)
        except Exception:
            return None

    def _current_rect_top(self, half_sizes: Sequence[float]) -> Point | None:
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

    def _is_tcp_near_target(self, target: Point, tol_xy: float = 0.08, tol_z: float = 0.08) -> bool:
        tcp = self._lookup_link_pose_in_base("suction_tcp_link")
        if tcp is None:
            return False
        dx = tcp.position.x - target.x
        dy = tcp.position.y - target.y
        dz = tcp.position.z - target.z
        return (math.hypot(dx, dy) <= max(0.01, tol_xy)) and (abs(dz) <= max(0.01, tol_z))

    def _suction_alignment_metrics(
        self, rect_half_sizes: Sequence[float]
    ) -> tuple[float, float, float, float, float, bool] | None:
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        top = self._current_rect_top_live(rect_half_sizes) or self._current_rect_top(rect_half_sizes)
        if cup_pose is None or top is None:
            if cup_pose is None and top is None:
                self.get_logger().warn("[alignment_metrics] 无法读取 suction_cup_link 和物体位姿")
            elif cup_pose is None:
                self.get_logger().warn("[alignment_metrics] 无法读取 suction_cup_link TF")
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
            lateral_tol = float(self.get_parameter("suction_touch_lateral_tol").value)
            vertical_tol = float(self.get_parameter("suction_touch_vertical_tol").value)
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

    def _check_arm_object_collision(self, rect_half: Sequence[float]) -> tuple[bool, str]:
        if not self._param_bool("arm_collision_check_enabled"):
            return False, ""
        top = self._current_rect_top_live(list(rect_half)) or self._current_rect_top(list(rect_half))
        if top is None:
            return False, ""
        obj_center_x = float(top.x)
        obj_center_y = float(top.y)
        obj_center_z = float(top.z) - float(rect_half[2])
        obj_half_x = float(rect_half[0])
        obj_half_y = float(rect_half[1])
        obj_half_z = float(rect_half[2])
        margin = float(self.get_parameter("arm_collision_margin").value)
        link_radii = list(self.get_parameter("arm_collision_link_radii").value)
        link_names = [
            "shoulder_link",
            "upperarm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        suction_offset_z = float(self.get_parameter("suction_contact_offset_z").value)
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        for idx, link_name in enumerate(link_names):
            link_radius = float(link_radii[idx]) if idx < len(link_radii) else 0.04
            try:
                from rclpy.duration import Duration as RclDuration
                tf = self._tf_buffer.lookup_transform(
                    "base_link",
                    link_name,
                    rclpy.time.Time(),
                    timeout=RclDuration(seconds=0.1),
                )
            except Exception:
                continue
            lx = float(tf.transform.translation.x)
            ly = float(tf.transform.translation.y)
            lz = float(tf.transform.translation.z)
            eff_half_x = obj_half_x + link_radius + margin
            eff_half_y = obj_half_y + link_radius + margin
            eff_half_z_bottom = obj_half_z + link_radius + margin
            eff_half_z_top = obj_half_z + link_radius + margin + suction_offset_z
            dx = abs(lx - obj_center_x)
            dy = abs(ly - obj_center_y)
            dz_bottom = lz - (obj_center_z - obj_half_z)
            dz_top = (obj_center_z + obj_half_z + suction_offset_z) - lz
            if dx < eff_half_x and dy < eff_half_y and dz_bottom < eff_half_z_bottom and lz < (obj_center_z + obj_half_z + suction_offset_z) and lz > (obj_center_z - obj_half_z - link_radius - margin):
                desc = (
                    f"arm link '{link_name}' at ({lx:.4f},{ly:.4f},{lz:.4f}) "
                    f"collides with object AABB center=({obj_center_x:.4f},{obj_center_y:.4f},{obj_center_z:.4f}) "
                    f"half=({obj_half_x:.4f},{obj_half_y:.4f},{obj_half_z:.4f}) "
                    f"link_radius={link_radius:.4f} margin={margin:.4f}"
                )
                return True, desc
        return False, ""

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
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
            if cup_pose is None:
                self.get_logger().warn(f"{label}: 无法读取 suction_cup_link TF，跳过朝向验证")
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

        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
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

        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        live_top = self._current_rect_top_live(list(half_sizes)) or self._current_rect_top(list(half_sizes))
        target_top = live_top if live_top is not None else top

        if cup_pose is None:
            self.get_logger().warn(
                "[centerline_pre_touch] 无法读取 suction_cup_link TF，跳过最终对齐校验"
            )
            return True

        cup_bottom = self._point_with_local_offset(
            cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
        )
        dx = cup_bottom.x - target_top.x
        dy = cup_bottom.y - target_top.y
        lateral_err = math.hypot(dx, dy)
        vertical_err = abs(cup_bottom.z - (target_top.z + suction_contact_offset))
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
            cup_pose_new = self._lookup_link_pose_in_base("suction_cup_link")
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

        cup_pose_final = self._lookup_link_pose_in_base("suction_cup_link")
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
        """approach 点就位后验证 XY 对齐和朝向。Z 误差仅记录日志，
        不阻塞下压——笛卡尔触碰阶段会做最终 Z 贴合。"""
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        max_verify_iters = 3
        lateral_max = max(0.010, float(self.get_parameter("approach_verify_lateral_tol_m").value))
        min_cos = max(0.90, float(self.get_parameter("orientation_min_cos_before_touch").value))
        target_orient = orientations[0] if orientations else _suction_down_quat(0.0)
        max_fix_step = max(0.01, float(self.get_parameter("approach_verify_max_correction_step_m").value))
        reject_large_err = max(0.10, float(self.get_parameter("approach_verify_reject_large_error_m").value))
        relaxed_lateral_tol = max(
            lateral_max,
            float(self.get_parameter("approach_verify_relaxed_lateral_tol_m").value),
        )

        top = self._current_rect_top(list(half_sizes))
        if top is None:
            top = self._current_rect_top_live(list(half_sizes))
        if top is None:
            self.get_logger().warn(f"{label}: 无法读取物体顶面坐标，跳过 approach 验证")
            return True

        current_approach = Point(x=approach.x, y=approach.y, z=approach.z)

        for attempt in range(1, max_verify_iters + 1):
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
            if cup_pose is None:
                self.get_logger().warn(f"{label}[{attempt}]: 无法读取 suction_cup_link TF，跳过验证")
                return True

            down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
            down_cos = -down_axis[2]

            cup_bottom = self._point_with_local_offset(
                cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
            )
            dx = cup_bottom.x - top.x
            dy = cup_bottom.y - top.y
            lateral_err = math.hypot(dx, dy)
            z_err = abs(cup_bottom.z - approach.z)

            self.get_logger().info(
                f"{label}[{attempt}]: lateral={lateral_err:.4f}m(max={lateral_max:.4f}), "
                f"z_err={z_err:.4f}m, down_cos={down_cos:.4f}(min={min_cos:.4f})"
            )
            if lateral_err > reject_large_err:
                self.get_logger().warn(
                    f"{label}[{attempt}]: lateral_err={lateral_err:.3f}m 过大，拒绝大步修正并回到安全分支"
                )
                return False

            xy_ok = lateral_err <= lateral_max
            orient_ok = down_cos >= min_cos

            # XY + 朝向通过即视为 approach 有效；Z 由后续笛卡尔下压贴合。
            if xy_ok and orient_ok:
                self.get_logger().info(
                    f"{label}[{attempt}]: approach 验证通过（xy+orient OK, "
                    f"z_err={z_err:.4f}m 由下压阶段修正）"
                )
                return True

            self.get_logger().warn(
                f"{label}[{attempt}]: approach 未通过 "
                f"xy={'OK' if xy_ok else 'FAIL'} "
                f"orient={'OK' if orient_ok else 'FAIL'}"
            )

            # 若当前关节状态命中坏姿态，终止 approach 校正。
            if not self._pick_pose_guard_ok(label):
                self.get_logger().warn(
                    f"{label}[{attempt}]: 当前关节命中抓取护栏，停止 approach 校正"
                )
                return False

            step_norm = math.hypot(dx, dy)
            if step_norm > max_fix_step and step_norm > 1e-9:
                scale = max_fix_step / step_norm
                dx *= scale
                dy *= scale
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

        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose is not None:
            cup_bottom_final = self._point_with_local_offset(
                cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
            )
            final_lateral = math.hypot(cup_bottom_final.x - top.x, cup_bottom_final.y - top.y)
            self.get_logger().warn(
                f"{label}: 验证结束 lateral={final_lateral:.4f}m, "
                f"strict_xy={'OK' if final_lateral <= lateral_max else 'FAIL'}, "
                f"relaxed_xy={'OK' if final_lateral <= relaxed_lateral_tol else 'FAIL'}"
            )
        return final_lateral <= lateral_max if cup_pose is not None else False

    def _verify_place_hover_pose(
        self,
        hover_target: Point,
        label: str,
    ) -> bool:
        suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
        lateral_max = max(0.010, float(self.get_parameter("place_verify_lateral_tol_m").value))
        min_cos = max(0.95, float(self.get_parameter("orientation_min_cos_before_touch").value))
        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose is None:
            self.get_logger().warn(f"{label}: 无法读取 suction_cup_link TF，跳过放置 hover 验证")
            return True
        cup_bottom = self._point_with_local_offset(
            cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
        )
        dx = cup_bottom.x - hover_target.x
        dy = cup_bottom.y - hover_target.y
        lateral_err = math.hypot(dx, dy)
        down_axis = _quat_rotate_vec(cup_pose.orientation, 0.0, 0.0, 1.0)
        down_cos = -down_axis[2]
        self.get_logger().info(
            f"{label}: lateral={lateral_err:.4f}m(max={lateral_max:.4f}), "
            f"down_cos={down_cos:.4f}(min={min_cos:.4f})"
        )
        ok = lateral_err <= lateral_max and down_cos >= min_cos and self._place_pose_guard_ok(label)
        if not ok:
            self.get_logger().warn(
                f"{label}: 放置 hover 未通过，"
                f"xy={'OK' if lateral_err <= lateral_max else 'FAIL'} "
                f"orient={'OK' if down_cos >= min_cos else 'FAIL'}"
            )
        return ok

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
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
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
    ) -> CollisionObject:
        co = CollisionObject()
        co.id = object_id
        co.header.frame_id = "base_link"
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
        for ee_link in ("suction_cup_link",):
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
            f"[suction_cup_link] ↔ scene_rect_pickup allow={allow}"
        )
        return True

    def _build_joint_goal(self, positions: List[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = False
        req = goal.request
        req.group_name = "arm"
        req.num_planning_attempts = 15
        req.allowed_planning_time = 8.0
        vel, acc = self._current_motion_scales()
        req.max_velocity_scaling_factor = min(max(vel, 0.01), 1.0)
        req.max_acceleration_scaling_factor = min(max(acc, 0.01), 1.0)
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnect"
        c = Constraints()
        for name, pos in zip(_ARM_JOINTS, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.12
            jc.tolerance_below = 0.12
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

    def _build_pose_goal(self, target: Point, orientation: Quaternion, label: str = "") -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = False
        req = goal.request
        req.group_name = "arm"
        req.num_planning_attempts = 30
        req.allowed_planning_time = 15.0
        vel, acc = self._current_motion_scales()
        req.max_velocity_scaling_factor = min(max(vel, 0.01), 1.0)
        req.max_acceleration_scaling_factor = min(max(acc, 0.01), 1.0)
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnect"

        pos_tol = max(0.001, float(self.get_parameter("pose_position_tolerance").value))
        # 不设下限裁剪，允许极紧容差生效
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
        # Z 轴（yaw）收紧但不过度，防止旋转一整圈
        oc.absolute_z_axis_tolerance = min(ori_tol, 0.04)
        oc.weight = 50.0

        c = Constraints()
        c.position_constraints = [pc]
        c.orientation_constraints = [oc]
        if self._pick_pose_guard_active(label):
            c.joint_constraints = self._pick_pose_joint_constraints()
        elif self._place_pose_guard_active(label):
            c.joint_constraints = self._place_pose_joint_constraints()
        req.goal_constraints = [c]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        return goal

    def _range_joint_constraint(
        self, joint_name: str, lower: float, upper: float, weight: float = 0.8
    ) -> JointConstraint:
        center = 0.5 * (float(lower) + float(upper))
        jc = JointConstraint()
        jc.joint_name = joint_name
        jc.position = center
        jc.tolerance_below = max(0.001, center - float(lower))
        jc.tolerance_above = max(0.001, float(upper) - center)
        jc.weight = float(weight)
        return jc

    def _pick_pose_joint_constraints(self) -> list[JointConstraint]:
        j2_min = float(self.get_parameter("pick_pose_guard_joint2_min").value)
        j2_max = float(self.get_parameter("pick_pose_guard_joint2_max").value)
        j3_min = float(self.get_parameter("pick_pose_guard_joint3_min").value)
        j3_max = float(self.get_parameter("pick_pose_guard_joint3_max").value)
        j5_min = float(self.get_parameter("pick_pose_guard_joint5_min").value)
        j5_max = float(self.get_parameter("pick_pose_guard_joint5_max").value)
        j4_abs_max = float(self.get_parameter("pick_pose_guard_joint4_abs_max").value)
        j6_abs_max = float(self.get_parameter("pick_pose_guard_joint6_abs_max").value)
        return [
            self._range_joint_constraint("shoulder_lift_joint", j2_min, j2_max),
            self._range_joint_constraint("elbow_joint", j3_min, j3_max),
            self._range_joint_constraint("wrist_1_joint", -j4_abs_max, j4_abs_max),
            self._range_joint_constraint("wrist_2_joint", j5_min, j5_max),
            self._range_joint_constraint("wrist_3_joint", -j6_abs_max, j6_abs_max),
        ]

    def _place_pose_joint_constraints(self) -> list[JointConstraint]:
        j2_min = float(self.get_parameter("place_pose_guard_joint2_min").value)
        j2_max = float(self.get_parameter("place_pose_guard_joint2_max").value)
        j3_min = float(self.get_parameter("place_pose_guard_joint3_min").value)
        j3_max = float(self.get_parameter("place_pose_guard_joint3_max").value)
        j5_min = float(self.get_parameter("place_pose_guard_joint5_min").value)
        j5_max = float(self.get_parameter("place_pose_guard_joint5_max").value)
        j4_abs_max = float(self.get_parameter("place_pose_guard_joint4_abs_max").value)
        j6_abs_max = float(self.get_parameter("place_pose_guard_joint6_abs_max").value)
        return [
            self._range_joint_constraint("shoulder_lift_joint", j2_min, j2_max),
            self._range_joint_constraint("elbow_joint", j3_min, j3_max),
            self._range_joint_constraint("wrist_1_joint", -j4_abs_max, j4_abs_max),
            self._range_joint_constraint("wrist_2_joint", j5_min, j5_max),
            self._range_joint_constraint("wrist_3_joint", -j6_abs_max, j6_abs_max),
        ]

    def _current_motion_scales(self) -> tuple[float, float]:
        if self._motion_profile == "far":
            return (
                float(self.get_parameter("far_move_velocity_scale").value),
                float(self.get_parameter("far_move_acceleration_scale").value),
            )
        if self._motion_profile == "near":
            return (
                float(self.get_parameter("near_move_velocity_scale").value),
                float(self.get_parameter("near_move_acceleration_scale").value),
            )
        return (
            float(self.get_parameter("move_velocity_scale").value),
            float(self.get_parameter("move_acceleration_scale").value),
        )

    def _set_motion_profile(self, profile: str) -> None:
        if profile not in ("default", "far", "near"):
            profile = "default"
        self._motion_profile = profile

    def _run_stage(self, stage: StageName, fn, on_fail: str) -> bool:
        self.get_logger().info(f"[阶段] {stage.name} 开始")
        ok = bool(fn())
        if ok:
            self.get_logger().info(f"[阶段] {stage.name} 完成")
            return True
        self.get_logger().error(f"[阶段] {stage.name} 失败: {on_fail}")
        return False

    def _apply_stage_motion_profile(self, stage: StageName) -> None:
        # 只调整速度/加速度，不改变原有抓放路径方向与拓扑
        if stage in (StageName.PRE_PICK, StageName.TRANSFER, StageName.HOME):
            self._set_motion_profile("far")
            return
        if stage in (StageName.APPROACH, StageName.TOUCH, StageName.PLACE, StageName.RETREAT):
            self._set_motion_profile("near")
            return
        self._set_motion_profile("default")

    def _send_pose_goal(self, target: Point, orientation: Quaternion, label: str) -> bool:
        if not self._action.wait_for_server(timeout_sec=60.0):
            self.get_logger().error("move_action 不可用")
            return False
        goal = self._build_pose_goal(target, orientation, label)
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
            # MoveIt 规划成功后，Gazebo 执行可能存在跟踪偏差 → 强制笛卡尔朝向校正
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
        # 0.95 ≈ 18° — 若偏差小于此值则视为可接受
        min_cos_snap = 0.95
        if down_cos >= min_cos_snap:
            self.get_logger().info(
                f"{label}: post-MoveIt 朝向已满足 cos={down_cos:.5f} >= {min_cos_snap}"
            )
            return
        self.get_logger().warn(
            f"{label}: post-MoveIt 朝向偏差 cos={down_cos:.5f} < {min_cos_snap}，"
            f"执行笛卡尔朝向校正"
        )
        snap_target = Point(
            x=float(cup_pose.position.x),
            y=float(cup_pose.position.y),
            z=float(cup_pose.position.z),
        )
        correction_weight = max(5.0, float(self.get_parameter("orientation_correction_weight").value))
        for attempt in range(3):
            moved = self._move_cartesian_direct(
                snap_target,
                orientation,
                mode="place" if "place" in label else "pick",
                label=f"{label}_orient_snap[{attempt}]",
                keep_xy_from_current=True,
                orientation_weight_override=correction_weight,
                joint_step_limit_override=0.06,
            )
            if not moved:
                self.get_logger().warn(
                    f"{label}: 笛卡尔朝向校正 attempt {attempt + 1} 失败"
                )
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
        self.get_logger().warn(
            f"{label}: post-MoveIt 朝向校正未达标，将依赖后续 _ensure_suction_facing_down"
        )

    def _move_target_with_moveit_pose(
        self, target: Point, orientations: List[Quaternion], label: str
    ) -> bool:
        for idx, ori in enumerate(orientations, start=1):
            if self._send_pose_goal(target, ori, f"{label}_pose[{idx}]"):
                if self._pick_pose_guard_ok(label):
                    return True
                self.get_logger().warn(
                    f"{label}_pose[{idx}]: 命中抓取姿态护栏，放弃该解并尝试其它规划分支"
                )
        return False

    def _pick_pose_guard_ok(self, label: str) -> bool:
        if not self._pick_pose_guard_active(label):
            return True
        joints = self._current_arm_positions()
        if joints is None or len(joints) != 6:
            return True
        j2, j3, j4, j5, j6 = (
            _wrap_to_pi(float(joints[1])),
            _wrap_to_pi(float(joints[2])),
            _wrap_to_pi(float(joints[3])),
            _wrap_to_pi(float(joints[4])),
            _wrap_to_pi(float(joints[5])),
        )
        j2_min = float(self.get_parameter("pick_pose_guard_joint2_min").value)
        j2_max = float(self.get_parameter("pick_pose_guard_joint2_max").value)
        j3_min = float(self.get_parameter("pick_pose_guard_joint3_min").value)
        j3_max = float(self.get_parameter("pick_pose_guard_joint3_max").value)
        j5_min = float(self.get_parameter("pick_pose_guard_joint5_min").value)
        j5_max = float(self.get_parameter("pick_pose_guard_joint5_max").value)
        j4_abs_max = float(self.get_parameter("pick_pose_guard_joint4_abs_max").value)
        j6_abs_max = float(self.get_parameter("pick_pose_guard_joint6_abs_max").value)
        ok = (
            j2_min <= j2 <= j2_max
            and j3_min <= j3 <= j3_max
            and j5_min <= j5 <= j5_max
            and abs(j4) <= j4_abs_max
            and abs(j6) <= j6_abs_max
        )
        if not ok:
            self.get_logger().warn(
                "抓取姿态护栏触发: "
                f"j2={j2:.3f}[{j2_min:.2f},{j2_max:.2f}], "
                f"j3={j3:.3f}[{j3_min:.2f},{j3_max:.2f}], "
                f"j4={j4:.3f}|<= {j4_abs_max:.2f}, "
                f"j5={j5:.3f}[{j5_min:.2f},{j5_max:.2f}], "
                f"j6={j6:.3f}|<= {j6_abs_max:.2f}"
            )
        return ok

    def _pick_pose_guard_active(self, label: str) -> bool:
        if not self._param_bool("pick_pose_guard_enabled"):
            return False
        # 仅在抓取关键阶段启用，避免影响放置阶段和常规运动。
        guarded_tokens = ("pre_pick", "approach", "pick_", "touch", "realign")
        return any(token in label for token in guarded_tokens)

    def _place_pose_guard_active(self, label: str) -> bool:
        if not self._param_bool("place_pose_guard_enabled"):
            return False
        guarded_tokens = ("place_", "retreat", "transfer")
        return any(token in label for token in guarded_tokens)

    def _place_pose_guard_ok(self, label: str) -> bool:
        if not self._place_pose_guard_active(label):
            return True
        joints = self._current_arm_positions()
        if joints is None or len(joints) != 6:
            return True
        j2, j3, j4, j5, j6 = (
            _wrap_to_pi(float(joints[1])),
            _wrap_to_pi(float(joints[2])),
            _wrap_to_pi(float(joints[3])),
            _wrap_to_pi(float(joints[4])),
            _wrap_to_pi(float(joints[5])),
        )
        j2_min = float(self.get_parameter("place_pose_guard_joint2_min").value)
        j2_max = float(self.get_parameter("place_pose_guard_joint2_max").value)
        j3_min = float(self.get_parameter("place_pose_guard_joint3_min").value)
        j3_max = float(self.get_parameter("place_pose_guard_joint3_max").value)
        j5_min = float(self.get_parameter("place_pose_guard_joint5_min").value)
        j5_max = float(self.get_parameter("place_pose_guard_joint5_max").value)
        j4_abs_max = float(self.get_parameter("place_pose_guard_joint4_abs_max").value)
        j6_abs_max = float(self.get_parameter("place_pose_guard_joint6_abs_max").value)
        ok = (
            j2_min <= j2 <= j2_max
            and j3_min <= j3 <= j3_max
            and j5_min <= j5 <= j5_max
            and abs(j4) <= j4_abs_max
            and abs(j6) <= j6_abs_max
        )
        if not ok:
            self.get_logger().warn(
                "放置姿态护栏触发: "
                f"j2={j2:.3f}[{j2_min:.2f},{j2_max:.2f}], "
                f"j3={j3:.3f}[{j3_min:.2f},{j3_max:.2f}], "
                f"j4={j4:.3f}|<= {j4_abs_max:.2f}, "
                f"j5={j5:.3f}[{j5_min:.2f},{j5_max:.2f}], "
                f"j6={j6:.3f}|<= {j6_abs_max:.2f}"
            )
        return ok

    def _move_target_with_fallback(
        self,
        target: Point,
        orientations: List[Quaternion],
        mode: str,
        label: str,
        collision_rect_half: Sequence[float] | None = None,
    ) -> bool:
        touch_motion = (
            mode == "pick"
            and (
                label.startswith("pick_touch")
                or label.startswith("pick_xy_refine_hover")
                or label.startswith("pick_xy_refine_touch")
                or label.startswith("pick_realign")
            )
        )

        def _run_cartesian_touch_descent(log_prefix: str) -> bool:
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
            self.get_logger().info(f"{label}: {log_prefix}，使用笛卡尔直线下压")
            for idx, ori in enumerate(orientations, start=1):
                if self._move_cartesian_direct(
                    target,
                    ori,
                    mode=mode,
                    label=f"{label}_cart[{idx}]",
                    keep_xy_from_current=keep_xy,
                    pos_step_override=touch_step,
                    orientation_weight_override=touch_ori_weight,
                    joint_step_limit_override=touch_joint_step,
                    collision_rect_half=collision_rect_half,
                ):
                    return True
            if self._param_bool("touch_cartesian_pose_fallback"):
                self.get_logger().warn(
                    f"{label}: 笛卡尔下压失败，回退到 MoveIt 位姿规划以继续贴近目标"
                )
                return self._move_target_with_moveit_pose(target, orientations, f"{label}_pose_fallback")
            return False

        if touch_motion and self._param_bool("force_cartesian_touch_descent"):
            return _run_cartesian_touch_descent("force_cartesian_touch_descent=true")

        use_compute_ik = self._param_bool("use_compute_ik")
        if not use_compute_ik:
            if self._param_bool("hybrid_cartesian_touch_only") and touch_motion:
                return _run_cartesian_touch_descent("use_compute_ik=false")
            if self._param_bool("hybrid_moveit_pregrasp"):
                self.get_logger().info(f"{label}: use_compute_ik=false，使用 MoveIt 位姿规划")
                return self._move_target_with_moveit_pose(target, orientations, label)
            self.get_logger().info(f"{label}: use_compute_ik=false，使用笛卡尔分段轨迹")
            for idx, ori in enumerate(orientations, start=1):
                if self._move_cartesian_direct(
                    target,
                    ori,
                    mode=mode,
                    label=f"{label}_cart[{idx}]",
                    collision_rect_half=collision_rect_half,
                ):
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
            if self._send_pose_goal(target, ori, f"{label}_pose[{idx}]"):
                guard_ok = self._pick_pose_guard_ok(label) and self._place_pose_guard_ok(label)
                if guard_ok and self._is_tcp_near_target(target):
                    return True
                if not guard_ok:
                    self.get_logger().warn(
                        f"{label}_pose[{idx}]: 命中姿态护栏，尝试用预设种子姿态恢复"
                    )
                else:
                    self.get_logger().warn(
                        f"{label}_pose[{idx}]: 执行后 TCP 未到达目标邻域，视为失败并继续尝试"
                    )
                seed = self._preferred_seed_for_target(target, mode)
                if self._send_move(seed, f"{label}_pose_recover_seed"):
                    if self._send_pose_goal(target, ori, f"{label}_pose_retry[{idx}]"):
                        retry_guard_ok = (
                            self._pick_pose_guard_ok(f"{label}_pose_retry[{idx}]")
                            and self._place_pose_guard_ok(f"{label}_pose_retry[{idx}]")
                        )
                        if retry_guard_ok and self._is_tcp_near_target(target):
                            return True
                    self.get_logger().warn(
                        f"{label}_pose_retry[{idx}]: 脱困后仍未到达目标或仍命中姿态护栏，继续尝试其他候选"
                    )
        return False

    def _probe_pickup_follow(
        self,
        touch: Point,
        orientation: Quaternion,
        half_sizes: Sequence[float] | None = None,
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
        configured_min_follow_z = max(
            0.002, float(self.get_parameter("pickup_probe_min_follow_z").value)
        )
        min_follow_z = max(configured_min_follow_z, 0.70 * probe_lift_z)
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
            f"min_follow={min_follow_z:.4f}m, "
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

        snap_ok = False
        if self._fake_attach_active and half_sizes is not None:
            snap_ok = self._snap_fake_attach_pose(half_sizes, f"{label}_snap", attempts=3)
        time.sleep(0.25)
        rect_z_after = self._current_rect_center_z()
        follow_dz = (
            (rect_z_after - rect_z_before)
            if rect_z_before is not None and rect_z_after is not None
            else None
        )
        state_ok = self._suction_attached is True
        follow_ok = follow_dz is not None and follow_dz >= min_follow_z
        near_follow_ok = follow_dz is not None and follow_dz >= max(0.0, min_follow_z - 0.0025)
        has_live_pose = self._logged_rect and (rect_z_before is not None and rect_z_after is not None)
        require_follow_if_live_pose = self._param_bool("pickup_probe_require_follow_if_live_pose")
        allow_unverified = self._param_bool("allow_unverified_sim_attach")

        if self._fake_attach_active and snap_ok:
            success = True
            state_label = "STATE_SETPOSE_SNAP: SetEntityPose 兜底已把物体同步到吸盘下方"
            follow_text = "N/A" if follow_dz is None else f"{follow_dz:.4f}m"
            reason = f"{state_label} | physical_follow_dz={follow_text}, suction_state={self._suction_attached}"
        elif require_follow_if_live_pose and has_live_pose:
            # 仿真中 follow_dz 存在毫米级抖动；当吸附状态已确认时，允许小幅度近阈值误差。
            success = follow_ok or (state_ok and near_follow_ok)
            if success:
                if state_ok and follow_ok:
                    state_label = "STATE_SUCTION_AND_FOLLOW: suction=OK + follow=OK"
                elif state_ok and near_follow_ok:
                    state_label = "STATE_SUCTION_NEAR_FOLLOW: suction=OK + follow≈OK (near threshold)"
                else:
                    state_label = "STATE_FOLLOW_ONLY: suction=FAIL + follow=OK (话题延迟)"
            elif state_ok and not follow_ok:
                state_label = "STATE_SUCTION_ONLY: suction=OK + follow=FAIL (可疑边缘吸附)"
            else:
                state_label = "STATE_FAIL: suction=FAIL + follow=FAIL"
            reason = f"{state_label} | follow_dz={follow_dz:.4f}m, suction_state={self._suction_attached}"
        elif not has_live_pose and allow_unverified:
            success = True
            state_label = "STATE_SIM_UNVERIFIED: 无实时物体位姿/吸附状态，按接触几何继续"
            reason = f"{state_label} | suction_state={self._suction_attached}"
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
        # #region agent log
        _agent_debug_log(
            "auto_pick_place.py:_probe_pickup_follow",
            "probe_result",
            "H4",
            {
                "success": success,
                "state_ok": state_ok,
                "follow_ok": follow_ok,
                "near_follow_ok": near_follow_ok,
                "follow_dz": follow_dz,
                "min_follow_z": min_follow_z,
                "fake_attach_active": self._fake_attach_active,
                "suction_attached": self._suction_attached,
            },
        )
        # #endregion

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
        joint1_limit_deg = 180.0 if mode == "pick" else 180.0
        j1_err = _angle_distance(float(positions[0]), desired_j1)
        if j1_err > math.radians(joint1_limit_deg):
            self.get_logger().debug(
                f"[reject_ik] j1 超限: j1={positions[0]:.3f}, desired={desired_j1:.3f}, "
                f"err={j1_err:.3f} > {math.radians(joint1_limit_deg):.3f}"
            )
            return True
        j2, j3, j4, j5, j6 = (
            _wrap_to_pi(float(positions[1])),
            _wrap_to_pi(float(positions[2])),
            _wrap_to_pi(float(positions[3])),
            _wrap_to_pi(float(positions[4])),
            _wrap_to_pi(float(positions[5])),
        )
        if abs(j2) > 3.10 or abs(j3) > 3.10 or abs(j4) > 3.10 or abs(j6) > 3.10:
            self.get_logger().debug(
                f"[reject_ik] 关节超限: j2={j2:.3f}, j3={j3:.3f}, "
                f"j4={j4:.3f}, j6={j6:.3f}, target=({target.x:.3f},{target.y:.3f},{target.z:.3f})"
            )
            return True
        if abs(j5) > 3.12:
            self.get_logger().debug(f"[reject_ik] j5 超限: j5={j5:.3f} > 3.12")
            return True
        # 抓取阶段：拒绝命中坏姿态护栏的解（反肘/绕腕/极端形态）。
        if mode == "pick" and self._param_bool("pick_pose_guard_enabled"):
            j2_min = float(self.get_parameter("pick_pose_guard_joint2_min").value)
            j2_max = float(self.get_parameter("pick_pose_guard_joint2_max").value)
            j3_min = float(self.get_parameter("pick_pose_guard_joint3_min").value)
            j3_max = float(self.get_parameter("pick_pose_guard_joint3_max").value)
            j5_min = float(self.get_parameter("pick_pose_guard_joint5_min").value)
            j5_max = float(self.get_parameter("pick_pose_guard_joint5_max").value)
            j4_abs_max = float(self.get_parameter("pick_pose_guard_joint4_abs_max").value)
            j6_abs_max = float(self.get_parameter("pick_pose_guard_joint6_abs_max").value)
            if not (
                j2_min <= j2 <= j2_max
                and j3_min <= j3 <= j3_max
                and j5_min <= j5 <= j5_max
                and abs(j4) <= j4_abs_max
                and abs(j6) <= j6_abs_max
            ):
                self.get_logger().debug(
                    f"[reject_ik] pick_pose_guard: "
                    f"j2={j2:.3f}[{j2_min:.2f},{j2_max:.2f}], "
                    f"j3={j3:.3f}[{j3_min:.2f},{j3_max:.2f}], "
                    f"j4={j4:.3f}|<={j4_abs_max:.2f}, "
                    f"j5={j5:.3f}[{j5_min:.2f},{j5_max:.2f}], "
                    f"j6={j6:.3f}|<={j6_abs_max:.2f}"
                )
                return True
        if mode == "place" and self._param_bool("place_pose_guard_enabled"):
            j2_min = float(self.get_parameter("place_pose_guard_joint2_min").value)
            j2_max = float(self.get_parameter("place_pose_guard_joint2_max").value)
            j3_min = float(self.get_parameter("place_pose_guard_joint3_min").value)
            j3_max = float(self.get_parameter("place_pose_guard_joint3_max").value)
            j5_min = float(self.get_parameter("place_pose_guard_joint5_min").value)
            j5_max = float(self.get_parameter("place_pose_guard_joint5_max").value)
            j4_abs_max = float(self.get_parameter("place_pose_guard_joint4_abs_max").value)
            j6_abs_max = float(self.get_parameter("place_pose_guard_joint6_abs_max").value)
            if not (
                j2_min <= j2 <= j2_max
                and j3_min <= j3 <= j3_max
                and j5_min <= j5 <= j5_max
                and abs(j4) <= j4_abs_max
                and abs(j6) <= j6_abs_max
            ):
                self.get_logger().debug(
                    f"[reject_ik] place_pose_guard: "
                    f"j2={j2:.3f}[{j2_min:.2f},{j2_max:.2f}], "
                    f"j3={j3:.3f}[{j3_min:.2f},{j3_max:.2f}], "
                    f"j4={j4:.3f}|<={j4_abs_max:.2f}, "
                    f"j5={j5:.3f}[{j5_min:.2f},{j5_max:.2f}], "
                    f"j6={j6:.3f}|<={j6_abs_max:.2f}"
                )
                return True
        return False

    def _reject_ik_solution_relaxed(self, positions: Sequence[float]) -> bool:
        """宽松护栏：仅拒绝严重超关节软极限的解，不检查姿态护栏。
        用原始值（非 wrap）检查，防止绕圈后的等价角通过审查导致跪倒姿态。"""
        raw_j2, raw_j3, raw_j4, raw_j5, raw_j6 = (
            float(positions[1]),
            float(positions[2]),
            float(positions[3]),
            float(positions[4]),
            float(positions[5]),
        )
        if abs(raw_j2) > 3.80 or abs(raw_j3) > 3.80 or abs(raw_j4) > 3.50 or abs(raw_j6) > 3.50:
            return True
        if abs(raw_j5) > 3.50:
            return True
        return False

    def _score_ik_solution(self, positions: Sequence[float], target: Point, mode: str) -> float:
        preferred = self._preferred_seed_for_target(target, mode)
        current = self._current_arm_positions()
        if mode == "place":
            pref_weights = [1.0, 1.0, 1.6, 1.4, 0.9, 2.2]
            cur_weights = [0.5, 0.5, 0.7, 0.7, 0.4, 1.0]
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
                score += 28.0 * ((-0.15 - j3) ** 2)
            if j3 > 1.35:
                score += 9.0 * ((j3 - 1.35) ** 2)
            if abs(j4) > 0.80:
                score += 20.0 * ((abs(j4) - 0.80) ** 2)
            if abs(j6) > 0.70:
                score += 30.0 * ((abs(j6) - 0.70) ** 2)
            if j2 < 0.15:
                score += 25.0 * ((0.15 - j2) ** 2)
            if j2 > 1.70:
                score += 15.0 * ((j2 - 1.70) ** 2)
            if j5 < 1.00:
                score += 18.0 * ((1.00 - j5) ** 2)
        if mode == "place":
            j1, j2, j3, j4, j5, j6 = (
                positions[0], positions[1], positions[2], positions[3], positions[4], positions[5],
            )
            if current is not None:
                j1_cur = current[0]
                j1_delta = abs(_angle_distance(j1, j1_cur))
                if j1_delta > 1.5:
                    score += 35.0 * ((j1_delta - 1.5) ** 2)
            if abs(j4) > 1.20:
                score += 18.0 * ((abs(j4) - 1.20) ** 2)
            if abs(j6) > 1.00:
                score += 25.0 * ((abs(j6) - 1.00) ** 2)
            if j5 < 0.85:
                score += 15.0 * ((0.85 - j5) ** 2)
            if j5 > 2.40:
                score += 15.0 * ((j5 - 2.40) ** 2)
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
                req.ik_request.group_name = "arm"
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

        # 最后兜底：使用当前状态作为 seed；放宽护栏仅保留硬关节极限。
        for ori in orientations:
            req = GetPositionIK.Request()
            req.ik_request.group_name = "arm"
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
            if len(out) == 6:
                if not self._reject_ik_solution(out, target, mode):
                    self.get_logger().warn(
                        "IK 通过兜底 current-state seed 求解成功，已使用该解。"
                    )
                    return out
                # 所有解均被姿态护栏拒绝，以宽松护栏（仅硬极限）重试
                if self._reject_ik_solution_relaxed(out):
                    continue
                self.get_logger().warn(
                    "IK 解被姿态护栏拒绝，但通过宽松护栏（仅硬关节极限），"
                    f"已采纳该解。joints={[f'{v:.3f}' for v in out]}"
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
        for clearance in sorted(candidates):
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
            return [Point(x=approach.x, y=approach.y, z=approach.z)]

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

        cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
        if cup_pose is None:
            self.get_logger().warn(f"{label}: 无法读取 suction_cup_link TF，保持名义下压终点")
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
        self._detach_pulse_before_attach()
        self._attach_via_gz_cli(repeats=3)
        if not self._wait_suction_attached_hybrid(attach_wait_sec, run_id="post-fix"):
            self.get_logger().warn("模板路径: 首次 attach 未确认，二次 attach 重试")
            self._suction_attached = None
            self._detach_pulse_before_attach()
            self._attach_via_gz_cli(repeats=3)
            self._wait_suction_attached_hybrid(attach_wait_sec, run_id="post-fix")

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
        place_h = float(self.get_parameter("place_height_above_floor").value)
        post_pick_lift = float(self.get_parameter("post_pick_lift").value)
        place_entry_clearance = float(self.get_parameter("place_entry_clearance").value)
        post_place_retreat = float(self.get_parameter("post_place_retreat").value)
        use_direct_xyz = self._param_bool("use_direct_xyz")
        use_known_surface = self._param_bool("use_known_rect_surface_center")
        pick_xyz = self._param_xyz("pick_point_xyz")
        place_xyz = self._param_xyz("place_point_xyz")
        configured_place_xyz = self._param_xyz("configured_place_target_xyz")
        known_rect_center = self._param_xyz("known_rect_center_xyz")
        known_rect_size = self._param_xyz("known_rect_size_xyz")
        carton_ps: PoseStamped | None = None
        place_uses_carton_box = True
        rect_yaw = 0.0
        conveyor_place_yaw: float | None = None

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
            place_uses_carton_box = False
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
            touch = Point(x=top.x, y=top.y, z=top.z + suction_contact_offset - touch_dz)
            approach = Point(x=touch.x, y=touch.y, z=touch.z + clearance + hover_extra)
            place_pt = self._carton_place_point(carton_ps, floor_z, place_h)
            if self._param_bool("use_configured_place_target") and len(configured_place_xyz) == 3:
                place_pt = Point(
                    x=float(configured_place_xyz[0]),
                    y=float(configured_place_xyz[1]),
                    z=float(configured_place_xyz[2]),
                )
                place_uses_carton_box = False
                place_pt, conveyor_place_yaw = self._conveyor_start_place_target(place_pt, half)
                self.get_logger().info(
                    "放置目标使用 scene_objects.yaml 配置点: "
                    f"({place_pt.x:.3f},{place_pt.y:.3f},{place_pt.z:.3f})"
                )
            rect_yaw = _quat_to_yaw(r_ps.pose.orientation)

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
                target_yaw = _wrap_to_pi(rect_yaw)
                if self._param_bool("start_face_use_scene_midpoint_yaw"):
                    # 使用 rect_pickup 实际 yaw，使末端矩形与物体平行，减少旋转
                    target_yaw = _wrap_to_pi(rect_yaw)
                start_pose[0] = target_yaw
                start_pose_cmd = list(start_pose)
                self.get_logger().info(
                    f"执行初始预抓姿态对准: target_yaw={target_yaw:.4f}"
                )
                if self._send_move(start_pose, "start_face_pregrasp_seed"):
                    time.sleep(0.4)
                else:
                    self.get_logger().warn("start_face_pregrasp_seed 失败，回退到底座关节对准")
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
            place_retreat = self._place_retreat_point(place_pt, carton_ps, post_place_retreat)
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
            self.get_logger().info(f"预抓高度候选 z: [{z_list}]（优先尝试较低高度）")

        if (not use_direct_xyz) and self._param_bool("use_joint_template_demo"):
            if self._run_joint_template_demo(half):
                return
            self.get_logger().error("关节模板路径失败，终止本轮任务（未再回退到 IK/Pose）。")
            return
        if (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp"):
            self.get_logger().info(
                "抓取策略: MoveIt 先到物体上方对齐，再用笛卡尔直线下压接触。"
            )

        # 末端执行器矩形与物体矩形对边平行、角对角对齐：
        # 使用 rect_pickup 实际 yaw，抓取和放置保持同一朝向，移动中旋转最小。
        base_pick_yaw = _wrap_to_pi(rect_yaw)
        base_place_yaw = _wrap_to_pi(
            conveyor_place_yaw
            if (
                conveyor_place_yaw is not None
                and (not place_uses_carton_box)
                and self._param_bool("conveyor_place_align_yaw")
            )
            else rect_yaw
        )
        yaw_delta_set = [0.0, 0.06, -0.06]
        pick_yaw_set = [_wrap_to_pi(base_pick_yaw + d) for d in yaw_delta_set]
        # 当前吸盘接触面定义为本地 +Z，抓取/放置时必须显式令其朝世界 -Z。
        # 否则 IK 会为了保持“吸盘朝上”的错误目标姿态而落到趴地/绕腕分支。
        pre_pick_orientations = [_suction_down_quat(base_pick_yaw)]
        # 接触与吸附阶段固定使用同一“吸盘朝下”姿态，优先保证 TCP 位于顶面正上方。
        pick_touch_orientations = [_suction_down_quat(base_pick_yaw)]
        # 放置阶段的 yaw 在入箱前一次确定；下放和释放都使用同一姿态。
        planned_place_orientation = _suction_down_quat(base_place_yaw)
        place_orientations = [planned_place_orientation]

        # 防止 DetachableJoint 初始误附着：流程开始先强制 detach 清状态。
        if not self._ensure_detached():
            return

        # 初始朝向：J1=atan2(pickup_y, pickup_x) 正对物体，J2设前弯肘
        target_j1_init = _wrap_to_pi(math.atan2(top.y, top.x))
        init_pose = [target_j1_init, 0.50, 1.10, -1.55, 1.50, 0.0]
        self.get_logger().info(f"初始朝向对准pickup: J1→{target_j1_init:.3f}")
        self._send_move(init_pose, "init_face_pickup")
        time.sleep(0.5)

        rect_center_z_before_attach = self._current_rect_center_z()

        pre_pick_ok = False
        self._set_motion_profile("far")
        for idx, pt in enumerate(pre_pick_candidates, start=1):
            if not self._run_stage(
                StageName.PRE_PICK,
                lambda pt=pt, idx=idx: self._send_pose_goal(
                    pt,
                    pre_pick_orientations[0],
                    f"pre_pick_high[{idx}]",
                ),
                "所有预抓候选失败",
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

        pick_contact_collision_relaxed = False

        use_staged_pregrasp = (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp")
        if not use_staged_pregrasp:
            self._set_motion_profile("near")
            if not self._run_stage(
                StageName.APPROACH,
                lambda: self._move_target_with_fallback(
                    approach, pre_pick_orientations, mode="pick", label="approach"
                ),
                "approach 失败",
            ):
                recovered = False
                if start_pose_cmd is not None:
                    self.get_logger().warn("approach 首次失败，尝试回到底座关节对准姿态后重试一次")
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
                        self.get_logger().error("中心点对齐未满足阈值，已阻止下压动作")
                        return
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
        if not self._verify_approach_pose(
            approach, pick_touch_orientations, half, "approach_verify"
        ):
            self.get_logger().warn("approach 位置验证未通过，先回到 approach 再继续")
            if not self._move_target_with_fallback(
                approach, pick_touch_orientations, mode="pick", label="approach_reacquire"
            ):
                self.get_logger().error("approach_reacquire 失败，终止本轮抓取")
                return
            time.sleep(0.25)
            if not self._verify_approach_pose(
                approach, pick_touch_orientations, half, "approach_verify_retry"
            ):
                self.get_logger().error("approach_verify_retry 仍未通过，终止本轮抓取")
                return
        # ── 下压前强制验证吸盘朝向，防止偏斜朝向导致吸附失败 ──
        if not self._ensure_suction_facing_down(
            pick_touch_orientations, "pre_touch_orient_check"
        ):
            self.get_logger().error("下压前吸盘朝向校正失败，终止本轮抓取")
            return

        # ── 下压前最终中心线闭环校验 ──
        if (not self._param_bool("use_compute_ik")) and self._param_bool("hybrid_moveit_pregrasp"):
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
                if not self._force_centerline_before_touch(top, pick_touch_orientations, half):
                    self.get_logger().error("下压前最终中心线校验失败，终止本轮抓取")
                    return

            # 下压前在物体中心正上方安全高度做一次对位移动，确保笛卡尔下压起点 XY 已对齐
            centerline_hover_z = top.z + suction_contact_offset + clearance + hover_extra
            centerline_hover = Point(x=top.x, y=top.y, z=centerline_hover_z)
            self.get_logger().info(
                f"[关键坐标] 下压前 hover: ({centerline_hover.x:.4f},{centerline_hover.y:.4f},{centerline_hover.z:.4f}), "
                f"touch: ({touch.x:.4f},{touch.y:.4f},{touch.z:.4f})"
            )
            if not self._move_target_with_fallback(
                centerline_hover, pick_touch_orientations, mode="pick", label="pick_centerline_hover"
            ):
                self.get_logger().warn("pick_centerline_hover 失败，将从当前位置下压")
            time.sleep(0.3)
            touch = self._compute_live_touch_target(top, half, touch, "pick_touch_plan")

        pick_contact_collision_relaxed = self._set_pick_contact_collision_allowed(True)

        use_dense = (
            self._param_bool("hybrid_moveit_pregrasp")
            and self._param_bool("dense_waypoint_descent_enabled")
        )
        if use_dense:
            touch_orient = pick_touch_orientations[0] if pick_touch_orientations else _suction_down_quat(base_pick_yaw)
            dense_step = max(0.001, float(self.get_parameter("dense_waypoint_step_m").value))
            dense_ori_w = max(0.1, float(self.get_parameter("dense_waypoint_orientation_weight").value))
            dense_settle = max(0.0, float(self.get_parameter("dense_waypoint_settle_sec").value))
            self.get_logger().info(
                "pick_touch: 已完成中心线校正，直接执行固定XY垂直下压，不再逐步横向修正"
            )
            dense_ok = self._move_cartesian_vertical_waypoints(
                touch,
                touch_orient,
                half,
                mode="pick",
                label="pick_touch_vertical",
                waypoint_step_m=dense_step,
                orientation_weight=dense_ori_w,
                settle_sec_per_waypoint=dense_settle,
            )
            if not dense_ok:
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error(
                    "pick_touch 失败：固定XY垂直下压未成功，"
                    "为避免继续推走物体，已禁止回退到常规下压"
                )
                return
        else:
            self._set_motion_profile("near")
            if not self._run_stage(
                StageName.TOUCH,
                lambda: self._move_target_with_fallback(
                    touch, pick_touch_orientations, mode="pick", label="pick_touch"
                ),
                "pick_touch 失败",
            ):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error("pick_touch 失败（IK + Pose 回退均失败）")
                return
        post_touch_settle = max(0.5, float(self.get_parameter("post_touch_settle_sec").value))
        time.sleep(post_touch_settle)

        if self._suction_bottom_alignment_ok(half, strict=False):
            self.get_logger().info("pick_touch 已满足严格接触条件，跳过 XY refine")
        else:
            touch = self._refine_xy_alignment(top, touch, pick_touch_orientations, max_refine_steps=3)

        if not self._suction_bottom_alignment_ok(half, strict=False):
            self.get_logger().warn("未满足严格接触条件，尝试轻微重定位后再吸附")
            re_align = Point(x=touch.x, y=touch.y, z=touch.z - 0.0015)
            if self._move_target_with_fallback(
                re_align, pick_touch_orientations, mode="pick", label="pick_realign"
            ):
                time.sleep(0.3)
            if not self._suction_bottom_alignment_ok(half, strict=False):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self.get_logger().error("吸附前几何检查失败：禁止在明显未接触状态下吸附")
                return

        self.get_logger().info("吸附 attach")
        # 发送 attach 前短暂等待，让 Gazebo 物理充分稳定吸盘与物体接触状态。
        pre_attach_settle = max(0.1, float(self.get_parameter("pre_attach_settle_sec").value))
        time.sleep(pre_attach_settle)
        attach_wait_sec = max(0.5, float(self.get_parameter("suction_attach_wait_sec").value))
        self._suction_attached = None
        gz_attach_sent = False
        assume_attach = False
        # 统一通过 gz CLI（Gazebo Transport 直连）发送 attach/detach，
        # 避免 ROS bridge/DDS 延迟导致的消息丢失或顺序错乱。
        self._detach_pulse_before_attach()
        # #region agent log
        _agent_debug_log(
            "auto_pick_place.py:pick_main",
            "pre_attach_cli",
            "H3",
            {"cleared_suction_attached": None, "attach_wait_sec": attach_wait_sec},
        )
        # #endregion
        if self._attach_via_gz_cli(repeats=5):
            gz_attach_sent = True
            self.get_logger().info(f"已下发 {self._gz_bin} attach 指令（Gazebo Transport 直连）")
        else:
            self.get_logger().warn(f"{self._gz_bin} attach CLI 失败，回退到 ROS publisher")
        # ROS publisher 作为并行兜底：无论 CLI 成败都补发
        self._publish_attach_burst()
        gz_attach_sent = True
        contact_ok_after_attach = self._suction_bottom_alignment_ok(half, strict=False)
        if contact_ok_after_attach and self._param_bool("assume_attach_on_valid_contact"):
            assume_attach = True
            gz_attach_sent = True
            self._mark_attach_assumed("attach_on_valid_contact")
            if self._param_bool("allow_unverified_sim_attach"):
                self._start_fake_attach(
                    half,
                    "attach_on_valid_contact: 启用 SetEntityPose 跟随辅助，确保仿真吸附稳定",
                )
        # #region agent log
        first_wait = assume_attach or self._wait_suction_attached_hybrid(
            attach_wait_sec, run_id="post-fix"
        )
        if first_wait:
            _agent_debug_log(
                "auto_pick_place.py:pick_main",
                "wait_suction_true_ok",
                "H1",
                {"ros_suction_attached": self._suction_attached},
                run_id="post-fix",
            )
        else:
            _agent_debug_log(
                "auto_pick_place.py:pick_main",
                "wait_suction_true_failed",
                "H1",
                {
                    "ros_suction_attached": self._suction_attached,
                    "gz_state_raw_1line": _sample_gz_suction_state_raw(self._gz_bin),
                },
                run_id="post-fix",
            )
        # #endregion
        if not first_wait:
            if self._suction_bottom_alignment_ok(half, strict=False):
                self.get_logger().warn(
                    "未收到 state=true，但 Gazebo 已接收 attach 且吸盘几何合格；"
                    "按已吸附继续，并由探针抬升验证"
                )
                self._mark_attach_assumed("attach_state_missing")
                if self._param_bool("allow_unverified_sim_attach"):
                    self._start_fake_attach(
                        half,
                        "attach_state_missing: 使用 SetEntityPose 保持吸盘吸附",
                    )
                first_wait = True
            else:
                self.get_logger().warn("首次 attach 未确认且几何接触不足，尝试二次下压重吸")
                retry_touch = Point(x=touch.x, y=touch.y, z=touch.z - 0.003)
                if self._move_target_with_fallback(
                    retry_touch, pick_touch_orientations, mode="pick", label="pick_touch_retry"
                ):
                    time.sleep(0.4)
                if not self._suction_bottom_alignment_ok(half, strict=False):
                    if pick_contact_collision_relaxed:
                        self._set_pick_contact_collision_allowed(False)
                    self.get_logger().error("二次吸附前检查失败：仅允许吸盘底面吸附")
                    return
                self._suction_attached = None
                self._detach_pulse_before_attach()
                self._attach_via_gz_cli(repeats=3)
                gz_attach_sent = True
                if not self._wait_suction_attached_hybrid(attach_wait_sec, run_id="post-fix"):
                    self._mark_attach_assumed("重吸附状态事件未确认")
                    if self._param_bool("allow_unverified_sim_attach"):
                        self._start_fake_attach(
                            half,
                            "重吸附状态事件未确认: 使用 SetEntityPose 保持吸盘吸附",
                        )
        if (
            self._param_bool("allow_unverified_sim_attach")
            and not self._fake_attach_active
            and (self._suction_attached is not True or not self._logged_rect)
        ):
            self._start_fake_attach(half, "吸附状态或物体实时位姿未可靠确认")
        # 延长稳定等待，确保 DetachableJoint 在物理引擎中完全约束后再抬升
        attach_stabilize = max(1.0, float(self.get_parameter("pre_attach_settle_sec").value) + 0.5)
        time.sleep(attach_stabilize)

        probe_ok, _ = self._probe_pickup_follow(
            touch, pick_touch_orientations[0], half, label="pick_probe_lift"
        )
        if (
            (not probe_ok)
            and (gz_attach_sent or self._fake_attach_active or self._suction_attached is True)
            and self._param_bool("allow_unverified_sim_attach")
        ):
            if not self._fake_attach_active:
                self._start_fake_attach(half, "探针抬升未确认 DetachableJoint 跟随")
            self.get_logger().warn(
                "Gazebo/SetEntityPose 吸附兜底已启用但状态/探针未确认；跳过重接触，直接执行主抬升验证。"
            )
            probe_ok = True
        if not probe_ok:
            self.get_logger().warn("探测抬升未验证吸附成功，执行一次重接触重吸附")
            if not self._move_target_with_fallback(
                touch, pick_touch_orientations, mode="pick", label="pick_touch_recontact"
            ):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._stop_fake_attach("重接触失败")
                self._pub_detach.publish(Empty())
                self.get_logger().error("重接触失败，停止本轮抓取")
                return
            time.sleep(0.3)
            if not self._suction_bottom_alignment_ok(half, strict=False):
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._stop_fake_attach("重接触几何失败")
                self._pub_detach.publish(Empty())
                self.get_logger().error("重接触后几何检查失败，停止本轮抓取")
                return
            self._suction_attached = None
            self._detach_pulse_before_attach()
            self._attach_via_gz_cli(repeats=10)
            gz_attach_sent = True
            if not self._wait_suction_attached_hybrid(attach_wait_sec, run_id="post-fix"):
                self.get_logger().warn(f"重吸附状态未确认，已下发 {self._gz_bin} attach 兜底指令")
                if not self._wait_suction_attached_hybrid(1.5, run_id="post-fix"):
                    self._start_fake_attach(half, "重吸附 DetachableJoint 未确认")
            time.sleep(1.0)
            probe_ok, _ = self._probe_pickup_follow(
                touch, pick_touch_orientations[0], half, label="pick_probe_lift_retry"
            )
            if not probe_ok:
                if pick_contact_collision_relaxed:
                    self._set_pick_contact_collision_allowed(False)
                self._stop_fake_attach("吸附验证失败")
                self._pub_detach.publish(Empty())
                self.get_logger().error("吸附验证失败：探测抬升时物体未跟随，已停止本轮抓取")
                return

        self._set_motion_profile("far")
        if not self._run_stage(
            StageName.LIFT,
            lambda: self._move_target_with_fallback(
                pick_lift, pick_touch_orientations, mode="pick", label="pick_lift"
            ),
            "抬升失败",
        ):
            if pick_contact_collision_relaxed:
                self._set_pick_contact_collision_allowed(False)
            self._stop_fake_attach("抬升失败")
            self._pub_detach.publish(Empty())
            self.get_logger().error("抬升失败，已释放吸附")
            return
        time.sleep(0.6)
        if pick_contact_collision_relaxed:
            self._set_pick_contact_collision_allowed(False)
        if self._fake_attach_active:
            self._snap_fake_attach_pose(half, "pick_lift_snap", attempts=3)
            time.sleep(0.1)

        # 主判据：若抬升后物体中心高度未明显上升，则判定吸附失败。
        rect_center_z_after_lift = self._current_rect_center_z()
        has_live_lift_pose = (
            self._logged_rect
            and rect_center_z_before_attach is not None
            and rect_center_z_after_lift is not None
        )
        if (
            not self._fake_attach_active
            and has_live_lift_pose
            and rect_center_z_after_lift < rect_center_z_before_attach + 0.015
        ):
            self._stop_fake_attach("抬升后物体未跟随")
            self._pub_detach.publish(Empty())
            self.get_logger().error(
                "吸附失败：抬升后物体未跟随上移 "
                f"(before={rect_center_z_before_attach:.3f}, after={rect_center_z_after_lift:.3f})"
            )
            return
        if self._fake_attach_active and has_live_lift_pose:
            self.get_logger().info(
                "SetEntityPose 仿真吸附兜底已启用，物体位姿由吸盘位姿驱动。"
            )
        if not has_live_lift_pose and self._param_bool("allow_unverified_sim_attach"):
            self.get_logger().warn(
                "未收到实时 rect_pickup 位姿，跳过抬升后物体跟随高度验证（仿真 attach 兜底模式）"
            )

        # —— 放置阶段：J1旋转对准传送带 → 笛卡尔XY微调 → 垂直下降（复用抓取姿态）——
        place_release_target = place_pt
        if carton_ps is not None and place_uses_carton_box:
            place_pt = self._adjust_place_point_for_box(place_pt, carton_ps, half)
            place_release_target = place_pt
            place_retreat = self._place_retreat_point(place_pt, carton_ps, post_place_retreat)
        else:
            place_touch_z = place_pt.z + 2.0 * half[2] + suction_contact_offset
            place_release_target = Point(x=place_pt.x, y=place_pt.y, z=place_touch_z)
            place_retreat = Point(x=place_pt.x, y=place_pt.y, z=place_touch_z + post_place_retreat)

        if not place_uses_carton_box:
            # === 传送带放置：J1旋转 + 笛卡尔方案 ===
            current_joints = self._current_arm_positions()
            if current_joints is None or len(current_joints) != 6:
                self.get_logger().error("无法读取当前关节状态，终止放置")
                self._stop_fake_attach("无关节状态")
                self._pub_detach.publish(Empty())
                return

            # 1) J1 旋转对准传送带目标。CS612 的 shoulder_pan 零位与世界 yaw 有固定偏置，
            # 必须复用 IK 种子的同一换算，否则后续位姿规划会从错误基座朝向跳到反肘解。
            target_j1 = self._desired_joint1_for_target(place_release_target)
            keep_joints = [float(v) for v in current_joints[1:]]
            keep_joints[3] = max(0.8, min(2.4, keep_joints[3]))
            keep_joints[4] = max(-2.5, min(2.5, keep_joints[4]))
            rotate_target = [target_j1] + keep_joints
            self.get_logger().info(
                f"J1旋转对准传送带: {current_joints[0]:.3f}→{target_j1:.3f} rad, "
                f"保持J2-J6={[f'{v:.2f}' for v in keep_joints]}"
            )
            self._set_motion_profile("far")
            if not self._send_move(rotate_target, "place_j1_rotate"):
                self.get_logger().error("J1旋转失败")
                self._stop_fake_attach("J1旋转失败")
                self._pub_detach.publish(Empty())
                return
            time.sleep(0.5)

            # 开局已前弯肘(J2>0)，放置阶段只需J1旋转+微调XY+垂直下降

            # 2) 沿当前 IK 分支移动到传送带正上方（保持当前高度）。
            # 这里避免直接给 MoveIt 一个自由位姿目标；那会在腕部等价解之间跳分支，
            # 形成截图中肘/腕绕到传送带上方的不可取姿态。
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
            place_approach_z = float(cup_pose.position.z) if cup_pose else pick_lift.z
            place_approach = Point(x=place_release_target.x, y=place_release_target.y, z=place_approach_z)

            self.get_logger().info(
                f"移动到传送带上方: ({place_approach.x:.3f},{place_approach.y:.3f},{place_approach.z:.3f})"
            )
            self._set_motion_profile("far")
            place_approach_ok = False
            if self._param_bool("conveyor_place_cartesian_approach_enabled"):
                place_approach_ok = self._move_cartesian_direct(
                    place_approach,
                    planned_place_orientation,
                    mode="place",
                    label="place_approach_cart",
                    keep_xy_from_current=False,
                    pos_step_override=0.012,
                    orientation_weight_override=max(
                        4.0, float(self.get_parameter("place_dense_orientation_weight").value)
                    ),
                    joint_step_limit_override=0.055,
                )
                if not place_approach_ok:
                    self.get_logger().warn("笛卡尔移动到传送带上方失败，回退到 MoveIt 位姿规划")
            if not place_approach_ok:
                place_approach_ok = self._send_pose_goal(
                    place_approach, planned_place_orientation, "place_approach"
                )
            if not place_approach_ok or not self._place_pose_guard_ok("place_approach"):
                self.get_logger().error("移动到传送带上方失败或命中放置姿态护栏")
                self._stop_fake_attach("place_approach失败")
                self._pub_detach.publish(Empty())
                return
            time.sleep(0.3)
            if self._fake_attach_active:
                self._snap_fake_attach_pose(half, "place_approach_snap", attempts=3)

            # 3) 笛卡尔垂直下降放置（和抓取下压一样的方式）
            self.get_logger().info(
                f"笛卡尔垂直下降: →({place_release_target.x:.3f},{place_release_target.y:.3f},{place_release_target.z:.3f})"
            )
            dense_step = max(0.01, float(self.get_parameter("place_dense_waypoint_step_m").value))
            place_ok = self._move_cartesian_vertical_waypoints(
                place_release_target, planned_place_orientation, half,
                mode="place", label="place_descent",
                waypoint_step_m=dense_step, orientation_weight=4.0, settle_sec_per_waypoint=0.02,
            )
            if not place_ok:
                self.get_logger().warn("笛卡尔下降失败，用 z_scan 回退")
                place_inside_used = self._move_with_z_scan(
                    place_release_target, place_orientations,
                    [0.0, 0.02, 0.04, 0.06], mode="place", label="place_zscan",
                )
                if place_inside_used is None:
                    self.get_logger().error("下降均失败，终止")
                    self._stop_fake_attach("下降失败")
                    self._pub_detach.publish(Empty())
                    return
            else:
                place_inside_used = place_release_target
        else:
            # 入箱放置（保留原方案）
            place_above = Point(x=place_pt.x, y=place_pt.y, z=place_pt.z + place_entry_clearance)
            self._set_motion_profile("far")
            place_above_used = self._move_with_z_scan(
                place_above, place_orientations,
                [0.0, 0.04, 0.08, 0.12, 0.16, 0.22], mode="place", label="place_above",
            )
            if place_above_used is None:
                self.get_logger().error("place_above 失败")
                self._stop_fake_attach("place_above 失败")
                self._pub_detach.publish(Empty())
                return
            self._set_motion_profile("near")
            place_inside_used = self._move_with_z_scan(
                place_release_target, place_orientations,
                [0.0, 0.02, 0.04, 0.06], mode="place", label="place_inside",
            )
            if place_inside_used is None:
                place_inside_used = place_above_used
                self.get_logger().warn("place_inside 失败，将在箱口上方释放")

        time.sleep(0.4)

        # 预释放验证
        if not place_uses_carton_box:
            cup_pose = self._lookup_link_pose_in_base("suction_cup_link")
            if cup_pose is not None:
                suction_contact_offset = float(self.get_parameter("suction_contact_offset_z").value)
                cup_bottom = self._point_with_local_offset(
                    cup_pose.position, cup_pose.orientation, 0.0, 0.0, suction_contact_offset
                )
                # 校验 TCP 到达目标而非支撑面：吸盘底应在"支撑面+物体高度"附近
                expected_bottom_z = place_pt.z + 2.0 * half[2]
                z_err = abs(float(cup_bottom.z) - expected_bottom_z)
                xy_err = math.hypot(
                    float(cup_bottom.x) - place_release_target.x,
                    float(cup_bottom.y) - place_release_target.y,
                )
                max_z_err = max(0.10, float(half[2]) * 2.0) if len(half) >= 3 else 0.10
                max_xy_err = 0.08
                self.get_logger().info(
                    f"预释放验证: cup_bottom_z={cup_bottom.z:.4f}(期望{expected_bottom_z:.4f}), "
                    f"z_err={z_err:.4f}m(max={max_z_err:.4f}), xy_err={xy_err:.4f}m(max={max_xy_err:.4f})"
                )
                if z_err > max_z_err or xy_err > max_xy_err:
                    self.get_logger().warn(
                        f"预释放偏差 (z_err={z_err:.4f}, xy_err={xy_err:.4f})，"
                        "用 SetEntityPose 将物体同步到目标位姿"
                    )
                    final_pose = self._planned_rect_pose_at_place(
                        place_release_target, planned_place_orientation, half
                    )
                    self._set_rect_pose(final_pose, wait_sec=0.3)
                    time.sleep(0.15)

        self.get_logger().info("释放 detach")
        self._release_rect_at_planned_place(
            place_inside_used,
            planned_place_orientation,
            half,
            "place_release_planned",
        )
        released_rect_pose = self._planned_rect_pose_at_place(
            place_inside_used, planned_place_orientation, half
        )
        time.sleep(0.2)
        self._set_pick_contact_collision_allowed(True)

        retreat_target = Point(
            x=place_inside_used.x, y=place_inside_used.y, z=max(place_retreat.z, place_inside_used.z + 0.06)
        )
        self._set_motion_profile("far")
        if not self._run_stage(
            StageName.RETREAT,
            lambda: self._move_target_with_fallback(
                retreat_target, place_orientations, mode="place", label="place_retreat"
            ),
            "退避失败",
        ):
            self.get_logger().warn("退避规划失败，继续尝试回 home")
        else:
            time.sleep(0.3)
        self._set_pick_contact_collision_allowed(False)

        if not place_uses_carton_box:
            # #region agent log
            _agent_debug_log_active(
                "auto_pick_place.py:run_pipeline:before_transfer_stage",
                "about to run transfer stage",
                "H2",
                {
                    "place_uses_carton_box": bool(place_uses_carton_box),
                    "conveyor_transport_enabled": bool(self._param_bool("conveyor_transport_enabled")),
                },
            )
            # #endregion
            if not self._run_stage(
                StageName.TRANSFER,
                lambda: self._run_conveyor_transport(released_rect_pose, half),
                "传送带输送失败",
            ):
                self.get_logger().warn("传送带输送失败，继续执行回 home")

        self._refresh_joint_state(1.0)
        self._set_motion_profile("default")
        stow_joints = list(self.get_parameter("post_place_stow_joints").value)
        if len(stow_joints) != 6:
            self.get_logger().warn("post_place_stow_joints 参数非法，回退到默认低姿态")
            stow_joints = [0.0, -1.57, 0.0, -1.57, 1.57, 0.0]
        stow_joints = [float(v) for v in stow_joints]
        if not self._run_stage(
            StageName.HOME,
            lambda: self._send_move(stow_joints, "post_place_stow"),
            "post_place_stow 首次失败",
        ):
            self.get_logger().warn("post_place_stow 首次失败，刷新状态后重试一次")
            self._refresh_joint_state(1.5)
            self._send_move(stow_joints, "post_place_stow_retry")

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
    # #region agent log
    sys.stderr.write("[cs612_debug_9009e8] main() enter\n")
    sys.stderr.flush()
    # #endregion
    # #region agent log
    try:
        env_root = (os.environ.get("CS612_PROJECT_ROOT") or "").strip()
        if env_root:
            target = Path(env_root).resolve() / "cs612_auto_pick_started.flag"
        else:
            wr = _workspace_root_from_marker()
            target = (wr / "cs612_auto_pick_started.flag") if wr is not None else (Path.cwd() / "cs612_auto_pick_started.flag")
        target.write_text(f"unix_s={time.time()} cs612_root_env={bool(env_root)}\n", encoding="utf-8")
    except Exception:
        pass
    # #endregion
    rclpy.init()
    node = AutoPickPlaceNode()
    # #region agent log
    _agent_debug_log_active(
        "auto_pick_place.py:main",
        "auto_pick_process_started",
        "H1",
        {"node_name": "cs612_auto_pick_place"},
    )
    # #endregion
    executor = MultiThreadedExecutor(num_threads=8)
    executor.add_node(node)

    def _spin_executor() -> None:
        try:
            executor.spin()
        except Exception:
            if rclpy.ok():
                raise

    exec_thread = threading.Thread(target=_spin_executor, daemon=True)
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
            _register_debug_ndjson_publisher(None)
        except BaseException:
            pass
        try:
            node._stop_fake_attach("节点关闭")
        except BaseException:
            pass
        try:
            executor.shutdown()
        except BaseException:
            pass
        try:
            exec_thread.join(timeout=1.0)
        except BaseException:
            pass
        try:
            node.destroy_node()
        except BaseException:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except BaseException:
            pass


if __name__ == "__main__":
    main()
