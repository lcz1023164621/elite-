#!/usr/bin/env python3
"""将 Gazebo 关节状态规范为 URDF 关节名并发布到 /joint_states。

优先订阅 ros_gz_bridge 转发的 ``/joint_states_gz``。
若在设定时间内仍未收到真实关节状态，则自动回退到原生 ``ign topic`` 读取
``/world/arm_world/model/CS612_arm/joint_state``，绕开 Jazzy/Jetty 下
``ignition.msgs.Model -> sensor_msgs/JointState`` 桥接偶发断流的问题。
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState

# 必须与 robot_state_publisher 对 /joint_states 的订阅 QoS 一致（Humble 下为 SensorData/BEST_EFFORT）。
# 使用 RELIABLE 发布时，部分环境下与 RSP 的 BEST_EFFORT 订阅匹配失败 → 整臂 TF 缺失、RViz 全红。
# depth 过小在 Gazebo Model 关节状态较大时易触发 DDS “sequence size exceeds remaining buffer”。
_JOINT_STATES_PUBLISHER_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    # 采用 RELIABLE，兼容 RSP 可能的默认可靠订阅配置，避免“整链 TF 缺失”。
    # RELIABLE 发布可被 BEST_EFFORT 订阅端接收；反向不成立。
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)

_JOINT_STATES_SUBSCRIBER_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=50,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)

# Gazebo SDF 中的关节名（内部）→ URDF 标准关节名（ROS 侧）
_GZ_TO_URDF = {
    "joint1": "shoulder_pan_joint",
    "joint2": "shoulder_lift_joint",
    "joint3": "elbow_joint",
    "joint4": "wrist_1_joint",
    "joint5": "wrist_2_joint",
    "joint6": "wrist_3_joint",
}

# URDF 中可动关节名（顺序固定，便于 robot_state_publisher）
_ARM = tuple(_GZ_TO_URDF.values())
_HOME_POSITIONS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 0.0,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0,
}


def _strip_scope(name: str) -> str:
    if "::" in name:
        return name.split("::")[-1]
    return name


def _canonical_joint_name(raw: str) -> str | None:
    """将 Gazebo / 桥接可能输出的 joint1、Joint_2、model::joint3 等映射到 URDF 中的标准关节名。"""
    key = _strip_scope(raw)
    if key in _ARM:
        return key
    if key in _GZ_TO_URDF:
        return _GZ_TO_URDF[key]
    compact = key.replace("_", "").lower()
    if compact.startswith("joint") and len(compact) > 5:
        suf = compact[5:]
        if suf.isdigit():
            cand = f"joint{int(suf)}"
            if cand in _GZ_TO_URDF:
                return _GZ_TO_URDF[cand]
    return None


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


def _json_objects_from_stream(stream) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    buf = ""
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break
        buf += chunk
        while True:
            stripped = buf.lstrip()
            if stripped != buf:
                buf = stripped
            if not buf:
                break
            try:
                obj, idx = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                break
            if isinstance(obj, dict):
                yield obj
            buf = buf[idx:]


def _iter_model_dicts(payload: object) -> Iterator[dict]:
    if not isinstance(payload, dict):
        return
    yield payload
    nested = payload.get("model")
    if isinstance(nested, list):
        for item in nested:
            if isinstance(item, dict):
                yield from _iter_model_dicts(item)


def _joint_state_from_gz_model(msg: dict) -> JointState | None:
    pos: dict[str, float] = {}
    vel: dict[str, float] = {}
    eff: dict[str, float] = {}

    for model in _iter_model_dicts(msg):
        joints = model.get("joint")
        if not isinstance(joints, list):
            continue
        for joint in joints:
            if not isinstance(joint, dict):
                continue
            key = _canonical_joint_name(str(joint.get("name", "")))
            if key is None:
                continue
            axis1 = joint.get("axis1")
            if not isinstance(axis1, dict):
                continue
            if "position" in axis1:
                pos[key] = float(axis1["position"])
            if "velocity" in axis1:
                vel[key] = float(axis1["velocity"])
            if "force" in axis1:
                eff[key] = float(axis1["force"])

    if not pos:
        return None

    out = JointState()
    for j in _ARM:
        out.name.append(j)
        out.position.append(pos.get(j, 0.0))
        out.velocity.append(vel.get(j, 0.0))
        out.effort.append(eff.get(j, 0.0))
    return out


class JointStatesBridge(Node):
    def __init__(self) -> None:
        super().__init__("cs612_joint_states_bridge")
        self.declare_parameter("use_gz_native_fallback", True)
        self.declare_parameter("ros_bridge_timeout_sec", 3.0)
        self.declare_parameter("gz_joint_state_topic", "/world/arm_world/model/CS612_arm/joint_state")
        # robot_state_publisher 对 joint_states 要求 name 与 position 等长；发布端与 RSP 默认 SensorDataQoS 对齐。
        self._pub = self.create_publisher(JointState, "joint_states", _JOINT_STATES_PUBLISHER_QOS)
        self.create_subscription(JointState, "joint_states_gz", self._cb, _JOINT_STATES_SUBSCRIBER_QOS)
        self._logged_first = False

        self._last: JointState | None = None
        self._logged_seed = False
        self._start_monotonic = time.monotonic()
        self._native_started = False
        self._native_seen = False
        self._gz = _find_gz_executable()
        self._stop = threading.Event()
        self._native_thread: threading.Thread | None = None
        self._native_proc: subprocess.Popen[str] | None = None
        # 在 Gazebo 首帧前先发布全零姿态，避免 RViz / RobotModel 因整链缺 TF 而全红。
        # 这里必须使用稳定时钟：use_sim_time=true 且 /clock 尚未桥接时，ROS 时间不会前进，
        # 普通定时器不会触发，/joint_states 也就不会发布，RViz 会报 link1..6 无 TF。
        # 收到真实 joint_states_gz 后会立刻切换为仿真状态。
        self.create_timer(
            1.0 / 20.0,
            self._tick,
            clock=Clock(clock_type=ClockType.STEADY_TIME),
        )

    def _make_zero(self) -> JointState:
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        for j in _ARM:
            out.name.append(j)
            out.position.append(_HOME_POSITIONS.get(j, 0.0))
            out.velocity.append(0.0)
            out.effort.append(0.0)
        return out

    def _build_from_gz(self, msg: JointState) -> JointState:
        pos: dict[str, float] = {}
        vel: dict[str, float] = {}
        eff: dict[str, float] = {}
        # Gazebo Model 关节状态常含 world_fixed 等与 URDF 可动关节无关的项：必须按名称对齐，
        # 禁止在「有 name 但含额外关节」时用 position[0..5] 误映射到 joint1..6。
        if msg.name:
            for i, raw in enumerate(msg.name):
                key = _canonical_joint_name(raw)
                if key is None:
                    continue
                if i < len(msg.position):
                    pos[key] = float(msg.position[i])
                if i < len(msg.velocity):
                    vel[key] = float(msg.velocity[i])
                if i < len(msg.effort):
                    eff[key] = float(msg.effort[i])

        # 仅当完全没有关节名、且仅有 6 个标量时，才按顺序对应标准关节
        if not pos and (not msg.name or len(msg.name) == 0) and len(msg.position) >= len(_ARM):
            for i, jn in enumerate(_ARM):
                pos[jn] = float(msg.position[i])
            if len(msg.velocity) >= len(_ARM):
                for i, jn in enumerate(_ARM):
                    vel[jn] = float(msg.velocity[i])
            if len(msg.effort) >= len(_ARM):
                for i, jn in enumerate(_ARM):
                    eff[jn] = float(msg.effort[i])

        out = JointState()
        out.header = msg.header
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        for j in _ARM:
            out.name.append(j)
            out.position.append(pos.get(j, 0.0))
            out.velocity.append(vel.get(j, 0.0))
            out.effort.append(eff.get(j, 0.0))
        return out

    def _publish_stamped(self, template: JointState) -> None:
        stamped = JointState()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.name = list(template.name)
        stamped.position = list(template.position)
        stamped.velocity = list(template.velocity)
        stamped.effort = list(template.effort)
        self._pub.publish(stamped)

    def _cb(self, msg: JointState) -> None:
        # 仅更新状态，由 _tick 统一按固定频率发布，QoS 与 RSP 一致且首帧即有 TF
        if not self._logged_first and msg.name:
            sample = next((n for n in msg.name if _canonical_joint_name(n)), msg.name[0])
            self.get_logger().info(
                f"已收到 Gazebo 关节状态（示例关节名: {sample}），/joint_states 将与仿真同步。"
            )
            self._logged_first = True
        self._last = self._build_from_gz(msg)

    def _ensure_native_fallback_started(self) -> None:
        if self._native_started:
            return
        if not bool(self.get_parameter("use_gz_native_fallback").value):
            return
        timeout_sec = max(0.0, float(self.get_parameter("ros_bridge_timeout_sec").value))
        if self._last is not None or time.monotonic() - self._start_monotonic < timeout_sec:
            return
        self._native_started = True
        topic = str(self.get_parameter("gz_joint_state_topic").value)
        self.get_logger().warn(
            "在限定时间内未收到 /joint_states_gz，切换到原生 ign topic 关节状态回退通道: "
            f"{topic}"
        )
        self._native_thread = threading.Thread(
            target=self._native_gz_loop,
            daemon=True,
            name="cs612-gz-joint-states",
        )
        self._native_thread.start()

    def _native_gz_loop(self) -> None:
        topic = str(self.get_parameter("gz_joint_state_topic").value)
        while rclpy.ok() and not self._stop.is_set():
            proc: subprocess.Popen[str] | None = None
            try:
                proc = subprocess.Popen(
                    [self._gz, "topic", "-e", "-t", topic, "--json-output"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self._native_proc = proc
                if proc.stdout is None:
                    raise RuntimeError(f"{os.path.basename(self._gz)} topic stdout is unavailable")
                for obj in _json_objects_from_stream(proc.stdout):
                    js = _joint_state_from_gz_model(obj)
                    if js is None:
                        continue
                    js.header.stamp = self.get_clock().now().to_msg()
                    self._last = js
                    if not self._native_seen:
                        sample = js.name[0] if js.name else "shoulder_pan_joint"
                        self.get_logger().info(
                            f"已收到 Gazebo 原生关节状态回退流（示例关节名: {sample}），"
                            "/joint_states 将直接与仿真同步。"
                        )
                        self._native_seen = True
                        self._logged_first = True
                if not self._stop.is_set():
                    stderr = ""
                    if proc.stderr is not None:
                        stderr = proc.stderr.read().strip()
                    if stderr:
                        self.get_logger().warn(f"`{os.path.basename(self._gz)} topic` 退出，joint_state 回退通道将重试: {stderr}")
            except Exception as exc:
                if not self._stop.is_set():
                    self.get_logger().warn(f"原生 joint_state 回退通道异常，将重试: {exc}")
            finally:
                self._native_proc = None
                if proc is not None:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    try:
                        proc.wait(timeout=0.5)
                    except Exception:
                        try:
                            proc.kill()
                        except Exception:
                            pass
            if not self._stop.is_set():
                time.sleep(1.0)

    def _tick(self) -> None:
        self._ensure_native_fallback_started()
        if self._last is None:
            self._publish_stamped(self._make_zero())
            if not self._logged_seed:
                self.get_logger().info(
                    "尚未收到 Gazebo 关节状态，先发布零位 /joint_states 以保持 TF 链完整。"
                )
                self._logged_seed = True
            return
        self._publish_stamped(self._last)

    def destroy_node(self) -> bool:
        self._stop.set()
        proc = self._native_proc
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        if self._native_thread is not None:
            self._native_thread.join(timeout=0.5)
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = JointStatesBridge()
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
