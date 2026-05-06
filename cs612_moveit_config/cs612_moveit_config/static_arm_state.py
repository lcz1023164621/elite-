"""在命名空间内发布固定关节状态，配合该命名空间下的 robot_state_publisher 使用。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

# 与 eli_cs_robot_description/config/initial_positions.yaml 一致（GZ 静止臂初始姿态）
_DEFAULT_POSITIONS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 0.0,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": 1.57,
    "wrist_3_joint": 0.0,
}
_DEBUG_LOG_PATH = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-a97e6b.log")
_DEBUG_SESSION_ID = "a97e6b"


def _debug_log(location: str, message: str, hypothesis_id: str, data: dict) -> None:
    import os

    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": os.environ.get("CS612_DEBUG_RUN_ID", "pre-fix"),
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


class StaticArmStateNode(Node):
    def __init__(self) -> None:
        super().__init__("cs612_static_arm_state")
        self.declare_parameter("publish_hz", 50.0)
        hz = max(1.0, float(self.get_parameter("publish_hz").value))
        # 与 robot_state_publisher 订阅 joint_states 的 SensorData QoS（BEST_EFFORT）一致，否则不可靠↔最优努力不配对
        self._pub = self.create_publisher(JointState, "joint_states", qos_profile_sensor_data)
        self.create_timer(1.0 / hz, self._tick)
        self._msg = JointState()
        self._msg.name = list(_DEFAULT_POSITIONS.keys())
        self._msg.position = list(_DEFAULT_POSITIONS.values())
        self._tick()
        self.get_logger().info(
            f"固定 joint_states ({len(self._msg.name)} 关节)，namespace={self.get_namespace()!r}"
        )
        # #region agent log
        self._tf_ns = self.get_namespace().strip("/")
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._tf_probe_fired = False
        self._tf_probe_timer = self.create_timer(15.0, self._tf_probe_once)
        # #endregion

    def _tick(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)

    def _tf_probe_once(self) -> None:
        # #region agent log
        if getattr(self, "_tf_probe_fired", True):
            return
        self._tf_probe_fired = True
        try:
            self._tf_probe_timer.cancel()
        except BaseException:
            pass

        ns = getattr(self, "_tf_ns", "")
        now = self.get_clock().now()
        chains = {
            "world_to_ns_world": ("world", f"{ns}/world"),
            "world_to_ns_base": ("world", f"{ns}/base_link"),
        }
        results: dict[str, object] = {}
        for label, (src, dst) in chains.items():
            try:
                tr = self._tf_buffer.lookup_transform(src, dst, now, timeout=Duration(seconds=4.0))
                results[label] = {
                    "ok": True,
                    "t": [
                        tr.transform.translation.x,
                        tr.transform.translation.y,
                        tr.transform.translation.z,
                    ],
                }
            except Exception as ex:  # noqa: BLE001 - 调试探针
                results[label] = {"ok": False, "err": f"{type(ex).__name__}: {ex}"[:400]}

        frames_blob = ""
        try:
            frames_blob = self._tf_buffer.all_frames_as_yaml()  # type: ignore[union-attr]
        except BaseException:
            pass

        _debug_log(
            "static_arm_state.py:_tf_probe_once",
            "tf_chain_probe",
            "H_tf",
            {
                "namespace": ns,
                "chains": results,
                "frames_has_ns": ns in frames_blob if ns else False,
                "frames_yaml_head": frames_blob[:800] if frames_blob else "",
            },
        )
        # #endregion


def main() -> None:
    # #region agent log
    _debug_log(
        "static_arm_state.py:main",
        "entry_before_rclpy_init",
        "H1",
        {},
    )
    # #endregion
    rclpy.init()
    node = StaticArmStateNode()
    # #region agent log
    _debug_log(
        "static_arm_state.py:main",
        "node_constructed",
        "H4",
        {"namespace": node.get_namespace()},
    )
    # #endregion
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
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
