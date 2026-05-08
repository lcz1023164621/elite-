"""在命名空间内发布固定关节状态，配合该命名空间下的 robot_state_publisher 使用。"""
from __future__ import annotations

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

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

    def _tick(self) -> None:
        self._msg.header.stamp = self.get_clock().now().to_msg()
        self._pub.publish(self._msg)


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
