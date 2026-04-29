"""Regression checks for sync, TF and base anchor stability."""
from __future__ import annotations

import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


class RegressionChecker(Node):
    def __init__(self) -> None:
        super().__init__("cs612_regression_checker")
        self.declare_parameter("timeout_sec", 30.0)
        self.declare_parameter("base_pose_topic", "/model/cs612/pose")
        self.declare_parameter("base_drift_tol", 0.01)

        self._clock_ok = False
        self._js_ok = False
        self._rect_ok = False
        self._carton_ok = False
        self._base_ok = False
        self._base_samples = 0

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint, 10)
        self.create_subscription(PoseStamped, "/model/rect_pickup/pose", self._on_rect, 10)
        self.create_subscription(PoseStamped, "/model/carton_box/pose", self._on_carton, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("base_pose_topic").value), self._on_base, 10)

    def _on_clock(self, msg: Clock) -> None:
        if msg.clock.sec != 0 or msg.clock.nanosec != 0:
            self._clock_ok = True

    def _on_joint(self, msg: JointState) -> None:
        if msg.name and msg.position and len(msg.name) == len(msg.position):
            self._js_ok = True

    def _on_rect(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) > 1e-6 or abs(p.y) > 1e-6 or abs(p.z) > 1e-6:
            self._rect_ok = True

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) > 1e-6 or abs(p.y) > 1e-6 or abs(p.z) > 1e-6:
            self._carton_ok = True

    def _on_base(self, msg: PoseStamped) -> None:
        tol = max(0.001, float(self.get_parameter("base_drift_tol").value))
        p = msg.pose.position
        self._base_samples += 1
        if abs(p.x) <= tol and abs(p.y) <= tol and abs(p.z) <= tol:
            self._base_ok = True

    def run(self) -> int:
        timeout = max(5.0, float(self.get_parameter("timeout_sec").value))
        t0 = time.monotonic()
        while rclpy.ok() and (time.monotonic() - t0) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            try:
                tf_ok = self._tf_buffer.can_transform("base_link", "suction_tcp_link", rclpy.time.Time())
            except Exception:
                tf_ok = False
            if self._clock_ok and self._js_ok and self._rect_ok and self._carton_ok and self._base_ok and tf_ok:
                self.get_logger().info("REGRESSION PASS: sync/tf/base checks all passed")
                return 0
        try:
            tf_ok = self._tf_buffer.can_transform("base_link", "suction_tcp_link", rclpy.time.Time())
        except Exception:
            tf_ok = False
        self.get_logger().error(
            "REGRESSION FAIL: "
            f"clock={self._clock_ok}, joint_states={self._js_ok}, rect_pose={self._rect_ok}, "
            f"carton_pose={self._carton_ok}, base_anchor={self._base_ok} (samples={self._base_samples}), tf={tf_ok}"
        )
        return 2


def main() -> None:
    rclpy.init()
    node = RegressionChecker()
    code = 2
    try:
        code = node.run()
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
    raise SystemExit(code)


if __name__ == "__main__":
    main()
