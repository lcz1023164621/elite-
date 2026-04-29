"""Runtime checks for Gazebo <-> ROS sync chain health."""
from __future__ import annotations

import time

import rclpy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage

from .gazebo_pose_sync import extract_model_pose


class SystemWatchdog(Node):
    def __init__(self) -> None:
        super().__init__("cs612_system_watchdog")
        self.declare_parameter("check_period_sec", 2.0)
        self.declare_parameter("warn_after_sec", 12.0)
        self.declare_parameter("frame_root", "base_link")
        self.declare_parameter("frame_tip", "suction_tcp_link")

        self._t0 = time.monotonic()
        self._clock_ok = False
        self._js_ok = False
        self._rect_pose_ok = False
        self._carton_pose_ok = False
        self._tf_ok = False
        self._ready_logged = False
        self._warned = False

        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        self.create_subscription(Clock, "/clock", self._on_clock, 10)
        self.create_subscription(JointState, "/joint_states", self._on_joint, 10)
        self.create_subscription(PoseStamped, "/model/rect_pickup/pose", self._on_rect, 10)
        self.create_subscription(PoseStamped, "/model/carton_box/pose", self._on_carton, 10)
        self.create_subscription(TFMessage, "/world/arm_world/pose/info", self._on_world_pose_info, 10)
        self.create_subscription(TFMessage, "/world/arm_world/dynamic_pose/info", self._on_world_pose_info, 10)
        period = max(0.2, float(self.get_parameter("check_period_sec").value))
        self.create_timer(period, self._tick)

    def _on_clock(self, msg: Clock) -> None:
        if msg.clock.sec != 0 or msg.clock.nanosec != 0:
            self._clock_ok = True

    def _on_joint(self, msg: JointState) -> None:
        if msg.name and msg.position and len(msg.name) == len(msg.position):
            self._js_ok = True

    def _on_rect(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) > 1e-6 or abs(p.y) > 1e-6 or abs(p.z) > 1e-6:
            self._rect_pose_ok = True

    def _on_carton(self, msg: PoseStamped) -> None:
        p = msg.pose.position
        if abs(p.x) > 1e-6 or abs(p.y) > 1e-6 or abs(p.z) > 1e-6:
            self._carton_pose_ok = True

    def _on_world_pose_info(self, msg: TFMessage) -> None:
        rect = extract_model_pose(msg, "rect_pickup")
        if rect is not None:
            self._on_rect(rect)
        carton = extract_model_pose(msg, "carton_box")
        if carton is not None:
            self._on_carton(carton)

    def _tick(self) -> None:
        frame_root = str(self.get_parameter("frame_root").value)
        frame_tip = str(self.get_parameter("frame_tip").value)
        try:
            if self._tf_buffer.can_transform(frame_root, frame_tip, rclpy.time.Time()):
                self._tf_ok = True
        except Exception:
            pass

        all_ok = (
            self._clock_ok
            and self._js_ok
            and self._rect_pose_ok
            and self._carton_pose_ok
            and self._tf_ok
        )
        if all_ok and not self._ready_logged:
            self._ready_logged = True
            self.get_logger().info(
                "同步链路自检通过: /clock, /joint_states, /model/*/pose, TF(base_link->suction_tcp_link)"
            )
            return

        elapsed = time.monotonic() - self._t0
        warn_after = max(1.0, float(self.get_parameter("warn_after_sec").value))
        if (not all_ok) and elapsed >= warn_after and not self._warned:
            self._warned = True
            self.get_logger().warn(
                "同步链路未完全就绪: "
                f"clock={self._clock_ok}, joint_states={self._js_ok}, "
                f"rect_pose={self._rect_pose_ok}, carton_pose={self._carton_pose_ok}, tf={self._tf_ok}"
            )


def main() -> None:
    rclpy.init()
    node = SystemWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
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
