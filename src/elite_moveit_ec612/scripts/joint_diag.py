#!/usr/bin/env python3
"""Diagnostic script: prints actual joint positions from /joint_states in real-time.

Run alongside the bringup launch:
  ros2 run elite_moveit_ec612 cs612_joint_diag

Or directly:
  python3 /mnt/e/gazebo_projects/my_first_world/src/elite_moveit_ec612/scripts/joint_diag.py
"""
from __future__ import annotations

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from control_msgs.msg import JointTrajectoryControllerState

_URDF_ORDER = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

_SHORT = {
    "shoulder_pan_joint": "J1_pan",
    "shoulder_lift_joint": "J2_lift",
    "elbow_joint": "J3_elbow",
    "wrist_1_joint": "J4_wr1",
    "wrist_2_joint": "J5_wr2",
    "wrist_3_joint": "J6_wr3",
}


class JointDiag(Node):
    def __init__(self):
        super().__init__("cs612_joint_diag")
        self._actual: dict[str, float] = {}
        self._cmd: dict[str, float] = {}
        self._actual_stamp: float = 0.0
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_js,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_subscription(
            JointTrajectoryControllerState,
            "/joint_trajectory_controller/controller_state",
            self._on_controller_state,
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        self.create_timer(0.1, self._print)
        self.get_logger().info(
            "Joint diagnostic started. Columns: Joint | Actual(deg) | Cmd(deg) | Err(deg)"
        )

    def _on_js(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name in _URDF_ORDER and i < len(msg.position):
                self._actual[name] = float(msg.position[i])
        self._actual_stamp = time.monotonic()

    def _on_controller_state(self, msg: JointTrajectoryControllerState) -> None:
        for i, name in enumerate(msg.joint_names):
            if name in _URDF_ORDER and i < len(msg.actual.positions):
                self._actual[name] = float(msg.actual.positions[i])
            if name in _URDF_ORDER and i < len(msg.desired.positions):
                self._cmd[name] = float(msg.desired.positions[i])

    def _print(self) -> None:
        if not self._actual:
            self.get_logger().info("Waiting for /joint_states ...")
            return
        lines = ["--- Joint Diagnostic ---"]
        total_err = 0.0
        count = 0
        for j in _URDF_ORDER:
            a_rad = self._actual.get(j, float("nan"))
            c_rad = self._cmd.get(j, float("nan"))
            a_deg = math.degrees(a_rad) if not math.isnan(a_rad) else float("nan")
            c_deg = math.degrees(c_rad) if not math.isnan(c_rad) else float("nan")
            if not math.isnan(a_rad) and not math.isnan(c_rad):
                err = math.degrees(math.atan2(math.sin(a_rad - c_rad), math.cos(a_rad - c_rad)))
                total_err += abs(err)
                count += 1
            else:
                err = float("nan")
            s = _SHORT[j]
            lines.append(f"  {s:8s} | actual={a_deg:8.2f}° | cmd={c_deg:8.2f}° | err={err:+7.2f}°")
        if count > 0:
            worst = 0.0
            for j in _URDF_ORDER:
                if j in self._actual and j in self._cmd:
                    a = self._actual[j]
                    c = self._cmd[j]
                    e = abs(math.degrees(math.atan2(math.sin(a - c), math.cos(a - c))))
                    if e > worst:
                        worst = e
            lines.append(f"  TOTAL | mean_err={total_err / count:.2f}° | worst_err={worst:.2f}°")
        self.get_logger().info("\n".join(lines))


def main() -> None:
    rclpy.init()
    node = JointDiag()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()