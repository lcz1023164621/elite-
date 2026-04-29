#!/usr/bin/env python3
"""Bridge MoveIt FollowJointTrajectory to Gazebo joint position command topics."""
from __future__ import annotations

import math
import time
from typing import Dict, List

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

# MoveIt / URDF 标准关节名 → Gazebo SDF 内部关节名
_URDF_TO_GZ = {
    "shoulder_pan_joint": "joint1",
    "shoulder_lift_joint": "joint2",
    "elbow_joint": "joint3",
    "wrist_1_joint": "joint4",
    "wrist_2_joint": "joint5",
    "wrist_3_joint": "joint6",
}

_ARM_JOINTS = list(_URDF_TO_GZ.keys())
_JOINT_COMMAND_TOPICS = {
    joint: f"/cs612/joint_command/{_URDF_TO_GZ[joint]}" for joint in _ARM_JOINTS
}


class TrajectoryActionBridge(Node):
    def __init__(self) -> None:
        super().__init__("cs612_trajectory_action_bridge")
        self._cb = ReentrantCallbackGroup()
        self._latest_positions: Dict[str, float] = {j: 0.0 for j in _ARM_JOINTS}
        self._cmd_pubs = {
            joint: self.create_publisher(Float64, topic, 10)
            for joint, topic in _JOINT_COMMAND_TOPICS.items()
        }
        # 发送整条轨迹（必要时抽样），并在末点确认关节基本收敛后再向 MoveIt 返回成功。
        self.declare_parameter("max_command_points", 60)
        self.declare_parameter("point_wait_cap_sec", 0.02)
        # Gazebo wrist joints may converge slowly or report an equivalent wrapped
        # angle.  Pre-grasp may soft-pass, then auto_pick_place verifies and
        # corrects the real suction-cup TF before descending.
        self.declare_parameter("goal_tolerance", 0.03)
        self.declare_parameter("goal_soft_tolerance", 0.04)
        self.declare_parameter("loose_tolerance_joints", ["wrist_3_joint"])
        self.declare_parameter("loose_goal_tolerance", 0.06)
        self.declare_parameter("loose_goal_soft_tolerance", 0.20)
        self.declare_parameter("goal_settle_timeout_sec", 120.0)
        self.declare_parameter("goal_hold_publish_period_sec", 0.05)
        self.create_subscription(
            JointState,
            "/joint_states",
            self._on_joint_states,
            qos_profile_sensor_data,
            callback_group=self._cb,
        )

        self._action_server = ActionServer(
            self,
            FollowJointTrajectory,
            "/manipulator_controller/follow_joint_trajectory",
            callback_group=self._cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            execute_callback=self._execute_cb,
        )
        self.get_logger().info(
            "已启动 FollowJointTrajectory 桥接：/manipulator_controller/follow_joint_trajectory"
        )

    def _on_joint_states(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            if name in self._latest_positions and idx < len(msg.position):
                self._latest_positions[name] = float(msg.position[idx])

    def _goal_cb(self, goal_request: FollowJointTrajectory.Goal) -> GoalResponse:
        jt = goal_request.trajectory
        if not jt.joint_names:
            self.get_logger().warn("拒绝空 joint_names 目标")
            return GoalResponse.REJECT
        if not jt.points:
            self.get_logger().warn("拒绝空轨迹目标")
            return GoalResponse.REJECT
        unknown = [j for j in jt.joint_names if j not in _ARM_JOINTS]
        if unknown:
            self.get_logger().warn(f"拒绝未知关节: {unknown}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle) -> CancelResponse:
        self.get_logger().info("收到取消请求")
        return CancelResponse.ACCEPT

    def _publish_joint_positions(self, names: List[str], positions: List[float]) -> None:
        for idx, joint in enumerate(names):
            if joint in self._cmd_pubs and idx < len(positions):
                msg = Float64()
                msg.data = float(positions[idx])
                try:
                    self._cmd_pubs[joint].publish(msg)
                except Exception:
                    # Launch shutdown may destroy publishers while execute_cb is still unwinding.
                    return

    def _duration_to_sec(self, duration_msg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    def _joint_abs_errors(self, names: List[str], positions: List[float]) -> Dict[str, float]:
        errors: Dict[str, float] = {}
        for idx, joint in enumerate(names):
            if idx >= len(positions):
                continue
            actual = self._latest_positions.get(joint)
            if actual is None:
                continue
            errors[joint] = abs(math.atan2(math.sin(actual - float(positions[idx])), math.cos(actual - float(positions[idx]))))
        return errors

    def _joint_tol(self, joint_name: str, base_tol: float, soft_mode: bool) -> float:
        loose = set(str(v) for v in self.get_parameter("loose_tolerance_joints").value)
        if joint_name in loose:
            if soft_mode:
                return max(base_tol, float(self.get_parameter("loose_goal_soft_tolerance").value))
            return max(base_tol, float(self.get_parameter("loose_goal_tolerance").value))
        # 非 loose 关节：soft 模式使用更宽的 base_tol（soft_tolerance），
        # 允许 Gazebo 关节控制器有限收敛后再报告成功。
        if soft_mode:
            return max(base_tol, float(self.get_parameter("goal_soft_tolerance").value))
        return float(self.get_parameter("goal_tolerance").value)

    def _goal_reached(
        self,
        names: List[str],
        positions: List[float],
        tolerance: float,
        soft_mode: bool = False,
    ) -> bool:
        errors = self._joint_abs_errors(names, positions)
        if len(errors) != len(names):
            return False
        for joint in names:
            err = errors.get(joint)
            if err is None:
                return False
            if err > self._joint_tol(joint, tolerance, soft_mode):
                return False
        return True

    def _wait_until_goal_reached(self, names: List[str], positions: List[float]) -> str:
        strict_tolerance = float(self.get_parameter("goal_tolerance").value)
        soft_tolerance = max(strict_tolerance, float(self.get_parameter("goal_soft_tolerance").value))
        timeout_sec = float(self.get_parameter("goal_settle_timeout_sec").value)
        republish_period = max(0.01, float(self.get_parameter("goal_hold_publish_period_sec").value))
        deadline = self._now_sec() + max(0.1, timeout_sec)
        next_publish = 0.0
        log_interval = 10.0
        next_log = self._now_sec() + log_interval
        while self._now_sec() < deadline:
            if self._goal_reached(names, positions, strict_tolerance, soft_mode=False):
                return "strict"
            if self._goal_reached(names, positions, soft_tolerance, soft_mode=True):
                return "soft"
            now = self._now_sec()
            if now >= next_publish:
                self._publish_joint_positions(names, positions)
                next_publish = now + republish_period
            if now >= next_log:
                errors = self._joint_abs_errors(names, positions)
                detail = ", ".join(
                    f"{j}={e:.3f}" for j, e in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:3]
                )
                self.get_logger().info(f"收敛中(worst3): {detail}")
                next_log = now + log_interval
            time.sleep(0.02)
        if self._goal_reached(names, positions, strict_tolerance, soft_mode=False):
            return "strict"
        if self._goal_reached(names, positions, soft_tolerance, soft_mode=True):
            return "soft"
        errors = self._joint_abs_errors(names, positions)
        detail = ", ".join(
            f"{joint}={err:.3f}" for joint, err in sorted(errors.items(), key=lambda item: item[1], reverse=True)
        )
        self.get_logger().warn(f"末点未收敛，关节误差: {detail}")
        return "fail"

    def _now_sec(self) -> float:
        """Return elapsed seconds on the ROS clock (sim_time when use_sim_time=true)."""
        t = self.get_clock().now()
        return t.nanoseconds * 1e-9

    def _execute_cb(self, goal_handle):
        goal = goal_handle.request
        traj = goal.trajectory
        joint_names = list(traj.joint_names)
        points = self._downsample_points(list(traj.points))
        if not points:
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Empty trajectory points"
            return result
        self.get_logger().info(
            f"执行轨迹点数: {len(points)}（joint_names={joint_names}）"
        )

        feedback = FollowJointTrajectory.Feedback()
        feedback.joint_names = joint_names

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = "Goal canceled"
            return result

        start = self._now_sec()
        wait_cap = float(self.get_parameter("point_wait_cap_sec").value)

        for point in points:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Goal canceled"
                return result

            target_t = max(0.0, self._duration_to_sec(point.time_from_start))
            while True:
                elapsed = self._now_sec() - start
                remain = target_t - elapsed
                if remain <= 0.0:
                    break
                time.sleep(min(wait_cap, remain))

            self._publish_joint_positions(joint_names, list(point.positions))

            feedback.desired = point
            feedback.actual.positions = [self._latest_positions.get(j, 0.0) for j in joint_names]
            feedback.error.positions = [
                d - a for d, a in zip(feedback.desired.positions, feedback.actual.positions)
            ]
            goal_handle.publish_feedback(feedback)

        final_positions = list(points[-1].positions)
        reach_status = self._wait_until_goal_reached(joint_names, final_positions)
        if reach_status == "fail":
            goal_handle.abort()
            result = FollowJointTrajectory.Result()
            result.error_code = FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED
            result.error_string = "Gazebo joints did not converge to the final command in time"
            return result
        if reach_status == "soft":
            errors = self._joint_abs_errors(joint_names, final_positions)
            detail = ", ".join(
                f"{joint}={err:.3f}" for joint, err in sorted(errors.items(), key=lambda item: item[1], reverse=True)
            )
            self.get_logger().warn(f"末点以软容差通过，关节误差: {detail}")

        goal_handle.succeed()
        result = FollowJointTrajectory.Result()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        result.error_string = "Trajectory executed through ROS->Gazebo joint command bridges"
        return result

    def _downsample_points(self, points: List) -> List:
        max_pts = int(self.get_parameter("max_command_points").value)
        if max_pts <= 0 or len(points) <= max_pts:
            return points
        # 均匀抽样并始终保留最后一点，避免极长轨迹把 ROS/Gazebo 桥接消息堆得过密。
        step = max(1, math.ceil(len(points) / max_pts))
        sampled = points[::step]
        if sampled[-1] is not points[-1]:
            sampled.append(points[-1])
        return sampled


def main() -> None:
    rclpy.init()
    node = TrajectoryActionBridge()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
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
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
