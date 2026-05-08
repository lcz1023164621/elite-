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

_ELITE_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_ARM_JOINTS = list(_ELITE_ARM_JOINTS)
_LOCAL_JOINTS = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
_ELITE_TO_LOCAL = {e: j for e, j in zip(_ELITE_ARM_JOINTS, _LOCAL_JOINTS)}
_LOCAL_TO_ELITE = {j: e for e, j in _ELITE_TO_LOCAL.items()}
_JOINT_COMMAND_TOPICS = {
    joint: f"/cs612/joint_command/{_ELITE_TO_LOCAL[joint]}" for joint in _ARM_JOINTS
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
        self.declare_parameter("max_command_points", 24)
        self.declare_parameter("point_wait_cap_sec", 0.02)
        # Gazebo 位置控制器对末端腕部关节收敛较慢；这里适度放宽执行判据，
        # 避免机械臂已到达可吸附姿态却被桥接层误判失败。
        self.declare_parameter("goal_tolerance", 0.16)
        self.declare_parameter("goal_soft_tolerance", 0.40)
        # wrist_2 / wrist_3 对吸盘中心位置和“是否位于物体顶面”影响较小，允许更宽收敛阈值。
        self.declare_parameter("loose_tolerance_joints", ["wrist_2_joint", "wrist_3_joint"])
        self.declare_parameter("loose_goal_tolerance", 1.20)
        self.declare_parameter("loose_goal_soft_tolerance", 1.40)
        self.declare_parameter("goal_settle_timeout_sec", 45.0)
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
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=self._cb,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            execute_callback=self._execute_cb,
        )
        self.get_logger().info(
            "已启动 FollowJointTrajectory 桥接：/joint_trajectory_controller/follow_joint_trajectory"
        )

    def _canonical_joint(self, name: str) -> str | None:
        if name in _ARM_JOINTS:
            return name
        if name in _LOCAL_TO_ELITE:
            return _LOCAL_TO_ELITE[name]
        return None

    def _on_joint_states(self, msg: JointState) -> None:
        for idx, name in enumerate(msg.name):
            key = self._canonical_joint(name)
            if key in self._latest_positions and idx < len(msg.position):
                self._latest_positions[key] = float(msg.position[idx])

    def _goal_cb(self, goal_request: FollowJointTrajectory.Goal) -> GoalResponse:
        jt = goal_request.trajectory
        if not jt.joint_names:
            self.get_logger().warn("拒绝空 joint_names 目标")
            return GoalResponse.REJECT
        if not jt.points:
            self.get_logger().warn("拒绝空轨迹目标")
            return GoalResponse.REJECT
        unknown = [j for j in jt.joint_names if j not in _ARM_JOINTS]
        unknown = [j for j in unknown if j not in _ELITE_TO_LOCAL]
        if unknown:
            self.get_logger().warn(f"拒绝未知关节: {unknown}")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _cancel_cb(self, _goal_handle) -> CancelResponse:
        self.get_logger().info("收到取消请求")
        return CancelResponse.ACCEPT

    def _publish_joint_positions(self, names: List[str], positions: List[float]) -> None:
        for idx, joint in enumerate(names):
            key = self._canonical_joint(joint)
            if key in self._cmd_pubs and idx < len(positions):
                msg = Float64()
                msg.data = float(positions[idx])
                self._cmd_pubs[key].publish(msg)

    def _duration_to_sec(self, duration_msg) -> float:
        return float(duration_msg.sec) + float(duration_msg.nanosec) * 1e-9

    def _joint_abs_errors(self, names: List[str], positions: List[float]) -> Dict[str, float]:
        errors: Dict[str, float] = {}
        for idx, joint in enumerate(names):
            key = self._canonical_joint(joint)
            if idx >= len(positions):
                continue
            actual = self._latest_positions.get(key) if key is not None else None
            if actual is None:
                continue
            errors[joint] = abs(actual - float(positions[idx]))
        return errors

    def _joint_tol(self, joint_name: str, base_tol: float, soft_mode: bool) -> float:
        loose = set(str(v) for v in self.get_parameter("loose_tolerance_joints").value)
        if joint_name in loose:
            if soft_mode:
                return max(base_tol, float(self.get_parameter("loose_goal_soft_tolerance").value))
            return max(base_tol, float(self.get_parameter("loose_goal_tolerance").value))
        # 非 loose 关节必须按严格阈值收敛，避免“软通过”后当前状态掉进自碰撞。
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
        deadline = time.monotonic() + max(0.1, timeout_sec)
        next_publish = 0.0
        while time.monotonic() < deadline:
            if not rclpy.ok():
                return "fail"
            if self._goal_reached(names, positions, strict_tolerance, soft_mode=False):
                return "strict"
            if self._goal_reached(names, positions, soft_tolerance, soft_mode=True):
                return "soft"
            now = time.monotonic()
            if now >= next_publish:
                try:
                    self._publish_joint_positions(names, positions)
                except Exception:
                    return "fail"
                next_publish = now + republish_period
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

        start = time.monotonic()
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
                elapsed = time.monotonic() - start
                remain = target_t - elapsed
                if remain <= 0.0:
                    break
                time.sleep(min(wait_cap, remain))

            try:
                self._publish_joint_positions(joint_names, list(point.positions))
            except Exception:
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Joint command publish failed during shutdown or bridge teardown"
                return result

            feedback.desired = point
            feedback.actual.positions = [self._latest_positions.get(j, 0.0) for j in joint_names]
            feedback.error.positions = [
                d - a for d, a in zip(feedback.desired.positions, feedback.actual.positions)
            ]
            try:
                goal_handle.publish_feedback(feedback)
            except Exception:
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.INVALID_GOAL
                result.error_string = "Action feedback publish failed during shutdown or bridge teardown"
                return result

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
