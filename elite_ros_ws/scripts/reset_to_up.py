#!/usr/bin/env python3
import sys
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint


class ResetToUp(Node):
    def __init__(self):
        super().__init__("reset_to_up")

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # 对应你 SRDF 里的 up 姿态
        self.up_positions = [
            0.0,
            -1.5707,
            0.0,
            -1.5707,
            0.0,
            0.0,
        ]

        # 常见 FollowJointTrajectory action 名称
        self.candidate_actions = [
            "/joint_trajectory_controller/follow_joint_trajectory",
            "/scaled_joint_trajectory_controller/follow_joint_trajectory",
            "/cs_joint_trajectory_controller/follow_joint_trajectory",
            "/arm_controller/follow_joint_trajectory",
            "/trajectory_controller/follow_joint_trajectory",
        ]

        self.client = None
        self.action_name = None

    def connect_controller(self):
        self.get_logger().info("正在尝试连接 FollowJointTrajectory 控制器...")

        for action_name in self.candidate_actions:
            self.get_logger().info(f"尝试 action: {action_name}")

            client = ActionClient(
                self,
                FollowJointTrajectory,
                action_name,
            )

            if client.wait_for_server(timeout_sec=2.0):
                self.client = client
                self.action_name = action_name
                self.get_logger().info(f"已连接轨迹控制器: {action_name}")
                return True

        self.get_logger().error("没有连接到任何 FollowJointTrajectory action。")
        self.get_logger().error("请执行下面命令检查实际 action 名称：")
        self.get_logger().error("ros2 action list | grep trajectory")
        return False

    def send_up_goal(self):
        if self.client is None:
            self.get_logger().error("轨迹控制器 client 为空")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = self.up_positions
        point.velocities = [0.0] * len(self.joint_names)
        point.accelerations = [0.0] * len(self.joint_names)

        # 给 5 秒完成复位，避免动作太猛
        point.time_from_start.sec = 5
        point.time_from_start.nanosec = 0

        goal.trajectory.points.append(point)

        self.get_logger().info("发送复位到 up 姿态的关节轨迹...")
        self.get_logger().info(f"joint_names: {self.joint_names}")
        self.get_logger().info(f"positions: {self.up_positions}")

        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()

        if goal_handle is None:
            self.get_logger().error("复位目标发送失败")
            return False

        if not goal_handle.accepted:
            self.get_logger().error("复位目标被控制器拒绝")
            return False

        self.get_logger().info("复位目标已接受，等待执行完成...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result_wrapper = result_future.result()

        if result_wrapper is None:
            self.get_logger().error("复位没有返回结果")
            return False

        result = result_wrapper.result

        self.get_logger().info(f"控制器返回 error_code: {result.error_code}")

        if result.error_code == 0:
            self.get_logger().info("机械臂已复位到 up 姿态")
            return True

        self.get_logger().error("控制器执行复位失败")
        return False


def main():
    rclpy.init()

    node = ResetToUp()

    ok = node.connect_controller()

    if ok:
        ok = node.send_up_goal()

    node.destroy_node()
    rclpy.shutdown()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
