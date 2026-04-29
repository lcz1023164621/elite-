"""最小执行链验证：home -> test_pose -> home。"""
from __future__ import annotations

import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState
from rclpy.action import ActionClient
from rclpy.node import Node

_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class MotionSmokeTest(Node):
    def __init__(self) -> None:
        super().__init__("cs612_motion_smoke_test")
        self._action = ActionClient(self, MoveGroup, "move_action")
        self.declare_parameter("config_file", "")
        self.declare_parameter("move_velocity_scale", 0.25)
        self.declare_parameter("move_acceleration_scale", 0.25)

        cfg_param = self.get_parameter("config_file").get_parameter_value().string_value
        if cfg_param:
            cfg_path = Path(cfg_param)
        else:
            share = Path(get_package_share_directory("cs612_moveit_config"))
            cfg_path = share / "config" / "motion_smoke_test.yaml"

        self._cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.get_logger().info(f"已加载最小运动验证配置: {cfg_path}")

    def _build_joint_goal(self, positions: list[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        goal.planning_options.plan_only = False
        req = goal.request
        req.group_name = "arm"
        req.num_planning_attempts = 15
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = float(self.get_parameter("move_velocity_scale").value)
        req.max_acceleration_scaling_factor = float(self.get_parameter("move_acceleration_scale").value)
        req.pipeline_id = "ompl"
        req.planner_id = "RRTConnect"
        c = Constraints()
        for name, pos in zip(_ARM_JOINTS, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = 0.1
            jc.tolerance_below = 0.1
            jc.weight = 1.0
            c.joint_constraints.append(jc)
        req.goal_constraints = [c]
        req.start_state = RobotState()
        req.start_state.is_diff = True
        return goal

    def _send_move(self, positions: list[float], label: str) -> bool:
        if len(positions) != 6:
            self.get_logger().error(f"{label}: 需要 6 个关节角")
            return False
        if not self._action.wait_for_server(timeout_sec=30.0):
            self.get_logger().error("move_action 不可用（move_group 未就绪）")
            return False
        goal = self._build_joint_goal(positions)
        self.get_logger().info(f"执行 {label}")
        send_future = self._action.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        gh = send_future.result()
        if gh is None or not gh.accepted:
            self.get_logger().error(f"{label}: 目标被拒绝")
            return False
        result_future = gh.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        res = result_future.result()
        if res is None:
            self.get_logger().error(f"{label}: 未收到结果")
            return False
        err = res.result.error_code.val
        if err != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"{label}: MoveIt 错误码 {err}")
            return False
        self.get_logger().info(f"{label}: 成功")
        return True

    def run(self) -> None:
        home = list(self._cfg.get("home", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        test_pose = list(self._cfg.get("test_pose", [0.65, -0.52, 0.78, 0.0, 1.05, 0.0]))

        if not self._send_move(home, "home_start"):
            return
        time.sleep(0.5)
        if not self._send_move(test_pose, "test_pose"):
            return
        time.sleep(0.5)
        self._send_move(home, "home_end")


def main() -> None:
    rclpy.init()
    node = MotionSmokeTest()
    try:
        node.run()
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
