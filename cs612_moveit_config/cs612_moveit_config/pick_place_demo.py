"""MoveIt 关节空间抓取演示：预定位 → 吸附 → 搬运 → 松开（需先启动 bringup.launch.py）。"""
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
from std_msgs.msg import Empty

_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class PickPlaceDemo(Node):
    def __init__(self) -> None:
        super().__init__("cs612_pick_place_demo")
        self._action = ActionClient(self, MoveGroup, "move_action")
        self._pub_attach = self.create_publisher(Empty, "/cs612/suction/attach", 10)
        self._pub_detach = self.create_publisher(Empty, "/cs612/suction/detach", 10)
        self.declare_parameter("config_file", "")
        cfg_param = self.get_parameter("config_file").get_parameter_value().string_value
        if cfg_param:
            cfg_path = Path(cfg_param)
        else:
            share = Path(get_package_share_directory("cs612_moveit_config"))
            cfg_path = share / "config" / "pick_place_demo.yaml"
        self._cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        self.get_logger().info(f"已加载关节目标配置: {cfg_path}")

    def _build_joint_goal(self, positions: list[float]) -> MoveGroup.Goal:
        goal = MoveGroup.Goal()
        req = goal.request
        req.group_name = "arm"
        req.num_planning_attempts = 15
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2
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
        if not self._action.wait_for_server(timeout_sec=60.0):
            self.get_logger().error("move_action 不可用（move_group 是否已启动？）")
            return False
        goal = self._build_joint_goal(positions)
        self.get_logger().info(f"规划并执行: {label}")
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
            return False
        err = res.result.error_code.val
        if err == MoveItErrorCodes.SUCCESS:
            self.get_logger().info(f"{label}: 成功 (planning_time={res.result.planning_time:.3f}s)")
            return True
        self.get_logger().error(f"{label}: MoveIt 错误码 {err}")
        return False

    def run(self) -> None:
        h = self._cfg["home"]
        pre = self._cfg["pick_pre"]
        touch = self._cfg["pick_touch"]
        place = self._cfg["place"]

        self.get_logger().info("启动先 detach 清状态（DetachableJoint 默认初始附着）")
        self._pub_detach.publish(Empty())
        time.sleep(0.2)

        if not self._send_move(h, "home"):
            return
        time.sleep(0.8)
        if not self._send_move(pre, "pick_pre"):
            return
        time.sleep(0.8)
        if not self._send_move(touch, "pick_touch"):
            return
        time.sleep(0.5)
        self.get_logger().info("发送吸附 attach（DetachableJoint）")
        self._pub_attach.publish(Empty())
        time.sleep(0.6)
        if not self._send_move(place, "place"):
            self._pub_detach.publish(Empty())
            return
        time.sleep(0.5)
        self.get_logger().info("发送释放 detach")
        self._pub_detach.publish(Empty())
        time.sleep(0.3)
        self._send_move(h, "回 home")


def main() -> None:
    rclpy.init()
    node = PickPlaceDemo()
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
