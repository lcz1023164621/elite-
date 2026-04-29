#!/usr/bin/env python3
"""
严格顺序点位测试脚本 V2：限制 IK 分支跳变 / 防止机械臂绕一大圈

流程：
1. 先到零点关节姿态；
2. 再到高空安全点；
3. 到物料箱上方，停留几秒；
4. 回安全点；
5. 到包装箱上方，停留几秒；
6. 回安全点。

相比上一版新增：
- 订阅 /joint_states，规划每个笛卡尔点位前读取当前关节角；
- 对每段运动添加 path_constraints，限制各关节不要跳到另一组 IK 解；
- 特别限制 wrist_3_joint，避免末端绕自身轴转一大圈；
- 使用你通过 tf2_echo 读取到的 tool0 当前四元数作为姿态约束。
"""

import math
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    PositionConstraint,
    OrientationConstraint,
    JointConstraint,
    BoundingVolume,
)
from shape_msgs.msg import SolidPrimitive

from scene_params import get_scene


class Tool0NoFlipTester(Node):
    def __init__(self):
        super().__init__("test_tool0_points_no_flip")

        self.client = ActionClient(self, MoveGroup, "/move_action")

        self.group_name = "cs_manipulator"
        self.base_frame = "base_link"
        self.ee_link = "tool0"

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        self.latest_joints: Dict[str, float] = {}
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 20)

        # 真实“零点”：六轴 0 位。
        # 如果你的机器人 SRDF / 示教器零点不是这个，请按实际零点修改。
        self.zero_positions = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        # 是否启用末端姿态约束。
        self.use_orientation_constraint = True

        # 是否限制每一段运动的关节跳变。
        # 这是解决“明明点位很近，机械臂却绕一大圈”的关键开关。
        self.use_near_current_joint_path_constraint = True

        # 每段运动中，各关节允许偏离当前角度的范围，单位 rad。
        # 如果某一步规划失败，优先适当放宽对应关节，而不是直接关掉全部约束。
        self.near_current_tolerance = {
            "shoulder_pan_joint": 1.60,
            "shoulder_lift_joint": 1.20,
            "elbow_joint": 1.40,
            "wrist_1_joint": 1.20,
            "wrist_2_joint": 1.00,
            "wrist_3_joint": 0.45,
        }

        # 你刚才 tf2_echo base_link tool0 读到的四元数。
        # 这会让 tool0 尽量保持当前这个末端姿态去移动到各个点位。
        self.tool0_down_orientation = Quaternion()
        self.tool0_down_orientation.x = 0.441
        self.tool0_down_orientation.y = 0.504
        self.tool0_down_orientation.z = 0.488
        self.tool0_down_orientation.w = 0.560

    def joint_state_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                self.latest_joints[name] = float(pos)

    def wait_for_moveit(self):
        self.get_logger().info("等待 MoveIt move_action...")
        self.client.wait_for_server()
        self.get_logger().info("MoveIt move_action 已连接")

    def wait_for_joint_states(self, timeout_sec: float = 5.0) -> bool:
        self.get_logger().info("等待 /joint_states...")
        start = time.time()
        while rclpy.ok() and time.time() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(name in self.latest_joints for name in self.joint_names):
                text = ", ".join(
                    f"{name}={self.latest_joints[name]:.3f}" for name in self.joint_names
                )
                self.get_logger().info(f"已获取当前关节角: {text}")
                return True
        self.get_logger().error("超时：没有收到完整 /joint_states")
        return False

    def send_move_group_goal(self, req: MotionPlanRequest, name: str) -> bool:
        goal_msg = MoveGroup.Goal()
        goal_msg.request = req

        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.look_around = False

        # 避免失败后反复重规划，看起来像一直在动。
        goal_msg.planning_options.replan = False
        goal_msg.planning_options.replan_attempts = 0

        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().error(f"{name} 目标发送失败")
            return False

        if not goal_handle.accepted:
            self.get_logger().error(f"{name} 目标被 MoveIt 拒绝")
            return False

        self.get_logger().info(f"{name} 目标已接受，等待执行完成...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if result.error_code.val == 1:
            self.get_logger().info(f"{name} 执行完成")
            return True

        self.get_logger().error(f"{name} 执行失败，MoveIt error_code={result.error_code.val}")
        return False

    def pause(self, seconds: float, reason: str):
        self.get_logger().info(f"{reason}：停留 {seconds:.1f} 秒")
        time.sleep(float(seconds))

    def move_to_joint_positions(self, positions, name: str, tolerance: float = 0.015) -> bool:
        self.get_logger().info(f"准备移动到关节姿态：{name}")

        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 20
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.05
        req.max_acceleration_scaling_factor = 0.05
        req.start_state.is_diff = True

        constraints = Constraints()
        constraints.name = name

        for joint_name, joint_pos in zip(self.joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_pos)
            jc.tolerance_above = float(tolerance)
            jc.tolerance_below = float(tolerance)
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req.goal_constraints.append(constraints)
        return self.send_move_group_goal(req, name)

    def add_near_current_path_constraints(self, req: MotionPlanRequest, name: str):
        if not self.use_near_current_joint_path_constraint:
            return

        if not all(joint in self.latest_joints for joint in self.joint_names):
            self.get_logger().warn("没有完整当前关节角，跳过 near-current path constraints")
            return

        pc = Constraints()
        pc.name = f"near_current_{name}"

        for joint_name in self.joint_names:
            current = float(self.latest_joints[joint_name])
            tol = float(self.near_current_tolerance[joint_name])

            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = current
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            pc.joint_constraints.append(jc)

        req.path_constraints = pc
        self.get_logger().info(
            "已启用关节防大幅跳变约束: "
            + ", ".join(
                f"{jn}=±{self.near_current_tolerance[jn]:.2f}" for jn in self.joint_names
            )
        )

    def move_tool0_to_xyz(self, x: float, y: float, z: float, name: str, goal_tolerance: float) -> bool:
        # 在每段规划前刷新一下当前关节角，作为 IK 分支约束的中心。
        rclpy.spin_once(self, timeout_sec=0.05)

        radius = float(max(goal_tolerance, 0.015))

        self.get_logger().info(
            f"准备移动 tool0 到 {name}: "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            f"radius={radius:.3f}, "
            f"orientation_constraint={self.use_orientation_constraint}, "
            f"near_current_constraint={self.use_near_current_joint_path_constraint}"
        )

        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 40
        req.allowed_planning_time = 12.0
        req.max_velocity_scaling_factor = 0.04
        req.max_acceleration_scaling_factor = 0.04
        req.start_state.is_diff = True

        # 关键：限制每一段规划不要跳到另一组 IK 解。
        self.add_near_current_path_constraints(req, name)

        constraints = Constraints()
        constraints.name = name

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_frame
        pos_constraint.link_name = self.ee_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [radius]

        target_pose = Pose()
        target_pose.position.x = float(x)
        target_pose.position.y = float(y)
        target_pose.position.z = float(z)
        target_pose.orientation.w = 1.0

        bv = BoundingVolume()
        bv.primitives.append(sphere)
        bv.primitive_poses.append(target_pose)

        pos_constraint.constraint_region = bv
        pos_constraint.weight = 1.0
        constraints.position_constraints.append(pos_constraint)

        if self.use_orientation_constraint:
            ori_constraint = OrientationConstraint()
            ori_constraint.header.frame_id = self.base_frame
            ori_constraint.link_name = self.ee_link
            ori_constraint.orientation = self.tool0_down_orientation

            # 先不要太紧，否则可能规划失败；稳定后可以逐步缩小到 0.10~0.20。
            ori_constraint.absolute_x_axis_tolerance = 0.20
            ori_constraint.absolute_y_axis_tolerance = 0.20
            ori_constraint.absolute_z_axis_tolerance = 0.25
            ori_constraint.weight = 1.0
            constraints.orientation_constraints.append(ori_constraint)

        req.goal_constraints.append(constraints)
        return self.send_move_group_goal(req, name)


def main():
    rclpy.init()

    scene_cfg = get_scene()
    material = scene_cfg["material_box"]
    packing = scene_cfg["packing_box"]
    test = scene_cfg["test"]

    safe_x = test["safe_x"]
    safe_y = test["safe_y"]
    safe_z = test["safe_z"]

    above_clearance = test["above_clearance"]
    goal_tolerance = test["goal_tolerance"]

    material_above_x = material["x"]
    material_above_y = material["y"]
    material_above_z = material["z"] + material["height"] + above_clearance

    packing_above_x = packing["x"]
    packing_above_y = packing["y"]
    packing_above_z = packing["z"] + packing["height"] + above_clearance

    node = Tool0NoFlipTester()
    node.wait_for_moveit()

    if not node.wait_for_joint_states():
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("============================================================")
    node.get_logger().info("开始严格顺序点位测试 V2：防 IK 大幅跳变")
    node.get_logger().info(f"零点关节: {node.zero_positions}")
    node.get_logger().info(f"安全点: x={safe_x}, y={safe_y}, z={safe_z}")
    node.get_logger().info(
        f"物料箱上方: x={material_above_x}, y={material_above_y}, z={material_above_z}"
    )
    node.get_logger().info(
        f"包装箱上方: x={packing_above_x}, y={packing_above_y}, z={packing_above_z}"
    )
    node.get_logger().info("============================================================")

    try:
        # 0. 从零点开始。
        # 如果零点本身不安全，可以先注释这一段，改成从当前姿态直接去安全点。
        if not node.move_to_joint_positions(node.zero_positions, "zero_joint_home"):
            node.get_logger().error("无法移动到零点，停止测试。")
            return

        node.pause(1.0, "零点到达")

        # 1. 零点 -> 安全点
        if not node.move_tool0_to_xyz(safe_x, safe_y, safe_z, "safe_high_point_1", goal_tolerance):
            return
        node.pause(1.0, "安全点到达")

        # 2. 安全点 -> 物料箱上方
        if not node.move_tool0_to_xyz(material_above_x, material_above_y, material_above_z, "material_box_above_high", goal_tolerance):
            return
        node.pause(3.0, "物料箱上方到达")

        # 3. 物料箱上方 -> 安全点
        if not node.move_tool0_to_xyz(safe_x, safe_y, safe_z, "safe_high_point_2", goal_tolerance):
            return
        node.pause(1.0, "回到安全点")

        # 4. 安全点 -> 包装箱上方
        if not node.move_tool0_to_xyz(packing_above_x, packing_above_y, packing_above_z, "packing_box_above_high", goal_tolerance):
            return
        node.pause(3.0, "包装箱上方到达")

        # 5. 包装箱上方 -> 安全点
        node.move_tool0_to_xyz(safe_x, safe_y, safe_z, "safe_high_point_3", goal_tolerance)

        node.get_logger().info("严格顺序点位测试 V2 完成。")

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

