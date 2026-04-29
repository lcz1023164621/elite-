#!/usr/bin/env python3
# ============================================================
# test_tool0_points_v3_down_no_flip
#
# 目标：
# 1. 不再使用旧版 move_to_up 流程；
# 2. 按固定顺序运动：
#    当前姿态 -> 零点关节姿态 -> 安全点 -> 物料箱上方 -> 停顿
#    -> 安全点 -> 包装箱上方 -> 停顿 -> 安全点
# 3. tool0 使用朝下姿态约束；
# 4. 每段笛卡尔点位运动增加 near-current joint path constraints，
#    尽量避免机械臂突然换 IK 解、绕大圈。
# ============================================================

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, Quaternion

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


class Tool0PointsV3(Node):
    def __init__(self):
        super().__init__("test_tool0_points_v3_down_no_flip")

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

        # 真正的六轴零点。若你的机器人零点会碰撞，可以后续改成品牌推荐 home 位。
        self.zero_positions = [
            0.0,
            -1.5707,
            0.0,
            -1.5707,
            0.0,
            0.0,
        ]

        # 这是你第二次 tf2_echo 读到的 tool0 姿态，更接近末端朝下。
        self.tool0_down_orientation = Quaternion()
        self.tool0_down_orientation.x = 1.000
        self.tool0_down_orientation.y = 0.000
        self.tool0_down_orientation.z = 0.000
        self.tool0_down_orientation.w = 0.026

        # 防止突然跳到另一组 IK 解。
        # 如果某一步规划失败，优先放宽 shoulder_pan / wrist_1 / wrist_2。
        self.near_current_tolerance = {
            "shoulder_pan_joint": 1.80,
            "shoulder_lift_joint": 1.40,
            "elbow_joint": 1.60,
            "wrist_1_joint": 1.40,
            "wrist_2_joint": 1.20,
            "wrist_3_joint": 0.60,
        }

        self.current_joints = {}

        self.create_subscription(
            JointState,
            "/joint_states",
            self.joint_state_callback,
            10,
        )

    def joint_state_callback(self, msg):
        for name, pos in zip(msg.name, msg.position):
            if name in self.joint_names:
                self.current_joints[name] = float(pos)

    def wait_for_moveit(self):
        self.get_logger().info("等待 MoveIt move_action...")
        self.client.wait_for_server()
        self.get_logger().info("MoveIt move_action 已连接")

    def wait_for_joint_states(self, timeout_sec=5.0):
        start = time.time()
        while rclpy.ok():
            if all(name in self.current_joints for name in self.joint_names):
                self.get_logger().info("已读取 /joint_states")
                return True

            if time.time() - start > timeout_sec:
                self.get_logger().error(
                    "等待 /joint_states 超时，请确认 joint_state_broadcaster 正常。"
                )
                return False

            rclpy.spin_once(self, timeout_sec=0.1)

    def make_near_current_path_constraints(self, name):
        constraints = Constraints()
        constraints.name = name

        for joint_name in self.joint_names:
            if joint_name not in self.current_joints:
                continue

            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(self.current_joints[joint_name])
            tol = self.near_current_tolerance.get(joint_name, 1.5)
            jc.tolerance_above = float(tol)
            jc.tolerance_below = float(tol)
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        return constraints

    def send_move_group_goal(self, req, name):
        goal_msg = MoveGroup.Goal()
        goal_msg.request = req

        goal_msg.planning_options.plan_only = False
        goal_msg.planning_options.look_around = False

        # 先关掉自动 replan，避免失败后反复找奇怪路径。
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

        self.get_logger().info(f"{name} 目标已接受，等待执行结果...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result

        if result.error_code.val == 1:
            self.get_logger().info(f"{name} 移动成功")
            return True

        self.get_logger().error(
            f"{name} 移动失败，error_code={result.error_code.val}"
        )
        return False

    def move_to_joint_positions(self, positions, name):
        self.get_logger().info(f"准备移动到关节姿态: {name}")

        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 20
        req.allowed_planning_time = 10.0
        req.max_velocity_scaling_factor = 0.05
        req.max_acceleration_scaling_factor = 0.05
        req.start_state.is_diff = True

        goal = Constraints()
        goal.name = name

        for joint_name, joint_pos in zip(self.joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = joint_name
            jc.position = float(joint_pos)
            jc.tolerance_above = 0.02
            jc.tolerance_below = 0.02
            jc.weight = 1.0
            goal.joint_constraints.append(jc)

        req.goal_constraints.append(goal)

        return self.send_move_group_goal(req, name)

    def move_to_tool0_pose(self, x, y, z, name, goal_tolerance):
        self.get_logger().info(
            f"准备移动到 {name}: "
            f"x={x:.3f}, y={y:.3f}, z={z:.3f}, "
            f"goal_tolerance={goal_tolerance:.3f}"
        )

        req = MotionPlanRequest()
        req.group_name = self.group_name
        req.num_planning_attempts = 40
        req.allowed_planning_time = 15.0
        req.max_velocity_scaling_factor = 0.04
        req.max_acceleration_scaling_factor = 0.04
        req.start_state.is_diff = True

        # 暂时关闭防跳解约束，先验证点位本身是否可达。
        # req.path_constraints = self.make_near_current_path_constraints(
        #     f"{name}_near_current"
        # )

        goal = Constraints()
        goal.name = name

        pos_constraint = PositionConstraint()
        pos_constraint.header.frame_id = self.base_frame
        pos_constraint.link_name = self.ee_link
        pos_constraint.target_point_offset.x = 0.0
        pos_constraint.target_point_offset.y = 0.0
        pos_constraint.target_point_offset.z = 0.0

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [float(max(goal_tolerance, 0.08))]

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

        ori_constraint = OrientationConstraint()
        ori_constraint.header.frame_id = self.base_frame
        ori_constraint.link_name = self.ee_link
        ori_constraint.orientation = self.tool0_down_orientation

        # x/y 限制末端大体朝下，z 稍微放宽，避免绕自身轴导致规划失败。
        ori_constraint.absolute_x_axis_tolerance = 0.35
        ori_constraint.absolute_y_axis_tolerance = 0.35
        ori_constraint.absolute_z_axis_tolerance = 1.00
        ori_constraint.weight = 0.8

        goal.position_constraints.append(pos_constraint)
        # 暂时关闭姿态约束，先验证位置是否可达
        # goal.orientation_constraints.append(ori_constraint)
        req.goal_constraints.append(goal)

        return self.send_move_group_goal(req, name)


def main():
    rclpy.init()

    scene_cfg = get_scene()
    material = scene_cfg["material_box"]
    packing = scene_cfg["packing_box"]
    test = scene_cfg["test"]

    safe_x = test["safe_x"]
    safe_y = test["safe_y"]
    safe_z = max(test["safe_z"], 1.00)

    above_clearance = max(test["above_clearance"], 0.60)
    goal_tolerance = test["goal_tolerance"]

    material_above_x = material["x"]
    material_above_y = material["y"]
    material_above_z = material["z"] + material["height"] + above_clearance

    packing_above_x = packing["x"]
    packing_above_y = packing["y"]
    packing_above_z = packing["z"] + packing["height"] + above_clearance

    node = Tool0PointsV3()
    node.wait_for_moveit()

    if not node.wait_for_joint_states():
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("============================================================")
    node.get_logger().info("开始严格顺序点位测试 V3")
    node.get_logger().info(f"零点关节: {node.zero_positions}")
    node.get_logger().info(f"安全点: x={safe_x}, y={safe_y}, z={safe_z}")
    node.get_logger().info(
        f"物料箱上方: x={material_above_x}, "
        f"y={material_above_y}, z={material_above_z}"
    )
    node.get_logger().info(
        f"包装箱上方: x={packing_above_x}, "
        f"y={packing_above_y}, z={packing_above_z}"
    )
    node.get_logger().info("============================================================")

    # 0. 当前姿态 -> 六轴零点
    if not node.move_to_joint_positions(node.zero_positions, "joint_zero"):
        node.get_logger().error("无法移动到六轴零点，停止测试。")
        node.destroy_node()
        rclpy.shutdown()
        return

    time.sleep(1.0)

    # 1. 当前 joint_zero / up 姿态本身就是安全姿态
    node.get_logger().info("已到达竖直安全姿态，停顿 1 秒...")
    time.sleep(1.0)

    # 2. 竖直安全姿态 -> 物料箱上方
    if not node.move_to_tool0_pose(
        material_above_x,
        material_above_y,
        material_above_z,
        "material_box_above_high",
        goal_tolerance,
    ):
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("已到达物料箱上方，停顿 3 秒...")
    time.sleep(3.0)

    # 3. 物料箱上方 -> 竖直安全姿态
    if not node.move_to_joint_positions(
        node.zero_positions,
        "safe_joint_after_material",
    ):
        node.destroy_node()
        rclpy.shutdown()
        return

    time.sleep(1.0)

    # 4. 竖直安全姿态 -> 包装箱上方
    if not node.move_to_tool0_pose(
        packing_above_x,
        packing_above_y,
        packing_above_z,
        "packing_box_above_high",
        goal_tolerance,
    ):
        node.destroy_node()
        rclpy.shutdown()
        return

    node.get_logger().info("已到达包装箱上方，停顿 3 秒...")
    time.sleep(3.0)

    # 5. 包装箱上方 -> 竖直安全姿态
    node.move_to_joint_positions(
        node.zero_positions,
        "safe_joint_after_packing",
    )

    node.get_logger().info("严格顺序点位测试 V3 完成。")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
