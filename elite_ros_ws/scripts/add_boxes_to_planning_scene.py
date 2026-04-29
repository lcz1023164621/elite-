#!/usr/bin/env python3
import sys

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive

from scene_params import get_scene


class AddBoxesToPlanningScene(Node):
    def __init__(self):
        super().__init__("add_boxes_to_planning_scene")
        self.client = self.create_client(
            ApplyPlanningScene,
            "/apply_planning_scene",
        )

    def make_box_part(self, object_id, x, y, z, sx, sy, sz):
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.id = object_id

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [
            float(sx),
            float(sy),
            float(sz),
        ]

        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.w = 1.0

        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)
        obj.operation = CollisionObject.ADD
        return obj

    def make_remove_object(self, object_id):
        obj = CollisionObject()
        obj.header.frame_id = "base_link"
        obj.id = object_id
        obj.operation = CollisionObject.REMOVE
        return obj

    def add_ground_with_base_clearance(self, objects):
        """
        分块地面：
        1. 防止 MoveIt 规划出从地下绕行的路径；
        2. 给机器人底座附近留空，避免初始状态碰撞。
        """

        z_center = -0.04
        thickness = 0.04

        objects.append(
            self.make_box_part(
                "ground_front",
                0.35,
                0.85,
                z_center,
                2.3,
                0.7,
                thickness,
            )
        )

        objects.append(
            self.make_box_part(
                "ground_back",
                0.35,
                -0.85,
                z_center,
                2.3,
                0.7,
                thickness,
            )
        )

        objects.append(
            self.make_box_part(
                "ground_left",
                -0.75,
                0.0,
                z_center,
                0.7,
                1.0,
                thickness,
            )
        )

        objects.append(
            self.make_box_part(
                "ground_right",
                1.45,
                0.0,
                z_center,
                0.7,
                1.0,
                thickness,
            )
        )

    def add_open_top_box(
        self,
        objects,
        prefix,
        cx,
        cy,
        cz,
        length,
        width,
        height,
        thickness,
    ):
        """
        添加上开口箱体：
        - bottom
        - front_wall
        - back_wall
        - left_wall
        - right_wall
        """

        bottom_z = cz + thickness / 2.0
        wall_z = cz + height / 2.0

        objects.append(
            self.make_box_part(
                f"{prefix}_bottom",
                cx,
                cy,
                bottom_z,
                length,
                width,
                thickness,
            )
        )

        objects.append(
            self.make_box_part(
                f"{prefix}_front_wall",
                cx,
                cy + width / 2.0 - thickness / 2.0,
                wall_z,
                length,
                thickness,
                height,
            )
        )

        objects.append(
            self.make_box_part(
                f"{prefix}_back_wall",
                cx,
                cy - width / 2.0 + thickness / 2.0,
                wall_z,
                length,
                thickness,
                height,
            )
        )

        objects.append(
            self.make_box_part(
                f"{prefix}_left_wall",
                cx - length / 2.0 + thickness / 2.0,
                cy,
                wall_z,
                thickness,
                width,
                height,
            )
        )

        objects.append(
            self.make_box_part(
                f"{prefix}_right_wall",
                cx + length / 2.0 - thickness / 2.0,
                cy,
                wall_z,
                thickness,
                width,
                height,
            )
        )

    def remove_old_objects(self):
        old_ids = [
            "ground_plane",
            "ground_front",
            "ground_back",
            "ground_left",
            "ground_right",
            "material_box_bottom",
            "material_box_front_wall",
            "material_box_back_wall",
            "material_box_left_wall",
            "material_box_right_wall",
            "packing_box_bottom",
            "packing_box_front_wall",
            "packing_box_back_wall",
            "packing_box_left_wall",
            "packing_box_right_wall",
        ]

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects = [
            self.make_remove_object(object_id) for object_id in old_ids
        ]

        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        self.get_logger().info("正在清除旧的 Planning Scene 碰撞体...")
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()
        if result is None or not result.success:
            self.get_logger().warn("清除旧碰撞体可能失败，继续添加新碰撞体。")
        else:
            self.get_logger().info("旧碰撞体清除完成。")

    def apply_scene(self):
        self.get_logger().info("等待 /apply_planning_scene 服务...")

        if not self.client.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                "没有找到 /apply_planning_scene 服务，请确认 move_group 已启动。"
            )
            return False

        self.remove_old_objects()

        scene_cfg = get_scene()
        material = scene_cfg["material_box"]
        packing = scene_cfg["packing_box"]

        objects = []

        self.add_ground_with_base_clearance(objects)

        self.add_open_top_box(
            objects=objects,
            prefix="material_box",
            cx=material["x"],
            cy=material["y"],
            cz=material["z"],
            length=material["length"],
            width=material["width"],
            height=material["height"],
            thickness=material["wall_thickness"],
        )

        self.add_open_top_box(
            objects=objects,
            prefix="packing_box",
            cx=packing["x"],
            cy=packing["y"],
            cz=packing["z"],
            length=packing["length"],
            width=packing["width"],
            height=packing["height"],
            thickness=packing["wall_thickness"],
        )

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects = objects

        request = ApplyPlanningScene.Request()
        request.scene = planning_scene

        self.get_logger().info("正在向 MoveIt Planning Scene 添加新碰撞体...")
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        result = future.result()

        if result is None:
            self.get_logger().error("调用 /apply_planning_scene 失败，没有返回结果。")
            return False

        if not result.success:
            self.get_logger().error("MoveIt 返回失败，Planning Scene 没有更新成功。")
            return False

        self.get_logger().info("成功添加地面、物料箱、包装箱到 MoveIt Planning Scene。")
        self.get_logger().info("添加的碰撞体如下：")

        for obj in objects:
            self.get_logger().info(f"  - {obj.id}")

        self.get_logger().info(
            f"物料箱: center=({material['x']}, {material['y']}, {material['z']}), "
            f"size=({material['length']}, {material['width']}, {material['height']})"
        )

        self.get_logger().info(
            f"包装箱: spec={packing['spec']}, "
            f"center=({packing['x']}, {packing['y']}, {packing['z']}), "
            f"size=({packing['length']}, {packing['width']}, {packing['height']})"
        )

        return True


def main():
    rclpy.init()

    node = AddBoxesToPlanningScene()
    ok = node.apply_scene()

    node.destroy_node()
    rclpy.shutdown()

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
