#!/usr/bin/env python3
import math
import random
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_bool(text: str) -> str:
    return "true" if str(text).lower() in ("1", "true", "yes", "on") else "false"


def _rand_color(rng: random.Random):
    # 柔和但明显的随机颜色
    return (
        round(rng.uniform(0.2, 0.95), 3),
        round(rng.uniform(0.2, 0.95), 3),
        round(rng.uniform(0.2, 0.95), 3),
    )


def _build_box_item_sdf(
    name: str,
    length: float,
    width: float,
    height: float,
    mass: float,
    color_rgb,
) -> str:
    # 简单长方体的惯性矩
    ixx = (1.0 / 12.0) * mass * (width * width + height * height)
    iyy = (1.0 / 12.0) * mass * (length * length + height * height)
    izz = (1.0 / 12.0) * mass * (length * length + width * width)

    r, g, b = color_rgb

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{name}">
    <static>false</static>
    <allow_auto_disable>false</allow_auto_disable>

    <link name="body">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>{iyy}</iyy>
          <iyz>0.0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>

      <collision name="collision">
        <geometry>
          <box>
            <size>{length} {width} {height}</size>
          </box>
        </geometry>
        <surface>
          <friction>
            <ode>
              <mu>0.9</mu>
              <mu2>0.9</mu2>
            </ode>
          </friction>
          <bounce>
            <restitution_coefficient>0.02</restitution_coefficient>
          </bounce>
        </surface>
      </collision>

      <visual name="visual">
        <geometry>
          <box>
            <size>{length} {width} {height}</size>
          </box>
        </geometry>
        <material>
          <ambient>{r} {g} {b} 1.0</ambient>
          <diffuse>{r} {g} {b} 1.0</diffuse>
          <specular>0.15 0.15 0.15 1.0</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
    return sdf


def launch_setup(context, *args, **kwargs):
    # 箱体参数（与你当前空心箱参数保持一致）
    box_x = float(LaunchConfiguration("box_x").perform(context))
    box_y = float(LaunchConfiguration("box_y").perform(context))
    box_z = float(LaunchConfiguration("box_z").perform(context))
    box_length = float(LaunchConfiguration("box_length").perform(context))
    box_width = float(LaunchConfiguration("box_width").perform(context))
    box_height = float(LaunchConfiguration("box_height").perform(context))
    wall_thickness = float(LaunchConfiguration("wall_thickness").perform(context))

    # 随机物体参数
    item_count = int(LaunchConfiguration("item_count").perform(context))
    seed = int(LaunchConfiguration("seed").perform(context))
    allow_renaming = _parse_bool(LaunchConfiguration("allow_renaming").perform(context))

    min_length = float(LaunchConfiguration("min_length").perform(context))
    max_length = float(LaunchConfiguration("max_length").perform(context))
    min_width = float(LaunchConfiguration("min_width").perform(context))
    max_width = float(LaunchConfiguration("max_width").perform(context))
    min_height = float(LaunchConfiguration("min_height").perform(context))
    max_height = float(LaunchConfiguration("max_height").perform(context))

    mass_per_item = float(LaunchConfiguration("mass_per_item").perform(context))
    edge_margin = float(LaunchConfiguration("edge_margin").perform(context))
    spawn_gap = float(LaunchConfiguration("spawn_gap").perform(context))

    # 基本合法性检查
    if item_count <= 0:
        raise RuntimeError("item_count 必须大于 0。")

    if box_length <= 0 or box_width <= 0 or box_height <= 0:
        raise RuntimeError("箱体尺寸必须大于 0。")

    if wall_thickness <= 0:
        raise RuntimeError("wall_thickness 必须大于 0。")

    inner_length = box_length - 2.0 * wall_thickness
    inner_width = box_width - 2.0 * wall_thickness
    inner_height = box_height - wall_thickness

    if inner_length <= 0 or inner_width <= 0 or inner_height <= 0:
        raise RuntimeError("箱体内腔尺寸无效，请检查箱体尺寸和壁厚。")

    if min_length > max_length or min_width > max_width or min_height > max_height:
        raise RuntimeError("随机物体尺寸范围非法：min 不能大于 max。")

    # 随机物体尺寸不能大于箱体内部
    usable_length = inner_length - 2.0 * edge_margin
    usable_width = inner_width - 2.0 * edge_margin

    if usable_length <= 0 or usable_width <= 0:
        raise RuntimeError("edge_margin 过大，导致箱内没有可用摆放区域。")

    if min_length >= usable_length or min_width >= usable_width:
        raise RuntimeError("最小物体尺寸已经大于箱内可用区域，请减小物体尺寸或减小 edge_margin。")

    rng = random.Random(seed)

    actions = [
        LogInfo(
            msg=(
                f"[spawn_random_items_in_box] box center(bottom)=({box_x}, {box_y}, {box_z}), "
                f"box outer size=({box_length}, {box_width}, {box_height}), "
                f"inner size≈({inner_length:.3f}, {inner_width:.3f}, {inner_height:.3f})"
            )
        ),
        LogInfo(
            msg=(
                f"[spawn_random_items_in_box] item_count={item_count}, seed={seed}, "
                f"size ranges L[{min_length}, {max_length}] "
                f"W[{min_width}, {max_width}] "
                f"H[{min_height}, {max_height}]"
            )
        ),
    ]

    current_spawn_z = box_z + box_height + spawn_gap

    for i in range(item_count):
        length = round(rng.uniform(min_length, min(max_length, usable_length)), 4)
        width = round(rng.uniform(min_width, min(max_width, usable_width)), 4)
        height = round(rng.uniform(min_height, max_height), 4)

        # 保证物体中心放在箱内有效范围
        x_low = box_x - inner_length / 2.0 + edge_margin + length / 2.0
        x_high = box_x + inner_length / 2.0 - edge_margin - length / 2.0
        y_low = box_y - inner_width / 2.0 + edge_margin + width / 2.0
        y_high = box_y + inner_width / 2.0 - edge_margin - width / 2.0

        if x_low > x_high or y_low > y_high:
            raise RuntimeError(
                f"第 {i} 个物体尺寸过大，无法放入箱内。"
            )

        item_x = round(rng.uniform(x_low, x_high), 4)
        item_y = round(rng.uniform(y_low, y_high), 4)

        # 从箱口上方开始生成，让它自然掉落进箱子
        item_z = round(current_spawn_z + height / 2.0, 4)
        current_spawn_z += height + spawn_gap

        roll = round(rng.uniform(-0.08, 0.08), 4)
        pitch = round(rng.uniform(-0.08, 0.08), 4)
        yaw = round(rng.uniform(-math.pi, math.pi), 4)

        item_name = f"random_item_{i:03d}"
        color_rgb = _rand_color(rng)

        sdf_content = _build_box_item_sdf(
            name=item_name,
            length=length,
            width=width,
            height=height,
            mass=mass_per_item,
            color_rgb=color_rgb,
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".sdf",
            prefix=f"{item_name}_",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(sdf_content)
            tmp_sdf_path = tmp.name

        actions.append(
            LogInfo(
                msg=(
                    f"[spawn_random_items_in_box] {item_name}: "
                    f"size=({length}, {width}, {height}), "
                    f"pose=({item_x}, {item_y}, {item_z}), "
                    f"rpy=({roll}, {pitch}, {yaw}), "
                    f"sdf={tmp_sdf_path}"
                )
            )
        )

        actions.append(
            Node(
                package="ros_gz_sim",
                executable="create",
                arguments=[
                    "-file", tmp_sdf_path,
                    "-name", item_name,
                    "-x", str(item_x),
                    "-y", str(item_y),
                    "-z", str(item_z),
                    "-R", str(roll),
                    "-P", str(pitch),
                    "-Y", str(yaw),
                    "-allow_renaming", allow_renaming,
                ],
                output="screen",
            )
        )

    return actions


def generate_launch_description():
    return LaunchDescription([
        # 箱体参数（需要和 spawn_material_box.launch.py 使用的参数保持一致）
        DeclareLaunchArgument("box_x", default_value="0.8", description="箱体底部中心 X"),
        DeclareLaunchArgument("box_y", default_value="0.0", description="箱体底部中心 Y"),
        DeclareLaunchArgument("box_z", default_value="0.0", description="箱体底部中心 Z"),
        DeclareLaunchArgument("box_length", default_value="0.6", description="箱体外部长"),
        DeclareLaunchArgument("box_width", default_value="1.0", description="箱体外部宽"),
        DeclareLaunchArgument("box_height", default_value="0.3", description="箱体外部高"),
        DeclareLaunchArgument("wall_thickness", default_value="0.015", description="箱壁厚度"),

        # 物体生成参数
        DeclareLaunchArgument("item_count", default_value="6", description="随机物体数量"),
        DeclareLaunchArgument("seed", default_value="42", description="随机种子"),
        DeclareLaunchArgument("allow_renaming", default_value="true", description="允许重名自动改名"),

        # 随机尺寸范围（单位：米）
        DeclareLaunchArgument("min_length", default_value="0.05", description="最小长度"),
        DeclareLaunchArgument("max_length", default_value="0.12", description="最大长度"),
        DeclareLaunchArgument("min_width", default_value="0.04", description="最小宽度"),
        DeclareLaunchArgument("max_width", default_value="0.10", description="最大宽度"),
        DeclareLaunchArgument("min_height", default_value="0.02", description="最小高度"),
        DeclareLaunchArgument("max_height", default_value="0.06", description="最大高度"),

        # 其他参数
        DeclareLaunchArgument("mass_per_item", default_value="0.08", description="每个物体质量(kg)"),
        DeclareLaunchArgument("edge_margin", default_value="0.01", description="离箱壁最小边距"),
        DeclareLaunchArgument("spawn_gap", default_value="0.04", description="物体生成时的竖直间隔"),

        OpaqueFunction(function=launch_setup),
    ])
