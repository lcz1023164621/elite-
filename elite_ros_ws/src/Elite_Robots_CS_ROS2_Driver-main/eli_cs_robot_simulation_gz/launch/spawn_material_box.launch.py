#!/usr/bin/env python3
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _parse_bool(text: str) -> str:
    return "true" if str(text).lower() in ("1", "true", "yes", "on") else "false"


def _build_hollow_box_sdf(
    box_name: str,
    length: float,
    width: float,
    height: float,
    thickness: float,
    alpha: float,
) -> str:
    if length <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("box_length / box_width / box_height 必须都大于 0。")

    if thickness <= 0:
        raise RuntimeError("wall_thickness 必须大于 0。")

    if thickness * 2 >= length:
        raise RuntimeError("wall_thickness 过大，必须满足 2 * wall_thickness < box_length。")

    if thickness * 2 >= width:
        raise RuntimeError("wall_thickness 过大，必须满足 2 * wall_thickness < box_width。")

    if not (0.0 <= alpha <= 1.0):
        raise RuntimeError("wall_alpha 必须在 [0, 1] 范围内。")

    # 模型原点定义在箱体底部中心
    # 这样传入的 box_z 就是“箱底高度”
    bottom_z = thickness / 2.0
    wall_z = height / 2.0

    front_back_y = width / 2.0 - thickness / 2.0
    left_right_x = length / 2.0 - thickness / 2.0

    # 左右侧壁长度减去前后壁厚度，避免过度重叠
    side_wall_width = max(width - 2.0 * thickness, 1e-6)

    blue_r = 0.10
    blue_g = 0.40
    blue_b = 0.85

    material_block = f"""
          <material>
            <ambient>{blue_r} {blue_g} {blue_b} {alpha}</ambient>
            <diffuse>{blue_r} {blue_g} {blue_b} {alpha}</diffuse>
            <specular>0.15 0.15 0.15 1.0</specular>
          </material>"""

    sdf = f"""<?xml version="1.0"?>
<sdf version="1.8">
  <model name="{box_name}">
    <static>true</static>
    <self_collide>false</self_collide>

    <link name="box_link">

      <!-- 底板 -->
      <visual name="bottom_visual">
        <pose>0 0 {bottom_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {width} {thickness}</size>
          </box>
        </geometry>
{material_block}
      </visual>

      <collision name="bottom_collision">
        <pose>0 0 {bottom_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {width} {thickness}</size>
          </box>
        </geometry>
      </collision>

      <!-- 前壁 -->
      <visual name="front_visual">
        <pose>0 {front_back_y} {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
{material_block}
      </visual>

      <collision name="front_collision">
        <pose>0 {front_back_y} {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
      </collision>

      <!-- 后壁 -->
      <visual name="back_visual">
        <pose>0 {-front_back_y} {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
{material_block}
      </visual>

      <collision name="back_collision">
        <pose>0 {-front_back_y} {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{length} {thickness} {height}</size>
          </box>
        </geometry>
      </collision>

      <!-- 左壁 -->
      <visual name="left_visual">
        <pose>{-left_right_x} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{thickness} {side_wall_width} {height}</size>
          </box>
        </geometry>
{material_block}
      </visual>

      <collision name="left_collision">
        <pose>{-left_right_x} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{thickness} {side_wall_width} {height}</size>
          </box>
        </geometry>
      </collision>

      <!-- 右壁 -->
      <visual name="right_visual">
        <pose>{left_right_x} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{thickness} {side_wall_width} {height}</size>
          </box>
        </geometry>
{material_block}
      </visual>

      <collision name="right_collision">
        <pose>{left_right_x} 0 {wall_z} 0 0 0</pose>
        <geometry>
          <box>
            <size>{thickness} {side_wall_width} {height}</size>
          </box>
        </geometry>
      </collision>

    </link>
  </model>
</sdf>
"""
    return sdf


def launch_setup(context, *args, **kwargs):
    box_name = LaunchConfiguration("box_name").perform(context)
    box_x = float(LaunchConfiguration("box_x").perform(context))
    box_y = float(LaunchConfiguration("box_y").perform(context))
    box_z = float(LaunchConfiguration("box_z").perform(context))

    box_length = float(LaunchConfiguration("box_length").perform(context))
    box_width = float(LaunchConfiguration("box_width").perform(context))
    box_height = float(LaunchConfiguration("box_height").perform(context))

    wall_thickness = float(LaunchConfiguration("wall_thickness").perform(context))
    wall_alpha = float(LaunchConfiguration("wall_alpha").perform(context))
    allow_renaming = _parse_bool(LaunchConfiguration("allow_renaming").perform(context))

    sdf_content = _build_hollow_box_sdf(
        box_name=box_name,
        length=box_length,
        width=box_width,
        height=box_height,
        thickness=wall_thickness,
        alpha=wall_alpha,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".sdf",
        prefix=f"{box_name}_",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(sdf_content)
        tmp_sdf_path = tmp.name

    spawn_box = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-file", tmp_sdf_path,
            "-name", box_name,
            "-x", str(box_x),
            "-y", str(box_y),
            "-z", str(box_z),
            "-allow_renaming", allow_renaming,
        ],
        output="screen",
    )

    return [
        LogInfo(msg=f"[spawn_material_box] 使用临时 SDF: {tmp_sdf_path}"),
        LogInfo(
            msg=(
                f"[spawn_material_box] name={box_name}, "
                f"size=({box_length}, {box_width}, {box_height}), "
                f"thickness={wall_thickness}, "
                f"pose=({box_x}, {box_y}, {box_z})"
            )
        ),
        spawn_box,
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "box_name",
            default_value="material_box",
            description="生成到 Gazebo 中的箱体名称"
        ),
        DeclareLaunchArgument(
            "box_x",
            default_value="0.5",
            description="箱体底部中心的 X 坐标"
        ),
        DeclareLaunchArgument(
            "box_y",
            default_value="0.0",
            description="箱体底部中心的 Y 坐标"
        ),
        DeclareLaunchArgument(
            "box_z",
            default_value="0.0",
            description="箱体底部中心的 Z 坐标"
        ),
        DeclareLaunchArgument(
            "box_length",
            default_value="0.6",
            description="箱体外部长（X 方向）"
        ),
        DeclareLaunchArgument(
            "box_width",
            default_value="0.4",
            description="箱体外部宽（Y 方向）"
        ),
        DeclareLaunchArgument(
            "box_height",
            default_value="0.3",
            description="箱体外部高（Z 方向）"
        ),
        DeclareLaunchArgument(
            "wall_thickness",
            default_value="0.02",
            description="箱壁厚度"
        ),
        DeclareLaunchArgument(
            "wall_alpha",
            default_value="0.35",
            description="箱壁透明度 alpha，范围 [0,1]"
        ),
        DeclareLaunchArgument(
            "allow_renaming",
            default_value="true",
            description="同名实体已存在时是否允许 Gazebo 自动重命名"
        ),
        OpaqueFunction(function=launch_setup),
    ])
