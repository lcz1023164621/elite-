#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,TimerAction  
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def launch_setup(context, *args, **kwargs):
    # Initialize Arguments
    cs_type = LaunchConfiguration("cs_type")
    safety_limits = LaunchConfiguration("safety_limits")
    # General arguments
    runtime_config_package = LaunchConfiguration("runtime_config_package")
    controllers_file = LaunchConfiguration("controllers_file")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    moveit_config_file = LaunchConfiguration("moveit_config_file")
    prefix = LaunchConfiguration("prefix")

    # ===== 原有 launch 文件 =====
    cs_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("eli_cs_robot_simulation_gz"), "/launch", "/cs_sim_control.launch.py"]
        ),
        launch_arguments={
            "cs_type": cs_type,
            "safety_limits": safety_limits,
            "runtime_config_package": runtime_config_package,
            "controllers_file": controllers_file,
            "description_package": description_package,
            "description_file": description_file,
            "prefix": prefix,
            "launch_rviz": "false",
        }.items(),
    )

    cs_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("eli_cs_robot_moveit_config"), "/launch", "/cs_moveit.launch.py"]
        ),
        launch_arguments={
            "cs_type": cs_type,
            "safety_limits": safety_limits,
            "description_package": description_package,
            "description_file": description_file,
            "moveit_config_package": moveit_config_package,
            "moveit_config_file": moveit_config_file,
            "prefix": prefix,
            "use_sim_time": "true",
            "launch_rviz": "true",
        }.items(),
    )

    # ===== 新增：物料框启动（放在 launch_setup 内部）=====
    spawn_box = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('eli_cs_robot_simulation_gz'),
                'launch',
                'spawn_material_box.launch.py'
            )
        ]),
        launch_arguments={
            'box_x': LaunchConfiguration('box_x'),
            'box_y': LaunchConfiguration('box_y'),
            'box_z': LaunchConfiguration('box_z'),
            'box_length': LaunchConfiguration('box_length'),
            'box_width': LaunchConfiguration('box_width'),
            'box_height': LaunchConfiguration('box_height'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('spawn_box'))
    )
    
    spawn_box_delayed = TimerAction(
    	period=12.0,
    	actions=[spawn_box]
	)

    # ===== 返回所有节点 =====
    nodes_to_launch = [
        cs_control_launch,
        cs_moveit_launch,
        spawn_box_delayed,  # ← 添加到这里
    ]

    return nodes_to_launch


def generate_launch_description():
    declared_arguments = []
    # CS specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "cs_type",
            description="Type/series of used ELITE CS robot.",
            choices=["cs63", "cs66", "cs612", "cs616", "cs620", "cs625"],
            default_value="cs66",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "runtime_config_package",
            default_value="eli_cs_robot_simulation_gz",
            description='Package with the controller\'s configuration in "config" folder. \
        Usually the argument is not set, it enables use of a custom setup.',
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "controllers_file",
            default_value="cs_controllers.yaml",
            description="YAML file with the controllers configuration.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="eli_cs_robot_description",
            description="Description package with robot URDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="cs.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="eli_cs_robot_moveit_config",
            description="MoveIt config package with robot SRDF/XACRO files. Usually the argument \
        is not set, it enables use of a custom moveit config.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_file",
            default_value="cs.srdf.xacro",
            description="MoveIt SRDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "prefix",
            default_value='""',
            description="Prefix of the joint names, useful for \
        multi-robot setup. If changed than also joint names in the controllers' configuration \
        have to be updated.",
        )
    )
    
    # ===== 物料框相关参数 =====
    declared_arguments.append(
        DeclareLaunchArgument(
            "spawn_box",
            default_value="true",
            description="Whether to spawn the material box in simulation.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_x",
            default_value="0.5",
            description="Material box X position.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_y",
            default_value="0.0",
            description="Material box Y position.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_z",
            default_value="0.15",
            description="Material box Z position (bottom on ground).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_length",
            default_value="0.6",
            description="Material box length (X dimension).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_width",
            default_value="0.4",
            description="Material box width (Y dimension).",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "box_height",
            default_value="0.3",
            description="Material box height (Z dimension).",
        )
    )

    # 简化：只返回 declared_arguments + OpaqueFunction
    return LaunchDescription(
        declared_arguments + [OpaqueFunction(function=launch_setup)]
    )
