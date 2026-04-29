#!/usr/bin/env python3
"""Compatibility entrypoint for the refactored CS612 simulation bringup.

The original ros2_control path now depends on an `ign_ros2_control` runtime
plugin that is not available in this workspace. Route this launch file to the
working Gazebo + bridge bringup so existing user commands keep working.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    rmw_implementation = LaunchConfiguration("rmw_implementation")
    auto_pick = LaunchConfiguration("auto_pick")

    legacy_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [FindPackageShare("elite_moveit_ec612"), "launch", "bringup.launch.py"]
            )
        ),
        launch_arguments={
            "use_sim_time": use_sim_time,
            "rmw_implementation": rmw_implementation,
            "auto_pick": auto_pick,
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("rmw_implementation", default_value="rmw_fastrtps_cpp"),
            DeclareLaunchArgument("auto_pick", default_value="true"),
            legacy_bringup,
        ]
    )
