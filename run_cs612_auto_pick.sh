#!/usr/bin/env bash
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source system ROS2 Humble + workspaces
source /opt/ros/humble/setup.bash
source "$ROOT_DIR/elite_ros_ws/install/setup.bash"
source "$ROOT_DIR/install/setup.bash"

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$ROOT_DIR/src/elite_moveit_ec612/config/fastdds_cs612.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

exec ros2 launch "$ROOT_DIR/install/elite_moveit_ec612/share/elite_moveit_ec612/launch/bringup_ros2_control.launch.py" auto_pick:=true