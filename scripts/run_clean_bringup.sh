#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOMAIN_ID="${ROS_DOMAIN_ID_OVERRIDE:-66}"

pkill -9 -f "ros2 launch cs612_moveit_config bringup.launch.py" 2>/dev/null || true
pkill -9 -f "cs612_joint_states_bridge" 2>/dev/null || true
pkill -9 -f "cs612_trajectory_action_bridge" 2>/dev/null || true
pkill -9 -f "move_group" 2>/dev/null || true
pkill -9 -f "robot_state_publisher" 2>/dev/null || true
pkill -9 -f "rviz2" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "ros_gz_bridge" 2>/dev/null || true
pkill -9 -f "ign gazebo" 2>/dev/null || true
pkill -9 -f "/usr/bin/ign" 2>/dev/null || true
pkill -9 -f "ign-gazebo" 2>/dev/null || true
pkill -9 -f "gz-sim" 2>/dev/null || true
sleep 1

source /opt/ros/humble/setup.bash
source "${ROOT}/install/setup.bash"
export ROS_DOMAIN_ID="${DOMAIN_ID}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="${ROOT}/src/elite_moveit_ec612/config/fastdds_cs612.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

echo "[run_clean_bringup] ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
echo "[run_clean_bringup] starting bringup..."
exec ros2 launch cs612_moveit_config bringup.launch.py