#!/bin/bash
# 一键启动Gazebo仿真 + 自动抓取功能（CS612，使用系统 ROS2 Humble）

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== 启动CS612机械臂自动抓取系统 ==="
echo "1. 启动Gazebo仿真环境..."
echo "2. 启动MoveIt运动规划..."
echo "3. 8秒后自动开始抓取任务..."
echo ""

# 设置环境。Gazebo/bridge/MoveIt 都由 bringup.launch.py 统一启动，
# 避免脚本先起一个 gz sim 后 launch 再起第二个仿真实例。
export GZ_SIM_RESOURCE_PATH=$PROJECT_DIR/arms_models:$PROJECT_DIR/my_arms:$PROJECT_DIR/models:$PROJECT_DIR/models/gazebo_models:$PROJECT_DIR/worlds:${GZ_SIM_RESOURCE_PATH}
unset WAYLAND_DISPLAY
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

# Source system ROS2 Humble + workspace
cd "$PROJECT_DIR"
source scripts/source_workspace.sh

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$PROJECT_DIR/src/elite_moveit_ec612/config/fastdds_cs612.xml"
export RMW_FASTRTPS_USE_QOS_FROM_XML=1

echo ""
echo "=== 启动自动抓取程序 ==="
exec ros2 launch cs612_moveit_config full_auto_pick.launch.py use_sim_time:=true