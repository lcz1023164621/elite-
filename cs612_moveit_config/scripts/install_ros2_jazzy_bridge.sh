#!/usr/bin/env bash
# 已弃用：本项目现在统一使用 conda ros2 环境（RoboStack）自带的 ros_gz_bridge。
# 不再依赖系统 ROS 2（包括 Jazzy 或 Humble）的 ros_gz_bridge。
# 请确保已激活 conda ros2 环境并重新 colcon build：
#   conda activate ros2
#   colcon build --symlink-install
#   source install/setup.bash
set -euo pipefail
echo "[INFO] 本脚本已弃用。项目现在统一使用 conda ros2 环境的 ros_gz_bridge。"
echo "[INFO] 请执行：conda activate ros2 && colcon build --symlink-install && source install/setup.bash"
