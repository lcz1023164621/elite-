#!/usr/bin/env bash
# 在终端直接运行 ign gazebo 时，避免从已激活的环境继承 LD_LIBRARY_PATH 导致崩溃。
# 本脚本用最小环境启动系统 /usr/bin/ign。
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GZ_SIM_RESOURCE_PATH="${ROOT}/arms_models:${ROOT}/my_arms:${ROOT}/worlds"
export PATH="/usr/bin:/bin:/usr/local/bin"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
export HOME="${HOME:?}"
export USER="${USER:-}"
export LANG="${LANG:-C.UTF-8}"
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-}"
export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export GALLIUM_DRIVER="${GALLIUM_DRIVER:-llvmpipe}"
unset PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_DISTRO 2>/dev/null || true
if [[ $# -eq 0 ]]; then
  set -- -r "${ROOT}/worlds/my_world.sdf"
fi
exec /usr/bin/ign gazebo "$@"