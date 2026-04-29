#!/usr/bin/env bash
# Source this file from a shell to set up the ROS2 workspace.
# Uses system ROS2 Humble (apt-installed), not conda.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source scripts/source_workspace.sh" >&2
  exit 2
fi

_ws_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Avoid conda Python (e.g. 3.13) shadowing ROS Humble Python 3.10 runtime.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  if declare -F conda >/dev/null 2>&1; then
    conda deactivate >/dev/null 2>&1 || true
  fi
fi
unset PYTHONHOME PYTHONPATH
export PATH="/usr/bin:/bin:/usr/local/bin:${PATH}"

# Source system ROS2 Humble
if [[ -f /opt/ros/humble/setup.bash ]]; then
  source /opt/ros/humble/setup.bash
else
  echo "ERROR: /opt/ros/humble/setup.bash not found. Install ROS2 Humble first." >&2
  return 1
fi

# Source the main workspace
if [[ -f "${_ws_root}/install/setup.bash" ]]; then
  source "${_ws_root}/install/setup.bash"
fi

# Source the elite_ros_ws workspace (overlay)
if [[ -f "${_ws_root}/elite_ros_ws/install/setup.bash" ]]; then
  source "${_ws_root}/elite_ros_ws/install/setup.bash"
fi

unset _ws_root