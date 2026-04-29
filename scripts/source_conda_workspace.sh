#!/usr/bin/env bash
# Source this file from a shell to run the workspace with the conda ros2 env.
#
# It intentionally removes any system ROS underlay from the current shell first.
# Otherwise conda's ros2 may import system ROS Python modules and fail with
# rclpy ABI errors such as "undefined symbol: rcutils_log_internal".

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed:" >&2
  echo "  source scripts/source_conda_workspace.sh" >&2
  exit 2
fi

_cs612_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_cs612_strip_ros_paths() {
  local var_name="$1"
  local old_value="${!var_name:-}"
  local new_value=""
  local item

  IFS=':' read -ra _cs612_items <<< "${old_value}"
  for item in "${_cs612_items[@]}"; do
    [[ -z "${item}" ]] && continue
    case "${item}" in
      /opt/ros/*) continue ;;
    esac
    if [[ -z "${new_value}" ]]; then
      new_value="${item}"
    else
      new_value="${new_value}:${item}"
    fi
  done
  export "${var_name}=${new_value}"
}

for _cs612_var in PYTHONPATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH LD_LIBRARY_PATH; do
  _cs612_strip_ros_paths "${_cs612_var}"
done
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_ROOT ROS_PACKAGE_PATH

# 清除可能残留的 FastDDS/CycloneDDS 配置文件路径，避免旧配置导致 DDS init 失败
unset FASTRTPS_DEFAULT_PROFILES_FILE CYCLONEDDS_URI

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate ros2

source "${_cs612_root}/install/local_setup.bash"

unset _cs612_var _cs612_root
