#!/usr/bin/env bash
# 将 cs612_moveit_config 的 Python 包安装到 colcon 前缀下（供 lib/.../ 启动脚本 import）。
# 在部分环境下 colcon build 的 ament_python 步骤会失败，导致 install/.../site-packages 为空。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PKG="${WS}/cs612_moveit_config"
PREFIX="${WS}/install/cs612_moveit_config"
if [[ ! -f "${PKG}/setup.py" ]]; then
  echo "找不到 ${PKG}/setup.py，请在工作空间根目录旁保留 cs612_moveit_config 包。" >&2
  exit 1
fi
python3 -m pip install --no-build-isolation --no-deps --prefix "${PREFIX}" "${PKG}"
echo "已安装到 ${PREFIX}/lib/python*/site-packages；请 source ${WS}/install/setup.bash"
