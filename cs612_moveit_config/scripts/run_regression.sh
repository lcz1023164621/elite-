#!/usr/bin/env bash
set -euo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WS_ROOT}"

source "${WS_ROOT}/scripts/source_workspace.sh"

echo "[1/2] 校验 scene_objects 与 world 一致性"
python3 "${WS_ROOT}/cs612_moveit_config/scripts/cs612_scene_alignment_check"

echo "[2/2] 启动后执行最小回归检查"
ros2 launch cs612_moveit_config full_auto_pick.launch.py use_sim_time:=true > /tmp/cs612_reg_launch.log 2>&1 &
LPID=$!
sleep 20
PYTHONPATH="${WS_ROOT}/cs612_moveit_config:${PYTHONPATH:-}" /usr/bin/python3 \
  "${WS_ROOT}/cs612_moveit_config/scripts/cs612_regression_check"
RC=$?
kill -INT "${LPID}" >/dev/null 2>&1 || true
wait "${LPID}" >/dev/null 2>&1 || true
exit "${RC}"
