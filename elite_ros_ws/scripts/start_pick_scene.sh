#!/usr/bin/env bash

cd ~/NZCX/elite_ros_ws
source install/setup.bash

# ============================================================
# 一键启动仿真场景
#
# 用法：
#   ./scripts/start_pick_scene.sh
#   ./scripts/start_pick_scene.sh c
#   ./scripts/start_pick_scene.sh c 42
#
# 参数：
#   第一个参数：包装箱规格，默认 c
#   第二个参数：随机种子，默认当前时间
#
# 所有箱子位置、尺寸都从 scripts/scene_params.py 读取。
# ============================================================

PACK_BOX_SPEC_ARG=${1:-c}
SEED=${2:-$(date +%s)}

# 从统一配置文件读取参数
if ! eval "$(python3 scripts/scene_params.py --bash --spec "${PACK_BOX_SPEC_ARG}")"; then
  echo "错误：读取 scripts/scene_params.py 失败"
  exit 1
fi

cleanup() {
  echo ""
  echo "正在关闭所有仿真进程..."
  jobs -pr | xargs -r kill
  wait 2>/dev/null || true
  echo "已关闭。"
}

# 注意：不要 trap EXIT，否则某个中间命令失败会把 Gazebo 杀空
trap cleanup INT TERM

echo "============================================================"
echo "启动抓取仿真场景"
echo "============================================================"
echo "物料箱位置: x=${MATERIAL_BOX_X}, y=${MATERIAL_BOX_Y}, z=${MATERIAL_BOX_Z}"
echo "物料箱尺寸: ${MATERIAL_BOX_L} x ${MATERIAL_BOX_W} x ${MATERIAL_BOX_H}"
echo "包装箱规格: ${PACK_BOX_SPEC}"
echo "包装箱位置: x=${PACK_BOX_X}, y=${PACK_BOX_Y}, z=${PACK_BOX_Z}"
echo "包装箱尺寸: ${PACK_BOX_L} x ${PACK_BOX_W} x ${PACK_BOX_H}"
echo "随机种子: ${SEED}"
echo "============================================================"

echo ""
echo "启动机械臂 + Gazebo + MoveIt + RViz..."
ros2 launch eli_cs_robot_simulation_gz cs_sim_moveit.launch.py spawn_box:=false &
SIM_PID=$!

echo "主仿真 launch PID: ${SIM_PID}"

# 等 Gazebo、MoveIt、RViz 初始化
sleep 12

echo ""
echo "生成物料箱 material_box..."
ros2 launch eli_cs_robot_simulation_gz spawn_material_box.launch.py \
  box_name:=material_box \
  box_x:=${MATERIAL_BOX_X} \
  box_y:=${MATERIAL_BOX_Y} \
  box_z:=${MATERIAL_BOX_Z} \
  box_length:=${MATERIAL_BOX_L} \
  box_width:=${MATERIAL_BOX_W} \
  box_height:=${MATERIAL_BOX_H} \
  wall_thickness:=${MATERIAL_WALL_THICKNESS} \
  wall_alpha:=${MATERIAL_WALL_ALPHA} &

sleep 2

echo ""
echo "生成包装箱 packing_box..."
ros2 launch eli_cs_robot_simulation_gz spawn_material_box.launch.py \
  box_name:=packing_box \
  box_x:=${PACK_BOX_X} \
  box_y:=${PACK_BOX_Y} \
  box_z:=${PACK_BOX_Z} \
  box_length:=${PACK_BOX_L} \
  box_width:=${PACK_BOX_W} \
  box_height:=${PACK_BOX_H} \
  wall_thickness:=${PACK_WALL_THICKNESS} \
  wall_alpha:=${PACK_WALL_ALPHA} &

sleep 3

echo ""
echo "向 MoveIt Planning Scene 添加物料箱和包装箱碰撞体..."
python3 scripts/add_boxes_to_planning_scene.py
ADD_SCENE_RET=$?

if [ ${ADD_SCENE_RET} -ne 0 ]; then
  echo "警告：添加 Planning Scene 碰撞体失败。"
  echo "仿真不会因此关闭。你可以稍后手动运行："
  echo "cd ~/NZCX/elite_ros_ws"
  echo "source install/setup.bash"
  echo "python3 scripts/add_boxes_to_planning_scene.py"
else
  echo "Planning Scene 碰撞体添加成功。"
fi

sleep 1

echo ""
echo "生成物料箱内随机物体..."
ros2 launch eli_cs_robot_simulation_gz spawn_random_items_in_box.launch.py \
  box_x:=${MATERIAL_BOX_X} \
  box_y:=${MATERIAL_BOX_Y} \
  box_z:=${MATERIAL_BOX_Z} \
  box_length:=${MATERIAL_BOX_L} \
  box_width:=${MATERIAL_BOX_W} \
  box_height:=${MATERIAL_BOX_H} \
  wall_thickness:=${MATERIAL_WALL_THICKNESS} \
  item_count:=6 \
  seed:=${SEED} &

echo ""
echo "============================================================"
echo "全部启动完成"
echo "============================================================"
echo "按 Ctrl+C 可以关闭所有进程。"
echo ""
echo "测试机械臂点位时，另开一个终端执行："
echo "cd ~/NZCX/elite_ros_ws"
echo "source install/setup.bash"
echo "python3 scripts/test_tool0_points.py"
echo "============================================================"

wait
