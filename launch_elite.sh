#!/bin/bash
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 资源路径
export GZ_SIM_RESOURCE_PATH=$PROJECT_DIR/arms_models:$PROJECT_DIR/src/elite_description:$PROJECT_DIR/models:$PROJECT_DIR/models/gazebo_models:$PROJECT_DIR/worlds:${GZ_SIM_RESOURCE_PATH}

# 渲染设置
unset WAYLAND_DISPLAY
export DISPLAY=:0
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=llvmpipe

echo "资源路径: $GZ_SIM_RESOURCE_PATH"
# -r: 启动后立即运行仿真
ign gazebo -r "$PROJECT_DIR/worlds/my_world.sdf" &
GZ_PID=$!

# DetachableJoint 默认初始附着；启动后持续发送 detach
for _ in 1 2 3 4 5 6 7 8; do
  sleep 0.8
  ign topic -t /ec612/suction/detach -m ignition.msgs.Empty -p "unused: true" >/dev/null 2>&1 || true
done

wait "$GZ_PID"