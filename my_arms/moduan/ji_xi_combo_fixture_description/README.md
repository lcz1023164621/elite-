# ji_xi_combo_fixture_description

这是根据 `夹吸组合夹具.STEP` 自动生成的 URDF 包。

## 生成方式与限制
- 当前环境无法直接使用 CAD 内核把 STEP 精确转换为 URDF 可用的 STL/DAE。
- 已采用 **STEP 点集 -> 三维凸包 -> STL** 的方式生成近似外形。
- 因此：
  - `urdf/ji_xi_combo_fixture.urdf` 是 **凸包近似版**；
  - `urdf/ji_xi_combo_fixture_box.urdf` 是 **包围盒简化版**；
  - 质量和惯量是占位值，需要你按真实夹具再改。

## 从 STEP 估计出的整体尺寸
- 尺寸（m）: 0.509403 x 0.114889 x 0.216800
- 包围盒中心（m）: 0.152202, -0.012445, 0.061600
- CAD 原点已保留为 link 原点。

## 目录
- `meshes/ji_xi_combo_fixture_convex_hull.stl` 近似网格
- `urdf/ji_xi_combo_fixture.urdf` 网格版 URDF
- `urdf/ji_xi_combo_fixture_box.urdf` 包围盒版 URDF
- `urdf/attach_example.xacro` 安装到机械臂末端的示例

## 使用
把整个目录放到 ROS 2 工作空间下，例如：
`src/ji_xi_combo_fixture_description`
然后：
```bash
colcon build --packages-select ji_xi_combo_fixture_description
source install/setup.bash
check_urdf $(ros2 pkg prefix ji_xi_combo_fixture_description)/share/ji_xi_combo_fixture_description/urdf/ji_xi_combo_fixture.urdf
```

## 更高精度建议
若你要真正在 Gazebo / MoveIt 中精确碰撞，建议在本机用 FreeCAD 或 SolidWorks 再做一次：
STEP -> STL/DAE -> 替换 `meshes/` 中的文件。
