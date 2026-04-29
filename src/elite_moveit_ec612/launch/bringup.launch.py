"""
CS612 机械臂完整启动文件（Gazebo 仿真 + MoveIt）
启动顺序：
  1. Gazebo Sim（仿真环境）
  2. ros_gz_bridge（Gazebo <-> ROS2 话题桥接：吸盘 + 状态 + 位姿）
  3. cs612_joint_states_bridge：/joint_states_gz → /joint_states（规范化关节名）
  4. cs612_trajectory_action_bridge：MoveIt FollowJointTrajectory → GZ cmd topics
  5. robot_state_publisher（发布 TF）
  6. move_group（MoveIt 规划节点）
  7. RViz2

仿真与 RViz「同一机械臂」同步依赖：Gazebo joint_state → 桥 → /joint_states → TF。
"""
import os
import shutil
import sys
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    LogInfo,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import LaunchConfigurationEquals
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _system_ros_setup_for_gz_bridge() -> tuple[Path | None, str]:
    # Ubuntu 22.04 + Humble: 优先使用 Humble 的 ros_gz_bridge（已安装）。
    # Gazebo Fortress 使用 ignition-transport11，与 Humble 的桥接完全兼容。
    for distro_name, distro_path in [
        ("humble", Path("/opt/ros/humble/setup.bash")),
    ]:
        bridge_bin = distro_path.parent / "lib" / "ros_gz_bridge" / "bridge_node"
        if distro_path.is_file() and bridge_bin.is_file():
            return distro_path, distro_name

    return None, "none"


def _clean_env_for_system_ros_bridge() -> tuple[dict, dict]:
    runtime_root = Path("/tmp/cs612_runtime")
    ros_log_dir = runtime_root / "ros_logs"
    gz_home = runtime_root / "home"
    xdg_config_home = runtime_root / "xdg_config"
    xdg_cache_home = runtime_root / "xdg_cache"
    for path in (ros_log_dir, gz_home, xdg_config_home, xdg_cache_home):
        path.mkdir(parents=True, exist_ok=True)

    extra = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(gz_home),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "DISPLAY": os.environ.get("DISPLAY", ":0"),
        "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", ""),
        "ROS_LOG_DIR": str(ros_log_dir),
        "XDG_CONFIG_HOME": str(xdg_config_home),
        "XDG_CACHE_HOME": str(xdg_cache_home),
    }
    extra = {k: v for k, v in extra.items() if v}
    return {}, extra


def _get_project_paths(launch_file: Path) -> tuple[Path, Path]:
    launch_file = launch_file.resolve()
    cwd = Path.cwd().resolve()

    source_candidates: list[Path] = []
    env_root = os.environ.get("CS612_PROJECT_ROOT", "").strip() or os.environ.get("EC612_PROJECT_ROOT", "").strip()
    if env_root:
        source_candidates.append(Path(env_root).expanduser().resolve())
    source_candidates.append(cwd)
    for parent in launch_file.parents:
        if parent.name == "install":
            source_candidates.append(parent.parent.resolve())
            break

    urdf_rel = Path("my_arms") / "urdf" / "CS612.urdf"

    seen: set[Path] = set()
    for cand in source_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / urdf_rel).is_file() and (cand / "src" / "elite_moveit_ec612").is_dir():
            return cand, cand / "src" / "elite_moveit_ec612"

    from ament_index_python.packages import get_package_share_directory

    try:
        share_dir = Path(get_package_share_directory("CS612urdf"))
        if (share_dir / "urdf" / "CS612.urdf").is_file():
            return share_dir.parent.parent.parent.parent, Path(get_package_share_directory("elite_moveit_ec612"))
    except Exception:
        pass

    raise FileNotFoundError(
        "找不到 my_arms/urdf/CS612.urdf。请在项目根目录启动，或设置 CS612_PROJECT_ROOT。"
    )


def _load_urdf_with_mesh_paths(project_root: Path) -> str:
    urdf_path = project_root / "my_arms" / "urdf" / "CS612.urdf"
    text = urdf_path.read_text(encoding="utf-8")
    mesh_uri = (project_root / "my_arms" / "meshes").resolve().as_uri()
    text = text.replace('<robot name="CS612urdf">', '<robot name="cs612">')
    for idx in range(1, 7):
        text = text.replace(f'name="Joint{idx}"', f'name="joint{idx}"')
    text = text.replace("package://CS612urdf/meshes/", mesh_uri + "/")
    if '<link name="flan"' not in text:
        flan = """
  <link name="flan" />
  <joint name="flan_joint" type="fixed">
    <origin xyz="0 0 0" rpy="0 0 0" />
    <parent link="wrist_3_link" />
    <child link="flan" />
  </joint>
"""
        text = text.replace('  <link name="suction_cup_link">', flan + '\n  <link name="suction_cup_link">', 1)
    return text


def _load_scene_defaults(moveit_config_dir: Path) -> tuple[list[float], list[float], list[float], list[float], float]:
    rect_center = [0.45, 0.0, 0.03]
    rect_size = [0.14, 0.10, 0.06]
    suction_pick_center = [0.45, 0.0, 0.065]
    suction_pick_tolerance = 0.05
    carton_pose = [0.82, -0.32, 0.0]
    cfg = moveit_config_dir / "config" / "scene_objects.yaml"
    try:
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        rect = doc.get("rect_pickup") or {}
        carton = doc.get("carton_box") or {}
        rect_center = [float(v) for v in rect.get("center_xyz", rect_center)]
        rect_size = [float(v) for v in rect.get("size_xyz", rect_size)]
        suction_pick_center = [
            float(v)
            for v in rect.get(
                "suction_pick_center_xyz",
                [rect_center[0], rect_center[1], rect_center[2] + 0.5 * rect_size[2] + 0.005],
            )
        ]
        suction_pick_tolerance = float(rect.get("suction_pick_tolerance_m", suction_pick_tolerance))
        carton_pose = [float(v) for v in carton.get("model_pose_xyz", carton_pose)]
    except Exception:
        pass
    return rect_center, rect_size, carton_pose, suction_pick_center, suction_pick_tolerance


def _gz_executable() -> str:
    # Gazebo Fortress: 使用 ign 命令
    if os.path.isfile("/usr/bin/ign") and os.access("/usr/bin/ign", os.X_OK):
        return "/usr/bin/ign"
    w = shutil.which("ign")
    if w and not ("miniconda" in w or "anaconda" in w or "conda/envs" in w.lower()):
        return w
    # Fallback: try gz (Gazebo Harmonic+)
    if os.path.isfile("/usr/bin/gz") and os.access("/usr/bin/gz", os.X_OK):
        return "/usr/bin/gz"
    w = shutil.which("gz")
    if w and not ("miniconda" in w or "anaconda" in w or "conda/envs" in w.lower()):
        return w
    return "ign"


def _gz_clean_env_for_sim(gz_resource_path: str) -> tuple[dict, dict]:
    runtime_root = Path("/tmp/cs612_runtime")
    ros_log_dir = runtime_root / "ros_logs"
    gz_home = runtime_root / "home"
    xdg_config_home = runtime_root / "xdg_config"
    xdg_cache_home = runtime_root / "xdg_cache"
    for path in (ros_log_dir, gz_home, xdg_config_home, xdg_cache_home):
        path.mkdir(parents=True, exist_ok=True)

    sys_ld = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu"
    xdg_rt = os.environ.get("XDG_RUNTIME_DIR", "")
    if not xdg_rt:
        xdg_rt = f"/tmp/runtime-{os.environ.get('USER', 'user')}"
    extra = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(gz_home),
        "USER": os.environ.get("USER", ""),
        "DISPLAY": os.environ.get("DISPLAY", ":0"),
        "XAUTHORITY": os.environ.get("XAUTHORITY", ""),
        "WAYLAND_DISPLAY": os.environ.get("WAYLAND_DISPLAY", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "GZ_SIM_RESOURCE_PATH": gz_resource_path,
        "LD_LIBRARY_PATH": sys_ld,
        "ROS_LOG_DIR": str(ros_log_dir),
        "XDG_CONFIG_HOME": str(xdg_config_home),
        "XDG_CACHE_HOME": str(xdg_cache_home),
        "LIBGL_ALWAYS_SOFTWARE": os.environ.get("LIBGL_ALWAYS_SOFTWARE", "1"),
        "GALLIUM_DRIVER": os.environ.get("GALLIUM_DRIVER", "llvmpipe"),
        "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "xcb"),
        "XDG_RUNTIME_DIR": xdg_rt,
    }
    extra = {k: v for k, v in extra.items() if v}
    return {}, extra


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    # 手动解析命令行中的 use_sim_time，用于 ExecuteProcess 的 cmd 字符串
    _use_sim_time_str = "false"
    for arg in sys.argv:
        if arg.startswith("use_sim_time:="):
            _use_sim_time_str = arg.split(":=", 1)[1]
        elif arg == "use_sim_time:=true":
            _use_sim_time_str = "true"

    launch_file = Path(__file__)
    project_root, moveit_config_dir = _get_project_paths(launch_file)

    robot_description_content = _load_urdf_with_mesh_paths(project_root)

    srdf_path = moveit_config_dir / "config" / "cs612.srdf"
    robot_description_semantic_content = srdf_path.read_text(encoding="utf-8")

    with open(moveit_config_dir / "config" / "ompl_planning.yaml", "r", encoding="utf-8") as f:
        ompl_cfg = yaml.safe_load(f)

    with open(moveit_config_dir / "config" / "joint_limits.yaml", "r", encoding="utf-8") as f:
        joint_limits_doc = yaml.safe_load(f)

    with open(moveit_config_dir / "config" / "kinematics.yaml", "r", encoding="utf-8") as f:
        robot_description_kinematics = yaml.safe_load(f)

    rect_center, rect_size, carton_pose, suction_pick_center, suction_pick_tolerance = _load_scene_defaults(
        moveit_config_dir
    )

    world_file = project_root / "worlds" / "my_world.sdf"

    cyclone_cfg = moveit_config_dir / "config" / "cyclonedds.xml"
    fastdds_cfg = moveit_config_dir / "config" / "fastdds_cs612.xml"
    rmw_impl = LaunchConfiguration("rmw_implementation")

    gz_parts = [
        project_root / "arms_models",
        # Required for Gazebo to resolve package://my_arms/...
        project_root,
        # Required for Gazebo to resolve package://elite_description/...
        project_root / "src",
        project_root / "src" / "elite_description",
        project_root / "models",
        project_root / "models" / "gazebo_models",
        project_root / "worlds",
    ]
    py_src_dir = (project_root / "src" / "elite_moveit_ec612").resolve()
    old_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_value = str(py_src_dir) if not old_pythonpath else f"{py_src_dir}:{old_pythonpath}"

    gz_resource_path = ":".join(str(p) for p in gz_parts if p.is_dir())
    gz_empty_env, gz_add_env = _gz_clean_env_for_sim(gz_resource_path)
    gz_bin = _gz_executable()
    # Determine subcommand and message prefix based on Gazebo version
    # Fortress uses 'ign gazebo' / 'ign topic' / 'ignition.msgs'
    # Harmonic+ uses 'gz sim' / 'gz topic' / 'gz.msgs'
    _is_ign = os.path.basename(gz_bin).startswith("ign")
    _sim_cmd = "gazebo" if _is_ign else "sim"
    _msg_prefix = "ignition.msgs" if _is_ign else "gz.msgs"

    gazebo = ExecuteProcess(
        cmd=[gz_bin, _sim_cmd, "-r", str(world_file)],
        env=gz_empty_env,
        additional_env=gz_add_env,
        output="screen",
    )

    def _startup_detach_action(delay_sec: float) -> TimerAction:
        return TimerAction(
            period=delay_sec,
            actions=[
                ExecuteProcess(
                    cmd=[
                        gz_bin,
                        "topic",
                        "-t",
                        "/cs612/suction/detach",
                        "-m",
                        _msg_prefix + ".Empty",
                        "-p",
                        "unused: true",
                    ],
                    env=gz_empty_env,
                    additional_env=gz_add_env,
                    output="screen",
                )
            ],
        )

    startup_detach_actions = [
        _startup_detach_action(1.6),
        _startup_detach_action(2.2),
        _startup_detach_action(2.8),
        _startup_detach_action(3.6),
        _startup_detach_action(4.6),
        _startup_detach_action(6.0),
        _startup_detach_action(7.5),
        _startup_detach_action(9.0),
    ]

    bridge_config_file = moveit_config_dir / "config" / "gz_bridge_topics.yaml"
    setup_path, bridge_distro = _system_ros_setup_for_gz_bridge()
    if setup_path is not None:
        bridge_bin = setup_path.parent / "lib" / "ros_gz_bridge" / "bridge_node"
        bridge_unset = (
            "unset PYTHONPATH LD_LIBRARY_PATH AMENT_PREFIX_PATH CMAKE_PREFIX_PATH "
            "COLCON_PREFIX_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION;"
        )
        bridge_empty_env, bridge_add_env = _clean_env_for_system_ros_bridge()
        bridge_rmw_impl = os.environ.get(
            "CS612_BRIDGE_RMW_IMPLEMENTATION",
            os.environ.get("EC612_BRIDGE_RMW_IMPLEMENTATION", "rmw_fastrtps_cpp"),
        )
        bridge_add_env = {**bridge_add_env, "RMW_IMPLEMENTATION": bridge_rmw_impl}
        if bridge_rmw_impl == "rmw_fastrtps_cpp":
            bridge_add_env = {
                **bridge_add_env,
                "FASTDDS_BUILTIN_TRANSPORTS": "UDPv4",
            }
            if fastdds_cfg.is_file():
                bridge_add_env["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(fastdds_cfg.resolve())
        if bridge_rmw_impl == "rmw_cyclonedds_cpp" and cyclone_cfg.is_file():
            bridge_add_env = {**bridge_add_env, "CYCLONEDDS_URI": f"file://{cyclone_cfg.resolve()}"}
        gz_bridge = ExecuteProcess(
            cmd=[
                "bash",
                "-c",
                f"{bridge_unset} source {setup_path} && "
                f"exec {bridge_bin} --ros-args "
                f"--disable-rosout-logs --disable-external-lib-logs "
                f"-p start_parameter_services:=false "
                f"-p start_parameter_event_publisher:=false "
                f"-p config_file:={bridge_config_file} -p use_sim_time:=true",
            ],
            env=bridge_empty_env,
            additional_env=bridge_add_env,
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
        gz_set_pose_bridge = ExecuteProcess(
            cmd=[
                "bash",
                "-c",
                f"{bridge_unset} source {setup_path} && "
                "exec "
                f"{setup_path.parent / 'lib' / 'ros_gz_bridge' / 'parameter_bridge'} "
                "/world/arm_world/set_pose@ros_gz_interfaces/srv/SetEntityPose "
                "--ros-args --disable-rosout-logs --disable-external-lib-logs "
                "-p start_parameter_services:=false "
                "-p start_parameter_event_publisher:=false",
            ],
            env=bridge_empty_env,
            additional_env=bridge_add_env,
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
        bridge_hint = LogInfo(
            msg=(
                f"[elite_moveit_cs612] 使用系统 ROS {bridge_distro} 的 ros_gz_bridge bridge_node"
                f"（{setup_path}），bridge RMW={bridge_rmw_impl}。"
                "Gazebo Fortress + Humble bridge 已就绪。"
            ),
        )
    else:
        gz_bridge = Node(
            package="ros_gz_bridge",
            executable="bridge_node",
            name="gz_ros2_bridge",
            parameters=[
                {"config_file": str(bridge_config_file)},
                {"use_sim_time": _use_sim_time_str},
            ],
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
        gz_set_pose_bridge = Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            arguments=[
                "/world/arm_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
            ],
            output="screen",
            respawn=True,
            respawn_delay=2.0,
        )
        bridge_hint = LogInfo(
            msg=(
                "[elite_moveit_cs612] 未找到系统 ros_gz_bridge，使用节点方式启动 bridge。"
                "请确保 ros-humble-ros-gz-bridge 已安装。"
            ),
        )

    joint_states_bridge = Node(
        package="elite_moveit_ec612",
        executable="cs612_joint_states_bridge",
        name="cs612_joint_states_bridge",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )
    gz_pose_bridge = Node(
        package="elite_moveit_ec612",
        executable="cs612_gz_pose_bridge",
        name="cs612_gz_pose_bridge",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"frame_id": "base_link"},
        ],
    )
    trajectory_action_bridge = Node(
        package="elite_moveit_ec612",
        executable="cs612_trajectory_action_bridge",
        name="cs612_trajectory_action_bridge",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"goal_tolerance": 0.03},
            {"goal_soft_tolerance": 0.04},
            {"loose_tolerance_joints": ["wrist_3_joint"]},
            {"loose_goal_tolerance": 0.06},
            {"loose_goal_soft_tolerance": 0.20},
            {"goal_settle_timeout_sec": 45.0},
            {"goal_hold_publish_period_sec": 0.02},
        ],
    )
    world_markers = Node(
        package="elite_moveit_ec612",
        executable="cs612_world_markers",
        name="cs612_world_markers",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"use_scene_yaml_fallback": True},
        ],
    )

    planning_scene_spawner = Node(
        package="elite_moveit_ec612",
        executable="cs612_planning_scene_spawner",
        name="cs612_planning_scene_spawner",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
    )

    auto_pick = Node(
        package="elite_moveit_ec612",
        executable="cs612_auto_pick_place",
        name="cs612_auto_pick_place",
        output="screen",
        condition=IfCondition(LaunchConfiguration("auto_pick")),
        parameters=[
            {"use_sim_time": ParameterValue(use_sim_time, value_type=bool)},
            {"wait_poses_sec": 90.0},
            {"require_joint_states": True},
            {"use_known_rect_surface_center": True},
            {"known_rect_center_xyz": rect_center},
            {"known_rect_size_xyz": rect_size},
            {"use_known_suction_pick_center": False},
            {"known_suction_pick_center_xyz": suction_pick_center},
            {"suction_pick_center_tolerance_m": suction_pick_tolerance},
            {"rect_fallback_pose_xyz": rect_center},
            {"rect_fallback_wait_sec": 8.0},
            {"carton_fallback_pose_xyz": carton_pose},
            {"carton_fallback_wait_sec": 8.0},
            # 使用 Elite MoveIt 规划到物体中心上方；接触阶段再做短距离笛卡尔下压。
            {"use_compute_ik": False},
            {"hybrid_moveit_pregrasp": True},
            {"hybrid_cartesian_touch_only": True},
            {"prefer_upright_joint_goal_for_pick": False},
            {"pick_posture_hint": [0.0, -0.10, -1.80, 0.30, 1.57, 0.0]},
            {"pick_pregrasp_min_joint3_origin_z": 0.12},
            {"pick_pregrasp_elbow_filter_min_target_z": 0.45},
            {"use_joint_template_demo": False},
            {"move_velocity_scale": 0.20},
            {"move_acceleration_scale": 0.20},
            {"joint_goal_tolerance": 0.04},
            {"pose_orientation_tolerance": 0.025},
            {"move_to_start_face_pose": False},
            {"pre_pick_safe_clearance": 0.22},
            {"approach_clearance": 0.18},
            {"pre_touch_hover_extra_z": 0.04},
            {"touch_delta_z": 0.005},
            {"suction_contact_offset_z": 0.0},
            {"place_object_bottom_clearance": 0.008},
            {"place_entry_clearance": 0.12},
            {"post_place_retreat": 0.14},
            {"place_inner_margin_xy": 0.025},
            {"refresh_top_from_live_pose": True},
            {"refresh_top_max_delta_xy": 0.20},
            {"refresh_top_max_delta_z": 0.08},
            {"centerline_use_object_center_only": True},
            {"pregrasp_alignment_gate_enabled": True},
            {"allow_unverified_sim_attach": False},
            {"fake_attach_set_pose_fallback": True},
            {"fake_attach_service": "/world/arm_world/set_pose"},
            {"fake_attach_update_hz": 8.0},
            {"pregrasp_xy_align_tol": 0.008},
            {"pregrasp_xy_comp_max_step_m": 0.020},
            {"pregrasp_xy_comp_gain": 0.80},
            {"pregrasp_xy_comp_retries": 6},
            {"pregrasp_cartesian_center_enabled": True},
            {"orientation_min_cos_before_touch": 0.995},
            {"orientation_correction_weight": 6.0},
            {"orientation_correction_retries": 3},
            {"post_moveit_orientation_snap_enabled": False},
            {"touch_cartesian_keep_xy": True},
            {"touch_cartesian_step_max_m": 0.004},
            {"touch_cartesian_joint_step_limit_rad": 0.040},
            {"touch_cartesian_orientation_weight": 6.0},
            # 当前自写笛卡尔 IK 与真实 TF 存在可观偏差；触碰阶段先使用 MoveIt 对
            # suction_tcp_link 的位姿约束，避免密途径点下压把 TCP 横向推偏。
            {"dense_waypoint_descent_enabled": False},
            {"dense_waypoint_step_m": 0.004},
            {"dense_waypoint_max_xy_drift_m": 0.035},
            {"staged_pregrasp_enabled": True},
            {"staged_pregrasp_clearances": [0.24]},
            {"dense_waypoint_orientation_weight": 6.0},
            {"suction_touch_lateral_tol": 0.040},
            {"suction_touch_vertical_tol": 0.060},
            {"suction_touch_axis_down_min": 0.960},
            {"require_dual_bottom_contact_before_attach": True},
            {"suction_cup_offsets_xy": [-0.018, 0.0, 0.018, 0.0]},
            {"suction_cup_lip_radius": 0.010},
            {"suction_rubber_compression_m": 0.015},
            {"suction_attach_lateral_tol": 0.055},
            {"suction_attach_vertical_tol": 0.055},
            {"suction_attach_axis_down_min": 0.940},
            {"suction_attach_burst_count": 6},
            {"suction_attach_burst_interval_sec": 0.06},
            {"suction_attach_wait_sec": 2.0},
            {"pickup_probe_lift_z": 0.020},
            {"pickup_probe_min_follow_z": 0.008},
            {"pickup_probe_require_follow_if_live_pose": False},
            {"pre_touch_settle_sec": 0.8},
            {"approach_verify_enabled": True},
            {"post_touch_settle_sec": 0.6},
            {"pre_attach_settle_sec": 0.4},
            {"cartesian_settle_timeout_sec": 12.0},
            {"cartesian_settle_tol_rad": 0.045},
            {"cartesian_bridge_wait_sec": 35.0},
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {"robot_description": robot_description_content},
            {"use_sim_time": use_sim_time},
            {"ignore_timestamp": True},
            {"publish_frequency": 50.0},
            {"publish_robot_description": True},
        ],
    )

    moveit_simple_controller_manager = {
        "controller_names": ["manipulator_controller"],
        "manipulator_controller": {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
        },
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            {"robot_description": robot_description_content},
            {"robot_description_semantic": robot_description_semantic_content},
            {"robot_description_kinematics": robot_description_kinematics},
            {"robot_description_planning": joint_limits_doc},
            {"planning_pipelines": ["ompl"]},
            {"ompl": ompl_cfg},
            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"moveit_simple_controller_manager": moveit_simple_controller_manager},
            {"allow_trajectory_execution": True},
            {"trajectory_execution.allowed_start_tolerance": 0.04},
            {"trajectory_execution.allowed_execution_duration_scaling": 10.0},
            {"trajectory_execution.allowed_goal_duration_margin": 5.0},
            {"trajectory_execution.execution_duration_monitoring": False},
            {"publish_robot_description_semantic": True},
            {"publish_robot_description": False},
            {"monitor_dynamics": False},
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_config_file = moveit_config_dir / "config" / "cs612.rviz"
    if not rviz_config_file.is_file():
        rviz_config_file = moveit_config_dir / "config" / "moveit.rviz"
    rviz_args = (
        ["-d", str(rviz_config_file), "--ros-args", "-p", ["use_sim_time:=", use_sim_time]]
        if rviz_config_file.is_file()
        else ["--ros-args", "-p", ["use_sim_time:=", use_sim_time]]
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=rviz_args,
        parameters=[
            {"robot_description": robot_description_content},
            {"robot_description_semantic": robot_description_semantic_content},
            {"robot_description_kinematics": robot_description_kinematics},
            {"use_sim_time": use_sim_time},
        ],
    )

    map_to_base_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_base_link",
        arguments=[
            "--frame-id", "map",
            "--child-frame-id", "base_link",
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="log",
    )

    delayed_ros = TimerAction(
        period=1.5,
        actions=[
            gz_bridge,
            gz_set_pose_bridge,
            joint_states_bridge,
            gz_pose_bridge,
            trajectory_action_bridge,
            world_markers,
            map_to_base_tf,
            robot_state_publisher,
            move_group_node,
            planning_scene_spawner,
            rviz_node,
        ],
    )

    delayed_auto_pick = TimerAction(
        period=10.0,
        actions=[auto_pick],
    )

    dds_env: list = [
        SetEnvironmentVariable(
            name="RMW_IMPLEMENTATION",
            value=rmw_impl,
        )
    ]
    if cyclone_cfg.is_file():
        dds_env.append(
            SetEnvironmentVariable(
                name="CYCLONEDDS_URI",
                value=f"file://{cyclone_cfg.resolve()}",
                condition=LaunchConfigurationEquals("rmw_implementation", "rmw_cyclonedds_cpp"),
            )
        )
    dds_env.append(
        SetEnvironmentVariable(
            name="FASTDDS_BUILTIN_TRANSPORTS",
            value="UDPv4",
            condition=LaunchConfigurationEquals("rmw_implementation", "rmw_fastrtps_cpp"),
        )
    )
    if fastdds_cfg.is_file():
        dds_env.append(
            SetEnvironmentVariable(
                name="FASTRTPS_DEFAULT_PROFILES_FILE",
                value=str(fastdds_cfg.resolve()),
                condition=LaunchConfigurationEquals("rmw_implementation", "rmw_fastrtps_cpp"),
            ),
        )
        dds_env.append(
            SetEnvironmentVariable(
                name="RMW_FASTRTPS_USE_QOS_FROM_XML",
                value="1",
                condition=LaunchConfigurationEquals("rmw_implementation", "rmw_fastrtps_cpp"),
            ),
        )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("rmw_implementation", default_value="rmw_fastrtps_cpp"),
            DeclareLaunchArgument(
                "auto_pick",
                default_value="true",
                description="启动完成后自动执行 CS612 抓取并入箱流程；如需手动规划可设为 false。",
            ),
            SetEnvironmentVariable(
                name="PYTHONPATH",
                value=pythonpath_value,
            ),
            *dds_env,
            bridge_hint,
            gazebo,
            *startup_detach_actions,
            delayed_ros,
            delayed_auto_pick,
        ]
    )
