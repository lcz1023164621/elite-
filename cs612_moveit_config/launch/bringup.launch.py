"""CS612 Gazebo + MoveIt bringup using the official Elite CS ROS 2 stack."""
import json
import os
import subprocess
import time
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit, OnProcessStart, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
_DEBUG_LOG_PATH = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-a97e6b.log")
_DEBUG_SESSION_ID = "a97e6b"
_DEBUG_LOG_PATH_ACTIVE = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-3e253c.log")
_DEBUG_SESSION_ID_ACTIVE = "3e253c"
_AGENT_9009e8_ID = "9009e8"
# 与 auto_pick_place._IDE_CURSOR_MIRROR_LOG 一致：便于对照 launch 子进程与 Cursor 工作区是否同一挂载
_IDE_CURSOR_AGENT_LOG = Path("/mnt/e/gazebo_projects/my_first_world/.cursor/debug-9009e8.log")


def _append_line(path: Path, line: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _debug_log_9009e8_bridge(project_root: Path, data: dict) -> None:
    """由 launch 进程写入（与 _debug_log 同源），用于验证 IDE 侧能否看到 debug-9009e8.log（H_sync）。"""
    payload = {
        "sessionId": _AGENT_9009e8_ID,
        "runId": "pre-fix",
        "hypothesisId": "H_bridge",
        "location": "bringup.launch.py:_launch_setup",
        "message": "bringup_workspace_bridge",
        "data": {"project_root": str(project_root.resolve()), **data},
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    _append_line(_DEBUG_LOG_PATH, line)
    paths = [project_root / ".cursor" / "debug-9009e8.log", _IDE_CURSOR_AGENT_LOG]
    seen: set[str] = set()
    for p in paths:
        try:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            _append_line(p, line)
        except Exception:
            pass


def _debug_log(location: str, message: str, hypothesis_id: str, data: dict) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    candidates = [
        _DEBUG_LOG_PATH,
        Path.cwd() / ".cursor" / "debug-a97e6b.log",
        Path("/tmp/cs612_runtime/home/.cursor/debug-a97e6b.log"),
    ]
    env_path = os.environ.get("CS612_DEBUG_NDJSON_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))
    seen: set[str] = set()
    for p in candidates:
        try:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            _append_line(p, line)
        except Exception:
            pass


def _debug_log_active(location: str, message: str, hypothesis_id: str, data: dict) -> None:
    payload = {
        "sessionId": _DEBUG_SESSION_ID_ACTIVE,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    _append_line(_DEBUG_LOG_PATH_ACTIVE, json.dumps(payload, ensure_ascii=True) + "\n")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _find_project_root() -> Path:
    candidates = [Path.cwd().resolve()]
    try:
        share = Path(get_package_share_directory("cs612_moveit_config")).resolve()
        candidates.append(share)
        for parent in share.parents:
            candidates.append(parent)
    except Exception:
        pass
    for cand in candidates:
        if (cand / "worlds" / "my_world.sdf").is_file() and (cand / "cs612_moveit_config").is_dir():
            return cand
    return Path.cwd().resolve()


def _launch_setup(context, *args, **kwargs):
    project_root = _find_project_root()
    moveit_config_dir = Path(get_package_share_directory("cs612_moveit_config"))
    world_file = project_root / "worlds" / "my_world.sdf"
    controllers_file = moveit_config_dir / "config" / "ros2_controllers.yaml"

    cs_type = LaunchConfiguration("cs_type")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    auto_pick = LaunchConfiguration("auto_pick")
    auto_pick_delay_sec = LaunchConfiguration("auto_pick_delay_sec")
    launch_gz_gui = LaunchConfiguration("launch_gz_gui")
    rviz_config_arg = LaunchConfiguration("rviz_config")
    helper_modules = {
        "cs612_joint_states_bridge": "cs612_moveit_config.joint_states_bridge",
        "cs612_trajectory_action_bridge": "cs612_moveit_config.trajectory_action_bridge",
        "cs612_world_markers": "cs612_moveit_config.world_markers",
        "cs612_planning_scene_spawner": "cs612_moveit_config.planning_scene_spawner",
        "cs612_system_watchdog": "cs612_moveit_config.system_watchdog",
        "cs612_auto_pick_place": "cs612_moveit_config.auto_pick_place",
    }

    xacro_file = moveit_config_dir / "config" / "cs612_suction_control.urdf.xacro"
    robot_description_content = subprocess.check_output(
        [
            "xacro",
            str(xacro_file),
            f"cs_type:={cs_type.perform(context)}",
            "name:=cs612",
            "sim_ignition:=true",
            f"simulation_controllers:={controllers_file}",
        ],
        text=True,
    )
    robot_description = {"robot_description": robot_description_content}
    robot_description_semantic = {
        "robot_description_semantic": (moveit_config_dir / "config" / "CS612.srdf").read_text(encoding="utf-8")
    }
    robot_description_kinematics = {"robot_description_kinematics": _load_yaml(moveit_config_dir / "config" / "kinematics.yaml")}
    robot_description_planning = {"robot_description_planning": _load_yaml(moveit_config_dir / "config" / "joint_limits.yaml")}
    ompl_cfg = _load_yaml(moveit_config_dir / "config" / "ompl_planning.yaml")
    moveit_controllers = _load_yaml(moveit_config_dir / "config" / "moveit_controllers.yaml")

    gz_gui_enabled = launch_gz_gui.perform(context).strip().lower() in ("1", "true", "yes", "on")
    gz_flags = "-r -v 4" if gz_gui_enabled else "-r -s -v 4"
    rviz_config_value = rviz_config_arg.perform(context).strip()
    rviz_config_file = Path(rviz_config_value)
    if not rviz_config_file.is_absolute():
        rviz_config_file = moveit_config_dir / "config" / rviz_config_value
    rviz_fixed_frame = "unknown"
    rviz_static_models: dict[str, bool] = {}
    try:
        rviz_cfg = _load_yaml(rviz_config_file)
        vm = rviz_cfg.get("Visualization Manager", {}) if isinstance(rviz_cfg, dict) else {}
        go = vm.get("Global Options", {}) if isinstance(vm, dict) else {}
        rviz_fixed_frame = str(go.get("Fixed Frame", "unknown"))
        displays = vm.get("Displays", []) if isinstance(vm, dict) else []
        if isinstance(displays, list):
            for d in displays:
                if not isinstance(d, dict):
                    continue
                name = str(d.get("Name", ""))
                if name in ("StaticArm2RobotModel", "StaticArm3RobotModel"):
                    rviz_static_models[name] = bool(d.get("Enabled", False))
    except Exception:
        pass
    # #region agent log
    _debug_log(
        "bringup.launch.py:_launch_setup",
        "launch_setup_values",
        "H4",
        {
            "launch_rviz": launch_rviz.perform(context),
            "launch_gz_gui": launch_gz_gui.perform(context),
            "auto_pick": auto_pick.perform(context),
            "rviz_config": str(rviz_config_file),
            "world_file_exists": world_file.is_file(),
        },
    )
    _debug_log(
        "bringup.launch.py:_launch_setup",
        "rviz_display_config",
        "H9",
        {
            "fixed_frame": rviz_fixed_frame,
            "static_models_enabled": rviz_static_models,
        },
    )
    try:
        _ap_delay_bridge = float(auto_pick_delay_sec.perform(context))
    except (ValueError, TypeError):
        _ap_delay_bridge = 25.0
    _debug_log_9009e8_bridge(
        project_root,
        {
            "auto_pick": auto_pick.perform(context),
            "auto_pick_delay_sec": _ap_delay_bridge,
        },
    )
    # #endregion

    gz_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": f" {gz_flags} {world_file}"}.items(),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-string", robot_description_content, "-name", "cs612", "-allow_renaming", "false"],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
            {"publish_robot_description": True},
        ],
    )

    static_arm_ns_poses = {
        "cs612_static_2": ("2.01650", "-0.71094", "0.00141"),
        "cs612_static_3": ("2.60725", "-0.99329", "0.03013"),
    }
    static_robot_state_publishers = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=ns,
            output="both",
            parameters=[
                robot_description,
                {"use_sim_time": False},
                {"publish_robot_description": True},
                {"frame_prefix": f"{ns}/"},
                # 仅靠 joint_states；仿真心跳/stamp 漂移时避免因时间戳拒发 TF
                {"ignore_timestamp": True},
                {"publish_frequency": 50.0},
            ],
            remappings=[
                ("tf", "/tf"),
                ("tf_static", "/tf_static"),
            ],
        )
        for ns in static_arm_ns_poses
    ]
    static_world_tfs = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{ns}_world_anchor",
            arguments=[
                "--x",
                xyz[0],
                "--y",
                xyz[1],
                "--z",
                xyz[2],
                "--qx",
                "0",
                "--qy",
                "0",
                "--qz",
                "0",
                "--qw",
                "1",
                "--frame-id",
                "world",
                "--child-frame-id",
                f"{ns}/world",
            ],
            parameters=[{"use_sim_time": use_sim_time}],
            output="log",
        )
        for ns, xyz in static_arm_ns_poses.items()
    ]

    static_arm_joint_publishers = [
        ExecuteProcess(
            cmd=[
                "/usr/bin/python3",
                "-m",
                "cs612_moveit_config.static_arm_state",
                "--ros-args",
                "-r",
                f"__ns:=/{ns}",
                "-p",
                "publish_hz:=50.0",
                "-p",
                "use_sim_time:=false",
            ],
            output="screen",
        )
        for ns in static_arm_ns_poses
    ]
    # #region agent log
    _debug_log(
        "bringup.launch.py:_launch_setup",
        "static_arm_publishers_configured",
        "H3",
        {
            "namespaces": list(static_arm_ns_poses.keys()),
            "world_children": [f"{ns}/world" for ns in static_arm_ns_poses],
            "rsp_frame_prefix_mode": "tf_namespace_slash",
            "static_arm_use_sim_time": False,
            "joint_state_executable": "cs612_static_arm_state",
            "joint_state_exec_mode": "/usr/bin/python3 -m cs612_moveit_config.static_arm_state",
        },
    )
    # #endregion

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="bridge_node",
        name="gz_ros2_bridge",
        parameters=[
            {"config_file": str(moveit_config_dir / "config" / "gz_bridge_topics.yaml")},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )
    gz_set_pose_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/world/arm_world/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        output="screen",
    )
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )
    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "--controller-manager",
            "/controller_manager",
        ],
        output="screen",
    )

    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            {"planning_pipelines": ["ompl"]},
            {"ompl": ompl_cfg},
            moveit_controllers,
            {
                "allow_trajectory_execution": True,
                "moveit_manage_controllers": False,
                "trajectory_execution.allowed_start_tolerance": 0.10,
                "trajectory_execution.allowed_execution_duration_scaling": 6.0,
                "trajectory_execution.allowed_goal_duration_margin": 5.0,
                "trajectory_execution.execution_duration_monitoring": True,
                "publish_robot_description": False,
                "publish_robot_description_semantic": True,
                "publish_planning_scene": True,
                "publish_geometry_updates": True,
                "publish_state_updates": True,
                "publish_transforms_updates": True,
                "use_sim_time": use_sim_time,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_config_file)],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(launch_rviz),
    )

    _ap_ndjson = str((project_root / ".cursor" / "debug-9009e8.log").resolve())
    source_pythonpath = str((project_root / "cs612_moveit_config").resolve())
    inherited_pythonpath = os.environ.get("PYTHONPATH", "").strip()
    if inherited_pythonpath:
        source_pythonpath = f"{source_pythonpath}:{inherited_pythonpath}"

    def package_script(name: str, *ros_args: str) -> ExecuteProcess:
        add_env: dict[str, str] = {
            # 优先从工作区源码导入，避免 install/ 中遗留的 conda Python 版本 site-packages 污染。
            "PYTHONPATH": source_pythonpath,
        }
        if name == "cs612_auto_pick_place":
            # 强制子进程继承 NDJSON 绝对路径（H_exec：部分环境下 ExecuteProcess 未合并上层 env）
            add_env["CS612_DEBUG_NDJSON_PATH"] = _ap_ndjson
        return ExecuteProcess(
            cmd=[
                "/usr/bin/python3",
                "-m",
                helper_modules[name],
                "--ros-args",
                "-p",
                f"use_sim_time:={use_sim_time.perform(context)}",
                *ros_args,
            ],
            output="screen",
            additional_env=add_env,
        )

    helpers = [
        package_script("cs612_world_markers"),
        package_script("cs612_planning_scene_spawner"),
        package_script("cs612_system_watchdog"),
    ]
    auto_pick_node = package_script("cs612_auto_pick_place")

    def _startup_detach_action(delay_sec: float) -> TimerAction:
        return TimerAction(
            period=delay_sec,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "/usr/bin/ign",
                        "topic",
                        "-t",
                        "/cs612/suction/detach",
                        "-m",
                        "ignition.msgs.Empty",
                        "-p",
                        "unused: true",
                    ],
                    output="log",
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
    # #region agent log
    _debug_log(
        "bringup.launch.py:_launch_setup",
        "actions_registered",
        "H2",
        {
            "helpers_count": len(helpers),
            "detach_action_count": len(startup_detach_actions),
            "arm_joint_count": len(_ARM_JOINTS),
            "main_joint_state_bridge": False,
            "main_trajectory_bridge": False,
            "ros2_control_spawners": [
                "joint_state_broadcaster",
                "joint_trajectory_controller",
            ],
        },
    )
    # #endregion

    try:
        ap_delay = float(auto_pick_delay_sec.perform(context))
    except (ValueError, TypeError):
        ap_delay = 25.0
    ap_delay = max(0.0, min(float(ap_delay), 600.0))
    # #region agent log
    _debug_log_active(
        "bringup.launch.py:_launch_setup",
        "auto_pick_timer_config",
        "H1",
        {
            "auto_pick": auto_pick.perform(context),
            "auto_pick_delay_sec": ap_delay,
            "auto_pick_module": helper_modules["cs612_auto_pick_place"],
        },
    )
    # #endregion

    return [
        gz_launch,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[gz_bridge, gz_set_pose_bridge]),
        TimerAction(
            period=5.0,
            actions=[
                joint_state_broadcaster_spawner,
                joint_trajectory_controller_spawner,
                move_group,
                *helpers,
                rviz,
                # 次序：锚定 → joint_states → RSP（避免订阅晚于首轮关节消息）
                *static_world_tfs,
                *static_arm_joint_publishers,
                *static_robot_state_publishers,
            ],
        ),
        TimerAction(period=ap_delay, actions=[auto_pick_node], condition=IfCondition(auto_pick)),
        RegisterEventHandler(
            OnProcessStart(
                target_action=auto_pick_node,
                on_start=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log_active(
                                "bringup.launch.py:OnProcessStart(cs612_auto_pick_place)",
                                "process_start",
                                "H2",
                                {"process": "cs612_auto_pick_place"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=move_group,
                on_exit=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnProcessExit(move_group)",
                                "process_exit",
                                "H2",
                                {"process": "move_group"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=auto_pick_node,
                on_exit=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnProcessExit(cs612_auto_pick_place)",
                                "process_exit",
                                "H6",
                                {"process": "cs612_auto_pick_place"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=rviz,
                on_exit=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnProcessExit(rviz2)",
                                "process_exit",
                                "H2",
                                {"process": "rviz2"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=robot_state_publisher,
                on_exit=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnProcessExit(robot_state_publisher)",
                                "process_exit",
                                "H6",
                                {"process": "robot_state_publisher"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnProcessExit(
                target_action=gz_bridge,
                on_exit=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnProcessExit(gz_ros2_bridge)",
                                "process_exit",
                                "H6",
                                {"process": "gz_ros2_bridge"},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ],
            )
        ),
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[
                    OpaqueFunction(
                        function=lambda context: (
                            # #region agent log
                            _debug_log(
                                "bringup.launch.py:OnShutdown",
                                "launch_shutdown",
                                "H2",
                                {"reason": str(getattr(context.locals.event, "reason", "unknown"))},
                            ),
                            # #endregion
                            []
                        )[1]
                    )
                ]
            )
        ),
        *startup_detach_actions,
    ]


def generate_launch_description():
    project_root = _find_project_root()
    moveit_config_dir = Path(get_package_share_directory("cs612_moveit_config"))
    runtime_root = Path("/tmp/cs612_runtime")
    runtime_home = runtime_root / "home"
    runtime_xdg_config = runtime_root / "xdg_config"
    runtime_xdg_cache = runtime_root / "xdg_cache"
    runtime_ros_logs = runtime_root / "ros_logs"
    for path in (runtime_home, runtime_xdg_config, runtime_xdg_cache, runtime_ros_logs):
        path.mkdir(parents=True, exist_ok=True)
    resource_paths = [
        moveit_config_dir / "models",
        project_root / "models",
        project_root / "models" / "gazebo_models",
        project_root / "worlds",
    ]
    return LaunchDescription(
        [
            DeclareLaunchArgument("cs_type", default_value="cs612"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("launch_rviz", default_value="true"),
            DeclareLaunchArgument("auto_pick", default_value="true"),
            DeclareLaunchArgument(
                "auto_pick_delay_sec",
                default_value="25.0",
                description="Launch 后延迟多少秒再启动 cs612_auto_pick_place（调试时可改小以更快产生日志）",
            ),
            DeclareLaunchArgument("launch_gz_gui", default_value="false"),
            DeclareLaunchArgument("rviz_config", default_value="cs612.rviz"),
            SetEnvironmentVariable("HOME", str(runtime_home)),
            SetEnvironmentVariable("XDG_CONFIG_HOME", str(runtime_xdg_config)),
            SetEnvironmentVariable("XDG_CACHE_HOME", str(runtime_xdg_cache)),
            SetEnvironmentVariable("ROS_LOG_DIR", str(runtime_ros_logs)),
            SetEnvironmentVariable("CS612_PROJECT_ROOT", str(project_root)),
            SetEnvironmentVariable(
                "CS612_DEBUG_NDJSON_PATH",
                str((project_root / ".cursor" / "debug-9009e8.log").resolve()),
            ),
            SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            SetEnvironmentVariable("GALLIUM_DRIVER", "llvmpipe"),
            SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", ":".join(str(p) for p in resource_paths if p.is_dir())),
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", ":".join(str(p) for p in resource_paths if p.is_dir())),
            OpaqueFunction(function=_launch_setup),
        ]
    )
