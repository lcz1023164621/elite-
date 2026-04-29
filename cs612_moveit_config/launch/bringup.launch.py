"""CS612 Gazebo + MoveIt bringup using the official Elite CS ROS 2 stack."""
import subprocess
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
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
    launch_gz_gui = LaunchConfiguration("launch_gz_gui")
    helper_modules = {
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

    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )
    joint_trajectory_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

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
        arguments=["-d", str(moveit_config_dir / "config" / "cs612.rviz")],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            {"use_sim_time": use_sim_time},
        ],
        condition=IfCondition(launch_rviz),
    )

    def package_script(name: str, *ros_args: str) -> ExecuteProcess:
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

    return [
        gz_launch,
        robot_state_publisher,
        TimerAction(period=2.0, actions=[spawn_robot]),
        TimerAction(period=4.0, actions=[joint_state_broadcaster, joint_trajectory_controller, gz_bridge]),
        TimerAction(period=5.0, actions=[move_group, *helpers, rviz]),
        TimerAction(period=25.0, actions=[auto_pick_node], condition=IfCondition(auto_pick)),
        *startup_detach_actions,
    ]


def generate_launch_description():
    project_root = _find_project_root()
    runtime_root = Path("/tmp/cs612_runtime")
    runtime_home = runtime_root / "home"
    runtime_xdg_config = runtime_root / "xdg_config"
    runtime_xdg_cache = runtime_root / "xdg_cache"
    runtime_ros_logs = runtime_root / "ros_logs"
    for path in (runtime_home, runtime_xdg_config, runtime_xdg_cache, runtime_ros_logs):
        path.mkdir(parents=True, exist_ok=True)
    resource_paths = [
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
            DeclareLaunchArgument("launch_gz_gui", default_value="false"),
            SetEnvironmentVariable("HOME", str(runtime_home)),
            SetEnvironmentVariable("XDG_CONFIG_HOME", str(runtime_xdg_config)),
            SetEnvironmentVariable("XDG_CACHE_HOME", str(runtime_xdg_cache)),
            SetEnvironmentVariable("ROS_LOG_DIR", str(runtime_ros_logs)),
            SetEnvironmentVariable("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4"),
            SetEnvironmentVariable("LIBGL_ALWAYS_SOFTWARE", "1"),
            SetEnvironmentVariable("GALLIUM_DRIVER", "llvmpipe"),
            SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
            SetEnvironmentVariable("GZ_SIM_RESOURCE_PATH", ":".join(str(p) for p in resource_paths if p.is_dir())),
            SetEnvironmentVariable("IGN_GAZEBO_RESOURCE_PATH", ":".join(str(p) for p in resource_paths if p.is_dir())),
            OpaqueFunction(function=_launch_setup),
        ]
    )
