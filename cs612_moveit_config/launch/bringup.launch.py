"""CS612 Gazebo + MoveIt bringup using the official Elite CS ROS 2 stack."""
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from xml.etree import ElementTree as ET

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


def _prefixed_model_name(name: str, prefix: str) -> str:
    if not name or name == "world" or name.startswith(prefix):
        return name
    return f"{prefix}{name}"


def _prefix_srdf(srdf_text: str, *, prefix: str, robot_name: str) -> str:
    root = ET.fromstring(srdf_text)
    root.set("name", robot_name)
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "chain":
            for attr in ("base_link", "tip_link"):
                if attr in elem.attrib:
                    elem.set(attr, _prefixed_model_name(elem.attrib[attr], prefix))
        elif tag in ("link", "joint"):
            if "name" in elem.attrib:
                elem.set("name", _prefixed_model_name(elem.attrib["name"], prefix))
        elif tag == "end_effector":
            if "parent_link" in elem.attrib:
                elem.set("parent_link", _prefixed_model_name(elem.attrib["parent_link"], prefix))
        elif tag == "disable_collisions":
            for attr in ("link1", "link2"):
                if attr in elem.attrib:
                    elem.set(attr, _prefixed_model_name(elem.attrib[attr], prefix))
    return ET.tostring(root, encoding="unicode")


def _prefix_joint_limits(joint_limits_doc: dict, prefix: str) -> dict:
    doc = copy.deepcopy(joint_limits_doc)
    limits = doc.get("joint_limits")
    if isinstance(limits, dict):
        doc["joint_limits"] = {
            _prefixed_model_name(str(name), prefix): value
            for name, value in limits.items()
        }
    return doc


def _prefix_ompl_config(ompl_cfg: dict, prefix: str) -> dict:
    cfg = copy.deepcopy(ompl_cfg)
    arm_cfg = cfg.get("arm")
    if isinstance(arm_cfg, dict):
        projection = arm_cfg.get("projection_evaluator")
        if isinstance(projection, str) and projection.startswith("joints(") and projection.endswith(")"):
            joints = [j.strip() for j in projection[len("joints("):-1].split(",") if j.strip()]
            prefixed_joints = ",".join(_prefixed_model_name(j, prefix) for j in joints)
            arm_cfg["projection_evaluator"] = f"joints({prefixed_joints})"
    return cfg


def _prefixed_moveit_controllers(prefix: str) -> dict:
    controller_name = f"{prefix.rstrip('_')}_joint_trajectory_controller"
    return {
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
        "moveit_simple_controller_manager": {
            "controller_names": [controller_name],
            controller_name: {
                "type": "FollowJointTrajectory",
                "action_ns": "follow_joint_trajectory",
                "default": True,
                "joints": [_prefixed_model_name(j, prefix) for j in _ARM_JOINTS],
            },
        },
    }


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
    cs612_2_controllers_file = moveit_config_dir / "config" / "cs612_2_ros2_controllers.yaml"

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
            "controller_manager_name:=controller_manager",
            "robot_param_node:=robot_state_publisher",
        ],
        text=True,
    )
    cs612_2_description_content = subprocess.check_output(
        [
            "xacro",
            str(xacro_file),
            f"cs_type:={cs_type.perform(context)}",
            "name:=cs612_2",
            "prefix:=cs612_2_",
            "sim_ignition:=true",
            f"simulation_controllers:={cs612_2_controllers_file}",
            "controller_manager_name:=cs612_2_controller_manager",
            "robot_param_node:=/cs612_2/robot_state_publisher",
            "suction_topic_ns:=/cs612_2/suction",
            "base_origin_xyz:=2.30000 -0.60000 0.00141",
        ],
        text=True,
    )
    srdf_text = (moveit_config_dir / "config" / "CS612.srdf").read_text(encoding="utf-8")
    robot_description = {"robot_description": robot_description_content}
    cs612_2_robot_description = {"robot_description": cs612_2_description_content}
    robot_description_semantic = {
        "robot_description_semantic": srdf_text
    }
    cs612_2_robot_description_semantic = {
        "robot_description_semantic": _prefix_srdf(srdf_text, prefix="cs612_2_", robot_name="cs612")
    }
    robot_description_kinematics = {"robot_description_kinematics": _load_yaml(moveit_config_dir / "config" / "kinematics.yaml")}
    robot_description_planning = {"robot_description_planning": _load_yaml(moveit_config_dir / "config" / "joint_limits.yaml")}
    cs612_2_robot_description_kinematics = copy.deepcopy(robot_description_kinematics)
    cs612_2_robot_description_planning = {
        "robot_description_planning": _prefix_joint_limits(robot_description_planning["robot_description_planning"], "cs612_2_")
    }
    ompl_cfg = _load_yaml(moveit_config_dir / "config" / "ompl_planning.yaml")
    cs612_2_ompl_cfg = _prefix_ompl_config(ompl_cfg, "cs612_2_")
    moveit_controllers = _load_yaml(moveit_config_dir / "config" / "moveit_controllers.yaml")
    cs612_2_moveit_controllers = _prefixed_moveit_controllers("cs612_2_")

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
    spawn_robot2 = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=[
            "-string",
            cs612_2_description_content,
            "-name",
            "cs612_2",
            "-allow_renaming",
            "false",
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.0",
        ],
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

    cs612_2_robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace="cs612_2",
        output="both",
        parameters=[
            cs612_2_robot_description,
            {"use_sim_time": use_sim_time},
            {"publish_robot_description": True},
        ],
        remappings=[
            ("joint_states", "/cs612_2_joint_state_broadcaster/joint_states"),
            ("tf", "/tf"),
            ("tf_static", "/tf_static"),
        ],
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
            "--param-file",
            str(controllers_file),
            "--switch-timeout",
            "20.0",
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
            "--param-file",
            str(controllers_file),
            "--switch-timeout",
            "20.0",
        ],
        output="screen",
    )
    cs612_2_joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "cs612_2_joint_state_broadcaster",
            "--controller-manager",
            "/cs612_2_controller_manager",
            "--param-file",
            str(cs612_2_controllers_file),
            "--switch-timeout",
            "20.0",
        ],
        output="screen",
    )
    cs612_2_joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "cs612_2_joint_trajectory_controller",
            "--controller-manager",
            "/cs612_2_controller_manager",
            "--param-file",
            str(cs612_2_controllers_file),
            "--switch-timeout",
            "20.0",
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
    cs612_2_move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        namespace="cs612_2",
        name="move_group",
        output="screen",
        parameters=[
            cs612_2_robot_description,
            cs612_2_robot_description_semantic,
            cs612_2_robot_description_kinematics,
            cs612_2_robot_description_planning,
            {"planning_pipelines": ["ompl"]},
            {"ompl": cs612_2_ompl_cfg},
            cs612_2_moveit_controllers,
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
        remappings=[
            ("joint_states", "/cs612_2_joint_state_broadcaster/joint_states"),
            (
                "cs612_2_joint_trajectory_controller/follow_joint_trajectory",
                "/cs612_2_joint_trajectory_controller/follow_joint_trajectory",
            ),
            (
                "/cs612_2/cs612_2_joint_trajectory_controller/follow_joint_trajectory",
                "/cs612_2_joint_trajectory_controller/follow_joint_trajectory",
            ),
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
            "cs612_2_move_group": True,
            "main_joint_state_bridge": False,
            "main_trajectory_bridge": False,
            "ros2_control_spawners": [
                "joint_state_broadcaster",
                "joint_trajectory_controller",
                "cs612_2_joint_state_broadcaster",
                "cs612_2_joint_trajectory_controller",
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
        cs612_2_robot_state_publisher,
        TimerAction(period=2.0, actions=[spawn_robot, spawn_robot2]),
        TimerAction(period=4.0, actions=[gz_bridge, gz_set_pose_bridge]),
        TimerAction(
            period=5.0,
            actions=[
                joint_state_broadcaster_spawner,
                joint_trajectory_controller_spawner,
            ],
        ),
        TimerAction(
            period=9.0,
            actions=[
                cs612_2_joint_state_broadcaster_spawner,
                cs612_2_joint_trajectory_controller_spawner,
            ],
        ),
        TimerAction(
            period=11.0,
            actions=[
                move_group,
                cs612_2_move_group,
                *helpers,
                rviz,
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
