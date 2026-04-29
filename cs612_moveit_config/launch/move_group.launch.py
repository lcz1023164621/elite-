"""
MoveIt move_group launch file for CS612 机械臂

- 默认 use_gazebo=true：与「以前一起出 Gazebo + RViz」一致，内部直接包含 bringup.launch.py
  （Gazebo Sim、ros_gz_bridge、关节桥、MoveIt、RViz）。
- use_gazebo=false：仅本机 MoveIt + RViz，无仿真；此时 use_sim_time 须为 false（默认），否则无 /clock。

也可继续直接使用：ros2 launch cs612_moveit_config bringup.launch.py
"""
import os
from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _get_project_paths(launch_file: Path) -> tuple[Path, Path]:
    """
    Returns (project_root, moveit_config_dir).
    After colcon install, URDF/meshes live under package share; in a source tree,
    project_root is the repo root and moveit_config_dir is cs612_moveit_config/.
    """
    launch_file = launch_file.resolve()
    urdf_rel = Path("my_arms/urdf/CS612urdf.urdf")

    # 优先使用源码树（避免 install/share 与源码未同步时读取旧模型）
    source_candidates: list[Path] = []
    env_root = os.environ.get("CS612_PROJECT_ROOT", "").strip()
    if env_root:
        source_candidates.append(Path(env_root).expanduser().resolve())
    source_candidates.append(Path.cwd().resolve())
    for parent in launch_file.parents:
        if parent.name == "install":
            source_candidates.append(parent.parent.resolve())
            break

    seen: set[Path] = set()
    for cand in source_candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if (cand / urdf_rel).is_file() and (cand / "cs612_moveit_config").is_dir():
            return cand, cand / "cs612_moveit_config"

    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("cs612_moveit_config"))
        if (share / urdf_rel).is_file():
            return share, share
    except Exception:
        pass

    moveit_config_dir = launch_file.parent.parent
    project_root = moveit_config_dir.parent
    if (project_root / "my_arms" / "urdf" / "CS612urdf.urdf").is_file():
        return project_root, moveit_config_dir

    raise FileNotFoundError(
        "找不到 my_arms/urdf/CS612urdf.urdf。请先在工作空间执行 colcon build 并 source install/setup.bash，"
        "或确认 my_arms 与 cs612_moveit_config 位于同一工程根目录下。"
    )


def _load_urdf_with_mesh_paths(project_root: Path) -> str:
    urdf_path = project_root / "my_arms" / "urdf" / "CS612urdf.urdf"
    text = urdf_path.read_text(encoding="utf-8")
    mesh_uri = (project_root / "my_arms" / "meshes").resolve().as_uri()
    return text.replace("package://CS612urdf/meshes/", mesh_uri + "/")


def _launch_setup(context, *args, **kwargs):
    launch_file = Path(__file__)
    project_root, moveit_config_dir = _get_project_paths(launch_file)

    use_gazebo_str = LaunchConfiguration("use_gazebo").perform(context)
    if use_gazebo_str.lower() in ("true", "1", "yes"):
        bringup_py = launch_file.resolve().parent / "bringup.launch.py"
        if not bringup_py.is_file():
            raise FileNotFoundError(f"找不到仿真启动文件: {bringup_py}")
        return [
            IncludeLaunchDescription(PythonLaunchDescriptionSource([str(bringup_py)])),
        ]

    use_sim_time_str = LaunchConfiguration("use_sim_time").perform(context)
    use_sim_time = use_sim_time_str.lower() in ("true", "1", "yes")

    robot_description_content = _load_urdf_with_mesh_paths(project_root)
    robot_description = {"robot_description": robot_description_content}

    srdf_file = moveit_config_dir / "config" / "CS612.srdf"
    robot_description_semantic = {
        "robot_description_semantic": srdf_file.read_text(encoding="utf-8")
    }

    kinematics_yaml = moveit_config_dir / "config" / "kinematics.yaml"
    with open(kinematics_yaml, "r", encoding="utf-8") as f:
        robot_description_kinematics = yaml.safe_load(f)

    with open(moveit_config_dir / "config" / "ompl_planning.yaml", "r", encoding="utf-8") as f:
        ompl_cfg = yaml.safe_load(f)

    with open(moveit_config_dir / "config" / "joint_limits.yaml", "r", encoding="utf-8") as f:
        joint_limits_doc = yaml.safe_load(f)

    urdf_path = project_root / "my_arms" / "urdf" / "CS612urdf.urdf"
    if not urdf_path.is_file():
        raise FileNotFoundError(f"找不到 URDF 文件: {urdf_path}")

    moveit_simple_controller_manager = {
        "controller_names": ["joint_trajectory_controller"],
        "joint_trajectory_controller": {
            "type": "FollowJointTrajectory",
            "action_ns": "follow_joint_trajectory",
            "default": True,
            "joints": [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        },
    }

    # joint_state_publisher 在 ROS2 Humble 中：若不传 URDF 文件路径，则不会使用 launch 里的
    # robot_description 参数，而是订阅 /robot_description 话题；若未及时收到则永远不发布
    # /joint_states，导致 TF 断链。传入 URDF 绝对路径可从磁盘直接解析关节并立即发布。
    # rate 在源码中为 PARAMETER_INTEGER，勿传浮点。
    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        arguments=[str(urdf_path.resolve())],
        parameters=[
            {"use_sim_time": use_sim_time},
            {"rate": 50},
        ],
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {"use_sim_time": use_sim_time},
            {"ignore_timestamp": True},
            {"publish_frequency": 50.0},
            {"publish_robot_description": True},
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

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": robot_description_kinematics},
            {"robot_description_planning": joint_limits_doc},
            {"planning_pipelines": ["ompl"]},
            {"ompl": ompl_cfg},
            {"moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager"},
            {"moveit_simple_controller_manager": moveit_simple_controller_manager},
            {"publish_robot_description_semantic": True},
            {"publish_robot_description": False},
            {"allow_trajectory_execution": True},
            {"capabilities": ""},
            {"disable_capabilities": ""},
            {"monitor_dynamics": False},
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_config = moveit_config_dir / "config" / "cs612.rviz"
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", str(rviz_config)] if rviz_config.is_file() else [],
        parameters=[
            robot_description,
            robot_description_semantic,
            {"robot_description_kinematics": robot_description_kinematics},
            {"use_sim_time": use_sim_time},
        ],
    )

    return [
        map_to_base_tf,
        joint_state_publisher,
        robot_state_publisher,
        move_group_node,
        rviz_node,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_gazebo",
                default_value="true",
                description="true：Gazebo + RViz + MoveIt（等同 bringup）；false：仅 RViz + MoveIt。",
            ),
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="false",
                description="仅在 use_gazebo=false 时生效；无 Gazebo 时必须为 false。",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
