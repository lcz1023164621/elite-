"""
一键全自动：Gazebo + MoveIt + RViz（bringup）+ TF + 约 8s 后启动 cs612_auto_pick_place（吸盘 DetachableJoint + MoveIt 关节目标）。

社区参考实现（思路同 MoveIt PlanningScene + 轨迹执行 + Gazebo 插件）：
- https://github.com/moveit/moveit2_tutorials — PlanningScene / pick_place 示例
- https://github.com/moveit/moveit_task_constructor — MTC 复杂抓放流水线
- https://github.com/AndrejOrsula/pymoveit2 — 脚本化 MoveIt2 抓放

若只想手动控制启动时机：先 bringup，再另一终端
  ros2 run cs612_moveit_config cs612_auto_pick_place
"""
from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_scene_defaults(share: Path) -> tuple[list[float], list[float], list[float]]:
    rect_center = [-0.82, 0.30, 0.046]
    rect_size = [0.20, 0.14, 0.08]
    carton_pose = [-0.82, 0.30, 0.0]
    cfg = share / "config" / "scene_objects.yaml"
    try:
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        rect = doc.get("rect_pickup") or {}
        carton = doc.get("carton_box") or {}
        rect_center = list(rect.get("center_xyz", rect_center))
        rect_size = list(doc.get("size_xyz", rect_size) if doc.get("size_xyz") else rect.get("size_xyz", rect_size))
        carton_pose = list(carton.get("model_pose_xyz", carton_pose))
    except Exception:
        pass
    return rect_center, rect_size, carton_pose


def generate_launch_description():
    share = Path(get_package_share_directory("cs612_moveit_config"))
    bringup = share / "launch" / "bringup.launch.py"
    rect_center, rect_size, carton_pose = _load_scene_defaults(share)
    map_bridge = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="arm_world_to_map",
        arguments=[
            "--frame-id",
            "arm_world",
            "--child-frame-id",
            "map",
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "0",
            "--qx",
            "0",
            "--qy",
            "0",
            "--qz",
            "0",
            "--qw",
            "1",
        ],
        parameters=[{"use_sim_time": LaunchConfiguration("use_sim_time")}],
        output="log",
    )
    auto_pick = Node(
        package="cs612_moveit_config",
        executable="cs612_auto_pick_place",
        name="cs612_auto_pick_place",
        prefix="/usr/bin/python3",
        output="screen",
        parameters=[
            {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)},
            {"use_direct_xyz": ParameterValue(LaunchConfiguration("use_direct_xyz"), value_type=bool)},
            {"pick_point_xyz": LaunchConfiguration("pick_point_xyz")},
            {"place_point_xyz": LaunchConfiguration("place_point_xyz")},
            {"use_known_rect_surface_center": ParameterValue(LaunchConfiguration("use_known_rect_surface_center"), value_type=bool)},
            {"known_rect_center_xyz": LaunchConfiguration("known_rect_center_xyz")},
            {"known_rect_size_xyz": LaunchConfiguration("known_rect_size_xyz")},
            {"wait_poses_sec": ParameterValue(LaunchConfiguration("wait_poses_sec"), value_type=float)},
            {"require_joint_states": ParameterValue(LaunchConfiguration("require_joint_states"), value_type=bool)},
            {"use_compute_ik": ParameterValue(LaunchConfiguration("use_compute_ik"), value_type=bool)},
            {"use_joint_template_demo": ParameterValue(LaunchConfiguration("use_joint_template_demo"), value_type=bool)},
            {"pose_goal_fallback": ParameterValue(LaunchConfiguration("pose_goal_fallback"), value_type=bool)},
            {"pose_position_tolerance": ParameterValue(LaunchConfiguration("pose_position_tolerance"), value_type=float)},
            {"pose_orientation_tolerance": ParameterValue(LaunchConfiguration("pose_orientation_tolerance"), value_type=float)},
            {"pre_pick_safe_clearance": ParameterValue(LaunchConfiguration("pre_pick_safe_clearance"), value_type=float)},
            {"pre_touch_hover_extra_z": ParameterValue(LaunchConfiguration("pre_touch_hover_extra_z"), value_type=float)},
            {"suction_contact_offset_z": ParameterValue(LaunchConfiguration("suction_contact_offset_z"), value_type=float)},
            {"xy_refine_safe_clearance": ParameterValue(LaunchConfiguration("xy_refine_safe_clearance"), value_type=float)},
            {"move_velocity_scale": ParameterValue(LaunchConfiguration("move_velocity_scale"), value_type=float)},
            {"move_acceleration_scale": ParameterValue(LaunchConfiguration("move_acceleration_scale"), value_type=float)},
            {"suction_attach_lateral_tol": ParameterValue(LaunchConfiguration("suction_attach_lateral_tol"), value_type=float)},
            {"suction_attach_vertical_tol": ParameterValue(LaunchConfiguration("suction_attach_vertical_tol"), value_type=float)},
            {"suction_attach_axis_down_min": ParameterValue(LaunchConfiguration("suction_attach_axis_down_min"), value_type=float)},
            {"suction_attach_burst_count": ParameterValue(LaunchConfiguration("suction_attach_burst_count"), value_type=int)},
            {"suction_attach_burst_interval_sec": ParameterValue(LaunchConfiguration("suction_attach_burst_interval_sec"), value_type=float)},
            {"suction_attach_wait_sec": ParameterValue(LaunchConfiguration("suction_attach_wait_sec"), value_type=float)},
            {"pickup_probe_lift_z": ParameterValue(LaunchConfiguration("pickup_probe_lift_z"), value_type=float)},
            {"pickup_probe_min_follow_z": ParameterValue(LaunchConfiguration("pickup_probe_min_follow_z"), value_type=float)},
            {"touch_cartesian_keep_xy": ParameterValue(LaunchConfiguration("touch_cartesian_keep_xy"), value_type=bool)},
            {"touch_cartesian_step_max_m": ParameterValue(LaunchConfiguration("touch_cartesian_step_max_m"), value_type=float)},
            {
                "touch_cartesian_joint_step_limit_rad": ParameterValue(
                    LaunchConfiguration("touch_cartesian_joint_step_limit_rad"), value_type=float
                )
            },
            {
                "touch_cartesian_orientation_weight": ParameterValue(
                    LaunchConfiguration("touch_cartesian_orientation_weight"), value_type=float
                )
            },
            {"move_to_start_face_pose": ParameterValue(LaunchConfiguration("move_to_start_face_pose"), value_type=bool)},
            {"start_face_use_scene_midpoint_yaw": ParameterValue(LaunchConfiguration("start_face_use_scene_midpoint_yaw"), value_type=bool)},
            {"start_face_joint1_rad": ParameterValue(LaunchConfiguration("start_face_joint1_rad"), value_type=float)},
            {"joint1_world_yaw_offset_rad": ParameterValue(LaunchConfiguration("joint1_world_yaw_offset_rad"), value_type=float)},
            {"start_face_posture_hint": LaunchConfiguration("start_face_posture_hint")},
            {"refresh_top_from_live_pose": ParameterValue(LaunchConfiguration("refresh_top_from_live_pose"), value_type=bool)},
            {"refresh_top_max_delta_xy": ParameterValue(LaunchConfiguration("refresh_top_max_delta_xy"), value_type=float)},
            {"refresh_top_max_delta_z": ParameterValue(LaunchConfiguration("refresh_top_max_delta_z"), value_type=float)},
            {"ik_pick_avoid_collisions": ParameterValue(LaunchConfiguration("ik_pick_avoid_collisions"), value_type=bool)},
            {"ik_timeout_sec": ParameterValue(LaunchConfiguration("ik_timeout_sec"), value_type=float)},
            {"ik_call_wait_sec": ParameterValue(LaunchConfiguration("ik_call_wait_sec"), value_type=float)},
            {"ik_search_wall_time_sec": ParameterValue(LaunchConfiguration("ik_search_wall_time_sec"), value_type=float)},
            {"rect_fallback_pose_xyz": LaunchConfiguration("rect_fallback_pose_xyz")},
            {"rect_fallback_wait_sec": ParameterValue(LaunchConfiguration("rect_fallback_wait_sec"), value_type=float)},
            {"carton_fallback_pose_xyz": LaunchConfiguration("carton_fallback_pose_xyz")},
            {"carton_fallback_wait_sec": ParameterValue(LaunchConfiguration("carton_fallback_wait_sec"), value_type=float)},
            {"hybrid_moveit_pregrasp": ParameterValue(LaunchConfiguration("hybrid_moveit_pregrasp"), value_type=bool)},
            {"hybrid_cartesian_touch_only": ParameterValue(LaunchConfiguration("hybrid_cartesian_touch_only"), value_type=bool)},
            {"pregrasp_xy_align_tol": ParameterValue(LaunchConfiguration("pregrasp_xy_align_tol"), value_type=float)},
            {"pregrasp_alignment_gate_enabled": ParameterValue(LaunchConfiguration("pregrasp_alignment_gate_enabled"), value_type=bool)},
            {"pregrasp_xy_comp_max_step_m": ParameterValue(LaunchConfiguration("pregrasp_xy_comp_max_step_m"), value_type=float)},
            {"pregrasp_xy_comp_gain": ParameterValue(LaunchConfiguration("pregrasp_xy_comp_gain"), value_type=float)},
            {"pregrasp_xy_comp_retries": ParameterValue(LaunchConfiguration("pregrasp_xy_comp_retries"), value_type=int)},
            {
                "centerline_use_object_center_only": ParameterValue(
                    LaunchConfiguration("centerline_use_object_center_only"), value_type=bool
                ),
                "orientation_min_cos_before_touch": ParameterValue(
                    LaunchConfiguration("orientation_min_cos_before_touch"), value_type=float
                ),
                "orientation_correction_weight": ParameterValue(
                    LaunchConfiguration("orientation_correction_weight"), value_type=float
                ),
                "orientation_correction_retries": ParameterValue(
                    LaunchConfiguration("orientation_correction_retries"), value_type=int
                ),
                "touch_delta_z": ParameterValue(LaunchConfiguration("touch_delta_z"), value_type=float),
                "pre_touch_settle_sec": ParameterValue(LaunchConfiguration("pre_touch_settle_sec"), value_type=float),
                "post_touch_settle_sec": ParameterValue(LaunchConfiguration("post_touch_settle_sec"), value_type=float),
                "pre_attach_settle_sec": ParameterValue(LaunchConfiguration("pre_attach_settle_sec"), value_type=float),
            },
            {"dense_waypoint_descent_enabled": ParameterValue(LaunchConfiguration("dense_waypoint_descent_enabled"), value_type=bool)},
            {"dense_waypoint_step_m": ParameterValue(LaunchConfiguration("dense_waypoint_step_m"), value_type=float)},
            {"dense_waypoint_xy_correction_gain": ParameterValue(LaunchConfiguration("dense_waypoint_xy_correction_gain"), value_type=float)},
            {"dense_waypoint_xy_correction_max_m": ParameterValue(LaunchConfiguration("dense_waypoint_xy_correction_max_m"), value_type=float)},
            {"dense_waypoint_orientation_weight": ParameterValue(LaunchConfiguration("dense_waypoint_orientation_weight"), value_type=float)},
            {"dense_waypoint_settle_sec": ParameterValue(LaunchConfiguration("dense_waypoint_settle_sec"), value_type=float)},
            {"dense_waypoint_max_xy_drift_m": ParameterValue(LaunchConfiguration("dense_waypoint_max_xy_drift_m"), value_type=float)},
        ],
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_direct_xyz", default_value="false"),
            DeclareLaunchArgument("pick_point_xyz", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("place_point_xyz", default_value="[0.0, 0.0, 0.0]"),
            DeclareLaunchArgument("use_known_rect_surface_center", default_value="true"),
            DeclareLaunchArgument("known_rect_center_xyz", default_value=str(rect_center)),
            DeclareLaunchArgument("known_rect_size_xyz", default_value=str(rect_size)),
            DeclareLaunchArgument("wait_poses_sec", default_value="90.0"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("require_joint_states", default_value="true"),
            DeclareLaunchArgument("use_compute_ik", default_value="false"),
            DeclareLaunchArgument("use_joint_template_demo", default_value="false"),
            DeclareLaunchArgument("pose_goal_fallback", default_value="true"),
            DeclareLaunchArgument("pose_position_tolerance", default_value="0.01"),
            DeclareLaunchArgument("pose_orientation_tolerance", default_value="0.04"),
            DeclareLaunchArgument("pre_pick_safe_clearance", default_value="0.32"),
            DeclareLaunchArgument("pre_touch_hover_extra_z", default_value="0.04"),
            DeclareLaunchArgument("suction_contact_offset_z", default_value="0.214"),
            DeclareLaunchArgument("move_velocity_scale", default_value="0.35"),
            DeclareLaunchArgument("move_acceleration_scale", default_value="0.35"),
            DeclareLaunchArgument("suction_attach_lateral_tol", default_value="0.045"),
            DeclareLaunchArgument("suction_attach_vertical_tol", default_value="0.040"),
            DeclareLaunchArgument("suction_attach_axis_down_min", default_value="0.92"),
            DeclareLaunchArgument("suction_attach_burst_count", default_value="10"),
            DeclareLaunchArgument("suction_attach_burst_interval_sec", default_value="0.04"),
            DeclareLaunchArgument("suction_attach_wait_sec", default_value="2.8"),
            DeclareLaunchArgument("pickup_probe_lift_z", default_value="0.025"),
            DeclareLaunchArgument("pickup_probe_min_follow_z", default_value="0.010"),
            DeclareLaunchArgument("touch_cartesian_keep_xy", default_value="true"),
            DeclareLaunchArgument("touch_cartesian_step_max_m", default_value="0.004"),
            DeclareLaunchArgument("touch_cartesian_joint_step_limit_rad", default_value="0.04"),
            DeclareLaunchArgument("touch_cartesian_orientation_weight", default_value="2.50"),
            DeclareLaunchArgument("move_to_start_face_pose", default_value="true"),
            DeclareLaunchArgument("start_face_use_scene_midpoint_yaw", default_value="true"),
            DeclareLaunchArgument("start_face_joint1_rad", default_value="0.0"),
            DeclareLaunchArgument("joint1_world_yaw_offset_rad", default_value="-1.5708"),
            DeclareLaunchArgument("start_face_posture_hint", default_value="[0.0, -0.68, 1.02, 0.0, 1.18, 0.0]"),
            DeclareLaunchArgument("refresh_top_from_live_pose", default_value="true"),
            DeclareLaunchArgument("xy_refine_safe_clearance", default_value="0.03"),
            DeclareLaunchArgument("refresh_top_max_delta_xy", default_value="0.20"),
            DeclareLaunchArgument("refresh_top_max_delta_z", default_value="0.08"),
            DeclareLaunchArgument("ik_pick_avoid_collisions", default_value="true"),
            DeclareLaunchArgument("ik_timeout_sec", default_value="5.0"),
            DeclareLaunchArgument("ik_call_wait_sec", default_value="4.0"),
            DeclareLaunchArgument("ik_search_wall_time_sec", default_value="30.0"),
            DeclareLaunchArgument("rect_fallback_pose_xyz", default_value=str(rect_center)),
            DeclareLaunchArgument("rect_fallback_wait_sec", default_value="8.0"),
            DeclareLaunchArgument("carton_fallback_pose_xyz", default_value=str(carton_pose)),
            DeclareLaunchArgument("carton_fallback_wait_sec", default_value="8.0"),
            DeclareLaunchArgument("hybrid_moveit_pregrasp", default_value="true"),
            DeclareLaunchArgument("hybrid_cartesian_touch_only", default_value="true"),
            DeclareLaunchArgument("pregrasp_xy_align_tol", default_value="0.015"),
            DeclareLaunchArgument("pregrasp_alignment_gate_enabled", default_value="true"),
            DeclareLaunchArgument("pregrasp_xy_comp_max_step_m", default_value="0.03"),
            DeclareLaunchArgument("pregrasp_xy_comp_gain", default_value="1.0"),
            DeclareLaunchArgument("pregrasp_xy_comp_retries", default_value="3"),
            DeclareLaunchArgument("centerline_use_object_center_only", default_value="true"),
            DeclareLaunchArgument("orientation_min_cos_before_touch", default_value="0.95"),
            DeclareLaunchArgument("orientation_correction_weight", default_value="3.0"),
            DeclareLaunchArgument("orientation_correction_retries", default_value="2"),
            DeclareLaunchArgument("touch_delta_z", default_value="0.003"),
            DeclareLaunchArgument("pre_touch_settle_sec", default_value="1.5"),
            DeclareLaunchArgument("post_touch_settle_sec", default_value="1.2"),
            DeclareLaunchArgument("pre_attach_settle_sec", default_value="0.6"),
            DeclareLaunchArgument("dense_waypoint_descent_enabled", default_value="true"),
            DeclareLaunchArgument("dense_waypoint_step_m", default_value="0.004"),
            DeclareLaunchArgument("dense_waypoint_xy_correction_gain", default_value="0.85"),
            DeclareLaunchArgument("dense_waypoint_xy_correction_max_m", default_value="0.008"),
            DeclareLaunchArgument("dense_waypoint_orientation_weight", default_value="3.0"),
            DeclareLaunchArgument("dense_waypoint_settle_sec", default_value="0.04"),
            DeclareLaunchArgument("dense_waypoint_max_xy_drift_m", default_value="0.03"),
            # 主系统节点默认使用 CycloneDDS；bridge 统一使用 FastDDS。
            DeclareLaunchArgument("rmw_implementation", default_value="rmw_fastrtps_cpp"),
            SetEnvironmentVariable(
                name="RMW_IMPLEMENTATION",
                value=LaunchConfiguration("rmw_implementation"),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([str(bringup)]),
                launch_arguments={
                    "use_sim_time": LaunchConfiguration("use_sim_time"),
                    "rmw_implementation": LaunchConfiguration("rmw_implementation"),
                }.items(),
            ),
            map_bridge,
            # 尽快自动开始；节点内部仍会等待位姿与服务就绪。
            TimerAction(period=8.0, actions=[auto_pick]),
        ]
    )
