"""
稳定演示入口（推荐）：
- 主系统节点使用 CycloneDDS，bridge 统一使用 FastDDS(UDPv4)
- 复用 full_auto_pick 全流程参数与自动抓放节点
"""
from pathlib import Path
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def _load_scene_defaults(share: Path) -> tuple[str, str, str]:
    rect_center = [0.68, 0.16, 0.03]
    rect_size = [0.14, 0.10, 0.06]
    carton_pose = [0.82, -0.32, 0.0]
    cfg = share / "config" / "scene_objects.yaml"
    try:
        doc = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        rect = doc.get("rect_pickup") or {}
        carton = doc.get("carton_box") or {}
        rect_center = list(rect.get("center_xyz", rect_center))
        rect_size = list(rect.get("size_xyz", rect_size))
        carton_pose = list(carton.get("model_pose_xyz", carton_pose))
    except Exception:
        pass
    return str(rect_center), str(rect_size), str(carton_pose)


def generate_launch_description():
    share = Path(get_package_share_directory("cs612_moveit_config"))
    full_auto = share / "launch" / "full_auto_pick.launch.py"
    rect_center, rect_size, carton_pose = _load_scene_defaults(share)
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([str(full_auto)]),
                launch_arguments={
                    # 与系统 ros_gz_bridge（FastDDS）对齐，避免跨 RMW/跨发行版时的反序列化异常。
                    "rmw_implementation": "rmw_fastrtps_cpp",
                    "use_sim_time": "false",
                    "wait_poses_sec": "90.0",
                    "use_known_rect_surface_center": "true",
                    "known_rect_center_xyz": rect_center,
                    "known_rect_size_xyz": rect_size,
                    "rect_fallback_pose_xyz": rect_center,
                    "carton_fallback_pose_xyz": carton_pose,
                    "rect_fallback_wait_sec": "8.0",
                    "carton_fallback_wait_sec": "8.0",
                    "move_velocity_scale": "0.12",
                    "move_acceleration_scale": "0.12",
                    "pre_pick_safe_clearance": "0.32",
                    "pre_touch_hover_extra_z": "0.06",
                    "touch_delta_z": "0.006",
                    "suction_contact_offset_z": "0.214",
                    "xy_refine_safe_clearance": "0.03",
                    "move_to_start_face_pose": "false",
                    "start_face_use_scene_midpoint_yaw": "true",
                    "start_face_joint1_rad": "0.0",
                    "joint1_world_yaw_offset_rad": "-1.5708",
                    "refresh_top_from_live_pose": "true",
                    "suction_attach_lateral_tol": "0.045",
                    "suction_attach_vertical_tol": "0.040",
                    "suction_attach_axis_down_min": "0.92",
                    "suction_touch_lateral_tol": "0.015",
                    "suction_touch_vertical_tol": "0.022",
                    "suction_touch_axis_down_min": "0.95",
                    "suction_attach_burst_count": "10",
                    "suction_attach_burst_interval_sec": "0.04",
                    "suction_attach_wait_sec": "3.2",
                    "pickup_probe_lift_z": "0.030",
                    "pickup_probe_min_follow_z": "0.012",
                    "pickup_probe_require_follow_if_live_pose": "true",
                    "touch_cartesian_keep_xy": "true",
                    "touch_cartesian_step_max_m": "0.0035",
                    "touch_cartesian_joint_step_limit_rad": "0.030",
                    "touch_cartesian_orientation_weight": "2.50",
                    "cartesian_settle_timeout_sec": "3.2",
                    "cartesian_settle_tol_rad": "0.035",
                    # 关键：禁用历史硬编码模板路径，启用按目标点 IK 抓取
                    "use_joint_template_demo": "false",
                    "use_compute_ik": "false",
                    "hybrid_moveit_pregrasp": "true",
                    "hybrid_cartesian_touch_only": "true",
                    "pregrasp_xy_align_tol": "0.015",
                    "pregrasp_alignment_gate_enabled": "true",
                    "pregrasp_xy_comp_max_step_m": "0.015",
                    "pregrasp_xy_comp_gain": "0.80",
                    "pregrasp_xy_comp_retries": "4",
                    "pregrasp_cartesian_center_enabled": "true",
                    "centerline_use_object_center_only": "true",
                    "orientation_min_cos_before_touch": "0.95",
                    "orientation_correction_weight": "3.0",
                    "orientation_correction_retries": "2",
                    "pre_touch_settle_sec": "1.5",
                    "post_touch_settle_sec": "1.2",
                    "pre_attach_settle_sec": "0.8",
                    "ik_pick_avoid_collisions": "true",
                    "ik_search_wall_time_sec": "30.0",
                    "ik_call_wait_sec": "6.0",
                    "pose_goal_fallback": "false",
                    "pose_position_tolerance": "0.004",
                    "pose_orientation_tolerance": "0.03",
                }.items(),
            )
        ]
    )
