cmake_minimum_required(VERSION 3.22)
project(elite_moveit_ec612)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/auto_pick_place.py
  scripts/gz_pose_bridge.py
  scripts/joint_states_bridge.py
  scripts/planning_scene_spawner.py
  scripts/trajectory_action_bridge.py
  scripts/world_markers.py
  scripts/joint_diag.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS scripts/auto_pick_place.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_auto_pick_place)
install(PROGRAMS scripts/planning_scene_spawner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_planning_scene_spawner)
install(PROGRAMS scripts/gz_pose_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_gz_pose_bridge)
install(PROGRAMS scripts/joint_states_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_joint_states_bridge)
install(PROGRAMS scripts/trajectory_action_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_trajectory_action_bridge)
install(PROGRAMS scripts/world_markers.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_world_markers)
install(PROGRAMS scripts/joint_diag.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME cs612_joint_diag)

ament_package()

install(DIRECTORY launch DESTINATION share/${PROJECT_NAME}
  PATTERN "setup_assistant.launch" EXCLUDE)
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})
install(FILES .setup_assistant DESTINATION share/${PROJECT_NAME})
