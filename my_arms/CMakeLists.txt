cmake_minimum_required(VERSION 3.16)
project(CS612urdf)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch meshes urdf textures
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
