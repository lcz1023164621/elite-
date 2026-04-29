"""
ros_gz_bridge launch file：桥接 Gazebo Sim 与 ROS2 的关节状态和命令话题
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bridge_params = [
        "/world/arm_world/model/cs612/joint_state"
        "@sensor_msgs/msg/JointState[gz.msgs.Model",
        "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
    ]

    gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="gz_ros2_bridge",
        arguments=bridge_params,
        remappings=[
            (
                "/world/arm_world/model/cs612/joint_state",
                "/joint_states",
            ),
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    return LaunchDescription([gz_bridge])
