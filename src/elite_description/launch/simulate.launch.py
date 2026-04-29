from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch import LaunchDescription

def generate_launch_description():
    DeclareLaunchArgument("model", default_value="ec66")
    model = LaunchConfiguration("model")

    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare("elite_description"), '/launch', '/simulate_ec63.launch.py']), 
                                condition=IfCondition(PythonExpression(["'", model, "' == 'ec63'"]))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare("elite_description"), '/launch', '/simulate_ec64.launch.py']), 
                                condition=IfCondition(PythonExpression(["'", model, "' == 'ec64'"]))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare("elite_description"), '/launch', '/simulate_ec66.launch.py']), 
                                 condition=IfCondition(PythonExpression(["'", model, "' == 'ec66'"]))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare("elite_description"), '/launch', '/simulate_ec68.launch.py']), 
                                 condition=IfCondition(PythonExpression(["'", model, "' == 'ec68'"]))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource([FindPackageShare("elite_description"), '/launch', '/simulate_ec612.launch.py']), 
                                 condition=IfCondition(PythonExpression(["'", model, "' == 'ec612'"])))
    ])

