from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    world = LaunchConfiguration("world")
    return LaunchDescription([
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([
                FindPackageShare("raceline_gazebo"), "worlds", "track.sdf",
            ]),
            description="SDF world exported from pipeline artifacts",
        ),
        ExecuteProcess(
            cmd=["gz", "sim", "-r", world],
            output="screen",
        ),
    ])
