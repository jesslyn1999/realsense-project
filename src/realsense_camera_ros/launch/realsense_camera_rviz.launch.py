from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = Path(get_package_share_directory("realsense_camera_ros"))
    default_config = package_share / "config" / "camera_d435i.json"

    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(package_share / "launch" / "realsense_camera.launch.py")
                ),
                launch_arguments={
                    "config": LaunchConfiguration("config"),
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    str(package_share / "config" / "camera_d435i.rviz"),
                ],
                output="screen",
            ),
        ]
    )
