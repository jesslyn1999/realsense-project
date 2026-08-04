from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration("config")
    output_directory = LaunchConfiguration("output_directory")
    target_rgb = LaunchConfiguration("target_rgb")
    lab_threshold = LaunchConfiguration("lab_threshold")

    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("realsense_camera_ros"),
                    "launch",
                    "realsense_camera.launch.py",
                ]
            )
        ),
        launch_arguments={"config": config}.items(),
    )
    scanner = Node(
        package="object_scanner",
        executable="scanner_node",
        parameters=[
            {
                "output_directory": output_directory,
                "target_rgb": ParameterValue(target_rgb),
                "lab_threshold": ParameterValue(
                    lab_threshold,
                    value_type=float,
                ),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("realsense_camera_ros"),
                        "config",
                        "camera_d435i.json",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "output_directory",
                default_value="scans",
            ),
            DeclareLaunchArgument(
                "target_rgb",
                default_value="[0, 255, 0]",
            ),
            DeclareLaunchArgument(
                "lab_threshold",
                default_value="15.0",
            ),
            camera_launch,
            scanner,
        ]
    )
