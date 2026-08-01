from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    output_directory = LaunchConfiguration("output_directory")
    target_rgb = LaunchConfiguration("target_rgb")
    lab_threshold = LaunchConfiguration("lab_threshold")

    scanner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("object_scanner"),
                    "launch",
                    "object_scanner.launch.py",
                ]
            )
        ),
        launch_arguments={
            "output_directory": output_directory,
            "target_rgb": target_rgb,
            "lab_threshold": lab_threshold,
        }.items(),
    )
    web_server = Node(
        package="object_scanner_web",
        executable="web_server",
        parameters=[
            {
                "target_rgb": ParameterValue(target_rgb),
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
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
            scanner,
            web_server,
        ]
    )
