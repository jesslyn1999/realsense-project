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

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("orbbec_camera"),
                    "launch",
                    "dabai.launch.py",
                ]
            )
        ),
        launch_arguments={
            "camera_name": "camera",
            "device_num": "1",
            "uvc_backend": "libuvc",
            "depth_registration": "true",
            "align_mode": "HW",
            "enable_point_cloud": "false",
            "enable_colored_point_cloud": "true",
            "enable_ir": "false",
            "color_qos": "SENSOR_DATA",
            "depth_qos": "SENSOR_DATA",
            "point_cloud_qos": "SENSOR_DATA",
        }.items(),
    )
    scanner = Node(
        package="object_scanner_web_orbbec",
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
    web_server = Node(
        package="object_scanner_web_orbbec",
        executable="web_server",
        parameters=[
            {
                "output_directory": ParameterValue(output_directory),
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
            camera,
            scanner,
            web_server,
        ]
    )
