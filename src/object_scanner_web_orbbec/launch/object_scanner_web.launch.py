import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


_REQUIRED_CAMERA_PARAMETERS = {
    "camera_name": str,
    "device_num": int,
    "uvc_backend": str,
    "depth_registration": bool,
    "align_mode": str,
    "enable_point_cloud": bool,
    "enable_colored_point_cloud": bool,
    "enable_ir": bool,
    "color_qos": str,
    "depth_qos": str,
    "point_cloud_qos": str,
    "enable_frame_sync": bool,
    "frame_aggregate_mode": str,
}


def _load_camera_config(config_path):
    path = Path(config_path).expanduser()
    try:
        with path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load Orbbec camera config '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise RuntimeError(f"Orbbec camera config '{path}' must contain a JSON object")

    missing = sorted(_REQUIRED_CAMERA_PARAMETERS.keys() - config.keys())
    if missing:
        raise RuntimeError(
            f"Orbbec camera config '{path}' is missing: {', '.join(missing)}"
        )

    for name, expected_type in _REQUIRED_CAMERA_PARAMETERS.items():
        if not isinstance(config[name], expected_type):
            raise RuntimeError(
                f"Orbbec camera config parameter '{name}' must be "
                f"{expected_type.__name__}"
            )

    return config


def _launch_argument(value):
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _launch_camera(context):
    config = _load_camera_config(LaunchConfiguration("camera_config").perform(context))
    return [
        IncludeLaunchDescription(
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
                name: _launch_argument(value) for name, value in config.items()
            }.items(),
        )
    ]


def generate_launch_description():
    output_directory = LaunchConfiguration("output_directory")
    target_rgb = LaunchConfiguration("target_rgb")
    lab_threshold = LaunchConfiguration("lab_threshold")
    default_camera_config = (
        Path(get_package_share_directory("object_scanner_web_orbbec"))
        / "config"
        / "camera_dabai_dc1.json"
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
            DeclareLaunchArgument(
                "camera_config",
                default_value=str(default_camera_config),
            ),
            OpaqueFunction(function=_launch_camera),
            scanner,
            web_server,
        ]
    )
