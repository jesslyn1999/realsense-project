import json
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


_REQUIRED_PARAMETERS = {
    "camera_name": str,
    "camera_namespace": str,
    "device_type": str,
    "enable_color": bool,
    "rgb_camera.color_profile": str,
    "enable_depth": bool,
    "depth_module.depth_profile": str,
}


def _load_config(config_path):
    path = Path(config_path).expanduser()
    try:
        with path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load RealSense config '{path}': {exc}") from exc

    if not isinstance(config, dict):
        raise RuntimeError(f"RealSense config '{path}' must contain a JSON object")

    missing = sorted(_REQUIRED_PARAMETERS.keys() - config.keys())
    if missing:
        raise RuntimeError(
            f"RealSense config '{path}' is missing: {', '.join(missing)}"
        )

    for name, expected_type in _REQUIRED_PARAMETERS.items():
        if not isinstance(config[name], expected_type):
            raise RuntimeError(
                f"RealSense config parameter '{name}' must be "
                f"{expected_type.__name__}"
            )

    return config


def _launch_camera(context):
    config = _load_config(LaunchConfiguration("config").perform(context))
    return [
        Node(
            package="realsense2_camera",
            executable="realsense2_camera_node",
            namespace=config["camera_namespace"],
            name=config["camera_name"],
            parameters=[config],
            output="screen",
        )
    ]


def generate_launch_description():
    default_config = (
        Path(get_package_share_directory("realsense_camera_ros"))
        / "config"
        / "camera_d435i.json"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=str(default_config)),
            OpaqueFunction(function=_launch_camera),
        ]
    )
