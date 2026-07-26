import importlib.util
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.actions import Node


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "realsense_camera.launch.py"
RVIZ_LAUNCH_PATH = PACKAGE_ROOT / "launch" / "realsense_camera_rviz.launch.py"
CONFIG_PATH = PACKAGE_ROOT / "config" / "camera_d435i.json"
RVIZ_CONFIG_PATH = PACKAGE_ROOT / "config" / "camera_d435i.rviz"

spec = importlib.util.spec_from_file_location("realsense_camera_launch", LAUNCH_PATH)
launch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_module)

rviz_spec = importlib.util.spec_from_file_location(
    "realsense_camera_rviz_launch", RVIZ_LAUNCH_PATH
)
rviz_launch_module = importlib.util.module_from_spec(rviz_spec)
rviz_spec.loader.exec_module(rviz_launch_module)


def test_d435i_config_contract():
    config = launch_module._load_config(CONFIG_PATH)

    assert config["device_type"] == "D435I"
    assert config["rgb_camera.color_profile"] == "1280x720x30"
    assert config["depth_module.depth_profile"] == "848x480x30"
    assert config["depth_module.infra_profile"] == "848x480x30"
    assert config["enable_infra"] is False
    assert config["enable_infra1"] is True
    assert config["enable_infra2"] is True
    assert config["pointcloud.enable"] is True
    assert "serial_no" not in config


def test_launch_exposes_only_config_argument():
    launch_description = launch_module.generate_launch_description()
    arguments = [
        action.name
        for action in launch_description.entities
        if isinstance(action, DeclareLaunchArgument)
    ]

    assert arguments == ["config"]


def test_rviz_wrapper_starts_camera_and_rviz():
    entities = rviz_launch_module.generate_launch_description().entities

    assert any(isinstance(entity, IncludeLaunchDescription) for entity in entities)
    assert any(isinstance(entity, Node) for entity in entities)


def test_rviz_config_uses_camera_topics():
    rviz_config = RVIZ_CONFIG_PATH.read_text(encoding="utf-8")

    assert "/realsense/camera0/depth/color/points" in rviz_config
    assert "/realsense/camera0/color/image_raw" in rviz_config
    assert "/realsense/camera0/depth/image_rect_raw" in rviz_config
    assert "/realsense/camera0/infra1/image_rect_raw" in rviz_config
    assert "/realsense/camera0/infra2/image_rect_raw" in rviz_config


def test_missing_required_parameter_is_rejected(tmp_path):
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="is missing"):
        launch_module._load_config(invalid_config)
