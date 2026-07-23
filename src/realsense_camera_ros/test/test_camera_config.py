import importlib.util
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "realsense_camera.launch.py"
CONFIG_PATH = PACKAGE_ROOT / "config" / "camera_d435i.json"

spec = importlib.util.spec_from_file_location("realsense_camera_launch", LAUNCH_PATH)
launch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_module)


def test_d435i_config_contract():
    config = launch_module._load_config(CONFIG_PATH)

    assert config["device_type"] == "D435I"
    assert config["rgb_camera.color_profile"] == "1280x720x30"
    assert config["depth_module.depth_profile"] == "1280x720x30"
    assert config["enable_infra"] is False
    assert config["enable_infra1"] is False
    assert config["enable_infra2"] is False
    assert "serial_no" not in config


def test_launch_exposes_only_config_argument():
    launch_description = launch_module.generate_launch_description()
    arguments = [
        action.name
        for action in launch_description.entities
        if isinstance(action, DeclareLaunchArgument)
    ]

    assert arguments == ["config"]


def test_missing_required_parameter_is_rejected(tmp_path):
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="is missing"):
        launch_module._load_config(invalid_config)
