import importlib.util
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument


PACKAGE_ROOT = Path(__file__).parents[1]
LAUNCH_PATH = PACKAGE_ROOT / "launch" / "object_scanner_web.launch.py"
CONFIG_PATH = PACKAGE_ROOT / "config" / "camera_dabai_dc1.json"

spec = importlib.util.spec_from_file_location("object_scanner_web_orbbec_launch", LAUNCH_PATH)
launch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch_module)


def test_dabai_dc1_config_contract():
    config = launch_module._load_camera_config(CONFIG_PATH)

    assert config["depth_registration"] is True
    assert config["align_mode"] == "HW"
    assert config["enable_colored_point_cloud"] is True
    assert config["enable_frame_sync"] is True
    assert config["frame_aggregate_mode"] == "full_frame"


def test_launch_exposes_camera_config_argument():
    launch_description = launch_module.generate_launch_description()
    arguments = [
        action.name
        for action in launch_description.entities
        if isinstance(action, DeclareLaunchArgument)
    ]

    assert arguments == [
        "output_directory",
        "target_rgb",
        "lab_threshold",
        "camera_config",
    ]


def test_missing_required_camera_parameter_is_rejected(tmp_path):
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="is missing"):
        launch_module._load_camera_config(invalid_config)
