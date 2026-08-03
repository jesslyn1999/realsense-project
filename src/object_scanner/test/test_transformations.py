import json
from pathlib import Path

from builtin_interfaces.msg import Time
import numpy as np
from object_scanner.transformations import (
    load_transformation_matrices,
    transformation_to_message,
)
import pytest


RESOURCE_PATH = (
    Path(__file__).parents[1] / "resource" / "transformation_matrices.json"
)


def test_identity_resource_loads_and_converts_to_stamped_message():
    transformations = load_transformation_matrices(RESOURCE_PATH)
    stamp = Time(sec=12, nanosec=34)

    assert [item.name for item in transformations] == ["identity"]
    message = transformation_to_message(
        transformations[0],
        stamp,
        "camera0_depth_optical_frame",
    )
    assert message.header.stamp == stamp
    assert message.header.frame_id == "world"
    assert message.child_frame_id == "camera0_depth_optical_frame"
    assert message.transformation_name == "identity"
    np.testing.assert_array_equal(
        np.asarray(message.matrix).reshape(4, 4),
        np.eye(4),
    )


def test_rotation_and_translation_round_trip(tmp_path):
    expected = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    path = tmp_path / "transforms.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "quarter_turn",
                    "parent_frame_id": "world",
                    "matrix": expected.tolist(),
                }
            ]
        ),
        encoding="utf-8",
    )

    transformation = load_transformation_matrices(path)[0]
    message = transformation_to_message(
        transformation,
        Time(),
        "camera",
    )

    assert message.transformation_name == "quarter_turn"
    np.testing.assert_array_equal(
        np.asarray(message.matrix).reshape(4, 4),
        expected,
    )


def test_invalid_rotation_is_rejected(tmp_path):
    path = tmp_path / "transforms.json"
    path.write_text(
        json.dumps(
            [
                {
                    "name": "scaled",
                    "parent_frame_id": "world",
                    "matrix": np.diag([2.0, 1.0, 1.0, 1.0]).tolist(),
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="proper rotation"):
        load_transformation_matrices(path)
