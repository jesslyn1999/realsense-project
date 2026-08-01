import math
import struct

from geometry_msgs.msg import TransformStamped
import numpy as np
from object_scanner.pointcloud import (
    filter_colored_points,
    transform_filtered_cloud,
    transform_to_matrix,
)
import pytest
from sensor_msgs.msg import PointCloud2, PointField


def make_cloud(points):
    message = PointCloud2()
    message.height = 1
    message.width = len(points)
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = message.width * message.point_step
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    data = bytearray()
    for x, y, z, (red, green, blue) in points:
        packed = (red << 16) | (green << 8) | blue
        packed_float = struct.unpack("<f", struct.pack("<I", packed))[0]
        data.extend(struct.pack("<ffff", x, y, z, packed_float))
    message.data = bytes(data)
    return message


def test_green_filter_keeps_close_finite_points():
    cloud = make_cloud(
        [
            (0.0, 0.0, 0.5, (0, 255, 0)),
            (0.1, 0.0, 0.5, (0, 240, 0)),
            (0.2, 0.0, 0.5, (255, 0, 0)),
            (0.3, 0.0, -0.5, (0, 255, 0)),
            (float("nan"), 0.0, 0.5, (0, 255, 0)),
        ]
    )

    xyz, rgb = filter_colored_points(cloud, (0, 255, 0), 15.0)

    np.testing.assert_allclose(xyz, [[0.0, 0.0, 0.5], [0.1, 0.0, 0.5]])
    np.testing.assert_array_equal(rgb, [[0, 255, 0], [0, 240, 0]])


def test_transform_rotates_translates_and_preserves_color():
    transform = TransformStamped()
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    transform.transform.translation.z = 3.0
    transform.transform.rotation.z = math.sin(math.pi / 4)
    transform.transform.rotation.w = math.cos(math.pi / 4)
    xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    rgb = np.array([[0, 255, 0], [0, 240, 0]], dtype=np.uint8)

    matrix = transform_to_matrix(transform)
    world_xyz, world_rgb = transform_filtered_cloud(matrix, xyz, rgb)

    np.testing.assert_allclose(
        world_xyz,
        [[1.0, 3.0, 3.0], [0.0, 2.0, 3.0]],
        atol=1e-6,
    )
    np.testing.assert_array_equal(world_rgb, rgb)


def test_invalid_transforms_are_rejected():
    transform = TransformStamped()
    transform.transform.rotation.w = 0.0
    with pytest.raises(ValueError, match="non-zero"):
        transform_to_matrix(transform)

    with pytest.raises(ValueError, match="homogeneous"):
        transform_filtered_cloud(
            np.zeros((4, 4)),
            np.empty((0, 3)),
            np.empty((0, 3)),
        )
