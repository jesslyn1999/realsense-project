from array import array
import sqlite3
import struct

from geometry_msgs.msg import TransformStamped
import numpy as np
from object_scanner.scanner_node import ObjectScannerNode, RecordingState
import rclpy
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_srvs.srv import Trigger


def make_cloud(points, stamp):
    message = PointCloud2()
    message.header.stamp = stamp
    message.header.frame_id = "camera0_depth_optical_frame"
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


def call(node, callback):
    return callback(Trigger.Request(), Trigger.Response())


def test_recording_services_append_to_one_sqlite_session(tmp_path):
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"output_directory:={tmp_path}",
        ]
    )
    node = ObjectScannerNode()
    try:
        assert not call(node, node._on_pause_recording).success

        response = call(node, node._on_start_recording)
        assert response.success
        assert node._state is RecordingState.RECORDING
        database_path = node._database_path
        assert database_path is not None
        assert response.message == str(database_path)
        assert not call(node, node._on_start_recording).success

        stamp = PointCloud2().header.stamp
        stamp.sec = 1
        stamp.nanosec = 2
        cloud = make_cloud(
            [
                (0.5, 0.0, 1.0, (0, 255, 0)),
                (1.0, 0.0, 1.0, (255, 0, 0)),
            ],
            stamp,
        )
        image = Image()
        image.header.stamp = stamp
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = "world"
        transform.child_frame_id = cloud.header.frame_id
        transform.transform.translation.x = 1.0
        node._on_synchronized_frame(cloud, image, transform)

        response = call(node, node._on_pause_recording)
        assert response.success
        assert node._state is RecordingState.PAUSED
        with sqlite3.connect(database_path) as connection:
            row = connection.execute(
                """
                SELECT source_sec, source_nanosec, frame_id,
                       point_count, xyz, rgb
                FROM frames
                """
            ).fetchone()
        assert row[:4] == (1, 2, "world", 1)
        np.testing.assert_allclose(
            np.frombuffer(row[4], dtype="<f4").reshape(-1, 3),
            [[1.5, 0.0, 1.0]],
        )
        np.testing.assert_array_equal(
            np.frombuffer(row[5], dtype=np.uint8).reshape(-1, 3),
            [[0, 255, 0]],
        )

        assert call(node, node._on_resume_recording).success
        assert node._state is RecordingState.RECORDING
        node._on_synchronized_frame(cloud, image, transform)

        response = call(node, node._on_stop_recording)
        assert response.success
        assert response.message == str(database_path)
        assert node._state is RecordingState.STOPPED
        with sqlite3.connect(database_path) as connection:
            frame_count = connection.execute(
                "SELECT COUNT(*) FROM frames"
            ).fetchone()[0]
        assert frame_count == 2
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


def test_reference_color_changes_only_while_stopped(tmp_path):
    rclpy.init(
        args=[
            "--ros-args",
            "-p",
            f"output_directory:={tmp_path}",
        ]
    )
    node = ObjectScannerNode()
    try:
        assert node._validated_target_rgb(
            array("q", [255, 0, 0])
        ) == (255, 0, 0)
        result = node.set_parameters(
            [Parameter("target_rgb", value=[255, 0, 0])]
        )[0]
        assert result.successful
        assert node._target_rgb == (255, 0, 0)

        assert call(node, node._on_start_recording).success
        result = node.set_parameters(
            [Parameter("target_rgb", value=[0, 0, 255])]
        )[0]
        assert not result.successful
        assert node._target_rgb == (255, 0, 0)

        stamp = PointCloud2().header.stamp
        cloud = make_cloud(
            [
                (0.0, 0.0, 1.0, (255, 0, 0)),
                (1.0, 0.0, 1.0, (0, 255, 0)),
            ],
            stamp,
        )
        image = Image()
        transform = TransformStamped()
        transform.header.frame_id = "world"
        transform.child_frame_id = cloud.header.frame_id
        node._on_synchronized_frame(cloud, image, transform)

        assert call(node, node._on_pause_recording).success
        result = node.set_parameters(
            [Parameter("target_rgb", value=[0, 0, 255])]
        )[0]
        assert not result.successful
        with sqlite3.connect(node._database_path) as connection:
            rgb_blob = connection.execute(
                "SELECT rgb FROM frames"
            ).fetchone()[0]
        np.testing.assert_array_equal(
            np.frombuffer(rgb_blob, dtype=np.uint8).reshape(-1, 3),
            [[255, 0, 0]],
        )
        assert call(node, node._on_stop_recording).success
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()
