import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_web.sqlite_reader import (
    build_array_payload,
    build_frame_payload,
    build_point_payload,
    build_replay_frame_payload,
    list_frames,
    PAYLOAD_HEADER,
    PAYLOAD_MAGIC,
    read_frame_matrix,
    read_sampled_frame,
    read_sampled_points,
)
import pytest


def test_reader_samples_committed_open_database_and_encodes_payload(tmp_path):
    recording = SqliteRecording(tmp_path / "scan")
    database_path = recording.path
    try:
        for frame_index in range(2):
            first = frame_index * 6
            xyz = np.column_stack(
                (
                    np.arange(first, first + 6, dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                    np.ones(6, dtype=np.float32),
                )
            )
            rgb = np.column_stack(
                (
                    np.zeros(6, dtype=np.uint8),
                    np.full(6, 255, dtype=np.uint8),
                    np.arange(first, first + 6, dtype=np.uint8),
                )
            )
            recording.append_frame(
                recorded_perf_counter_ns=frame_index,
                source_sec=2 - frame_index,
                source_nanosec=0,
                frame_id="world",
                transformation_name="identity",
                transformation_matrix=np.eye(4),
                xyz=xyz,
                rgb=rgb,
            )

        xyz, rgb, total = read_sampled_points(database_path, max_points=5)
        assert total == 12
        np.testing.assert_array_equal(xyz[:, 0], [0, 3, 6, 9])
        np.testing.assert_array_equal(rgb[:, 2], [0, 3, 6, 9])

        payload = build_point_payload(database_path, max_points=5)
        magic, displayed, total = PAYLOAD_HEADER.unpack_from(payload)
        assert magic == PAYLOAD_MAGIC
        assert displayed == 4
        assert total == 12
        xyz_bytes = displayed * 3 * np.dtype("<f4").itemsize
        positions = np.frombuffer(
            payload,
            dtype="<f4",
            count=displayed * 3,
            offset=PAYLOAD_HEADER.size,
        ).reshape(-1, 3)
        colors = np.frombuffer(
            payload,
            dtype=np.uint8,
            count=displayed * 3,
            offset=PAYLOAD_HEADER.size + xyz_bytes,
        ).reshape(-1, 3)
        np.testing.assert_array_equal(positions, xyz)
        np.testing.assert_array_equal(colors, rgb)

        frames = list_frames(database_path)
        assert [frame["id"] for frame in frames] == [2, 1]
        assert frames[0]["transformation_name"] == "identity"
        assert frames[0]["parent_frame_id"] == "world"
        np.testing.assert_array_equal(frames[0]["matrix"], np.eye(4))
        frame_xyz, frame_rgb, frame_total = read_sampled_frame(
            database_path,
            frame_id=2,
            max_points=5,
        )
        assert frame_total == 6
        np.testing.assert_array_equal(frame_xyz[:, 0], [6, 8, 10])
        np.testing.assert_array_equal(frame_rgb[:, 2], [6, 8, 10])
        frame_payload = build_frame_payload(
            database_path,
            frame_id=2,
            max_points=5,
        )
        assert PAYLOAD_HEADER.unpack_from(frame_payload) == (
            PAYLOAD_MAGIC,
            3,
            6,
        )
        array_payload = build_array_payload(xyz, rgb, total_points=12)
        assert PAYLOAD_HEADER.unpack_from(array_payload) == (
            PAYLOAD_MAGIC,
            4,
            12,
        )
    finally:
        recording.close()


def test_builds_raw_filtered_and_aligned_replay_stages(tmp_path):
    raw = SqliteRecording(tmp_path / "raw")
    aligned = SqliteRecording(tmp_path / "aligned")
    initial_pose = np.eye(4)
    initial_pose[0, 3] = 1.0
    optimized_pose = np.eye(4)
    optimized_pose[0, 3] = 3.0
    rgb = np.tile([0, 255, 0], (3, 1)).astype(np.uint8)
    try:
        raw.append_frame(
            recorded_perf_counter_ns=1,
            source_sec=1,
            source_nanosec=0,
            frame_id="world",
            transformation_name="identity",
            transformation_matrix=initial_pose,
            xyz=np.array(
                [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [101.0, 0.0, 1.0]],
                dtype="<f4",
            ),
            rgb=rgb,
        )
        aligned.append_frame(
            recorded_perf_counter_ns=1,
            source_sec=1,
            source_nanosec=0,
            frame_id="world",
            transformation_name="charuco_pose_graph_optimized",
            transformation_matrix=optimized_pose,
            xyz=np.array(
                [[3.0, 0.0, 1.0], [4.0, 0.0, 1.0]],
                dtype="<f4",
            ),
            rgb=rgb[:2],
        )

        def positions(stage):
            payload = build_replay_frame_payload(
                raw.path,
                aligned.path,
                frame_id=1,
                max_points=10,
                stage=stage,
            )
            _, displayed, total = PAYLOAD_HEADER.unpack_from(payload)
            xyz = np.frombuffer(
                payload,
                dtype="<f4",
                count=displayed * 3,
                offset=PAYLOAD_HEADER.size,
            ).reshape(-1, 3)
            return xyz, total

        raw_xyz, raw_total = positions("raw")
        filtered_xyz, filtered_total = positions("filtered")
        aligned_xyz, aligned_total = positions("aligned")

        np.testing.assert_array_equal(raw_xyz[:, 0], [1.0, 2.0, 101.0])
        np.testing.assert_array_equal(filtered_xyz[:, 0], [1.0, 2.0])
        np.testing.assert_array_equal(aligned_xyz[:, 0], [3.0, 4.0])
        assert (raw_total, filtered_total, aligned_total) == (3, 2, 2)
        np.testing.assert_array_equal(read_frame_matrix(raw.path, 1), initial_pose)
        with pytest.raises(ValueError, match="stage must be one of"):
            positions("unknown")
    finally:
        raw.close()
        aligned.close()
