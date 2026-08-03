import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_web.sqlite_reader import (
    build_frame_payload,
    build_point_payload,
    list_frames,
    PAYLOAD_HEADER,
    PAYLOAD_MAGIC,
    read_sampled_frame,
    read_sampled_points,
)


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
    finally:
        recording.close()
