import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_web.sqlite_reader import (
    build_point_payload,
    PAYLOAD_HEADER,
    PAYLOAD_MAGIC,
    read_sampled_points,
)


def test_reader_samples_committed_open_database_and_encodes_payload(tmp_path):
    database_path = tmp_path / "scan.sqlite3"
    recording = SqliteRecording(database_path)
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
                source_sec=frame_index,
                source_nanosec=0,
                frame_id="world",
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
    finally:
        recording.close()
