import sqlite3

import numpy as np
from object_scanner_processing.recording import read_frames
import pytest


def _write_frame_database(database_path):
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TABLE frames (
                id INTEGER PRIMARY KEY,
                recorded_perf_counter_ns INTEGER NOT NULL,
                source_sec INTEGER NOT NULL,
                source_nanosec INTEGER NOT NULL,
                point_count INTEGER NOT NULL,
                frame_id TEXT NOT NULL,
                transformation_name TEXT NOT NULL,
                transformation_matrix BLOB NOT NULL,
                xyz BLOB NOT NULL,
                rgb BLOB NOT NULL
            )
            """
        )
        for frame_id, source_sec in ((1, 2), (2, 1)):
            xyz = np.array(
                [[frame_id, 0.0, 0.5], [frame_id, 0.1, 0.5]],
                dtype="<f4",
            )
            rgb = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
            connection.execute(
                """
                INSERT INTO frames (
                    id, recorded_perf_counter_ns, source_sec, source_nanosec,
                    point_count, frame_id,
                    transformation_name, transformation_matrix, xyz, rgb
                ) VALUES (?, ?, ?, 0, ?, 'world', 'charuco', ?, ?, ?)
                """,
                (
                    frame_id,
                    frame_id * 100,
                    source_sec,
                    len(xyz),
                    np.eye(4, dtype="<f8").tobytes(),
                    xyz.tobytes(),
                    rgb.tobytes(),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _write_charuco_observation(database_path):
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE charuco_observations (
                frame_id INTEGER PRIMARY KEY,
                corner_count INTEGER NOT NULL,
                valid_depth_corner_count INTEGER NOT NULL,
                initial_reprojection_rmse_px REAL NOT NULL,
                camera_matrix BLOB NOT NULL,
                distortion_count INTEGER NOT NULL,
                distortion BLOB NOT NULL,
                color_from_child BLOB NOT NULL
            );
            CREATE TABLE charuco_corners (
                frame_id INTEGER NOT NULL,
                corner_id INTEGER NOT NULL,
                image_u REAL NOT NULL,
                image_v REAL NOT NULL,
                depth_valid INTEGER NOT NULL,
                child_x REAL,
                child_y REAL,
                child_z REAL,
                valid_pixel_count INTEGER NOT NULL,
                inlier_pixel_count INTEGER NOT NULL,
                depth_mad_m REAL,
                invalid_reason TEXT NOT NULL,
                initial_reprojection_error_px REAL NOT NULL,
                PRIMARY KEY (frame_id, corner_id)
            );
            """
        )
        camera_matrix = np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            dtype="<f8",
        )
        connection.execute(
            """
            INSERT INTO charuco_observations VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                20,
                20,
                0.2,
                camera_matrix.tobytes(),
                5,
                np.zeros(5, dtype="<f8").tobytes(),
                np.eye(4, dtype="<f8").tobytes(),
            ),
        )
        connection.executemany(
            """
            INSERT INTO charuco_corners VALUES (
                1, ?, ?, ?, 1, ?, 0.0, 0.4, 25, 25, 0.0, '', 0.2
            )
            """,
            [
                (corner_id, corner_id, corner_id, corner_id * 0.001)
                for corner_id in range(20)
            ],
        )
        connection.commit()
    finally:
        connection.close()


def test_reads_complete_frames_without_modifying_database(tmp_path):
    database_path = tmp_path / "recording.sqlite3"
    _write_frame_database(database_path)
    before = database_path.read_bytes()

    frames = read_frames(database_path)

    assert [frame.id for frame in frames] == [1, 2]
    assert frames[0].parent_frame_id == "world"
    assert frames[0].transformation_name == "charuco"
    assert frames[0].recorded_perf_counter_ns == 100
    np.testing.assert_array_equal(frames[0].matrix, np.eye(4))
    np.testing.assert_array_equal(frames[0].xyz[:, 0], [1.0, 1.0])
    np.testing.assert_array_equal(frames[0].rgb, [[10, 20, 30], [40, 50, 60]])
    assert database_path.read_bytes() == before


def test_reads_validated_charuco_observation(tmp_path):
    database_path = tmp_path / "recording.sqlite3"
    _write_frame_database(database_path)
    _write_charuco_observation(database_path)

    frames = read_frames(database_path)

    assert frames[0].charuco is not None
    assert frames[0].charuco.valid_depth_corner_count == 20
    np.testing.assert_array_equal(
        frames[0].charuco.corner_ids,
        np.arange(20),
    )
    assert frames[1].charuco is None


def test_rejects_invalid_frame_blob_lengths(tmp_path):
    database_path = tmp_path / "recording.sqlite3"
    _write_frame_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("UPDATE frames SET xyz = X'00' WHERE id = 1")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(ValueError, match="invalid point data"):
        read_frames(database_path)
