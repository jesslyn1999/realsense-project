from contextlib import closing
import sqlite3

import numpy as np
from object_scanner_processing.charuco_observations import (
    CharucoFrameObservation,
)
from object_scanner_web_orbbec.sqlite_recording import SqliteRecording
import pytest


def _observation():
    count = 20
    return CharucoFrameObservation(
        corner_ids=np.arange(count, dtype=np.int32),
        image_points=np.column_stack(
            (np.arange(count), np.arange(count))
        ).astype(np.float64),
        depth_valid=np.ones(count, dtype=bool),
        child_points=np.column_stack(
            (
                np.arange(count) * 0.001,
                np.zeros(count),
                np.full(count, 0.4),
            )
        ),
        depth_valid_pixel_counts=np.full(count, 25, dtype=np.uint16),
        depth_inlier_pixel_counts=np.full(count, 25, dtype=np.uint16),
        depth_mad_m=np.zeros(count),
        depth_invalid_reasons=tuple("" for _ in range(count)),
        camera_matrix=np.array(
            [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
        ),
        distortion=np.zeros(5),
        color_from_child=np.eye(4),
        initial_reprojection_errors_px=np.full(count, 0.2),
        initial_reprojection_rmse_px=0.2,
    )


def _append_charuco(recording, observation):
    return recording.append_frame(
        recorded_perf_counter_ns=100,
        source_sec=1,
        source_nanosec=2,
        frame_id="world",
        transformation_name="charuco",
        transformation_matrix=np.eye(4),
        xyz=np.array([[0.0, 0.0, 0.4]], dtype=np.float32),
        rgb=np.array([[0, 255, 0]], dtype=np.uint8),
        charuco_observation=observation,
    )


def test_persists_charuco_observation_with_its_frame(tmp_path):
    recording = SqliteRecording(tmp_path / "scan")

    frame_id = _append_charuco(recording, _observation())

    with closing(sqlite3.connect(recording.path)) as connection:
        assert connection.execute(
            "SELECT value FROM metadata WHERE key = 'format_version'"
        ).fetchone()[0] == "5"
        assert connection.execute(
            "SELECT corner_count, valid_depth_corner_count "
            "FROM charuco_observations WHERE frame_id = ?",
            (frame_id,),
        ).fetchone() == (20, 20)
        assert connection.execute(
            "SELECT COUNT(*) FROM charuco_corners WHERE frame_id = ?",
            (frame_id,),
        ).fetchone()[0] == 20
    recording.close()


def test_corner_insert_failure_rolls_back_the_frame(tmp_path):
    recording = SqliteRecording(tmp_path / "scan")
    recording._connection.execute(
        """
        CREATE TRIGGER reject_charuco_corner
        BEFORE INSERT ON charuco_corners
        BEGIN
            SELECT RAISE(ABORT, 'synthetic corner failure');
        END
        """
    )
    recording._connection.commit()

    with pytest.raises(sqlite3.IntegrityError, match="corner failure"):
        _append_charuco(recording, _observation())

    assert recording._connection.execute(
        "SELECT COUNT(*) FROM frames"
    ).fetchone()[0] == 0
    recording.close()
