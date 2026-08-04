from contextlib import closing
import sqlite3

import cv2
import numpy as np
from object_scanner.sqlite_recording import SqliteRecording
from object_scanner_processing.aligned_recording import (
    generate_aligned_recording,
    read_fused_cloud,
    source_revision,
)
from object_scanner_processing.charuco_observations import (
    board_points_opencv,
    board_points_world,
    CharucoFrameObservation,
    WORLD_FROM_OPENCV_BOARD,
)
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
        ).fetchone()[0] == "3"
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


def _synthetic_camera_pose(rotation_vector, translation):
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64)
    )[0]
    camera_from_board[:3, 3] = translation
    return WORLD_FROM_OPENCV_BOARD @ np.linalg.inv(camera_from_board)


def _synthetic_observation(pose):
    corner_ids = np.arange(30, dtype=np.int32)
    board_opencv = board_points_opencv(corner_ids)
    world_points = board_points_world(corner_ids)
    child_from_world = np.linalg.inv(pose)
    child_points = (
        world_points @ child_from_world[:3, :3].T
        + child_from_world[:3, 3]
    )
    intrinsics = np.array(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
    )
    camera_from_board = child_from_world @ WORLD_FROM_OPENCV_BOARD
    projected, _ = cv2.projectPoints(
        board_opencv,
        cv2.Rodrigues(camera_from_board[:3, :3])[0],
        camera_from_board[:3, 3],
        intrinsics,
        np.zeros(5),
    )
    return CharucoFrameObservation(
        corner_ids=corner_ids,
        image_points=projected.reshape(-1, 2),
        depth_valid=np.ones(30, dtype=bool),
        child_points=child_points,
        depth_valid_pixel_counts=np.full(30, 25, dtype=np.uint16),
        depth_inlier_pixel_counts=np.full(30, 25, dtype=np.uint16),
        depth_mad_m=np.zeros(30),
        depth_invalid_reasons=tuple("" for _ in range(30)),
        camera_matrix=intrinsics,
        distortion=np.zeros(5),
        color_from_child=np.eye(4),
        initial_reprojection_errors_px=np.zeros(30),
        initial_reprojection_rmse_px=0.0,
    )


def test_saved_observations_generate_strict_aligned_metrics(tmp_path):
    recording = SqliteRecording(tmp_path / "scan")
    axis = np.linspace(-0.06, 0.06, 61)
    x, y = np.meshgrid(axis, axis)
    world_xyz = np.column_stack(
        (
            x.ravel(),
            y.ravel(),
            0.35
            + 0.012 * np.sin(40.0 * x.ravel())
            + 0.008 * np.cos(35.0 * y.ravel()),
        )
    )
    rgb = np.tile([0, 255, 0], (len(world_xyz), 1)).astype(np.uint8)
    true_poses = [
        _synthetic_camera_pose([0.0, 0.0, 0.0], [-0.09, -0.06, 0.40]),
        _synthetic_camera_pose(
            [0.02, -0.03, 0.01],
            [-0.08, -0.055, 0.41],
        ),
    ]
    recorded_poses = [true_poses[0], true_poses[1].copy()]
    recorded_poses[1][:3, 3] += [0.002, -0.001, 0.001]
    try:
        for index, (true_pose, recorded_pose) in enumerate(
            zip(true_poses, recorded_poses, strict=True),
            start=1,
        ):
            child_from_world = np.linalg.inv(true_pose)
            child_xyz = (
                world_xyz @ child_from_world[:3, :3].T
                + child_from_world[:3, 3]
            )
            recorded_xyz = (
                child_xyz @ recorded_pose[:3, :3].T
                + recorded_pose[:3, 3]
            )
            recording.append_frame(
                recorded_perf_counter_ns=index,
                source_sec=index,
                source_nanosec=0,
                frame_id="world",
                transformation_name="charuco",
                transformation_matrix=recorded_pose,
                xyz=recorded_xyz,
                rgb=rgb,
                charuco_observation=_synthetic_observation(true_pose),
            )

        aligned_path = generate_aligned_recording(recording.path)
        fused = read_fused_cloud(
            aligned_path,
            source_revision(recording.path),
        )

        assert fused.charuco_frame_count == 2
        assert fused.charuco_reprojection_max_px < 0.001
        assert fused.cloud_overlap_fraction_3mm == pytest.approx(1.0)
    finally:
        recording.close()
