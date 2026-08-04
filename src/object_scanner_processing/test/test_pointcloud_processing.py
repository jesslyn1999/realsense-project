from dataclasses import replace

import numpy as np
from object_scanner_processing.charuco_observations import (
    board_points_opencv,
    board_points_world,
    CharucoCalibrationError,
    CharucoFrameObservation,
    WORLD_FROM_OPENCV_BOARD,
)
import object_scanner_processing.pointcloud_processing as processing
from object_scanner_processing.pointcloud_processing import (
    _fuse_voxels,
    _sequential_charuco_poses,
    CharucoPoseEvidence,
    charuco_frame_weight,
    PointCloudProcessingError,
    process_frames,
    validate_sequential_charuco_capture,
)
from object_scanner_processing.recording import RecordedFrame
import pytest


def _transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _pose(translation, rotation_axis=None, rotation_degrees=0.0):
    matrix = np.eye(4)
    if rotation_axis is not None:
        axis = np.asarray(rotation_axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        angle = np.deg2rad(rotation_degrees)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        matrix[:3, :3] = (
            np.eye(3)
            + np.sin(angle) * skew
            + (1.0 - np.cos(angle)) * (skew @ skew)
        )
    matrix[:3, 3] = translation
    return matrix


def _surface():
    axis = np.linspace(-0.06, 0.06, 61)
    x, y = np.meshgrid(axis, axis)
    z = (
        0.35
        + 0.012 * np.sin(40.0 * x)
        + 0.008 * np.cos(35.0 * y)
        + 0.1 * x * y
    )
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    rgb = np.column_stack(
        (
            np.full(len(xyz), 30, dtype=np.uint8),
            np.clip(180 + x.ravel() * 500, 0, 255).astype(np.uint8),
            np.clip(80 + y.ravel() * 500, 0, 255).astype(np.uint8),
        )
    )
    return xyz, rgb


def _recorded_frame(
    frame_id,
    world_xyz,
    rgb,
    true_pose,
    recorded_pose,
    charuco=None,
):
    camera_xyz = _transform(world_xyz, np.linalg.inv(true_pose))
    recorded_world = _transform(camera_xyz, recorded_pose)
    return RecordedFrame(
        id=frame_id,
        recorded_perf_counter_ns=frame_id,
        source_sec=frame_id,
        source_nanosec=0,
        parent_frame_id="world",
        transformation_name="charuco" if charuco is not None else "identity",
        matrix=recorded_pose,
        xyz=recorded_world.astype(np.float32),
        rgb=rgb,
        charuco=charuco,
    )


def _pose_error(estimated, expected):
    relative_rotation = estimated[:3, :3] @ expected[:3, :3].T
    translation = np.linalg.norm(estimated[:3, 3] - expected[:3, 3])
    angle = np.arccos(
        np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    )
    return translation, np.rad2deg(angle)


def _charuco_camera_pose(rotation_vector, translation):
    camera_from_board = np.eye(4)
    camera_from_board[:3, :3] = processing.cv2.Rodrigues(
        np.asarray(rotation_vector, dtype=np.float64)
    )[0]
    camera_from_board[:3, 3] = translation
    return WORLD_FROM_OPENCV_BOARD @ np.linalg.inv(camera_from_board)


def _charuco_observation(true_pose, corner_count=30):
    corner_ids = np.arange(corner_count, dtype=np.int32)
    board_opencv = board_points_opencv(corner_ids)
    world_points = board_points_world(corner_ids)
    child_points = _transform(world_points, np.linalg.inv(true_pose))
    intrinsics = np.array(
        [[700.0, 0.0, 320.0], [0.0, 700.0, 240.0], [0.0, 0.0, 1.0]]
    )
    camera_from_world = np.linalg.inv(true_pose)
    camera_from_board = camera_from_world @ WORLD_FROM_OPENCV_BOARD
    projected, _ = processing.cv2.projectPoints(
        board_opencv,
        processing.cv2.Rodrigues(camera_from_board[:3, :3])[0],
        camera_from_board[:3, 3],
        intrinsics,
        np.zeros(5),
    )
    return CharucoFrameObservation(
        corner_ids=corner_ids,
        image_points=projected.reshape(-1, 2),
        depth_valid=np.ones(corner_count, dtype=bool),
        child_points=child_points,
        depth_valid_pixel_counts=np.full(corner_count, 25, dtype=np.uint16),
        depth_inlier_pixel_counts=np.full(corner_count, 25, dtype=np.uint16),
        depth_mad_m=np.zeros(corner_count),
        depth_invalid_reasons=tuple("" for _ in range(corner_count)),
        camera_matrix=intrinsics,
        distortion=np.zeros(5),
        color_from_child=np.eye(4),
        initial_reprojection_errors_px=np.zeros(corner_count),
        initial_reprojection_rmse_px=0.0,
    )


def test_corner_frame_weights_decay_by_one_half():
    assert [charuco_frame_weight(index) for index in range(3)] == [
        1.0,
        0.5,
        0.25,
    ]


def test_sequential_corner_solve_anchors_first_frame_and_improves_later_poses():
    xyz, rgb = _surface()
    true_poses = [
        _charuco_camera_pose([0.0, 0.0, 0.0], [-0.09, -0.06, 0.40]),
        _charuco_camera_pose([0.02, -0.03, 0.01], [-0.08, -0.055, 0.41]),
        _charuco_camera_pose([-0.015, 0.025, -0.01], [-0.10, -0.05, 0.39]),
    ]
    recorded_poses = [
        true_poses[0],
        _pose([0.002, -0.001, 0.001], [0, 1, 0], 0.3)
        @ true_poses[1],
        _pose([-0.002, 0.001, -0.001], [1, 0, 0], -0.25)
        @ true_poses[2],
    ]
    frames = [
        _recorded_frame(
            index + 1,
            xyz,
            rgb,
            true_pose,
            recorded_poses[index],
            _charuco_observation(true_pose),
        )
        for index, true_pose in enumerate(true_poses)
    ]

    poses = _sequential_charuco_poses(frames)

    np.testing.assert_allclose(poses[0], recorded_poses[0])
    for index in (1, 2):
        before = _pose_error(recorded_poses[index], true_poses[index])
        after = _pose_error(poses[index], true_poses[index])
        assert after[0] < before[0]
        assert after[1] < before[1]
        assert after[0] < 1e-6
        assert after[1] < 1e-4


def test_capture_sequential_check_accepts_small_pose_correction():
    first_pose = _charuco_camera_pose(
        [0.0, 0.0, 0.0],
        [-0.09, -0.06, 0.40],
    )
    second_pose = _charuco_camera_pose(
        [0.02, -0.03, 0.01],
        [-0.08, -0.055, 0.41],
    )
    recorded_second = (
        _pose([0.002, -0.001, 0.001], [0, 1, 0], 0.3) @ second_pose
    )

    metrics = validate_sequential_charuco_capture(
        (
            CharucoPoseEvidence(
                first_pose,
                _charuco_observation(first_pose),
            ),
        ),
        CharucoPoseEvidence(
            recorded_second,
            _charuco_observation(second_pose),
        ),
    )

    assert metrics.correction_m < processing.MAX_CORRECTION_M
    assert metrics.correction_deg < processing.MAX_CORRECTION_DEG


def test_capture_sequential_check_rejects_large_pose_correction():
    first_pose = _charuco_camera_pose(
        [0.0, 0.0, 0.0],
        [-0.09, -0.06, 0.40],
    )
    second_pose = _charuco_camera_pose(
        [0.02, -0.03, 0.01],
        [-0.08, -0.055, 0.41],
    )
    recorded_second = _pose([0.020, 0.0, 0.0]) @ second_pose

    with pytest.raises(
        CharucoCalibrationError,
        match=r"maximums are 10.00 mm and 2.000 deg",
    ):
        validate_sequential_charuco_capture(
            (
                CharucoPoseEvidence(
                    first_pose,
                    _charuco_observation(first_pose),
                ),
            ),
            CharucoPoseEvidence(
                recorded_second,
                _charuco_observation(second_pose),
            ),
        )


def test_charuco_processing_refuses_missing_observations():
    xyz, rgb = _surface()
    frames = [
        _recorded_frame(1, xyz, rgb, np.eye(4), np.eye(4)),
        _recorded_frame(2, xyz, rgb, np.eye(4), np.eye(4)),
    ]
    for frame in frames:
        object.__setattr__(frame, "transformation_name", "charuco")

    with pytest.raises(PointCloudProcessingError, match="no persisted"):
        _sequential_charuco_poses(frames)


def test_charuco_pipeline_meets_corner_and_cloud_acceptance_contract():
    xyz, rgb = _surface()
    true_poses = [
        _charuco_camera_pose([0.0, 0.0, 0.0], [-0.09, -0.06, 0.40]),
        _charuco_camera_pose([0.02, -0.03, 0.01], [-0.08, -0.055, 0.41]),
    ]
    recorded_poses = [
        true_poses[0],
        _pose([0.002, -0.001, 0.001], [0, 1, 0], 0.3)
        @ true_poses[1],
    ]
    frames = [
        _recorded_frame(
            index + 1,
            xyz,
            rgb,
            true_pose,
            recorded_poses[index],
            _charuco_observation(true_pose),
        )
        for index, true_pose in enumerate(true_poses)
    ]

    result = process_frames(list(reversed(frames)))

    assert result.cloud_overlap_fraction_3mm == pytest.approx(1.0)
    assert [frame.source.id for frame in result.aligned_frames] == [1, 2]
    np.testing.assert_allclose(result.optimized_poses[0], true_poses[0])
    assert [item.temporal_weight for item in result.charuco_frames] == [
        1.0,
        0.5,
    ]
    assert [item.matched_prior_count for item in result.charuco_frames] == [
        0,
        30,
    ]
    assert max(
        item.reprojection_max_px for item in result.charuco_frames
    ) < 1e-3
    assert len(result.charuco_corners) == 60


def test_charuco_pipeline_rejects_one_corner_over_one_pixel():
    xyz, rgb = _surface()
    true_poses = [
        _charuco_camera_pose([0.0, 0.0, 0.0], [-0.09, -0.06, 0.40]),
        _charuco_camera_pose([0.02, -0.03, 0.01], [-0.08, -0.055, 0.41]),
    ]
    first_observation = _charuco_observation(true_poses[0])
    shifted_pixels = first_observation.image_points.copy()
    shifted_pixels[0, 0] += 1.1
    first_observation = replace(
        first_observation,
        image_points=shifted_pixels,
    )
    frames = [
        _recorded_frame(
            1,
            xyz,
            rgb,
            true_poses[0],
            true_poses[0],
            first_observation,
        ),
        _recorded_frame(
            2,
            xyz,
            rgb,
            true_poses[1],
            true_poses[1],
            _charuco_observation(true_poses[1]),
        ),
    ]

    with pytest.raises(PointCloudProcessingError, match="every corner"):
        process_frames(frames)


def test_charuco_pipeline_rejects_cloud_below_99_percent_at_three_mm(
    monkeypatch,
):
    xyz, rgb = _surface()
    deformed = xyz.copy()
    deformed[deformed[:, 0] < -0.045, 2] += 0.005
    true_poses = [
        _charuco_camera_pose([0.0, 0.0, 0.0], [-0.09, -0.06, 0.40]),
        _charuco_camera_pose([0.02, -0.03, 0.01], [-0.08, -0.055, 0.41]),
    ]
    frames = [
        _recorded_frame(
            1,
            xyz,
            rgb,
            true_poses[0],
            true_poses[0],
            _charuco_observation(true_poses[0]),
        ),
        _recorded_frame(
            2,
            deformed,
            rgb,
            true_poses[1],
            true_poses[1],
            _charuco_observation(true_poses[1]),
        ),
    ]
    monkeypatch.setattr(
        processing,
        "_robust_point_to_plane_icp",
        lambda _source, _target, _distance, initial, _prior: initial,
    )

    with pytest.raises(PointCloudProcessingError, match="99% are required"):
        process_frames(frames)


def test_rejects_invalid_transformation_before_processing():
    xyz, rgb = _surface()
    invalid = np.eye(4)
    invalid[0, 0] = 2.0
    frames = [
        _recorded_frame(1, xyz, rgb, np.eye(4), invalid),
        _recorded_frame(2, xyz, rgb, np.eye(4), np.eye(4)),
    ]

    with pytest.raises(PointCloudProcessingError, match="orthonormal"):
        process_frames(frames)


def test_pose_error_does_not_turn_rotation_into_translation():
    expected = _pose([0.4, -0.2, 0.5])
    estimated = _pose([0.4, -0.2, 0.5], [0, 0, 1], 1.5)

    translation, rotation = _pose_error(estimated, expected)

    assert translation == pytest.approx(0.0)
    assert rotation == pytest.approx(1.5)


def test_fusion_gives_each_frame_one_vote_per_voxel():
    xyz_parts = [
        np.array(
            [[0.0001, 0, 0], [0.0002, 0, 0], [0.0003, 0, 0]],
            dtype=np.float64,
        ),
        np.array([[0.0024, 0, 0]], dtype=np.float64),
    ]
    rgb_parts = [
        np.tile([0, 0, 0], (3, 1)).astype(np.uint8),
        np.array([[200, 100, 50]], dtype=np.uint8),
    ]

    xyz, rgb, observations = _fuse_voxels(xyz_parts, rgb_parts)

    np.testing.assert_allclose(xyz, [[0.0013, 0, 0]], atol=1e-7)
    np.testing.assert_array_equal(rgb, [[100, 50, 25]])
    np.testing.assert_array_equal(observations, [2])


def test_accepts_already_aligned_overlapping_frames():
    xyz, rgb = _surface()
    frames = [
        _recorded_frame(1, xyz, rgb, np.eye(4), np.eye(4)),
        _recorded_frame(2, xyz, rgb, np.eye(4), np.eye(4)),
    ]

    result = process_frames(frames)

    assert result.accepted_edges == 1
    assert result.edges[0].rmse_before_m == pytest.approx(
        result.edges[0].rmse_after_m,
        abs=1e-6,
    )
    assert [frame.source.id for frame in result.aligned_frames] == [1, 2]
    for aligned, pose, diagnostics in zip(
        result.aligned_frames,
        result.optimized_poses,
        result.frames,
        strict=True,
    ):
        np.testing.assert_allclose(aligned.optimized_pose, pose)
        assert len(aligned.xyz) == (
            diagnostics.cleaned_points - diagnostics.temporal_removed
        )
        assert aligned.xyz.dtype == np.dtype("<f4")
        assert aligned.rgb.dtype == np.dtype(np.uint8)


def test_registers_pose_graph_and_temporally_removes_weak_ghost():
    xyz, rgb = _surface()
    true_poses = [
        np.eye(4),
        _pose([0.020, -0.004, 0.006], [0, 1, 0], 5.0),
        _pose([-0.018, 0.006, 0.004], [1, 0, 0], -4.0),
    ]
    recorded_poses = [
        true_poses[0],
        _pose([0.002, 0.0, 0.0], [0, 0, 1], 0.3) @ true_poses[1],
        _pose([-0.0025, 0.001, 0.0], [1, 0, 0], -0.4) @ true_poses[2],
    ]

    patch_axis = np.linspace(-0.006, 0.006, 7)
    patch_x, patch_y = np.meshgrid(patch_axis - 0.04, patch_axis + 0.04)
    patch_z = np.full(patch_x.size, 0.348)
    ghost = np.column_stack((patch_x.ravel(), patch_y.ravel(), patch_z))
    ghost_rgb = np.tile([255, 0, 0], (len(ghost), 1)).astype(np.uint8)

    frames = []
    for index in range(3):
        frame_xyz = xyz
        frame_rgb = rgb
        if index == 1:
            frame_xyz = np.concatenate((frame_xyz, ghost))
            frame_rgb = np.concatenate((frame_rgb, ghost_rgb))
        frames.append(
            _recorded_frame(
                index + 1,
                frame_xyz,
                frame_rgb,
                true_poses[index],
                recorded_poses[index],
            )
        )

    result = process_frames(frames)

    assert result.accepted_edges >= 3
    assert any(
        edge.accepted
        and abs(edge.source_frame_id - edge.target_frame_id) > 1
        for edge in result.edges
    )
    for index in (1, 2):
        before = _pose_error(recorded_poses[index], true_poses[index])
        after = _pose_error(result.optimized_poses[index], true_poses[index])
        assert after[0] < before[0]
        before_score = before[0] / 0.010 + before[1] / 2.0
        after_score = after[0] / 0.010 + after[1] / 2.0
        assert after_score < before_score
    assert result.frames[1].temporal_removed > 0
    assert len(result.xyz) > 1000
    assert np.max(result.observation_counts) >= 2


def test_rejects_one_failed_icp_edge_when_other_edges_connect(
    monkeypatch,
):
    xyz, rgb = _surface()
    poses = [
        np.eye(4),
        _pose([0.015, 0.0, 0.005], [0, 1, 0], 3.0),
        _pose([-0.012, 0.004, 0.003], [1, 0, 0], -3.0),
    ]
    frames = [
        _recorded_frame(index + 1, xyz, rgb, pose, pose)
        for index, pose in enumerate(poses)
    ]
    original = processing._robust_point_to_plane_icp
    failed = False

    def fail_first_edge(*args, **kwargs):
        nonlocal failed
        if not failed:
            failed = True
            raise PointCloudProcessingError("synthetic solver failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        processing,
        "_robust_point_to_plane_icp",
        fail_first_edge,
    )

    result = process_frames(frames)

    assert result.accepted_edges == 2
    assert any(
        not edge.accepted and "synthetic solver failure" in edge.reason
        for edge in result.edges
    )


def test_rejects_disconnected_low_overlap_frames():
    xyz, rgb = _surface()
    frames = [
        _recorded_frame(1, xyz, rgb, np.eye(4), np.eye(4)),
        _recorded_frame(
            2,
            xyz + [0.5, 0.0, 0.0],
            rgb,
            np.eye(4),
            np.eye(4),
        ),
    ]

    with pytest.raises(
        PointCloudProcessingError,
        match="do not connect",
    ) as caught:
        process_frames(frames)

    message = str(caught.value)
    assert "1/2 frames connect" in message
    assert "2/2 are required" in message
    assert "Accepted 0/1 candidate edges" in message
    assert "at least 1 accepted edge is necessary" in message
    assert "60% within 12 mm" in message
    assert "15% within 5 mm" in message
    assert "Rejected 1/1 candidate edges" in message
    assert "1->2: initial 12 mm overlap" in message
