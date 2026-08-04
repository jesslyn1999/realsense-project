import cv2
import numpy as np

from object_scanner_processing.charuco_observations import (
    annotate_charuco,
    board_points_world,
    build_charuco_observation,
    calibrate_charuco,
    calibrate_charuco_points,
    CharucoCalibration,
    CharucoCalibrationError,
    CharucoFrameObservation,
    depth_image_to_meters,
    detect_charuco,
    DEPTH_CONSISTENCY_MEDIAN_LIMIT_M,
    DEPTH_CONSISTENCY_P95_LIMIT_M,
    MAX_REPROJECTION_RMSE_PX,
    MIN_CORNER_COUNT,
    validate_charuco_depth_consistency,
)
import pytest


CAMERA_MATRIX = np.array(
    [
        [1000.0, 0.0, 500.0],
        [0.0, 1000.0, 350.0],
        [0.0, 0.0, 1.0],
    ]
)
DISTORTION = np.zeros(8)
BOARD = cv2.aruco.CharucoBoard(
    (10, 7),
    0.020,
    0.015,
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
)


def projected_board_points():
    object_points = BOARD.getChessboardCorners().astype(np.float64)
    rotation_vector = np.zeros(3)
    translation = np.array([-0.1, -0.07, 0.2])
    image_points, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        CAMERA_MATRIX,
        DISTORTION,
    )
    return object_points, image_points


def test_generated_board_recovers_camera_to_upright_world():
    image = BOARD.generateImage((1000, 700))

    calibration = calibrate_charuco(image, CAMERA_MATRIX, DISTORTION)

    expected = np.diag([1.0, -1.0, -1.0, 1.0])
    expected[:3, 3] = [0.1, -0.07, 0.2]
    assert calibration.corner_count == 54
    assert calibration.reprojection_rmse_px < 0.1
    np.testing.assert_array_equal(calibration.corner_ids, np.arange(54))
    np.testing.assert_allclose(
        calibration.camera_to_world,
        expected,
        atol=1e-4,
    )


def test_full_board_preview_draws_markers_and_corners_on_a_copy():
    grayscale = BOARD.generateImage((1000, 700))
    image = np.repeat(grayscale[:, :, None], 3, axis=2)
    original = image.copy()

    detection = detect_charuco(image)
    annotated = annotate_charuco(image, detection)

    assert detection.corner_count == 54
    assert len(detection.marker_corners) == 35
    np.testing.assert_array_equal(image, original)
    assert not np.shares_memory(annotated, image)
    assert np.any(annotated != image)


def test_partial_board_preview_draws_every_visible_detection():
    grayscale = BOARD.generateImage((1000, 700))
    image = np.repeat(grayscale[:, :650, None], 3, axis=2)

    detection = detect_charuco(image)
    annotated = annotate_charuco(image, detection)

    assert 0 < detection.corner_count < 54
    assert np.any(annotated != image)


def test_blank_preview_returns_an_unchanged_copy():
    image = np.full((120, 160, 3), 255, dtype=np.uint8)

    detection = detect_charuco(image)
    annotated = annotate_charuco(image, detection)

    assert detection.corner_count == 0
    assert detection.marker_ids is None
    assert not np.shares_memory(annotated, image)
    np.testing.assert_array_equal(annotated, image)


def test_blank_image_is_rejected_without_a_pose():
    with pytest.raises(CharucoCalibrationError) as caught:
        calibrate_charuco(
            np.full((700, 1000), 255, dtype=np.uint8),
            CAMERA_MATRIX,
            DISTORTION,
        )

    assert caught.value.corner_count == 0
    assert caught.value.reprojection_rmse_px is None


def test_too_few_matched_corners_are_rejected():
    object_points, image_points = projected_board_points()

    with pytest.raises(CharucoCalibrationError) as caught:
        calibrate_charuco_points(
            object_points[: MIN_CORNER_COUNT - 1],
            image_points[: MIN_CORNER_COUNT - 1],
            CAMERA_MATRIX,
            DISTORTION,
        )

    assert caught.value.corner_count == MIN_CORNER_COUNT - 1


def test_high_reprojection_error_is_rejected():
    object_points, image_points = projected_board_points()
    noisy_points = image_points.reshape(-1, 2).copy()
    noisy_points[::2, 0] += 5.0
    noisy_points[1::2, 1] -= 5.0

    with pytest.raises(CharucoCalibrationError) as caught:
        calibrate_charuco_points(
            object_points,
            noisy_points,
            CAMERA_MATRIX,
            DISTORTION,
        )

    assert caught.value.corner_count == 54
    assert (
        caught.value.reprojection_rmse_px
        > MAX_REPROJECTION_RMSE_PX
    )


@pytest.mark.parametrize(
    ("encoding", "dtype", "scale"),
    [
        ("16UC1", "<u2", 1000.0),
        ("32FC1", "<f4", 1.0),
    ],
)
def test_decodes_registered_depth_with_padded_rows(encoding, dtype, scale):
    source = np.array(
        [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        dtype=np.float64,
    )
    encoded = (source * scale).astype(dtype)
    row_bytes = encoded.shape[1] * encoded.dtype.itemsize
    step = row_bytes + 4
    data = bytearray(step * len(encoded))
    for row_index, row in enumerate(encoded):
        start = row_index * step
        data[start:start + row_bytes] = row.tobytes()

    depth = depth_image_to_meters(
        bytes(data),
        width=3,
        height=2,
        step=step,
        encoding=encoding,
        is_bigendian=False,
    )

    np.testing.assert_allclose(depth, source, atol=1e-6)


def _depth_calibration(corner_count=20):
    grid_x, grid_y = np.meshgrid(
        np.arange(10, 60, 10),
        np.arange(10, 50, 10),
    )
    image_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))[
        :corner_count
    ].astype(np.float64)
    return CharucoCalibration(
        camera_to_world=np.eye(4),
        corner_count=corner_count,
        reprojection_rmse_px=0.2,
        corner_ids=np.arange(corner_count, dtype=np.int32),
        image_points=image_points,
        board_points_opencv=np.zeros((corner_count, 3)),
        reprojection_errors_px=np.full(corner_count, 0.2),
    )


def test_samples_robust_registered_depth_in_child_frame():
    intrinsics = np.array(
        [[100.0, 0.0, 35.0], [0.0, 100.0, 30.0], [0.0, 0.0, 1.0]]
    )
    depth = np.full((60, 70), 0.5, dtype=np.float64)
    depth[8, 8] = 2.0
    color_from_child = np.eye(4)
    color_from_child[0, 3] = 0.02

    observation = build_charuco_observation(
        _depth_calibration(),
        depth,
        intrinsics,
        np.zeros(5),
        color_from_child,
    )

    assert observation.valid_depth_corner_count == 20
    assert observation.invalid_depth_corner_count == 0
    assert np.all(observation.depth_inlier_pixel_counts >= 24)
    first_u, first_v = observation.image_points[0]
    expected_color = np.array(
        [
            (first_u - intrinsics[0, 2]) / intrinsics[0, 0] * 0.5,
            (first_v - intrinsics[1, 2]) / intrinsics[1, 1] * 0.5,
            0.5,
        ]
    )
    np.testing.assert_allclose(
        observation.child_points[0],
        expected_color - [0.02, 0.0, 0.0],
    )


def test_reports_too_few_valid_depth_corners():
    calibration = _depth_calibration()
    depth = np.full((60, 70), 0.5, dtype=np.float64)
    first_u, first_v = calibration.image_points[0].astype(int)
    depth[first_v - 2:first_v + 3, first_u - 2:first_u + 3] = 0

    with pytest.raises(CharucoCalibrationError) as caught:
        build_charuco_observation(
            calibration,
            depth,
            np.array(
                [
                    [100.0, 0.0, 35.0],
                    [0.0, 100.0, 30.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            np.zeros(5),
            np.eye(4),
        )

    assert caught.value.valid_depth_corner_count == 19
    assert caught.value.invalid_depth_corner_count == 1
    assert caught.value.invalid_depth_reasons == {
        "insufficient_valid_depth": 1
    }


def _consistent_observation(errors_m):
    errors = np.asarray(errors_m, dtype=np.float64)
    corner_ids = np.arange(len(errors), dtype=np.int32)
    child_points = board_points_world(corner_ids)
    child_points[:, 2] += errors
    return CharucoFrameObservation(
        corner_ids=corner_ids,
        image_points=np.zeros((len(errors), 2)),
        depth_valid=np.ones(len(errors), dtype=bool),
        child_points=child_points,
        depth_valid_pixel_counts=np.full(len(errors), 25, dtype=np.uint16),
        depth_inlier_pixel_counts=np.full(len(errors), 25, dtype=np.uint16),
        depth_mad_m=np.zeros(len(errors)),
        depth_invalid_reasons=("",) * len(errors),
        camera_matrix=np.eye(3),
        distortion=np.zeros(5),
        color_from_child=np.eye(4),
        initial_reprojection_errors_px=np.full(len(errors), 0.2),
        initial_reprojection_rmse_px=0.2,
    )


def test_depth_consistency_accepts_errors_within_limits():
    metrics = validate_charuco_depth_consistency(
        _consistent_observation(np.full(20, 0.002)),
        np.eye(4),
    )

    assert metrics.median_error_m == pytest.approx(0.002)
    assert metrics.p95_error_m == pytest.approx(0.002)


def test_depth_consistency_rejects_median_over_limit():
    observation = _consistent_observation(
        np.full(20, DEPTH_CONSISTENCY_MEDIAN_LIMIT_M + 0.001)
    )

    with pytest.raises(
        CharucoCalibrationError,
        match=r"median corner error 6.00 mm \(maximum 5.00 mm\)",
    ):
        validate_charuco_depth_consistency(observation, np.eye(4))


def test_depth_consistency_rejects_p95_over_limit():
    errors = np.full(20, 0.002)
    errors[-2:] = DEPTH_CONSISTENCY_P95_LIMIT_M + 0.002

    with pytest.raises(
        CharucoCalibrationError,
        match=r"95th percentile 12.00 mm \(maximum 10.00 mm\)",
    ):
        validate_charuco_depth_consistency(
            _consistent_observation(errors),
            np.eye(4),
        )
