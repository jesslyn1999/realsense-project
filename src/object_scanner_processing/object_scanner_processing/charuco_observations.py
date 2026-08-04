"""Detect ChArUco corners and sample registered depth observations."""

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


SQUARES_X = 10
SQUARES_Y = 7
SQUARE_SIZE_M = 0.020
MARKER_SIZE_M = 0.015
MIN_CORNER_COUNT = 20
MAX_REPROJECTION_RMSE_PX = 1.5
DEPTH_WINDOW_RADIUS_PX = 2
MIN_DEPTH_INLIERS = 9
DEPTH_MAD_SCALE = 3.0
MIN_DEPTH_TOLERANCE_M = 0.001
DEPTH_CONSISTENCY_MEDIAN_LIMIT_M = 0.005
DEPTH_CONSISTENCY_P95_LIMIT_M = 0.010

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_BOARD = cv2.aruco.CharucoBoard(
    (SQUARES_X, SQUARES_Y),
    SQUARE_SIZE_M,
    MARKER_SIZE_M,
    _DICTIONARY,
)

# OpenCV's board frame has +Y down and +Z through the printed board. The world
# frame keeps the same top-left origin but points +Y up and +Z out of the board.
WORLD_FROM_OPENCV_BOARD = np.diag([1.0, -1.0, -1.0, 1.0])


def _empty_int32() -> np.ndarray:
    return np.empty(0, dtype=np.int32)


def _empty_points(columns: int) -> np.ndarray:
    return np.empty((0, columns), dtype=np.float64)


@dataclass(frozen=True)
class CharucoCalibration:
    """One accepted ChArUco camera pose and its image observations."""

    camera_to_world: np.ndarray
    corner_count: int
    reprojection_rmse_px: float
    corner_ids: np.ndarray = field(default_factory=_empty_int32)
    image_points: np.ndarray = field(
        default_factory=lambda: _empty_points(2)
    )
    board_points_opencv: np.ndarray = field(
        default_factory=lambda: _empty_points(3)
    )
    reprojection_errors_px: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def as_dict(self) -> dict:
        return {
            "corner_count": self.corner_count,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "matrix": self.camera_to_world.tolist(),
        }


@dataclass(frozen=True)
class CharucoDetection:
    """Detected marker and ChArUco image points before pose quality checks."""

    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray | None

    @property
    def corner_count(self) -> int:
        return 0 if self.charuco_ids is None else len(self.charuco_ids)


@dataclass(frozen=True)
class CharucoFrameObservation:
    """Serializable image and registered-depth evidence for one capture."""

    corner_ids: np.ndarray
    image_points: np.ndarray
    depth_valid: np.ndarray
    child_points: np.ndarray
    depth_valid_pixel_counts: np.ndarray
    depth_inlier_pixel_counts: np.ndarray
    depth_mad_m: np.ndarray
    depth_invalid_reasons: tuple[str, ...]
    camera_matrix: np.ndarray
    distortion: np.ndarray
    color_from_child: np.ndarray
    initial_reprojection_errors_px: np.ndarray
    initial_reprojection_rmse_px: float

    @property
    def valid_depth_corner_count(self) -> int:
        return int(np.sum(self.depth_valid))

    @property
    def invalid_depth_corner_count(self) -> int:
        return len(self.corner_ids) - self.valid_depth_corner_count

    def as_dict(self) -> dict:
        reason_counts = {}
        for valid, reason in zip(
            self.depth_valid,
            self.depth_invalid_reasons,
            strict=True,
        ):
            if valid:
                continue
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return {
            "corner_count": len(self.corner_ids),
            "reprojection_rmse_px": self.initial_reprojection_rmse_px,
            "valid_depth_corner_count": self.valid_depth_corner_count,
            "invalid_depth_corner_count": self.invalid_depth_corner_count,
            "invalid_depth_reasons": reason_counts,
        }


@dataclass(frozen=True)
class CharucoDepthConsistency:
    """Capture-time agreement between 2D board pose and sampled depth."""

    median_error_m: float
    p95_error_m: float

    def as_dict(self) -> dict:
        """Return browser-facing consistency metrics in millimeters."""
        return {
            "depth_consistency_median_mm": self.median_error_m * 1000.0,
            "depth_consistency_p95_mm": self.p95_error_m * 1000.0,
            "depth_consistency_median_limit_mm": (
                DEPTH_CONSISTENCY_MEDIAN_LIMIT_M * 1000.0
            ),
            "depth_consistency_p95_limit_mm": (
                DEPTH_CONSISTENCY_P95_LIMIT_M * 1000.0
            ),
        }


class CharucoCalibrationError(ValueError):
    """A rejected ChArUco capture with metrics suitable for the web UI."""

    def __init__(
        self,
        message: str,
        *,
        corner_count: int = 0,
        reprojection_rmse_px: float | None = None,
        valid_depth_corner_count: int = 0,
        invalid_depth_corner_count: int = 0,
        invalid_depth_reasons: dict[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.corner_count = corner_count
        self.reprojection_rmse_px = reprojection_rmse_px
        self.valid_depth_corner_count = valid_depth_corner_count
        self.invalid_depth_corner_count = invalid_depth_corner_count
        self.invalid_depth_reasons = invalid_depth_reasons or {}

    def as_dict(self) -> dict:
        return {
            "success": False,
            "message": str(self),
            "corner_count": self.corner_count,
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "valid_depth_corner_count": self.valid_depth_corner_count,
            "invalid_depth_corner_count": self.invalid_depth_corner_count,
            "invalid_depth_reasons": self.invalid_depth_reasons,
        }


def detect_charuco(image: np.ndarray) -> CharucoDetection:
    """Detect all visible board markers and interpolated ChArUco corners."""
    pixels = np.asarray(image)
    if pixels.ndim == 3 and pixels.shape[2] == 3:
        grayscale = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    elif pixels.ndim == 2:
        grayscale = pixels
    else:
        raise ValueError("image must have shape (H, W) or (H, W, 3)")
    if grayscale.dtype != np.uint8:
        raise ValueError("image must use uint8 pixels")

    try:
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            cv2.aruco.CharucoDetector(_BOARD).detectBoard(grayscale)
        )
    except cv2.error as error:
        raise CharucoCalibrationError(
            f"Cannot detect the ChArUco board: {error}"
        ) from error
    return CharucoDetection(
        charuco_corners=charuco_corners,
        charuco_ids=charuco_ids,
        marker_corners=tuple(marker_corners),
        marker_ids=marker_ids,
    )


def annotate_charuco(
    image: np.ndarray,
    detection: CharucoDetection | None = None,
) -> np.ndarray:
    """Return an RGB copy with every detected marker and corner drawn."""
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("preview image must have shape (H, W, 3)")
    if pixels.dtype != np.uint8:
        raise ValueError("preview image must use uint8 pixels")

    result = np.ascontiguousarray(pixels).copy()
    visible = detection if detection is not None else detect_charuco(result)
    if visible.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(
            result,
            visible.marker_corners,
            visible.marker_ids,
            borderColor=(0, 255, 0),
        )
    if visible.charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            result,
            visible.charuco_corners,
            visible.charuco_ids,
            cornerColor=(255, 0, 0),
        )
    return result


def _validated_corner_ids(ids: np.ndarray, corner_count: int) -> np.ndarray:
    corner_ids = np.asarray(ids, dtype=np.int32).reshape(-1)
    maximum_id = len(_BOARD.getChessboardCorners()) - 1
    if (
        len(corner_ids) != corner_count
        or len(np.unique(corner_ids)) != corner_count
        or np.any((corner_ids < 0) | (corner_ids > maximum_id))
    ):
        raise ValueError("ChArUco corner IDs are invalid")
    return corner_ids


def board_points_opencv(corner_ids: np.ndarray) -> np.ndarray:
    """Return exact OpenCV-board points for validated corner IDs."""
    ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
    return np.asarray(
        _BOARD.getChessboardCorners(),
        dtype=np.float64,
    )[ids].copy()


def board_points_world(corner_ids: np.ndarray) -> np.ndarray:
    """Return exact scanner-world board points for corner IDs."""
    points = board_points_opencv(corner_ids)
    return points @ WORLD_FROM_OPENCV_BOARD[:3, :3].T


def validate_charuco_depth_consistency(
    observation: CharucoFrameObservation,
    world_from_child: np.ndarray,
) -> CharucoDepthConsistency:
    """Reject a capture when its sampled depth disagrees with its 2D pose."""
    matrix = np.asarray(world_from_child, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0])
    ):
        raise ValueError("world_from_child must be a finite 4x4 transform")

    valid = observation.depth_valid
    observed_world = (
        observation.child_points[valid] @ matrix[:3, :3].T
        + matrix[:3, 3]
    )
    expected_world = board_points_world(observation.corner_ids[valid])
    errors_m = np.linalg.norm(observed_world - expected_world, axis=1)
    consistency = CharucoDepthConsistency(
        median_error_m=float(np.median(errors_m)),
        p95_error_m=float(np.percentile(errors_m, 95)),
    )
    if (
        consistency.median_error_m > DEPTH_CONSISTENCY_MEDIAN_LIMIT_M
        or consistency.p95_error_m > DEPTH_CONSISTENCY_P95_LIMIT_M
    ):
        raise CharucoCalibrationError(
            "ChArUco depth consistency failed: median corner error "
            f"{consistency.median_error_m * 1000.0:.2f} mm "
            f"(maximum {DEPTH_CONSISTENCY_MEDIAN_LIMIT_M * 1000.0:.2f} mm), "
            "95th percentile "
            f"{consistency.p95_error_m * 1000.0:.2f} mm "
            f"(maximum {DEPTH_CONSISTENCY_P95_LIMIT_M * 1000.0:.2f} mm)",
            corner_count=len(observation.corner_ids),
            reprojection_rmse_px=(
                observation.initial_reprojection_rmse_px
            ),
            valid_depth_corner_count=(
                observation.valid_depth_corner_count
            ),
            invalid_depth_corner_count=(
                observation.invalid_depth_corner_count
            ),
        )
    return consistency


def calibrate_charuco(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> CharucoCalibration:
    """Detect the board and return its accepted pose and matched corners."""
    detection = detect_charuco(image)
    corner_count = detection.corner_count
    if corner_count < MIN_CORNER_COUNT:
        raise CharucoCalibrationError(
            "ChArUco capture requires at least "
            f"{MIN_CORNER_COUNT} corners; detected {corner_count}",
            corner_count=corner_count,
        )
    corner_ids = _validated_corner_ids(detection.charuco_ids, corner_count)
    image_points = np.asarray(
        detection.charuco_corners,
        dtype=np.float64,
    ).reshape(-1, 2)
    return calibrate_charuco_points(
        board_points_opencv(corner_ids),
        image_points,
        camera_matrix,
        distortion,
        corner_ids=corner_ids,
    )


def calibrate_charuco_points(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    *,
    corner_ids: np.ndarray | None = None,
) -> CharucoCalibration:
    """Estimate and quality-check a pose from matched ChArUco corners."""
    board_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    pixel_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    corner_count = len(board_points)
    if corner_count != len(pixel_points):
        raise ValueError("object_points and image_points must have equal length")
    if corner_count < MIN_CORNER_COUNT:
        raise CharucoCalibrationError(
            "ChArUco capture requires at least "
            f"{MIN_CORNER_COUNT} corners; detected {corner_count}",
            corner_count=corner_count,
        )
    if not np.isfinite(board_points).all() or not np.isfinite(pixel_points).all():
        raise ValueError("ChArUco points must contain only finite values")
    if corner_ids is None:
        ids = np.arange(corner_count, dtype=np.int32)
    else:
        ids = _validated_corner_ids(corner_ids, corner_count)

    intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if (
        intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0])
    ):
        raise ValueError("camera_matrix must contain valid pinhole intrinsics")
    if coefficients.size not in {0, 4, 5, 8, 12, 14}:
        raise ValueError("distortion must contain 0, 4, 5, 8, 12, or 14 values")
    if not np.isfinite(coefficients).all():
        raise ValueError("distortion must contain only finite values")

    try:
        solved, rotation_vector, translation = cv2.solvePnP(
            board_points,
            pixel_points,
            intrinsics,
            coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as error:
        raise CharucoCalibrationError(
            f"Cannot estimate the ChArUco pose: {error}",
            corner_count=corner_count,
        ) from error
    if not solved or translation[2, 0] <= 0:
        raise CharucoCalibrationError(
            "Cannot estimate a valid ChArUco pose",
            corner_count=corner_count,
        )

    try:
        projected, _ = cv2.projectPoints(
            board_points,
            rotation_vector,
            translation,
            intrinsics,
            coefficients,
        )
    except cv2.error as error:
        raise CharucoCalibrationError(
            f"Cannot evaluate the ChArUco pose: {error}",
            corner_count=corner_count,
        ) from error
    residuals = projected.reshape(-1, 2) - pixel_points
    reprojection_errors_px = np.linalg.norm(residuals, axis=1)
    reprojection_rmse_px = math.sqrt(
        float(np.mean(reprojection_errors_px * reprojection_errors_px))
    )
    if (
        not math.isfinite(reprojection_rmse_px)
        or reprojection_rmse_px > MAX_REPROJECTION_RMSE_PX
    ):
        raise CharucoCalibrationError(
            "ChArUco reprojection RMSE "
            f"{reprojection_rmse_px:.3f} px exceeds "
            f"{MAX_REPROJECTION_RMSE_PX:.3f} px",
            corner_count=corner_count,
            reprojection_rmse_px=reprojection_rmse_px,
        )

    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_from_board = np.eye(4, dtype=np.float64)
    camera_from_board[:3, :3] = rotation
    camera_from_board[:3, 3] = translation.reshape(3)
    camera_to_world = WORLD_FROM_OPENCV_BOARD @ np.linalg.inv(
        camera_from_board
    )
    return CharucoCalibration(
        camera_to_world=camera_to_world,
        corner_count=corner_count,
        reprojection_rmse_px=reprojection_rmse_px,
        corner_ids=np.ascontiguousarray(ids),
        image_points=np.ascontiguousarray(pixel_points),
        board_points_opencv=np.ascontiguousarray(board_points),
        reprojection_errors_px=np.ascontiguousarray(
            reprojection_errors_px,
        ),
    )


def depth_image_to_meters(
    data: bytes,
    *,
    width: int,
    height: int,
    step: int,
    encoding: str,
    is_bigendian: bool,
) -> np.ndarray:
    """Decode one registered ROS depth image into float64 metres."""
    if width <= 0 or height <= 0:
        raise ValueError("depth image dimensions must be positive")
    if encoding == "16UC1":
        dtype = np.dtype(">u2" if is_bigendian else "<u2")
        scale = 0.001
    elif encoding == "32FC1":
        dtype = np.dtype(">f4" if is_bigendian else "<f4")
        scale = 1.0
    else:
        raise ValueError(
            "registered depth image must use 16UC1 or 32FC1 encoding"
        )
    row_bytes = width * dtype.itemsize
    if step < row_bytes or len(data) < step * height:
        raise ValueError("registered depth image data is truncated")
    pixels = np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=data,
        strides=(step, dtype.itemsize),
    )
    return pixels.astype(np.float64) * scale


def validated_charuco_observation(
    *,
    corner_ids,
    image_points,
    depth_valid,
    child_points,
    depth_valid_pixel_counts,
    depth_inlier_pixel_counts,
    depth_mad_m,
    depth_invalid_reasons,
    camera_matrix,
    distortion,
    color_from_child,
    initial_reprojection_errors_px,
    initial_reprojection_rmse_px,
) -> CharucoFrameObservation:
    """Validate and normalize one observation at a transport boundary."""
    ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
    count = len(ids)
    _validated_corner_ids(ids, count)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    valid = np.asarray(depth_valid, dtype=bool).reshape(-1)
    points = np.asarray(child_points, dtype=np.float64).reshape(-1, 3)
    valid_counts = np.asarray(
        depth_valid_pixel_counts,
        dtype=np.uint16,
    ).reshape(-1)
    inlier_counts = np.asarray(
        depth_inlier_pixel_counts,
        dtype=np.uint16,
    ).reshape(-1)
    mad = np.asarray(depth_mad_m, dtype=np.float64).reshape(-1)
    reasons = tuple(str(reason) for reason in depth_invalid_reasons)
    errors = np.asarray(
        initial_reprojection_errors_px,
        dtype=np.float64,
    ).reshape(-1)
    if any(
        len(values) != count
        for values in (
            pixels,
            valid,
            points,
            valid_counts,
            inlier_counts,
            mad,
            reasons,
            errors,
        )
    ):
        raise ValueError("ChArUco observation arrays have inconsistent lengths")
    if count < MIN_CORNER_COUNT or int(np.sum(valid)) < MIN_CORNER_COUNT:
        raise ValueError(
            "ChArUco observation requires at least 20 corners and "
            "20 valid depth corners"
        )
    if (
        not np.isfinite(pixels).all()
        or not np.isfinite(points[valid]).all()
        or not np.isfinite(errors).all()
        or np.any(errors < 0.0)
        or np.any(inlier_counts > valid_counts)
        or np.any(inlier_counts[valid] < MIN_DEPTH_INLIERS)
        or np.any(np.isfinite(mad) & (mad < 0.0))
    ):
        raise ValueError("ChArUco observation contains invalid corner metrics")
    if any(reasons[index] for index in np.flatnonzero(valid)):
        raise ValueError("Valid ChArUco depth corners cannot have error reasons")

    intrinsics = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    extrinsic = np.asarray(color_from_child, dtype=np.float64).reshape(4, 4)
    rmse = float(initial_reprojection_rmse_px)
    if (
        not np.isfinite(intrinsics).all()
        or intrinsics[0, 0] <= 0.0
        or intrinsics[1, 1] <= 0.0
        or not np.allclose(intrinsics[2], [0.0, 0.0, 1.0])
        or coefficients.size not in {0, 4, 5, 8, 12, 14}
        or not np.isfinite(coefficients).all()
        or not np.isfinite(extrinsic).all()
        or not np.allclose(extrinsic[3], [0.0, 0.0, 0.0, 1.0])
        or not math.isfinite(rmse)
        or rmse < 0.0
    ):
        raise ValueError("ChArUco observation contains invalid camera metadata")

    return CharucoFrameObservation(
        corner_ids=np.ascontiguousarray(ids),
        image_points=np.ascontiguousarray(pixels),
        depth_valid=np.ascontiguousarray(valid),
        child_points=np.ascontiguousarray(points),
        depth_valid_pixel_counts=np.ascontiguousarray(valid_counts),
        depth_inlier_pixel_counts=np.ascontiguousarray(inlier_counts),
        depth_mad_m=np.ascontiguousarray(mad),
        depth_invalid_reasons=reasons,
        camera_matrix=np.ascontiguousarray(intrinsics),
        distortion=np.ascontiguousarray(coefficients),
        color_from_child=np.ascontiguousarray(extrinsic),
        initial_reprojection_errors_px=np.ascontiguousarray(errors),
        initial_reprojection_rmse_px=rmse,
    )


def observation_from_message(message) -> CharucoFrameObservation | None:
    """Decode an optional generated ROS observation message."""
    if not message.corner_ids:
        return None
    return validated_charuco_observation(
        corner_ids=message.corner_ids,
        image_points=message.image_points,
        depth_valid=message.depth_valid,
        child_points=message.child_points,
        depth_valid_pixel_counts=message.depth_valid_pixel_counts,
        depth_inlier_pixel_counts=message.depth_inlier_pixel_counts,
        depth_mad_m=message.depth_mad_m,
        depth_invalid_reasons=message.depth_invalid_reasons,
        camera_matrix=message.camera_matrix,
        distortion=message.distortion,
        color_from_child=message.color_from_child,
        initial_reprojection_errors_px=(
            message.initial_reprojection_errors_px
        ),
        initial_reprojection_rmse_px=(
            message.initial_reprojection_rmse_px
        ),
    )


def build_charuco_observation(
    calibration: CharucoCalibration,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    color_from_child: np.ndarray,
) -> CharucoFrameObservation:
    """Sample robust 3D corner positions from RGB-registered depth."""
    depth = np.asarray(depth_m, dtype=np.float64)
    intrinsics = np.asarray(camera_matrix, dtype=np.float64)
    coefficients = np.asarray(distortion, dtype=np.float64).reshape(-1)
    extrinsic = np.asarray(color_from_child, dtype=np.float64)
    if depth.ndim != 2 or not np.isfinite(depth[np.isfinite(depth)]).all():
        raise ValueError("registered depth must have shape (H, W)")
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("camera_matrix must be a finite 3x3 matrix")
    if extrinsic.shape != (4, 4) or not np.isfinite(extrinsic).all():
        raise ValueError("color_from_child must be a finite 4x4 matrix")
    if not np.allclose(extrinsic[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("color_from_child has an invalid homogeneous row")

    corner_ids = np.asarray(calibration.corner_ids, dtype=np.int32)
    image_points = np.asarray(calibration.image_points, dtype=np.float64)
    corner_count = len(corner_ids)
    if corner_count < MIN_CORNER_COUNT or image_points.shape != (
        corner_count,
        2,
    ):
        raise ValueError("calibration does not contain matched corner data")
    try:
        normalized = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            intrinsics,
            coefficients,
        ).reshape(-1, 2)
    except cv2.error as error:
        raise ValueError(f"Cannot undistort ChArUco corners: {error}") from error

    depth_valid = np.zeros(corner_count, dtype=bool)
    child_points = np.zeros((corner_count, 3), dtype=np.float64)
    valid_counts = np.zeros(corner_count, dtype=np.uint16)
    inlier_counts = np.zeros(corner_count, dtype=np.uint16)
    depth_mad = np.full(corner_count, np.nan, dtype=np.float64)
    invalid_reasons = []
    child_from_color = np.linalg.inv(extrinsic)
    height, width = depth.shape

    for index, ((u, v), (normalized_x, normalized_y)) in enumerate(
        zip(image_points, normalized, strict=True)
    ):
        center_x = int(np.rint(u))
        center_y = int(np.rint(v))
        x0 = max(0, center_x - DEPTH_WINDOW_RADIUS_PX)
        x1 = min(width, center_x + DEPTH_WINDOW_RADIUS_PX + 1)
        y0 = max(0, center_y - DEPTH_WINDOW_RADIUS_PX)
        y1 = min(height, center_y + DEPTH_WINDOW_RADIUS_PX + 1)
        if x0 >= x1 or y0 >= y1:
            invalid_reasons.append("out_of_bounds")
            continue

        values = depth[y0:y1, x0:x1].reshape(-1)
        values = values[np.isfinite(values) & (values > 0.0)]
        valid_counts[index] = len(values)
        if len(values) < MIN_DEPTH_INLIERS:
            invalid_reasons.append("insufficient_valid_depth")
            continue

        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        depth_mad[index] = mad
        tolerance = max(
            DEPTH_MAD_SCALE * 1.4826 * mad,
            MIN_DEPTH_TOLERANCE_M,
        )
        inliers = values[np.abs(values - median) <= tolerance]
        inlier_counts[index] = len(inliers)
        if len(inliers) < MIN_DEPTH_INLIERS:
            invalid_reasons.append("insufficient_depth_inliers")
            continue

        z = float(np.median(inliers))
        color_point = np.array(
            [normalized_x * z, normalized_y * z, z, 1.0],
            dtype=np.float64,
        )
        child_point = child_from_color @ color_point
        if not np.isfinite(child_point[:3]).all():
            invalid_reasons.append("non_finite_3d_point")
            continue
        depth_valid[index] = True
        child_points[index] = child_point[:3]
        invalid_reasons.append("")

    valid_corner_count = int(np.sum(depth_valid))
    if valid_corner_count < MIN_CORNER_COUNT:
        reason_counts = {}
        for reason in invalid_reasons:
            if reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        raise CharucoCalibrationError(
            "ChArUco capture requires at least "
            f"{MIN_CORNER_COUNT} valid depth corners; sampled "
            f"{valid_corner_count} from {corner_count}",
            corner_count=corner_count,
            reprojection_rmse_px=calibration.reprojection_rmse_px,
            valid_depth_corner_count=valid_corner_count,
            invalid_depth_corner_count=corner_count - valid_corner_count,
            invalid_depth_reasons=reason_counts,
        )

    return validated_charuco_observation(
        corner_ids=corner_ids,
        image_points=image_points,
        depth_valid=depth_valid,
        child_points=child_points,
        depth_valid_pixel_counts=valid_counts,
        depth_inlier_pixel_counts=inlier_counts,
        depth_mad_m=depth_mad,
        depth_invalid_reasons=invalid_reasons,
        camera_matrix=intrinsics,
        distortion=coefficients,
        color_from_child=extrinsic,
        initial_reprojection_errors_px=calibration.reprojection_errors_px,
        initial_reprojection_rmse_px=calibration.reprojection_rmse_px,
    )
