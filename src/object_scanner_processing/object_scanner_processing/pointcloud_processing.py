"""Non-destructive filtering, registration, and fusion of recorded frames."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path

import cv2
import numpy as np
from object_scanner_processing.charuco_observations import (
    board_points_opencv,
    board_points_world,
    MIN_CORNER_COUNT,
    WORLD_FROM_OPENCV_BOARD,
)
from object_scanner_processing.recording import read_frames, RecordedFrame
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
try:
    import open3d as o3d
except ModuleNotFoundError:
    o3d = None


LOGGER = logging.getLogger(__name__)

VOXEL_SIZE_M = 0.002
RADIUS_M = 0.005
RADIUS_NEIGHBORS = 8
SOR_NEIGHBORS = 8
SOR_STD_RATIO = 2.0
MAX_FILTERED_FRACTION = 0.15
BOARD_CLEARANCE_M = 0.002
OBJECT_CLUSTER_EPS_M = 0.006
OBJECT_CLUSTER_MIN_POINTS = 10
OVERLAP_VOXEL_M = 0.003
OVERLAP_DISTANCE_M = 0.012
MIN_OVERLAP = 0.60
STRICT_OVERLAP_DISTANCE_M = 0.005
MIN_STRICT_OVERLAP = 0.15
MAX_CORRECTION_M = 0.010
MAX_CORRECTION_DEG = 2.0
ICP_LEVELS = (
    (0.006, 0.020),
    (0.003, 0.012),
    (0.002, 0.006),
)
ICP_MAX_ITERATIONS = 30
POSE_PRIOR_WEIGHT = 0.01
POSE_GRAPH_ROTATION_PRIOR = 100.0
POSE_GRAPH_TRANSLATION_PRIOR = 1000.0
TEMPORAL_SUPPORT_M = 0.004
TEMPORAL_COVERAGE_M = 0.012
TEMPORAL_WEAK_NEIGHBORS = 12
FUSION_VOXEL_M = 0.0025
CHARUCO_FRAME_WEIGHT_DECAY = 0.5
CHARUCO_3D_RESIDUAL_WEIGHT = 0.25
CHARUCO_3D_RESIDUAL_SCALE_M = 0.003
CHARUCO_MAX_REPROJECTION_ERROR_PX = 1.0
CHARUCO_CLOUD_DISTANCE_M = 0.003
CHARUCO_CLOUD_ACCEPTANCE_FRACTION = 0.99


class PointCloudProcessingError(RuntimeError):
    """Raised when a session cannot be refined without unsafe assumptions."""


def _require_open3d() -> None:
    if o3d is None:
        raise PointCloudProcessingError(
            "Open3D is unavailable; install open3d-cpu==0.19.0"
        )


@dataclass(frozen=True)
class FrameDiagnostics:
    """Point counts recorded for one frame at each processing stage."""

    frame_id: int
    input_points: int
    voxel_points: int
    radius_points: int
    cleaned_points: int
    temporal_removed: int = 0


@dataclass(frozen=True)
class EdgeDiagnostics:
    """Quality metrics for one candidate registration edge."""

    source_frame_id: int
    target_frame_id: int
    accepted: bool
    overlap_before: float
    overlap_after: float
    rmse_before_m: float
    rmse_after_m: float
    correction_m: float
    correction_deg: float
    reason: str


@dataclass(frozen=True)
class CharucoCornerDiagnostics:
    """Final residuals for one persisted ChArUco corner."""

    frame_id: int
    corner_id: int
    retained: bool
    depth_valid: bool
    reprojection_error_px: float
    corner_3d_residual_m: float | None


@dataclass(frozen=True)
class CharucoFrameDiagnostics:
    """Final corner and cloud acceptance metrics for one frame."""

    frame_id: int
    temporal_weight: float
    matched_prior_count: int
    corner_count: int
    valid_depth_count: int
    invalid_depth_count: int
    reprojection_rmse_px: float
    reprojection_max_px: float
    corner_3d_rmse_m: float
    corner_3d_max_m: float
    cloud_overlap_fraction_3mm: float


@dataclass(frozen=True)
class AlignedFrame:
    """One cleaned frame transformed by its optimized camera pose."""

    source: RecordedFrame
    optimized_pose: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    diagnostics: FrameDiagnostics


@dataclass(frozen=True)
class ProcessingResult:
    """Fused world cloud and the diagnostics that justify it."""

    xyz: np.ndarray
    rgb: np.ndarray
    observation_counts: np.ndarray
    optimized_poses: tuple[np.ndarray, ...]
    aligned_frames: tuple[AlignedFrame, ...]
    frames: tuple[FrameDiagnostics, ...]
    edges: tuple[EdgeDiagnostics, ...]
    charuco_frames: tuple[CharucoFrameDiagnostics, ...] = ()
    charuco_corners: tuple[CharucoCornerDiagnostics, ...] = ()
    cloud_overlap_fraction_3mm: float | None = None
    quality_warning: str | None = None

    @property
    def raw_points(self) -> int:
        return sum(frame.input_points for frame in self.frames)

    @property
    def cleaned_points(self) -> int:
        return sum(
            frame.cleaned_points - frame.temporal_removed
            for frame in self.frames
        )

    @property
    def accepted_edges(self) -> int:
        return sum(edge.accepted for edge in self.edges)

    @property
    def rejected_edges(self) -> int:
        return len(self.edges) - self.accepted_edges

    @property
    def summary(self) -> str:
        return (
            f"fused {len(self.xyz):,} points from {len(self.frames)} frames; "
            f"{self.accepted_edges} registration edges accepted, "
            f"{self.rejected_edges} rejected"
        )


@dataclass
class _PreparedFrame:
    recorded: RecordedFrame
    camera_xyz: np.ndarray
    rgb: np.ndarray
    initial_pose: np.ndarray
    cloud: o3d.geometry.PointCloud
    pyramids: dict[float, o3d.geometry.PointCloud]
    weak: np.ndarray
    diagnostics: FrameDiagnostics


@dataclass(frozen=True)
class _AcceptedEdge:
    source_index: int
    target_index: int
    transformation: np.ndarray
    information: np.ndarray
    diagnostics: EdgeDiagnostics


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]


def _rotation_degrees(matrix: np.ndarray) -> float:
    cosine = np.clip((np.trace(matrix[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _pose_delta(current: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return uncoupled rotation and camera-origin translation differences."""
    delta = np.eye(4)
    delta[:3, :3] = current[:3, :3] @ reference[:3, :3].T
    delta[:3, 3] = current[:3, 3] - reference[:3, 3]
    return delta


def _validate_matrix(frame_id: int, matrix: np.ndarray) -> None:
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise PointCloudProcessingError(
            f"Frame {frame_id} camera-to-world matrix must be finite and 4x4"
        )
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise PointCloudProcessingError(
            f"Frame {frame_id} camera-to-world matrix has an invalid last row"
        )
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise PointCloudProcessingError(
            f"Frame {frame_id} camera-to-world rotation is not orthonormal"
        )
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-6):
        raise PointCloudProcessingError(
            f"Frame {frame_id} camera-to-world rotation determinant is "
            f"{determinant:.6f}, not +1"
        )


def _validate_and_recover(
    frames: list[RecordedFrame],
) -> list[tuple[RecordedFrame, np.ndarray]]:
    if len(frames) < 2:
        raise PointCloudProcessingError(
            "Point-cloud refinement requires at least two captured frames"
        )
    parent_frames = {frame.parent_frame_id for frame in frames}
    if len(parent_frames) != 1:
        raise PointCloudProcessingError(
            "Recorded frames use inconsistent world parent frame IDs"
        )

    recovered = []
    for frame in frames:
        _validate_matrix(frame.id, frame.matrix)
        if frame.xyz.ndim != 2 or frame.xyz.shape[1:] != (3,):
            raise PointCloudProcessingError(
                f"Frame {frame.id} XYZ data must have shape (N, 3)"
            )
        if frame.rgb.shape != frame.xyz.shape:
            raise PointCloudProcessingError(
                f"Frame {frame.id} RGB data does not match its XYZ data"
            )
        if not len(frame.xyz):
            raise PointCloudProcessingError(f"Frame {frame.id} is empty")
        if not np.isfinite(frame.xyz).all():
            raise PointCloudProcessingError(
                f"Frame {frame.id} contains non-finite XYZ values"
            )

        camera_xyz = _transform_points(frame.xyz, np.linalg.inv(frame.matrix))
        round_trip = _transform_points(camera_xyz, frame.matrix)
        max_error = float(np.max(np.linalg.norm(round_trip - frame.xyz, axis=1)))
        if max_error > 2e-6:
            raise PointCloudProcessingError(
                f"Frame {frame.id} camera/world round-trip error is "
                f"{max_error * 1000.0:.4f} mm"
            )
        recovered.append((frame, camera_xyz))
    return recovered


def charuco_frame_weight(frame_index: int) -> float:
    """Return the replaceable capture-order weight used by all corner priors."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return CHARUCO_FRAME_WEIGHT_DECAY**frame_index


def _weighted_rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Fit one proper rigid transform from source points to target points."""
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    point_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if (
        source_points.shape != target_points.shape
        or source_points.ndim != 2
        or source_points.shape[1:] != (3,)
        or len(source_points) < 3
        or point_weights.shape != (len(source_points),)
        or not np.isfinite(source_points).all()
        or not np.isfinite(target_points).all()
        or not np.isfinite(point_weights).all()
        or np.any(point_weights <= 0.0)
    ):
        raise PointCloudProcessingError(
            "Cannot fit an invalid weighted ChArUco consensus"
        )
    normalized = point_weights / np.sum(point_weights)
    source_center = np.sum(source_points * normalized[:, None], axis=0)
    target_center = np.sum(target_points * normalized[:, None], axis=0)
    source_centered = source_points - source_center
    target_centered = target_points - target_center
    covariance = source_centered.T @ (
        normalized[:, None] * target_centered
    )
    left, _, right = np.linalg.svd(covariance)
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = target_center - rotation @ source_center
    return matrix


def _weighted_board_consensus(
    frames: list[RecordedFrame],
    poses: list[np.ndarray],
    current_index: int,
) -> tuple[np.ndarray, set[int]]:
    """Rigidly project prior weighted 3D observations onto the exact board."""
    weighted_points: dict[int, list[tuple[float, np.ndarray]]] = {}
    for frame_index in range(current_index):
        observation = frames[frame_index].charuco
        assert observation is not None
        weight = charuco_frame_weight(frame_index)
        valid_indices = np.flatnonzero(observation.depth_valid)
        world_points = _transform_points(
            observation.child_points[valid_indices],
            poses[frame_index],
        )
        for corner_id, world_point in zip(
            observation.corner_ids[valid_indices],
            world_points,
            strict=True,
        ):
            weighted_points.setdefault(int(corner_id), []).append(
                (weight, world_point)
            )

    current = frames[current_index].charuco
    assert current is not None
    current_valid_ids = {
        int(corner_id)
        for corner_id in current.corner_ids[current.depth_valid]
    }
    matched_ids = sorted(current_valid_ids & weighted_points.keys())
    if len(matched_ids) < MIN_CORNER_COUNT:
        raise PointCloudProcessingError(
            f"Frame {frames[current_index].id} shares only "
            f"{len(matched_ids)} valid depth corners with prior frames; "
            f"{MIN_CORNER_COUNT} are required"
        )

    consensus = []
    consensus_weights = []
    for corner_id in matched_ids:
        samples = weighted_points[corner_id]
        weights = np.asarray([sample[0] for sample in samples])
        points = np.asarray([sample[1] for sample in samples])
        consensus.append(np.average(points, axis=0, weights=weights))
        consensus_weights.append(float(np.sum(weights)))
    board_transform = _weighted_rigid_transform(
        board_points_world(np.asarray(matched_ids, dtype=np.int32)),
        np.asarray(consensus),
        np.asarray(consensus_weights),
    )
    return board_transform, set(matched_ids)


def _pose_from_update(update: np.ndarray, initial: np.ndarray) -> np.ndarray:
    delta = np.eye(4)
    delta[:3, :3] = cv2.Rodrigues(update[:3])[0]
    delta[:3, 3] = update[3:]
    return delta @ initial


def _solve_hybrid_charuco_pose(
    frame: RecordedFrame,
    board_transform: np.ndarray,
    matched_ids: set[int],
) -> np.ndarray:
    """Solve one pose from image reprojection and lower-weight 3D corners."""
    observation = frame.charuco
    assert observation is not None
    board_opencv = board_points_opencv(observation.corner_ids)
    solved, rotation_vector, translation = cv2.solvePnP(
        board_opencv,
        observation.image_points,
        observation.camera_matrix,
        observation.distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not solved or translation[2, 0] <= 0.0:
        raise PointCloudProcessingError(
            f"Frame {frame.id} ChArUco pose cannot be re-estimated"
        )
    color_from_board = np.eye(4)
    color_from_board[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    color_from_board[:3, 3] = translation.reshape(3)
    world_from_color = (
        WORLD_FROM_OPENCV_BOARD @ np.linalg.inv(color_from_board)
    )
    initial = world_from_color @ observation.color_from_child

    matched_indices = np.asarray(
        [
            index
            for index, corner_id in enumerate(observation.corner_ids)
            if observation.depth_valid[index]
            and int(corner_id) in matched_ids
        ],
        dtype=np.int64,
    )
    target_world = _transform_points(
        board_points_world(observation.corner_ids[matched_indices]),
        board_transform,
    )
    source_child = observation.child_points[matched_indices]
    world_from_opencv_board = WORLD_FROM_OPENCV_BOARD

    def residual(update: np.ndarray) -> np.ndarray:
        pose = _pose_from_update(update, initial)
        color_from_world = observation.color_from_child @ np.linalg.inv(pose)
        color_from_board_candidate = (
            color_from_world @ world_from_opencv_board
        )
        rotation = cv2.Rodrigues(
            color_from_board_candidate[:3, :3]
        )[0]
        projected, _ = cv2.projectPoints(
            board_opencv,
            rotation,
            color_from_board_candidate[:3, 3],
            observation.camera_matrix,
            observation.distortion,
        )
        reprojection = (
            projected.reshape(-1, 2) - observation.image_points
        ).reshape(-1)
        corner_3d = (
            _transform_points(source_child, pose) - target_world
        ).reshape(-1)
        corner_3d *= (
            math.sqrt(CHARUCO_3D_RESIDUAL_WEIGHT)
            / CHARUCO_3D_RESIDUAL_SCALE_M
        )
        return np.concatenate((reprojection, corner_3d))

    solution = least_squares(
        residual,
        np.zeros(6),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=100,
    )
    if not solution.success or not np.isfinite(solution.x).all():
        raise PointCloudProcessingError(
            f"Frame {frame.id} hybrid ChArUco pose solve failed: "
            f"{solution.message}"
        )
    pose = _pose_from_update(solution.x, initial)
    correction = _pose_delta(pose, frame.matrix)
    correction_m = float(np.linalg.norm(correction[:3, 3]))
    correction_deg = _rotation_degrees(correction)
    if (
        correction_m > MAX_CORRECTION_M
        or correction_deg > MAX_CORRECTION_DEG
    ):
        raise PointCloudProcessingError(
            f"Frame {frame.id} corner solve moved "
            f"{correction_m * 1000.0:.2f} mm and "
            f"{correction_deg:.3f} deg from its captured pose"
        )
    return pose


def _sequential_charuco_poses(
    frames: list[RecordedFrame],
) -> tuple[np.ndarray, ...]:
    """Anchor frame 1 and solve each later ChArUco pose from prior evidence."""
    charuco_frames = [
        frame for frame in frames if frame.transformation_name == "charuco"
    ]
    if not charuco_frames:
        return tuple(frame.matrix.astype(np.float64, copy=True) for frame in frames)
    if len(charuco_frames) != len(frames):
        raise PointCloudProcessingError(
            "A recording cannot mix ChArUco and non-ChArUco frames"
        )
    for frame in frames:
        if frame.charuco is None:
            raise PointCloudProcessingError(
                f"Frame {frame.id} has no persisted ChArUco corner observation"
            )

    poses = [frames[0].matrix.astype(np.float64, copy=True)]
    for frame_index in range(1, len(frames)):
        board_transform, matched_ids = _weighted_board_consensus(
            frames,
            poses,
            frame_index,
        )
        poses.append(
            _solve_hybrid_charuco_pose(
                frames[frame_index],
                board_transform,
                matched_ids,
            )
        )
    return tuple(poses)


def _fit_similarity_transform(
    source: np.ndarray,
    target: np.ndarray,
    frame_id: int,
) -> tuple[np.ndarray, float]:
    """Fit the 3D scale and rigid correction indicated by board corners."""
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    source_variance = float(np.mean(np.sum(source_zero**2, axis=1)))
    if source_variance <= np.finfo(np.float64).eps:
        raise PointCloudProcessingError(
            f"Frame {frame_id} ChArUco depth corners have no 3D extent"
        )

    u, singular_values, vt = np.linalg.svd(
        target_zero.T @ source_zero / len(source)
    )
    signs = np.ones(3)
    signs[-1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag(signs) @ vt
    scale = float(np.sum(singular_values * signs) / source_variance)
    if not np.isfinite(scale) or scale <= 0.0:
        raise PointCloudProcessingError(
            f"Frame {frame_id} ChArUco depth scale is invalid: {scale}"
        )

    correction = np.eye(4)
    correction[:3, :3] = scale * rotation
    correction[:3, 3] = (
        target_center - scale * rotation @ source_center
    )
    return correction, scale


def _correct_charuco_depth_geometry(
    recovered: list[tuple[RecordedFrame, np.ndarray]],
) -> list[tuple[RecordedFrame, np.ndarray]]:
    """Correct processed geometry from saved corners without changing raw data."""
    corrected = []
    for frame, camera_xyz in recovered:
        if frame.transformation_name != "charuco":
            corrected.append((frame, camera_xyz))
            continue
        observation = frame.charuco
        if observation is None:
            raise PointCloudProcessingError(
                f"Frame {frame.id} has no persisted ChArUco corner observation"
            )
        valid = observation.depth_valid
        expected_camera = _transform_points(
            board_points_world(observation.corner_ids[valid]),
            np.linalg.inv(frame.matrix),
        )
        correction, scale = _fit_similarity_transform(
            observation.child_points[valid],
            expected_camera,
            frame.id,
        )
        corrected_camera = _transform_points(camera_xyz, correction)
        corrected_child_points = observation.child_points.copy()
        corrected_child_points[valid] = _transform_points(
            corrected_child_points[valid],
            correction,
        )
        corrected_observation = replace(
            observation,
            child_points=corrected_child_points,
        )
        residuals = np.linalg.norm(
            corrected_child_points[valid] - expected_camera,
            axis=1,
        )
        LOGGER.info(
            "Frame %s ChArUco geometry correction: scale=%.6f "
            "corner RMSE=%.3f mm max=%.3f mm",
            frame.id,
            scale,
            math.sqrt(float(np.mean(residuals**2))) * 1000.0,
            float(np.max(residuals)) * 1000.0,
        )
        corrected.append(
            (
                replace(frame, charuco=corrected_observation),
                corrected_camera,
            )
        )
    return corrected


def _make_cloud(xyz: np.ndarray, rgb: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(xyz, dtype=np.float64)
    )
    cloud.colors = o3d.utility.Vector3dVector(
        np.ascontiguousarray(rgb, dtype=np.float64) / 255.0
    )
    return cloud


def _estimate_normals(
    cloud: o3d.geometry.PointCloud,
    voxel_size: float,
    frame_id: int,
) -> None:
    if len(cloud.points) < SOR_NEIGHBORS + 1:
        raise PointCloudProcessingError(
            f"Frame {frame_id} has too few points for normal estimation"
        )
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=max(voxel_size * 3.0, RADIUS_M),
            max_nn=30,
        )
    )
    cloud.normalize_normals()
    normals = np.asarray(cloud.normals)
    if normals.shape != (len(cloud.points), 3) or not np.isfinite(normals).all():
        raise PointCloudProcessingError(
            f"Frame {frame_id} normal estimation produced invalid normals"
        )


def _isolate_main_object(
    frame: RecordedFrame,
    camera_xyz: np.ndarray,
    initial_pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the ChArUco plane and retain the largest 3D component."""
    keep = np.ones(len(camera_xyz), dtype=bool)
    if frame.transformation_name == "charuco":
        initial_world_xyz = _transform_points(camera_xyz, initial_pose)
        keep = initial_world_xyz[:, 2] > BOARD_CLEARANCE_M
    candidate_xyz = camera_xyz[keep]
    candidate_rgb = frame.rgb[keep]
    if len(candidate_xyz) < OBJECT_CLUSTER_MIN_POINTS:
        raise PointCloudProcessingError(
            f"Frame {frame.id} has only {len(candidate_xyz)} points above the "
            f"{BOARD_CLEARANCE_M * 1000.0:.1f} mm board clearance"
        )

    candidate_cloud = _make_cloud(candidate_xyz, candidate_rgb)
    labels = np.asarray(
        candidate_cloud.cluster_dbscan(
            eps=OBJECT_CLUSTER_EPS_M,
            min_points=OBJECT_CLUSTER_MIN_POINTS,
            print_progress=False,
        )
    )
    clustered = labels >= 0
    if not np.any(clustered):
        raise PointCloudProcessingError(
            f"Frame {frame.id} has no connected object with at least "
            f"{OBJECT_CLUSTER_MIN_POINTS} points within "
            f"{OBJECT_CLUSTER_EPS_M * 1000.0:.1f} mm"
        )
    component_sizes = np.bincount(labels[clustered])
    main_component = int(np.argmax(component_sizes))
    object_mask = labels == main_component
    LOGGER.info(
        "Frame %s object isolation: input=%s above_board=%s object=%s",
        frame.id,
        len(frame.xyz),
        len(candidate_xyz),
        int(np.sum(object_mask)),
    )
    return candidate_xyz[object_mask], candidate_rgb[object_mask]


def _clean_frame(
    frame: RecordedFrame,
    camera_xyz: np.ndarray,
    initial_pose: np.ndarray,
) -> _PreparedFrame:
    object_xyz, object_rgb = _isolate_main_object(
        frame,
        camera_xyz,
        initial_pose,
    )
    cloud = _make_cloud(object_xyz, object_rgb)
    voxel_cloud = cloud.voxel_down_sample(VOXEL_SIZE_M)
    voxel_points = len(voxel_cloud.points)
    if voxel_points < SOR_NEIGHBORS + 1:
        raise PointCloudProcessingError(
            f"Frame {frame.id} has too few points after voxel downsampling"
        )

    radius_cloud, _ = voxel_cloud.remove_radius_outlier(
        nb_points=RADIUS_NEIGHBORS,
        radius=RADIUS_M,
    )
    radius_points = len(radius_cloud.points)
    if radius_points < SOR_NEIGHBORS + 1:
        raise PointCloudProcessingError(
            f"Frame {frame.id} has too few points after radius filtering"
        )
    cleaned, _ = radius_cloud.remove_statistical_outlier(
        nb_neighbors=SOR_NEIGHBORS,
        std_ratio=SOR_STD_RATIO,
    )
    cleaned_points = len(cleaned.points)
    if not cleaned_points:
        raise PointCloudProcessingError(
            f"Frame {frame.id} is empty after outlier filtering"
        )
    removed_fraction = 1.0 - cleaned_points / voxel_points
    if removed_fraction > MAX_FILTERED_FRACTION:
        raise PointCloudProcessingError(
            f"Frame {frame.id} outlier filters removed "
            f"{removed_fraction:.1%} of voxel points; safety limit is "
            f"{MAX_FILTERED_FRACTION:.0%}"
        )

    _, strong_indices = cleaned.remove_radius_outlier(
        nb_points=TEMPORAL_WEAK_NEIGHBORS,
        radius=RADIUS_M,
    )
    weak = np.ones(cleaned_points, dtype=bool)
    weak[np.asarray(strong_indices, dtype=np.int64)] = False

    pyramids = {}
    for voxel_size, _ in ICP_LEVELS:
        level = cleaned.voxel_down_sample(voxel_size)
        _estimate_normals(level, voxel_size, frame.id)
        pyramids[voxel_size] = level

    diagnostics = FrameDiagnostics(
        frame_id=frame.id,
        input_points=len(frame.xyz),
        voxel_points=voxel_points,
        radius_points=radius_points,
        cleaned_points=cleaned_points,
    )
    LOGGER.info(
        "Frame %s cleanup: input=%s voxel=%s radius=%s SOR=%s",
        frame.id,
        len(frame.xyz),
        voxel_points,
        radius_points,
        cleaned_points,
    )
    return _PreparedFrame(
        recorded=frame,
        camera_xyz=np.asarray(cleaned.points).copy(),
        rgb=np.rint(np.asarray(cleaned.colors) * 255.0).astype(np.uint8),
        initial_pose=initial_pose.astype(np.float64, copy=True),
        cloud=cleaned,
        pyramids=pyramids,
        weak=weak,
        diagnostics=diagnostics,
    )


def _symmetric_overlap(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    transformation: np.ndarray,
    max_distance: float,
) -> float:
    source_in_target = copy.deepcopy(source)
    source_in_target.transform(transformation)
    source_distances = np.asarray(
        source_in_target.compute_point_cloud_distance(target)
    )

    target_in_source = copy.deepcopy(target)
    target_in_source.transform(np.linalg.inv(transformation))
    target_distances = np.asarray(
        target_in_source.compute_point_cloud_distance(source)
    )
    if not len(source_distances) or not len(target_distances):
        return 0.0
    return float(
        (
            np.mean(source_distances <= max_distance)
            + np.mean(target_distances <= max_distance)
        )
        / 2.0
    )


def _robust_point_to_plane_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    max_distance: float,
    initial: np.ndarray,
    prior: np.ndarray,
) -> np.ndarray:
    """Refine one pyramid level with Tukey loss and a weak pose prior."""
    source_points = np.asarray(source.points)
    target_points = np.asarray(target.points)
    target_normals = np.asarray(target.normals)
    target_tensor = o3d.core.Tensor(target_points.astype(np.float32))
    nearest = o3d.core.nns.NearestNeighborSearch(target_tensor)
    nearest.knn_index()
    transformation = initial.copy()

    for _ in range(ICP_MAX_ITERATIONS):
        moved = _transform_points(source_points, transformation)
        indices, squared_distances = nearest.knn_search(
            o3d.core.Tensor(moved.astype(np.float32)),
            1,
        )
        indices = indices.numpy().ravel()
        squared_distances = squared_distances.numpy().ravel()
        within_distance = squared_distances <= max_distance**2
        if np.count_nonzero(within_distance) < SOR_NEIGHBORS + 1:
            raise PointCloudProcessingError(
                "Point-to-plane ICP has too few correspondences"
            )

        matched_source = moved[within_distance]
        matched_target = target_points[indices[within_distance]]
        matched_normals = target_normals[indices[within_distance]]
        residuals = np.sum(
            matched_normals * (matched_source - matched_target),
            axis=1,
        )

        median = float(np.median(residuals))
        mad = 1.4826 * float(np.median(np.abs(residuals - median)))
        tukey_scale = max(max_distance * 0.15, 4.685 * mad)
        normalized = residuals / tukey_scale
        weights = np.where(
            np.abs(normalized) < 1.0,
            (1.0 - normalized**2) ** 2,
            0.0,
        )
        robust = weights > 1e-6
        if np.count_nonzero(robust) < SOR_NEIGHBORS + 1:
            raise PointCloudProcessingError(
                "Tukey loss rejected too many ICP correspondences"
            )

        matched_source = matched_source[robust]
        matched_normals = matched_normals[robust]
        residuals = residuals[robust]
        weights = weights[robust]
        jacobian = np.column_stack(
            (
                np.cross(matched_source, matched_normals),
                matched_normals,
            )
        )
        hessian = jacobian.T @ (weights[:, None] * jacobian)
        gradient = jacobian.T @ (weights * residuals)

        correction = _pose_delta(transformation, prior)
        rotation_vector = cv2.Rodrigues(correction[:3, :3])[0].ravel()
        prior_residual = np.concatenate(
            (rotation_vector, correction[:3, 3])
        )
        diagonal = np.diag(hessian)
        prior_diagonal = np.concatenate(
            (
                np.full(3, max(diagonal[:3]) * POSE_PRIOR_WEIGHT),
                np.full(3, max(diagonal[3:]) * POSE_PRIOR_WEIGHT),
            )
        )
        hessian += np.diag(prior_diagonal)
        gradient += prior_diagonal * prior_residual

        try:
            update = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise PointCloudProcessingError(
                "Point-to-plane ICP normal equations are singular"
            ) from error

        update_matrix = np.eye(4)
        update_matrix[:3, :3] = cv2.Rodrigues(update[:3])[0]
        update_matrix[:3, 3] = update[3:]
        transformation = update_matrix @ transformation
        if (
            np.linalg.norm(update[:3]) < 1e-6
            and np.linalg.norm(update[3:]) < 1e-6
        ):
            break

    return transformation


def _register_edge(
    source_index: int,
    target_index: int,
    prepared: list[_PreparedFrame],
) -> tuple[_AcceptedEdge | None, EdgeDiagnostics]:
    source_frame = prepared[source_index]
    target_frame = prepared[target_index]
    source_overlap = source_frame.pyramids[OVERLAP_VOXEL_M]
    target_overlap = target_frame.pyramids[OVERLAP_VOXEL_M]
    initial = (
        np.linalg.inv(target_frame.initial_pose) @ source_frame.initial_pose
    )
    overlap_before = _symmetric_overlap(
        source_overlap,
        target_overlap,
        initial,
        OVERLAP_DISTANCE_M,
    )
    strict_overlap = _symmetric_overlap(
        source_overlap,
        target_overlap,
        initial,
        STRICT_OVERLAP_DISTANCE_M,
    )
    initial_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_overlap,
        target_overlap,
        OVERLAP_DISTANCE_M,
        initial,
    )
    if (
        overlap_before < MIN_OVERLAP
        or strict_overlap < MIN_STRICT_OVERLAP
    ):
        reasons = []
        if overlap_before < MIN_OVERLAP:
            reasons.append(
                f"initial 12 mm overlap {overlap_before:.1%} is below 60%"
            )
        if strict_overlap < MIN_STRICT_OVERLAP:
            reasons.append(
                f"initial 5 mm overlap {strict_overlap:.1%} is below "
                f"{MIN_STRICT_OVERLAP:.0%}"
            )
        diagnostics = EdgeDiagnostics(
            source_frame_id=source_frame.recorded.id,
            target_frame_id=target_frame.recorded.id,
            accepted=False,
            overlap_before=overlap_before,
            overlap_after=overlap_before,
            rmse_before_m=float(initial_evaluation.inlier_rmse),
            rmse_after_m=float(initial_evaluation.inlier_rmse),
            correction_m=0.0,
            correction_deg=0.0,
            reason="; ".join(reasons),
        )
        return None, diagnostics

    transformation = initial
    try:
        for voxel_size, correspondence_distance in ICP_LEVELS:
            # ponytail: the weak ChArUco prior suppresses pipe-axis drift; use a
            # constrained optimizer if corrections larger than 10 mm are required.
            transformation = _robust_point_to_plane_icp(
                source_frame.pyramids[voxel_size],
                target_frame.pyramids[voxel_size],
                correspondence_distance,
                transformation,
                initial,
            )
    except (PointCloudProcessingError, RuntimeError) as error:
        diagnostics = EdgeDiagnostics(
            source_frame_id=source_frame.recorded.id,
            target_frame_id=target_frame.recorded.id,
            accepted=False,
            overlap_before=overlap_before,
            overlap_after=overlap_before,
            rmse_before_m=float(initial_evaluation.inlier_rmse),
            rmse_after_m=float(initial_evaluation.inlier_rmse),
            correction_m=0.0,
            correction_deg=0.0,
            reason=f"ICP failed: {error}",
        )
        return None, diagnostics

    overlap_after = _symmetric_overlap(
        source_overlap,
        target_overlap,
        transformation,
        OVERLAP_DISTANCE_M,
    )
    final_evaluation = o3d.pipelines.registration.evaluate_registration(
        source_overlap,
        target_overlap,
        OVERLAP_DISTANCE_M,
        transformation,
    )
    correction = _pose_delta(transformation, initial)
    correction_m = float(np.linalg.norm(correction[:3, 3]))
    correction_deg = _rotation_degrees(correction)
    rmse_before = float(initial_evaluation.inlier_rmse)
    rmse_after = float(final_evaluation.inlier_rmse)
    accepted_reason = "accepted"
    if rmse_after > rmse_before + 1e-6:
        accepted_reason = (
            "accepted initial ChArUco alignment because ICP worsened RMSE "
            f"from {rmse_before * 1000.0:.2f} to "
            f"{rmse_after * 1000.0:.2f} mm"
        )
        transformation = initial
        overlap_after = overlap_before
        rmse_after = rmse_before
        correction_m = 0.0
        correction_deg = 0.0

    reasons = []
    if overlap_after < MIN_OVERLAP:
        reasons.append(f"overlap {overlap_after:.1%} is below 60%")
    if correction_m > MAX_CORRECTION_M:
        reasons.append(
            f"translation correction {correction_m * 1000.0:.2f} mm "
            "exceeds 10 mm"
        )
    if correction_deg > MAX_CORRECTION_DEG:
        reasons.append(
            f"rotation correction {correction_deg:.3f} deg exceeds 2 deg"
        )

    accepted = not reasons
    diagnostics = EdgeDiagnostics(
        source_frame_id=source_frame.recorded.id,
        target_frame_id=target_frame.recorded.id,
        accepted=accepted,
        overlap_before=overlap_before,
        overlap_after=overlap_after,
        rmse_before_m=rmse_before,
        rmse_after_m=rmse_after,
        correction_m=correction_m,
        correction_deg=correction_deg,
        reason=accepted_reason if accepted else "; ".join(reasons),
    )
    if not accepted:
        return None, diagnostics

    information = (
        o3d.pipelines.registration.get_information_matrix_from_point_clouds(
            source_frame.pyramids[ICP_LEVELS[-1][0]],
            target_frame.pyramids[ICP_LEVELS[-1][0]],
            ICP_LEVELS[-1][1],
            transformation,
        )
    )
    return (
        _AcceptedEdge(
            source_index=source_index,
            target_index=target_index,
            transformation=transformation,
            information=np.asarray(information).copy(),
            diagnostics=diagnostics,
        ),
        diagnostics,
    )


def _connected_frame_indices(
    frame_count: int,
    edges: list[_AcceptedEdge],
) -> set[int]:
    adjacency = [set() for _ in range(frame_count)]
    for edge in edges:
        adjacency[edge.source_index].add(edge.target_index)
        adjacency[edge.target_index].add(edge.source_index)
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            pending.append(neighbor)
    return visited


def _spanning_tree_edge_indices(
    frame_count: int,
    edges: list[_AcceptedEdge],
) -> set[int]:
    """Return the highest-overlap edges that connect every graph node."""
    parents = list(range(frame_count))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    tree = set()
    ranked = sorted(
        enumerate(edges),
        key=lambda item: (
            item[1].diagnostics.overlap_after,
            -item[1].diagnostics.rmse_after_m,
        ),
        reverse=True,
    )
    for edge_index, edge in ranked:
        source_root = root(edge.source_index)
        target_root = root(edge.target_index)
        if source_root == target_root:
            continue
        parents[target_root] = source_root
        tree.add(edge_index)
        if len(tree) == frame_count - 1:
            break
    return tree


def _optimize_poses(
    prepared: list[_PreparedFrame],
    edges: list[_AcceptedEdge],
    edge_diagnostics: list[EdgeDiagnostics],
) -> tuple[np.ndarray, ...]:
    frame_count = len(prepared)
    connected = _connected_frame_indices(frame_count, edges)
    if len(connected) != frame_count:
        candidate_edge_count = frame_count * (frame_count - 1) // 2
        minimum_edge_count = frame_count - 1
        minimum_edge_label = (
            "edge" if minimum_edge_count == 1 else "edges"
        )
        minimum_edge_verb = "is" if minimum_edge_count == 1 else "are"
        disconnected_ids = [
            frame.recorded.id
            for index, frame in enumerate(prepared)
            if index not in connected
        ]
        rejected = [
            diagnostics
            for diagnostics in edge_diagnostics
            if not diagnostics.accepted
        ]
        rejection_details = "; ".join(
            f"{diagnostics.source_frame_id}->{diagnostics.target_frame_id}: "
            f"{diagnostics.reason}"
            for diagnostics in rejected[:3]
        )
        if len(rejected) > 3:
            rejection_details += f"; plus {len(rejected) - 3} more"
        raise PointCloudProcessingError(
            "Accepted registration edges do not connect every captured frame: "
            f"{len(connected)}/{frame_count} frames connect to frame 1; "
            f"{frame_count}/{frame_count} are required. Accepted "
            f"{len(edges)}/{candidate_edge_count} candidate edges; at least "
            f"{minimum_edge_count} accepted {minimum_edge_label} "
            f"{minimum_edge_verb} necessary. "
            f"Disconnected frame IDs: {disconnected_ids}. Edge overlap "
            f"thresholds are {MIN_OVERLAP:.0%} within "
            f"{OVERLAP_DISTANCE_M * 1000.0:.0f} mm and "
            f"{MIN_STRICT_OVERLAP:.0%} within "
            f"{STRICT_OVERLAP_DISTANCE_M * 1000.0:.0f} mm"
            f". Rejected {len(rejected)}/{candidate_edge_count} candidate "
            f"edges: {rejection_details}"
        )

    pose_graph = o3d.pipelines.registration.PoseGraph()
    for frame in prepared:
        pose_graph.nodes.append(
            o3d.pipelines.registration.PoseGraphNode(frame.initial_pose)
        )
    spanning_tree = _spanning_tree_edge_indices(len(prepared), edges)
    for edge_index, edge in enumerate(edges):
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                edge.source_index,
                edge.target_index,
                edge.transformation,
                edge.information,
                uncertain=edge_index not in spanning_tree,
            )
        )
    prior_information = np.diag(
        [
            POSE_GRAPH_ROTATION_PRIOR,
            POSE_GRAPH_ROTATION_PRIOR,
            POSE_GRAPH_ROTATION_PRIOR,
            POSE_GRAPH_TRANSLATION_PRIOR,
            POSE_GRAPH_TRANSLATION_PRIOR,
            POSE_GRAPH_TRANSLATION_PRIOR,
        ]
    )
    for target_index in range(1, len(prepared)):
        scale = (
            charuco_frame_weight(target_index)
            if prepared[target_index].recorded.charuco is not None
            else 1.0
        )
        initial_relative = (
            np.linalg.inv(prepared[target_index].initial_pose)
            @ prepared[0].initial_pose
        )
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(
                0,
                target_index,
                initial_relative,
                prior_information * scale,
                uncertain=False,
            )
        )

    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=ICP_LEVELS[-1][1],
            edge_prune_threshold=0.25,
            preference_loop_closure=1.0,
            reference_node=0,
        ),
    )

    optimized = []
    for frame, node in zip(prepared, pose_graph.nodes, strict=True):
        pose = np.asarray(node.pose).copy()
        correction = _pose_delta(pose, frame.initial_pose)
        correction_m = float(np.linalg.norm(correction[:3, 3]))
        correction_deg = _rotation_degrees(correction)
        if (
            correction_m > MAX_CORRECTION_M
            or correction_deg > MAX_CORRECTION_DEG
        ):
            raise PointCloudProcessingError(
                f"Optimized frame {frame.recorded.id} moved "
                f"{correction_m * 1000.0:.2f} mm and "
                f"{correction_deg:.3f} deg from its recorded pose"
            )
        optimized.append(pose)
    return tuple(optimized)


def _world_cloud(
    frame: _PreparedFrame,
    pose: np.ndarray,
) -> o3d.geometry.PointCloud:
    cloud = copy.deepcopy(frame.cloud)
    cloud.transform(pose)
    return cloud


def _validate_optimized_edges(
    prepared: list[_PreparedFrame],
    poses: tuple[np.ndarray, ...],
    edges: list[_AcceptedEdge],
) -> None:
    for edge in edges:
        source = prepared[edge.source_index].pyramids[OVERLAP_VOXEL_M]
        target = prepared[edge.target_index].pyramids[OVERLAP_VOXEL_M]
        optimized_relative = (
            np.linalg.inv(poses[edge.target_index]) @ poses[edge.source_index]
        )
        evaluation = o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            OVERLAP_DISTANCE_M,
            optimized_relative,
        )
        overlap = _symmetric_overlap(
            source,
            target,
            optimized_relative,
            OVERLAP_DISTANCE_M,
        )
        if (
            overlap < MIN_OVERLAP
            or evaluation.inlier_rmse
            > edge.diagnostics.rmse_before_m + 1e-6
        ):
            raise PointCloudProcessingError(
                "Pose-graph optimization degraded accepted edge "
                f"{edge.diagnostics.source_frame_id} -> "
                f"{edge.diagnostics.target_frame_id}: overlap "
                f"{overlap:.1%}, RMSE "
                f"{evaluation.inlier_rmse * 1000.0:.2f} mm"
            )


def _project_charuco_points(
    frame: RecordedFrame,
    pose: np.ndarray,
) -> np.ndarray:
    observation = frame.charuco
    assert observation is not None
    color_from_world = observation.color_from_child @ np.linalg.inv(pose)
    color_from_board = color_from_world @ WORLD_FROM_OPENCV_BOARD
    projected, _ = cv2.projectPoints(
        board_points_opencv(observation.corner_ids),
        cv2.Rodrigues(color_from_board[:3, :3])[0],
        color_from_board[:3, 3],
        observation.camera_matrix,
        observation.distortion,
    )
    return projected.reshape(-1, 2)


def _mutual_overlap_distances(
    source_world: np.ndarray,
    target_world: np.ndarray,
) -> np.ndarray:
    """Return reciprocal nearest-neighbor distances in the overlap band."""
    target_tree = cKDTree(target_world)
    source_distances, source_targets = target_tree.query(source_world, k=1)
    source_tree = cKDTree(source_world)
    _, target_sources = source_tree.query(target_world, k=1)
    source_indices = np.arange(len(source_world))
    mutual = target_sources[source_targets] == source_indices
    distances = source_distances[mutual]
    return distances[distances <= OVERLAP_DISTANCE_M]


def _validate_charuco_acceptance(
    prepared: list[_PreparedFrame],
    poses: tuple[np.ndarray, ...],
    edges: list[_AcceptedEdge],
    aligned_world_xyz: list[np.ndarray],
) -> tuple[
    tuple[CharucoFrameDiagnostics, ...],
    tuple[CharucoCornerDiagnostics, ...],
    float | None,
]:
    if prepared[0].recorded.charuco is None:
        return (), (), None

    per_frame_cloud_fractions = [[] for _ in prepared]
    all_cloud_fractions = []
    for edge in edges:
        source_world = aligned_world_xyz[edge.source_index]
        target_world = aligned_world_xyz[edge.target_index]
        distances = _mutual_overlap_distances(source_world, target_world)
        if len(distances) < SOR_NEIGHBORS + 1:
            raise PointCloudProcessingError(
                "Accepted cloud edge "
                f"{edge.diagnostics.source_frame_id} -> "
                f"{edge.diagnostics.target_frame_id} has too few mutual "
                "overlap correspondences for the 3 mm check"
            )
        fraction = float(np.mean(distances <= CHARUCO_CLOUD_DISTANCE_M))
        if fraction < CHARUCO_CLOUD_ACCEPTANCE_FRACTION:
            raise PointCloudProcessingError(
                "Accepted cloud edge "
                f"{edge.diagnostics.source_frame_id} -> "
                f"{edge.diagnostics.target_frame_id} has only "
                f"{fraction:.3%} mutual correspondences within 3 mm; "
                f"{CHARUCO_CLOUD_ACCEPTANCE_FRACTION:.0%} are required"
            )
        per_frame_cloud_fractions[edge.source_index].append(fraction)
        per_frame_cloud_fractions[edge.target_index].append(fraction)
        all_cloud_fractions.append(fraction)

    frame_diagnostics = []
    corner_diagnostics = []
    prior_valid_ids: set[int] = set()
    for frame_index, (frame, pose) in enumerate(
        zip(prepared, poses, strict=True)
    ):
        observation = frame.recorded.charuco
        assert observation is not None
        projected = _project_charuco_points(frame.recorded, pose)
        reprojection_errors = np.linalg.norm(
            projected - observation.image_points,
            axis=1,
        )
        reprojection_max = float(np.max(reprojection_errors))
        reprojection_rmse = math.sqrt(
            float(np.mean(reprojection_errors**2))
        )
        if (
            not np.isfinite(reprojection_errors).all()
            or reprojection_max > CHARUCO_MAX_REPROJECTION_ERROR_PX
        ):
            raise PointCloudProcessingError(
                f"Frame {frame.recorded.id} final ChArUco reprojection "
                f"maximum is {reprojection_max:.3f} px; every corner must "
                "be within 1 px"
            )

        valid_indices = np.flatnonzero(observation.depth_valid)
        observed_world = _transform_points(
            observation.child_points[valid_indices],
            pose,
        )
        expected_world = board_points_world(
            observation.corner_ids[valid_indices]
        )
        corner_3d_residuals = np.linalg.norm(
            observed_world - expected_world,
            axis=1,
        )
        residual_by_index = {
            int(index): float(residual)
            for index, residual in zip(
                valid_indices,
                corner_3d_residuals,
                strict=True,
            )
        }
        for corner_index, corner_id in enumerate(observation.corner_ids):
            corner_diagnostics.append(
                CharucoCornerDiagnostics(
                    frame_id=frame.recorded.id,
                    corner_id=int(corner_id),
                    retained=True,
                    depth_valid=bool(
                        observation.depth_valid[corner_index]
                    ),
                    reprojection_error_px=float(
                        reprojection_errors[corner_index]
                    ),
                    corner_3d_residual_m=residual_by_index.get(
                        corner_index
                    ),
                )
            )
        matched_prior = len(
            prior_valid_ids
            & {
                int(corner_id)
                for corner_id in observation.corner_ids[
                    observation.depth_valid
                ]
            }
        )
        prior_valid_ids.update(
            int(corner_id)
            for corner_id in observation.corner_ids[
                observation.depth_valid
            ]
        )
        cloud_fraction = min(per_frame_cloud_fractions[frame_index])
        frame_diagnostics.append(
            CharucoFrameDiagnostics(
                frame_id=frame.recorded.id,
                temporal_weight=charuco_frame_weight(frame_index),
                matched_prior_count=matched_prior,
                corner_count=len(observation.corner_ids),
                valid_depth_count=observation.valid_depth_corner_count,
                invalid_depth_count=(
                    observation.invalid_depth_corner_count
                ),
                reprojection_rmse_px=reprojection_rmse,
                reprojection_max_px=reprojection_max,
                corner_3d_rmse_m=math.sqrt(
                    float(np.mean(corner_3d_residuals**2))
                ),
                corner_3d_max_m=float(np.max(corner_3d_residuals)),
                cloud_overlap_fraction_3mm=cloud_fraction,
            )
        )
    return (
        tuple(frame_diagnostics),
        tuple(corner_diagnostics),
        min(all_cloud_fractions),
    )


def _temporal_filter(
    prepared: list[_PreparedFrame],
    poses: tuple[np.ndarray, ...],
    edges: list[_AcceptedEdge],
) -> tuple[list[np.ndarray], list[np.ndarray], list[FrameDiagnostics]]:
    world_clouds = [
        _world_cloud(frame, pose)
        for frame, pose in zip(prepared, poses, strict=True)
    ]
    support = [
        np.zeros(len(frame.camera_xyz), dtype=bool) for frame in prepared
    ]
    covered_before = [
        np.zeros(len(frame.camera_xyz), dtype=np.uint16) for frame in prepared
    ]
    covered_after = [
        np.zeros(len(frame.camera_xyz), dtype=np.uint16) for frame in prepared
    ]
    available_before = np.zeros(len(prepared), dtype=np.uint16)
    available_after = np.zeros(len(prepared), dtype=np.uint16)

    def compare(source_index: int, target_index: int) -> None:
        distances = np.asarray(
            world_clouds[source_index].compute_point_cloud_distance(
                world_clouds[target_index]
            )
        )
        support[source_index] |= distances <= TEMPORAL_SUPPORT_M
        covered = distances <= TEMPORAL_COVERAGE_M
        if target_index < source_index:
            covered_before[source_index] += covered
            available_before[source_index] += 1
        else:
            covered_after[source_index] += covered
            available_after[source_index] += 1

    for edge in edges:
        compare(edge.source_index, edge.target_index)
        compare(edge.target_index, edge.source_index)

    retained_xyz = []
    retained_rgb = []
    diagnostics = []
    for index, frame in enumerate(prepared):
        comparison_count = (
            available_before[index] + available_after[index]
        )
        if available_before[index] and available_after[index]:
            sufficiently_covered = (
                (covered_before[index] > 0)
                & (covered_after[index] > 0)
            )
        elif available_before[index]:
            sufficiently_covered = covered_before[index] >= 2
        else:
            sufficiently_covered = covered_after[index] >= 2
        remove = (
            frame.weak
            & sufficiently_covered
            & (comparison_count >= 2)
            & ~support[index]
        )
        keep = ~remove
        retained_xyz.append(np.asarray(world_clouds[index].points)[keep].copy())
        retained_rgb.append(frame.rgb[keep].copy())
        diagnostics.append(
            replace(frame.diagnostics, temporal_removed=int(np.sum(remove)))
        )
    return retained_xyz, retained_rgb, diagnostics


def _fuse_voxels(
    xyz_parts: list[np.ndarray],
    rgb_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.concatenate(xyz_parts)
    colors = np.concatenate(rgb_parts)
    frame_ids = np.concatenate(
        [
            np.full(len(frame_points), frame_index, dtype=np.int32)
            for frame_index, frame_points in enumerate(xyz_parts)
        ]
    )
    keys = np.floor(points / FUSION_VOXEL_M).astype(np.int64)
    order = np.lexsort(
        (frame_ids, keys[:, 2], keys[:, 1], keys[:, 0])
    )
    sorted_keys = keys[order]
    sorted_frame_ids = frame_ids[order]
    pair_starts = np.concatenate(
        (
            [0],
            np.flatnonzero(
                np.any(np.diff(sorted_keys, axis=0), axis=1)
                | (np.diff(sorted_frame_ids) != 0)
            )
            + 1,
        )
    )
    pair_stops = np.concatenate((pair_starts[1:], [len(order)]))
    pair_xyz = np.empty((len(pair_starts), 3), dtype=np.float64)
    pair_rgb = np.empty((len(pair_starts), 3), dtype=np.float64)
    for pair_index, (start, stop) in enumerate(
        zip(pair_starts, pair_stops, strict=True)
    ):
        members = order[start:stop]
        pair_xyz[pair_index] = np.median(points[members], axis=0)
        pair_rgb[pair_index] = np.median(colors[members], axis=0)

    pair_keys = sorted_keys[pair_starts]
    voxel_starts = np.concatenate(
        (
            [0],
            np.flatnonzero(np.any(np.diff(pair_keys, axis=0), axis=1)) + 1,
        )
    )
    voxel_stops = np.concatenate(
        (voxel_starts[1:], [len(pair_starts)])
    )

    fused_xyz = np.empty((len(voxel_starts), 3), dtype=np.float32)
    fused_rgb = np.empty((len(voxel_starts), 3), dtype=np.uint8)
    observation_counts = np.empty(len(voxel_starts), dtype=np.uint16)
    for output_index, (start, stop) in enumerate(
        zip(voxel_starts, voxel_stops, strict=True)
    ):
        fused_xyz[output_index] = np.median(pair_xyz[start:stop], axis=0)
        fused_rgb[output_index] = np.rint(
            np.median(pair_rgb[start:stop], axis=0)
        ).astype(np.uint8)
        observation_counts[output_index] = stop - start
    return fused_xyz, fused_rgb, observation_counts


def _process_frames(
    frames: list[RecordedFrame],
    allow_degraded_quality: bool,
) -> ProcessingResult:
    if len({frame.id for frame in frames}) != len(frames):
        raise PointCloudProcessingError(
            "Captured frame IDs must be unique"
        )
    frames = sorted(frames, key=lambda frame: frame.id)
    recovered = _correct_charuco_depth_geometry(
        _validate_and_recover(frames)
    )
    processed_frames = [frame for frame, _ in recovered]
    initial_poses = _sequential_charuco_poses(processed_frames)
    prepared = [
        _clean_frame(frame, camera_xyz, initial_pose)
        for (frame, camera_xyz), initial_pose in zip(
            recovered,
            initial_poses,
            strict=True,
        )
    ]

    accepted_edges = []
    edge_diagnostics = []
    for source_index in range(len(prepared)):
        for target_index in range(source_index + 1, len(prepared)):
            edge, diagnostics = _register_edge(
                source_index,
                target_index,
                prepared,
            )
            edge_diagnostics.append(diagnostics)
            if edge is None:
                LOGGER.warning(
                    "Rejected registration edge %s -> %s: %s",
                    diagnostics.source_frame_id,
                    diagnostics.target_frame_id,
                    diagnostics.reason,
                )
            else:
                accepted_edges.append(edge)
                LOGGER.info(
                    "Accepted registration edge %s -> %s: overlap %.1f%% "
                    "to %.1f%%, RMSE %.2f to %.2f mm, correction %.2f mm "
                    "%.3f deg",
                    diagnostics.source_frame_id,
                    diagnostics.target_frame_id,
                    diagnostics.overlap_before * 100.0,
                    diagnostics.overlap_after * 100.0,
                    diagnostics.rmse_before_m * 1000.0,
                    diagnostics.rmse_after_m * 1000.0,
                    diagnostics.correction_m * 1000.0,
                    diagnostics.correction_deg,
                )

    poses = _optimize_poses(
        prepared,
        accepted_edges,
        edge_diagnostics,
    )
    try:
        _validate_optimized_edges(prepared, poses, accepted_edges)
    except PointCloudProcessingError as error:
        LOGGER.warning(
            "%s; using corrected ChArUco poses instead",
            error,
        )
        poses = tuple(frame.initial_pose.copy() for frame in prepared)
        _validate_optimized_edges(prepared, poses, accepted_edges)
    xyz_parts, rgb_parts, frame_diagnostics = _temporal_filter(
        prepared,
        poses,
        accepted_edges,
    )
    quality_warning = None
    try:
        (
            charuco_frame_diagnostics,
            charuco_corner_diagnostics,
            cloud_overlap_fraction_3mm,
        ) = _validate_charuco_acceptance(
            prepared,
            poses,
            accepted_edges,
            xyz_parts,
        )
    except PointCloudProcessingError as error:
        if not allow_degraded_quality:
            raise
        quality_warning = str(error)
        LOGGER.warning(
            "Saving best-effort aligned output without strict ChArUco "
            "acceptance metrics: %s",
            quality_warning,
        )
        charuco_frame_diagnostics = ()
        charuco_corner_diagnostics = ()
        cloud_overlap_fraction_3mm = None
    fused_xyz, fused_rgb, observation_counts = _fuse_voxels(
        xyz_parts,
        rgb_parts,
    )
    if not len(fused_xyz):
        raise PointCloudProcessingError("Fusion produced an empty point cloud")

    aligned_frames = tuple(
        AlignedFrame(
            source=frame.recorded,
            optimized_pose=pose.copy(),
            xyz=np.ascontiguousarray(xyz, dtype="<f4"),
            rgb=np.ascontiguousarray(rgb, dtype=np.uint8),
            diagnostics=diagnostics,
        )
        for frame, pose, xyz, rgb, diagnostics in zip(
            prepared,
            poses,
            xyz_parts,
            rgb_parts,
            frame_diagnostics,
            strict=True,
        )
    )
    result = ProcessingResult(
        xyz=fused_xyz,
        rgb=fused_rgb,
        observation_counts=observation_counts,
        optimized_poses=poses,
        aligned_frames=aligned_frames,
        frames=tuple(frame_diagnostics),
        edges=tuple(edge_diagnostics),
        charuco_frames=charuco_frame_diagnostics,
        charuco_corners=charuco_corner_diagnostics,
        cloud_overlap_fraction_3mm=cloud_overlap_fraction_3mm,
        quality_warning=quality_warning,
    )
    LOGGER.info("Point-cloud refinement complete: %s", result.summary)
    return result


def process_frames(
    frames: list[RecordedFrame],
    *,
    allow_degraded_quality: bool = False,
) -> ProcessingResult:
    """Validate, register, and fuse complete recorded frames."""
    _require_open3d()
    try:
        return _process_frames(frames, allow_degraded_quality)
    except PointCloudProcessingError:
        raise
    except RuntimeError as error:
        raise PointCloudProcessingError(
            f"Open3D processing failed: {error}"
        ) from error


def process_recording(
    database_path: Path,
    *,
    allow_degraded_quality: bool = False,
) -> ProcessingResult:
    """Process one SQLite recording without writing to it."""
    return process_frames(
        read_frames(database_path),
        allow_degraded_quality=allow_degraded_quality,
    )
