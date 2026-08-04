"""Read complete point-cloud frames from the shared scanner SQLite schema."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import numpy as np

from object_scanner_processing.charuco_observations import (
    CharucoFrameObservation,
    validated_charuco_observation,
)


@dataclass(frozen=True)
class RecordedFrame:
    """One complete world-frame capture and its camera-to-world matrix."""

    id: int
    recorded_perf_counter_ns: int
    source_sec: int
    source_nanosec: int
    parent_frame_id: str
    transformation_name: str
    matrix: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    charuco: CharucoFrameObservation | None = None


def _read_charuco_observations(
    connection: sqlite3.Connection,
) -> dict[int, CharucoFrameObservation]:
    tables = {
        row[0]
        for row in connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table'
              AND name IN ('charuco_observations', 'charuco_corners')
            """
        )
    }
    if not tables:
        return {}
    if tables != {"charuco_observations", "charuco_corners"}:
        raise ValueError("Recording has an incomplete ChArUco schema")

    frame_rows = connection.execute(
        """
        SELECT frame_id, corner_count, valid_depth_corner_count,
               initial_reprojection_rmse_px, camera_matrix,
               distortion_count, distortion, color_from_child
        FROM charuco_observations
        """
    ).fetchall()
    corner_rows = connection.execute(
        """
        SELECT frame_id, corner_id, image_u, image_v, depth_valid,
               child_x, child_y, child_z, valid_pixel_count,
               inlier_pixel_count, depth_mad_m, invalid_reason,
               initial_reprojection_error_px
        FROM charuco_corners
        ORDER BY frame_id, corner_id
        """
    ).fetchall()
    corners_by_frame: dict[int, list[tuple]] = {}
    for row in corner_rows:
        corners_by_frame.setdefault(int(row[0]), []).append(row[1:])

    observations = {}
    for (
        frame_id,
        corner_count,
        valid_depth_corner_count,
        initial_rmse,
        camera_matrix_blob,
        distortion_count,
        distortion_blob,
        color_from_child_blob,
    ) in frame_rows:
        frame_id = int(frame_id)
        corners = corners_by_frame.pop(frame_id, [])
        if len(corners) != corner_count:
            raise ValueError(
                f"Frame {frame_id} contains invalid ChArUco corner rows"
            )
        camera_matrix = np.frombuffer(camera_matrix_blob, dtype="<f8")
        distortion = np.frombuffer(distortion_blob, dtype="<f8")
        color_from_child = np.frombuffer(
            color_from_child_blob,
            dtype="<f8",
        )
        if (
            camera_matrix.size != 9
            or distortion.size != distortion_count
            or color_from_child.size != 16
        ):
            raise ValueError(
                f"Frame {frame_id} contains invalid ChArUco camera metadata"
            )

        ids = []
        image_points = []
        depth_valid = []
        child_points = []
        valid_counts = []
        inlier_counts = []
        depth_mad = []
        reasons = []
        errors = []
        for (
            corner_id,
            image_u,
            image_v,
            valid,
            child_x,
            child_y,
            child_z,
            valid_count,
            inlier_count,
            mad,
            reason,
            error,
        ) in corners:
            is_valid = bool(valid)
            if is_valid and (
                child_x is None or child_y is None or child_z is None
            ):
                raise ValueError(
                    f"Frame {frame_id} has a valid corner without XYZ"
                )
            ids.append(corner_id)
            image_points.append((image_u, image_v))
            depth_valid.append(is_valid)
            child_points.append(
                (
                    child_x if is_valid else 0.0,
                    child_y if is_valid else 0.0,
                    child_z if is_valid else 0.0,
                )
            )
            valid_counts.append(valid_count)
            inlier_counts.append(inlier_count)
            depth_mad.append(np.nan if mad is None else mad)
            reasons.append(reason)
            errors.append(error)
        try:
            observation = validated_charuco_observation(
                corner_ids=ids,
                image_points=image_points,
                depth_valid=depth_valid,
                child_points=child_points,
                depth_valid_pixel_counts=valid_counts,
                depth_inlier_pixel_counts=inlier_counts,
                depth_mad_m=depth_mad,
                depth_invalid_reasons=reasons,
                camera_matrix=camera_matrix.reshape(3, 3),
                distortion=distortion,
                color_from_child=color_from_child.reshape(4, 4),
                initial_reprojection_errors_px=errors,
                initial_reprojection_rmse_px=initial_rmse,
            )
        except ValueError as error:
            raise ValueError(
                f"Frame {frame_id} contains invalid ChArUco data: {error}"
            ) from error
        if observation.valid_depth_corner_count != valid_depth_corner_count:
            raise ValueError(
                f"Frame {frame_id} has an invalid depth-corner count"
            )
        observations[frame_id] = observation
    if corners_by_frame:
        raise ValueError("Recording has orphaned ChArUco corner rows")
    return observations


def read_frames(database_path: Path) -> list[RecordedFrame]:
    """Return complete frames in capture order without writing."""
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT id, recorded_perf_counter_ns, source_sec, source_nanosec, frame_id,
                   transformation_name, transformation_matrix,
                   point_count, xyz, rgb
            FROM frames
            ORDER BY id
            """
        ).fetchall()
        charuco_observations = _read_charuco_observations(connection)
    finally:
        connection.close()

    frames = []
    for (
        frame_id,
        recorded_perf_counter_ns,
        source_sec,
        source_nanosec,
        parent_frame_id,
        transformation_name,
        matrix_blob,
        point_count,
        xyz_blob,
        rgb_blob,
    ) in rows:
        matrix = np.frombuffer(matrix_blob, dtype="<f8")
        expected_xyz_bytes = point_count * 3 * np.dtype("<f4").itemsize
        expected_rgb_bytes = point_count * 3
        if matrix.size != 16:
            raise ValueError(
                f"Frame {frame_id} contains invalid transformation metadata"
            )
        if (
            point_count < 0
            or len(xyz_blob) != expected_xyz_bytes
            or len(rgb_blob) != expected_rgb_bytes
        ):
            raise ValueError(f"Frame {frame_id} contains invalid point data")
        frames.append(
            RecordedFrame(
                id=int(frame_id),
                recorded_perf_counter_ns=int(recorded_perf_counter_ns),
                source_sec=int(source_sec),
                source_nanosec=int(source_nanosec),
                parent_frame_id=parent_frame_id,
                transformation_name=transformation_name,
                matrix=matrix.reshape(4, 4).copy(),
                xyz=np.frombuffer(xyz_blob, dtype="<f4").reshape(-1, 3).copy(),
                rgb=np.frombuffer(rgb_blob, dtype=np.uint8).reshape(-1, 3).copy(),
                charuco=charuco_observations.get(int(frame_id)),
            )
        )
    return frames
