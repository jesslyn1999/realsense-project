"""Persist and read non-destructive aligned scanner results."""

from dataclasses import dataclass
from pathlib import Path
import sqlite3
from time import perf_counter_ns

import numpy as np

from object_scanner_processing.pointcloud_processing import (
    process_recording,
    ProcessingResult,
)


ALIGNED_DATABASE_FILENAME = "aligned_recording.sqlite3"
FORMAT_VERSION = "2"


class AlignedRecordingError(RuntimeError):
    """Raised when aligned output cannot be safely written or read."""


class StaleAlignedRecordingError(AlignedRecordingError):
    """Raised when aligned output does not match its raw recording."""


@dataclass(frozen=True)
class SourceRevision:
    """Logical revision of an append-only raw frames table."""

    frame_count: int
    maximum_frame_id: int
    total_points: int


@dataclass(frozen=True)
class FusedAlignedCloud:
    """Saved fused cloud and the metrics used by the web preview."""

    xyz: np.ndarray
    rgb: np.ndarray
    observation_counts: np.ndarray
    source_revision: SourceRevision
    raw_points: int
    cleaned_points: int
    accepted_edges: int
    rejected_edges: int
    charuco_frame_count: int
    charuco_reprojection_max_px: float | None
    cloud_overlap_fraction_3mm: float | None


def aligned_database_path(raw_database_path: Path) -> Path:
    """Return the derived database path beside one raw recording."""
    return Path(raw_database_path).parent / ALIGNED_DATABASE_FILENAME


def source_revision(raw_database_path: Path) -> SourceRevision:
    """Read the logical revision without modifying the raw database."""
    path = Path(raw_database_path)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        frame_count, maximum_frame_id, total_points = connection.execute(
            """
            SELECT COUNT(*), COALESCE(MAX(id), 0),
                   COALESCE(SUM(point_count), 0)
            FROM frames
            """
        ).fetchone()
    finally:
        connection.close()
    return SourceRevision(
        frame_count=int(frame_count),
        maximum_frame_id=int(maximum_frame_id),
        total_points=int(total_points),
    )


def _validated_matrix(value: np.ndarray, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype="<f8")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise AlignedRecordingError(f"{label} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise AlignedRecordingError(f"{label} has an invalid last row")
    rotation = matrix[:3, :3]
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise AlignedRecordingError(f"{label} is not a rigid transformation")
    return np.ascontiguousarray(matrix)


def _validated_xyz_rgb(
    xyz: np.ndarray,
    rgb: np.ndarray,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(xyz, dtype="<f4")
    colors = np.asarray(rgb)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise AlignedRecordingError(f"{label} XYZ must have shape (N, 3)")
    if colors.shape != points.shape:
        raise AlignedRecordingError(f"{label} RGB must match XYZ")
    if not np.isfinite(points).all():
        raise AlignedRecordingError(f"{label} XYZ contains non-finite values")
    if not np.issubdtype(colors.dtype, np.integer):
        raise AlignedRecordingError(f"{label} RGB must contain integers")
    if np.any((colors < 0) | (colors > 255)):
        raise AlignedRecordingError(f"{label} RGB values must be from 0 to 255")
    return (
        np.ascontiguousarray(points, dtype="<f4"),
        np.ascontiguousarray(colors, dtype=np.uint8),
    )


def _prepared_rows(
    result: ProcessingResult,
    revision: SourceRevision,
):
    if revision.frame_count != len(result.aligned_frames):
        raise AlignedRecordingError(
            "Processed frame count does not match the raw recording revision"
        )
    if revision.total_points != result.raw_points:
        raise AlignedRecordingError(
            "Processed raw point count does not match the raw recording revision"
        )

    frame_rows = []
    diagnostic_rows = []
    parent_frame_ids = set()
    maximum_frame_id = 0
    for aligned in result.aligned_frames:
        source = aligned.source
        diagnostics = aligned.diagnostics
        xyz, rgb = _validated_xyz_rgb(
            aligned.xyz,
            aligned.rgb,
            f"Aligned frame {source.id}",
        )
        pose = _validated_matrix(
            aligned.optimized_pose,
            f"Aligned frame {source.id} pose",
        )
        expected_points = (
            diagnostics.cleaned_points - diagnostics.temporal_removed
        )
        if diagnostics.frame_id != source.id or len(xyz) != expected_points:
            raise AlignedRecordingError(
                f"Aligned frame {source.id} diagnostics do not match its points"
            )
        maximum_frame_id = max(maximum_frame_id, source.id)
        parent_frame_ids.add(source.parent_frame_id)
        frame_rows.append(
            (
                source.id,
                source.recorded_perf_counter_ns,
                source.source_sec,
                source.source_nanosec,
                len(xyz),
                source.parent_frame_id,
                f"{source.transformation_name}_pose_graph_optimized",
                pose.tobytes(),
                xyz.tobytes(),
                rgb.tobytes(),
            )
        )
        diagnostic_rows.append(
            (
                source.id,
                diagnostics.input_points,
                diagnostics.voxel_points,
                diagnostics.radius_points,
                diagnostics.cleaned_points,
                diagnostics.temporal_removed,
            )
        )
    if maximum_frame_id != revision.maximum_frame_id:
        raise AlignedRecordingError(
            "Processed maximum frame ID does not match the raw recording revision"
        )
    if len(parent_frame_ids) != 1:
        raise AlignedRecordingError(
            "Aligned frames use inconsistent world parent frame IDs"
        )

    fused_xyz, fused_rgb = _validated_xyz_rgb(
        result.xyz,
        result.rgb,
        "Fused cloud",
    )
    observations = np.asarray(result.observation_counts)
    if (
        observations.shape != (len(fused_xyz),)
        or not np.issubdtype(observations.dtype, np.integer)
        or np.any((observations < 1) | (observations > np.iinfo(np.uint16).max))
    ):
        raise AlignedRecordingError(
            "Fused observation counts must contain one positive uint16 per point"
        )
    observations = np.ascontiguousarray(observations, dtype="<u2")
    fused_row = (
        1,
        len(fused_xyz),
        fused_xyz.tobytes(),
        fused_rgb.tobytes(),
        observations.tobytes(),
    )

    edge_rows = [
        (
            edge.source_frame_id,
            edge.target_frame_id,
            int(edge.accepted),
            edge.overlap_before,
            edge.overlap_after,
            edge.rmse_before_m,
            edge.rmse_after_m,
            edge.correction_m,
            edge.correction_deg,
            edge.reason,
        )
        for edge in result.edges
    ]
    charuco_frame_rows = []
    for diagnostics in result.charuco_frames:
        values = (
            diagnostics.temporal_weight,
            diagnostics.reprojection_rmse_px,
            diagnostics.reprojection_max_px,
            diagnostics.corner_3d_rmse_m,
            diagnostics.corner_3d_max_m,
            diagnostics.cloud_overlap_fraction_3mm,
        )
        if (
            diagnostics.frame_id not in {
                aligned.source.id for aligned in result.aligned_frames
            }
            or not np.isfinite(values).all()
            or diagnostics.temporal_weight <= 0.0
            or diagnostics.matched_prior_count < 0
            or diagnostics.corner_count < 20
            or diagnostics.valid_depth_count < 20
            or diagnostics.invalid_depth_count < 0
            or (
                diagnostics.valid_depth_count
                + diagnostics.invalid_depth_count
                != diagnostics.corner_count
            )
        ):
            raise AlignedRecordingError(
                "ChArUco frame diagnostics are inconsistent"
            )
        charuco_frame_rows.append(
            (
                diagnostics.frame_id,
                diagnostics.temporal_weight,
                diagnostics.matched_prior_count,
                diagnostics.corner_count,
                diagnostics.valid_depth_count,
                diagnostics.invalid_depth_count,
                diagnostics.reprojection_rmse_px,
                diagnostics.reprojection_max_px,
                diagnostics.corner_3d_rmse_m,
                diagnostics.corner_3d_max_m,
                diagnostics.cloud_overlap_fraction_3mm,
            )
        )
    charuco_corner_rows = []
    for diagnostics in result.charuco_corners:
        if (
            not np.isfinite(diagnostics.reprojection_error_px)
            or (
                diagnostics.corner_3d_residual_m is not None
                and not np.isfinite(diagnostics.corner_3d_residual_m)
            )
        ):
            raise AlignedRecordingError(
                "ChArUco corner diagnostics contain non-finite residuals"
            )
        charuco_corner_rows.append(
            (
                diagnostics.frame_id,
                diagnostics.corner_id,
                int(diagnostics.retained),
                int(diagnostics.depth_valid),
                diagnostics.reprojection_error_px,
                diagnostics.corner_3d_residual_m,
            )
        )
    if bool(charuco_frame_rows) != bool(charuco_corner_rows):
        raise AlignedRecordingError(
            "ChArUco frame and corner diagnostics must be saved together"
        )
    if charuco_frame_rows:
        expected_corners = sum(row[3] for row in charuco_frame_rows)
        if (
            len(charuco_frame_rows) != revision.frame_count
            or len(charuco_corner_rows) != expected_corners
            or result.cloud_overlap_fraction_3mm is None
        ):
            raise AlignedRecordingError(
                "ChArUco diagnostics do not cover every processed frame/corner"
            )
    elif result.cloud_overlap_fraction_3mm is not None:
        raise AlignedRecordingError(
            "Cloud corner acceptance metric has no ChArUco diagnostics"
        )
    charuco_max_reprojection = (
        max(row[7] for row in charuco_frame_rows)
        if charuco_frame_rows
        else None
    )
    metadata = {
        "format_version": FORMAT_VERSION,
        "source_frame_count": str(revision.frame_count),
        "source_max_frame_id": str(revision.maximum_frame_id),
        "source_total_points": str(revision.total_points),
        "processed_perf_counter_ns": str(perf_counter_ns()),
        "parent_frame_id": next(iter(parent_frame_ids)),
        "raw_points": str(result.raw_points),
        "cleaned_points": str(result.cleaned_points),
        "fused_points": str(len(fused_xyz)),
        "accepted_edges": str(result.accepted_edges),
        "rejected_edges": str(result.rejected_edges),
        "charuco_frame_count": str(len(charuco_frame_rows)),
        "charuco_reprojection_max_px": (
            "" if charuco_max_reprojection is None
            else str(charuco_max_reprojection)
        ),
        "cloud_overlap_fraction_3mm": (
            "" if result.cloud_overlap_fraction_3mm is None
            else str(result.cloud_overlap_fraction_3mm)
        ),
    }
    return (
        frame_rows,
        fused_row,
        diagnostic_rows,
        edge_rows,
        charuco_frame_rows,
        charuco_corner_rows,
        metadata,
    )


def _create_schema(connection: sqlite3.Connection) -> None:
    statements = (
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS frames (
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
        """,
        """
        CREATE TABLE IF NOT EXISTS frame_diagnostics (
            frame_id INTEGER PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
            input_points INTEGER NOT NULL,
            voxel_points INTEGER NOT NULL,
            radius_points INTEGER NOT NULL,
            cleaned_points INTEGER NOT NULL,
            temporal_removed INTEGER NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS registration_edges (
            source_frame_id INTEGER NOT NULL,
            target_frame_id INTEGER NOT NULL,
            accepted INTEGER NOT NULL,
            overlap_before REAL NOT NULL,
            overlap_after REAL NOT NULL,
            rmse_before_m REAL NOT NULL,
            rmse_after_m REAL NOT NULL,
            correction_m REAL NOT NULL,
            correction_deg REAL NOT NULL,
            reason TEXT NOT NULL,
            PRIMARY KEY (source_frame_id, target_frame_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS charuco_frame_diagnostics (
            frame_id INTEGER PRIMARY KEY REFERENCES frames(id) ON DELETE CASCADE,
            temporal_weight REAL NOT NULL,
            matched_prior_count INTEGER NOT NULL,
            corner_count INTEGER NOT NULL,
            valid_depth_count INTEGER NOT NULL,
            invalid_depth_count INTEGER NOT NULL,
            reprojection_rmse_px REAL NOT NULL,
            reprojection_max_px REAL NOT NULL,
            corner_3d_rmse_m REAL NOT NULL,
            corner_3d_max_m REAL NOT NULL,
            cloud_overlap_fraction_3mm REAL NOT NULL
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS charuco_corner_diagnostics (
            frame_id INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
            corner_id INTEGER NOT NULL,
            retained INTEGER NOT NULL,
            depth_valid INTEGER NOT NULL,
            reprojection_error_px REAL NOT NULL,
            corner_3d_residual_m REAL,
            PRIMARY KEY (frame_id, corner_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS fused_cloud (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            point_count INTEGER NOT NULL,
            xyz BLOB NOT NULL,
            rgb BLOB NOT NULL,
            observation_counts BLOB NOT NULL
        )
        """,
    )
    for statement in statements:
        connection.execute(statement)


def _validate_existing_format(
    connection: sqlite3.Connection,
) -> str | None:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if not tables:
        return None
    if "metadata" not in tables:
        raise AlignedRecordingError(
            "Existing aligned database has an unknown schema"
        )
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = 'format_version'"
    ).fetchone()
    if row is not None and row[0] not in {"1", FORMAT_VERSION}:
        raise AlignedRecordingError(
            f"Aligned database format {row[0]} is unsupported"
        )
    return None if row is None else row[0]


def write_aligned_recording(
    raw_database_path: Path,
    result: ProcessingResult,
) -> Path:
    """Transactionally refresh all derived rows beside the raw recording."""
    raw_path = Path(raw_database_path)
    revision = source_revision(raw_path)
    prepared = _prepared_rows(result, revision)
    aligned_path = aligned_database_path(raw_path)
    existed = aligned_path.exists()
    connection = None
    failed = False
    try:
        connection = sqlite3.connect(aligned_path, isolation_level=None)
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        existing_version = _validate_existing_format(connection)
        connection.execute("BEGIN IMMEDIATE")
        if existing_version == "1":
            for table in (
                "frame_diagnostics",
                "registration_edges",
                "fused_cloud",
                "frames",
                "metadata",
            ):
                connection.execute(f"DROP TABLE {table}")
        _create_schema(connection)
        for table in (
            "charuco_corner_diagnostics",
            "charuco_frame_diagnostics",
            "frame_diagnostics",
            "registration_edges",
            "fused_cloud",
            "frames",
            "metadata",
        ):
            connection.execute(f"DELETE FROM {table}")

        (
            frame_rows,
            fused_row,
            diagnostic_rows,
            edge_rows,
            charuco_frame_rows,
            charuco_corner_rows,
            metadata,
        ) = prepared
        connection.executemany(
            """
            INSERT INTO frames (
                id, recorded_perf_counter_ns, source_sec, source_nanosec,
                point_count, frame_id, transformation_name,
                transformation_matrix, xyz, rgb
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            frame_rows,
        )
        connection.executemany(
            """
            INSERT INTO frame_diagnostics (
                frame_id, input_points, voxel_points, radius_points,
                cleaned_points, temporal_removed
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            diagnostic_rows,
        )
        connection.executemany(
            """
            INSERT INTO registration_edges (
                source_frame_id, target_frame_id, accepted, overlap_before,
                overlap_after, rmse_before_m, rmse_after_m, correction_m,
                correction_deg, reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            edge_rows,
        )
        connection.executemany(
            """
            INSERT INTO charuco_frame_diagnostics (
                frame_id, temporal_weight, matched_prior_count,
                corner_count, valid_depth_count, invalid_depth_count,
                reprojection_rmse_px, reprojection_max_px,
                corner_3d_rmse_m, corner_3d_max_m,
                cloud_overlap_fraction_3mm
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            charuco_frame_rows,
        )
        connection.executemany(
            """
            INSERT INTO charuco_corner_diagnostics (
                frame_id, corner_id, retained, depth_valid,
                reprojection_error_px, corner_3d_residual_m
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            charuco_corner_rows,
        )
        connection.execute(
            """
            INSERT INTO fused_cloud (
                id, point_count, xyz, rgb, observation_counts
            ) VALUES (?, ?, ?, ?, ?)
            """,
            fused_row,
        )
        connection.executemany(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            metadata.items(),
        )
        connection.execute("COMMIT")
    except Exception:
        failed = True
        if connection is not None and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        if connection is not None:
            connection.close()
        if failed and not existed:
            for suffix in ("", "-journal", "-wal", "-shm"):
                candidate = Path(f"{aligned_path}{suffix}")
                if candidate.exists():
                    candidate.unlink()
    return aligned_path


def generate_aligned_recording(raw_database_path: Path) -> Path:
    """Run the full pipeline and transactionally persist its derived result."""
    raw_path = Path(raw_database_path)
    return write_aligned_recording(raw_path, process_recording(raw_path))


def _metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def _revision_from_metadata(metadata: dict[str, str]) -> SourceRevision:
    try:
        if metadata["format_version"] != FORMAT_VERSION:
            raise AlignedRecordingError(
                f"Aligned database format {metadata['format_version']} "
                "is unsupported"
            )
        return SourceRevision(
            frame_count=int(metadata["source_frame_count"]),
            maximum_frame_id=int(metadata["source_max_frame_id"]),
            total_points=int(metadata["source_total_points"]),
        )
    except KeyError as error:
        raise AlignedRecordingError(
            f"Aligned database metadata is missing {error.args[0]}"
        ) from error
    except ValueError as error:
        raise AlignedRecordingError(
            "Aligned database revision metadata is invalid"
        ) from error


def read_fused_cloud(
    aligned_path: Path,
    expected_revision: SourceRevision | None = None,
) -> FusedAlignedCloud:
    """Read and validate the saved fused cloud and preview metrics."""
    path = Path(aligned_path)
    uri = f"{path.resolve().as_uri()}?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError as error:
        raise AlignedRecordingError(
            f"Aligned recording does not exist: {path}"
        ) from error
    try:
        metadata = _metadata(connection)
        revision = _revision_from_metadata(metadata)
        if expected_revision is not None and revision != expected_revision:
            raise StaleAlignedRecordingError(
                "Aligned recording does not match the current raw recording"
            )
        row = connection.execute(
            """
            SELECT point_count, xyz, rgb, observation_counts
            FROM fused_cloud
            WHERE id = 1
            """
        ).fetchone()
    except sqlite3.Error as error:
        raise AlignedRecordingError(
            f"Cannot read aligned recording: {error}"
        ) from error
    finally:
        connection.close()
    if row is None:
        raise AlignedRecordingError("Aligned recording has no fused cloud")

    point_count, xyz_blob, rgb_blob, observations_blob = row
    expected_xyz_bytes = point_count * 3 * np.dtype("<f4").itemsize
    expected_rgb_bytes = point_count * 3
    expected_observation_bytes = point_count * np.dtype("<u2").itemsize
    if (
        point_count < 1
        or len(xyz_blob) != expected_xyz_bytes
        or len(rgb_blob) != expected_rgb_bytes
        or len(observations_blob) != expected_observation_bytes
    ):
        raise AlignedRecordingError(
            "Aligned recording contains invalid fused point data"
        )
    try:
        raw_points = int(metadata["raw_points"])
        cleaned_points = int(metadata["cleaned_points"])
        fused_points = int(metadata["fused_points"])
        accepted_edges = int(metadata["accepted_edges"])
        rejected_edges = int(metadata["rejected_edges"])
        charuco_frame_count = int(metadata["charuco_frame_count"])
        charuco_reprojection_max_px = (
            None
            if not metadata["charuco_reprojection_max_px"]
            else float(metadata["charuco_reprojection_max_px"])
        )
        cloud_overlap_fraction_3mm = (
            None
            if not metadata["cloud_overlap_fraction_3mm"]
            else float(metadata["cloud_overlap_fraction_3mm"])
        )
    except (KeyError, ValueError) as error:
        raise AlignedRecordingError(
            "Aligned recording contains invalid processing metrics"
        ) from error

    xyz = np.frombuffer(xyz_blob, dtype="<f4").reshape(-1, 3).copy()
    observations = np.frombuffer(
        observations_blob,
        dtype="<u2",
    ).copy()
    if not np.isfinite(xyz).all():
        raise AlignedRecordingError(
            "Aligned recording contains non-finite fused XYZ values"
        )
    if (
        raw_points != revision.total_points
        or cleaned_points < point_count
        or fused_points != point_count
        or accepted_edges < 0
        or rejected_edges < 0
        or charuco_frame_count not in {0, revision.frame_count}
        or (
            charuco_frame_count == 0
            and (
                charuco_reprojection_max_px is not None
                or cloud_overlap_fraction_3mm is not None
            )
        )
        or (
            charuco_frame_count > 0
            and (
                charuco_reprojection_max_px is None
                or not np.isfinite(charuco_reprojection_max_px)
                or charuco_reprojection_max_px > 1.0
                or cloud_overlap_fraction_3mm is None
                or not np.isfinite(cloud_overlap_fraction_3mm)
                or cloud_overlap_fraction_3mm < 0.99
            )
        )
        or np.any(observations < 1)
    ):
        raise AlignedRecordingError(
            "Aligned recording contains inconsistent processing metrics"
        )
    return FusedAlignedCloud(
        xyz=xyz,
        rgb=np.frombuffer(rgb_blob, dtype=np.uint8).reshape(-1, 3).copy(),
        observation_counts=observations,
        source_revision=revision,
        raw_points=raw_points,
        cleaned_points=cleaned_points,
        accepted_edges=accepted_edges,
        rejected_edges=rejected_edges,
        charuco_frame_count=charuco_frame_count,
        charuco_reprojection_max_px=charuco_reprojection_max_px,
        cloud_overlap_fraction_3mm=cloud_overlap_fraction_3mm,
    )
