"""Append transformed colored point-cloud frames to SQLite."""

import json
from pathlib import Path
import re
import sqlite3

import numpy as np
from object_scanner_processing.charuco_observations import (
    CharucoFrameObservation,
    validated_charuco_observation,
)


SESSION_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def validated_session_name(value) -> str:
    """Return a safe bare SQLite session name."""
    if (
        not isinstance(value, str)
        or SESSION_NAME_PATTERN.fullmatch(value) is None
    ):
        raise ValueError(
            "session_name must use only letters, numbers, '_' or '-'"
        )
    return value


class SqliteRecording:
    """One appendable point-cloud recording session."""

    def __init__(self, session_directory: Path) -> None:
        if session_directory.exists():
            raise FileExistsError(
                f"Recording session already exists: {session_directory}"
            )

        self.session_directory = session_directory
        self.path = session_directory / "recording.sqlite3"
        self.metadata_path = session_directory / "metadata.json"
        session_directory.mkdir(parents=True)
        self._connection = sqlite3.connect(self.path)
        try:
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT INTO metadata (key, value)
                VALUES ('format_version', '3');

                CREATE TABLE frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_perf_counter_ns INTEGER NOT NULL,
                    source_sec INTEGER NOT NULL,
                    source_nanosec INTEGER NOT NULL,
                    frame_id TEXT NOT NULL,
                    transformation_name TEXT NOT NULL,
                    transformation_matrix BLOB NOT NULL,
                    point_count INTEGER NOT NULL,
                    xyz BLOB NOT NULL,
                    rgb BLOB NOT NULL
                );

                CREATE TABLE charuco_observations (
                    frame_id INTEGER PRIMARY KEY REFERENCES frames(id),
                    corner_count INTEGER NOT NULL,
                    valid_depth_corner_count INTEGER NOT NULL,
                    initial_reprojection_rmse_px REAL NOT NULL,
                    camera_matrix BLOB NOT NULL,
                    distortion_count INTEGER NOT NULL,
                    distortion BLOB NOT NULL,
                    color_from_child BLOB NOT NULL
                );

                CREATE TABLE charuco_corners (
                    frame_id INTEGER NOT NULL REFERENCES frames(id),
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
            self._connection.commit()
        except Exception:
            self._connection.close()
            self.path.unlink(missing_ok=True)
            session_directory.rmdir()
            raise

    def append_frame(
        self,
        *,
        recorded_perf_counter_ns: int,
        source_sec: int,
        source_nanosec: int,
        frame_id: str,
        transformation_name: str,
        transformation_matrix: np.ndarray,
        xyz: np.ndarray,
        rgb: np.ndarray,
        charuco_observation: CharucoFrameObservation | None = None,
    ) -> int:
        """Append and commit one world-frame XYZ/RGB array pair."""
        points = np.asarray(xyz, dtype="<f4")
        colors = np.asarray(rgb)
        matrix = np.asarray(transformation_matrix, dtype="<f8")
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("xyz must have shape (N, 3)")
        if colors.shape != points.shape:
            raise ValueError("rgb must have the same (N, 3) shape as xyz")
        if not np.isfinite(points).all():
            raise ValueError("xyz must contain only finite values")
        if np.any((colors < 0) | (colors > 255)):
            raise ValueError("rgb values must be from 0 to 255")
        if not transformation_name:
            raise ValueError("transformation_name must not be empty")
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(
                "transformation_matrix must be a finite 4x4 matrix"
            )
        observation = self._validated_observation(
            transformation_name,
            charuco_observation,
        )

        points = np.ascontiguousarray(points)
        colors = np.ascontiguousarray(colors, dtype=np.uint8)
        matrix = np.ascontiguousarray(matrix)
        try:
            cursor = self._connection.execute(
                """
                INSERT INTO frames (
                    recorded_perf_counter_ns,
                    source_sec,
                    source_nanosec,
                    frame_id,
                    transformation_name,
                    transformation_matrix,
                    point_count,
                    xyz,
                    rgb
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(recorded_perf_counter_ns),
                    int(source_sec),
                    int(source_nanosec),
                    frame_id,
                    transformation_name,
                    sqlite3.Binary(matrix.tobytes()),
                    len(points),
                    sqlite3.Binary(points.tobytes()),
                    sqlite3.Binary(colors.tobytes()),
                ),
            )
            sqlite_frame_id = int(cursor.lastrowid)
            if observation is not None:
                self._insert_charuco_observation(
                    sqlite_frame_id,
                    observation,
                )
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise
        return sqlite_frame_id

    @staticmethod
    def _validated_observation(
        transformation_name: str,
        observation: CharucoFrameObservation | None,
    ) -> CharucoFrameObservation | None:
        if transformation_name == "charuco" and observation is None:
            raise ValueError("ChArUco frames require corner observations")
        if transformation_name != "charuco" and observation is not None:
            raise ValueError(
                "Corner observations are valid only for ChArUco frames"
            )
        if observation is None:
            return None
        return validated_charuco_observation(
            **observation.__dict__,
        )

    def _insert_charuco_observation(
        self,
        frame_id: int,
        observation: CharucoFrameObservation,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO charuco_observations (
                frame_id, corner_count, valid_depth_corner_count,
                initial_reprojection_rmse_px, camera_matrix,
                distortion_count, distortion, color_from_child
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                frame_id,
                len(observation.corner_ids),
                observation.valid_depth_corner_count,
                observation.initial_reprojection_rmse_px,
                sqlite3.Binary(
                    np.ascontiguousarray(
                        observation.camera_matrix,
                        dtype="<f8",
                    ).tobytes()
                ),
                len(observation.distortion),
                sqlite3.Binary(
                    np.ascontiguousarray(
                        observation.distortion,
                        dtype="<f8",
                    ).tobytes()
                ),
                sqlite3.Binary(
                    np.ascontiguousarray(
                        observation.color_from_child,
                        dtype="<f8",
                    ).tobytes()
                ),
            ),
        )
        rows = []
        for index, corner_id in enumerate(observation.corner_ids):
            valid = bool(observation.depth_valid[index])
            child_point = observation.child_points[index]
            mad = observation.depth_mad_m[index]
            rows.append(
                (
                    frame_id,
                    int(corner_id),
                    float(observation.image_points[index, 0]),
                    float(observation.image_points[index, 1]),
                    int(valid),
                    float(child_point[0]) if valid else None,
                    float(child_point[1]) if valid else None,
                    float(child_point[2]) if valid else None,
                    int(observation.depth_valid_pixel_counts[index]),
                    int(observation.depth_inlier_pixel_counts[index]),
                    float(mad) if np.isfinite(mad) else None,
                    observation.depth_invalid_reasons[index],
                    float(
                        observation.initial_reprojection_errors_px[index]
                    ),
                )
            )
        self._connection.executemany(
            """
            INSERT INTO charuco_corners (
                frame_id, corner_id, image_u, image_v, depth_valid,
                child_x, child_y, child_z, valid_pixel_count,
                inlier_pixel_count, depth_mad_m, invalid_reason,
                initial_reprojection_error_px
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def commit(self) -> None:
        """Make all appended frames visible to other SQLite connections."""
        self._connection.commit()

    def close(self) -> None:
        """Finalize SQLite and atomically write per-frame metadata."""
        self._connection.commit()
        metadata = self._frame_metadata()
        temporary_metadata_path = self.metadata_path.with_suffix(".json.tmp")
        try:
            with temporary_metadata_path.open("w", encoding="utf-8") as file:
                json.dump(
                    {
                        "format_version": 1,
                        "session_name": self.session_directory.name,
                        "database_file": self.path.name,
                        "frames": metadata,
                    },
                    file,
                    indent=2,
                )
                file.write("\n")

            busy, _, _ = self._connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if busy:
                raise sqlite3.OperationalError(
                    "Cannot finalize recording while another SQLite reader "
                    "is active"
                )
            journal_mode = self._connection.execute(
                "PRAGMA journal_mode=DELETE"
            ).fetchone()[0]
            if journal_mode.lower() != "delete":
                raise sqlite3.OperationalError(
                    "Cannot finalize recording in DELETE mode: "
                    f"{journal_mode}"
                )
            temporary_metadata_path.replace(self.metadata_path)
        except Exception:
            temporary_metadata_path.unlink(missing_ok=True)
            raise
        self._connection.close()

    def _frame_metadata(self) -> list[dict]:
        frames = []
        rows = self._connection.execute(
            """
            SELECT id, source_sec, source_nanosec, frame_id,
                   transformation_name, transformation_matrix
            FROM frames
            ORDER BY id
            """
        )
        for (
            frame_id,
            source_sec,
            source_nanosec,
            parent_frame_id,
            transformation_name,
            matrix_blob,
        ) in rows:
            matrix = np.frombuffer(matrix_blob, dtype="<f8")
            if matrix.size != 16:
                raise sqlite3.DatabaseError(
                    f"Frame {frame_id} contains invalid transformation metadata"
                )
            frames.append(
                {
                    "sqlite_frame_id": frame_id,
                    "source_sec": source_sec,
                    "source_nanosec": source_nanosec,
                    "transformation_name": transformation_name,
                    "parent_frame_id": parent_frame_id,
                    "matrix": matrix.reshape(4, 4).tolist(),
                }
            )
        return frames
