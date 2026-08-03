"""Append transformed colored point-cloud frames to SQLite."""

import json
from pathlib import Path
import re
import sqlite3

import numpy as np


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
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.executescript(
                """
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                INSERT INTO metadata (key, value)
                VALUES ('format_version', '2');

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
    ) -> None:
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

        points = np.ascontiguousarray(points)
        colors = np.ascontiguousarray(colors, dtype=np.uint8)
        matrix = np.ascontiguousarray(matrix)
        self._connection.execute(
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
        self._connection.commit()

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
