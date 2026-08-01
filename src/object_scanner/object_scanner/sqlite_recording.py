"""Append transformed colored point-cloud frames to SQLite."""

from pathlib import Path
import sqlite3

import numpy as np


class SqliteRecording:
    """One appendable point-cloud recording session."""

    def __init__(self, path: Path) -> None:
        if path.exists():
            raise FileExistsError(f"Recording already exists: {path}")

        self.path = path
        self._connection = sqlite3.connect(path)
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
                VALUES ('format_version', '1');

                CREATE TABLE frames (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    recorded_perf_counter_ns INTEGER NOT NULL,
                    source_sec INTEGER NOT NULL,
                    source_nanosec INTEGER NOT NULL,
                    frame_id TEXT NOT NULL,
                    point_count INTEGER NOT NULL,
                    xyz BLOB NOT NULL,
                    rgb BLOB NOT NULL
                );
                """
            )
            self._connection.commit()
        except Exception:
            self._connection.close()
            raise

    def append_frame(
        self,
        *,
        recorded_perf_counter_ns: int,
        source_sec: int,
        source_nanosec: int,
        frame_id: str,
        xyz: np.ndarray,
        rgb: np.ndarray,
    ) -> None:
        """Append and commit one world-frame XYZ/RGB array pair."""
        points = np.asarray(xyz, dtype="<f4")
        colors = np.asarray(rgb)
        if points.ndim != 2 or points.shape[1:] != (3,):
            raise ValueError("xyz must have shape (N, 3)")
        if colors.shape != points.shape:
            raise ValueError("rgb must have the same (N, 3) shape as xyz")
        if not np.isfinite(points).all():
            raise ValueError("xyz must contain only finite values")
        if np.any((colors < 0) | (colors > 255)):
            raise ValueError("rgb values must be from 0 to 255")

        points = np.ascontiguousarray(points)
        colors = np.ascontiguousarray(colors, dtype=np.uint8)
        self._connection.execute(
            """
            INSERT INTO frames (
                recorded_perf_counter_ns,
                source_sec,
                source_nanosec,
                frame_id,
                point_count,
                xyz,
                rgb
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(recorded_perf_counter_ns),
                int(source_sec),
                int(source_nanosec),
                frame_id,
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
        """Commit and close this recording session."""
        self._connection.commit()
        self._connection.close()
