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
                VALUES ('format_version', '4');

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

                CREATE TABLE stream_timestamps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stream TEXT NOT NULL,
                    source_sec INTEGER NOT NULL,
                    source_nanosec INTEGER NOT NULL,
                    arrival_perf_counter_ns INTEGER NOT NULL
                );

                CREATE TABLE publish_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    received_perf_counter_ns INTEGER NOT NULL,
                    transformation_name TEXT NOT NULL,
                    depth_sec INTEGER NOT NULL,
                    depth_nanosec INTEGER NOT NULL,
                    pointcloud_received INTEGER NOT NULL,
                    color_sec INTEGER,
                    color_nanosec INTEGER,
                    color_delta_ns INTEGER,
                    captured_frame_id INTEGER
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

        points = np.ascontiguousarray(points)
        colors = np.ascontiguousarray(colors, dtype=np.uint8)
        matrix = np.ascontiguousarray(matrix)
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
        self._connection.commit()
        return int(cursor.lastrowid)

    def append_stream_timestamp(
        self,
        *,
        stream: str,
        source_sec: int,
        source_nanosec: int,
        arrival_perf_counter_ns: int,
    ) -> None:
        """Append one independently received ROS stream timestamp."""
        if stream not in {"pointcloud", "transformation", "color_image"}:
            raise ValueError(f"Unsupported diagnostic stream: {stream}")
        self._connection.execute(
            """
            INSERT INTO stream_timestamps (
                stream,
                source_sec,
                source_nanosec,
                arrival_perf_counter_ns
            ) VALUES (?, ?, ?, ?)
            """,
            (
                stream,
                int(source_sec),
                int(source_nanosec),
                int(arrival_perf_counter_ns),
            ),
        )

    def append_publish_attempt(
        self,
        *,
        received_perf_counter_ns: int,
        transformation_name: str,
        depth_sec: int,
        depth_nanosec: int,
        pointcloud_received: bool,
        color_stamp: tuple[int, int] | None,
    ) -> int:
        """Append one transformation publish attempt."""
        color_sec = color_stamp[0] if color_stamp is not None else None
        color_nanosec = color_stamp[1] if color_stamp is not None else None
        color_delta_ns = (
            self._timestamp_delta_ns(
                color_stamp,
                (depth_sec, depth_nanosec),
            )
            if color_stamp is not None
            else None
        )
        cursor = self._connection.execute(
            """
            INSERT INTO publish_attempts (
                received_perf_counter_ns,
                transformation_name,
                depth_sec,
                depth_nanosec,
                pointcloud_received,
                color_sec,
                color_nanosec,
                color_delta_ns
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(received_perf_counter_ns),
                transformation_name,
                int(depth_sec),
                int(depth_nanosec),
                int(pointcloud_received),
                color_sec,
                color_nanosec,
                color_delta_ns,
            ),
        )
        return int(cursor.lastrowid)

    def update_publish_attempt_color(
        self,
        attempt_id: int,
        color_stamp: tuple[int, int],
    ) -> None:
        """Store the closest observed color timestamp for one attempt."""
        depth_sec, depth_nanosec = self._connection.execute(
            """
            SELECT depth_sec, depth_nanosec
            FROM publish_attempts
            WHERE id = ?
            """,
            (int(attempt_id),),
        ).fetchone()
        self._connection.execute(
            """
            UPDATE publish_attempts
            SET color_sec = ?, color_nanosec = ?, color_delta_ns = ?
            WHERE id = ?
            """,
            (
                int(color_stamp[0]),
                int(color_stamp[1]),
                self._timestamp_delta_ns(
                    color_stamp,
                    (depth_sec, depth_nanosec),
                ),
                int(attempt_id),
            ),
        )

    def mark_publish_attempt_pointcloud_received(self, attempt_id: int) -> None:
        """Mark that this scanner received the web-selected depth cloud."""
        self._connection.execute(
            """
            UPDATE publish_attempts
            SET pointcloud_received = 1
            WHERE id = ?
            """,
            (int(attempt_id),),
        )

    def mark_publish_attempt_captured(
        self,
        attempt_id: int,
        frame_id: int,
        color_stamp: tuple[int, int],
    ) -> None:
        """Link a publish attempt to its recorded synchronized frame."""
        self.update_publish_attempt_color(attempt_id, color_stamp)
        self._connection.execute(
            """
            UPDATE publish_attempts
            SET captured_frame_id = ?
            WHERE id = ?
            """,
            (int(frame_id), int(attempt_id)),
        )
        self._connection.commit()

    @staticmethod
    def _timestamp_delta_ns(
        left: tuple[int, int],
        right: tuple[int, int],
    ) -> int:
        return (
            (int(left[0]) - int(right[0])) * 1_000_000_000
            + int(left[1])
            - int(right[1])
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
