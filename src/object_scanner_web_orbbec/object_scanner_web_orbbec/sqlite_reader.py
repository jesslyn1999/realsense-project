"""Read and encode point-cloud frames from scanner SQLite sessions."""

from math import ceil
from pathlib import Path
import sqlite3
import struct

import numpy as np


PAYLOAD_HEADER = struct.Struct("<4sII")
PAYLOAD_MAGIC = b"PCD1"


def list_frames(database_path: Path) -> list[dict]:
    """Return replay frame metadata in source-timestamp order."""
    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            """
            SELECT id, source_sec, source_nanosec, point_count, frame_id,
                   transformation_name, transformation_matrix
            FROM frames
            ORDER BY source_sec, source_nanosec, id
            """
        ).fetchall()
    finally:
        connection.close()
    frames = []
    for (
        frame_id,
        source_sec,
        source_nanosec,
        point_count,
        parent_frame_id,
        transformation_name,
        matrix_blob,
    ) in rows:
        matrix = np.frombuffer(matrix_blob, dtype="<f8")
        if matrix.size != 16:
            raise ValueError(
                f"Frame {frame_id} contains invalid transformation metadata"
            )
        frames.append(
            {
                "id": frame_id,
                "source_sec": source_sec,
                "source_nanosec": source_nanosec,
                "point_count": point_count,
                "parent_frame_id": parent_frame_id,
                "transformation_name": transformation_name,
                "matrix": matrix.reshape(4, 4).tolist(),
            }
        )
    return frames


def read_sampled_frame(
    database_path: Path,
    frame_id: int,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return one uniformly sampled replay frame."""
    if frame_id < 1:
        raise ValueError("frame_id must be at least one")
    if max_points < 1:
        raise ValueError("max_points must be at least one")

    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        row = connection.execute(
            "SELECT point_count, xyz, rgb FROM frames WHERE id = ?",
            (frame_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        raise LookupError(f"Frame {frame_id} does not exist")

    point_count, xyz_blob, rgb_blob = row
    expected_xyz_bytes = point_count * 3 * np.dtype("<f4").itemsize
    expected_rgb_bytes = point_count * 3
    if (
        len(xyz_blob) != expected_xyz_bytes
        or len(rgb_blob) != expected_rgb_bytes
    ):
        raise ValueError(f"Frame {frame_id} contains invalid point data")

    stride = max(1, ceil(point_count / max_points))
    xyz = np.frombuffer(xyz_blob, dtype="<f4").reshape(-1, 3)[::stride]
    rgb = np.frombuffer(rgb_blob, dtype=np.uint8).reshape(-1, 3)[::stride]
    return (
        np.ascontiguousarray(xyz, dtype="<f4"),
        np.ascontiguousarray(rgb, dtype=np.uint8),
        point_count,
    )


def read_sampled_points(
    database_path: Path,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return uniformly sampled XYZ/RGB arrays and the uncapped point count."""
    if max_points < 1:
        raise ValueError("max_points must be at least one")

    uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        total_points = int(
            connection.execute(
                "SELECT COALESCE(SUM(point_count), 0) FROM frames"
            ).fetchone()[0]
        )
        stride = max(1, ceil(total_points / max_points))
        xyz_parts = []
        rgb_parts = []
        point_offset = 0

        for frame_id, point_count, xyz_blob, rgb_blob in connection.execute(
            "SELECT id, point_count, xyz, rgb FROM frames ORDER BY id"
        ):
            expected_xyz_bytes = point_count * 3 * np.dtype("<f4").itemsize
            expected_rgb_bytes = point_count * 3
            xyz_size_is_valid = len(xyz_blob) == expected_xyz_bytes
            rgb_size_is_valid = len(rgb_blob) == expected_rgb_bytes
            if not xyz_size_is_valid or not rgb_size_is_valid:
                raise ValueError(f"Frame {frame_id} contains invalid point data")

            start = (-point_offset) % stride
            frame_xyz = np.frombuffer(xyz_blob, dtype="<f4").reshape(-1, 3)
            frame_rgb = np.frombuffer(rgb_blob, dtype=np.uint8).reshape(-1, 3)
            xyz_parts.append(frame_xyz[start::stride])
            rgb_parts.append(frame_rgb[start::stride])
            point_offset += point_count
    finally:
        connection.close()

    if not xyz_parts:
        return (
            np.empty((0, 3), dtype="<f4"),
            np.empty((0, 3), dtype=np.uint8),
            total_points,
        )
    return (
        np.ascontiguousarray(np.concatenate(xyz_parts), dtype="<f4"),
        np.ascontiguousarray(np.concatenate(rgb_parts), dtype=np.uint8),
        total_points,
    )


def build_point_payload(database_path: Path, max_points: int) -> bytes:
    """Encode a sampled cloud for direct Three.js typed-array loading."""
    xyz, rgb, total_points = read_sampled_points(database_path, max_points)
    return build_array_payload(xyz, rgb, total_points)


def build_array_payload(
    xyz: np.ndarray,
    rgb: np.ndarray,
    total_points: int | None = None,
) -> bytes:
    """Encode validated XYZ/RGB arrays using the existing PCD1 contract."""
    points = np.asarray(xyz, dtype="<f4")
    colors = np.asarray(rgb)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("xyz must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("rgb must have the same (N, 3) shape as xyz")
    if not np.isfinite(points).all():
        raise ValueError("xyz must contain only finite values")
    if not np.issubdtype(colors.dtype, np.integer):
        raise ValueError("rgb must contain integer values")
    if np.any((colors < 0) | (colors > 255)):
        raise ValueError("rgb values must be from 0 to 255")
    if total_points is None:
        total_points = len(points)
    if total_points < len(points):
        raise ValueError("total_points must not be smaller than displayed points")

    points = np.ascontiguousarray(points, dtype="<f4")
    colors = np.ascontiguousarray(colors, dtype=np.uint8)
    header = PAYLOAD_HEADER.pack(PAYLOAD_MAGIC, len(points), total_points)
    return header + points.tobytes() + colors.tobytes()


def build_frame_payload(
    database_path: Path,
    frame_id: int,
    max_points: int,
) -> bytes:
    """Encode one sampled replay frame for Three.js."""
    xyz, rgb, total_points = read_sampled_frame(
        database_path,
        frame_id,
        max_points,
    )
    return build_array_payload(xyz, rgb, total_points)
