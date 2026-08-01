"""Read and encode point-cloud frames from scanner SQLite sessions."""

from math import ceil
from pathlib import Path
import sqlite3
import struct

import numpy as np


PAYLOAD_HEADER = struct.Struct("<4sII")
PAYLOAD_MAGIC = b"PCD1"


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
    header = PAYLOAD_HEADER.pack(PAYLOAD_MAGIC, len(xyz), total_points)
    return header + xyz.tobytes() + rgb.tobytes()
