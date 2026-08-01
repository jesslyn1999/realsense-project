"""Color filtering and rigid transformation for colored point clouds."""

import cv2
from geometry_msgs.msg import TransformStamped
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField


def _pointcloud_records(message: PointCloud2) -> np.ndarray:
    fields = {field.name: field for field in message.fields}
    required = ("x", "y", "z", "rgb")
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Point cloud is missing fields: {', '.join(missing)}")

    for name in required:
        field = fields[name]
        if field.datatype != PointField.FLOAT32 or field.count != 1:
            raise ValueError(f"Point cloud field '{name}' must be FLOAT32")

    byte_order = ">" if message.is_bigendian else "<"
    dtype = np.dtype(
        {
            "names": list(required),
            "formats": [byte_order + "f4"] * len(required),
            "offsets": [fields[name].offset for name in required],
            "itemsize": message.point_step,
        }
    )
    point_count = message.width * message.height
    packed_row_size = message.width * message.point_step
    if message.row_step == packed_row_size:
        return np.frombuffer(message.data, dtype=dtype, count=point_count)

    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(
        message.height,
        message.row_step,
    )
    packed = rows[:, :packed_row_size].copy()
    return np.frombuffer(packed, dtype=dtype, count=point_count)


def filter_colored_points(
    message: PointCloud2,
    target_rgb: tuple[int, int, int],
    lab_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite XYZ/RGB points within a CIELAB distance of target_rgb."""
    target = np.asarray(target_rgb)
    if (
        target.shape != (3,)
        or not np.issubdtype(target.dtype, np.number)
        or not np.isfinite(target).all()
        or np.any((target < 0) | (target > 255))
    ):
        raise ValueError("target_rgb must contain three values from 0 to 255")
    if not np.isfinite(lab_threshold) or lab_threshold < 0:
        raise ValueError("lab_threshold must be finite and non-negative")
    target = target.astype(np.uint8)

    records = _pointcloud_records(message)
    xyz = np.column_stack(
        (records["x"], records["y"], records["z"])
    ).astype(np.float32, copy=False)

    byte_order = ">" if message.is_bigendian else "<"
    packed_rgb = (
        np.ascontiguousarray(records["rgb"])
        .view(byte_order + "u4")
        .astype(np.uint32, copy=False)
    )
    rgb = np.column_stack(
        (
            (packed_rgb >> 16) & 0xFF,
            (packed_rgb >> 8) & 0xFF,
            packed_rgb & 0xFF,
        )
    ).astype(np.uint8)

    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0)
    xyz = xyz[valid]
    rgb = rgb[valid]
    if not len(xyz):
        return xyz, rgb

    lab = cv2.cvtColor(
        (rgb.astype(np.float32) / 255.0).reshape(-1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(-1, 3)
    target_lab = cv2.cvtColor(
        (target.astype(np.float32) / 255.0).reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(3)
    keep = np.linalg.norm(lab - target_lab, axis=1) <= lab_threshold
    return xyz[keep], rgb[keep]


def transform_to_matrix(message: TransformStamped) -> np.ndarray:
    """Convert a TransformStamped translation and quaternion to a 4x4 matrix."""
    translation = np.array(
        [
            message.transform.translation.x,
            message.transform.translation.y,
            message.transform.translation.z,
        ],
        dtype=np.float64,
    )
    quaternion = np.array(
        [
            message.transform.rotation.x,
            message.transform.rotation.y,
            message.transform.rotation.z,
            message.transform.rotation.w,
        ],
        dtype=np.float64,
    )
    if not np.isfinite(translation).all() or not np.isfinite(quaternion).all():
        raise ValueError("Transform must contain only finite values")

    norm = np.linalg.norm(quaternion)
    if norm == 0:
        raise ValueError("Transform quaternion must have non-zero length")
    x, y, z, w = quaternion / norm

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    matrix[:3, 3] = translation
    return matrix


def transform_filtered_cloud(
    transform_matrix: np.ndarray,
    xyz: np.ndarray,
    rgb: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform filtered XYZ into world coordinates and preserve its RGB."""
    matrix = np.asarray(transform_matrix, dtype=np.float64)
    points = np.asarray(xyz)
    colors = np.asarray(rgb)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform_matrix must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("transform_matrix must be a rigid homogeneous matrix")
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("xyz must have shape (N, 3)")
    if colors.shape != points.shape:
        raise ValueError("rgb must have the same (N, 3) shape as xyz")
    if not np.isfinite(points).all():
        raise ValueError("xyz must contain only finite values")

    world_xyz = (
        points.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
    ).astype(np.float32)
    return world_xyz, colors.astype(np.uint8, copy=True)
