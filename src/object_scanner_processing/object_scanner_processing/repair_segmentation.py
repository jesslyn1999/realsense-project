"""CAD-guided repair-region segmentation for the demo5 scan."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d

from .aligned_recording import read_fused_cloud


REPAIR_CENTER_WORLD_M = np.array([0.125, -0.2342, 0.0802])
REPAIR_DISTANCE_M = 0.005

# The print is 5.7x the STL's millimetre geometry. This rotation is the
# demo5 broken-part registration; the translation is set from the repair centre.
REPAIR_LINEAR_WORLD_FROM_STL = np.array(
    [
        [0.0045241111, 0.0033756128, 0.0007922481],
        [-0.0029776391, 0.0031150922, 0.0037309337],
        [0.0017765372, -0.0033751206, 0.0042358561],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class RepairSegmentation:
    """Repair points and the native-STL-to-world transform used to select them."""

    xyz: np.ndarray
    transform: np.ndarray
    scale_m_per_stl_unit: float


def _validated_similarity_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("transform must have the homogeneous last row [0, 0, 0, 1]")

    linear = matrix[:3, :3]
    scales = np.linalg.norm(linear, axis=0)
    scale = float(scales.mean())
    if scale <= 0.0 or not np.allclose(scales, scale, rtol=1e-3, atol=1e-9):
        raise ValueError("transform must use one positive uniform scale")
    rotation = linear / scale
    if (
        not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
        or np.linalg.det(rotation) <= 0.0
    ):
        raise ValueError("transform must contain a proper rotation without shear")
    return matrix.copy()


def _initial_transform(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        raise ValueError("repair STL has no vertices")
    transform = np.eye(4)
    transform[:3, :3] = REPAIR_LINEAR_WORLD_FROM_STL
    transform[:3, 3] = (
        REPAIR_CENTER_WORLD_M
        - REPAIR_LINEAR_WORLD_FROM_STL @ vertices.mean(axis=0)
    )
    return _validated_similarity_transform(transform)


def _load_repair_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise ValueError(f"Cannot read a triangle mesh from repair STL: {path}")
    return mesh


def segment_repair(
    aligned_recording_path: Path,
    repair_stl_path: Path,
    transform: np.ndarray | None = None,
) -> RepairSegmentation:
    """Select fused scan points supported by the positioned repair surface."""
    mesh = _load_repair_mesh(Path(repair_stl_path))
    matrix = (
        _initial_transform(mesh)
        if transform is None
        else _validated_similarity_transform(transform)
    )
    mesh.transform(matrix)

    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    cloud = read_fused_cloud(Path(aligned_recording_path))
    query = o3d.core.Tensor(
        np.asarray(cloud.xyz, dtype=np.float32),
        dtype=o3d.core.Dtype.Float32,
    )
    distances = scene.compute_distance(query).numpy()
    selected = np.asarray(cloud.xyz)[distances <= REPAIR_DISTANCE_M]
    scale = float(np.linalg.norm(matrix[:3, 0]))
    return RepairSegmentation(
        xyz=np.ascontiguousarray(selected, dtype=np.float32),
        transform=matrix,
        scale_m_per_stl_unit=scale,
    )
