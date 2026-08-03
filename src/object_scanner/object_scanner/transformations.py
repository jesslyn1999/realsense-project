"""Load named rigid transforms and convert them to ROS messages."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from object_scanner_interfaces.msg import NamedTransform


@dataclass(frozen=True)
class TransformationMatrix:
    """One named camera-to-world homogeneous transformation."""

    name: str
    parent_frame_id: str
    matrix: tuple[tuple[float, ...], ...]

    def as_dict(self) -> dict:
        """Return the JSON-safe representation used by the web GUI."""
        return {
            "name": self.name,
            "parent_frame_id": self.parent_frame_id,
            "matrix": [list(row) for row in self.matrix],
        }


def load_transformation_matrices(path: Path) -> list[TransformationMatrix]:
    """Load and validate a non-empty JSON list of rigid 4x4 matrices."""
    with path.open(encoding="utf-8") as file:
        entries = json.load(file)

    if not isinstance(entries, list) or not entries:
        raise ValueError("transformation matrix file must contain a non-empty list")

    transformations = []
    names = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"transformation entry {index} must be an object")

        name = entry.get("name")
        parent_frame_id = entry.get("parent_frame_id")
        if not isinstance(name, str) or not name:
            raise ValueError(f"transformation entry {index} needs a non-empty name")
        if name in names:
            raise ValueError(f"duplicate transformation name: {name}")
        if not isinstance(parent_frame_id, str) or not parent_frame_id:
            raise ValueError(
                f"transformation '{name}' needs a non-empty parent_frame_id"
            )

        matrix = _validated_rigid_matrix(entry.get("matrix"), name)
        transformations.append(
            TransformationMatrix(
                name=name,
                parent_frame_id=parent_frame_id,
                matrix=tuple(tuple(float(value) for value in row) for row in matrix),
            )
        )
        names.add(name)

    return transformations


def transformation_to_message(
    transformation: TransformationMatrix,
    stamp,
    child_frame_id: str,
) -> NamedTransform:
    """Create a stamped ROS transform from one validated matrix."""
    if not child_frame_id:
        raise ValueError("transform child frame must not be empty")

    message = NamedTransform()
    message.header.stamp = stamp
    message.header.frame_id = transformation.parent_frame_id
    message.child_frame_id = child_frame_id
    message.transformation_name = transformation.name
    message.matrix = np.asarray(
        transformation.matrix,
        dtype=np.float64,
    ).reshape(-1).tolist()
    return message


def _validated_rigid_matrix(value, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"transformation '{name}' matrix must contain only numbers"
        ) from error

    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(
            f"transformation '{name}' matrix must be a finite 4x4 matrix"
        )
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError(
            f"transformation '{name}' matrix must have homogeneous bottom row"
        )

    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        np.linalg.det(rotation),
        1.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"transformation '{name}' matrix must contain a proper rotation"
        )
    return matrix
