from types import SimpleNamespace

import numpy as np
import open3d as o3d
import object_scanner_processing.repair_segmentation as repair
import pytest


def _triangle_mesh():
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(
            np.array(
                [
                    [-0.02, -0.02, 0.0],
                    [0.02, -0.02, 0.0],
                    [0.0, 0.02, 0.0],
                ]
            )
        ),
        o3d.utility.Vector3iVector(np.array([[0, 1, 2]], dtype=np.int32)),
    )
    mesh.compute_triangle_normals()
    return mesh


def test_initial_transform_places_repair_center_at_requested_position():
    mesh = _triangle_mesh()

    transform = repair._initial_transform(mesh)
    centre = np.asarray(mesh.vertices).mean(axis=0)
    world_centre = transform[:3, :3] @ centre + transform[:3, 3]

    np.testing.assert_allclose(world_centre, repair.REPAIR_CENTER_WORLD_M)
    np.testing.assert_allclose(
        np.linalg.norm(transform[:3, :3], axis=0),
        0.0057,
        atol=1e-10,
    )


def test_segment_repair_uses_refined_transform_and_mesh_distance(
    tmp_path,
    monkeypatch,
):
    repair_path = tmp_path / "repair.stl"
    assert o3d.io.write_triangle_mesh(str(repair_path), _triangle_mesh())
    cloud = SimpleNamespace(
        xyz=np.array(
            [
                [0.10, 0.00, 0.001],
                [0.10, 0.00, 0.004],
                [0.10, 0.00, 0.006],
                [0.20, 0.00, 0.000],
            ],
            dtype=np.float32,
        )
    )
    monkeypatch.setattr(repair, "read_fused_cloud", lambda _: cloud)
    transform = np.eye(4)
    transform[0, 3] = 0.10

    result = repair.segment_repair(
        tmp_path / "aligned_recording.sqlite3",
        repair_path,
        transform,
    )

    np.testing.assert_allclose(result.transform, transform)
    np.testing.assert_allclose(
        result.xyz,
        cloud.xyz[:2],
        atol=1e-7,
    )
    assert result.scale_m_per_stl_unit == 1.0


@pytest.mark.parametrize(
    "transform",
    [
        np.eye(3),
        np.full((4, 4), np.nan),
        np.diag([1.0, 2.0, 1.0, 1.0]),
        np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_rejects_invalid_similarity_transform(transform):
    with pytest.raises(ValueError):
        repair._validated_similarity_transform(transform)
