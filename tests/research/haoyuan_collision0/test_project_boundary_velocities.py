import json

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    _write_boundary_grid,
    create_boundary_grid,
)
from hamageolib.research.haoyuan_collision0.scripts.project_boundary_velocities import (
    boundary_outward_normals,
    project_boundary_meshes,
    project_velocity,
)


@pytest.mark.parametrize(
    ("boundary_name", "point", "expected_normal"),
    [
        ("east", (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ("west", (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ("north", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("south", (1.0, 0.0, 0.0), (0.0, 0.0, -1.0)),
    ],
)
def test_boundary_outward_normals_follow_boundary_orientation(
    boundary_name, point, expected_normal
):
    normals = boundary_outward_normals(
        np.asarray([point], dtype=float), boundary_name
    )

    assert normals[0] == pytest.approx(expected_normal)


def test_project_velocity_splits_normal_and_parallel_vectors():
    points = np.asarray([(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)])
    velocity = np.asarray([(2.0, 3.0, 4.0), (5.0, 6.0, 7.0)])

    normals, signed_normal, normal_velocity, parallel_velocity = (
        project_velocity(points, velocity, "east")
    )

    assert np.linalg.norm(normals, axis=1) == pytest.approx(1.0)
    assert normal_velocity == pytest.approx(signed_normal[:, None] * normals)
    assert normal_velocity + parallel_velocity == pytest.approx(velocity)
    assert np.einsum("ij,ij->i", parallel_velocity, normals) == pytest.approx(
        0.0, abs=1e-12
    )


def test_project_boundary_meshes_updates_only_metadata_meshes(tmp_path):
    boundary_paths = {}
    for boundary_name in ("west", "east", "north", "south"):
        grid = create_boundary_grid(
            boundary_name,
            (1.0, 2.0),
            (-30.0, -20.0),
            (150.0, 160.0),
            radius_spacing=1.0,
            lateral_spacing=10.0,
        )
        velocity = np.tile((2.0, 3.0, 4.0), (grid.GetNumberOfPoints(), 1))
        velocity_array = numpy_to_vtk(velocity, deep=True)
        velocity_array.SetName("velocity")
        grid.GetPointData().AddArray(velocity_array)
        boundary_path = tmp_path / f"chunk_3d_{boundary_name}_mesh.vts"
        _write_boundary_grid(grid, boundary_path)
        boundary_paths[boundary_name] = str(boundary_path)

    metadata_path = tmp_path / "boundary_mesh_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "solution": str(tmp_path / "must-not-be-read.pvtu"),
                "boundary_directory": str(tmp_path),
                "longitude_bounds": [150.0, 160.0],
                "radial_bounds": [1.0, 2.0],
                "boundary_meshes": boundary_paths,
            }
        ),
        encoding="utf-8",
    )

    projection = project_boundary_meshes(metadata_path)

    assert set(projection["boundaries"]) == {
        "west",
        "east",
        "north",
        "south",
    }
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["velocity_projection"] == projection
    for boundary_path in boundary_paths.values():
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(boundary_path)
        reader.Update()
        point_data = reader.GetOutput().GetPointData()
        arrays = {
            name: vtk_to_numpy(point_data.GetArray(name))
            for name in (
                "velocity",
                "boundary_normal",
                "velocity_normal",
                "velocity_parallel",
                "velocity_normal_signed",
            )
        }
        assert np.linalg.norm(arrays["boundary_normal"], axis=1) == pytest.approx(
            1.0
        )
        assert arrays["velocity_normal"] + arrays["velocity_parallel"] == (
            pytest.approx(arrays["velocity"])
        )
        assert np.einsum(
            "ij,ij->i",
            arrays["velocity_parallel"],
            arrays["boundary_normal"],
        ) == pytest.approx(0.0, abs=1e-12)
