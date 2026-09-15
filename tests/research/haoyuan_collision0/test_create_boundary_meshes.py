import numpy as np
import pytest
import vtk

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    create_boundary_grid,
    uniform_coordinates,
    write_boundary_meshes,
)


RADIUS_BOUNDS = (4_760e3, 6_360e3)
LATITUDE_BOUNDS = (-55.0, -20.0)
LONGITUDE_BOUNDS = (150.0, 210.0)


def test_uniform_coordinates_include_evenly_divisible_bounds():
    coordinates = uniform_coordinates(*RADIUS_BOUNDS, spacing=100e3)

    assert len(coordinates) == 17
    assert coordinates[0] == RADIUS_BOUNDS[0]
    assert coordinates[-1] == RADIUS_BOUNDS[1]
    assert np.allclose(np.diff(coordinates), 100e3)


def test_uniform_coordinates_reject_nondivisible_interval():
    with pytest.raises(ValueError, match="divide the interval evenly"):
        uniform_coordinates(4_770e3, 6_360e3, spacing=100e3)


@pytest.mark.parametrize(
    ("boundary", "dimensions", "number_of_cells"),
    [
        ("west", (17, 15, 1), 224),
        ("east", (17, 15, 1), 224),
        ("north", (17, 25, 1), 384),
        ("south", (17, 25, 1), 384),
    ],
)
def test_create_boundary_grid_has_connected_surface_cells(
    boundary, dimensions, number_of_cells
):
    grid = create_boundary_grid(
        boundary,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    actual_dimensions = [0, 0, 0]
    grid.GetDimensions(actual_dimensions)
    assert tuple(actual_dimensions) == dimensions
    assert grid.GetNumberOfPoints() == np.prod(dimensions)
    assert grid.GetNumberOfCells() == number_of_cells
    assert grid.GetPoints().GetDataType() == vtk.VTK_DOUBLE


def test_write_boundary_meshes_creates_readable_vts_files(tmp_path):
    output_paths = write_boundary_meshes(
        tmp_path,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    assert set(output_paths) == {"west", "east", "north", "south"}
    for boundary, output_path in output_paths.items():
        assert output_path == tmp_path / f"chunk_3d_{boundary}_mesh.vts"
        assert output_path.is_file()

        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(output_path))
        reader.Update()
        grid = reader.GetOutput()
        assert grid.GetNumberOfPoints() > 0
        assert grid.GetNumberOfCells() > 0
