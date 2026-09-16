import re
import runpy

import numpy as np
import pytest
import vtk

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    create_boundary_grid,
    interpolate_velocity,
    uniform_coordinates,
    write_boundary_meshes,
    write_boundary_velocity_meshes,
    write_visualization_script,
)


RADIUS_BOUNDS = (4_760e3, 6_360e3)
LATITUDE_BOUNDS = (-55.0, -20.0)
LONGITUDE_BOUNDS = (150.0, 210.0)
TIMESTAMP_PATTERN = re.compile(
    r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [^\]]+\] .+$"
)


def assert_timestamped(lines):
    assert lines
    assert all(TIMESTAMP_PATTERN.match(line) for line in lines)


def create_linear_velocity_source():
    """Create a volume with velocity=(x, 2*y, 3*z) at its vertices."""
    source = vtk.vtkImageData()
    source.SetOrigin(-7e6, -7e6, -7e6)
    source.SetSpacing(14e6, 14e6, 14e6)
    source.SetDimensions(2, 2, 2)

    velocity = vtk.vtkDoubleArray()
    velocity.SetName("velocity")
    velocity.SetNumberOfComponents(3)
    for point_index in range(source.GetNumberOfPoints()):
        x, y, z = source.GetPoint(point_index)
        velocity.InsertNextTuple3(x, 2.0 * y, 3.0 * z)
    source.GetPointData().AddArray(velocity)
    return source


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


def test_interpolate_velocity_adds_vector_and_scalar_components():
    source = create_linear_velocity_source()
    boundary_grid = create_boundary_grid(
        "west",
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )
    original_number_of_arrays = boundary_grid.GetPointData().GetNumberOfArrays()

    interpolated_grid = interpolate_velocity(source, boundary_grid)

    interpolated_dimensions = [0, 0, 0]
    boundary_dimensions = [0, 0, 0]
    interpolated_grid.GetDimensions(interpolated_dimensions)
    boundary_grid.GetDimensions(boundary_dimensions)
    assert interpolated_dimensions == boundary_dimensions
    assert interpolated_grid.GetNumberOfPoints() == boundary_grid.GetNumberOfPoints()
    assert boundary_grid.GetPointData().GetNumberOfArrays() == original_number_of_arrays

    velocity = interpolated_grid.GetPointData().GetArray("velocity")
    components = [
        interpolated_grid.GetPointData().GetArray(name)
        for name in ("Vx", "Vy", "Vz")
    ]
    assert velocity is not None
    assert all(component is not None for component in components)

    for point_index in range(interpolated_grid.GetNumberOfPoints()):
        x, y, z = interpolated_grid.GetPoint(point_index)
        expected_velocity = (x, 2.0 * y, 3.0 * z)
        assert velocity.GetTuple3(point_index) == pytest.approx(expected_velocity)
        assert tuple(
            component.GetValue(point_index) for component in components
        ) == pytest.approx(expected_velocity)


def test_interpolate_velocity_rejects_points_outside_source():
    source = create_linear_velocity_source()
    boundary_grid = vtk.vtkStructuredGrid()
    boundary_grid.SetDimensions(1, 1, 1)
    points = vtk.vtkPoints()
    points.InsertNextPoint(20e6, 0.0, 0.0)
    boundary_grid.SetPoints(points)

    with pytest.raises(ValueError, match="outside the source mesh"):
        interpolate_velocity(source, boundary_grid)


def test_interpolate_velocity_requires_velocity_array_name():
    source = create_linear_velocity_source()
    source.GetPointData().GetArray("velocity").SetName("other_velocity")
    boundary_grid = create_boundary_grid(
        "west",
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    with pytest.raises(ValueError, match="point-data array 'velocity'"):
        interpolate_velocity(source, boundary_grid)


def test_write_boundary_velocity_meshes_preserves_interpolated_arrays(tmp_path):
    output_paths = write_boundary_velocity_meshes(
        tmp_path,
        create_linear_velocity_source(),
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    for output_path in output_paths.values():
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(output_path))
        reader.Update()
        point_data = reader.GetOutput().GetPointData()
        assert point_data.GetArray("velocity") is not None
        assert point_data.GetArray("Vx") is not None
        assert point_data.GetArray("Vy") is not None
        assert point_data.GetArray("Vz") is not None


def test_write_boundary_velocity_meshes_reports_timestamped_progress(
    tmp_path, capsys
):
    write_boundary_velocity_meshes(
        tmp_path,
        create_linear_velocity_source(),
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    lines = capsys.readouterr().out.splitlines()
    assert_timestamped(lines)
    for boundary_name in ("west", "east", "north", "south"):
        assert any(f"Creating {boundary_name} boundary grid" in line for line in lines)
        assert any(
            f"Interpolating velocity onto {boundary_name} boundary" in line
            for line in lines
        )
        assert any(f"Writing {boundary_name} boundary mesh" in line for line in lines)
        assert any(f"Finished {boundary_name} boundary" in line for line in lines)


def test_write_boundary_meshes_reports_progress_without_interpolation(
    tmp_path, capsys
):
    write_boundary_meshes(
        tmp_path,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )

    lines = capsys.readouterr().out.splitlines()
    assert_timestamped(lines)
    assert not any("Interpolating velocity" in line for line in lines)
    for boundary_name in ("west", "east", "north", "south"):
        assert any(f"Creating {boundary_name} boundary grid" in line for line in lines)
        assert any(f"Writing {boundary_name} boundary mesh" in line for line in lines)
        assert any(f"Finished {boundary_name} boundary" in line for line in lines)


def test_write_visualization_script_embeds_absolute_input_paths(tmp_path):
    output_directory = tmp_path / "boundary meshes"
    solution_path = tmp_path / "solution's mesh.pvtu"
    solution_path.touch()

    script_path = write_visualization_script(output_directory, solution_path)

    assert script_path == output_directory / "visualize_boundary_meshes.py"
    script_contents = script_path.read_text(encoding="utf-8")
    assert "__SOLUTION_PATH__" not in script_contents
    assert "__BOUNDARY_DIRECTORY__" not in script_contents
    assert "__STATE_FILE__" not in script_contents
    assert "__VALIDATION_FILE__" not in script_contents

    configured_values = runpy.run_path(str(script_path))
    assert configured_values["SOLUTION_PATH"] == solution_path.resolve()
    assert configured_values["BOUNDARY_DIRECTORY"] == output_directory.resolve()
    assert configured_values["STATE_FILE"] == (
        output_directory / "boundary_meshes.pvsm"
    ).resolve()
    assert configured_values["VALIDATION_FILE"] == (
        output_directory / "boundary_meshes_validation.json"
    ).resolve()
    assert configured_values["LONGITUDE_BOUNDS"] == LONGITUDE_BOUNDS
