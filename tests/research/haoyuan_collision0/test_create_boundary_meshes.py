import csv
import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vtk

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    create_boundary_grid,
    interpolate_velocity,
    load_boundary_mesh_config,
    read_solution_dataset,
    report_solution_bounds,
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


def write_boundary_mesh_config(path, **overrides):
    """Write a complete test configuration with optional value overrides."""
    values = {
        "solution": "solution/solution-00000.pvtu",
        "output_directory": "output",
        "radius_min": "4760000",
        "radius_max": "6360000",
        "latitude_min": "-55.0",
        "latitude_max": "-20.0",
        "longitude_min": "150.0",
        "longitude_max": "210.0",
        "radius_spacing": "100000",
        "lateral_spacing": "2.5",
    }
    values.update(overrides)
    path.write_text(
        "[paths]\n"
        f"solution = {values['solution']}\n"
        f"output_directory = {values['output_directory']}\n\n"
        "[geometry]\n"
        f"radius_min = {values['radius_min']}\n"
        f"radius_max = {values['radius_max']}\n"
        f"latitude_min = {values['latitude_min']}\n"
        f"latitude_max = {values['latitude_max']}\n"
        f"longitude_min = {values['longitude_min']}\n"
        f"longitude_max = {values['longitude_max']}\n\n"
        "[resolution]\n"
        f"radius_spacing = {values['radius_spacing']}\n"
        f"lateral_spacing = {values['lateral_spacing']}\n",
        encoding="utf-8",
    )


def write_parallel_unstructured_grid(path, include_velocity=True):
    """Write a one-piece PVTU containing wanted and unwanted data arrays."""
    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)):
        points.InsertNextPoint(*point)

    tetrahedron = vtk.vtkTetra()
    for point_index in range(4):
        tetrahedron.GetPointIds().SetId(point_index, point_index)

    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    grid.InsertNextCell(tetrahedron.GetCellType(), tetrahedron.GetPointIds())

    if include_velocity:
        velocity = vtk.vtkFloatArray()
        velocity.SetName("velocity")
        velocity.SetNumberOfComponents(3)
        for point_index in range(4):
            velocity.InsertNextTuple3(point_index, 2 * point_index, 3 * point_index)
        grid.GetPointData().AddArray(velocity)

    temperature = vtk.vtkFloatArray()
    temperature.SetName("temperature")
    for point_index in range(4):
        temperature.InsertNextValue(300 + point_index)
    grid.GetPointData().AddArray(temperature)

    material_id = vtk.vtkIntArray()
    material_id.SetName("material_id")
    material_id.InsertNextValue(7)
    grid.GetCellData().AddArray(material_id)

    writer = vtk.vtkXMLPUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetNumberOfPieces(1)
    writer.SetStartPiece(0)
    writer.SetEndPiece(0)
    assert writer.Write() == 1


def test_uniform_coordinates_include_evenly_divisible_bounds():
    coordinates = uniform_coordinates(*RADIUS_BOUNDS, spacing=100e3)

    assert len(coordinates) == 17
    assert coordinates[0] == RADIUS_BOUNDS[0]
    assert coordinates[-1] == RADIUS_BOUNDS[1]
    assert np.allclose(np.diff(coordinates), 100e3)


def test_uniform_coordinates_reject_nondivisible_interval():
    with pytest.raises(ValueError, match="divide the interval evenly"):
        uniform_coordinates(4_770e3, 6_360e3, spacing=100e3)


def test_load_boundary_mesh_config_reads_all_runtime_inputs(tmp_path, monkeypatch):
    config_path = tmp_path / "boundary_mesh_config.txt"
    write_boundary_mesh_config(config_path)
    monkeypatch.chdir(tmp_path)

    config = load_boundary_mesh_config(config_path)

    assert config.solution == (tmp_path / "solution/solution-00000.pvtu").resolve()
    assert config.output_directory == (tmp_path / "output").resolve()
    assert config.radius_bounds == RADIUS_BOUNDS
    assert config.latitude_bounds == LATITUDE_BOUNDS
    assert config.longitude_bounds == LONGITUDE_BOUNDS
    assert config.radius_spacing == 100e3
    assert config.lateral_spacing == 2.5


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"radius_spacing": "0"}, "radius_spacing must be positive"),
        ({"longitude_max": "140"}, "longitude_max must be greater"),
        ({"lateral_spacing": "4"}, "divide the interval evenly"),
        ({"radius_min": "not-a-number"}, "must be a number"),
    ],
)
def test_load_boundary_mesh_config_rejects_invalid_values(
    tmp_path, override, message
):
    config_path = tmp_path / "boundary_mesh_config.txt"
    write_boundary_mesh_config(config_path, **override)

    with pytest.raises(ValueError, match=message):
        load_boundary_mesh_config(config_path)


def test_load_boundary_mesh_config_reports_missing_key(tmp_path):
    config_path = tmp_path / "boundary_mesh_config.txt"
    config_path.write_text("[paths]\nsolution = solution.pvtu\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required configuration value"):
        load_boundary_mesh_config(config_path)


def test_standard_arushi_configuration_preserves_original_mesh_parameters():
    config_path = (
        Path(__file__).parents[3]
        / "hamageolib/research/haoyuan_collision0/files/arushi_dataset"
        / "boundary_mesh_config.txt"
    )

    config = load_boundary_mesh_config(config_path)

    assert config.radius_bounds == RADIUS_BOUNDS
    assert config.latitude_bounds == LATITUDE_BOUNDS
    assert config.longitude_bounds == LONGITUDE_BOUNDS
    assert config.radius_spacing == 100e3
    assert config.lateral_spacing == 2.5
    assert str(config.solution).endswith("solution/solution-00000.pvtu")


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


def test_interpolate_velocity_writes_invalid_points_before_raising(tmp_path):
    source = create_linear_velocity_source()
    boundary_grid = vtk.vtkStructuredGrid()
    boundary_grid.SetDimensions(2, 1, 1)
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    points.InsertNextPoint(20e6, 1e6, -2e6)
    boundary_grid.SetPoints(points)
    diagnostic_path = tmp_path / "chunk_3d_west_invalid_points.csv"

    with pytest.raises(ValueError, match=str(diagnostic_path)):
        interpolate_velocity(source, boundary_grid, diagnostic_path)

    with diagnostic_path.open(newline="", encoding="utf-8") as diagnostic_file:
        rows = list(csv.DictReader(diagnostic_file))
    assert rows == [
        {
            "point_index": "1",
            "x": "20000000.0",
            "y": "1000000.0",
            "z": "-2000000.0",
        }
    ]


def test_interpolate_velocity_removes_stale_diagnostic_when_all_points_valid(
    tmp_path,
):
    source = create_linear_velocity_source()
    boundary_grid = create_boundary_grid(
        "west",
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )
    diagnostic_path = tmp_path / "chunk_3d_west_invalid_points.csv"
    diagnostic_path.write_text("stale data\n", encoding="utf-8")

    interpolate_velocity(source, boundary_grid, diagnostic_path)

    assert not diagnostic_path.exists()


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


def test_read_pvtu_loads_only_velocity(tmp_path, capsys):
    solution_path = tmp_path / "solution.pvtu"
    write_parallel_unstructured_grid(solution_path)

    dataset = read_solution_dataset(solution_path)

    assert dataset.GetPointData().GetArray("velocity") is not None
    assert dataset.GetPointData().GetArray("temperature") is None
    assert dataset.GetCellData().GetArray("material_id") is None
    lines = capsys.readouterr().out.splitlines()
    assert_timestamped(lines)
    assert any("Reading VTK metadata" in line for line in lines)
    assert any("Loading mesh geometry and velocity only" in line for line in lines)


def test_read_pvtu_requires_velocity_array(tmp_path):
    solution_path = tmp_path / "solution.pvtu"
    write_parallel_unstructured_grid(solution_path, include_velocity=False)

    with pytest.raises(ValueError, match="point-data array 'velocity'"):
        read_solution_dataset(solution_path)


def test_read_solution_dataset_preserves_pyvista_fallback(monkeypatch, tmp_path):
    solution_path = tmp_path / "solution.pvd"
    expected_dataset = object()
    fake_pyvista = SimpleNamespace(
        read=lambda actual_path: (
            expected_dataset if actual_path == solution_path else None
        )
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_pyvista)

    assert read_solution_dataset(solution_path) is expected_dataset


def test_report_solution_bounds_supports_composite_dataset(capsys):
    source = create_linear_velocity_source()
    composite = vtk.vtkMultiBlockDataSet()
    composite.SetBlock(0, source)

    report_solution_bounds(composite)

    lines = capsys.readouterr().out.splitlines()
    assert_timestamped(lines)
    assert "Source solution bounds" in lines[0]
    assert "x=[-7.000000e+06, 7.000000e+06] m" in lines[0]
    assert "y=[-7.000000e+06, 7.000000e+06] m" in lines[0]
    assert "z=[-7.000000e+06, 7.000000e+06] m" in lines[0]


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


def test_write_boundary_velocity_meshes_names_invalid_point_file(tmp_path):
    source = create_linear_velocity_source()
    source.SetOrigin(-1e6, -1e6, -1e6)
    source.SetSpacing(2e6, 2e6, 2e6)
    expected_path = tmp_path / "chunk_3d_west_invalid_points.csv"

    with pytest.raises(ValueError, match=str(expected_path)):
        write_boundary_velocity_meshes(
            tmp_path,
            source,
            RADIUS_BOUNDS,
            LATITUDE_BOUNDS,
            LONGITUDE_BOUNDS,
            radius_spacing=100e3,
            lateral_spacing=2.5,
        )

    assert expected_path.is_file()
    with expected_path.open(newline="", encoding="utf-8") as diagnostic_file:
        rows = list(csv.DictReader(diagnostic_file))
    assert len(rows) == 255


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

    longitude_bounds = (151.0, 209.0)
    script_path = write_visualization_script(
        output_directory, solution_path, longitude_bounds
    )

    assert script_path == output_directory / "visualize_boundary_meshes.py"
    script_contents = script_path.read_text(encoding="utf-8")
    assert "__SOLUTION_PATH__" not in script_contents
    assert "__BOUNDARY_DIRECTORY__" not in script_contents
    assert "__STATE_FILE__" not in script_contents
    assert "__VALIDATION_FILE__" not in script_contents
    assert "__LONGITUDE_MIN__" not in script_contents
    assert "__LONGITUDE_MAX__" not in script_contents

    configured_values = runpy.run_path(str(script_path))
    assert configured_values["SOLUTION_PATH"] == solution_path.resolve()
    assert configured_values["BOUNDARY_DIRECTORY"] == output_directory.resolve()
    assert configured_values["STATE_FILE"] == (
        output_directory / "boundary_meshes.pvsm"
    ).resolve()
    assert configured_values["VALIDATION_FILE"] == (
        output_directory / "boundary_meshes_validation.json"
    ).resolve()
    assert configured_values["LONGITUDE_BOUNDS"] == longitude_bounds
