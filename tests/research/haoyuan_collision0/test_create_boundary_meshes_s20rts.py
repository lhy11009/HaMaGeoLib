import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import vtk_to_numpy


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SOLUTION_PATH = (
    PACKAGE_ROOT
    / "big_tests/Collision0/S20RTS/output-S20RTS/solution"
    / "solution-00000.pvtu"
)
SCRIPT_PATH = (
    PACKAGE_ROOT
    / "hamageolib/research/haoyuan_collision0/scripts"
    / "create_boundary_meshes.py"
)
GENERATOR_PATH = (
    PACKAGE_ROOT
    / "hamageolib/research/haoyuan_collision0/scripts"
    / "generate_boundary_visualization.py"
)
PROJECTOR_PATH = (
    PACKAGE_ROOT
    / "hamageolib/research/haoyuan_collision0/scripts"
    / "project_boundary_velocities.py"
)
TEST_DIRECTORY = (
    PACKAGE_ROOT
    / ".test/research-haoyuan_collision0-create-boundary-meshes-s20rts"
)


@pytest.mark.big_test
def test_create_boundary_meshes_with_s20rts_solution():
    """Run the boundary-mesh CLI against the checked-out S20RTS fixture."""
    if not SOLUTION_PATH.is_file():
        pytest.skip(f"S20RTS fixture is not available: {SOLUTION_PATH}")

    if TEST_DIRECTORY.exists():
        shutil.rmtree(TEST_DIRECTORY)
    TEST_DIRECTORY.mkdir(parents=True)
    config_path = TEST_DIRECTORY / "boundary_mesh_config.txt"
    config_path.write_text(
        "[paths]\n"
        f"solution = {SOLUTION_PATH}\n"
        f"output_directory = {TEST_DIRECTORY}\n\n"
        "[geometry]\n"
        "radius_min = 4760000\n"
        "radius_max = 6360000\n"
        "latitude_min = -55.0\n"
        "latitude_max = -20.0\n"
        "longitude_min = 150.0\n"
        "longitude_max = 210.0\n\n"
        "[resolution]\n"
        "radius_spacing = 100000\n"
        "lateral_spacing = 2.5\n\n"
        "[interpolation]\n"
        "retry_tolerance = 10.0\n"
        "use_nearest_valid_point = true\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--config", str(config_path)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (TEST_DIRECTORY / "create_boundary_meshes.stdout").write_text(
        result.stdout, encoding="utf-8"
    )
    (TEST_DIRECTORY / "create_boundary_meshes.stderr").write_text(
        result.stderr, encoding="utf-8"
    )
    assert result.returncode == 0, result.stderr
    assert "Finished boundary mesh processing" in result.stdout
    assert "Source solution radial bounds" in result.stdout

    expected_dimensions = {
        "west": (17, 15, 1),
        "east": (17, 15, 1),
        "north": (17, 25, 1),
        "south": (17, 25, 1),
    }
    for boundary_name, dimensions in expected_dimensions.items():
        mesh_path = TEST_DIRECTORY / f"chunk_3d_{boundary_name}_mesh.vts"
        assert mesh_path.is_file()
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(mesh_path))
        reader.Update()
        grid = reader.GetOutput()
        actual_dimensions = [0, 0, 0]
        grid.GetDimensions(actual_dimensions)
        assert tuple(actual_dimensions) == dimensions

        point_data = grid.GetPointData()
        for array_name in ("velocity", "Vx", "Vy", "Vz"):
            array = point_data.GetArray(array_name)
            assert array is not None
            values = vtk_to_numpy(array)
            assert np.isfinite(values).all()
            if array_name == "velocity":
                assert np.any(np.abs(values) > 0.0)

    metadata_path = TEST_DIRECTORY / "boundary_mesh_metadata.json"
    assert metadata_path.is_file()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["solution"] == str(SOLUTION_PATH.resolve())
    assert metadata["radial_bounds"] == pytest.approx(
        (3_481_000.0, 6_371_000.0), abs=2.0
    )

    projection_result = subprocess.run(
        [sys.executable, str(PROJECTOR_PATH), "--metadata", str(metadata_path)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (TEST_DIRECTORY / "project_boundary_velocities.stdout").write_text(
        projection_result.stdout, encoding="utf-8"
    )
    (TEST_DIRECTORY / "project_boundary_velocities.stderr").write_text(
        projection_result.stderr, encoding="utf-8"
    )
    assert projection_result.returncode == 0, projection_result.stderr
    assert "Finished boundary velocity projection" in projection_result.stdout

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    projection = metadata["velocity_projection"]
    assert set(projection["boundaries"]) == set(expected_dimensions)
    for boundary_name in expected_dimensions:
        reader = vtk.vtkXMLStructuredGridReader()
        reader.SetFileName(str(metadata["boundary_meshes"][boundary_name]))
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

    visualization_path = TEST_DIRECTORY / "visualize_boundary_meshes.py"
    assert not visualization_path.exists()
    visualization_result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--metadata", str(metadata_path)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    (TEST_DIRECTORY / "generate_boundary_visualization.stdout").write_text(
        visualization_result.stdout, encoding="utf-8"
    )
    (TEST_DIRECTORY / "generate_boundary_visualization.stderr").write_text(
        visualization_result.stderr, encoding="utf-8"
    )
    assert visualization_result.returncode == 0, visualization_result.stderr
    assert visualization_path.is_file()
    visualization = runpy.run_path(str(visualization_path))
    assert visualization["SOLUTION_PATH"] == SOLUTION_PATH.resolve()
    assert visualization["LOAD_ORIGINAL_SOLUTION"] is False
    assert visualization["RADIAL_BOUNDS"] == pytest.approx(
        (3_481_000.0, 6_371_000.0), abs=2.0
    )
    assert visualization["VELOCITY_GLYPH_SCALE_FACTOR"] == 1.0e6
    assert visualization["VELOCITY_COLOR_RANGE"] == (0.0, 1.0)
    assert visualization["VELOCITY_COLOR_PRESETS"] == (
        "Blue - Green - Orange",
        "Blue Green Orange",
    )
    assert visualization["SPHERE_OPACITIES"] == {
        "inner": 1.0,
        "outer": 0.2,
    }
