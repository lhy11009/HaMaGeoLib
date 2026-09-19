import json
import runpy

from hamageolib.research.haoyuan_collision0.scripts.generate_boundary_visualization import (
    generate_visualization_script,
)


def test_generate_visualization_script_reads_processing_metadata(tmp_path):
    output_directory = tmp_path / "boundary meshes"
    output_directory.mkdir()
    solution_path = tmp_path / "solution.pvtu"
    metadata_path = output_directory / "boundary_mesh_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "solution": str(solution_path.resolve()),
                "boundary_directory": str(output_directory.resolve()),
                "longitude_bounds": [151.0, 209.0],
                "radial_bounds": [4_760_000.0, 6_360_000.0],
                "boundary_meshes": {},
            }
        ),
        encoding="utf-8",
    )

    script_path = generate_visualization_script(metadata_path)

    assert script_path == output_directory / "visualize_boundary_meshes.py"
    script_contents = script_path.read_text(encoding="utf-8")
    for placeholder in (
        "__BOUNDARY_DIRECTORY__",
        "__STATE_FILE__",
        "__VALIDATION_FILE__",
        "__INNER_RADIUS__",
        "__OUTER_RADIUS__",
        "__SOLUTION_PATH__",
        "__LONGITUDE_MIN__",
        "__LONGITUDE_MAX__",
    ):
        assert placeholder not in script_contents

    configured_values = runpy.run_path(str(script_path))
    assert configured_values["BOUNDARY_DIRECTORY"] == output_directory.resolve()
    assert configured_values["STATE_FILE"] == (
        output_directory / "boundary_meshes.pvsm"
    ).resolve()
    assert configured_values["VALIDATION_FILE"] == (
        output_directory / "boundary_meshes_validation.json"
    ).resolve()
    assert configured_values["RADIAL_BOUNDS"] == (4_760_000.0, 6_360_000.0)
    assert configured_values["SOLUTION_PATH"] == solution_path.resolve()
    assert configured_values["LONGITUDE_BOUNDS"] == (151.0, 209.0)
    assert configured_values["LOAD_ORIGINAL_SOLUTION"] is False
