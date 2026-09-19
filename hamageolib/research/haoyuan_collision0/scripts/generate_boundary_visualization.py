"""Generate a configured ParaView script from boundary-mesh metadata."""

import argparse
import json
from pathlib import Path


VISUALIZATION_SCRIPT_NAME = "visualize_boundary_meshes.py"


def load_boundary_mesh_metadata(metadata_path):
    """Load and minimally validate boundary processing metadata."""
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"boundary mesh metadata does not exist: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_keys = {
        "solution",
        "boundary_directory",
        "longitude_bounds",
        "radial_bounds",
        "boundary_meshes",
    }
    missing_keys = sorted(required_keys - metadata.keys())
    if missing_keys:
        raise ValueError(
            "boundary mesh metadata is missing: " + ", ".join(missing_keys)
        )
    return metadata


def generate_visualization_script(metadata_path, output_path=None):
    """Configure the ParaView template using processed mesh metadata."""
    metadata = load_boundary_mesh_metadata(metadata_path)
    boundary_directory = Path(metadata["boundary_directory"]).resolve()
    solution_path = Path(metadata["solution"]).resolve()
    longitude_bounds = metadata["longitude_bounds"]
    radial_bounds = metadata["radial_bounds"]
    if output_path is None:
        output_path = boundary_directory / VISUALIZATION_SCRIPT_NAME
    else:
        output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    template_path = Path(__file__).with_name(VISUALIZATION_SCRIPT_NAME)
    configured_values = {
        "__SOLUTION_PATH__": solution_path,
        "__BOUNDARY_DIRECTORY__": boundary_directory,
        "__STATE_FILE__": boundary_directory / "boundary_meshes.pvsm",
        "__VALIDATION_FILE__": boundary_directory
        / "boundary_meshes_validation.json",
        "__INNER_RADIUS__": float(min(radial_bounds)),
        "__OUTER_RADIUS__": float(max(radial_bounds)),
        "__LONGITUDE_MIN__": float(min(longitude_bounds)),
        "__LONGITUDE_MAX__": float(max(longitude_bounds)),
    }

    configured_script = template_path.read_text(encoding="utf-8")
    for placeholder, configured_value in configured_values.items():
        quoted_placeholder = f'"{placeholder}"'
        if configured_script.count(quoted_placeholder) != 1:
            raise ValueError(
                f"visualization template must contain {quoted_placeholder} exactly once"
            )
        if isinstance(configured_value, Path):
            replacement = repr(str(configured_value))
        else:
            replacement = repr(configured_value)
        configured_script = configured_script.replace(
            quoted_placeholder, replacement
        )

    output_path.write_text(configured_script, encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate a configured ParaView boundary visualization script."
    )
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_path = generate_visualization_script(args.metadata, args.output)
    print(f"Created visualization script: {output_path}", flush=True)


if __name__ == "__main__":
    main()
