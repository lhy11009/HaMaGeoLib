"""Build a ParaView state containing chunk boundaries and radial spheres.

Run this file with ``pvpython`` after creating the four structured boundary
meshes with ``create_boundary_meshes.py``.  The configured copy embeds the
source mesh's inner and outer radii, so it does not load the original solution.
"""

import argparse
import json
from pathlib import Path

# These uppercase strings are replaced when ``create_boundary_meshes.py``
# writes a configured copy beside the generated boundary meshes.
BOUNDARY_DIRECTORY = Path("__BOUNDARY_DIRECTORY__")
STATE_FILE = Path("__STATE_FILE__")
VALIDATION_FILE = Path("__VALIDATION_FILE__")
RADIAL_BOUNDS = ("__INNER_RADIUS__", "__OUTER_RADIUS__")
BOUNDARY_NAMES = ("west", "east", "north", "south")
SPHERE_RESOLUTION = 128
SPHERE_OPACITY = 0.02


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Load chunk boundary meshes, create inner and outer radius "
            "spheres, and save the ParaView state."
        )
    )
    parser.add_argument(
        "--boundary-directory", type=Path, default=BOUNDARY_DIRECTORY
    )
    parser.add_argument("--state-file", type=Path, default=STATE_FILE)
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=VALIDATION_FILE,
        help="optional JSON file receiving pipeline validation details",
    )
    return parser.parse_args()


def create_radius_spheres(simple, radial_bounds):
    """Create smooth ParaView spheres at the inner and outer radii."""
    return {
        name: simple.Sphere(
            registrationName=f"{name.title()}RadiusSphere",
            Radius=float(radius),
            ThetaResolution=SPHERE_RESOLUTION,
            PhiResolution=SPHERE_RESOLUTION,
        )
        for name, radius in zip(("inner", "outer"), radial_bounds)
    }


def build_pipeline(boundary_directory, radial_bounds):
    """Create and validate boundary readers and radial sphere sources."""
    from paraview import servermanager, simple

    boundary_directory = Path(boundary_directory).resolve()
    boundary_paths = {
        name: boundary_directory / f"chunk_3d_{name}_mesh.vts"
        for name in BOUNDARY_NAMES
    }
    missing_paths = [path for path in boundary_paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "missing boundary mesh files: " + ", ".join(map(str, missing_paths))
        )

    simple._DisableFirstRenderCameraReset()
    boundaries = {}
    validation = {
        "boundary_directory": str(boundary_directory),
        "boundaries": {},
        "spheres": {},
    }
    for boundary_name, boundary_path in boundary_paths.items():
        boundary = simple.OpenDataFile(str(boundary_path))
        simple.RenameSource(f"{boundary_name.title()}Boundary", boundary)
        boundary.UpdatePipeline()
        boundaries[boundary_name] = boundary
        boundary_data = servermanager.Fetch(boundary)
        validation["boundaries"][boundary_name] = {
            "path": str(boundary_path),
            "point_count": boundary_data.GetNumberOfPoints(),
        }

    spheres = create_radius_spheres(simple, radial_bounds)
    for sphere_name, sphere in spheres.items():
        validation["spheres"][sphere_name] = {
            "radius": sphere.Radius,
            "theta_resolution": sphere.ThetaResolution,
            "phi_resolution": sphere.PhiResolution,
            "opacity": SPHERE_OPACITY,
        }

    return boundaries, spheres, validation


def show_pipeline(simple, boundaries, spheres):
    """Show boundary meshes and transparent radial spheres in one view."""
    render_view = simple.GetActiveViewOrCreate("RenderView")
    boundary_colors = {
        "west": (0.85, 0.15, 0.15),
        "east": (0.15, 0.35, 0.85),
        "north": (0.85, 0.65, 0.15),
        "south": (0.25, 0.75, 0.35),
    }
    for boundary_name, boundary in boundaries.items():
        boundary_display = simple.Show(boundary, render_view)
        boundary_display.Representation = "Surface With Edges"
        boundary_display.DiffuseColor = boundary_colors[boundary_name]

    sphere_colors = {
        "inner": (0.35, 0.55, 0.85),
        "outer": (0.75, 0.75, 0.75),
    }
    for sphere_name, sphere in spheres.items():
        sphere_display = simple.Show(sphere, render_view)
        sphere_display.ColorArrayName = [None, ""]
        sphere_display.Representation = "Surface"
        sphere_display.DiffuseColor = sphere_colors[sphere_name]
        sphere_display.Opacity = SPHERE_OPACITY

    render_view.ResetCamera()
    return render_view


def main():
    from paraview import simple

    args = _parse_arguments()
    radial_bounds = tuple(float(radius) for radius in RADIAL_BOUNDS)
    boundaries, spheres, validation = build_pipeline(
        args.boundary_directory, radial_bounds
    )
    show_pipeline(simple, boundaries, spheres)

    args.state_file.parent.mkdir(parents=True, exist_ok=True)
    simple.SaveState(str(args.state_file.resolve()))
    validation["state_file"] = str(args.state_file.resolve())

    if args.validation_file is not None:
        args.validation_file.parent.mkdir(parents=True, exist_ok=True)
        args.validation_file.write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
