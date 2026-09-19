"""Build a ParaView state containing chunk boundaries and radial spheres.

Run this file with ``pvpython`` after creating the four structured boundary
meshes with ``create_boundary_meshes.py``.  The configured copy embeds the
source mesh's radii and location.  Set ``LOAD_ORIGINAL_SOLUTION`` to ``True``
in that copy to load the original solution and create west/east slices.
"""

import argparse
import json
import math
from pathlib import Path

# These uppercase strings are replaced when ``create_boundary_meshes.py``
# writes a configured copy beside the generated boundary meshes.
LOAD_ORIGINAL_SOLUTION = False
SOLUTION_PATH = Path("__SOLUTION_PATH__")
BOUNDARY_DIRECTORY = Path("__BOUNDARY_DIRECTORY__")
STATE_FILE = Path("__STATE_FILE__")
VALIDATION_FILE = Path("__VALIDATION_FILE__")
RADIAL_BOUNDS = ("__INNER_RADIUS__", "__OUTER_RADIUS__")
LONGITUDE_BOUNDS = ("__LONGITUDE_MIN__", "__LONGITUDE_MAX__")
BOUNDARY_NAMES = ("west", "east", "north", "south")
SPHERE_RESOLUTION = 128
SPHERE_OPACITIES = {"inner": 1.0, "outer": 0.2}
VELOCITY_GLYPH_SCALE_FACTOR = 1.0e6
VELOCITY_COLOR_RANGE = (0.0, 1.0)
VELOCITY_COLOR_PRESETS = (
    "Blue - Green - Orange",
    "Blue Green Orange",
)
SOLID_WHITE = (1.0, 1.0, 1.0)


def boundary_longitude(boundary_name, longitude_bounds):
    """Return the longitude defining an east or west chunk boundary."""
    if boundary_name == "west":
        return min(longitude_bounds)
    if boundary_name == "east":
        return max(longitude_bounds)
    raise ValueError("boundary_name must be east or west")


def longitude_slice_normal(longitude):
    """Return a unit normal for the origin-crossing longitude plane."""
    longitude_radians = math.radians(longitude)
    return (
        -math.sin(longitude_radians),
        math.cos(longitude_radians),
        0.0,
    )


def maximum_plane_distance(points, normal, origin=(0.0, 0.0, 0.0)):
    """Return the greatest point-to-plane distance in a VTK point array."""
    if hasattr(points, "GetNumberOfTuples"):
        point_iterator = (
            points.GetTuple3(index) for index in range(points.GetNumberOfTuples())
        )
    else:
        point_iterator = points

    return max(
        abs(
            sum(
                (coordinate - origin_coordinate) * normal_component
                for coordinate, origin_coordinate, normal_component in zip(
                    point, origin, normal
                )
            )
        )
        for point in point_iterator
    )


def _number_of_points(dataset):
    """Return the total point count for a VTK dataset or composite dataset."""
    if hasattr(dataset, "GetNumberOfPoints"):
        return dataset.GetNumberOfPoints()

    point_count = 0
    iterator = dataset.NewIterator()
    iterator.InitTraversal()
    while not iterator.IsDoneWithTraversal():
        block = iterator.GetCurrentDataObject()
        if block is not None and hasattr(block, "GetNumberOfPoints"):
            point_count += block.GetNumberOfPoints()
        iterator.GoToNextItem()
    return point_count


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


def create_velocity_glyphs(simple, boundaries):
    """Create arrow glyphs oriented and scaled by boundary velocity."""
    glyphs = {}
    for boundary_name, boundary in boundaries.items():
        glyph = simple.Glyph(
            registrationName=f"{boundary_name.title()}VelocityGlyphs",
            Input=boundary,
            GlyphType="Arrow",
        )
        glyph.OrientationArray = ["POINTS", "velocity"]
        glyph.ScaleArray = ["POINTS", "velocity"]
        glyph.ScaleFactor = VELOCITY_GLYPH_SCALE_FACTOR
        glyph.GlyphMode = "All Points"
        glyphs[boundary_name] = glyph
    return glyphs


def apply_velocity_color_preset(lookup_table):
    """Apply the first Blue-Green-Orange preset supported by ParaView."""
    last_error = None
    for preset_name in VELOCITY_COLOR_PRESETS:
        try:
            lookup_table.ApplyPreset(preset_name, True)
            return preset_name
        except RuntimeError as error:
            last_error = error
    raise last_error


def build_pipeline(
    boundary_directory,
    radial_bounds,
    load_original_solution=False,
    solution_path=None,
    longitude_bounds=None,
):
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
        "glyphs": {},
        "velocity_coloring": {
            "array": "velocity",
            "component": "Magnitude",
            "preset_candidates": list(VELOCITY_COLOR_PRESETS),
            "range": list(VELOCITY_COLOR_RANGE),
        },
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
            "opacity": SPHERE_OPACITIES[sphere_name],
        }

    glyphs = create_velocity_glyphs(simple, boundaries)
    for boundary_name in glyphs:
        validation["glyphs"][boundary_name] = {
            "orientation_array": "velocity",
            "scale_array": "velocity",
            "scale_factor": VELOCITY_GLYPH_SCALE_FACTOR,
            "glyph_mode": "All Points",
        }

    solution = None
    slices = {}
    if load_original_solution:
        solution_path = Path(solution_path).resolve()
        if not solution_path.is_file():
            raise FileNotFoundError(
                f"solution file does not exist: {solution_path}"
            )
        solution = simple.OpenDataFile(str(solution_path))
        simple.RenameSource("GlobalSolution", solution)
        solution.UpdatePipeline()
        validation["solution"] = str(solution_path)
        validation["slices"] = {}
        for boundary_name in ("west", "east"):
            longitude = boundary_longitude(boundary_name, longitude_bounds)
            normal = longitude_slice_normal(longitude)
            global_slice = simple.Slice(
                registrationName=f"{boundary_name.title()}GlobalSlice",
                Input=solution,
            )
            global_slice.SliceType = "Plane"
            global_slice.SliceType.Origin = (0.0, 0.0, 0.0)
            global_slice.SliceType.Normal = normal
            global_slice.UpdatePipeline()
            slices[boundary_name] = global_slice

            boundary_data = servermanager.Fetch(boundaries[boundary_name])
            slice_data = servermanager.Fetch(global_slice)
            maximum_distance = maximum_plane_distance(
                boundary_data.GetPoints().GetData(), normal
            )
            slice_point_count = _number_of_points(slice_data)
            if slice_point_count == 0:
                raise RuntimeError(f"{boundary_name} global slice is empty")
            if maximum_distance > 1e-6:
                raise RuntimeError(
                    f"{boundary_name} boundary is {maximum_distance} m "
                    "from its slice plane"
                )
            validation["boundaries"][boundary_name][
                "maximum_plane_distance"
            ] = maximum_distance
            validation["slices"][boundary_name] = {
                "longitude": longitude,
                "origin": [0.0, 0.0, 0.0],
                "normal": list(normal),
                "point_count": slice_point_count,
            }

    return solution, boundaries, spheres, glyphs, slices, validation


def show_pipeline(
    simple, boundaries, spheres, glyphs, solution=None, slices=None
):
    """Show velocity-colored boundaries, glyphs, and radial spheres."""
    render_view = simple.GetActiveViewOrCreate("RenderView")
    velocity_lookup_table = simple.GetColorTransferFunction("velocity")
    apply_velocity_color_preset(velocity_lookup_table)
    velocity_lookup_table.RescaleTransferFunction(*VELOCITY_COLOR_RANGE)
    first_boundary_display = None
    for boundary in boundaries.values():
        boundary_display = simple.Show(boundary, render_view)
        boundary_display.Representation = "Surface"
        simple.ColorBy(
            boundary_display, ("POINTS", "velocity", "Magnitude")
        )
        if first_boundary_display is None:
            first_boundary_display = boundary_display
    if first_boundary_display is not None:
        first_boundary_display.SetScalarBarVisibility(render_view, True)

    for sphere_name, sphere in spheres.items():
        sphere_display = simple.Show(sphere, render_view)
        sphere_display.ColorArrayName = [None, ""]
        sphere_display.Representation = "Surface"
        sphere_display.DiffuseColor = SOLID_WHITE
        sphere_display.Opacity = SPHERE_OPACITIES[sphere_name]

    for glyph in glyphs.values():
        glyph_display = simple.Show(glyph, render_view)
        glyph_display.ColorArrayName = [None, ""]
        glyph_display.Representation = "Surface"
        glyph_display.DiffuseColor = SOLID_WHITE

    if solution is not None:
        solution_display = simple.Show(solution, render_view)
        solution_display.Representation = "Surface"
        solution_display.Opacity = 0.15

    if slices is not None:
        for boundary_name, global_slice in slices.items():
            slice_display = simple.Show(global_slice, render_view)
            slice_display.Representation = "Surface"
            slice_display.Opacity = 0.35
            slice_display.ColorArrayName = [None, ""]
            slice_display.DiffuseColor = SOLID_WHITE

    render_view.ResetCamera()
    return render_view


def main():
    from paraview import simple

    args = _parse_arguments()
    radial_bounds = tuple(float(radius) for radius in RADIAL_BOUNDS)
    longitude_bounds = tuple(float(value) for value in LONGITUDE_BOUNDS)
    solution, boundaries, spheres, glyphs, slices, validation = build_pipeline(
        args.boundary_directory,
        radial_bounds,
        LOAD_ORIGINAL_SOLUTION,
        SOLUTION_PATH,
        longitude_bounds,
    )
    show_pipeline(simple, boundaries, spheres, glyphs, solution, slices)

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
