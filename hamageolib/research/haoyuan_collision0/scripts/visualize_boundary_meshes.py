"""Build a ParaView state containing a global solution and chunk boundaries.

Run this file with ``pvpython`` after creating the four structured boundary
meshes with ``create_boundary_meshes.py``.  In addition to loading the global
solution and boundary meshes, the script slices the global solution at the
exact constant-longitude planes used by the west and east boundary meshes.
"""

import argparse
import json
import math
from pathlib import Path

try:
    from .create_boundary_meshes import LONGITUDE_BOUNDS
except ImportError:  # Direct execution with pvpython places this folder on sys.path.
    from create_boundary_meshes import LONGITUDE_BOUNDS


BOUNDARY_NAMES = ("west", "east", "north", "south")


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
            "Load a global solution and chunk boundary meshes, create matching "
            "east/west slices, and save the ParaView state."
        )
    )
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--boundary-directory", type=Path, required=True)
    parser.add_argument("--state-file", type=Path, required=True)
    parser.add_argument(
        "--validation-file",
        type=Path,
        help="optional JSON file receiving pipeline validation details",
    )
    return parser.parse_args()


def build_pipeline(solution_path, boundary_directory):
    """Create and validate the ParaView readers and longitude slices."""
    from paraview import servermanager, simple

    solution_path = Path(solution_path).resolve()
    boundary_directory = Path(boundary_directory).resolve()
    if not solution_path.is_file():
        raise FileNotFoundError(f"solution file does not exist: {solution_path}")

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
    solution = simple.OpenDataFile(str(solution_path))
    simple.RenameSource("GlobalSolution", solution)
    solution.UpdatePipeline()

    boundaries = {}
    for boundary_name, boundary_path in boundary_paths.items():
        boundary = simple.OpenDataFile(str(boundary_path))
        simple.RenameSource(f"{boundary_name.title()}Boundary", boundary)
        boundary.UpdatePipeline()
        boundaries[boundary_name] = boundary

    slices = {}
    validation = {
        "solution": str(solution_path),
        "boundary_directory": str(boundary_directory),
        "boundaries": {},
        "slices": {},
    }
    for boundary_name in ("west", "east"):
        longitude = boundary_longitude(boundary_name, LONGITUDE_BOUNDS)
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
        boundary_points = boundary_data.GetPoints().GetData()
        maximum_distance = maximum_plane_distance(boundary_points, normal)
        slice_point_count = _number_of_points(slice_data)
        if slice_point_count == 0:
            raise RuntimeError(f"{boundary_name} global slice is empty")
        if maximum_distance > 1e-6:
            raise RuntimeError(
                f"{boundary_name} boundary is {maximum_distance} m from its slice plane"
            )

        validation["boundaries"][boundary_name] = {
            "path": str(boundary_paths[boundary_name]),
            "point_count": boundary_data.GetNumberOfPoints(),
            "maximum_plane_distance": maximum_distance,
        }
        validation["slices"][boundary_name] = {
            "longitude": longitude,
            "origin": [0.0, 0.0, 0.0],
            "normal": list(normal),
            "point_count": slice_point_count,
        }

    for boundary_name in ("north", "south"):
        boundary_data = servermanager.Fetch(boundaries[boundary_name])
        validation["boundaries"][boundary_name] = {
            "path": str(boundary_paths[boundary_name]),
            "point_count": boundary_data.GetNumberOfPoints(),
        }

    return solution, boundaries, slices, validation


def show_pipeline(solution, boundaries, slices):
    """Show every pipeline object in one render view for the saved state."""
    from paraview import simple

    render_view = simple.GetActiveViewOrCreate("RenderView")
    solution_display = simple.Show(solution, render_view)
    solution_display.Representation = "Surface"
    solution_display.Opacity = 0.15

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

    for boundary_name, global_slice in slices.items():
        slice_display = simple.Show(global_slice, render_view)
        slice_display.Representation = "Surface"
        slice_display.Opacity = 0.35
        slice_display.DiffuseColor = boundary_colors[boundary_name]

    render_view.ResetCamera()
    return render_view


def main():
    from paraview import simple

    args = _parse_arguments()
    solution, boundaries, slices, validation = build_pipeline(
        args.solution, args.boundary_directory
    )
    show_pipeline(solution, boundaries, slices)

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
