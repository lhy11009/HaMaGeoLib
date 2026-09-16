"""Create and sample VTK structured-grid meshes for a spherical chunk.

The four meshes use the same sampling layout as ``extract_local_velocity.py``:
west and east vary in radius and latitude, while north and south vary in
radius and longitude.  Mesh geometry can be written by itself or populated
with velocity sampled from a model solution.
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import vtk


DEFAULT_OUTPUT_DIRECTORY = Path(
    "/mnt/lochy/ASPECT_DATA/Collision0/collision_test/S20RTS/"
    "output-S20RTS/regional_velocity_files"
)
RADIUS_BOUNDS = (4_760e3, 6_360e3)
LATITUDE_BOUNDS = (-55.0, -20.0)
LONGITUDE_BOUNDS = (150.0, 210.0)
RADIUS_SPACING = 100e3
LATERAL_SPACING = 2.5
VISUALIZATION_SCRIPT_NAME = "visualize_boundary_meshes.py"


def report_progress(message):
    """Print a timestamped progress message and flush it immediately."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def uniform_coordinates(minimum, maximum, spacing):
    """Return uniformly spaced coordinates including both exact bounds."""
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")

    number_of_intervals = (maximum - minimum) / spacing
    rounded_intervals = round(number_of_intervals)
    if not np.isclose(number_of_intervals, rounded_intervals):
        raise ValueError(
            "spacing must divide the interval evenly: "
            f"spacing={spacing}, interval=[{minimum}, {maximum}]"
        )

    return np.linspace(minimum, maximum, rounded_intervals + 1)


def spherical_to_cartesian(radius, latitude, longitude):
    """Convert radius (m), latitude (degrees), and longitude (degrees)."""
    latitude_radians = np.deg2rad(latitude)
    longitude_radians = np.deg2rad(longitude)
    x = radius * np.cos(latitude_radians) * np.cos(longitude_radians)
    y = radius * np.cos(latitude_radians) * np.sin(longitude_radians)
    z = radius * np.sin(latitude_radians)
    return x, y, z


def create_boundary_grid(
    boundary_name,
    radius_bounds,
    latitude_bounds,
    longitude_bounds,
    radius_spacing,
    lateral_spacing,
):
    """Create one connected VTK structured surface for a chunk boundary."""
    if boundary_name not in {"west", "east", "north", "south"}:
        raise ValueError(f"unknown boundary name: {boundary_name}")

    radii = uniform_coordinates(*radius_bounds, spacing=radius_spacing)
    if boundary_name in {"west", "east"}:
        # Match extract_local_velocity.py: latitude decreases within each slice.
        lateral_coordinates = uniform_coordinates(
            *latitude_bounds, spacing=lateral_spacing
        )[::-1]
    else:
        lateral_coordinates = uniform_coordinates(
            *longitude_bounds, spacing=lateral_spacing
        )

    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for lateral_coordinate in lateral_coordinates:
        for radius in radii:
            if boundary_name == "west":
                latitude = lateral_coordinate
                longitude = min(longitude_bounds)
            elif boundary_name == "east":
                latitude = lateral_coordinate
                longitude = max(longitude_bounds)
            elif boundary_name == "north":
                latitude = max(latitude_bounds)
                longitude = lateral_coordinate
            else:
                latitude = min(latitude_bounds)
                longitude = lateral_coordinate

            points.InsertNextPoint(
                spherical_to_cartesian(radius, latitude, longitude)
            )

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(len(radii), len(lateral_coordinates), 1)
    grid.SetPoints(points)
    return grid


def interpolate_velocity(source_dataset, boundary_grid):
    """Interpolate a Cartesian velocity vector onto a boundary grid.

    The vector is probed once so all components use the same source-cell
    lookup and validity mask.  The returned grid retains the three-component
    ``velocity`` array and also contains scalar ``Vx``, ``Vy``, and
    ``Vz`` arrays.  Neither input dataset is modified.

    Let M be the number of source-mesh cells and N the number of boundary-mesh
    vertices.  Building the spatial locator and probing the points generally
    costs O(M + N log M), though an efficient locator often gives near
    O(M + N) behavior in practice.  Splitting the resulting vector into three
    scalar arrays costs O(N).  Probing the vector once avoids repeating the
    source-cell search separately for Vx, Vy, and Vz.
    """
    if isinstance(source_dataset, vtk.vtkCompositeDataSet):
        probe = vtk.vtkCompositeDataProbeFilter()
    else:
        probe = vtk.vtkProbeFilter()
    probe.SetInputData(boundary_grid)
    probe.SetSourceData(source_dataset)
    probe.ComputeToleranceOn()
    probe.Update()

    interpolated_grid = vtk.vtkStructuredGrid()
    interpolated_grid.DeepCopy(probe.GetOutput())
    point_data = interpolated_grid.GetPointData()

    valid_point_mask = point_data.GetArray("vtkValidPointMask")
    if valid_point_mask is not None:
        invalid_point_count = sum(
            valid_point_mask.GetTuple1(index) == 0
            for index in range(valid_point_mask.GetNumberOfTuples())
        )
        if invalid_point_count:
            raise ValueError(
                f"{invalid_point_count} boundary points lie outside the source mesh"
            )

    velocity = point_data.GetArray("velocity")
    if velocity is None:
        raise ValueError("source mesh does not provide point-data array 'velocity'")
    if velocity.GetNumberOfComponents() != 3:
        raise ValueError("'velocity' must have exactly three components")

    for component_index, component_name in enumerate(("Vx", "Vy", "Vz")):
        component = vtk.vtkDoubleArray()
        component.SetName(component_name)
        component.SetNumberOfTuples(velocity.GetNumberOfTuples())
        for point_index in range(velocity.GetNumberOfTuples()):
            component.SetValue(
                point_index, velocity.GetComponent(point_index, component_index)
            )
        point_data.AddArray(component)

    return interpolated_grid


def _write_boundary_grid(grid, output_path):
    """Write one structured boundary grid and report write failures."""
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(output_path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise OSError(f"failed to write {output_path}")


def write_boundary_meshes(
    output_directory,
    radius_bounds,
    latitude_bounds,
    longitude_bounds,
    radius_spacing,
    lateral_spacing,
):
    """Write west, east, north, and south meshes as VTK XML grids."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    for boundary_name in ("west", "east", "north", "south"):
        report_progress(f"Creating {boundary_name} boundary grid")
        grid = create_boundary_grid(
            boundary_name,
            radius_bounds,
            latitude_bounds,
            longitude_bounds,
            radius_spacing,
            lateral_spacing,
        )
        output_path = output_directory / f"chunk_3d_{boundary_name}_mesh.vts"
        report_progress(f"Writing {boundary_name} boundary mesh: {output_path}")
        _write_boundary_grid(grid, output_path)
        output_paths[boundary_name] = output_path
        report_progress(f"Finished {boundary_name} boundary")

    return output_paths


def write_boundary_velocity_meshes(
    output_directory,
    source_dataset,
    radius_bounds,
    latitude_bounds,
    longitude_bounds,
    radius_spacing,
    lateral_spacing,
):
    """Create four boundary grids, sample velocity, and write VTK XML grids."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = {}

    for boundary_name in ("west", "east", "north", "south"):
        report_progress(f"Creating {boundary_name} boundary grid")
        boundary_grid = create_boundary_grid(
            boundary_name,
            radius_bounds,
            latitude_bounds,
            longitude_bounds,
            radius_spacing,
            lateral_spacing,
        )
        report_progress(
            f"Interpolating velocity onto {boundary_name} boundary"
        )
        interpolated_grid = interpolate_velocity(source_dataset, boundary_grid)
        output_path = output_directory / f"chunk_3d_{boundary_name}_mesh.vts"
        report_progress(f"Writing {boundary_name} boundary mesh: {output_path}")
        _write_boundary_grid(interpolated_grid, output_path)
        output_paths[boundary_name] = output_path
        report_progress(f"Finished {boundary_name} boundary")

    return output_paths


def read_solution_dataset(solution_path):
    """Read a VTK or ParaView solution file, including PVD collections."""
    import pyvista as pv

    return pv.read(solution_path)


def write_visualization_script(output_directory, solution_path):
    """Write a self-contained ParaView GUI script beside the boundary meshes."""
    output_directory = Path(output_directory).resolve()
    solution_path = Path(solution_path).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    template_path = Path(__file__).with_name(VISUALIZATION_SCRIPT_NAME)
    configured_path = output_directory / VISUALIZATION_SCRIPT_NAME
    configured_values = {
        "__SOLUTION_PATH__": solution_path,
        "__BOUNDARY_DIRECTORY__": output_directory,
        "__STATE_FILE__": output_directory / "boundary_meshes.pvsm",
        "__VALIDATION_FILE__": output_directory
        / "boundary_meshes_validation.json",
    }

    configured_script = template_path.read_text(encoding="utf-8")
    for placeholder, configured_value in configured_values.items():
        quoted_placeholder = f'"{placeholder}"'
        if configured_script.count(quoted_placeholder) != 1:
            raise ValueError(
                f"visualization template must contain {quoted_placeholder} exactly once"
            )
        configured_script = configured_script.replace(
            quoted_placeholder, repr(str(configured_value))
        )

    configured_path.write_text(configured_script, encoding="utf-8")
    return configured_path


def main():
    parser = argparse.ArgumentParser(
        description="Write the four structured boundary meshes for a spherical chunk."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="directory in which to write the four .vts files",
    )
    parser.add_argument(
        "--solution",
        type=Path,
        help="optional VTK/PVD solution whose velocity is sampled onto the meshes",
    )
    args = parser.parse_args()

    writer = write_boundary_meshes
    writer_arguments = ()
    if args.solution is not None:
        report_progress(f"Loading source solution: {args.solution}")
        writer = write_boundary_velocity_meshes
        writer_arguments = (read_solution_dataset(args.solution),)
        report_progress("Finished loading source solution")

    report_progress("Starting boundary mesh processing")
    output_paths = writer(
        args.output_directory,
        *writer_arguments,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        RADIUS_SPACING,
        LATERAL_SPACING,
    )
    for output_path in output_paths.values():
        report_progress(f"Created boundary mesh: {output_path}")
    if args.solution is not None:
        report_progress("Generating configured ParaView visualization script")
        visualization_script = write_visualization_script(
            args.output_directory, args.solution
        )
        report_progress(f"Created visualization script: {visualization_script}")
    report_progress("Finished boundary mesh processing")


if __name__ == "__main__":
    main()
