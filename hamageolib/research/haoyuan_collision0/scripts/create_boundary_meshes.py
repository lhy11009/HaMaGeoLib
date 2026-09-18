"""Create and sample VTK structured-grid meshes for a spherical chunk.

The four meshes use the same sampling layout as ``extract_local_velocity.py``:
west and east vary in radius and latitude, while north and south vary in
radius and longitude.  Mesh geometry can be written by itself or populated
with velocity sampled from a model solution.
"""

import argparse
import configparser
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import vtk


VISUALIZATION_SCRIPT_NAME = "visualize_boundary_meshes.py"


@dataclass(frozen=True)
class BoundaryMeshConfig:
    """Validated runtime inputs for boundary mesh generation."""

    solution: Path
    output_directory: Path
    radius_bounds: tuple
    latitude_bounds: tuple
    longitude_bounds: tuple
    radius_spacing: float
    lateral_spacing: float
    retry_tolerance: float
    use_nearest_valid_point: bool


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


def _required_config_value(parser, section, key):
    """Return a required configuration string with a concise error."""
    try:
        return parser[section][key]
    except KeyError as error:
        raise ValueError(
            f"missing required configuration value [{section}] {key}"
        ) from error


def _required_config_float(parser, section, key):
    """Return a required finite floating-point configuration value."""
    value = _required_config_value(parser, section, key)
    try:
        number = float(value)
    except ValueError as error:
        raise ValueError(
            f"configuration value [{section}] {key} must be a number"
        ) from error
    if not np.isfinite(number):
        raise ValueError(
            f"configuration value [{section}] {key} must be a finite number"
        )
    return number


def _config_boolean(parser, section, key, fallback=False):
    """Return a boolean configuration value or its fallback when omitted."""
    if not parser.has_option(section, key):
        return fallback
    value = parser[section][key].lower()
    if value not in parser.BOOLEAN_STATES:
        raise ValueError(
            f"configuration value [{section}] {key} must be a boolean"
        )
    return parser.BOOLEAN_STATES[value]


def load_boundary_mesh_config(config_path):
    """Load and validate all runtime inputs from an INI-style text file."""
    config_path = Path(config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")

    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    solution = Path(
        _required_config_value(parser, "paths", "solution")
    ).expanduser().resolve()
    output_directory = Path(
        _required_config_value(parser, "paths", "output_directory")
    ).expanduser().resolve()
    radius_bounds = (
        _required_config_float(parser, "geometry", "radius_min"),
        _required_config_float(parser, "geometry", "radius_max"),
    )
    latitude_bounds = (
        _required_config_float(parser, "geometry", "latitude_min"),
        _required_config_float(parser, "geometry", "latitude_max"),
    )
    longitude_bounds = (
        _required_config_float(parser, "geometry", "longitude_min"),
        _required_config_float(parser, "geometry", "longitude_max"),
    )
    radius_spacing = _required_config_float(
        parser, "resolution", "radius_spacing"
    )
    lateral_spacing = _required_config_float(
        parser, "resolution", "lateral_spacing"
    )
    retry_tolerance = _required_config_float(
        parser, "interpolation", "retry_tolerance"
    )
    use_nearest_valid_point = _config_boolean(
        parser, "interpolation", "use_nearest_valid_point"
    )

    for bounds_name, bounds in (
        ("radius", radius_bounds),
        ("latitude", latitude_bounds),
        ("longitude", longitude_bounds),
    ):
        if bounds[1] <= bounds[0]:
            raise ValueError(
                f"{bounds_name}_max must be greater than {bounds_name}_min"
            )
    if radius_spacing <= 0:
        raise ValueError("radius_spacing must be positive")
    if lateral_spacing <= 0:
        raise ValueError("lateral_spacing must be positive")
    if retry_tolerance <= 0:
        raise ValueError("retry_tolerance must be positive")

    uniform_coordinates(*radius_bounds, spacing=radius_spacing)
    uniform_coordinates(*latitude_bounds, spacing=lateral_spacing)
    uniform_coordinates(*longitude_bounds, spacing=lateral_spacing)
    return BoundaryMeshConfig(
        solution,
        output_directory,
        radius_bounds,
        latitude_bounds,
        longitude_bounds,
        radius_spacing,
        lateral_spacing,
        retry_tolerance,
        use_nearest_valid_point,
    )


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


def _write_invalid_points(
    boundary_grid,
    invalid_point_indices,
    output_path,
    retry_tolerance=None,
    retry_succeeded=None,
    nearest_point_fallback_used=None,
    nearest_point_indices=None,
    nearest_point_distances=None,
):
    """Write invalid boundary-point indices and Cartesian coordinates to CSV."""
    if retry_succeeded is None:
        retry_succeeded = [False] * len(invalid_point_indices)
    if nearest_point_fallback_used is None:
        nearest_point_fallback_used = [False] * len(invalid_point_indices)
    if nearest_point_indices is None:
        nearest_point_indices = [None] * len(invalid_point_indices)
    if nearest_point_distances is None:
        nearest_point_distances = [None] * len(invalid_point_indices)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "point_index",
                "x",
                "y",
                "z",
                "retry_tolerance",
                "retry_succeeded",
                "nearest_point_fallback_used",
                "nearest_point_index",
                "nearest_point_distance",
            )
        )
        tolerance_value = "" if retry_tolerance is None else retry_tolerance
        for point_index, succeeded, fallback_used, nearest_index, distance in zip(
            invalid_point_indices,
            retry_succeeded,
            nearest_point_fallback_used,
            nearest_point_indices,
            nearest_point_distances,
        ):
            writer.writerow(
                (
                    point_index,
                    *boundary_grid.GetPoint(point_index),
                    tolerance_value,
                    str(succeeded).lower(),
                    str(fallback_used).lower(),
                    "" if nearest_index is None else nearest_index,
                    "" if distance is None else distance,
                )
            )


def _probe_dataset(source_dataset, input_dataset, tolerance=None):
    """Probe an input dataset, optionally using an explicit tolerance."""
    if isinstance(source_dataset, vtk.vtkCompositeDataSet):
        probe = vtk.vtkCompositeDataProbeFilter()
    else:
        probe = vtk.vtkProbeFilter()
    probe.SetInputData(input_dataset)
    probe.SetSourceData(source_dataset)
    if tolerance is None:
        probe.ComputeToleranceOn()
    else:
        probe.ComputeToleranceOff()
        probe.SetTolerance(tolerance)
    probe.Update()

    output = probe.GetOutput().NewInstance()
    output.DeepCopy(probe.GetOutput())
    return output


def _point_subset(dataset, point_indices):
    """Return a point-only dataset containing the selected input points."""
    points = vtk.vtkPoints()
    points.SetDataTypeToDouble()
    for point_index in point_indices:
        points.InsertNextPoint(dataset.GetPoint(point_index))
    subset = vtk.vtkPolyData()
    subset.SetPoints(points)
    return subset


def interpolate_velocity(
    source_dataset,
    boundary_grid,
    invalid_points_path=None,
    retry_tolerance=None,
    use_nearest_valid_point=False,
):
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
    interpolated_grid = _probe_dataset(source_dataset, boundary_grid)
    point_data = interpolated_grid.GetPointData()

    valid_point_mask = point_data.GetArray("vtkValidPointMask")
    invalid_point_indices = []
    if valid_point_mask is not None:
        invalid_point_indices = [
            index
            for index in range(valid_point_mask.GetNumberOfTuples())
            if valid_point_mask.GetTuple1(index) == 0
        ]

    retry_succeeded = [False] * len(invalid_point_indices)
    if invalid_point_indices and retry_tolerance is not None:
        report_progress(
            f"Retrying {len(invalid_point_indices)} invalid boundary points "
            f"with tolerance {retry_tolerance} m"
        )
        retry_output = _probe_dataset(
            source_dataset,
            _point_subset(boundary_grid, invalid_point_indices),
            retry_tolerance,
        )
        retry_point_data = retry_output.GetPointData()
        retry_valid_mask = retry_point_data.GetArray("vtkValidPointMask")
        retry_velocity = retry_point_data.GetArray("velocity")
        velocity = point_data.GetArray("velocity")
        if retry_velocity is None or velocity is None:
            raise ValueError(
                "source mesh does not provide point-data array 'velocity'"
            )
        if retry_velocity.GetNumberOfComponents() != 3:
            raise ValueError("'velocity' must have exactly three components")
        if retry_valid_mask is not None:
            for retry_index, point_index in enumerate(invalid_point_indices):
                if retry_valid_mask.GetTuple1(retry_index) != 0:
                    velocity.SetTuple(
                        point_index, retry_velocity.GetTuple(retry_index)
                    )
                    valid_point_mask.SetTuple1(point_index, 1)
                    retry_succeeded[retry_index] = True
        report_progress(
            f"Recovered {sum(retry_succeeded)} of "
            f"{len(invalid_point_indices)} invalid boundary points"
        )

    nearest_point_fallback_used = [False] * len(invalid_point_indices)
    nearest_point_indices = [None] * len(invalid_point_indices)
    nearest_point_distances = [None] * len(invalid_point_indices)
    fallback_error = None
    unresolved_positions = [
        position
        for position, succeeded in enumerate(retry_succeeded)
        if not succeeded
    ]
    if unresolved_positions and use_nearest_valid_point:
        valid_indices = [
            point_index
            for point_index in range(valid_point_mask.GetNumberOfTuples())
            if valid_point_mask.GetTuple1(point_index) != 0
        ]
        if not valid_indices:
            fallback_error = (
                "nearest-point fallback cannot run because there are no valid "
                "boundary points"
            )
        else:
            report_progress(
                f"Applying nearest-valid-point fallback to "
                f"{len(unresolved_positions)} boundary points"
            )
            velocity = point_data.GetArray("velocity")
            if velocity is None:
                raise ValueError(
                    "source mesh does not provide point-data array 'velocity'"
                )
            if velocity.GetNumberOfComponents() != 3:
                raise ValueError("'velocity' must have exactly three components")
            valid_coordinates = np.array(
                [boundary_grid.GetPoint(index) for index in valid_indices]
            )
            for position in unresolved_positions:
                point_index = invalid_point_indices[position]
                point = np.array(boundary_grid.GetPoint(point_index))
                distances = np.linalg.norm(valid_coordinates - point, axis=1)
                nearest_position = int(np.argmin(distances))
                nearest_index = valid_indices[nearest_position]
                velocity.SetTuple(point_index, velocity.GetTuple(nearest_index))
                valid_point_mask.SetTuple1(point_index, 1)
                nearest_point_fallback_used[position] = True
                nearest_point_indices[position] = nearest_index
                nearest_point_distances[position] = distances[nearest_position]
            report_progress(
                f"Recovered {len(unresolved_positions)} boundary points with "
                "nearest-valid-point fallback"
            )

    if invalid_point_indices:
        diagnostic_message = ""
        if invalid_points_path is not None:
            _write_invalid_points(
                boundary_grid,
                invalid_point_indices,
                invalid_points_path,
                retry_tolerance,
                retry_succeeded,
                nearest_point_fallback_used,
                nearest_point_indices,
                nearest_point_distances,
            )
            report_progress(
                f"Wrote {len(invalid_point_indices)} invalid boundary points: "
                f"{invalid_points_path}"
            )
            diagnostic_message = f"; coordinates written to {invalid_points_path}"
        remaining_invalid_count = sum(
            not retry_success and not fallback_success
            for retry_success, fallback_success in zip(
                retry_succeeded, nearest_point_fallback_used
            )
        )
        if remaining_invalid_count:
            if fallback_error is not None:
                raise ValueError(f"{fallback_error}{diagnostic_message}")
            if retry_tolerance is None:
                raise ValueError(
                    f"{remaining_invalid_count} boundary points lie outside the "
                    f"source mesh{diagnostic_message}"
                )
            raise ValueError(
                f"{remaining_invalid_count} boundary points remain outside the "
                f"source mesh after retry{diagnostic_message}"
            )

    if invalid_points_path is not None and not invalid_point_indices:
        Path(invalid_points_path).unlink(missing_ok=True)

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
    retry_tolerance=None,
    use_nearest_valid_point=False,
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
        invalid_points_path = (
            output_directory
            / f"chunk_3d_{boundary_name}_invalid_points.csv"
        )
        interpolated_grid = interpolate_velocity(
            source_dataset,
            boundary_grid,
            invalid_points_path,
            retry_tolerance,
            use_nearest_valid_point,
        )
        output_path = output_directory / f"chunk_3d_{boundary_name}_mesh.vts"
        report_progress(f"Writing {boundary_name} boundary mesh: {output_path}")
        _write_boundary_grid(interpolated_grid, output_path)
        output_paths[boundary_name] = output_path
        report_progress(f"Finished {boundary_name} boundary")

    return output_paths


def _read_xml_unstructured_grid(solution_path):
    """Read PVTU/VTU geometry and velocity without unrelated data arrays."""
    reader_types = {
        ".pvtu": vtk.vtkXMLPUnstructuredGridReader,
        ".vtu": vtk.vtkXMLUnstructuredGridReader,
    }
    reader = reader_types[solution_path.suffix.lower()]()
    reader.SetFileName(str(solution_path))

    report_progress(f"Reading VTK metadata: {solution_path}")
    reader.UpdateInformation()
    point_arrays = {
        reader.GetPointArrayName(index)
        for index in range(reader.GetNumberOfPointArrays())
    }
    if "velocity" not in point_arrays:
        raise ValueError("source mesh does not provide point-data array 'velocity'")

    reader.GetPointDataArraySelection().DisableAllArrays()
    reader.GetPointDataArraySelection().EnableArray("velocity")
    reader.GetCellDataArraySelection().DisableAllArrays()
    report_progress("Loading mesh geometry and velocity only")
    reader.Update()
    return reader.GetOutput()


def read_solution_dataset(solution_path):
    """Read a VTK or ParaView solution file, including PVD collections."""
    solution_path = Path(solution_path)
    if solution_path.suffix.lower() in {".pvtu", ".vtu"}:
        return _read_xml_unstructured_grid(solution_path)

    import pyvista as pv

    report_progress(
        f"Loading all arrays with the PyVista fallback: {solution_path}"
    )
    return pv.read(solution_path)


def report_solution_bounds(source_dataset):
    """Print the Cartesian bounds of a VTK dataset in metres."""
    if isinstance(source_dataset, vtk.vtkCompositeDataSet):
        bounds = [0.0] * 6
        source_dataset.GetBounds(bounds)
    else:
        bounds = source_dataset.GetBounds()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    report_progress(
        "Source solution bounds: "
        f"x=[{x_min:.6e}, {x_max:.6e}] m, "
        f"y=[{y_min:.6e}, {y_max:.6e}] m, "
        f"z=[{z_min:.6e}, {z_max:.6e}] m"
    )


def _dataset_blocks(dataset):
    """Yield point-containing blocks from a dataset or composite dataset."""
    if not isinstance(dataset, vtk.vtkCompositeDataSet):
        yield dataset
        return

    iterator = dataset.NewIterator()
    iterator.InitTraversal()
    while not iterator.IsDoneWithTraversal():
        block = iterator.GetCurrentDataObject()
        if block is not None and hasattr(block, "GetNumberOfPoints"):
            yield block
        iterator.GoToNextItem()


def source_radial_bounds(source_dataset):
    """Return minimum and maximum source-mesh vertex radii in metres."""
    minimum_radius = float("inf")
    maximum_radius = 0.0
    point_count = 0
    for block in _dataset_blocks(source_dataset):
        for point_index in range(block.GetNumberOfPoints()):
            x, y, z = block.GetPoint(point_index)
            radius = np.sqrt(x * x + y * y + z * z)
            minimum_radius = min(minimum_radius, radius)
            maximum_radius = max(maximum_radius, radius)
            point_count += 1
    if point_count == 0:
        raise ValueError("source mesh does not contain any points")
    return float(minimum_radius), float(maximum_radius)


def write_visualization_script(
    output_directory, solution_path, longitude_bounds, radial_bounds
):
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

    configured_path.write_text(configured_script, encoding="utf-8")
    return configured_path


def main():
    parser = argparse.ArgumentParser(
        description="Write the four structured boundary meshes for a spherical chunk."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_boundary_mesh_config(args.config)

    report_progress(f"Configuration file: {args.config.resolve()}")
    report_progress(f"Source solution: {config.solution}")
    report_progress(f"Output directory: {config.output_directory}")
    report_progress(f"Radius bounds: {config.radius_bounds} m")
    report_progress(f"Latitude bounds: {config.latitude_bounds} degrees")
    report_progress(f"Longitude bounds: {config.longitude_bounds} degrees")
    report_progress(f"Radius spacing: {config.radius_spacing} m")
    report_progress(f"Lateral spacing: {config.lateral_spacing} degrees")
    report_progress(f"Invalid-point retry tolerance: {config.retry_tolerance} m")
    report_progress(
        "Use nearest valid point after retry: "
        f"{config.use_nearest_valid_point}"
    )
    if not config.solution.is_file():
        raise FileNotFoundError(
            f"configured solution file does not exist: {config.solution}"
        )

    report_progress(f"Loading source solution: {config.solution}")
    source_dataset = read_solution_dataset(config.solution)
    report_progress("Finished loading source solution")
    report_solution_bounds(source_dataset)
    radial_bounds = source_radial_bounds(source_dataset)
    report_progress(
        "Source solution radial bounds: "
        f"r=[{radial_bounds[0]:.6e}, {radial_bounds[1]:.6e}] m"
    )

    report_progress("Starting boundary mesh processing")
    output_paths = write_boundary_velocity_meshes(
        config.output_directory,
        source_dataset,
        config.radius_bounds,
        config.latitude_bounds,
        config.longitude_bounds,
        config.radius_spacing,
        config.lateral_spacing,
        config.retry_tolerance,
        config.use_nearest_valid_point,
    )
    for output_path in output_paths.values():
        report_progress(f"Created boundary mesh: {output_path}")
    report_progress("Generating configured ParaView visualization script")
    visualization_script = write_visualization_script(
        config.output_directory,
        config.solution,
        config.longitude_bounds,
        radial_bounds,
    )
    report_progress(f"Created visualization script: {visualization_script}")
    report_progress("Finished boundary mesh processing")


if __name__ == "__main__":
    main()
