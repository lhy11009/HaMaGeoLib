"""Create VTK structured-grid meshes for a regional spherical chunk.

The four meshes use the same sampling layout as ``extract_local_velocity.py``:
west and east vary in radius and latitude, while north and south vary in
radius and longitude.  Only mesh geometry is written; no model fields are
sampled by this script.
"""

import argparse
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
        grid = create_boundary_grid(
            boundary_name,
            radius_bounds,
            latitude_bounds,
            longitude_bounds,
            radius_spacing,
            lateral_spacing,
        )
        output_path = output_directory / f"chunk_3d_{boundary_name}_mesh.vts"
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(str(output_path))
        writer.SetInputData(grid)
        if writer.Write() != 1:
            raise OSError(f"failed to write {output_path}")
        output_paths[boundary_name] = output_path

    return output_paths


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
    args = parser.parse_args()

    output_paths = write_boundary_meshes(
        args.output_directory,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        RADIUS_SPACING,
        LATERAL_SPACING,
    )
    for output_path in output_paths.values():
        print(output_path)


if __name__ == "__main__":
    main()
