"""Project boundary velocities into outward-normal and tangential parts.

This processing stage reads only ``boundary_mesh_metadata.json`` and the four
structured boundary meshes referenced by it.  It does not load the original
solution dataset.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


BOUNDARY_NAMES = ("west", "east", "north", "south")
PROJECTION_ARRAYS = {
    "normal": "boundary_normal",
    "normal_velocity": "velocity_normal",
    "parallel_velocity": "velocity_parallel",
    "signed_normal_velocity": "velocity_normal_signed",
}


def report_progress(message):
    """Print a timestamped progress message and flush it immediately."""
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[{timestamp}] {message}", flush=True)


def boundary_outward_normals(points, boundary_name):
    """Return outward unit normals for one spherical-chunk side boundary."""
    if boundary_name not in BOUNDARY_NAMES:
        raise ValueError(f"unknown boundary name: {boundary_name}")

    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    radii = np.linalg.norm(points, axis=1)
    cylindrical_radii = np.linalg.norm(points[:, :2], axis=1)
    if np.any(radii == 0.0):
        raise ValueError("boundary points must not lie at the coordinate origin")
    if np.any(cylindrical_radii == 0.0):
        raise ValueError("boundary normals are undefined at a geographic pole")

    if boundary_name in {"west", "east"}:
        normals = np.column_stack(
            (
                -points[:, 1] / cylindrical_radii,
                points[:, 0] / cylindrical_radii,
                np.zeros(len(points)),
            )
        )
        if boundary_name == "west":
            normals *= -1.0
    else:
        normals = np.column_stack(
            (
                -points[:, 2] * points[:, 0]
                / (radii * cylindrical_radii),
                -points[:, 2] * points[:, 1]
                / (radii * cylindrical_radii),
                cylindrical_radii / radii,
            )
        )
        if boundary_name == "south":
            normals *= -1.0
    return normals


def project_velocity(points, velocity, boundary_name):
    """Split Cartesian velocity into normal and boundary-parallel vectors."""
    velocity = np.asarray(velocity, dtype=float)
    points = np.asarray(points, dtype=float)
    if velocity.shape != points.shape:
        raise ValueError("velocity and points must have the same (N, 3) shape")

    normals = boundary_outward_normals(points, boundary_name)
    signed_normal_velocity = np.einsum("ij,ij->i", velocity, normals)
    normal_velocity = signed_normal_velocity[:, None] * normals
    parallel_velocity = velocity - normal_velocity
    return (
        normals,
        signed_normal_velocity,
        normal_velocity,
        parallel_velocity,
    )


def _read_boundary_grid(mesh_path):
    """Read and detach one VTK XML structured grid."""
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    if reader.GetErrorCode() != vtk.vtkErrorCode.NoError:
        raise OSError(f"failed to read boundary mesh: {mesh_path}")
    grid = vtk.vtkStructuredGrid()
    grid.DeepCopy(reader.GetOutput())
    return grid


def _write_boundary_grid(grid, mesh_path):
    """Overwrite one generated boundary mesh with projected arrays."""
    writer = vtk.vtkXMLStructuredGridWriter()
    writer.SetFileName(str(mesh_path))
    writer.SetInputData(grid)
    if writer.Write() != 1:
        raise OSError(f"failed to write boundary mesh: {mesh_path}")


def _add_array(point_data, name, values):
    """Add or replace a NumPy array in VTK point data."""
    point_data.RemoveArray(name)
    vtk_array = numpy_to_vtk(np.asarray(values), deep=True)
    vtk_array.SetName(name)
    point_data.AddArray(vtk_array)


def _resolve_mesh_path(metadata_path, configured_path):
    mesh_path = Path(configured_path).expanduser()
    if not mesh_path.is_absolute():
        mesh_path = metadata_path.parent / mesh_path
    return mesh_path.resolve()


def project_boundary_meshes(metadata_path):
    """Project velocity arrays in the four metadata-referenced meshes."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata file does not exist: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    try:
        configured_meshes = metadata["boundary_meshes"]
    except KeyError as error:
        raise ValueError("metadata is missing boundary_meshes") from error
    missing_boundaries = [
        name for name in BOUNDARY_NAMES if name not in configured_meshes
    ]
    if missing_boundaries:
        raise ValueError(
            "metadata is missing boundary meshes: "
            + ", ".join(missing_boundaries)
        )

    grids = {}
    projection = {
        "arrays": PROJECTION_ARRAYS,
        "normal_convention": "outward from the spherical chunk",
        "boundaries": {},
    }
    for boundary_name in BOUNDARY_NAMES:
        mesh_path = _resolve_mesh_path(
            metadata_path, configured_meshes[boundary_name]
        )
        if not mesh_path.is_file():
            raise FileNotFoundError(
                f"{boundary_name} boundary mesh does not exist: {mesh_path}"
            )
        report_progress(f"Reading {boundary_name} boundary mesh: {mesh_path}")
        grid = _read_boundary_grid(mesh_path)
        point_data = grid.GetPointData()
        velocity_array = point_data.GetArray("velocity")
        if velocity_array is None:
            raise ValueError(
                f"{boundary_name} boundary mesh has no velocity point array"
            )
        if velocity_array.GetNumberOfComponents() != 3:
            raise ValueError(
                f"{boundary_name} boundary velocity must have three components"
            )

        points = vtk_to_numpy(grid.GetPoints().GetData())
        velocity = vtk_to_numpy(velocity_array)
        normals, signed, normal_velocity, parallel_velocity = project_velocity(
            points, velocity, boundary_name
        )
        for name, values in (
            (PROJECTION_ARRAYS["normal"], normals),
            (PROJECTION_ARRAYS["normal_velocity"], normal_velocity),
            (PROJECTION_ARRAYS["parallel_velocity"], parallel_velocity),
            (PROJECTION_ARRAYS["signed_normal_velocity"], signed),
        ):
            _add_array(point_data, name, values)

        reconstruction_error = np.linalg.norm(
            velocity - normal_velocity - parallel_velocity, axis=1
        )
        parallel_normal_dot = np.abs(
            np.einsum("ij,ij->i", parallel_velocity, normals)
        )
        normal_norm_error = np.abs(np.linalg.norm(normals, axis=1) - 1.0)
        projection["boundaries"][boundary_name] = {
            "mesh": str(mesh_path),
            "point_count": int(grid.GetNumberOfPoints()),
            "maximum_reconstruction_error": float(
                np.max(reconstruction_error, initial=0.0)
            ),
            "maximum_parallel_normal_dot": float(
                np.max(parallel_normal_dot, initial=0.0)
            ),
            "maximum_normal_norm_error": float(
                np.max(normal_norm_error, initial=0.0)
            ),
        }
        grids[boundary_name] = (grid, mesh_path)

    for boundary_name, (grid, mesh_path) in grids.items():
        report_progress(
            f"Writing projected {boundary_name} boundary mesh: {mesh_path}"
        )
        _write_boundary_grid(grid, mesh_path)

    metadata["velocity_projection"] = projection
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_progress(f"Updated projection metadata: {metadata_path}")
    return projection


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Project velocities in four spherical-chunk boundary meshes into "
            "outward-normal and boundary-parallel components."
        )
    )
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    project_boundary_meshes(args.metadata)
    report_progress("Finished boundary velocity projection")


if __name__ == "__main__":
    main()
