import numpy as np
import pytest

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    create_boundary_grid,
)
from hamageolib.research.haoyuan_collision0.scripts.visualize_boundary_meshes import (
    boundary_longitude,
    longitude_slice_normal,
    maximum_plane_distance,
)


RADIUS_BOUNDS = (4_760e3, 6_360e3)
LATITUDE_BOUNDS = (-55.0, -20.0)
LONGITUDE_BOUNDS = (150.0, 210.0)


@pytest.mark.parametrize(
    ("boundary_name", "expected_longitude"),
    [("west", 150.0), ("east", 210.0)],
)
def test_boundary_longitude_selects_exact_chunk_boundaries(
    boundary_name, expected_longitude
):
    assert boundary_longitude(boundary_name, LONGITUDE_BOUNDS) == expected_longitude


def test_boundary_longitude_rejects_non_longitudinal_boundary():
    with pytest.raises(ValueError, match="east or west"):
        boundary_longitude("north", LONGITUDE_BOUNDS)


@pytest.mark.parametrize(
    ("longitude", "expected_normal"),
    [
        (0.0, (0.0, 1.0, 0.0)),
        (90.0, (-1.0, 0.0, 0.0)),
        (150.0, (-0.5, -np.sqrt(3.0) / 2.0, 0.0)),
        (210.0, (0.5, -np.sqrt(3.0) / 2.0, 0.0)),
    ],
)
def test_longitude_slice_normal(longitude, expected_normal):
    assert longitude_slice_normal(longitude) == pytest.approx(expected_normal)


@pytest.mark.parametrize("boundary_name", ["west", "east"])
def test_boundary_mesh_points_lie_on_matching_longitude_plane(boundary_name):
    grid = create_boundary_grid(
        boundary_name,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )
    longitude = boundary_longitude(boundary_name, LONGITUDE_BOUNDS)

    distance = maximum_plane_distance(
        grid.GetPoints().GetData(), longitude_slice_normal(longitude)
    )

    assert distance < 1e-8
