"""
Unit tests for geometry_utilities.py

These tests verify the correctness of the compute_depth function for different
geometric configurations (Cartesian 3D, Cartesian 2D, and Spherical).
"""

import pytest
import numpy as np
from hamageolib.utils.geometry_utilities import *

@pytest.mark.parametrize(
    "point, reference_value, is_spherical, is_2d, expected",
    [
        # Cartesian 3D cases (depth = H_box - z)
        ((10, 20, 30), 100, False, False, 70),  # 100 - 30 = 70
        ((-5, 15, 50), 200, False, False, 150), # 200 - 50 = 150
        ((0, 0, 0), 50, False, False, 50),      # 50 - 0 = 50

        # Cartesian 2D cases (depth = H_box - y, z ignored)
        ((10, 20, 0), 50, False, True, 30),  # 50 - 20 = 30
        ((-5, 15, 10), 100, False, True, 85), # 100 - 15 = 85
        ((0, 0, 100), 200, False, True, 200), # 200 - 0 = 200

        # Spherical cases (depth = R_outer - r)
        ((0, 0, 6371), 6371, True, False, 0),  # Surface point (Earth)
        ((0, 0, 6370), 6371, True, False, 1),  # 1 km depth
        ((100, 200, 300), 7000, True, False, 7000 - np.sqrt(100**2 + 200**2 + 300**2)),  # Generic point
    ]
)
def test_compute_depth(point, reference_value, is_spherical, is_2d, expected):
    """
    Test compute_depth for various Cartesian and Spherical cases.
    """
    computed_depth = compute_depth(point, reference_value, is_spherical, is_2d)
    assert pytest.approx(computed_depth, rel=1e-5) == expected, f"Failed for {point}"


# ---- Precompute grid test cases ----
# Cartesian grid: shape (2, 2, 3) with (x, y, z) on the last axis
grid_cart = np.array([
    [[10., 20., 30.],  [-5., 15., 50.]],
    [[ 0.,  0.,  0.],  [ 1.,  2.,  3.]],
])
exp_cart_L0 = grid_cart[..., 2]  # z
exp_cart_L1 = grid_cart[..., 0]  # x
exp_cart_L2 = grid_cart[..., 1]  # y

# Spherical grid: shape (2, 2, 3) with (x, y, z)
grid_sph = np.array([
    [[1., 0., 0.],  [0., 1., 0.]],
    [[0., 0., 1.],  [1., 1., 1.]],
])
x, y, z = grid_sph[..., 0], grid_sph[..., 1], grid_sph[..., 2]
r = np.sqrt(x**2 + y**2 + z**2)
lon = np.arctan2(y, x)
lat = np.arcsin(np.divide(z, r, out=np.zeros_like(r), where=r != 0.0))

# For scaled=True (default): L0=r, L1=r*lon, L2=r*lat
exp_sph_L0, exp_sph_L1, exp_sph_L2 = r, r * lon, r * lat

@pytest.mark.parametrize(
    "point, is_spherical, expected, scaled",
    [
        # Cartesian 3D cases (should return unchanged values)
        ((10, 20, 30), False, (30, 10, 20), True),
        ((-5, 15, 50), False, (50, -5, 15), True),
        ((5054338.38061481, 3878569.90838686, 0.), True, (6371e3, 0.654528, 0.0), False), # with earth's parameters

        # Spherical 3D cases (should return arc-length lon, lat, and r)
        ((10, 20, 30), True, (
            np.sqrt(10**2 + 20**2 + 30**2),
            np.sqrt(10**2 + 20**2 + 30**2) * np.arctan2(20, 10),
            np.sqrt(10**2 + 20**2 + 30**2) * np.arcsin(30 / np.sqrt(10**2 + 20**2 + 30**2)),
        ), True),

        # NEW: Cartesian grid (2x2)
        (grid_cart, False, (exp_cart_L0, exp_cart_L1, exp_cart_L2), True),

        # NEW: Spherical grid (2x2), scaled=True (default)
        (grid_sph, True, (exp_sph_L0, exp_sph_L1, exp_sph_L2), True),
    ]
)
def test_points2unified3(point, is_spherical, expected, scaled):
    """
    Test points2unified3 for both Cartesian and Spherical cases.
    """
    computed = points2unified3(point, is_spherical, scaled)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)),\
        f"Failed for {point}, is_spherical {is_spherical}, scaled {scaled}, expected {expected}, computed {computed}"


@pytest.mark.parametrize(
    "point, is_spherical, reference_value, expected",
    [
        # Cartesian 3D cases (should return unchanged values)
        ((10, 20, 30), False, 1000, (30, 10, 20)),
        ((-5, 15, 50), False, 2000, (50, -5, 15)),

        # Spherical 3D cases (should return scaled arc length longitude and latitude, but actual r)
        ((10, 20, 30), True, 1000, (
            np.sqrt(10**2 + 20**2 + 30**2),  # ✅ Use actual r
            1000 * np.arctan2(20, 10),
            1000 * np.arcsin(30 / np.sqrt(10**2 + 20**2 + 30**2))
        )),
    ]
)
def points2unified3_reference_r(point, is_spherical, reference_value, expected):
    """
    Test convert_to_unified_coordinates_reference_3d for both Cartesian and Spherical cases.
    """
    computed = points2unified3(point, is_spherical, r0=reference_value)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, L",
    [
        # Cartesian 3D cases (unified: (z, x, y))
        ((10, 20, 30), False, (30, 10, 20)),
        ((-5, 15, 50), False, (50, -5, 15)),

        # Spherical 3D cases (unified: (r, r*lon, r*lat) with scaled=True)
        ((10, 20, 30), True, (
            np.sqrt(10**2 + 20**2 + 30**2),  # r
            np.sqrt(10**2 + 20**2 + 30**2) * np.arctan2(20, 10),  # r*lon
            np.sqrt(10**2 + 20**2 + 30**2) * np.arcsin(30 / np.sqrt(10**2 + 20**2 + 30**2))  # r*lat
        )),
    ]
)
def test_unified2points3(point, is_spherical, L):
    """
    Test unified2points3 for both Cartesian and Spherical cases.
    Uses the same dataset as the forward test (scaled=True).
    """
    computed = unified2points3(L, is_spherical, scaled=True)
    assert all(pytest.approx(c, rel=1e-5) == p for c, p in zip(computed, point)), f"Failed for L={L}, geom={'spherical' if is_spherical else 'cartesian'}"


@pytest.mark.parametrize(
    "point, is_spherical, expected",
    [
        # Cartesian 2D cases (should ignore z and return x, y)
        ((10, 20, 30), False, (20, 10)),
        ((-5, 15, 100), False, (15, -5)),

        # Spherical 2D cases (should return arc length longitude and r)
        ((10, 20, 30), True, (
            np.sqrt(10**2 + 20**2),  # ✅ Use r instead of reference_value
            np.sqrt(10**2 + 20**2) * np.arctan2(20, 10)
        )),
    ]
)
def test_points2unified2(point, is_spherical, expected):
    """
    Test points2unified2 for both Cartesian and Spherical cases.
    """
    computed = points2unified2(point, is_spherical)
    print(computed) # debug
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, reference_value, expected",
    [
        # Cartesian 2D cases (should ignore z and return x, y)
        ((10, 20, 30), False, 500, (20, 10)),
        ((-5, 15, 100), False, 800, (15, -5)),

        # Spherical 2D cases (should return scaled arc length longitude and actual r)
        ((10, 20, 30), True, 6371, (
            np.sqrt(10**2 + 20**2),  # ✅ Use actual r
            6371 * np.arctan2(20, 10)
        )),
    ]
)
def test_points2unified2_reference_r(point, is_spherical, reference_value, expected):
    """
    Test convert_to_unified_coordinates_reference_2d for both Cartesian and Spherical cases.
    """
    computed = points2unified2(point, is_spherical, r0=reference_value)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, L",
    [
        # Cartesian 2D cases (unified: (y, x); z ignored)
        ((10, 20, 30),   False, (20, 10)),
        ((-5, 15, 100),  False, (15, -5)),

        # Spherical 2D cases (unified: (r, r*lon) with scaled=True)
        ((10, 20, 30),   True, (
            np.sqrt(10**2 + 20**2),                        # r (from x,y)
            np.sqrt(10**2 + 20**2) * np.arctan2(20, 10)    # r*lon
        )),
    ]
)
def test_unified2points2(point, is_spherical, L):
    """
    Test unified2points2 for both Cartesian and Spherical 2D cases.
    Uses the same dataset as the forward test (scaled=True).
    """
    computed = unified2points2(L, is_spherical, scaled=True)
    # Expect to recover x,y; z should be 0.0 in 2D inverse
    assert pytest.approx(computed[0], rel=1e-5) == point[0]
    assert pytest.approx(computed[1], rel=1e-5) == point[1]
    assert computed[2] == 0.0


def test_profiles_with_different_lengths():
    # Profile 0 has 3 points
    X0 = np.array([0.0, 1.0, 2.0])
    Y0 = np.array([0.0, 0.0, 0.0])

    # Profile 1 has 2 points
    X1 = np.array([0.0, 0.0])
    Y1 = np.array([1.0, 2.0])

    distances = compute_pairwise_distances(X0, Y0, X1, Y1)

    # Expected shape: (3, 2) => 3 points in profile 0, 2 points in profile 1
    assert distances.shape == (3, 2)

    # Expected values
    expected = np.array([
        [1.0, 2.0],
        [np.sqrt(2), np.sqrt(5)],
        [np.sqrt(5), np.sqrt(8)]
    ])
    assert np.allclose(distances, expected)

    print("Test with different profile lengths passed!")
