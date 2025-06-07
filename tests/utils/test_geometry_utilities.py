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


@pytest.mark.parametrize(
    "point, is_spherical, expected",
    [
        # Cartesian 3D cases (should return unchanged values)
        ((10, 20, 30), False, (10, 20, 30)),
        ((-5, 15, 50), False, (-5, 15, 50)),

        # Spherical 3D cases (should return arc length longitude, latitude, and r)
        ((10, 20, 30), True, (
            np.sqrt(10**2 + 20**2 + 30**2) * np.arctan2(20, 10),
            np.sqrt(10**2 + 20**2 + 30**2) * np.arcsin(30 / np.sqrt(10**2 + 20**2 + 30**2)),
            np.sqrt(10**2 + 20**2 + 30**2)  # ✅ Use r instead of reference_value
        )),
    ]
)
def test_convert_to_unified_coordinates_3d(point, is_spherical, expected):
    """
    Test convert_to_unified_coordinates_3d for both Cartesian and Spherical cases.
    """
    computed = convert_to_unified_coordinates_3d(point, is_spherical)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, expected",
    [
        # Cartesian 2D cases (should ignore z and return x, y)
        ((10, 20, 30), False, (10, 20)),
        ((-5, 15, 100), False, (-5, 15)),

        # Spherical 2D cases (should return arc length longitude and r)
        ((10, 20, 30), True, (
            np.sqrt(10**2 + 20**2 + 30**2) * np.arctan2(20, 10),
            np.sqrt(10**2 + 20**2 + 30**2)  # ✅ Use r instead of reference_value
        )),
    ]
)
def test_convert_to_unified_coordinates_2d(point, is_spherical, expected):
    """
    Test convert_to_unified_coordinates_2d for both Cartesian and Spherical cases.
    """
    computed = convert_to_unified_coordinates_2d(point, is_spherical)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, reference_value, expected",
    [
        # Cartesian 3D cases (should return unchanged values)
        ((10, 20, 30), False, 1000, (10, 20, 30)),
        ((-5, 15, 50), False, 2000, (-5, 15, 50)),

        # Spherical 3D cases (should return scaled arc length longitude and latitude, but actual r)
        ((10, 20, 30), True, 1000, (
            1000 * np.arctan2(20, 10),
            1000 * np.arcsin(30 / np.sqrt(10**2 + 20**2 + 30**2)),
            np.sqrt(10**2 + 20**2 + 30**2)  # ✅ Use actual r
        )),
    ]
)
def test_convert_to_unified_coordinates_reference_3d(point, is_spherical, reference_value, expected):
    """
    Test convert_to_unified_coordinates_reference_3d for both Cartesian and Spherical cases.
    """
    computed = convert_to_unified_coordinates_reference_3d(point, is_spherical, reference_value)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"


@pytest.mark.parametrize(
    "point, is_spherical, reference_value, expected",
    [
        # Cartesian 2D cases (should ignore z and return x, y)
        ((10, 20, 30), False, 500, (10, 20)),
        ((-5, 15, 100), False, 800, (-5, 15)),

        # Spherical 2D cases (should return scaled arc length longitude and actual r)
        ((10, 20, 30), True, 6371, (
            6371 * np.arctan2(20, 10),
            np.sqrt(10**2 + 20**2 + 30**2)  # ✅ Use actual r
        )),
    ]
)

def test_convert_to_unified_coordinates_reference_2d(point, is_spherical, reference_value, expected):
    """
    Test convert_to_unified_coordinates_reference_2d for both Cartesian and Spherical cases.
    """
    computed = convert_to_unified_coordinates_reference_2d(point, is_spherical, reference_value)
    assert all(pytest.approx(c, rel=1e-5) == e for c, e in zip(computed, expected)), f"Failed for {point}"

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
