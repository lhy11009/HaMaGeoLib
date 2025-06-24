# test_vtk_utilities.py

import numpy as np
import pyvista as pv
import pytest
import os
from shutil import rmtree  # for remove directories
from hamageolib.utils.vtk_utilities import *

# ---------------------------------------------------------------------
# Check and make test directories
# ---------------------------------------------------------------------
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "vtk_utilities"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)


def test_single_triangle():
    """
    Test with a single triangular cell.
    """
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])  # 3 points
    cells = [[0, 1, 2]]  # Single triangle cell

    grid = create_vtk_grid(points, cells)
    resolution = calculate_resolution(grid)

    # Expected minimum pairwise distances: 
    # sqrt(1^2) = 1, sqrt(1^2) = 1, and sqrt(2) ~ 1.414
    assert np.isclose(resolution[0], 1.0)  # First point
    assert np.isclose(resolution[1], 1.0)  # Second point
    assert np.isclose(resolution[2], 1.0)  # Third point

def test_two_disconnected_lines():
    """
    Test with two disconnected line cells.
    """
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    cells = [[0, 1], [2, 3]]  # Two separate lines

    grid = create_vtk_grid(points, cells)
    resolution = calculate_resolution(grid)

    # Expected distances: 1.0 for connected points, infinity for isolated points
    assert np.isclose(resolution[0], 1.0)
    assert np.isclose(resolution[1], 1.0)
    assert np.isclose(resolution[2], 1.0)
    assert np.isclose(resolution[3], 1.0)

def test_empty_grid():
    """
    Test with an empty grid.
    """
    points = np.array([]).reshape(0, 3)
    cells = []

    grid = create_vtk_grid(points, cells)
    resolution = calculate_resolution(grid)

    assert len(resolution) == 0  # No points, so resolution should be empty.

def test_single_point():
    """
    Test with a grid containing a single point.
    """
    points = np.array([[0, 0, 0]])
    cells = []

    grid = create_vtk_grid(points, cells)
    resolution = calculate_resolution(grid)

    assert resolution[0] == float('inf')  # Single point has no resolution.

def test_multiple_triangles():
    """
    Test with two connected triangular cells.
    """
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    cells = [[0, 1, 2], [1, 2, 3]]  # Two triangles sharing an edge

    grid = create_vtk_grid(points, cells)
    resolution = calculate_resolution(grid)

    # Expected pairwise minimum distances: edges of length 1
    assert np.isclose(resolution[0], 1.0)
    assert np.isclose(resolution[1], 1.0)
    assert np.isclose(resolution[2], 1.0)
    assert np.isclose(resolution[3], 1.0)

@pytest.fixture
def synthetic_points():
    r_vals = np.linspace(0.1, 1.0, 20)
    theta_vals = np.linspace(0, np.pi / 2, 30)
    phi_vals = np.linspace(0, 2 * np.pi, 50)

    r_grid, theta_grid, phi_grid = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')

    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T


@pytest.fixture
def asymmetric_phi_points():
    # Create a thin spherical shell with φ varying with r and θ
    r_vals = np.linspace(0.5, 1.0, 10)
    theta_vals = np.linspace(0, np.pi / 2, 50)

    r_grid, theta_grid = np.meshgrid(r_vals, theta_vals, indexing='ij')

    # Nontrivial phi pattern
    phi_grid = np.pi * (0.2 + 0.5 * np.sin(theta_grid)**2 * np.cos(r_grid * 3 * np.pi))

    # Convert to Cartesian
    x = r_grid * np.sin(theta_grid) * np.cos(phi_grid)
    y = r_grid * np.sin(theta_grid) * np.sin(phi_grid)
    z = r_grid * np.cos(theta_grid)

    return np.vstack((x.ravel(), y.ravel(), z.ravel())).T
