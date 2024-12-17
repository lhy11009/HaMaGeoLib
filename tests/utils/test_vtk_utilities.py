# test_vtk_utilities.py

import numpy as np
from hamageolib.utils.vtk_utilities import create_vtk_grid, calculate_resolution

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