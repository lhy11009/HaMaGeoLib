"""
MIT License

Copyright (c) 2025 Haoyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

# ==============================================================================
# File: geometry_utilities.py
# Author: Haoyuan Li
# Description: Utility functions for computing geometric properties in 
#              geodynamic modeling. This module includes functions for 
#              computing depth in both Cartesian and Spherical geometries.
# ==============================================================================

import numpy as np

def compute_depth(point, reference_value, is_spherical, is_2d):
    """
    Computes the depth of a point based on the provided geometry type and dimensionality.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point.
        reference_value (float): 
            - In Cartesian, this is the height of the box (H_box).
            - In Spherical, this is the outer radius (R_outer).
        is_spherical (bool): 
            - True: Compute depth in spherical geometry.
            - False: Compute depth in Cartesian geometry.
        is_2d (bool): 
            - True: Use `y` as the vertical coordinate in Cartesian geometry.
            - False: Use `z` as the vertical coordinate in Cartesian geometry.

    Returns:
        float: The computed depth of the point.

    Notes:
        - The function expects a full (x, y, z) input, even for 2D cases, to maintain compatibility with VTK files.
        - In 2D Cartesian mode, `y` is used as the vertical axis, and `z` is ignored.
        - In 3D Cartesian mode, `z` is used as the vertical axis.
        - In Spherical mode, the radial distance `r` is computed from (x, y, z).
    """

    x, y, z = point  # Unpack coordinates (always expecting 3 values)

    if is_spherical:
        # Compute radial distance
        r = np.sqrt(x**2 + y**2 + z**2)  
        return reference_value - r  # Depth = R_outer - r

    elif is_2d:
        # Compute depth using y as the vertical axis (ignore z)
        return reference_value - y  

    else:
        # Compute depth using z as the vertical axis
        return reference_value - z

def convert_to_unified_coordinates_3d(point, is_spherical):
    """
    Converts 3D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point.
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.

    Returns:
        tuple of float: Transformed coordinates (X', Y', Z') where all values represent length.

    Notes:
        - In Cartesian 3D, returns (x, y, z) unchanged.
        - In Spherical 3D, converts to (arc_length_lon, arc_length_lat, r).
    """

    x, y, z = point  # Unpack coordinates

    if is_spherical:
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)  
        lon = np.arctan2(y, x)  # Longitude
        lat = np.arcsin(z / r)  # Latitude

        # Convert to arc lengths
        arc_length_lon = r * lon
        arc_length_lat = r * lat

        return (arc_length_lon, arc_length_lat, r)

    else:
        # Cartesian 3D remains unchanged
        return (x, y, z)


def convert_to_unified_coordinates_2d(point, is_spherical):
    """
    Converts 2D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point (z is ignored in 2D).
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.

    Returns:
        tuple of float: Transformed coordinates (X', Y') where all values represent length.

    Notes:
        - In Cartesian 2D, returns (x, y), ignoring z.
        - In Spherical 2D, converts to (arc_length_lon, r), ignoring latitude arc length.
    """

    x, y, z = point  # Unpack coordinates

    if is_spherical:
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)  
        lon = np.arctan2(y, x)  # Longitude

        # Convert to arc length
        arc_length_lon = r * lon

        return (arc_length_lon, r)  # Ignore lat arc length in 2D spherical mode

    else:
        # Cartesian 2D → Ignore z
        return (x, y)  # Keep only (x, y) for 2D Cartesian
    

def convert_to_unified_coordinates_reference_3d(point, is_spherical, reference_value):
    """
    Converts 3D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point.
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.
        reference_value (float):
            - In Cartesian mode, this parameter is ignored.
            - In Spherical mode, this acts as a multiplier for longitude and latitude arc lengths.

    Returns:
        tuple of float: Transformed coordinates (X', Y', Z') where all values represent length.

    Notes:
        - In Cartesian 3D, returns (x, y, z) unchanged.
        - In Spherical 3D, converts to (arc_length_lon, arc_length_lat, reference_value).
        - The reference value ensures that points with the same longitude share the same coordinates.
    """

    x, y, z = point  # Unpack coordinates

    if is_spherical:
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)  
        lon = np.arctan2(y, x)  # Longitude
        lat = np.arcsin(z / r)  # Latitude

        # Convert to arc lengths using the reference value
        arc_length_lon = reference_value * lon
        arc_length_lat = reference_value * lat

        return (arc_length_lon, arc_length_lat, r)

    else:
        # Cartesian 3D remains unchanged
        return (x, y, z)


def convert_to_unified_coordinates_reference_2d(point, is_spherical, reference_value):
    """
    Converts 2D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point (z is ignored in 2D).
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.
        reference_value (float):
            - In Cartesian mode, this parameter is ignored.
            - In Spherical mode, this acts as a multiplier for longitude arc length.

    Returns:
        tuple of float: Transformed coordinates (X', Y') where all values represent length.

    Notes:
        - In Cartesian 2D, returns (x, y), ignoring z.
        - In Spherical 2D, converts to (arc_length_lon, reference_value), ensuring consistent coordinates.
    """

    x, y, z = point  # Unpack coordinates

    if is_spherical:
        # Convert to spherical coordinates
        r = np.sqrt(x**2 + y**2 + z**2)  
        lon = np.arctan2(y, x)  # Longitude

        # Convert to arc length using the reference value
        arc_length_lon = reference_value * lon

        return (arc_length_lon, r)  # Ignore lat arc length in 2D spherical mode

    else:
        # Cartesian 2D → Ignore z
        return (x, y)  # Keep only (x, y) for 2D Cartesian

def offset_profile(Xs, Ys, offset_distance):
    """
    Compute a new profile offset from the original (X, Y) profile
    by a given perpendicular distance.

    Args:
        Xs (array-like): Original X coordinates of the profile
        Ys (array-like): Original Y coordinates of the profile
        offset_distance (float): Distance to offset (positive = left of profile)

    Returns:
        offset_Xs (np.ndarray), offset_Ys (np.ndarray): Offset profile coordinates
    """
    Xs = np.asarray(Xs)
    Ys = np.asarray(Ys)

    # Compute segment tangents
    dx = np.gradient(Xs)
    dy = np.gradient(Ys)

    # Normalize to get unit tangent vectors
    lengths = np.sqrt(dx**2 + dy**2)
    dx /= lengths
    dy /= lengths

    # Normal vector is perpendicular (dy, -dx)
    nx = -dy
    ny = dx

    # Apply offset
    offset_Xs = Xs + offset_distance * nx
    offset_Ys = Ys + offset_distance * ny

    return offset_Xs, offset_Ys


def compute_pairwise_distances(X0, Y0, X1, Y1):
    """
    Compute the pairwise Euclidean distances between two 2D profiles.

    Parameters:
    - X0, Y0: arrays of shape (N0,) for profile 0
    - X1, Y1: arrays of shape (N1,) for profile 1

    Returns:
    - distances: array of shape (N0, N1), distances from each point in profile 0 to each in profile 1
    """
    points0 = np.stack((X0, Y0), axis=-1)  # Shape (N0, 2)
    points1 = np.stack((X1, Y1), axis=-1)  # Shape (N1, 2)

    deltas = points0[:, np.newaxis, :] - points1[np.newaxis, :, :]  # Shape (N0, N1, 2)
    distances = np.linalg.norm(deltas, axis=2)  # Shape (N0, N1)

    return distances


def cartesian_to_spherical(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, theta, phi