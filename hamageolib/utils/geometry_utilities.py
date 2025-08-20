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
from types import SimpleNamespace
import hamageolib.utils.nump_utilities as np_util

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


def points2unified3(point, is_spherical, scaled=True, **kwargs):
    """
    Converts 3D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point.
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.
        scaled (bool):
            - True: return arc length in lon and lat direction
            - False: return lon and lat
        kwargs:
            r0:
                - None (default): use r for scale of longitude and latitude
                - float number: use the given value for scale of longitude and latitude

    Returns:
        tuple of float: Transformed coordinates (L0, L1, L2) where all values represent length.
            L0, L1, L2 represents the 1st, 2nd, 3rd dimenstion in model setup.
            In cartesian, these are z, x, y.
            In spherical, there are r, arc along longtitude, arc along latitude

    Notes:
        - In Cartesian 3D, returns (z, x, y)
        - In Spherical 3D, converts to (r, arc_length_lon, arc_length_lat).
    """
    r0 = kwargs.get("r0", None)

    x, y, z = np_util.unpack_array(point, 3)

    if is_spherical:
        # Ensure array-like for safe vectorized math
        x = np.asarray(x)
        y = np.asarray(y)
        z = np.asarray(z)

        r = np.sqrt(x**2 + y**2 + z**2)
        zero = (r == 0)

        # Angles
        lon = np.arctan2(y, x)
        # Avoid divide-by-zero and guard arcsin domain with clip
        with np.errstate(divide='ignore', invalid='ignore'):
            lat = np.where(zero, 0.0, np.arcsin(z /r))

        if scaled:
            scale = r if (r0 is None) else np.asarray(r0)
            arc_length_lon = scale * lon
            arc_length_lat = scale * lat

            # If inputs were pure scalars, return pure scalars for backward compatibility
            if r.ndim == 0:
                return (float(r), float(arc_length_lon), float(arc_length_lat))
            return (r, arc_length_lon, arc_length_lat)
        else:
            if r.ndim == 0:
                return (float(r), float(lon), float(lat))
            else:
                return (r, lon, lat)
    else:
        # Cartesian 3D remains unchanged
        return (z, x, y)
    

def unified2points3(L, is_spherical, scaled=True, **kwargs):
    """
    Inverse of points2unified3.

    Parameters:
        L (tuple of float): Unified coordinates (L0, L1, L2).
            - Cartesian geometry:   L0=z, L1=x, L2=y
            - Spherical geometry:   L0=r, L1=(arc_lon or lon), L2=(arc_lat or lat)
        is_spherical (bool):
            - True: interpret L as spherical-unified and convert to Cartesian (x,y,z).
            - False: interpret L as Cartesian-unified and convert to Cartesian (x,y,z).
        scaled (bool):
            - True: L1, L2 are arc lengths (lon*base, lat*base)
            - False: L1, L2 are angles in radians (lon, lat)
        kwargs:
            r0 (float or None):
                - None (default): base = r (L0) for scaling lon/lat when scaled=True
                - float: base = r0 for both lon and lat when scaled=True

    Returns:
        tuple of float: (x, y, z) in Cartesian coordinates.
    """
    r0 = kwargs.get("r0", None)
    L0, L1, L2 = np_util.unpack_array(L, 3)

    if is_spherical:

        L0 = np.asarray(L0, dtype=float)
        L1 = np.asarray(L1, dtype=float)
        L2 = np.asarray(L2, dtype=float)

        r = L0
        if scaled:
            # If r0 is given, use it as the scale (can be scalar or array); otherwise use r
            base = r if (r0 is None) else np.asarray(r0, dtype=float)
            lon = np.where(base == 0.0, 0.0, L1 / base)
            lat = np.where(base == 0.0, 0.0, L2 / base)
        else:
            lon = L1
            lat = L2

        coslat = np.cos(lat)
        x = r * coslat * np.cos(lon)
        y = r * coslat * np.sin(lon)
        z = r * np.sin(lat)

        # Preserve scalar returns when inputs are scalars
        if np.ndim(x) == 0:
            return (float(x), float(y), float(z))
        else:
            return (x, y, z)
    else:
        # Cartesian unified: (L0=z, L1=x, L2=y) → (x,y,z)
        return (L1, L2, L0)



def points2unified2(point, is_spherical, scaled=True, **kwargs):
    """
    Converts 2D Cartesian or Spherical coordinates into a unified coordinate system where 
    all dimensions have length interpretations.

    Parameters:
        point (tuple of float): The (x, y, z) coordinates of the point (z is ignored in 2D).
        is_spherical (bool): 
            - True: Convert from spherical geometry.
            - False: Treat input as Cartesian coordinates.
        scaled (bool):
            - True: return arc length in lon and lat direction
        kwargs:
            r0:
                - None (default): use r for scale of longitude and latitude
                - float number: use the given value for scale of longitude and latitude

    Returns:
        tuple of float: Transformed coordinates (X', Y') where all values represent length.

    Notes:
        - In Cartesian 2D, returns (y, x), ignoring z.
        - In Spherical 2D, converts to (r, arc_length_lon), ignoring latitude arc length.
    """
    r0 = kwargs.get("r0", None)

    x, y, _ = np_util.unpack_array(point, 3)

    if is_spherical:
        # Convert to spherical coordinates
        # Ensure array-like for safe vectorized math
        r = np.sqrt(x**2 + y**2)
        lon = np.arctan2(y, x)  # Longitude

        if scaled:
            if r0 is None:
                # Convert to arc length
                arc_length_lon = r * lon
            else:
                # Convert to arc length
                arc_length_lon = r0 * lon
            return (r, arc_length_lon)  # Ignore lat arc length in 2D spherical mode
        else:
            return (r, lon)  # Ignore lat arc length in 2D spherical mode
    else:
        # Cartesian 2D → Ignore z
        return (y, x)  # Keep only (x, y) for 2D Cartesian


def unified2points2(L, is_spherical, scaled=True, **kwargs):
    """
    Inverse of points2unified2.

    Parameters:
        L (tuple of float):
            - Cartesian: (y, x)
            - Spherical: (r, arc_lon) if scaled=True, or (r, lon) if scaled=False
        is_spherical (bool):
            - True: interpret L as spherical-unified and convert to Cartesian (x,y,0).
            - False: interpret L as Cartesian-unified and convert to Cartesian (x,y,0).
        scaled (bool):
            - True: L[1] is arc length (lon * base), with base=r or r0
            - False: L[1] is angle in radians (lon)
        kwargs:
            r0 (float or None):
                - None (default): base = r for scaling lon when scaled=True
                - float: base = r0 for scaling lon when scaled=True

    Returns:
        tuple of float: (x, y, z) in Cartesian coordinates with z=0.0.
    """
    r0 = kwargs.get("r0", None)
    L0, L1 = np_util.unpack_array(L, 2)

    if is_spherical:
        # make as array
        L0 = np.asarray(L0, dtype=float)
        L1 = np.asarray(L1, dtype=float)
        
        r = L0

        if scaled:
            # If r0 is given, use it as the scale (can be scalar or array); otherwise use r
            base = r if (r0 is None) else np.asarray(r0, dtype=float)
            lon = np.where(base == 0.0, 0.0, L1 / base)
        else:
            lon = L1

        x = r * np.cos(lon)
        y = r * np.sin(lon)
        
        # Preserve scalar returns when inputs are scalars
        if np.ndim(x) == 0:
            return (float(x), float(y), 0.0)
        else:
            return (x, y, 0.0)
    else:
        # Cartesian unified (y, x) -> (x, y, 0)
        return (L1, L0, 0.0)


# put functions in a simple name space 
PUnified = SimpleNamespace()
PUnified.points2unified3 = points2unified3
PUnified.points2unified2 = points2unified2
PUnified.unified2points2 = unified2points2
PUnified.unified2points3 = unified2points3


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

def cartesian_to_spherical_2d(x, y):
    """
    Convert 2D Cartesian coordinates (x, y) to polar coordinates (r, theta).

    Parameters:
        x (float or np.ndarray): x-coordinate(s)
        y (float or np.ndarray): y-coordinate(s)

    Returns:
        r (same type as input): radial distance(s)
        phi (same type as input): angle(s) in radians, in range [0, 2π)
    """
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x) % (2 * np.pi)
    return r, phi


def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z