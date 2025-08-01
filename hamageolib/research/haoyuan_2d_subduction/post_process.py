"""
case_options_haoyuan.py

Author: Haoyuan Li
License: MIT

Description:
    This module provides research-specific utilities for managing and processing 
    simulation cases in the Haoyuan 2D subduction research project. It includes 
    functions for handling case metadata, locating input files, and managing 
    time step tracking.

Usage:
    from hamageolib.research.haoyuan_2d_subduction.case_options_haoyuan import (
        CASE_SUMMARY, find_case_files
    )

Contents:
    - CASE_SUMMARY: Class for storing and managing case metadata.
    - find_case_files: Function to locate relevant input files in a case directory.
"""

import os
import sys
import pandas as pd
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy

# include package root
path_ = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if path_ not in sys.path:
    sys.path.append(path_)

from hamageolib.utils import case_options as C_Options
from hamageolib.utils.geometry_utilities import convert_to_unified_coordinates_2d


# todo_2d
class CASE_SUMMARY(C_Options.CASE_SUMMARY):
    """
    Manages case metadata, including file paths, simulation parameters, and time step tracking.

    Attributes:
        df (pandas.DataFrame): Stores case metadata with the following columns
    """

    def __init__(self):
        # call init function of C_Options.CASE_SUMMARY
        C_Options.CASE_SUMMARY.__init__()

# todo_2d
def extract_slab_outline_binned_unified(slab_polydata, is_spherical, **kwargs):
    """
    Extracts the slab outline using depth binning, converting coordinates to a unified system first.

    Parameters:
        slab_polydata (vtk.vtkPolyData): Polydata containing slab points.
        is_spherical (bool): Whether the input coordinates are in spherical geometry.
        bin_size (float): Depth interval for binning (default: 10 km).

    Returns:
        vtk.vtkPolyData: Polydata containing the slab outline points.
    """
    bin_size = kwargs.get("bin_size", 10e3)

    points = slab_polydata.GetPoints()
    num_points = slab_polydata.GetNumberOfPoints()

    if num_points == 0:
        return None  # Return None if no points exist

    # Convert VTK points to unified coordinates
    coords = vtk_to_numpy(points.GetData()).reshape(-1, 3)  # Reshape into Nx3 array
    unified_coords = np.array([convert_to_unified_coordinates_2d(points.GetPoint(i), is_spherical)
                               for i in range(num_points)])

    # Extract lateral (X') and vertical (Y') coordinates
    lateral_values = unified_coords[:, 0]
    vertical_values = unified_coords[:, 1]

    # Define bin ranges
    vertical_min, vertical_max = np.min(vertical_values), np.max(vertical_values)
    bins = np.arange(vertical_min, vertical_max + bin_size, bin_size)

    # Storage for outline points
    outline_points_0 = []
    outline_points_1 = []

    for z_start in bins[:-1]:
        z_end = z_start + bin_size

        # Select points within the vertical interval
        mask = (vertical_values >= z_start) & (vertical_values < z_end)
        bin_points = coords[mask]
        bin_points_unified = unified_coords[mask]

        if bin_points.shape[0] == 0:
            continue  # Skip empty bins

        # Find min/max lateral values within this bin
        min_idx = np.argmin(bin_points_unified[:, 0])  # Min lateral value
        max_idx = np.argmax(bin_points_unified[:, 0])  # Max lateral value

        # Store the min and max lateral points
        outline_points_0.append(bin_points[min_idx])
        outline_points_1.append(bin_points[max_idx])

    # Convert outline points to vtkPolyData
    vtk_outline_points_0 = vtk.vtkPoints()
    vtk_outline_points_1 = vtk.vtkPoints()
    vtk_outline_polydata_0 = vtk.vtkPolyData()
    vtk_outline_polydata_1 = vtk.vtkPolyData()

    for pt in outline_points_0:
        vtk_outline_points_0.InsertNextPoint(pt[0], pt[1], 0.0)  # Keep z = 0 for visualization
    
    for pt in outline_points_1:
        vtk_outline_points_1.InsertNextPoint(pt[0], pt[1], 0.0)  # Keep z = 0 for visualization

    vtk_outline_polydata_0.SetPoints(vtk_outline_points_0)
    vtk_outline_polydata_1.SetPoints(vtk_outline_points_1)

    return vtk_outline_polydata_0, vtk_outline_polydata_1  # Return the outline polydata