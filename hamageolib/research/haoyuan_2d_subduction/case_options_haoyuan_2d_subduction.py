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
import pandas as pd

from ...utils import case_options as C_Options

# todo_2d
class CASE_OPTIONS(C_Options.CASE_OPTIONS):
    """
    A class to substitute existing code with specific values and parameters for geodynamic modeling cases.

    Attributes:
        attributs from C_Options.CASE_OPTIONS
    """
    def __init__(self, case_dir):
        # call init function of C_Options.CASE_OPTIONS
        C_Options.CASE_OPTIONS.__init__(case_dir)

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