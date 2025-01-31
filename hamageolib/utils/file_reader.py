# =============================================================================
# MIT License
# 
# Copyright (c) 2024 Haoyuan Li
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================

"""
Module: file_reader.py

Author: Haoyuan Li

Purpose:
    This module is part of the HaMaGeoLib package, located under the `utils` submodule.
    It provides functions and classes to read and parse specific file types used
    in geodynamic modeling workflows, such as parameter files, output data files,
    and mesh formats.

Functionalities:
    - File reading utilities for various formats (e.g., .prm, .dat, .txt).
    - Parsing structured data for further processing or visualization.

Usage:
    Import this module to handle file input within geodynamic workflows.
    Example:
        ```python
        from hamageolib.utils.file_reader import read_parameter_file
        
        parameters = read_parameter_file("model_input.prm")
        ```

Dependencies:
    - List required Python modules (e.g., NumPy, pandas).
    - Ensure compatibility with the broader HaMaGeoLib package.

"""

# Add your imports here
import numpy as np
import os
import pandas as pd
import re


def parse_header_from_lines(lines):
    """
    Parses the structured header from the provided lines and extracts column names and their units.

    Args:
        lines (list): List of lines from the file.

    Returns:
        tuple:
            - list: A list of cleaned column names extracted from the header (without units).
            - dict: A dictionary mapping the cleaned column names to their units.
            - int: The line index where the data starts.

    Raises:
        ValueError: If no header lines are found or if the header is improperly formatted.
    """
    column_names = []
    units_map = {}
    data_start_index = len(lines)

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('#'):
            # Format: "# <index>: <column name> (<unit>)"
            parts = line[1:].split(':', 1)
            if len(parts) != 2:
                raise ValueError(f"Improperly formatted header line: {line}")

            raw_name = parts[1].strip()
            # Extract column name and unit (if present)
            match = re.match(r"(.+?)\s*\((.+?)\)$", raw_name)
            if match:
                column_name, unit = match.groups()
                column_name = column_name.strip()
                units_map[column_name] = unit.strip()
            else:
                column_name = raw_name.strip()
                units_map[column_name] = None  # No unit provided

            column_names.append(column_name)
        else:
            # Record the first line of data
            # Otherwise the start line of data is directed to the
            # end of the file.
            data_start_index = i
            break

    if not column_names:
        raise ValueError("No header lines found in the provided lines.")

    return column_names, units_map, data_start_index


def read_aspect_header_file(file_path):
    """
    Reads a simulation log file with specific column headers and data format.

    Args:
        file_path (str): Path to the simulation log file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the parsed data.
                     Units are stored in the `attrs` attribute of the DataFrame.
    """
    # Open the file once and read all lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the header and determine where the data starts
    column_names, units_map, data_start_index = parse_header_from_lines(lines)

    # Read the data starting from the determined index
    data = pd.read_csv(
        file_path,
        delim_whitespace=True,
        skiprows=data_start_index,
        names=column_names,
        skip_blank_lines=True
    )

    # Store units_map as metadata in the DataFrame
    data.attrs["units"] = units_map

    return data