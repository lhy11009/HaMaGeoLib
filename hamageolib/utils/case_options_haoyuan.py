# dealii_param_parser.py
# MIT License
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

"""
case_options_haoyuan.py

Author: Haoyuan Li
Description: A utility for parsing and handling deal.ii parameter files 
             in the hamageolib package.

Functions:
    - parse_parameters: Parses a deal.ii parameter file and returns a dictionary 
                        of parameters.
    - save_parameters: Saves a dictionary of parameters to a deal.ii formatted file.
"""

import json
import os
import re
import subprocess
import pandas as pd
from .dealii_param_parser import parse_parameters_to_dict
from .exception_handler import my_assert
from .handy_shortcuts_haoyuan import func_name
from .file_reader import read_aspect_header_file

class CASE_OPTIONS:
    """
    A class to substitute existing code with specific values and parameters for geodynamic modeling cases.

    Attributes:
        _case_dir (str): Directory path of the case.
        _output_dir (str): Directory path for output files.
        visit_file (str): Path to the visit file for case visualization.
        paraview_file (str): Path to the paraview file for case visualization.
        _img_dir (str): Directory path for image outputs.
        idict (dict): Dictionary containing parsed parameters from the .prm file.
        wb_dict (dict): Dictionary containing parsed parameters from the .wb file if it exists.
        options (dict): Dictionary storing interpreted options for data output and model parameters.
    """

    def __init__(self, case_dir):
        """
        Initializes the CASE_OPTIONS class by setting up file paths, checking directories,
        and loading parameters from .prm and .wb files if available.

        Args:
            case_dir (str): The directory of the case.
        """
        # todo_animate
        # Validate and set case directory
        self._case_dir = case_dir
        my_assert(os.path.isdir(self._case_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case directory - %s doesn\'t exist' % self._case_dir)

        # Validate and set output directory
        self._output_dir = os.path.join(case_dir, 'output')
        my_assert(os.path.isdir(self._output_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case output directory - %s doesn\'t exist' % self._output_dir)

        # Set paths for visualization files and validate access
        self.visit_file = None
        visit_file_tmp = os.path.join(self._output_dir, 'solution.visit')
        if os.path.isfile(visit_file_tmp):
            self.visit_file = visit_file_tmp

        self.paraview_file = None
        paraview_file_tmp = os.path.join(self._output_dir, 'solution.pvd')
        if os.path.isfile(paraview_file_tmp):
            self.paraview_file = paraview_file_tmp

        # Set or create image directory
        self._img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(self._img_dir):
            os.mkdir(self._img_dir)

        # Parse parameters from .prm file
        prm_file = os.path.join(self._case_dir, 'case.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case prm file - %s cannot be read' % prm_file)
        with open(prm_file, 'r') as fin:
            self.idict = parse_parameters_to_dict(fin)

        # Parse .wb file if available, or initialize an empty dictionary
        self.wb_dict = {}
        wb_file = os.path.join(self._case_dir, 'case.wb')
        if os.access(wb_file, os.R_OK):
            with open(wb_file, 'r') as fin:
                self.wb_dict = json.load(fin)

        # Initialize options dictionary
        self.options = {}

        # Read statistic results
        self.statistic_df = None
        statistic_path = os.path.join(case_dir, "output/statistics")
        if os.path.isfile(statistic_path):
            self.statistic_df = read_aspect_header_file(statistic_path)

        # Read visualization files
        # parse information of the output file
        try:
            time_between_graphical_output = self.idict['Postprocess']['Visualization']['Time between graphical output']
        except ValueError:
            time_between_graphical_output = 1e8 # default value

        try:
            graphical_output_format = self.idict['Postprocess']['Visualization']['Output format']
        except ValueError:
            graphical_output_format = "vtu" # default value

        self.visualization_df = None
        if self.statistic_df is not None:
            statistic_time = self.statistic_df["Time"]
            statistic_timestep = self.statistic_df["Time step number"]
            statistic_visualization_file_name = self.statistic_df["Visualization file name"]

            not_nan_indexes, non_nan_visualization_file_name, matched_series_list = \
                filter_and_match(statistic_visualization_file_name, statistic_time, statistic_timestep)

            data = {
                "Index": not_nan_indexes,
                "Visualization file name": non_nan_visualization_file_name.values,
                "Time": matched_series_list[0].values if matched_series_list else [],
                "Time step number": matched_series_list[1].values if matched_series_list else []
            }

            self.visualization_df = pd.DataFrame(data)
            if graphical_output_format == "vtu":
                self.visualization_df['File Exists'] = self.visualization_df["Visualization file name"].apply(
                    lambda file_name: os.path.isfile(os.path.join(case_dir, file_name + ".pvtu"))
                )
            else:
                raise NotImplementedError()

            self.visualization_df.attrs["Time between graphical output"] = time_between_graphical_output

    def resample_visualization_df(self, time_interval):

            resampled_df = resample_time_series_df(self.visualization_df, time_interval)
            resampled_df.attrs["Time between graphical output"] = self.visualization_df.attrs["Time between graphical output"]

            return resampled_df


# Parse run time results
def parse_log_file_for_visualization_snapshots(log_file_path, output_path, **kwargs):
    """
    Parses a log file to extract visualization snapshot information using an AWK script.

    Args:
        log_file_path (str): Path to the input log file containing snapshot data.
        output_path (str): Path to save the extracted snapshot information.
        **kwargs: Additional optional arguments:
            - debug (bool): If True, prints the executed command and its output for debugging purposes. Defaults to False.

    Details:
        - The function uses an AWK script to process the log file. The script's path is determined relative
          to the script's directory.
        - The `subprocess.run` command executes the AWK script with the provided log file and output path.
        - If `debug` is enabled, the executed command and the captured output are printed.

    Notes:
        - Ensure the AWK script (`parse_snapshots.awk`) exists at the expected relative location.
        - The output path should be writable, and the input log file should be readable.

    Example:
        parse_log_file_for_visualization_snapshots(
            log_file_path="/path/to/log_file.log",
            output_path="/path/to/output.txt",
            debug=True
        )
    """
    debug = kwargs.get("debug", False)

    awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts/parse_snapshots.awk"))
    assert(os.path.isfile(awk_configuration_file))

    with open(output_path, "w") as fout: 
        completed_process = subprocess.run(
            ["awk", "-f", awk_configuration_file, log_file_path],
            stdout=fout,
            stderr=subprocess.PIPE, 
            text=True
        )

    if debug: 
        print(f"Executed command:", completed_process.args)  # Debug: show the executed command
        # print(f"Output:\n", completed_process.stdout)  # Debug: show the captured output
        print(f"Error:\n", completed_process.stderr)  # Debug: show the captured output


# todo_animate
def parse_case_log(case_dir, **kwargs):
    """
    Parses a case directory's log file to extract and process visualization snapshot data.

    Args:
        case_dir (str): The directory containing the case data, including the log file to be processed.
        **kwargs: Additional optional arguments:
            - output directory (str): The name of the subdirectory within `case_dir` where the log file is located. Defaults to "output".
            - debug (bool): If True, enables debug output for logging and executed actions. Defaults to False.

    Details:
        - The function identifies the log file (`log.txt`) in the specified `case_dir` and creates necessary temporary directories
          for processing the output.
        - If the log file or output directories do not exist, they are created as needed.
        - Calls `parse_log_file_for_visualization_snapshots` to process the log file using an AWK script.
        - Ensures the existence of the processed output file after the AWK script execution.

    Steps:
        1. Validates the presence of `log.txt` in the specified case directory.
        2. Creates a temporary directory (`temp`) and a subdirectory (`awk_outputs`) to store intermediate results.
        3. Removes any pre-existing output file (`snapshots_outputs`) to ensure clean processing.
        4. Processes the log file using the `parse_log_file_for_visualization_snapshots` function.
        5. Verifies the successful creation of the output file.
        6. If debug mode is enabled, logs actions and created files.

    Example:
        parse_case_log(
            case_dir="/path/to/case_directory",
            output_directory="output",
            debug=True
        )
    """
    _output = kwargs.get("output directory", "output")
    debug = kwargs.get("debug", False)

    log_file = os.path.join(case_dir, _output, "log.txt")
    assert(os.path.isfile(log_file))

    temp_dir = os.path.join(case_dir, "temp")
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    
    temp_dir1 = os.path.join(case_dir, "temp", "awk_outputs")
    if not os.path.isdir(temp_dir1):
        os.mkdir(temp_dir1)

    output_file = os.path.join(temp_dir1, "snapshots_outputs")
    if os.path.isfile(output_file):
        os.remove(output_file)
 
    parse_log_file_for_visualization_snapshots(log_file, output_file, debug=debug)

    assert(os.path.isfile(output_file))

    if debug:
        print("created file %s" % output_file)


def filter_and_match(series1, *args):
    """
    Filters and matches values between a primary pandas Series and multiple other Series.
    The primary Series (series1) may contain NaN values, and the function matches corresponding values 
    from other Series based on the non-NaN indexes of series1.

    Args:
        series1 (pd.Series): The primary pandas Series object (may contain NaN values).
        *args: A variable number of pandas Series objects to match against series1.

    Returns:
        tuple: A tuple containing:
            - indexes (pd.Index): The indexes of non-NaN values in `series1`.
            - non_nan_first (pd.Series): The non-NaN values in `series1`.
            - matched_series_list (list of pd.Series): A list of Series objects corresponding to 
              the non-NaN indexes in `series1` from the provided `*args`.
    """
    # Get the indexes where `series1` is not NaN
    not_nan_indexes = series1.dropna().index

    # Get the corresponding non-NaN values in `series1`
    non_nan_first = series1.loc[not_nan_indexes]

    # Get the corresponding values for all Series in *args
    matched_series_list = [series.loc[not_nan_indexes] for series in args]

    return not_nan_indexes, non_nan_first, matched_series_list

# todo_animate
def resample_time_series_df(target_df, time_interval, time_column="Time"):
    """
    Resamples the DataFrame based on a given time interval from the "Time" column.

    Args:
        target_df (pd.DataFrame): The DataFrame to be resampled.
        time_interval (float): The time interval for resampling.
        time_column (str): The column name representing time. Defaults to "Time".

    Returns:
        pd.DataFrame: A resampled DataFrame containing rows closest to the specified time intervals.
    """
    # Sort the DataFrame by the time column
    target_df = target_df.sort_values(by=time_column)

    # Identify resampling intervals
    time_min = target_df[time_column].min()
    time_max = target_df[time_column].max()
    resample_points = pd.Series(
        [time_min + i * time_interval for i in range(int((time_max - time_min) / time_interval) + 1)]
    )

    # Find the closest rows to each resampling point
    resampled_rows = []
    for target_time in resample_points:
        # Calculate absolute differences
        time_diffs = (target_df[time_column] - target_time).abs()
        min_diff = time_diffs.min()
        
        # Get all rows with the same minimum difference
        closest_rows = target_df[time_diffs == min_diff]
        
        # Select the last row among those equidistant
        closest_row = closest_rows.iloc[-1]
        resampled_rows.append(closest_row)

    # Create a new DataFrame with the resampled rows
    resampled_df = pd.DataFrame(resampled_rows)

    return resampled_df