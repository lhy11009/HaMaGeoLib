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
case_options.py

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
import numpy as np
from .dealii_param_parser import parse_parameters_to_dict, save_parameters_from_dict
from .exception_handler import my_assert
from .handy_shortcuts_haoyuan import func_name, strip_and_split
from .file_reader import read_aspect_header_file
import copy

class CASE_OPTIONS:
    """
    A class to substitute existing code with specific values and parameters for geodynamic modeling cases.

    Attributes:
        _case_dir (str): Directory path of the case.
        output_dir (str): Directory path for output files.
        visit_file (str): Path to the visit file for case visualization.
        paraview_file (str): Path to the paraview file for case visualization.
        img_dir (str): Directory path for image outputs.
        idict (dict): Dictionary containing parsed parameters from the .prm file.
        wb_dict (dict): Dictionary containing parsed parameters from the .wb file if it exists.
        options (dict): Dictionary storing interpreted options for data output and model parameters.
        summary_df (panda object): case summary
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
        self.case_dir = case_dir
        my_assert(os.path.isdir(self.case_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case directory - %s doesn\'t exist' % self.case_dir)

        # Validate and set output directory
        self.output_dir = os.path.join(case_dir, 'output')
        my_assert(os.path.isdir(self.output_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case output directory - %s doesn\'t exist' % self.output_dir)

        # Set paths for visualization files and validate access
        self.visit_file = None
        visit_file_tmp = os.path.join(self.output_dir, 'solution.visit')
        if os.path.isfile(visit_file_tmp):
            self.visit_file = visit_file_tmp

        self.paraview_file = None
        paraview_file_tmp = os.path.join(self.output_dir, 'solution.pvd')
        if os.path.isfile(paraview_file_tmp):
            self.paraview_file = paraview_file_tmp

        # Set or create image directory
        self.img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)

        # Parse parameters from .prm file
        prm_file = os.path.join(self.case_dir, 'case.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case prm file - %s cannot be read' % prm_file)
        with open(prm_file, 'r') as fin:
            self.idict = parse_parameters_to_dict(fin)

        # Parse .wb file if available, or initialize an empty dictionary
        self.wb_dict = {}
        wb_file = os.path.join(self.case_dir, 'case.wb')
        if os.access(wb_file, os.R_OK):
            with open(wb_file, 'r') as fin:
                self.wb_dict = json.load(fin)

        # Initialize options dictionary
        self.options = {}
        
        # Initialize summary dictionary
        self.summary_df = {}

        # Read statistic results
        self.statistic_df = None
        statistic_path = os.path.join(case_dir, "output/statistics")
        if os.path.isfile(statistic_path):
            self.statistic_df = read_aspect_header_file(statistic_path)

        # Log file
        log_file_path = os.path.join(self.output_dir, "log.txt")
        # my_assert(os.access(log_file_path, os.R_OK), FileNotFoundError,
                #   'BASH_OPTIONS.__init__: log file - %s cannot be read' % log_file_path)
        if os.access(log_file_path, os.R_OK):
            # header infomation: versions
            self.time_df = parse_log_file_for_time_info_to_pd(log_file_path)
            results = parse_log_file_for_header_info(log_file_path)
            self.aspect_version = results[0]
            self.dealii_version = results[1]
            self.world_builder_version = results[2]
            self.n_mpi = int(results[3])
        else:
            self.time_df = None
            self.aspect_version = None
            self.dealii_version = None
            self.world_builder_version = None
            self.n_mpi = None
        

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

    def SummaryCaseVtuStep(self):
        '''
        Generate a Case Summary
        ofile (str): if this provided, output the csv summary
        '''
        my_assert(self.options != {}, ValueError, "The Case needs to be interpreted first")
        # todo_data
        # Get a summary of case status
        initial_adaptive_refinement = int(self.options['INITIAL_ADAPTIVE_REFINEMENT'])
        Time = self.visualization_df["Time"].iloc[initial_adaptive_refinement:]
        Time_step_number = self.visualization_df["Time step number"].iloc[initial_adaptive_refinement:]
        Vtu_step = np.arange(0, Time_step_number.size, dtype=int)
        Vtu_snapshot = Vtu_step + initial_adaptive_refinement
        self.summary_df = pd.DataFrame({
            "Vtu step": Vtu_step,
            "Time": Time,
            "Time step number": Time_step_number,
            "Vtu snapshot": Vtu_snapshot,
            "File found": [True for i in range(Time_step_number.size)]
            })

    def SummaryCaseVtuStepUpdateValue(self, field_name, vtu_step, value):
        '''
        Update the value of a give field at vtu_step with value
        '''
        # Get the matching index
        index_list = self.summary_df.index[self.summary_df["Vtu step"] == vtu_step].tolist()
        my_assert(len(index_list)==1, ValueError, "Given value of vtu_step should match exact one value in the recorded values")
        index = index_list[0]

        # assign value
        my_assert(field_name in self.summary_df.columns, ValueError, "Given field_name is not an entry of summary")
        self.summary_df.at[index, field_name] = value
    
    
    def SummaryCaseVtuStepExport(self, ofile):
        '''
        exprot summary based on vtu steps
        '''
        self.summary_df.to_csv(ofile, index=False)
        print("%s: saved file %s" % (func_name(), ofile))



    def resample_visualization_df(self, time_interval):

            resampled_df = resample_time_series_df(self.visualization_df, time_interval)
            resampled_df.attrs["Time between graphical output"] = self.visualization_df.attrs["Time between graphical output"]

            return resampled_df
    
    def Interpret(self, **kwargs):

        # paths
        self.options["OUTPUT_DIRECTORY"] = self.output_dir
        self.options["IMAGE_DIRECTORY"] = self.img_dir

        # dimension
        dimension = int(self.idict['Dimension'])
        self.options['DIMENSION'] = dimension

        # geometry
        geometry = self.idict['Geometry model']['Model name']
        self.options['GEOMETRY'] = geometry

        if geometry == 'chunk':
            self.options["BOTTOM"]  = float(self.idict['Geometry model']['Chunk']['Chunk inner radius'])
            self.options["TOP"]  = float(self.idict['Geometry model']['Chunk']['Chunk outer radius'])
            self.options["LEFT"] = float(self.idict['Geometry model']['Chunk']['Chunk minimum longitude'])
            self.options["RIGHT"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum longitude'])
            if dimension == 2:
                self.options["FRONT"] = None
                self.options["BACK"] = None
            elif dimension == 3:
                self.options["FRONT"] = float(self.idict['Geometry model']['Chunk']['Chunk minimum latitude'])
                self.options["BACK"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum latitude'])
            else: 
                raise ValueError("%d is not a dimension option" % self.options['DIMENSION'])
            
        elif geometry == 'box':
            self.options["LEFT"] = 0.0
            self.options["RIGHT"] = float(self.idict['Geometry model']['Box']['X extent'])
            if dimension == 2:
                self.options["BOTTOM"] = 0.0
                self.options["TOP"]  = float(self.idict['Geometry model']['Box']['Y extent'])
                self.options["FRONT"] = None
                self.options["BACK"] = None
            elif dimension == 3:
                self.options["BOTTOM"] = 0.0
                self.options["TOP"]  = float(self.idict['Geometry model']['Box']['Z extent']) 
                self.options["FRONT"] = 0.0
                self.options["BACK"] = float(self.idict['Geometry model']['Box']['Y extent'])
            else: 
                raise ValueError("%d is not a dimension option" % self.options['DIMENSION'])

        # adaptive mesh
        self.options['INITIAL_ADAPTIVE_REFINEMENT'] = self.idict['Mesh refinement'].get('Initial adaptive refinement', '0')
            
class CASE_SUMMARY:

    def __init__(self):

        # initiate a data object to save the case summary
        self.df = pd.DataFrame(columns=["basename", "name", "end time step", "end time", "wall clock", "abs path"])

        units_map = {}
        units_map["basename"] = None
        units_map["name"] = None
        units_map["end time step"] = None
        units_map["end time"] = 'yr'
        units_map["wall clock"] = 'hr'
    
        self.df.attrs["units"] = units_map


    def update_single_case(self, abs_path):
        """
        Updates or adds a new case entry in the DataFrame based on the provided absolute path.
        If an existing row is updated, non-trivial values (non-None, non-empty) from the original row 
        are preserved over default None values in the new entry.

        Parameters:
            self.df (pandas.DataFrame): The DataFrame containing case entries.
            abs_path (str): The absolute path of the case to update or add.

        Returns:
            pandas.DataFrame: The updated DataFrame with the modified or new case entry.

        Notes:
            - If `abs_path` already exists in the DataFrame, the corresponding row is updated.
            - If `abs_path` is new, a new row is appended to the DataFrame with default values.
            - When updating an existing row, None values in `new_row` are replaced with original row values.
        """

        # Define new row data with default placeholders
        new_row, Case_Options = self.update_single_case_attributes(abs_path)

        # Check if the absolute path already exists in the DataFrame
        abs_path_1 = os.path.abspath(abs_path)
        existing_index = self.df[self.df["abs path"] == abs_path_1].index

        if not existing_index.empty:
            # If abs_path exists, preserve non-trivial values from the existing row
            existing_row = self.df.loc[existing_index].to_dict(orient="records")[0]  # Convert to dictionary

            # Merge values, keeping existing non-trivial values over None values in new_row
            merged_row = {key: existing_row[key] if new_row[key] in [None, ""] and existing_row[key] not in [None, ""] 
                        else new_row[key] 
                        for key in new_row}

            # Update the row in the DataFrame
            self.df.loc[existing_index, :] = merged_row

        else:
            # If abs_path does not exist, append a new row
            self.df = self.df.append(new_row, ignore_index=True)


    def update_single_case_attributes(self, abs_path):
        """
        Updates or adds a new case entry in the DataFrame based on the provided absolute path.
        This is the one function to reload for different project.

        Parameters:
            df (pandas.DataFrame): The DataFrame containing case entries.
            abs_path (str): The absolute path of the case to update or add.
        """
        Case_Options = CASE_OPTIONS(abs_path)
        
        new_row = {
            "basename": os.path.basename(abs_path),
            "name": None,  # Placeholder for name
            "end time step": Case_Options.time_df["Time step number"].iloc[-1],  # end time step
            "end time": Case_Options.time_df["Time"].iloc[-1],  # end time
            "wall clock": Case_Options.time_df["Corrected Wall Clock"].iloc[-1],  # wall clock
            "abs path": os.path.abspath(abs_path)  # absolute path
        }

        return new_row, Case_Options


def df_to_latex(df, output_file=None, columns=None):
    """
    Converts a pandas DataFrame to a LaTeX table while preserving selected column names.

    Parameters:
        df (pandas.DataFrame): The DataFrame to convert.
        output_file (str, optional): Path to save the LaTeX table (if provided).
        columns (list of str, optional): List of column names to include in the LaTeX table. 
                                         If None, all columns are included.

    Returns:
        str: LaTeX formatted table as a string.

    Notes:
        - If `columns` is provided, only those columns are included.
        - Uses `to_latex()` with formatting options for better readability.
        - If `output_file` is specified, saves the LaTeX table instead of returning it.
    """

    # If specific columns are provided, filter the DataFrame
    if columns:
        df = df[columns]

    # Convert DataFrame to LaTeX format
    latex_table = df.to_latex(index=False, column_format="l" + "c" * (df.shape[1] - 1))

    # Save to file if an output path is provided
    if output_file:
        with open(output_file, "w") as f:
            f.write(latex_table)
        return f"LaTeX table saved to {output_file}"

    return latex_table

    

def find_case_files(directory_path):
    """
    Determines whether a given directory contains a case based on the presence of a .prm file.
    If a .prm file is found, the function also checks for a corresponding .wb file.
    
    Selection criteria:
    - The function first looks for "case.prm" as the preferred .prm file. If not found, 
      it selects the first available .prm file in the directory.
    - If a .prm file is found, the function then searches for a .wb file.
    - The function prioritizes "case.wb" as the preferred .wb file. If not found, 
      it selects the first available .wb file in the directory.
    
    Parameters:
        directory_path (str): Path to the directory being checked.

    Returns:
        tuple (str or None, str or None): A tuple containing:
            - The absolute path to the selected .prm file, or None if no .prm file is found.
            - The absolute path to the selected .wb file, or None if no .wb file is found.

    Notes:
        - If no .prm file exists, the function returns (None, None).
        - If no .wb file exists, only the .prm file path is returned with None for .wb.
    """

    # Ensure the provided path is a valid directory
    if not os.path.isdir(directory_path):
        return None, None

    # List all .prm and .wb files in the directory
    prm_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".prm")])
    wb_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".wb")])

    # Determine the .prm file
    prm_file = os.path.abspath(os.path.join(directory_path, "case.prm")) if "case.prm" in prm_files else (
        os.path.join(directory_path, prm_files[0]) if prm_files else None
    )

    # Determine the .wb file (only if a .prm file exists)
    wb_file = None
    if prm_file:
        wb_file = os.path.abspath(os.path.join(directory_path, "case.wb")) if "case.wb" in wb_files else (
            os.path.join(directory_path, wb_files[0]) if wb_files else None
        )

    return prm_file, wb_file


def parse_log_file_for_time_info(log_file_path, output_path, **kwargs):
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

    awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts/awk/parse_block_output.awk"))
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


def parse_log_file_for_header_info(log_file_path, **kwargs):
    """
    Parses a log file to extract header information using an AWK script.

    Args:
        log_file_path (str): Path to the input log file containing snapshot data.
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
        parse_log_file_for_header_info(
            log_file_path="/path/to/log_file.log",
            debug=True
        )
    """
    debug = kwargs.get("debug", False)

    awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "awk", "parse_block_header.awk"))
    assert(os.path.isfile(awk_configuration_file))

    completed_process = subprocess.run(
        ["awk", "-f", awk_configuration_file, log_file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, 
        text=True
    )

    results = completed_process.stdout.split("\n")[1]
    
    if debug: 
        print(f"Executed command:", completed_process.args)  # Debug: show the executed command
        print(f"Output:\n", completed_process.stdout)  # Debug: show the captured output
        print(f"Error:\n", completed_process.stderr)  # Debug: show the captured output

    return strip_and_split(results)


def parse_log_file_for_solver_info(log_file_path, output_path, **kwargs):
    """
    Parses a log file to solver information using an awk script.

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
        - Ensure the AWK script exists at the expected relative location.
        - The output path should be writable, and the input log file should be readable.

    Example:
        parse_log_file_for_solver_info(
            log_file_path="/path/to/log_file.log",
            output_path="/path/to/output.txt",
            debug=True
        )
    """
    debug = kwargs.get("debug", False)
    major_version = kwargs.get("major_version", 2)

    awk_configuration_file = None
    if major_version == 2:
        awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "awk", "parse_block_newton_v2_6_0.awk"))
    elif major_version == 3:
        awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "awk", "parse_block_newton_v3_1_0.awk"))
    else:
        raise NotImplementedError()

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



def parse_log_file_for_time_info_to_pd(log_file_path, **kwargs):
    """
    Parses an ASPECT log file to extract time step and wall clock time information, 
    corrects for restarts, and returns a pandas DataFrame with the corrected wall clock times.

    Parameters:
        log_file_path (str): Path to the ASPECT log file.
        **kwargs: Additional arguments passed to `parse_log_file_for_time_info`.

    Returns:
        pandas.DataFrame: A DataFrame containing:
            - "Time step number": The recorded time step index.
            - "Wall Clock": The original wall clock time recorded in the log.
            - "Corrected Wall Clock": The adjusted wall clock time accounting for restarts.

    Raises:
        AssertionError: If the parsed output file is not generated successfully.

    Notes:
        - The function first processes the log file using `parse_log_file_for_time_info`, 
          which extracts time information and stores it in an intermediate file.
        - It then loads the parsed data using `read_aspect_header_file`.
        - The function detects simulation restarts, where the wall clock time resets to zero, 
          and accumulates previous wall clock times to maintain a continuous time record.
    """

    awk_output_path = os.path.join(os.path.dirname(log_file_path), "parse_time_info_results")

    parse_log_file_for_time_info(log_file_path, awk_output_path, **kwargs)

    assert(os.path.isfile(awk_output_path))

    # Correct for restart
    time_dat = read_aspect_header_file(awk_output_path)

    corrected_wall_clock = time_dat["Wall Clock"].copy()

    offset = 0.0 # Track cumulative shift

    for i in range(1, len(time_dat)):
        # Detected a restart, add the last wall clock time before restart
        if time_dat["Time step number"].iloc[i] <= time_dat["Time step number"].iloc[i - 1]:
            offset += time_dat["Wall Clock"].iloc[i - 1]
        
        # Apply offset
        corrected_wall_clock.iloc[i] += offset

    time_dat["Corrected Wall Clock"] = corrected_wall_clock

    return time_dat



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

    awk_configuration_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts/awk/parse_snapshots.awk"))
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


class ModelConfigManager:
    """
    Manage and modify a finite element geodynamic model configuration.
    
    Attributes:
        config (dict): The current configuration dictionary.
    """
    def __init__(self, base_config: dict):
        self.config = copy.deepcopy(base_config)
    
    def __init__(self, base_file: str):
        assert(os.path.isfile(base_file))
        with open(base_file, "r") as fin:
            self.config = parse_parameters_to_dict(fin)

    def apply_patch(self, patch: dict):
        """Apply a flat dictionary patch to override settings."""
        self.config.update(patch)

    def apply_nested_patch(self, nested_patch: dict):
        """Recursively update config with nested dictionary patch."""
        self._recursive_update(self.config, nested_patch)

    def _recursive_update(self, base: dict, updates: dict):
        for key, value in updates.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._recursive_update(base[key], value)
            else:
                base[key] = value

    def get_config(self):
        return self.config

    def export_to_param_file(self, output_path: str):

        with open(output_path, "w") as fout:
            save_parameters_from_dict(fout, self.config)

# todo_data
class SimulationStatus:
    def __init__(self, initial_data=None):
        """
        Initialize the simulation status tracker.
        `initial_data` can be a list of dicts, a DataFrame, or None.
        """
        columns = ["time", "time_step", "vtu_step", "vtu_snapshot"]
        if initial_data is None:
            self.df = pd.DataFrame(columns=columns)
        elif isinstance(initial_data, pd.DataFrame):
            self.df = initial_data
        else:
            self.df = pd.DataFrame(initial_data, columns=columns)

    def add_entry(self, time, time_step, vtu_step, vtu_snapshot):
        """Add a new row to the DataFrame."""
        new_row = {"time": time, "time_step": time_step, "vtu_step": vtu_step, "vtu_snapshot": vtu_snapshot}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)

    def save_to_csv(self, filename):
        """Save the current DataFrame to a CSV file."""
        self.df.to_csv(filename, index=False)

    def load_from_csv(self, filename):
        """Load a DataFrame from a CSV file."""
        self.df = pd.read_csv(filename)

    def __repr__(self):
        return repr(self.df)
