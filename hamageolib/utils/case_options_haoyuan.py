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
from .dealii_param_parser import parse_parameters_to_dict
from .exception_handler import my_assert
from .handy_shortcuts_haoyuan import func_name

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

    def Interpret(self):
        """
        Interprets and assigns model configuration parameters to the options dictionary.
        This method processes directory paths, dimensions, geometry, and refinement settings.
        """
        # Set data and image output directories in options
        self.options["DATA_OUTPUT_DIR"] = os.path.abspath(self._output_dir)
        if not os.path.isdir(self._img_dir):
            os.mkdir(self._img_dir)
        self.options["IMG_OUTPUT_DIR"] = os.path.abspath(self._img_dir)

        # Set model dimension and adaptive refinement options
        self.options['DIMENSION'] = int(self.idict['Dimension'])
        self.options['INITIAL_ADAPTIVE_REFINEMENT'] = self.idict['Mesh refinement'].get('Initial adaptive refinement', '0')

        # Determine geometry settings and assign radius or extent based on geometry model
        geometry = self.idict['Geometry model']['Model name']
        self.options['GEOMETRY'] = geometry
        self.options["Y_EXTENT"] = -1.0

        if geometry == 'chunk':
            self.options["OUTER_RADIUS"] = float(self.idict['Geometry model']['Chunk']['Chunk outer radius'])
            self.options["INNER_RADIUS"] = float(self.idict['Geometry model']['Chunk']['Chunk inner radius'])
            self.options["XMAX"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum longitude'])
        elif geometry == 'box':
            if self.options['DIMENSION'] == 2:
                self.options["OUTER_RADIUS"] = float(self.idict['Geometry model']['Box']['Y extent'])
                self.options["INNER_RADIUS"] = 0.0
            elif self.options['DIMENSION'] == 3:
                self.options["OUTER_RADIUS"] = float(self.idict['Geometry model']['Box']['Z extent'])
                self.options["INNER_RADIUS"] = 0.0
            else:
                raise ValueError("%d is not a dimension option" % self.options['DIMENSION'])
            self.options["XMAX"] = float(self.idict['Geometry model']['Box']['X extent'])

    def read_contents(self, *paths):
        """
        Reads and concatenates contents from specified file paths.

        Args:
            paths (str): One or more file paths to read from.
        """
        # Concatenates file contents, separated by newlines
        self.contents = ''
        i = 0  # Count files for separating content with newlines
        for _path in paths:
            my_assert(os.access(_path, os.R_OK), FileNotFoundError, "%s: %s cannot be opened" % (func_name(), _path))
            with open(_path, 'r') as fin:
                if i > 0:
                    self.contents += '\n\n'
                self.contents += fin.read()
            i += 1

    def read_options(self, _path):
        """
        Reads options from a JSON file and loads them into the options dictionary.

        Args:
            _path (str): The path to the JSON file.
        """
        # Reads JSON file contents into options dictionary
        my_assert(os.access(_path, os.R_OK), FileNotFoundError, "%s: %s cannot be opened" % (func_name(), _path))
        with open(_path, 'r') as fin:
            self.options = json.load(fin)

    def substitute(self):
        """
        Replaces all instances of keys in the options dictionary with their values in the contents.
        """
        # Iterates over options, replacing keys with corresponding values in contents
        for key, value in self.options.items():
            self.contents = re.sub(key, str(value), self.contents)

    def save(self, _path):
        """
        Saves the modified contents to a new file, creating directories if necessary.

        Args:
            _path (str): The path to save the new file.

        Returns:
            str: The path of the saved file.
        """
        # Ensure the directory for the file exists
        dir_path = os.path.dirname(_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)

        # Write contents to file
        with open(_path, 'w') as fout:
            fout.write(self.contents)
        return _path
