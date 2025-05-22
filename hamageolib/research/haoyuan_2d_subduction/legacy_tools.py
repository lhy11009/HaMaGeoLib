# ==============================================================================
# MIT License
# 
# Copyright (c) 2025 Haoyuan Li
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================

"""
File: legacy_tools.py

Author: Haoyuan Li

Description:
    A collection of legacy functions and class definitions reused from a previous 
    geodynamic modeling project. These tools may not be fully refactored or 
    standardized, but provide useful utilities for 2D subduction model development.

    This module is part of the HaMaGeoLib research branch under:
    hamageolib/research/haoyuan_2d_subduction/

    Note:
    - Functions and classes here may undergo future cleanup and integration.
    - Use with discretion, as some may carry assumptions from earlier models.
"""
import os
import json
import re
import numpy as np
import warnings
import vtk
import time
import multiprocessing
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import gridspec, cm
from matplotlib import patches as mpatches 
from PIL import Image, ImageDraw, ImageFont
from cmcrameri import cm as ccm 
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize
from joblib import Parallel, delayed
from .legacy_utilities import JsonOptions, ReadHeader, CODESUB, cart2sph, SphBound, clamp, ggr2cart, point2dist, UNITCONVERT, ReadHeader2,\
ggr2cart, var_subs, JSON_OPT, string2list, re_neat_word
from ...utils.exception_handler import my_assert
from ...utils.handy_shortcuts_haoyuan import func_name
from ...utils.dealii_param_parser import parse_parameters_to_dict
from ...utils.world_builder_param_parser import find_wb_feature
from ...utils.geometry_utilities import offset_profile

JSON_FILE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "legacy_json_files")
LEGACY_FILE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "legacy_files")

# todo_cv
def ExportData(depth_average_path, output_dir, **kwargs):
    '''
    Export data of a step to separate file
    Inputs:
        kwargs:
            time_step - time_step to plot the figure, default is 0
            fix_time_step - fix time step if it's beyong the limit, default is False.
    Returns:
        odata (ndarray): selected data for outputing
        output_path (str): path of file generated
    '''
    time_step = kwargs.get('time_step', 0)
    fix_time_step = kwargs.get('fix_time_step', False)
    assert(os.access(depth_average_path, os.R_OK))
    # read that
    DepthAverage = DEPTH_AVERAGE_PLOT('DepthAverage')
    DepthAverage.ReadHeader(depth_average_path)
    DepthAverage.ReadData(depth_average_path)
    # manage data
    DepthAverage.SplitTimeStep()
    if fix_time_step and time_step > len(DepthAverage.time_step_times) - 1:
        time_step = len(DepthAverage.time_step_times) - 2
    try:
        i0 = DepthAverage.time_step_indexes[time_step][-1] * DepthAverage.time_step_length
        if time_step == len(DepthAverage.time_step_times) - 1:
            # this is the last step
            i1 = DepthAverage.data.shape[0]
        else:
            i1 = DepthAverage.time_step_indexes[time_step + 1][0] * DepthAverage.time_step_length
    except IndexError:
        print("PlotDaFigure: File (%s) may not contain any depth average output, abort" % depth_average_path)
        return
    names = kwargs.get('names', ['depth', 'temperature', 'adiabatic_density'])
    output_path = os.path.join(output_dir, 'depth_average_output_s%d' % time_step)
    odata = DepthAverage.export(output_path, names, rows=[i for i in range(i0, i1)], include_size=True)
    return odata, output_path

def GetSnapsSteps(case_dir, type_='graphical'):
    '''
    Get snaps for visualization from the record of statistic file.
    This function requires a consistent statistic file with respect to the vtu files.
    Checking the statistic file is more on the safe side from checking the vtu file, since a newer run might overight the older result.
    '''
    case_output_dir = os.path.join(case_dir, 'output')

    # import parameters
    prm_file = os.path.join(case_dir, 'output', 'original.prm')
    my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,
              'case prm file - %s cannot be read' % prm_file)
    with open(prm_file, 'r') as fin:
        idict = parse_parameters_to_dict(fin)

    # import statistics file
    Statistics = STATISTICS_PLOT_OLD('Statistics')
    statistic_file = os.path.join(case_output_dir, 'statistics')
    my_assert(os.access(statistic_file, os.R_OK), FileNotFoundError,
              'case statistic file - %s cannot be read' % prm_file)
    Statistics.ReadHeader(statistic_file)
    Statistics.ReadData(statistic_file)
    col_time = Statistics.header['Time']['col']
    col_step = Statistics.header['Time_step_number']['col']

    # final time and step
    final_time = Statistics.data[-1, col_time]
    final_step = int(Statistics.data[-1, col_step])

    total_graphical_outputs = 0
    graphical_times = []
    graphical_steps = []
    # time interval
    # graphical
    try:
        time_between_graphical_output = float(idict['Postprocess']['Visualization']['Time between graphical output'])
    except KeyError:
        time_between_graphical_output = 1e8
    if time_between_graphical_output < 1e-6:
        # in case of 0, results are written every step
        total_graphical_outputs = int(final_step) + 1
        graphical_times = Statistics.data[:, col_time]
        graphical_steps = range(final_step + 1)
    else:
        total_graphical_outputs = int(final_time / time_between_graphical_output) + 1
        graphical_times = [i*time_between_graphical_output for i in range(total_graphical_outputs)]
        graphical_steps = [Statistics.GetStep(time) for time in graphical_times]
    # particle
    try:
        time_between_particles_output = float(idict['Postprocess']['Particles']['Time between data output'])
        total_particles_outputs = int(final_time / time_between_particles_output) + 1
    except KeyError:
        time_between_particles_output = 1e8
        total_particles_outputs = 0
    particle_times = [i*time_between_particles_output for i in range(total_particles_outputs)]
    particle_steps = [Statistics.GetStep(time) for time in particle_times]

    # initial_snap
    try:
        initial_snap = int(idict['Mesh refinement']['Initial adaptive refinement'])
    except KeyError:
        initial_snap = 0

    # end snap
    snaps = [0]
    if type_ == 'graphical':
        start_ = initial_snap
        end_ = total_graphical_outputs + initial_snap
        snaps = list(range(start_, end_))
        times = graphical_times
        steps = graphical_steps
    elif type_ == 'particle':
        start_ = 0
        end_ = total_particles_outputs
        snaps = list(range(start_, end_))
        times = particle_times
        steps = particle_steps

    return snaps, times, steps

class CASE_OPTIONS(CODESUB):
    """
    parse .prm file to a option file that bash can easily read
    This inherit from CODESUB
    Attributes:
        _case_dir(str): path of this case
        _output_dir(str): path of the output
        visit_file(str): path of the visit file
        options(dict): dictionary of key and value to output
        i_dict(dict): dictionary for prm file
        wb_dict(dict): dictionary for wb file
    """
    def __init__(self, case_dir):
        """
        Initiation
        Args:
            case_dir(str): directory of case
        """
        CODESUB.__init__(self)
        # check directory
        self._case_dir = case_dir
        my_assert(os.path.isdir(self._case_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case directory - %s doesn\'t exist' % self._case_dir)
        self._output_dir = os.path.join(case_dir, 'output')
        my_assert(os.path.isdir(self._output_dir), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case output directory - %s doesn\'t exist' % self._output_dir)
        self.visit_file = os.path.join(self._output_dir, 'solution.visit')
        self.paraview_file = os.path.join(self._output_dir, 'solution.pvd')
        my_assert(os.access(self.visit_file, os.R_OK), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case visit file - %s cannot be read' % self.visit_file)
        # output dir
        self._output_dir = os.path.join(case_dir, 'output')
        if not os.path.isdir(self._output_dir):
            os.mkdir(self._output_dir)
        # img dir
        self._img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(self._img_dir):
            os.mkdir(self._img_dir)

        # get inputs from .prm file
        prm_file = os.path.join(self._case_dir, 'case.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,
                  'BASH_OPTIONS.__init__: case prm file - %s cannot be read' % prm_file)
        with open(prm_file, 'r') as fin:
            self.idict = parse_parameters_to_dict(fin)
        
        # wb inputs:
        #   if there is a .wb file, it is loaded. Otherwise, just start with
        # a vacant dictionary.
        self.wb_dict = {}
        wb_file = os.path.join(self._case_dir, 'case.wb')
        if os.access(wb_file, os.R_OK):
            with open(wb_file, 'r') as fin:
                self.wb_dict = json.load(fin)

        # initiate a dictionary
        self.options = {}

        # initiate a statistic data
        self.Statistics = STATISTICS_PLOT('Statistics')
        self.statistic_file = os.path.join(self._output_dir, 'statistics')
        self.Statistics.ReadHeader(self.statistic_file)
        try:
            self.Statistics.ReadData(self.statistic_file)
        except ValueError as e:
            raise ValueError("%s: error while reading file %s" % (func_name(), self.statistic_file)) from e

        # horiz_avg
        self.horiz_avg_file = os.path.join(self._output_dir, "depth_average.txt")


    def Interpret(self):
        """
        Interpret the inputs, to be reloaded in children
        """
        # directory to output data
        self.options["DATA_OUTPUT_DIR"] = self._output_dir
        # directory to output images
        if not os.path.isdir(self._img_dir):
            os.mkdir(self._img_dir)
        self.options["IMG_OUTPUT_DIR"] = self._img_dir
        # dimension
        self.options['DIMENSION'] = int(self.idict['Dimension'])
        # initial adaptive refinement
        self.options['INITIAL_ADAPTIVE_REFINEMENT'] = self.idict['Mesh refinement'].get('Initial adaptive refinement', '0')
        # geometry
        # some notes on the "OUTER_RADIUS", in the case of box geometry
        # I want this value to record "Y" or "Z" in order to write consistent
        # scripts for different geometry
        geometry = self.idict['Geometry model']['Model name']
        self.options['GEOMETRY'] = geometry
        self.options["Y_EXTENT"] = -1.0
        if geometry == 'chunk':
            self.options["OUTER_RADIUS"]  = float(self.idict['Geometry model']['Chunk']['Chunk outer radius'])
            self.options["INNER_RADIUS"]  = float(self.idict['Geometry model']['Chunk']['Chunk inner radius'])
            self.options["XMAX"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum longitude'])
        elif geometry == 'box':
            if self.options['DIMENSION'] == 2:
                self.options["OUTER_RADIUS"]  = float(self.idict['Geometry model']['Box']['Y extent'])
                self.options["INNER_RADIUS"] = 0.0
            elif self.options['DIMENSION'] == 3:
                self.options["OUTER_RADIUS"]  = float(self.idict['Geometry model']['Box']['Z extent']) 
                self.options["INNER_RADIUS"] = 0.0
            else: 
                raise ValueError("%d is not a dimension option" % self.options['DIMENSION'])
            self.options["XMAX"] = float(self.idict['Geometry model']['Box']['X extent'])
    
    def get_geometry(self):
        '''
        get the name of geomery
        '''
        return self.idict['Geometry model']['Model name']

    def save(self, _path, **kwargs):
        '''
        save contents to a new file
        Args:
            kwargs(dict):
                relative: use relative path
        '''
        use_relative_path = kwargs.get('relative', False)
        if use_relative_path:
            _path = os.path.join(self._case_dir, _path)
        o_path = CODESUB.save(self, _path)
        print("saved file: %s" % _path)
        return o_path

    def __call__(self, ofile, kwargs):
        """
        Call function
        Args:
            ofile(str): path of output
        """
        # interpret
        self.Interpret(kwargs)

        # open ofile for output
        # write outputs by keys and values
        with open(ofile, 'w') as fout:
            for key, value in self.options.items():
                fout.write("%s       %s\n" % (key, value))
        pass


class VISIT_OPTIONS_BASE(CASE_OPTIONS):
    """
    parse .prm file to a option file that bash can easily read
    """
    def Interpret(self, **kwargs):
        """
        Interpret the inputs, to be reloaded in children
        kwargs: options
            steps (int): plot some steps
            last_step(list): plot the last few steps
        """
        steps = kwargs.get('steps', None)
        last_step = kwargs.get('last_step', None)
        time_interval = kwargs.get('time_interval', None)
        plot_axis = kwargs.get('plot_axis', False)
        max_velocity = kwargs.get('max_velocity', -1.0)
        slices = kwargs.get('slices', 3)
        graphical_type = kwargs.get("graphical_type", "pvd")
        # call function from parent
        CASE_OPTIONS.Interpret(self)
        # particle file
        particle_file = os.path.join(self._output_dir, 'particles.visit')
        if os.access(particle_file, os.R_OK):
            self.options["VISIT_PARTICLE_FILE"] = particle_file
        # visit file
        self.options["VISIT_FILE"] = self.visit_file
        self.options["PARAVIEW_FILE"] = self.paraview_file
        # data types
        self.options["HAS_DYNAMIC_PRESSURE"] = '0'
        try:
            visualization_output_variables = self.idict['Postprocess']['Visualization']['List of output variables']
        except KeyError:
            pass
        else:
            if re.match('.*nonadiabatic\ pressure', visualization_output_variables):
                self.options["HAS_DYNAMIC_PRESSURE"] = '1'

        # plot options
        # plot axis
        if plot_axis:
            self.options["PLOT_AXIS"] = '1'
        else: 
            self.options["PLOT_AXIS"] = '0'
        # maximum velocity
        self.options["MAX_VELOCITY"] = str(max_velocity)
        self.options["PLOT_TYPES"] = str(kwargs.get('plot_types', []))
        # additional fields to load for model
        additional_fields = kwargs.get('additional_fields', [])
        assert(type(additional_fields) == list)
        self.options["ADDITIONAL_FIELDS"] = str(additional_fields)

        # get all the available snaps for ploting by checking on the existence of the pvtu file
        # the correspondent, time, time step are also figured out.
        graphical_snaps_guess, times_guess, time_steps_guess = GetSnapsSteps(self._case_dir, 'graphical')
        graphical_snaps = []
        time_steps = []
        times = []
        for i in range(len(graphical_snaps_guess)):
            graphical_snap = graphical_snaps_guess[i]
            time_step = time_steps_guess[i]
            _time = times_guess[i]
            graphical_file_path = None
            if graphical_type == "pvd":
                graphical_file_path = os.path.join(self.options["DATA_OUTPUT_DIR"], "solution", "solution-%05d.pvtu" % graphical_snap)
            elif graphical_type == "slice_center":
                graphical_file_path = os.path.join(self._case_dir, "vtk_outputs", "center_profile_%05d.txt" % graphical_snap)
            if os.path.isfile(graphical_file_path):
                graphical_snaps.append(graphical_snap)
                time_steps.append(time_step)
                times.append(_time)
        self.all_graphical_snaps = graphical_snaps
        self.all_graphical_timesteps = time_steps 
        self.all_graphical_times = times
        self.options['ALL_AVAILABLE_GRAPHICAL_SNAPSHOTS'] = str(graphical_snaps)
        self.options['ALL_AVAILABLE_GRAPHICAL_TIMESTEPS'] = str(time_steps)
        self.options['ALL_AVAILABLE_GRAPHICAL_TIMES'] = str(times)
        particle_snaps, _, _ = GetSnapsSteps(self._case_dir, 'particle')
        self.options['ALL_AVAILABLE_PARTICLE_SNAPSHOTS'] = str(particle_snaps)
        particle_output_dir = os.path.join(self._output_dir, "slab_morphs")
        self.options["PARTICLE_OUTPUT_DIR"] = particle_output_dir
        # get the last step in the series
        try:
            self.last_step = max(0, graphical_snaps[-1] - int(self.options['INITIAL_ADAPTIVE_REFINEMENT']))  # it is the last step we have outputs
        except IndexError:
            # no snaps, stay on the safe side
            self.last_step = -1
        # add an option of the last step
        self.options["LAST_STEP"] = self.last_step

        # set steps to plot
        # Priority:
        #   1. a list of steps
        #   2. the last few steps
        #   3. only the last step
        if type(steps) == list:
            for step in steps:
                assert(type(step) == int)
            self.options['GRAPHICAL_STEPS'] = steps  # always plot the 0 th step
        elif type(time_interval) is float and time_interval > 0.0:
            times_ndarray = np.array(times)
            time_series_from_interval = np.arange(0.0, times[-1], time_interval, dtype=float)
            self.options['GRAPHICAL_STEPS'] = []
            for i in range(time_series_from_interval.size):
                _time = time_series_from_interval[i]
                idx = np.argmin(abs(times_ndarray - _time))
                self.options['GRAPHICAL_STEPS'].append(graphical_snaps[idx] - int(self.options['INITIAL_ADAPTIVE_REFINEMENT']))
        elif type(last_step) == int:
            # by this option, plot the last few steps
            self.options['GRAPHICAL_STEPS'] = [0]  # always plot the 0 th step
            self.options['GRAPHICAL_STEPS'] += [i for i in range(max(self.last_step - last_step + 1, 0), self.last_step + 1)]
        elif type(steps) == str and steps == "auto":
            # 
            # determine the options by the number of steps and slice them by the number of slices
            assert(slices > 0)
            self.options['GRAPHICAL_STEPS'] = [int(i) for i in np.linspace(0 , int(self.options["LAST_STEP"]), slices)]
            # self.options['GRAPHICAL_STEPS'].append(int(self.options["LAST_STEP"]))
        else:
            # by default append the first and the computing step.
            self.options['GRAPHICAL_STEPS'] = [0]
            if self.last_step > 0:
                self.options['GRAPHICAL_STEPS'].append(self.last_step)

        # get time steps
        self.options['GRAPHICAL_TIME_STEPS'] = []
        for step in self.options['GRAPHICAL_STEPS']:
            found = False
            for i in range(len(graphical_snaps)):
                if step == max(0, graphical_snaps[i] - int(self.options['INITIAL_ADAPTIVE_REFINEMENT'])):
                    found = True
                    self.options['GRAPHICAL_TIME_STEPS'].append(time_steps[i])
            if not found:
                warnings.warn("%s: step %d is not found" % (func_name(), step))

    def visit_options(self, extra_options):
        '''
        deprecated
        '''
        # optional settings
        for key, value in extra_options.items():
            # slab
            if key == 'slab':
                self.options['IF_PLOT_SLAB'] = 'True'
                self.options['GRAPHICAL_STEPS'] = value.get('steps', [0])
                self.options['IF_DEFORM_MECHANISM'] = value.get('deform_mechanism', 0)
            # export particles for slab morph
            elif key == 'slab_morph':
                self.options['IF_EXPORT_SLAB_MORPH'] = 'True'
                # check directory
                if not os.path.isdir(self.options["PARTICLE_OUTPUT_DIR"]):
                    os.mkdir(self.options["PARTICLE_OUTPUT_DIR"])
    
    def vtk_options(self, **kwargs):
        '''
        options of vtk scripts
        '''
        generate_horiz_file = kwargs.get('generate_horiz', False)
        operation = kwargs.get('operation', 'default')
        vtu_step = int(kwargs.get('vtu_step', 0))
        # houriz_avg file
        if generate_horiz_file:
            _time, time_step = self.get_time_and_step(vtu_step)
            depth_average_path = os.path.join(self.options["DATA_OUTPUT_DIR"], 'depth_average.txt')
            assert(os.path.isfile(depth_average_path))
            output_dir = os.path.join(self._case_dir, 'temp_output')
            try:  # This works better in parallel
                os.mkdir(output_dir)
            except FileExistsError:
                pass
            _, ha_output_file = ExportData(depth_average_path, output_dir, time_step=time_step, fix_time_step=True)
            self.options['VTK_HORIZ_FILE'] = ha_output_file
        else:
            self.options['VTK_HORIZ_FILE'] = None # os.path.join(ASPECT_LAB_DIR, 'output', 'depth_average_output')
        # directory to output from vtk script
        self.options['VTK_OUTPUT_DIR'] = os.path.join(self._case_dir, "vtk_outputs")
        if not os.path.isdir(self.options['VTK_OUTPUT_DIR']):
            os.mkdir(self.options['VTK_OUTPUT_DIR'])
        # file to read in vtk
        my_assert((vtu_step >= 0 and vtu_step <= self.last_step), ValueError, "vtu_step needs to be within the range of [%d, %d]" % (0, self.last_step))  # check the range of steps
        self.options['PVTU_FILE'] = os.path.join(self._output_dir, "solution", "solution-%05d.pvtu" % (vtu_step + int(self.options['INITIAL_ADAPTIVE_REFINEMENT'])))
        # type of operation
        self.options['OPERATION'] = operation

    def get_time_and_step(self, vtu_step):
        '''
        Convert vtu_step to step and time in model
        ''' 
        assert(len(self.all_graphical_snaps) > 0)
        assert(len(self.all_graphical_timesteps) > 0)
        # find step in all available steps
        found = False
        i = 0
        for snap_shot in self.all_graphical_snaps:
            if vtu_step == max(0, int(snap_shot) - int(self.options['INITIAL_ADAPTIVE_REFINEMENT'])):
                found = True
                step = int(self.all_graphical_timesteps[i])
            i += 1
        my_assert(found, ValueError, "%s: vtu_step %d is not found" % (func_name(), vtu_step))
        time = self.Statistics.GetTime(step)
        return time, step
    
    def get_time_and_step_by_snapshot(self, vtu_snapshot):
        '''
        Convert vtu_snapshot to step and time in model
        ''' 
        assert(len(self.all_graphical_snaps) > 0)
        assert(len(self.all_graphical_timesteps) > 0)
        # find step in all available steps
        found = False
        i = 0
        for snap_shot in self.all_graphical_snaps:
            if vtu_snapshot == snap_shot:
                found = True
                step = int(self.all_graphical_timesteps[i])
            i += 1
        my_assert(found, ValueError, "%s: vtu_snapshot %d is not found" % (func_name(), vtu_snapshot))
        time = self.Statistics.GetTime(step)
        return time, step

    def get_timestep_by_time(self, _time: float):
        '''
        Retrieves the closest graphical time and its corresponding timestep based on the given time.
        
        Parameters:
        _time (float): The reference time for which the closest graphical time and timestep are to be found.
        
        Returns:
        Tuple: A tuple containing the closest graphical time and its associated timestep.
        '''
        index = np.argmin(np.abs(np.array(self.all_graphical_times) - _time))
        return self.all_graphical_times[index], self.all_graphical_timesteps[index], self.all_graphical_snaps[index] - int(self.options['INITIAL_ADAPTIVE_REFINEMENT'])

class COMPOSITION():
    """
    store value like
      'background:4.0e-6|1.5e-6, spcrust:0.0, spharz:4.0e-6, opcrust:4.0e-6, opharz:4.0e-6 '
    or parse value back
    """
    def __init__(self, in_var):
        # parse the format:
        # key1: val1|val2, key2: val3|val4
        # to a dictionary data where
        # data[key1] = [val1, val2]
        if type(in_var) == str:
            self.data = {}
            parts = in_var.split(',')
            for part in parts:
                key_str = part.split(':')[0]
                key = re_neat_word(key_str)
                values_str = part.split(':')[1].split('|')
                # convert string to float
                try:
                    values = [float(re_neat_word(val)) for val in values_str]
                except ValueError:
                    values = [re_neat_word(val) for val in values_str]
                self.data[key] = values
        elif type(in_var) == type(self):
            self.data = in_var.data.copy()
        else:
            raise NotImplementedError()

    def parse_back(self):
        """
        def parse_back(self)

        parse data back to a string
        """
        line = ''
        j = 0
        for key, values in self.data.items():
            # construct the format:
            # key1: val1|val2, key2: val3|val4
            if j > 0:
                part_of_line = ', ' + key + ':'
            else:
                part_of_line = key + ':'
            i = 0
            for val in values:
                if i == 0:
                    if type(val) == float:
                        part_of_line += '%.4e' % val
                    elif type(val) == str:
                        part_of_line += val
                else:
                    if type(val) == float:
                        part_of_line += '|' + '%.4e' % val
                    elif type(val) == str:
                        part_of_line += val
                i += 1
            line += part_of_line
            j += 1
        return line


class VISIT_OPTIONS(VISIT_OPTIONS_BASE):
    """
    parse .prm file to a option file that bash can easily read
    """
    def Interpret(self, **kwargs):
        """
        Interpret the inputs, to be reloaded in children
        kwargs: options
            last_step(list): plot the last few steps
        """
        # call function from parent
        VISIT_OPTIONS_BASE.Interpret(self, **kwargs)

        # default settings
        self.options['IF_PLOT_SHALLOW'] = kwargs.get('if_plot_shallow', "False") # plot the shallow part of the slab.
        self.options["PLOT_TYPES"] = str(kwargs.get('plot_types', ["upper_mantle"]))
        self.options['IF_EXPORT_SLAB_MORPH'] = 'False'
        self.options['IF_PLOT_SLAB'] = 'True'
        self.options['GLOBAL_UPPER_MANTLE_VIEW_BOX'] = 0.0
        self.options['ROTATION_ANGLE'] = 0.0

        # additional inputs
        rotation_plus = kwargs.get("rotation_plus", 0.0) # additional rotation

        # try using the value for the background
        try:
            self.options['ETA_MIN'] =\
                string2list(self.idict['Material model']['Visco Plastic TwoD']['Minimum viscosity'], float)[0]
        except ValueError:
            eta_min_inputs =\
                COMPOSITION(self.idict['Material model']['Visco Plastic TwoD']['Minimum viscosity']) 
            self.options['ETA_MIN'] = eta_min_inputs.data['background'][0] # use phases
        try:
            self.options['ETA_MAX'] =\
                string2list(self.idict['Material model']['Visco Plastic TwoD']['Maximum viscosity'], float)[0]
        except ValueError:
            eta_max_inputs =\
                COMPOSITION(self.idict['Material model']['Visco Plastic TwoD']['Maximum viscosity']) 
            self.options['ETA_MAX'] = eta_max_inputs.data['background'][0] # use phases
        # self.options['IF_DEFORM_MECHANISM'] = value.get('deform_mechanism', 0)
        self.options['IF_DEFORM_MECHANISM'] = 1
        # crustal layers
        # todo_2l
        is_crust_2l = False
        composition_fields = []
        temp_list = self.idict['Compositional fields']['Names of fields'].split(",")
        for temp in temp_list:
            temp1 = re_neat_word(temp)
            if temp1 != "":
                composition_fields.append(temp1)
        if "spcrust_up" in composition_fields:
            is_crust_2l = True
        # currently only works for chunk geometry
        if self.options['GEOMETRY'] == 'chunk':
            sp_age = -1.0
            ov_age = -1.0
            Ro = 6371e3
            try:
                index = find_wb_feature(self.wb_dict, "Subducting plate")
                index1 = find_wb_feature(self.wb_dict, "Overiding plate")
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                sp_age = -1.0
                ov_age = -1.0
            else:
                feature_sp = self.wb_dict['features'][index]
                feature_ov = self.wb_dict['features'][index1]
                trench_angle = feature_sp["coordinates"][2][0]
                spreading_velocity = feature_sp["temperature models"][0]["spreading velocity"]
                sp_age = trench_angle * np.pi / 180.0 * Ro/ spreading_velocity 
                ov_age = feature_ov["temperature models"][0]["plate age"]
            self.options['SP_AGE'] = sp_age
            self.options['OV_AGE'] =  ov_age
        elif self.options['GEOMETRY'] == 'box':
            sp_age = -1.0
            ov_age = -1.0
            try:
                # todo_ptable
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
                index1 = find_wb_feature(self.wb_dict, "Overiding plate")
                feature_sp = self.wb_dict['features'][index]
                feature_ov = self.wb_dict['features'][index1]
                trench_x = feature_sp["coordinates"][2][0]
                spreading_velocity = feature_sp["temperature models"][0]["spreading velocity"]
                sp_age = trench_x / spreading_velocity 
                ov_age = feature_ov["temperature models"][0]["plate age"]
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                sp_age = -1.0
                ov_age = -1.0
                pass
            self.options['SP_AGE'] = sp_age
            self.options['OV_AGE'] =  ov_age
        else:
            raise ValueError("Geometry should be \"chunk\" or \"box\"")
        # rotation of the domain
        if self.options['GEOMETRY'] == 'chunk':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                rotation_angle = 52.0 + rotation_plus
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                rotation_angle = 90.0 - feature_sp["coordinates"][2][0] - 2.0 + rotation_plus
            self.options['ROTATION_ANGLE'] = rotation_angle
        elif self.options['GEOMETRY'] == 'box':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                trench_x = 4e6
            else:
                # reset the view point
                feature_sp = self.wb_dict['features'][index]
                trench_x = feature_sp["coordinates"][2][0]
            window_width = 1.8e6
            self.options['GLOBAL_UPPER_MANTLE_VIEW_BOX'] =\
            "(%.4e, %.4e, 1.9e6, 2.9e6)" % (trench_x - window_width/2.0, trench_x + window_width/2.0)
        else:
            raise ValueError("Geometry should be \"chunk\" or \"box\"")
        # Slab configuration
        index = find_wb_feature(self.wb_dict, 'Subducting plate')
        feature_sp = self.wb_dict['features'][index]
        # shear zone:
        #   the initial thickness is parsed from the wb file
        #   parse the cutoff depth if the viscosity is decoupled from the eclogite transition
        use_lookup_table_morb = self.idict['Material model']['Visco Plastic TwoD'].get("Use lookup table morb", 'false')
        sz_method = 0
        if use_lookup_table_morb == 'true':
            phase_rheology_mixing_models_str = self.idict['Material model']['Visco Plastic TwoD'].get('Phase rheology mixing models', "0, 0, 0, 0, 0")
            phase_rheology_mixing_models = string2list(phase_rheology_mixing_models_str, int)
            sz_method = phase_rheology_mixing_models[1]
        elif use_lookup_table_morb == 'false':
            pass
        else:
            raise ValueError()
        self.options["SHEAR_ZONE_METHOD"] = sz_method
        self.options["INITIAL_SHEAR_ZONE_THICKNESS"] = feature_sp["composition models"][0]["max depth"]
        decoupling_eclogite_viscosity = self.idict['Material model']['Visco Plastic TwoD'].get('Decoupling eclogite viscosity', 'false')
        if decoupling_eclogite_viscosity == 'true':
            self.options["SHEAR_ZONE_CUTOFF_DEPTH"] = float(self.idict['Material model']['Visco Plastic TwoD']["Eclogite decoupled viscosity"]["Decoupled depth"])
        else:
            self.options["SHEAR_ZONE_CUTOFF_DEPTH"] = -1.0
        #  the shear zone constant viscosity is calculated from the prefactor of spcrust
        A_diff_inputs = COMPOSITION(self.idict['Material model']['Visco Plastic TwoD']['Prefactors for diffusion creep'])
        # todo_2l
        if is_crust_2l:
            self.options["N_CRUST"] = 2
            self.options["SHEAR_ZONE_CONSTANT_VISCOSITY"] = 1.0 / 2.0 / A_diff_inputs.data['spcrust_up'][0] # use phases
        else:
            self.options["N_CRUST"] = 1
            self.options["SHEAR_ZONE_CONSTANT_VISCOSITY"] = 1.0 / 2.0 / A_diff_inputs.data['spcrust'][0] # use phases
        # yield stress
        try:
            self.options["MAXIMUM_YIELD_STRESS"] = float(self.idict['Material model']['Visco Plastic TwoD']["Maximum yield stress"])
        except KeyError:
            self.options["MAXIMUM_YIELD_STRESS"] = 1e8

        # peierls rheology
        try:
            include_peierls_rheology = self.idict['Material model']['Visco Plastic TwoD']['Include Peierls creep']
            if include_peierls_rheology == 'true':
                self.options['INCLUDE_PEIERLS_RHEOLOGY'] = True
            else:
                self.options['INCLUDE_PEIERLS_RHEOLOGY'] = False
        except KeyError:
            self.options['INCLUDE_PEIERLS_RHEOLOGY'] = False

        # reference trench point
        self.options['THETA_REF_TRENCH'] = 0.0  # initial value
        if self.options['GEOMETRY'] == 'chunk':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                theta_ref_trench = 0.63
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                theta_ref_trench = feature_sp["coordinates"][2][0] / 180.0 * np.pi
        elif self.options['GEOMETRY'] == 'box':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                # for the box geometry, this is the x distance of the trench
                theta_ref_trench = 4000000.0
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                theta_ref_trench = feature_sp["coordinates"][2][0]        
        else:
            raise ValueError("Value of geometry must be either \"chunk\" or \"box\"")
        self.options['THETA_REF_TRENCH'] = theta_ref_trench

    def vtk_options(self, **kwargs):
        '''
        options of vtk scripts
        '''
        # call function from parent
        VISIT_OPTIONS_BASE.vtk_options(self, **kwargs)
        # reference trench point
        self.options['THETA_REF_TRENCH'] = 0.0  # initial value
        if self.options['GEOMETRY'] == 'chunk':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                theta_ref_trench = 0.63
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                theta_ref_trench = feature_sp["coordinates"][2][0] / 180.0 * np.pi
        elif self.options['GEOMETRY'] == 'box':
            try:
                index = find_wb_feature(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                # for the box geometry, this is the x distance of the trench
                theta_ref_trench = 4000000.0
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                theta_ref_trench = feature_sp["coordinates"][2][0]        
        else:
            raise ValueError("Value of geometry must be either \"chunk\" or \"box\"")
        self.options['THETA_REF_TRENCH'] = theta_ref_trench


    def get_snaps_for_slab_morphology(self, **kwargs):
        '''
        get the snaps for processing slab morphology
        kwargs (dict):
            time_interval (float)
        '''
        ptime_start = kwargs.get('time_start', None)
        ptime_interval = kwargs.get('time_interval', None)
        ptime_end = kwargs.get('time_end', None)
        assert(ptime_interval is None or type(ptime_interval) == float)      
        # steps for processing slab morphology
        snaps, times, _ = GetSnapsSteps(self._case_dir, 'graphical')
        psnaps = []
        ptimes = []
        last_time = -1e8  # initiate as a small value, so first step is included
        # loop within all the available steps, find steps satisfying the time interval requirement.
        for i in range(len(times)):
            time = times[i]
            snap = snaps[i]
            if ptime_start is not None and time < ptime_start:
                # check on the start
                continue
            if ptime_end is not None and time > ptime_end:
                # check on the end
                break
            if type(ptime_interval) == float:
                if (time - last_time) < ptime_interval:
                    continue  # continue if interval is not reached
            pvtu_file_path = os.path.join(self.options["DATA_OUTPUT_DIR"], "solution", "solution-%05d.pvtu" % snap)
            if os.path.isfile(pvtu_file_path):
                # append if the file is found
                last_time = time
                psnaps.append(snap)
                ptimes.append(time)
        return psnaps

    def get_times_for_slab_morphology(self, **kwargs):
        '''
        get the snaps for processing slab morphology
        kwargs (dict):
            time_interval (float)
        '''
        ptime_start = kwargs.get('time_start', None)
        ptime_interval = kwargs.get('time_interval', None)
        ptime_end = kwargs.get('time_end', None)
        assert(ptime_interval is None or type(ptime_interval) == float)      
        # steps for processing slab morphology
        snaps, times, _ = GetSnapsSteps(self._case_dir, 'graphical')
        psnaps = []
        ptimes = []
        last_time = -1e8  # initiate as a small value, so first step is included
        # loop within all the available steps, find steps satisfying the time interval requirement.
        for i in range(len(times)):
            time = times[i]
            snap = snaps[i]
            if ptime_start is not None and time < ptime_start:
                # check on the start
                continue
            if ptime_end is not None and time > ptime_end:
                # check on the end
                break
            if type(ptime_interval) == float:
                if (time - last_time) < ptime_interval:
                    continue  # continue if interval is not reached
            pvtu_file_path = os.path.join(self.options["DATA_OUTPUT_DIR"], "solution", "solution-%05d.pvtu" % snap)
            if os.path.isfile(pvtu_file_path):
                # append if the file is found
                last_time = time
                psnaps.append(snap)
                ptimes.append(time)
        return ptimes
    
    def get_snaps_for_slab_morphology_outputs(self, **kwargs):
        '''
        get the snaps for processing slab morphology, look for existing outputs
        kwargs (dict):
            time_interval (float)
        '''
        ptime_start = kwargs.get('time_start', None)
        ptime_interval = kwargs.get('time_interval', None)
        ptime_end = kwargs.get('time_end', None)
        assert(ptime_interval is None or type(ptime_interval) == float)      
        # steps for processing slab morphology
        snaps, times, _ = GetSnapsSteps(self._case_dir, 'graphical')
        psnaps = []
        ptimes = []
        last_time = -1e8  # initiate as a small value, so first step is included
        # loop within all the available steps, find steps satisfying the time interval requirement.
        for i in range(len(times)):
            time = times[i]
            snap = snaps[i]
            if ptime_start is not None and time < ptime_start:
                # check on the start
                continue
            if ptime_end is not None and time > ptime_end:
                # check on the end
                break
            if type(ptime_interval) == float:
                if (time - last_time) < ptime_interval:
                    continue  # continue if interval is not reached
            center_profile_file_path = os.path.join(self._case_dir, "vtk_outputs", "slab_morph_s%06d.txt" % snap)
            if os.path.isfile(center_profile_file_path):
                # append if the file is found
                last_time = time
                psnaps.append(snap)
                ptimes.append(time)
        return ptimes, psnaps


class LINEARPLOT():
    '''
    LINEARPLOT():
    A class for plotting linear plots based on input data and user-defined options.

    Attributes:
        name (str): Name of the plot configuration.
        options (dict): Dictionary containing plotting options, read from JSON.
        UnitConvert (function or None): Optional function to handle unit conversion.
        dim (int): Dimensionality of the data (must be 1, 2, or 3).
        header (dict): Parsed header information from the input file.
        data (ndarray): Loaded data from file.
        prm (dict): Parameters parsed from a .prm input file.
    '''
    def __init__(self, _name, options={}):
        '''
        Initialize the LINEARPLOT class with a given name and options.

        Parameters:
            _name (str): Name of the plotting task.
            options (dict): Configuration dictionary. Keys include:
                - 'unit_convert' (function): Function for converting units.
                - 'dim' (int): Dimension of the data.
                - 'options' (dict): Overrides for JSON options.
        '''
        self.name = _name
        self.options = JsonOptions(_name, JSON_FILE_DIR)
        self.UnitConvert = options.get('unit_convert', None)
        self.dim = options.get('dim', 2)  # dimension
        assert(self.dim in [1, 2, 3])  # dimension must be 1, 2, 3

        # reset the options with a option in the options
        json_file_path = os.path.join(JSON_FILE_DIR, "post_process_std.json")
        with open(json_file_path, 'r') as fin:
            all_options = json.load(fin)
        self.options = all_options.get(self.name, {})
        try:
            options = options['options']
        except KeyError:
            pass
        else:
            self.options.update(options)

    def __call__(self, _filename, **kwargs):
        '''
        Make the class callable: reads data from a file and generates a plot.

        Parameters:
            _filename (str): Path to the input data file.
            **kwargs:
                fileout (str): Output filename for the plot.

        Returns:
            _fileout (str): Name of the output file, or None if data is invalid.
        '''
        _fileout = kwargs.get('fileout', _filename + '.pdf')
        self.ReadHeader(_filename)
        state = self.ReadData(_filename)
        if state == 1:
            return None
        _data_list = self.ManageData()
        _fileout = self.PlotCombine(_data_list, _fileout)
        return _fileout

    class DataNotFoundWarning(UserWarning):
        '''
        Warning class used when a requested data field is missing but the plotting proceeds.
        '''
        pass

    def ReadHeader(self, _filename):
        '''
        Read header information from the data file.

        Parameters:
            _filename (str): Path to the data file.
        '''
        if not os.access(_filename, os.R_OK):
            raise FileExistsError("%s cannot be read." % _filename)
        with open(_filename, 'r') as fin:
            _texts = fin.readlines()
        try:
            self.header = ReadHeader(_texts)
        except:
            raise Exception('Header for file %s cannot be read' % _filename)

    def SortHeader(self):
        '''
        Sort the header by column index.

        Returns:
            tuple: Sorted arrays of column indices, field names, and units.
        '''
        names = []
        cols = []
        units = []
        for key, value in self.header.items():
            if key == 'total_col':
                continue
            names.append(key)
            cols.append(int(value['col']))
            units.append(value['unit'])
        names = np.array(names)
        cols = np.array(cols)
        units = np.array(units)
        sort_indexes = np.argsort(cols)
        names = names[sort_indexes]
        cols = cols[sort_indexes]
        units = units[sort_indexes]
        return cols, names, units

    def ReadData(self, _filename, **kwargs):
        '''
        Read data from the file using NumPy.

        Parameters:
            _filename (str): Path to the data file.
            dtype (type): Data type, either float or str (default is float).

        Returns:
            int: 1 if file is empty or unreadable, 0 otherwise.
        '''
        dtype = kwargs.get('dtype', float)
        if not os.access(_filename, os.R_OK):
            raise FileExistsError("%s cannot be read." % _filename)
        with warnings.catch_warnings(record=True) as w:
            if dtype == float:
                self.data = np.genfromtxt(_filename, comments='#', filling_values=0.0)
            elif dtype == str:
                self.data = np.loadtxt(_filename, comments='#', dtype=str)
            else:
                raise ValueError("dtype must be float or str")
            if (len(w) > 0):
                assert(issubclass(w[-1].category, UserWarning))
                assert('Empty input file' in str(w[-1].message))
                warnings.warn('ReadData: %s, abort' % str(w[-1].message))
                return 1
        if len(self.data.shape) == 1:
            self.data = np.array([self.data])
        return 0

    def ReadPrm(self, prm_file):
        '''
        Read parameter file from deal.II format.

        Parameters:
            prm_file (str): Path to the .prm file.
        '''
        assert(os.path.isfile(prm_file))
        with open(prm_file, 'r') as fin:
            self.prm = parse_parameters_to_dict(fin)

    def ManageData(self):
        '''
        Manage and organize raw data into a list for plotting.

        Returns:
            list: List of arrays for each data column.
        '''
        _data_list = []
        for i in range(self.data.shape[1]):
            _data_list.append(self.data[:, i])
        return _data_list

    def Has(self, field_name):
        '''
        Check if a field exists in the data header.

        Parameters:
            field_name (str): Field name to check.

        Returns:
            bool: True if field exists, False otherwise.
        '''
        has_field = (field_name in self.header)
        return has_field

    def HasData(self):
        '''
        Check if the dataset contains any data.

        Returns:
            bool: True if data exists, False otherwise.
        '''
        if (self.data.size == 0):
            return False
        else:
            return True

    def PlotCombine(self, _data_list, _fileout, **kwargs):
        '''
        Plot combined subplots using predefined canvas and types.

        Parameters:
            _data_list (list): List of data arrays.
            _fileout (str): Output filename for the plot.
            title (str): Optional plot title.

        Returns:
            str: Output filename of the saved plot.
        '''
        assert(type(_data_list) is list)
        _canvas = self.options.get('canvas', [1, 1])
        assert(type(_canvas) is list and len(_canvas) == 2)
        _types = self.options.get('types', [])
        assert(type(_types) is list and
            _canvas[0] * _canvas[1] >= len(_types))
        _size = self.options.get('size', (5, 5))
        _title = kwargs.get('title', None)

        fig, axs = plt.subplots(_canvas[0], _canvas[1], figsize=_size)
        for i in range(len(_types)):
            if _canvas[0] == 1 and _canvas[1] == 1:
                _ax = axs
            elif _canvas[0] == 1 or _canvas[1] == 1:
                _ax = axs[i]
            else:
                _id1 = i // _canvas[1]
                _id2 = i % _canvas[1]
                _ax = axs[_id1, _id2]
            _type = _types[i]
            if type(_type) == str:
                _opt = self.options[_type]
                self.Plot(_data_list, _ax, _opt)
            elif type(_type) == list:
                for _subtype in _type:
                    assert(type(_subtype) == str)
                    _opt = self.options[_subtype]
                    self.Plot(_data_list, _ax, _opt)
        fig.tight_layout()
        fig.savefig(_fileout)
        plt.close(fig)
        return _fileout

    def Plot(self, _data_list, _ax, _opt):
        '''
        Plot a single subplot using given axis and options.

        Parameters:
            _data_list (list): List of data arrays.
            _ax (matplotlib.axes): Axis object to plot on.
            _opt (dict): Dictionary of plot options.

        Returns:
            matplotlib.axes: Modified axis object.
        '''
        _xname = _opt.get('xname', 'Time')
        _yname = _opt.get('yname', 'Number_of_mesh_cells')
        _color = _opt.get('color', 'r')
        _label = _opt.get('label', None)
        _line = _opt.get('line', '-')
        _invert_x = _opt.get('invert_x', 0)
        _invert_y = _opt.get('invert_y', 0)
        _log_x = _opt.get('log_x', 0)
        _log_y = _opt.get('log_y', 0)
        _colx = self.header[_xname]['col']
        try:
            _coly = self.header[_yname]['col']
        except KeyError:
            warnings.warn('The field %s doesn\'t exist. We will keep ploting,\
but you will get a blank one for this field name' % _yname,
                           self.DataNotFoundWarning)
            return
        _unitx = self.header[_xname]['unit']
        _unity = self.header[_yname]['unit']
        _x = _data_list[_colx]
        _y = _data_list[_coly]
        _unit_x_plot = _opt.get('xunit', _unitx)
        _unit_y_plot = _opt.get('yunit', _unity)
        if self.UnitConvert is not None and _unit_x_plot != _unitx:
            x_convert_ratio = self.UnitConvert(_unitx, _unit_x_plot)
        else:
            x_convert_ratio = 1.0
        if self.UnitConvert is not None and _unit_y_plot != _unity:
            y_convert_ratio = self.UnitConvert(_unity, _unit_y_plot)
        else:
            y_convert_ratio = 1.0
        if _unit_x_plot is not None:
            _xlabel = re.sub("_", " ", _xname) + ' (' + _unit_x_plot + ')'
        else:
            _xlabel = re.sub("_", " ", _xname)
        if _unit_y_plot is not None:
            _ylabel = re.sub("_", " ", _yname) + ' (' + _unit_y_plot + ')'
        else:
            _ylabel = re.sub("_", " ", _yname)
        if _log_x and _log_y:
            _ax.loglog(_x*x_convert_ratio, _y*y_convert_ratio, _line, color=_color, label=_label)
        elif _log_x:
            _ax.semilogx(_x*x_convert_ratio, _y*y_convert_ratio, _line, color=_color, label=_label)
        elif _log_y:
            _ax.semilogy(_x*x_convert_ratio, _y*y_convert_ratio, _line, color=_color, label=_label)
        else:
            _ax.plot(_x*x_convert_ratio, _y*y_convert_ratio, _line, color=_color, label=_label)
        _ax.set(xlabel=_xlabel, ylabel=_ylabel)
        if _invert_x and ~_ax.xaxis_inverted():
            _ax.invert_xaxis()
        if _invert_y and ~_ax.yaxis_inverted():
            _ax.invert_yaxis()
        if _label is not None:
            _ax.legend()

    def export(self, output_path, names, **kwargs):
        '''
        Export specified fields to a text file.

        Parameters:
            output_path (str): Path for output file.
            names (list of str): Field names to export.
            rows (list, optional): Row indices to export.
            include_size (bool): Whether to include array shape in the output.
            data_only (bool): If True, return data but skip writing to file.

        Returns:
            ndarray: Exported data.
        '''
        my_assert(type(names) == list, TypeError, "%s: names must be a list" % func_name())
        cols = []
        for _name in names:
            cols.append(int(self.header[_name]['col']))
        rows = kwargs.get("rows", None)
        if rows != None:
            my_assert((type(rows) == list), TypeError, "%s: rows must be a list" % func_name())
            odata = self.data[np.ix_(rows, cols)]
        else:
            odata = self.data[:, cols]
        include_size=kwargs.get('include_size', False)
        data_only = kwargs.get('data_only', False)
        if not data_only:
            with open(output_path, 'w') as fout:
                i = 1
                for _name in names:
                    fout.write("# %d: %s\n" % (i, _name))
                    i += 1
                if include_size:
                    fout.write("%d %d\n" % (odata.shape[0], odata.shape[1]))
                np.savetxt(fout, odata, fmt='%-20.8e')
            print('Export data to %s' % output_path)
            print('\tData layout: ', odata.shape)
        return odata

    def export_field_as_array(self, _name):
        '''
        Export a single field from the data as a list.

        Parameters:
            _name (str): Name of the field to extract.

        Returns:
            list: Field values as a list, or empty list if not found.
        '''
        try:
            col = int(self.header[_name]['col'])
        except KeyError:
            o_array = []
        else:
            temp = self.data[:, col]
            o_array = temp.tolist()
        return o_array

class STATISTICS_PLOT(LINEARPLOT):
    '''
    Class for plotting depth average file.
    This is an inheritage of the LINEARPLOT class

    Attributes:
    Args:
    '''
    def __init__(self, _name, **kwargs):
        LINEARPLOT.__init__(self, _name, kwargs)  # call init from base function
    
    def GetStep(self, time):
        '''
        Inputs:
            time(double)
        get step corresponding to a value of model time
        '''
        # get data
        col_t = self.header['Time']['col']
        col_step = self.header['Time_step_number']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]

        # get step
        idx = np.argmin(abs(times - time))
        step = int(steps[idx])
        return step
    
    def GetTime(self, step):
        '''
        Inputs:
            step(int)
        get time to a value of model step
        '''
        col_t = self.header['Time']['col']
        col_step = self.header['Time_step_number']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        assert(len(times) == len(steps))
        time = 0.0
        found = False
        for i in range(len(steps)):  # search for step
            if step == int(steps[i]):
                time = times[i]
                found = True
        my_assert(found, ValueError, "step %d is not a valid step" % step)
        return time

    def GetLastStep(self):
        '''
        get step and time of the last time step
        Return:
            last step, model time of the last step
        '''
        # get data
        col_t = self.header['Time']['col']
        col_step = self.header['Time_step_number']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        idx = np.argmax(steps)
        return int(steps[idx]), times[idx]

    def PlotNumberOfCells(self, **kwargs):
        '''
        plot the number of cells
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_noc = self.header['Number_of_mesh_cells']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        nocs = self.data[:, col_noc]
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, nocs, label=label, color=color)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Number of mesh cells')
        pass
    
    def PlotNumberOfNonlinearIterations(self, **kwargs):
        '''
        plot the number of cells
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_noni = self.header['Number_of_nonlinear_iterations']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        nonis = self.data[:, col_noni]
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, nonis, '.', label=label, color=color)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Number of nonlinear iterations')
    
    def PlotNumberOfIterationsStokessolver(self, **kwargs):
        '''
        plot the number of cells
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_iter_stokes = self.header['Iterations_for_Stokes_solver']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        iter_stokes = self.data[:, col_iter_stokes]
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, iter_stokes, '.', label=label, color=color)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Number of Iterations for Stokes solver')
    
    def PlotDegreeOfFreedom(self, **kwargs):
        '''
        plot the number of cells
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        label_all = kwargs.get('label_all', False) # figure out labels
        if label_all:
            label_stokes = "(stokes)"
            label_temperature = "(temperature)"
            label_composition = "(composition)"
            if label == None:
                label_total = "(total)"
            else:
                label_total = label + " (total)"
        else:
            label_stokes = None
            label_temperature = None
            label_composition = None
            label_total = label
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_dof_stokes = self.header['Number_of_Stokes_degrees_of_freedom']['col']
        col_dof_temperature = self.header['Number_of_temperature_degrees_of_freedom']['col']
        col_dof_composition = self.header['Number_of_degrees_of_freedom_for_all_compositions']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        dofs_stokes = self.data[:, col_dof_stokes]
        dofs_temperature = self.data[:, col_dof_temperature]
        dofs_composition = self.data[:, col_dof_composition]
        dofs_total = dofs_stokes + dofs_temperature + dofs_composition
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, dofs_total, '-', color=color, label=label_total)
        ax.plot(times * to_myr, dofs_stokes, ':', color=color, label=label_stokes)
        ax.plot(times * to_myr, dofs_temperature, '--', color=color, label=label_temperature)
        ax.plot(times * to_myr, dofs_composition, '-.', color=color, label=label_composition)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Number of degree of freedoms')
        pass
    
    def PlotTemperature(self, **kwargs):
        '''
        plot the number of cells
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        label_all = kwargs.get('label_all', False) # figure out labels
        if label_all:
            label_average = "(average temperature)"
            label_minimum = "(minimum temperature)"
            label_maximum = "(maximum temperature)"
            if label == None:
                label_average = "(average temperature)"
            else:
                label_average = label + "(average temperature)"
        else:
            label_average = None
            label_minimum = None
            label_maximum = None
            label_average = label
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_min_T = self.header['Minimal_temperature']['col']
        col_avg_T = self.header['Average_temperature']['col']
        col_max_T = self.header['Maximal_temperature']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        min_Ts = self.data[:, col_min_T]
        avg_Ts = self.data[:, col_avg_T]
        max_Ts = self.data[:, col_max_T]
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, avg_Ts, '-', color=color, label=label_average)
        ax.plot(times * to_myr, min_Ts, '-.', color=color, label=label_minimum)
        ax.plot(times * to_myr, max_Ts, '--', color=color, label=label_maximum)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Temperature (K)')
        pass
    
    def PlotVelocity(self, **kwargs):
        '''
        plot the velocity outputs
        '''
        ax = kwargs.get('axis', None)
        label = kwargs.get('label', None)
        label_all = kwargs.get('label_all', False) # figure out labels
        if label_all:
            label_maximum = "(maximum velocity)"
            if label == None:
                label_rms = "(rms velocity)"
            else:
                label_rms = label + "(rms velocity)"
        else:
            label_maximum = None
            label_rms = label
        color = kwargs.get('color', None)
        if ax == None:
            raise ValueError("Not implemented")
        col_t = self.header['Time']['col']
        unit_t = self.header['Time']['unit']
        col_step = self.header['Time_step_number']['col']
        col_rms_V = self.header['RMS_velocity']['col']
        unit_V = self.header['RMS_velocity']['unit']
        col_max_V = self.header['Max._velocity']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]
        rms_Vs = self.data[:, col_rms_V]
        max_Vs = self.data[:, col_max_V]
        if self.UnitConvert is not None:
            to_myr = self.UnitConvert(unit_t, 'myr')
        else:
            raise ValueError("a UNITCONVERT class must be given")
        ax.plot(times * to_myr, rms_Vs, '-', color=color, label=label_rms)
        ax.plot(times * to_myr, max_Vs, '--', color=color, label=label_maximum)
        ax.set_xlabel('Time (myr)')
        ax.set_ylabel('Velocity (%s)' % unit_V)

class STATISTICS_PLOT_OLD(LINEARPLOT):
    '''
    Class for plotting depth average file.
    This is an inheritage of the LINEARPLOT class

    Attributes:
    Args:
    '''
    def __init__(self, _name, **kwargs):
        LINEARPLOT.__init__(self, _name, kwargs)  # call init from base function

    def GetStep(self, time):
        '''
        Inputs:
            time(double)
        get step corresponding to a value of model time
        '''
        # get data
        col_t = self.header['Time']['col']
        col_step = self.header['Time_step_number']['col']
        times = self.data[:, col_t]
        steps = self.data[:, col_step]

        # get step
        idx = np.argmin(abs(times - time))
        step = int(steps[idx])
        return step

    def GetTime(self, step):
        '''
        future
        Inputs:
            step(int)
        get time to a value of model step
        '''
        time = 0.0
        return time

class DEPTH_AVERAGE_PLOT(LINEARPLOT):
    '''
    Class for plotting depth average file.
    This is an inheritage of the LINEARPLOT class

    Attributes:
    Args:
    '''
    def __init__(self, _name, **kwargs):
        LINEARPLOT.__init__(self, _name, kwargs)  # call init from base function
        self.time_step_length = None
        # both these two arrays have the length of total time steps
        # the first records the time for each time step
        # the second points to the actual step within data
        self.time_step_times = None
        self.time_step_indexes = None

    def __call__(self, _filename, **kwargs):
        '''
        Read and plot
        Attributes:
            _filename(string):
                filename for data file
        Returns:
            _fileout(string or list):
                filename for output figure
        future:
            add in option for unit
        '''
        _fileout = kwargs.get('fileout', _filename + '.pdf')
        _time = kwargs.get('time', 'last')  # default is 'last' which means the last step
        self.ReadHeader(_filename)  # inteprate header information
        self.ReadData(_filename)  # read data
        self.ManageUnits()  # mange unit to output
        self.SplitTimeStep()  # split time step data
        # work out the name of output files
        _fname_base = _fileout.rpartition('.')[0]
        _fname_type = _fileout.rpartition('.')[2]
        _fname_list = []  # initialize a list for return
        if type(_time) in [float, int]:
            _time_list = [_time]
        elif type(_time) in [list, np.ndarray]:
            _time_list = _time
        else:
            raise TypeError('type of time needs to be in [float, int, list, np.ndarrayy]')
        for _t in _time_list:
            if type(_t) not in [float, int]:
                raise TypeError('type of values in time needs to be in [float, int, list, np.ndarrayy]')
            _data_list = self.ManageData(_t)  # manage output data
            _t_in_myr = _t * self.UnitConvert(self.header['time']['unit'], 'myr')
            _fname = "%s_t%.8e.%s" % (_fname_base, _t_in_myr, _fname_type)
            _figure_title = "Detph Average, t = %.10f myr" % _t_in_myr
            _fname = self.PlotCombine(_data_list, _fname, title=_figure_title)
            _fname_list.append(_fname)
        if len(_fname_list) == 0:
            # if there is only one name, just return this name
            return _fname_list[0]
        else:
            return _fname_list
        
    def ReadDataStep(self, _filename, **kwargs):
        '''
        Read data of a time step, currently only read the first time step.
        Attributes:
            _filename(string):
                filename for data file~/ASPECT_PROJECT/TwoDSubduction/non_linear32/eba1_MRf12_iter20_DET660/output$
        Returns:
            _datalist:
                a list of data for
        future:
            add in option for unit
        '''
        _fileout = kwargs.get('fileout', _filename + '.pdf')
        self.ReadHeader(_filename)  # inteprate header information
        self.ReadData(_filename)  # read data
        self.ManageUnits()  # mange unit to output
        self.SplitTimeStep()  # split time step data
        
        _t = 0.0  # t is the time of the 0th tep
        if type(_t) not in [float, int]:
            raise TypeError('type of values in time needs to be in [float, int, list, np.ndarrayy]')
        _data_list = self.ManageData(_t)  # manage output data

        data_type = kwargs.get('datatype', None)
        if data_type is None:
            return _data_list
        else:
            _data_list_o = []
            for _type in data_type:
                col = self.header[_type]['col']
                _data_list_o.append(_data_list[col])
            return _data_list_o
    
    def ExportDataByTime(self, time, names):
        '''
        Export data as ndarray by time and names
        '''
        assert(type(names)==list)
        time_step = np.argmin(abs(self.time_step_times - time))  # time_step
        i0 = self.time_step_indexes[time_step][-1] * self.time_step_length
        if time_step == len(self.time_step_times) - 1:
            # this is the last step
            i1 = self.data.shape[0]
        else:
            i1 = self.time_step_indexes[time_step + 1][0] * self.time_step_length
        odata = self.export("", names, rows=[i for i in range(i0, i1)], include_size=True, data_only=True)
        return odata, self.time_step_times[time_step]


    def ReadHeader(self, _filename):
        '''
        Read header information from file.
        overload base function, use ReadHeader2
        function in py
        Args:
            _filename(str):
                filename for data file
        '''
        assert(os.access(_filename, os.R_OK))
        with open(_filename, 'r') as fin:
            _texts = fin.readlines()  # read the text of the file header
        self.header = ReadHeader2(_texts)


    def Import(self, _filename):
        '''
        Combine a few functions to read data, header, as well as split
        data to steps
        '''
        self.ReadHeader(_filename)
        self.ReadData(_filename)
        self.SplitTimeStep()


    def SplitTimeStep(self):
        '''
        split time steps, since the data is a big chunck
        '''
        time_step_times = []  # initialize
        time_step_indexes = []
        _col_time = self.header['time']['col']
        _col_depth = self.header['depth']['col']
        _times = self.data[:, _col_time]
        _depths = self.data[:, _col_depth]
        # get the length of a single time step
        for i in range(1, _depths.size):
            if _depths[i] < _depths[i-1]:
                self.time_step_length = i
                break
            elif i == _depths.size - 1:
                # as the exiting value from python is simply _depths.size - 1
                self.time_step_length = i + 1
        # make a ndarray of different value of time
        _step_times = [_times[_idx] for _idx in range(0, _times.size, self.time_step_length)]
        i = 0  # first sub list for first step
        time_step_times.append(_step_times[0])
        time_step_indexes.append([0])
        # loop to group data at the same step
        for j in range(1, len(_step_times)):
            _time = _step_times[j]
            if abs(_time - _step_times[j-1]) > 1e-16:
                time_step_indexes.append([])
                time_step_times.append(_time)
                i += 1
            time_step_indexes[i].append(j)
        # both these two arrays have the length of total time steps
        # the first records the time for each time step
        # the second points to the actual step within data
        self.time_step_times = np.array(time_step_times)
        self.time_step_indexes = time_step_indexes
    
    def ManageData(self, _time):
        '''
        manage data, get new data for this class
        Returns:
            _data_list(list):
                list of data for ploting
            _time(float):
                time of plotting
        '''
        _data_list = []
        _time_step = np.argmin(abs(self.time_step_times - _time))  # time_step
        _index0 = self.time_step_indexes[_time_step][-1] * self.time_step_length
        if _time_step == len(self.time_step_times) - 1:
            # this is the last step
            _index1 = self.data.shape[0]
        else:
            _index1 = self.time_step_indexes[_time_step + 1][0] * self.time_step_length
        for i in range(self.data.shape[1]):
            _data_list.append(self.data[_index0 : _index1, i])
        # get the super adiabatic temperature
        _col_temperature = self.header['temperature']['col']
        _col_adiabatic_temperature = self.header['adiabatic_temperature']['col']
        _super_adiabatic_temperature = self.data[_index0 : _index1, _col_temperature] - self.data[_index0 : _index1, _col_adiabatic_temperature]
        _data_list.append(_super_adiabatic_temperature)
        self.header['super_adiabatic_temperature'] = {}
        self.header['super_adiabatic_temperature']['col'] = self.header['total_col']
        self.header['super_adiabatic_temperature']['unit'] = 'K'
        self.header['total_col'] += 1
        return _data_list
    
    def ManageUnits(self):
        '''
        manage units, get units for data.
        This is due to the bad form of the header of this file
        '''
        self.header['time']['unit'] = 'yr'
        self.header['depth']['unit'] = 'm'
        self.header['temperature']['unit'] = 'K'
        self.header['adiabatic_temperature']['unit'] = 'K'
        self.header['viscosity']['unit'] = 'Pa s'
        self.header['velocity_magnitude']['unit'] = 'm/yr'
        if self.dim == 2:
            self.header['vertical_heat_flux']['unit'] = 'mw/m'
        elif self.dim == 3:
            self.header['vertical_heat_flux']['unit'] = 'mw/m^2'

    def GetInterpolateFunc(self, time, field_name):
        names = ["depth", field_name]
        odata, _ = self.ExportDataByTime(time, names)
        _func = interp1d(odata[:, 0], odata[:, 1], assume_sorted=True, fill_value="extrapolate")
        return _func
    
    def GetIntegrateArray(self, time, field_name, dim, geometry, geometry_length, geometry_width = None, **kwargs):
        '''
        Returns:
            integretions - an array of added volume * field
            segmentations - an array of volume * field
        Note the data at depth 0 has to be fixed in some way (as depth 0 is not in the depth_average files)
        '''
        # initiating
        my_assert(geometry in ["cartesian", "spherical"], ValueError,\
            "geometry must be either cartesian or spherical")
        Ro = kwargs.get('Ro', 6371e3)
        # get raw data
        names = ["depth", field_name]
        odata, _ = self.ExportDataByTime(time, names)
        depths = odata[:, 0]
        vals = odata[:, 1]
        # compute integretion
        segmentations = np.zeros(self.time_step_length)
        integretions = np.zeros(self.time_step_length)
        if geometry == "cartesian" and dim == 2:
            # compute the value for the shallowes depth,
            # note this only takes the first value in
            # the data
            integretions[0] = depths[0] * geometry_length * vals[0]
            segmentations[0] = integretions[0]
        elif geometry == "spherical" and dim == 2:
            integretions[0] = geometry_length/2.0 * (Ro**2.0 - (Ro - depths[0])**2.0) * vals[0]
            segmentations[0] = integretions[0]
        else:
            raise NotImplementedError()
        for i in range(1, self.time_step_length):
            if geometry == "cartesian" and dim == 2:
                volume = (depths[i] - depths[i-1]) * geometry_length
            elif geometry == "spherical" and dim == 2:
                r_last = Ro - depths[i-1]
                r = Ro - depths[i]
                volume = geometry_length/2.0 * (r_last**2.0 - r**2.0) # geometry_length in theta
            else:
                raise NotImplementedError()
            # compute average between the current point and the last point
            integretions[i] = integretions[i-1] + (vals[i-1] + vals[i])/2.0 * volume 
            segmentations[i] = (vals[i-1] + vals[i])/2.0 * volume 
        return integretions, segmentations


def PlotNewtonSolverHistory(temp_path, fig_path_base, **kwargs):
    '''
    PlotNewtonSolverHistory:
    Reads runtime information from a Newton solver log file and produces a summary plot
    of solver residuals and iteration counts.

    Parameters:
        fig_path_base (str): Base path for saving the output figure.
        **kwargs:
            query_iterations (list[int], optional): List of nonlinear iteration indices
                to track residuals for across time steps.
            step_range (list[int], optional): Range of steps [start, end] to include in plots.

    Returns:
        str: Path to the saved figure.
    '''
    # Read log file path and handle user query options
    query_iterations = kwargs.get('query_iterations', None)
    trailer = None

    # Initialize the plotter and read the header/data
    Plotter = LINEARPLOT('SolverHistory', {})
    Plotter.ReadHeader(temp_path)
    try:
        Plotter.ReadData(temp_path)
    except ValueError as e:
        raise ValueError('Value error(columns are not uniform): check file %s' % temp_path)

    # Get column indices
    col_step = Plotter.header['Time_step_number']['col']
    col_number_of_iteration = Plotter.header['Index_of_nonlinear_iteration']['col']
    col_residual = Plotter.header['Relative_nonlinear_residual']['col']
    end_step = int(Plotter.data[-1, col_step])

    # Initialize arrays for plotting
    steps = np.array([i for i in range(end_step)])
    number_of_iterations = np.zeros(end_step)
    residuals = np.zeros(end_step)

    # If user requests residuals at certain iterations, prepare the array
    n_query_iteration = 0
    if query_iterations is not None:
        for iteration in query_iterations:
            assert(type(iteration) == int)
        n_query_iteration = len(query_iterations)
        residuals_at_iterations = np.zeros([n_query_iteration, end_step])

    # Populate residuals and iteration counts from data
    for i in range(steps.size):
        step = steps[i]
        mask_step = (Plotter.data[:, col_step] == step)
        data = Plotter.data[mask_step, :]
        try:
            number_of_iterations[step] = data[-1, col_number_of_iteration]
            residuals[step] = data[-1, col_residual]
        except IndexError:
            number_of_iterations[step] = 0
            residuals[step] = 1e-31

    # Query residuals at specified iterations
    for i in range(steps.size):
        step = steps[i]
        mask_step = (Plotter.data[:, col_step] == step)
        data = Plotter.data[mask_step, :]
        for j in range(n_query_iteration):
            try:
                residuals_at_iterations[j, step] = data[query_iterations[j], col_residual]
            except IndexError:
                residuals_at_iterations[j, step] = data[-1, col_residual]

    # Determine which steps to include in plots
    step_range = kwargs.get('step_range', None)
    if step_range == None:
        s_mask = (steps >= 0)
    else:
        my_assert(type(step_range) == list and len(step_range) == 2, TypeError, "%s: step_range must be a list of 2." % func_name())
        s_mask = ((steps >= step_range[0]) & (steps <= step_range[1]))
        trailer = "%d_%d" % (step_range[0], step_range[1])

    # Create figure with 3 subplots
    fig = plt.figure(tight_layout=True, figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3)
    color = 'tab:blue'

    # Subplot 1: Residuals over time
    ax = fig.add_subplot(gs[0, 0])
    ax.semilogy(steps[s_mask], residuals[s_mask], '-', linewidth=1.5, color=color, label='Residuals')
    if query_iterations is not None:
        for j in range(n_query_iteration):
            ax.semilogy(steps[s_mask], residuals_at_iterations[j, s_mask], '--', linewidth=0.5, label='Residuals at iteration %d' % query_iterations[j])
    ax.set_ylabel('Relative non-linear residual', color=color)
    ax.set_xlabel('Steps')
    ax.set_title('Solver History')
    ax.tick_params(axis='y', labelcolor=color)
    ax.legend()

    # Twin axis for iterations count
    ax2 = ax.twinx()
    color = 'tab:red'
    ax2.plot(steps[s_mask], number_of_iterations[s_mask], '.', color=color, label='Numbers of Iterations')
    ax2.set_ylabel('Numbers of Iterations', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # Subplot 2: Histogram of log residuals
    ax = fig.add_subplot(gs[0, 1])
    data = np.log10(residuals[s_mask])
    plt.hist(data, bins=np.arange(int(np.floor(min(data))), int(np.floor(max(data))) + 1, 0.25), edgecolor='black')
    plt.xlabel('log(relative non-linear residuals)')
    plt.ylabel('Frequency')
    plt.title('Bin Plot')

    # Subplot 3: Histogram of iteration counts
    ax = fig.add_subplot(gs[0, 2])
    data = number_of_iterations[s_mask]
    plt.hist(data, bins=np.arange(0, int(np.ceil(max(data)/10.0 + 1.0)*10.0), 10.0), color='red', edgecolor='black')
    plt.xlabel('Numbers of Iterations')
    plt.ylabel('Frequency')
    plt.title('Bin Plot')

    # Save figure to file
    fig.tight_layout()
    fig_path_base0 = fig_path_base.rpartition('.')[0]
    fig_path_type = fig_path_base.rpartition('.')[2]
    if trailer == None:
        fig_path = "%s.%s" % (fig_path_base0, fig_path_type)
    else:
        fig_path = "%s_%s.%s" % (fig_path_base0, trailer, fig_path_type)
    plt.savefig(fig_path)
    print("New figure (new): %s" % fig_path)
    return fig_path

#==================================================
# CodeSection: slab morpholog and temperature, using vtk pakage
#================================================== 
class VERTICAL_PROFILE():
    '''
    A vertical profile
    Attributes:
        rs - vertical coordinates
        n - points in the profile
        field_funcs (dict) - functions in the profile
        uniform - is the grid uniform
    '''
    def __init__(self, rs, field_names, field_datas, uniform=False):
        '''
        Initiation
        '''
        self.rs = rs
        self.n = rs.size
        self.uniform = uniform
        self.field_funcs = {}
        for i in range(len(field_names)):
            field_name = field_names[i]
            field_data = field_datas[i]
            field_func = interp1d(rs, field_data, assume_sorted=True, fill_value="extrapolate")
            self.field_funcs[field_name] = field_func
        pass

    def GetFunction(self, field_name):
        '''
        field_name(str): name of the field
        return
            function of the matched field
        '''
        return self.field_funcs[field_name]


class VTKP_BASE():
    '''
    Class for VTK post-process 

    Attributes:
        reader (vtk reader object): Reader for VTK file formats.
        i_poly_data (vtk.vtkPolyData): Input polydata object used for processing.
        include_cell_center (bool): Flag indicating if cell centers should be included.
        c_poly_data (vtk.vtkPolyData): Processed polydata object for cells.
        cell_sizes (array-like or None): Array to store cell sizes.
        dim (int): Dimensionality of the data, default is 2.
        grav_data (array or None): A 2-column array to store gravity data (depth in meters and gravitational acceleration).
        geometry (str): The type of geometry, default is 'chunk'.
        is_chunk (bool): Flag to determine if the geometry is a chunk, based on the value of 'geometry'.
        Ro (float): Outer radius of the spherical domain, default is 6371e3 meters.
        Xmax (float): Maximum x-coordinate in radians for chunk geometry, default is 61.0 * pi / 180.0.
        time (float or None): Simulation time, used for time-dependent calculations.
        spacing_n (int): Number of spacing intervals for domain splitting, default is 0.
        spacing (array-like or None): Array to store spacing values for splitting the domain.
        spacing_cell_ids (array-like or None): Array to store cell IDs corresponding to spacing.
        cell_array (array-like or None): Array for storing cell-related data.
        Tref_func (callable or None): Interpolated function for temperature reference, derived from depth-averaged data.
        density_ref_func (callable or None): Interpolated function for adiabatic density reference, derived from depth-averaged data.
    '''

    def __init__(self, **kwargs):
        '''
        Initialization of the VTKP_BASE class.

        Inputs:
            kwargs (dict): Additional keyword arguments for initialization:
                - dim (int, default=2): Dimensionality of the data.
                - geometry (str, default='chunk'): The geometry type, either 'chunk' or otherwise.
                - Ro (float, default=6371e3): Outer radius of the spherical domain.
                - Xmax (float, default=61.0 * pi / 180.0): Maximum x-coordinate in radians.
                - time (float or None): Simulation time for time-dependent calculations.
                - ha_file (str or None): Path to a file containing depth-averaged data.
        
        Description:
            - Initializes various VTK and domain-specific attributes.
            - Sets up the polydata objects and handles gravity and temperature reference data.
            - If 'ha_file' is provided, it imports the depth-averaged data and sets up interpolation functions.
        '''
        self.i_poly_data = vtk.vtkPolyData()
        self.include_cell_center = False
        self.c_poly_data = vtk.vtkPolyData()
        self.cell_sizes = None
        self.dim = kwargs.get('dim', 2)
        self.grav_data = None  # A 2-column array to save gravity data (depth in meters and grav_acc)
        self.geometry = kwargs.get('geometry', 'chunk')
        self.is_chunk = (self.geometry == 'chunk')
        self.Ro = kwargs.get('Ro', 6371e3)
        self.Xmax = kwargs.get('Xmax', 61.0 * np.pi / 180.0)
        self.time = kwargs.get('time', None)

        # For splitting the domain
        self.spacing_n = 0
        self.spacing = None
        self.spacing_cell_ids = None

        # For storing an array of cells
        self.cell_array = None

        ha_file = kwargs.get("ha_file", None)
        if ha_file is None:
            self.Tref_func = None
        else:
            assert(os.path.isfile(ha_file))
            DepthAverage = DEPTH_AVERAGE_PLOT('DepthAverage')
            DepthAverage.Import(ha_file)
            my_assert(self.time is not None, ValueError, 
                                "\"time\" is a required input if \"ha_file\" is provided")
            self.Tref_func = DepthAverage.GetInterpolateFunc(self.time, "temperature")
            self.density_ref_func = DepthAverage.GetInterpolateFunc(self.time, "adiabatic_density")
        pass


    def ReadFile(self, filein, **kwargs):
        '''
        Read file
        Inputs:
            filein (str): path to a input file
        '''
        quiet = kwargs.get("quiet", False)
        if not quiet:
            print("%s started" % func_name())
        start = time.time()
        assert(os.path.isfile(filein))
        file_extension = filein.split('.')[-1]
        if file_extension == 'pvtu':
            self.reader = vtk.vtkXMLPUnstructuredGridReader()
        elif file_extension == 'vtu':
            self.reader = vtk.vtkXMLUnstructuredGridReader()
        else:
            raise TypeError("%s: Wrong type of file" % func_name())
        self.reader.SetFileName(filein)
        self.reader.Update()
        end = time.time()
        if not quiet:
            print("\tReadFile: %s" % filein)
            print("\t%s, takes %.2f s" % (func_name(), end - start))
    
    def ConstructPolyData(self, field_names, **kwargs):
        '''
        construct poly data
        Inputs:
            field_names (list of field names)
            kwargs:
                include_cell_center - include_cell_center in the poly_data
        '''
        quiet = kwargs.get("quiet", False)
        if not quiet:
            print("%s started" % func_name())
        start = time.time()
        include_cell_center = kwargs.get('include_cell_center', False)
        fix_cell_value = kwargs.get('fix_cell_value', True)  # fix the value of cell centers
        construct_Tdiff = kwargs.get("construct_Tdiff", False)
        assert(type(field_names) == list and len(field_names) > 0)
        noP = 0
        noC = 0
        grid = self.reader.GetOutput()
        data_set = self.reader.GetOutputAsDataSet()
        points = grid.GetPoints()
        cells = grid.GetCells()
        self.cells = cells
        point_data = data_set.GetPointData()
        self.i_poly_data.SetPoints(points)
        self.i_poly_data.SetPolys(cells)
        # import polydata
        is_first = True
        for field_name in field_names:
            if is_first:
                self.i_poly_data.GetPointData().SetScalars(point_data.GetArray(field_name))  # put T into cell data
                is_first = False
            else:
                self.i_poly_data.GetPointData().AddArray(point_data.GetArray(field_name))  # put T into cell data
            noP = self.i_poly_data.GetNumberOfPoints()
        if construct_Tdiff:
            assert(self.Tref_func != None)
            self.ConstructTemperatureDifference()
        time_import = time.time()
        # include cell centers
        if include_cell_center:
            centers = vtk.vtkCellCenters()  # test the utilities of centers
            centers.SetInputData(self.i_poly_data)
            centers.Update()
            probeFilter = vtk.vtkProbeFilter()
            # probeFilter = vtk.vtkCompositeDataProbeFilter() # debug
            probeFilter.SetSourceData(self.i_poly_data)  # use the polydata
            probeFilter.SetInputData(centers.GetOutput()) # Interpolate 'Source' at these points
            probeFilter.Update()
            self.c_poly_data = probeFilter.GetOutput()  # poly data at the center of the point
            self.include_cell_center = True
            noC = self.c_poly_data.GetNumberOfPoints() 
            # self.c_poly_data.GetPointData().GetArray('T') = numpy_to_vtk(T_field)
            # cell sizes
            cell_size_filter = vtk.vtkCellSizeFilter()
            cell_size_filter.SetInputDataObject(grid)
            cell_size_filter.Update()
            cz_cell_data = cell_size_filter.GetOutputDataObject(0).GetCellData()
            if self.dim == 2:
                self.cell_sizes = vtk_to_numpy(cz_cell_data.GetArray('Area'))
            else:
                raise ValueError("Not implemented")
            # fix values of fields. This is needed because the interpolation is not correct where 
            # the mesh refines or coarsens. 
            # I am usign the 'T' field as an indicator:
            # every cell center with T = 0.0 is to be fixed (assuming a realistic T > 273.0)
            # the strategy is to take a nearby cell center and check its value.
            # Note: this will look into the adjacent cells until it finds one with a sufficently 
            # approximate location and a non-zero value of T.
            tolerance = 1.0
            T_field = vtk_to_numpy(self.c_poly_data.GetPointData().GetArray('T'))
            fields = []
            for field_name in field_names:
                fields.append(vtk_to_numpy(self.c_poly_data.GetPointData().GetArray(field_name)))
            # density_field =  vtk_to_numpy(self.c_poly_data_raw.GetPointData().GetArray('density'))
            if fix_cell_value:
                for i in range(noC):
                    if T_field[i] - 0.0 < tolerance:
                        xs = self.c_poly_data.GetPoint(i)
                        found = False
                        i1 = 0
                        j = 1
                        dist_max = 3*(self.cell_sizes[i]**0.5)  # compare to the cell size
                        while True:   # find a adjacent point
                            if i+j >= noC and i-j < 0:
                                break # end is reached
                            if i+j < noC:
                                xs1 = self.c_poly_data.GetPoint(i+j)
                                dist = ((xs1[0] - xs[0])**2.0 + (xs1[1] - xs[1])**2.0)**0.5
                                if T_field[i+j] - 0.0 > tolerance and dist < dist_max:
                                    i1 = i + j
                                    found = True
                                    break
                            if i-j >= 0:
                                xs1 = self.c_poly_data.GetPoint(i-j)
                                dist = ((xs1[0] - xs[0])**2.0 + (xs1[1] - xs[1])**2.0)**0.5
                                if i-j >= 0 and T_field[i-j] - 0.0 > tolerance and dist < dist_max:
                                    i1 = i - j
                                    found = True
                                    break
                            j += 1
                        if not found:
                            raise ValueError("A cell center (%.4e, %.4e, %.4e) is not in mesh, and the adjacent cells are not found" % (xs[0], xs[1], xs[2]))
                        for n in range(len(fields)):
                            fields[n][i] = fields[n][i1]
        time_center = time.time()
        # send out message
        message = "\tConstructPolyData: %d * (%d + %d) entries in the polydata imported and %d * (%d + %d) points in the data at cell center. \
import data takes %f, interpolate cell center data takes %f"\
        % (noP, self.dim, len(field_names), noC, self.dim, len(field_names), time_import - start, time_center - time_import)
        if not quiet:
            print(message)
        end = time.time()
        if not quiet:
            print("\tConstruct polydata, takes %.2f s" % (end - start))

    def SplitInSpace(self, spacing, **kwargs):
        '''
        Split the space
        '''
        print("%s started" % func_name())
        start = time.time()
        # options
        geometry = kwargs.get('geometry', 'box')
        if geometry == "box":
            is_cartesian = True
        elif geometry == "chunk":
            is_cartesian = False
        else:
            raise ValueError("%s: geometry needs to be cartesian or chunk" % func_name())
        dim = kwargs.get('dim', 2)

        # Get the bounds
        domain_bounds = self.i_poly_data.GetPoints().GetBounds()

        # spacing: split the domain into smaller space, record the number of the 3 axis and the total number
        self.spacing = spacing
        if dim == 2:
            return NotImplementedError
        elif dim == 3:
            spacing_x, spacing_y, spacing_z = spacing[0], spacing[1], spacing[2]
            interval_x = (domain_bounds[1] - domain_bounds[0]) / spacing_x
            interval_y = (domain_bounds[3] - domain_bounds[2]) / spacing_y
            interval_z = (domain_bounds[5] - domain_bounds[4]) / spacing_z
            spacing_n = spacing_x * spacing_y * spacing_z
            for ix in range(spacing_x):
                for jy in range(spacing_y):
                    for kz in range(spacing_z):
                        spacing_idx = kz + jy * spacing_z + ix * spacing_y * spacing_z
        self.spacing_n = spacing_n

        # distribute cells by looking up the cell center
        self.spacing_cell_ids = [[] for i in range(self.spacing_n)]
        centers = vtk.vtkCellCenters()  # test the utilities of centers
        centers.SetInputData(self.i_poly_data)
        centers.Update()
        cell_centers = vtk_to_numpy(centers.GetOutput().GetPoints().GetData())
        for iC in range(cell_centers.shape[0]):
            cell_coordinates = cell_centers[iC]
            cell_center_x, cell_center_y, cell_center_z = cell_coordinates[0], cell_coordinates[1], cell_coordinates[2]
            ix = int((cell_center_x - domain_bounds[0]) // interval_x)
            jy = int((cell_center_y - domain_bounds[2]) // interval_y)
            kz = int((cell_center_z - domain_bounds[4]) // interval_z)
            spacing_idx = kz + jy * spacing_z + ix * spacing_y * spacing_z
            self.spacing_cell_ids[spacing_idx].append(iC)
        end = time.time()
        print("\t%s takes %.2f s" % (func_name(), end-start))
    
    def InterpolateDomain(self, points, fields, **kwargs):
        '''
        Run interpolation
        '''
        quiet = kwargs.get("quiet", True)
        if not quiet:
            print("%s started" % func_name())
        start = time.time()
        cells_vtk = kwargs.get("cells_vtk", None) # for set connectivity
        points_found = kwargs.get("points_found", None)
        output_poly_data = kwargs.get("output_poly_data", True)
        interpolated_data = kwargs.get("interpolated_data", None)
        apply_additional_chunk_search = kwargs.get('apply_additional_chunk_search', True)
        is_box = (self.geometry == "box")
        if not is_box:
            my_assert(self.geometry == "chunk", ValueError, "%s: we only handle box and chunk geometry" % func_name())
        # check point dimension
        if points.shape[1] == 2:
            pass
        elif points.shape[1] == 3:
            pass
        else:
            raise ValueError("%s: points.shape[1] needs to either 2 or 3" % func_name())
        # initiate a ndarray to store found information 
        if points_found is None:
            points_found = np.zeros(points.shape[0], dtype=int)
        # Get the bounds
        domain_bounds = self.i_poly_data.GetPoints().GetBounds()
        # prepare the datafield to interpolate
        # field_type: 0 - scalar, 1 - vector
        raw_data = []
        field_type = []
        n_vector = 0
        for field in fields:
            if field in ["velocity"]:
                raw_data.append(self.i_poly_data.GetPointData().GetVectors(field))
                field_type.append(1)
                n_vector += 1
            else:
                raw_data.append(self.i_poly_data.GetPointData().GetArray(field))
                field_type.append(0)
        if interpolated_data is None:
            interpolated_data = np.zeros((len(fields), points.shape[0]))
        if n_vector > 0:
            interpolated_vector = [ [[0.0, 0.0, 0.0] for j in range(points.shape[0])] for i in range(n_vector)]
        end = time.time()
        if not quiet:
            print("\tInitiating takes %2f s" % (end - start))
            if (not is_box) and apply_additional_chunk_search:
                print("\tApply additional chunk search")
        
        # loop over the points, find cells containing the points and interpolate from the cell points
        start = end
        points_in_cell = [[0.0, 0.0, 0.0] for i in range(4)] # for 2d, parse vtk cell, chunk case
        n_found = 0
        n_out_of_bound = 0
        n_not_found = 0
        for i in range(points.shape[0]):
            if points_found[i]:
                # skip points found in other files
                continue
            if not PointInBound2D(points[i], domain_bounds):
                # coordinates out of range, append invalid values
                n_out_of_bound += 1
                continue
            for iC in range(self.i_poly_data.GetNumberOfCells()):
                cell = self.i_poly_data.GetCell(iC)
                bound = cell.GetBounds()
                if PointInBound2D(points[i], bound):
                    # mark points inside the cell iC and mark found of point i
                    # box: simply use the cell bound
                    # chunk: check first the cell bound, then convert both the query point and the bound
                    # to spherical coordinates and check again
                    if is_box:
                        found = True
                        points_found[i] = 1
                        cell_found = cell
                        break
                    else:
                        if apply_additional_chunk_search:
                            r, theta, phi = cart2sph(points[i][0], points[i][1], points[i][2])
                            cell_points = cell.GetPoints()
                            for i_pc in range(cell.GetNumberOfPoints()):
                                point_id = cell.GetPointId(i_pc)
                                cell_points.GetPoint(i_pc, points_in_cell[i_pc])
                            sph_bounds_cell = SphBound(points_in_cell)
                            if PointInBound([phi, theta, r], sph_bounds_cell):
                                found = True
                                points_found[i] = 1
                                cell_found = cell
                                break
                        else:
                            found = True
                            points_found[i] = 1
                            cell_found = cell
                            break
            if found:
                n_found += 1
                # Prepare variables for EvaluatePosition
                closest_point = [0.0, 0.0, 0.0]
                sub_id = vtk.reference(0)
                dist2 = vtk.reference(0.0)
                pcoords = [0.0, 0.0, 0.0]
                weights = [0.0] * cell.GetNumberOfPoints()
                # Evaluate the position to check if the point is inside the cell
                inside = cell.EvaluatePosition(points[i], closest_point, sub_id, pcoords, dist2, weights)
                for i_f in range(len(fields)):
                    fdata = raw_data[i_f]
                    if field_type[i_f] == 0:
                        interpolated_val = 0.0
                        for i_pc in range(cell_found.GetNumberOfPoints()):
                            point_id = cell_found.GetPointId(i_pc)
                            value = fdata.GetTuple1(point_id)  # Assuming scalar data
                            interpolated_val += value * weights[i_pc]
                        interpolated_data[i_f][i] = interpolated_val
                    elif field_type[i_f] == 1:
                        interpolated_val = np.array([0.0, 0.0, 0.0])
                        for i_pc in range(cell_found.GetNumberOfPoints()):
                            point_id = cell_found.GetPointId(i_pc)
                            value = fdata.GetTuple(point_id)  # Assuming scalar data
                            interpolated_val += np.array(value) * weights[i_pc]
                        interpolated_vector[0][i] = interpolated_val
            else:
                n_not_found += 1 
        total_n_found = np.sum(points_found==1)
        end = time.time()
        if not quiet:
            print("\t%s Searched %d points (%d found total; %d found current file; %d out of bound; %d not found)" % (func_name(), points.shape[0], total_n_found, n_found, n_out_of_bound, n_not_found))
            print("\tSearching takes %2f s" % (end - start))

        # construct new polydata
        # We construct the polydata ourself, now it only works for field data
        if output_poly_data:
            o_poly_data = vtk.vtkPolyData()
            points_vtk = vtk.vtkPoints()
            for i in range(points.shape[0]):
                points_vtk.InsertNextPoint(points[i])
            o_poly_data.SetPoints(points_vtk) # insert points
            if cells_vtk is not None:
                o_poly_data.SetPolys(cells_vtk)
            for i_f in range(len(fields)):
                interpolated_array = numpy_to_vtk(interpolated_data[i_f])
                interpolated_array.SetName(fields[i_f])
                if i_f == 0:
                    # the first array
                    o_poly_data.GetPointData().SetScalars(interpolated_array)
                else:
                    # following arrays
                    o_poly_data.GetPointData().AddArray(interpolated_array)
        else:
            o_poly_data = None

        return o_poly_data, points_found, interpolated_data, interpolated_vector

    
    def InterpolateSplitSpacing(self, points, fields, **kwargs):
        '''
        Run interpolation from split spacing
        '''
        print("%s started" % func_name())
        start = time.time()
        assert(self.spacing_n > 0)
        cells_vtk = kwargs.get("cells_vtk", None) # for set connectivity
        split_perturbation = kwargs.get("split_perturbation", 1) # number of split to acess during interpolation
        debug = kwargs.get("debug", False) # print debug information
        points_found = kwargs.get("points_found", None)
        output_poly_data = kwargs.get("output_poly_data", True)
        interpolated_data = kwargs.get("interpolated_data", None)
        apply_additional_chunk_search = kwargs.get('apply_additional_chunk_search', True)
        is_box = (self.geometry == "box")
        if not is_box:
            my_assert(self.geometry == "chunk", ValueError, "%s: we only handle box and chunk geometry" % func_name())
        # check point dimension
        if points.shape[1] == 2:
            raise NotImplementedError()
        elif points.shape[1] == 3:
            pass
        else:
            raise ValueError("%s: points.shape[1] needs to either 2 or 3" % func_name())
        # initiate a ndarray to store found information 
        if points_found is None:
            points_found = np.zeros(points.shape[0], dtype=int)
        # Get the bounds
        domain_bounds = self.i_poly_data.GetPoints().GetBounds()
        # Extract the range for x, y, and z
        spacing_x, spacing_y, spacing_z = self.spacing[0], self.spacing[1], self.spacing[2]
        interval_x = (domain_bounds[1] - domain_bounds[0]) / spacing_x
        interval_y = (domain_bounds[3] - domain_bounds[2]) / spacing_y
        interval_z = (domain_bounds[5] - domain_bounds[4]) / spacing_z

        # prepare the datafield to interpolate
        raw_data = []
        for field in fields:
            raw_data.append(self.i_poly_data.GetPointData().GetArray(field))
        if interpolated_data is None:
            interpolated_data = np.zeros((len(fields), points.shape[0]))

        # interpolate
        # Method to use: simply looking at the bound of each cell
        # Note there is a method of EvaluatePosition to decide wheter a point is in a cell.
        # But that doesn't work well. It give different results for the same point and cell, shame.
        # looking at the bound works well for cartesian geometry but needs double-check the spherical bound in chunk geometry
        n_found = 0  # record whether interpolation is successful
        n_out_of_bound = 0
        n_not_found = 0
        # apply search over multiple slices by perturbation on the
        # slice index in all 3 directions
        diffs = []
        for pert_x in range(split_perturbation):
            for pert_y in range(split_perturbation):
                for pert_z in range(split_perturbation):
                    diffs.append([pert_x, pert_y, pert_z])
                    if pert_x > 0:
                        diffs.append([-pert_x, pert_y, pert_z])
                        if pert_y > 0:
                            diffs.append([-pert_x, -pert_y, pert_z])
                            if pert_z > 0:
                                diffs.append([-pert_x, -pert_y, -pert_z])
                        if pert_z > 0:
                            diffs.append([-pert_x, pert_y, -pert_z])
                    if pert_y > 0:
                        diffs.append([pert_x, -pert_y, pert_z])
                        if pert_z > 0:
                            diffs.append([pert_x, -pert_y, -pert_z])
                    if pert_z > 0:
                        diffs.append([pert_x, pert_y, -pert_z])
        end = time.time()
        print("\tInitiating takes %2f s" % (end - start))
        if (not is_box) and apply_additional_chunk_search:
            print("\tApply additional chunk search")
        # loop over the points, find cells containing the points and interpolate from the cell points
        start = end
        points_in_cell = [[0.0, 0.0, 0.0] for i in range(8)] # for 3d, parse vtk cell, chunk case
        for i in range(points.shape[0]):
            if points_found[i]:
                # skip points found in other files
                continue
            if not PointInBound(points[i], domain_bounds):
                # coordinates out of range, append invalid values
                n_out_of_bound += 1
                continue
            ix = int((points[i][0] - domain_bounds[0]) // interval_x)
            jy = int((points[i][1] - domain_bounds[2]) // interval_y)
            kz = int((points[i][2] - domain_bounds[4]) // interval_z)
            # Evaluate the position
            found = False
            for diff in diffs:
                spacing_idx = clamp(kz + diff[2], 0, spacing_z-1)\
                             + clamp(jy + diff[1], 0, spacing_y-1) * spacing_z\
                             + clamp(ix + diff[0], 0, spacing_x-1) * spacing_y * spacing_z
                assert(spacing_idx < self.spacing_n) # make sure we have a valid value
                for iC in self.spacing_cell_ids[spacing_idx]:
                    cell = self.i_poly_data.GetCell(iC)
                    bound = cell.GetBounds()
                    if PointInBound(points[i], bound):
                        # mark points inside the cell iC and mark found of point i
                        # box: simply use the cell bound
                        # chunk: check first the cell bound, then convert both the query point and the bound
                        # to spherical coordinates and check again
                        if is_box:
                            found = True
                            points_found[i] = 1
                            break
                        else:
                            if apply_additional_chunk_search:
                                r, theta, phi = cart2sph(points[i][0], points[i][1], points[i][2])
                                cell_points = cell.GetPoints()
                                for i_pc in range(cell.GetNumberOfPoints()):
                                    point_id = cell.GetPointId(i_pc)
                                    cell_points.GetPoint(i_pc, points_in_cell[i_pc])
                                sph_bounds_cell = SphBound(points_in_cell)
                                if PointInBound([phi, theta, r], sph_bounds_cell):
                                    found = True
                                    points_found[i] = 1
                                    break
                            else:
                                found = True
                                points_found[i] = 1
                                break
                if found:
                    cell_found = cell
                    break
            if found:
                n_found += 1
                # Prepare variables for EvaluatePosition
                closest_point = [0.0, 0.0, 0.0]
                sub_id = vtk.reference(0)
                dist2 = vtk.reference(0.0)
                pcoords = [0.0, 0.0, 0.0]
                weights = [0.0] * cell.GetNumberOfPoints()
                # Evaluate the position to check if the point is inside the cell
                inside = cell.EvaluatePosition(points[i], closest_point, sub_id, pcoords, dist2, weights)
                for i_f in range(len(fields)):
                    fdata = raw_data[i_f]
                    interpolated_val = 0.0
                    for i_pc in range(cell_found.GetNumberOfPoints()):
                        point_id = cell_found.GetPointId(i_pc)
                        value = fdata.GetTuple1(point_id)  # Assuming scalar data
                        interpolated_val += value * weights[i_pc]
                    interpolated_data[i_f][i] = interpolated_val
            else:
                n_not_found += 1
        total_n_found = np.sum(points_found==1)
        end = time.time()
        print("\t%s Searched %d points (%d found total; %d found current file; %d out of bound; %d not found)" % (func_name(), points.shape[0], total_n_found, n_found, n_out_of_bound, n_not_found))
        print("\tSearching takes %2f s" % (end - start))
        # construct new polydata
        # We construct the polydata ourself, now it only works for field data
        if output_poly_data:
            o_poly_data = vtk.vtkPolyData()
            points_vtk = vtk.vtkPoints()
            for i in range(points.shape[0]):
                points_vtk.InsertNextPoint(points[i])
            o_poly_data.SetPoints(points_vtk) # insert points
            if cells_vtk is not None:
                o_poly_data.SetPolys(cells_vtk)
            for i_f in range(len(fields)):
                interpolated_array = numpy_to_vtk(interpolated_data[i_f])
                interpolated_array.SetName(fields[i_f])
                if i_f == 0:
                    # the first array
                    o_poly_data.GetPointData().SetScalars(interpolated_array)
                else:
                    # following arrays
                    o_poly_data.GetPointData().AddArray(interpolated_array)
        else:
            o_poly_data = None

        return o_poly_data, points_found, interpolated_data
            


    def ConstructTemperatureDifference(self):
        '''
        Construct a dT field of temperature differences
        '''
        T_field = vtk_to_numpy(self.i_poly_data.GetPointData().GetArray("T"))
        Tdiffs = vtk.vtkFloatArray()
        Tdiffs.SetName("dT")
        for i in range(self.i_poly_data.GetNumberOfPoints()):
            xs = self.i_poly_data.GetPoint(i)
            x =  xs[0]
            y =  xs[1]
            r = get_r(x, y, self.geometry)
            Tref = self.Tref_func(self.Ro - r)
            T = T_field[i]
            Tdiffs.InsertNextValue(T - Tref)
        self.i_poly_data.GetPointData().AddArray(Tdiffs)

    def VerticalProfile2D(self, x0_range, x1, n, **kwargs):
        '''
        Get vertical profile by looking at one vertical line
        Inputs:
            x0_range: range of the first coordinate (y or r)
            x1: second coordinate (x or theta)
            n (int): number of point in the profile
            kwargs (dict):
                geometry: spherical or cartesian
                fix_point_value: fix invalid value indicated by zero temperature
                                these points exist as a result of interpolation
                                in adaptive meshes.
        '''
        geometry = kwargs.get('geometry', 'chunk')
        fix_point_value = kwargs.get('fix_point_value', False)
        assert(len(x0_range) == 2)
        points = np.zeros((n, 2))
        x0s = np.linspace(x0_range[0], x0_range[1], n)
        x1s = np.ones(n) * x1
        if geometry == 'chunk':
            xs = x0s * np.cos(x1s)
            ys = x0s * np.sin(x1s)
        elif geometry == 'box':
            ys = x0s
            xs = x1s
        else:
            raise ValueError('Wrong option for geometry')
        points[:, 0] = xs
        points[:, 1] = ys
        v_poly_data = InterpolateGrid(self.i_poly_data, points, quiet=True)
        point_data = v_poly_data.GetPointData()
        fields = []
        field_names = []
        fix_value_mask = None
        for i in range(point_data.GetNumberOfArrays()):
            field_name = point_data.GetArrayName(i)
            field = vtk_to_numpy(point_data.GetArray(field_name))
            field_names.append(field_name)
            fields.append(field)
            if fix_point_value and field_name == "T":
                # only take the valid point value (T = 0.0)
                fix_value_mask = (field > 1e-6)
        fields1 = []
        if fix_value_mask is not None:
            x0s = x0s[fix_value_mask]
            for field in fields:
                fields1.append(field[fix_value_mask])
        else:
            fields1 = fields
        v_profile = VERTICAL_PROFILE(x0s, field_names, fields1, uniform=True)
        return v_profile
    
    def StaticPressure(self, x0_range, x1, n, **kwargs):
        '''
        Compute the static pressure at a point
        Inputs:
            x0_range: range of the first coordinate (y or r)
            x1: second coordinate (x or theta)
            n (int): number of point in the profile
            kwargs (dict):
                geometry: spherical or cartesian
        '''
        geometry = kwargs.get('geometry', 'chunk') # geometry
        use_gravity_profile = False
        if self.grav_data is not None:
            use_gravity_profile = True
        else:
            constant_grav_acc = kwargs['grav_acc']
        points = np.zeros((n, 2))
        assert(x0_range[0] < x0_range[1])
        x0s = np.linspace(x0_range[0], x0_range[1], n)
        interval = (x0_range[1] - x0_range[0]) / (n-1)
        x1s = np.ones(n) * x1
        if geometry == 'chunk':
            xs = x0s * np.cos(x1s)
            ys = x0s * np.sin(x1s)
        elif geometry == 'box':
            ys = x0s
            xs = x1s
        else:
            raise ValueError('Wrong option for geometry')
        points[:, 0] = xs
        points[:, 1] = ys
        v_poly_data = InterpolateGrid(self.i_poly_data, points, quiet=True)
        point_data = v_poly_data.GetPointData()
        density_field = vtk_to_numpy(point_data.GetArray('density'))
        static_pressure = 0.0  # integrate the static pressure
        for i in range(1, n):
            depth = self.Ro - (x0s[i] + x0s[i-1])/2.0
            if use_gravity_profile:
                grav_acc = self.GetGravityAcc(depth)
            else:
                grav_acc = constant_grav_acc # geometry
            static_pressure += (density_field[i-1] + density_field[i]) / 2.0 * grav_acc * interval
        return static_pressure
    
    def ImportGravityData(self, filein):
        '''
        Import gravity data, file shoud contain depth and 
        gravity accerleration.
        Inputs:
            filein (str): path to a input file
        '''
        assert(os.path.isfile(filein))
        self.grav_data = np.loadtxt(filein, skiprows=8)
    
    def GetGravityAcc(self, depths):
        '''
        Get gravity from a profile
        Inputs:
            depths - depth of point, float or a vector
        '''
        assert(self.grav_data.shape[1] == 2)
        grav_accs = np.interp(depths, self.grav_data[:, 0], self.grav_data[:, 1])
        return grav_accs


def InterpolateVtu(Visit_Options, filein, spacing, fields, target_points_np, **kwargs):
    '''
    Interpolation of vtu
    '''
    geometry = Visit_Options.options['GEOMETRY']
    Ro =  Visit_Options.options['OUTER_RADIUS']
    Ri = Visit_Options.options['INNER_RADIUS']
    Xmax = Visit_Options.options['XMAX']
    # read file
    VtkP = VTKP_BASE(geometry=geometry, Ro=Ro, Xmax=Xmax)
    VtkP.ReadFile(filein)
    # construct poly data
    VtkP.ConstructPolyData(fields, include_cell_center=False)
    
    ### split the space
    VtkP.SplitInSpace(spacing, dim=3)
    
    ### Interpolate onto a new mesh
    kwargs["fields"] = fields
    o_poly_data, points_found, interpolated_data = VtkP.InterpolateSplitSpacing(target_points_np, **kwargs)
    return o_poly_data, points_found, interpolated_data

# todo_export
def ExportPointGridFromPolyData(i_poly_data, ids, output_xy=False):
    '''
    export point grid from a given vtk poly data by the indexed
    '''
    assert(ids is not [])
    vtk_points = vtk.vtkPoints()
    for id in ids:
        xs = i_poly_data.GetPoint(id)
        vtk_points.InsertNextPoint(xs[0], xs[1], xs[2])
    point_grid = vtk.vtkUnstructuredGrid()
    point_grid.SetPoints(vtk_points)
    if output_xy:
        coords = vtk_to_numpy(point_grid.GetPoints().GetData())
        return coords
    else:
        return point_grid
    pass


def ExportContour(poly_data, field_name, contour_value, **kwargs):
    '''
    Export contour of a field with a value
    Inputs:
        field_name (str): name of the field
        contour_value (float): contour value
        kwargs:
            fileout (str): path to output
    '''
    print("Filter contour")
    fileout = kwargs.get('fileout', None)
    contour_filter = vtk.vtkContourFilter()
    # prepare poly data for contour
    c_poly_data = vtk.vtkPolyData()
    c_vtk_point_data = poly_data.GetPointData()  # vtkPointData
    c_poly_data.SetPoints(poly_data.GetPoints())  # import points and polys
    c_poly_data.SetPolys(poly_data.GetPolys())
    vtk_data_array = c_vtk_point_data.GetArray(field_name)
    assert(vtk_data_array != None)
    c_poly_data.GetPointData().SetScalars(vtk_data_array)
    # draw contour 
    contour_filter.SetInputData(c_poly_data)
    contour_filter.Update()
    contour_filter.GenerateValues(1, contour_value, contour_value)  # Extract just one contour
    contour_filter.Update()
    # write output if a path is provided
    if fileout != None:
        file_extension = fileout.split('.')[-1]
        if file_extension == 'txt':
            writer = vtk.vtkSimplePointsWriter()
        else:
            raise TypeError("%s: Wrong type of file" % func_name())
        writer.SetInputData(contour_filter.GetOutput())
        writer.SetFileName(fileout)
        writer.Update()
        writer.Write()
        print("%s, Generate output (contour) : %s" % (func_name(), fileout))
    return contour_filter.GetOutput()


def ExportPolyData(poly_data, fileout, **kwargs):
    '''
    Export poly data to vtp file
    '''
    indent = kwargs.get('indent', 0)
    # output
    file_extension = fileout.split('.')[-1]
    if file_extension == "vtp":
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(fileout)
        writer.SetInputData(poly_data)
        # writer.SetFileTypeToBinary()  # try this later to see if this works
        writer.Update()
        writer.Write()
    elif file_extension == 'txt':
        raise TypeError("%s: option for txt file is not implemented" % func_name())
        # ExportPolyDataAscii(poly_data, fileout)
    else:
        raise TypeError("%s: Wrong type of file" % func_name())
    print(' '*indent + "%s: Write file %s" % (func_name(), fileout))


def ExportPolyDataAscii(poly_data, field_names, file_out):
    '''
    Export Poly data to a ascii file
    '''
    print("%s: operating" % func_name())
    header = "# 1: x (m)\n# 2: y (m)"
    i = 3
    point_data_export = []  # point data
    for field_name in field_names:
        header += "\n# %d" % i + ": " + field_name
        vtk_data_array = poly_data.GetPointData().GetArray(field_name)
        my_assert(vtk_data_array != None, KeyError,\
            "Failed to get field %s from vtk point data" % field_name)
        point_data_export.append(vtk_to_numpy(vtk_data_array))
        i += 1
    header += '\n'
    # output data
    output= ""
    for i in range(poly_data.GetNumberOfPoints()):
        if i > 0:
            output += "\n"  # append new line for new point
        xs = poly_data.GetPoint(i)
        output += "%.4e" % xs[0] + "\t" + "%.4e" % xs[1]
        j = 0
        for field_name in field_names:
            val = point_data_export[j][i]; 
            output += "\t" + "%.4e" % val
            j += 1
    # write file
    with open(file_out, 'w') as fout:
        fout.write(header)
        fout.write(output)
    print("\tWrite ascii data file: %s" % file_out)


def ExportPolyDataFromRaw(Xs, Ys, Zs, Fs, fileout, **kwargs):
    '''
    Export poly data from raw data
    '''
    i_points = vtk.vtkPoints()
    field_name = kwargs.get("field_name", "foo")
    assert(Xs.size == Ys.size)
    if Zs != None:
        assert(Xs.size == Zs.size)
    for i in range(Xs.size):
        x = Xs[i]
        y = Ys[i]
        if Zs is not None:
            z = Zs[i]
        else:
            z = 0.0
        i_points.InsertNextPoint(x, y, z)
    i_poly_data = vtk.vtkPolyData()  # initiate poly daa
    i_poly_data.SetPoints(i_points) # insert points
    if Fs != None:
        # insert field data
        assert(Xs.size == Fs.size)
        # fvalues.SetName(field_name)
        i_poly_data.GetPointData().SetScalars(numpy_to_vtk(Fs, deep=1))
    ExportPolyData(i_poly_data, fileout, **kwargs)


def PointInBound(point, bound):
    '''
    Determine whether a 3d point is within a bound
    Inputs:
        points: 3d, list or np.ndarray
        bound: 6 component list, x_min, x_max, y_min, y_max, z_min, z_min
    '''
    if type(point) == np.ndarray:
        assert(point.size == 3)
    elif type(point) == list:
        assert(len(point) == 3)
    else:
        raise TypeError
    return (point[0] >= bound[0]) and (point[0] <= bound[1]) and (point[1] >= bound[2])\
        and (point[1] <= bound[3]) and (point[2] >= bound[4]) and (point[2] <= bound[5])


def PointInBound2D(point, bound):
    '''
    Determine whether a 2d point is within a bound
    Inputs:
        points: 2d, list or np.ndarray, not we still need 3 entries to be consistent with vtk
        bound: 4 component list, x_min, x_max, y_min, y_max
    '''
    if type(point) == np.ndarray:
        assert(point.size == 3)
    elif type(point) == list:
        assert(len(point) == 3)
    else:
        raise TypeError
    return (point[0] >= bound[0]) and (point[0] <= bound[1]) and (point[1] >= bound[2])\
        and (point[1] <= bound[3])


def InterpolateGrid(poly_data, points, **kwargs):
    '''
    Inputs:
        poly_data (vtkPolyData): input data set
        points (np array): grid to interpolate to
    Return:
        poly data on the new grid
    Output:
    '''
    quiet = kwargs.get('quiet', False)
    fileout = kwargs.get('fileout', None)
    polys = kwargs.get('polys', None)
    assert(points.ndim == 2)
    if points.shape[1] == 2:
        is_2d = True
    elif points.shape[1] == 3:
        is_2d = False
    else:
        raise ValueError("%s: points.shape[1] needs to either 2 or 3" % func_name())
    if not quiet:
        print("%s: Perform interpolation onto the new grid" % func_name())
    start = time.time()
    grid_points = vtk.vtkPoints()
    # for point in points:
    #    grid_points.InsertNextPoint(point)
    for i in range(points.shape[0]):
        x = points[i][0]
        y = points[i][1]
        if is_2d:
            z = 0.0
        else:
            z = points[i][2]
        grid_points.InsertNextPoint(x, y, z)  # this is always x, y, z
    grid_data = vtk.vtkPolyData()
    grid_data.SetPoints(grid_points)
    end = time.time()
    if not quiet:
        print("%s: Construct vtkPoints, take %.2f s" % (func_name(), end - start))
    start = end
    probeFilter = vtk.vtkProbeFilter()
    probeFilter.SetSourceData(poly_data)  # use the polydata
    probeFilter.SetInputData(grid_data) # Interpolate 'Source' at these points
    probeFilter.Update()
    o_poly_data = probeFilter.GetOutput()
    if polys is not None:
        o_poly_data.SetPolys(polys)
    end = time.time()
    if not quiet:
        print("%s: Interpolate onto new grid, take %.2f s" % (func_name(), end - start))
    start = end
    # export to file output if a valid file is provided
    if fileout != None:
        ExportPolyData(o_poly_data, fileout, indent=4)
        end = time.time()
        print("%s: Export data, takes %.2f s" % (func_name(), end - start))
    return o_poly_data


def ProjectVelocity(x_query, w_query, vs, geometry):
    '''
    project the velocity from vx, vy to vr, vh in different geometries
    Inputs:
        x_query - x coordinate of the point
        w_query - y coordinate of the point
        vs - (vx, vy)
        geometry - type of geometry
    ''' 
    if geometry == "chunk":
        cos_sp = x_query / (x_query**2.0 + w_query**2.0)**0.5
        sin_sp = w_query / (x_query**2.0 + w_query**2.0)**0.5
        if vs.ndim == 1:
            v_h = vs[0] * (-sin_sp) + vs[1] * cos_sp
            v_r = vs[0] * cos_sp + vs[1] * sin_sp
        elif vs.ndim == 2:
            v_h = vs[:, 0] * (-sin_sp) + vs[:, 1] * cos_sp
            v_r = vs[:, 0] * cos_sp + vs[:, 1] * sin_sp
        else:
            NotImplementedError()
    elif geometry == "box":
        if vs.ndim == 1:
            v_h = vs[0]
            v_r = vs[1]
        elif vs.ndim == 2:
            v_h = vs[:, 0]
            v_r = vs[:, 1]
        else:
            NotImplementedError()
    else:
        raise NotImplementedError()
    return v_h, v_r


def MakeTargetMesh(Visit_Options, n0, n1, d_lateral):
    '''
    Make a target mesh for slicing 3d dataset
    Inputs:
        Visit_Options - a VISIT_OPTIONS class
        n0, n1 - number of points along the 1st and 3rd dimention
        interval - this determines the interval of the slices
        d_lateral - the lateral distance, along the 2nd dimention
            take a minimum value of 1.0 to assure successful slicing of the geometry
    '''
    # get the options 
    geometry = Visit_Options.options['GEOMETRY']
    Ro =  Visit_Options.options['OUTER_RADIUS']
    Ri = Visit_Options.options['INNER_RADIUS']
    Xmax = Visit_Options.options['XMAX']
    N = n0 * n1
    # new mesh
    target_points_np = np.zeros((N, 3))
    if geometry == "box":
        for i0 in range(n0):
            for j1 in range(n1):
                ii = i0 * n1 + j1
                target_points_np[ii, 0] = Xmax * i0 / (n0 - 1)
                target_points_np[ii, 1] = d_lateral
                target_points_np[ii, 2] = Ro * j1 / (n1 - 1) # Ro and depth are the same in this case
    elif geometry == "chunk":
        for i0 in range(n0):
            for j1 in range(n1):
                # note we use theta = 0.0 here, but z0 = small value, this is to ensure a successful slice
                # of the chunk geometry
                # we take the slice at z = d_lateral, then x and y are between Ri and a R1 value
                if d_lateral < 1e3:
                    R1 = Ro
                else:
                    R1 = (Ro**2.0 - d_lateral**2.0)**0.5
                ii = i0 * n1 + j1
                phi = i0 / n0 * Xmax * np.pi / 180.0
                r = R1 * (j1 - 0.0) / (n1 - 0.0) + Ri * (j1 - n1)/ (0.0 - n1)  
                slice_x, slice_y, _  = ggr2cart(0.0, phi, r) 
                target_points_np[ii, 0] = slice_x
                target_points_np[ii, 1] = slice_y
                target_points_np[ii, 2] = d_lateral
    return target_points_np


def GetVtkCells2d(n_x, n_z):
    '''
    Get a vtk cells from number of points along x, z in 2 dimensions
    '''
  
    cells = vtk.vtkCellArray()
  
    for ix in range(n_x-1):
        for jz in range(n_z-1):
          cells.InsertNextCell(4)
          # cell = vtk.vtkIdType()
          cells.InsertCellPoint(ix*n_z + jz+1)
          cells.InsertCellPoint(ix*n_z + jz)
          cells.InsertCellPoint((ix+1)*n_z + jz)
          cells.InsertCellPoint((ix+1)*n_z + jz+1)
  
    return cells


def Interpolate3dVtkCaseBeta(case_dir, VISIT_OPTIONS, vtu_snapshot, fields, mesh_options, **kwargs):
    '''
    Inputs:
        case_dir - case directory
        VISIT_OPTIONS - a matching class containing the options
        vtu_snapshot (int) - vtu snapshot
        fields (list of str) - the fields to output
        mesh_options - dict for mesh options
            type - type of meshes
            resolution - mesh resolutions
            d_lateral - if type is slice_2nd, this is the distance measured on the 2nd dimension
        kwargs - dict
            by_part - if the interpolation is performed on individua vtu file (False: on the pvtu file)
            spacing - spacing of the domain, this determines the number of slices the algorithm produce
            split_perturbation - This determines the number of slices the algorithm searches for a query point
    ####
    Note on the trade off between spacing and the split_perturbation parameters:
    # The spacing parameter tends to divide the domain into multiple spaces and accelerate
    # the process of interpolation, but make it harder to find the cell for a query point.
    # The problem is caused by the location of the cell center. When the cell is big, the cell center
    # might be far from a point within the cell, and one cell could be splited into different pieces of spacing.
    # This tackled by a large number of split perturbation, which will tell the interpolation algorithm to look into mulitple pices of spacing
    # rather than one and increases the possibility to locate the query point in a cell.
    # In application, first start with smaller spacing numbers. If the interpolation is slow, increase
    # this number.
    '''
    #options
    # case directory
    case_dir = '/mnt/lochy/ASPECT_DATA/ThDSubduction/chunk_test/chunk_initial9'
    assert(os.path.isdir(case_dir))
    # algorithm
    by_part = kwargs.get("by_part", False)
    apply_additional_chunk_search = kwargs.get("apply_additional_chunk_search", True)
    spacing = kwargs.get("spacing", [10, 10, 10])
    split_perturbation = kwargs.get("split_perturbation", 2)
    fields = ["T", "density"]
    # mesh
    _type = mesh_options['type']
    resolutions = mesh_options['resolution']
    n0 = resolutions[0]
    n1 = resolutions[1]
    if _type == "slice_2nd":
        d_lateral = mesh_options["d_lateral"]
    else:
        raise NotImplementedError

    #Initiation 
    # class for the basic settings of the case
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    
    ### make a target mesh as well as a poly (connectivity in cells)
    start = time.time()
    print("Make a target mesh")
    target_points_np = MakeTargetMesh(Visit_Options, n0, n1, d_lateral)
    target_cells_vtk = GetVtkCells2d(n0, n1)
    end = time.time()
    print("\tPoints in target: %d" % (target_points_np.shape[0]))
    print("\tOperation takes %.2f s" % (end - start))
    
    ### Perform interpolation
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    _time, step = Visit_Options.get_time_and_step(vtu_step)
    print("\tTime = %.4e" % (float(_time) / 1e6))
    interpolated_data = np.zeros((len(fields), target_points_np.shape[0]))
    if by_part:
        for part in range(16):
            filein = os.path.join(case_dir, "output", "solution", "solution-%05d.%04d.vtu" % (vtu_snapshot, part))
            if part == 0:
                points_found = None
            print("-"*20 + "split" + "-"*20) # print a spliting
            print(filein)
            _, points_found, interpolated_data = InterpolateVtu(Visit_Options, filein, spacing, fields, target_points_np, points_found=points_found,\
                                                        split_perturbation=split_perturbation, interpolated_data=interpolated_data, output_poly_data=False,
                                                        apply_additional_chunk_search=apply_additional_chunk_search)
            if np.sum(points_found == 1) == points_found.size:
                print("All points have been found, exiting")
                break
    else:
        filein = os.path.join(case_dir, "output", "solution", "solution-%05d.pvtu" % vtu_snapshot)
        _, points_found, interpolated_data = InterpolateVtu(Visit_Options, filein, spacing, fields, target_points_np, split_perturbation=split_perturbation,\
                                                    interpolated_data=None, output_poly_data=False, apply_additional_chunk_search=apply_additional_chunk_search)
    pass

    # organize output
    o_poly_data = vtk.vtkPolyData()
    points_vtk = vtk.vtkPoints()
    for i in range(target_points_np.shape[0]):
        points_vtk.InsertNextPoint(target_points_np[i])
    o_poly_data.SetPoints(points_vtk) # insert points
    if target_cells_vtk is not None:
        o_poly_data.SetPolys(target_cells_vtk)
    for i_f in range(len(fields)):
        interpolated_array = numpy_to_vtk(interpolated_data[i_f])
        interpolated_array.SetName(fields[i_f])
        if i_f == 0:
            # the first array
            o_poly_data.GetPointData().SetScalars(interpolated_array)
        else:
            # following arrays
            o_poly_data.GetPointData().AddArray(interpolated_array)

    # write output
    dirout = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(dirout):
        os.mkdir(dirout)
    if _type == "slice_2nd":
        filename = "slice_2nd_%05d.vtp" % vtu_snapshot
    fileout = os.path.join(dirout, filename)
    ExportPolyData(o_poly_data, fileout, indent=4)
    assert(os.path.isfile(fileout))
    print("%s: Write output %s" % (func_name(), fileout))
    

#------------ Utility functions ---------------- #

def NpIntToIdList(numpy_int_array):
    '''
    Convert numpy in array to vtk id list
    Inputs:
        numpy_int_array - 1d numpy int array
    '''
    id_list = vtk.vtkIdList()
    if type(numpy_int_array) == list:
        for i in range(len(numpy_int_array)):
            id_list.InsertNextId(numpy_int_array[i])
    else:
        assert(numpy_int_array.ndim == 1)  # 1d array
        for i in range(numpy_int_array.size):
            id_list.InsertNextId(numpy_int_array[i])
    return id_list


def OperateDataArrays(poly_data, names, operations):
    '''
    perform operation to vtk_arrays
    Inputs:
        poly_data - a poly data to work with
        names - names of fields to operate on
        operations - 0 (add) or 1 (minus)
    '''
    assert(len(operations) == len(names) - 1)
    is_first = True
    i = 0
    for _name in names:
        if is_first:
            o_array = np.copy(vtk_to_numpy(poly_data.GetArray(_name)))
            is_first = False
        else:
            if operations[i] == 0:
                o_array += vtk_to_numpy(poly_data.GetArray(_name))
            elif operations[i] == 1:
                o_array -= vtk_to_numpy(poly_data.GetArray(_name))
            else:
                raise ValueError('operation for %d is not implemented.' % operations[i])
            i += 1
    return o_array


def get_r(x, y, geometry): 
    '''
    Get r (the first coordinate)
    Inputs:
        x - x coordinate
        y - y coordinate
        geometry - 'chunk' or 'box'
    '''
    if geometry == 'chunk':
        r = (x*x + y*y)**0.5
    elif geometry == 'box':
        r = y
    else:
        raise ValueError("not implemented")
    return r


def get_r3(x, y, z, is_chunk):
    '''
    Get r (the first coordinate)
    Inputs:
        x - x coordinate
        y - y coordinate
        is_chunk - whether the geometry is chunk
    '''
    if is_chunk:
        r = (x*x + y*y + z*z)**0.5
    else:
        r = z
    return r

class VTKP(VTKP_BASE):
    '''
    Class inherited from a parental class
    Attributes:
        slab_cells: cell id of internal points in the slab
        slab_envelop_cell_list0: cell id of slab envelop with smaller theta, slab bottom
        slab_envelop_cell_list1: cell id of slab envelop with bigger theta, slab surface
        trench: trench position, theta for a 'chunk' model and x for a 'box' model
        coord_100: position where slab is 100km deep, theta for a 'chunk' model and x for a 'box' model
        vsp : velocity of the subducting plate
        vov : velocity of the overiding plate
    '''
    def __init__(self, **kwargs):
        '''
        Initiation
        kwargs (dict):
            geometry - type of geometry
            Ro - outer radius
        '''
        VTKP_BASE.__init__(self, **kwargs)
        self.slab_cells = []
        self.crust_cells = []
        self.surface_cells = []
        self.slab_envelop_cell_list0 = []
        self.slab_envelop_cell_list1 = []
        self.moho_envelop_cell_list = []
        self.sz_geometry = None
        self.slab_depth = None
        self.trench = None
        self.coord_100 = None 
        self.coord_200 = None # relate with the previous one
        self.dip_100 = None
        self.vsp = None
        self.vov = None
        self.coord_distant_200 = None
        self.v_distant_200 = None
        self.visc_distant_200 = None
        self.slab_shallow_cutoff = kwargs.get("slab_shallow_cutoff", 50e3)  # depth limit to slab
        self.slab_envelop_interval = kwargs.get("slab_envelop_interval", 5e3)
        self.velocitw_query_depth = 5e3  # depth to look up plate velocities
        self.velocitw_query_disl_to_trench = 500e3  # distance to trench to look up plate velocities
        self.shallow_trench = None
        default_gravity_file = os.path.join(var_subs('${ASPECT_SOURCE_DIR}'),\
        "data", "gravity-model", "prem.txt") 
        gravity_file = kwargs.get('gravity_file', default_gravity_file)
        assert(os.path.isfile(gravity_file))
        self.ImportGravityData(gravity_file)

    def PrepareSlabShallow(self, **kwargs):
        '''
        Prepares and identifies shallow and bottom points of a slab for trench analysis,
        exporting relevant data to a file if specified. Calculates original and corrected
        distances between shallow and bottom points based on threshold values.
    
        Parameters:
            trench_initial (float): Initial trench position in radians.
            **kwargs: Additional options, including trench_lookup_range and export_shallow_file.
                n_crust - number of crust in the model
    
        Returns:
            dict: Contains information on original and corrected points with distances.
        '''
        shallow_cutoff = 1e3 # m
        bottom_start = 7.5e3
        bottom_cutoff = 30e3
        pinned_field_value_threshold = 0.8
    
        trench_lookup_range = kwargs.get("trench_lookup_range", 15.0 * np.pi / 180.0)
        export_shallow_file = kwargs.get("export_shallow_file", None)
        n_crust = kwargs.get("n_crust", 1)
    
        print("%s started" % func_name())
        start = time.time()
    
        # Sorts the shallow points and retrieves relevant data arrays for processing.
        points = vtk_to_numpy(self.i_poly_data.GetPoints().GetData())
        points_x = points[:, 0]
        points_y = points[:, 1]
        points_r =  (points_x**2.0+points_y**2.0)**0.5
        point_data = self.i_poly_data.GetPointData()

        if n_crust == 1:
            pinned_field = vtk_to_numpy(point_data.GetArray("spcrust"))
        elif n_crust == 2:
            pinned_field = vtk_to_numpy(point_data.GetArray("spcrust_up")) + vtk_to_numpy(point_data.GetArray("spcrust_low"))
        else:
            raise NotImplementedError()
        pinned_bottom_field = vtk_to_numpy(point_data.GetArray("spharz"))

        # points in the shallow crust and deep harzburgite layer
        x0 = self.Ro * np.cos(self.trench - trench_lookup_range)
        y0 = self.Ro * np.sin(self.trench - trench_lookup_range)
        x1 = self.Ro * np.cos(self.trench + trench_lookup_range)
        y1 = self.Ro * np.sin(self.trench + trench_lookup_range)
        mask = (points_x <= x0) & (points_x >= x1) & (points_y >= y0) &\
              (points_y <= y1) # mask the region
        mask1 = mask & (self.Ro - points_r < shallow_cutoff) &\
              (pinned_field > pinned_field_value_threshold) # mask top field
        mask2 = mask & (self.Ro - points_r > bottom_start) &\
              (self.Ro - points_r < bottom_cutoff) &\
              (pinned_bottom_field > pinned_field_value_threshold) # mask top field
        shallow_points_idx = np.where(mask1)[0]
        bottom_points_idx = np.where(mask2)[0]
    
        n_shallow_points = len(shallow_points_idx)
        n_bottom_points = len(bottom_points_idx)
    
        # Extracts coordinates of shallow and bottom points for further processing.
        shallow_points = np.zeros([n_shallow_points, 3])
        for i in range(n_shallow_points):
            idx = shallow_points_idx[i]
            shallow_points[i] = points[idx]
    
        bottom_points = np.zeros([n_bottom_points, 3])
        for i in range(n_bottom_points):
            idx = bottom_points_idx[i]
            bottom_points[i] = points[idx]
    
        end = time.time()
        print("\tSort shallow points, takes %.2f s" % (end - start))
        start = end
    
        # Exports shallow and bottom points to a specified file if export option is provided.
        if export_shallow_file is not None:
            export_shallow_outputs = np.zeros([n_shallow_points+n_bottom_points, 3])
            for i in range(n_shallow_points):
                idx = shallow_points_idx[i]
                export_shallow_outputs[i] = points[idx]
            for i in range(n_shallow_points, n_shallow_points+n_bottom_points):
                idx = bottom_points_idx[i-n_shallow_points]
                export_shallow_outputs[i] = points[idx]
            with open(export_shallow_file, 'w') as fout:
                np.savetxt(fout, export_shallow_outputs, header="X Y Z\n%d %d" % (n_shallow_points, n_bottom_points))
            print("\t%s: Write output file %s" % (func_name(), export_shallow_file))
            end = time.time()
            print("\tWrite output file, takes %.2f s" % (end - start))
            start = end
    
        # Identifies furthest shallow point based on angle for distance calculations.
        Phi = np.zeros(n_shallow_points)
        for i in range(n_shallow_points):
            idx = shallow_points_idx[i]
            x, y, z = points[idx]
            r, th, ph = cart2sph(x, y, z)
            Phi[i] = ph
    
        i_max = np.argmax(Phi)
        id_max = shallow_points_idx[i_max]
        phi_max = Phi[i_max]
    
        # Retrieves initial furthest point coordinates for calculating distances to bottom points.
        x_max, y_max, z_max = points[id_max]
        min_dist = float('inf')
        i_min, min_dist = minimum_distance_array(bottom_points, x_max, y_max, z_max)
        x_b_min, y_b_min, z_b_min = bottom_points[i_min]
    
        outputs = {}
        outputs["original"] = {"points": [x_max, y_max, z_max], "match points": [x_b_min, y_b_min, z_b_min], "distance": min_dist}
    
        end = time.time()
        print("\tCalculate original distance, takes %.2f s" % (end - start))
        start = end
    
        # Applies corrections to the furthest point based on incremental angle adjustments.
        if not self.is_chunk:
            # Ensures only chunk geometry is handled; raises exception otherwise.
            raise NotImplementedError()
    
        dphi = 0.001 * np.pi / 180.0
        dphi_increment = 0.001 * np.pi / 180.0
        phi_new = phi_max
        i_min = None
        while dphi < 1.0:
            phi = phi_max - dphi
            x_new, y_new, z_new = ggr2cart(0.0, phi, self.Ro)
            i_min, min_dist = minimum_distance_array(bottom_points, x_new, y_new, z_new)
            if min_dist < 9e3:
                phi_new = phi
                break
            dphi += dphi_increment
 
        x_new, y_new, z_new = ggr2cart(0.0, phi_new, self.Ro)
        x_b_min, y_b_min, z_b_min = bottom_points[i_min]
    
        outputs["corrected"] = {"points": [x_new, y_new, z_new], "match points": [x_b_min, y_b_min, z_b_min], "distance": min_dist}

        self.shallow_trench = [x_new, y_new, z_new]
    
        end = time.time()
        print("\tPerform correction, takes %.2f s" % (end - start))
        start = end
    
        return outputs

    def PrepareSlab(self, slab_field_names, **kwargs):
        '''
        Prepares slab composition by identifying slab and crustal regions in the model.
        
        Parameters:
            slab_field_names (list): Names of fields used to define slab composition.
            kwargs (dict): Optional keyword arguments:
                - prepare_moho (str, optional): Field name for core-mantle boundary (moho) preparation.
                - prepare_slab_distant_properties (optional): Additional properties for distant slab.
                - slab_threshold (float, optional): Threshold value for slab field to classify as slab material.
                - depth_lookup (float, optional): Depth for slab surface lookup, default is 100 km.
                - depth_distant_lookup (float, optional): Depth threshold for distant properties, default is 200 km.
        '''
        # Ensure cell centers are included in data
        assert(self.include_cell_center)

        print("%s started" % func_name())
        start = time.time()

        # Initialize optional parameters from kwargs
        prepare_moho = kwargs.get('prepare_moho', None)
        prepare_slab_distant_properties = kwargs.get('prepare_slab_distant_properties', None)
        slab_threshold = kwargs.get('slab_threshold', 0.2)
        depth_lookup = kwargs.get("depth_lookup", 100e3)
        depth_distant_lookup = kwargs.get('depth_distant_lookup', 200e3)

        # Extract point and cell data arrays from VTK data structures
        points = vtk_to_numpy(self.i_poly_data.GetPoints().GetData())
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())
        point_data = self.i_poly_data.GetPointData()
        cell_point_data = self.c_poly_data.GetPointData()
        
        # Get the primary slab field and initialize crust field if `prepare_moho` is set
        slab_field = OperateDataArrays(cell_point_data, slab_field_names,\
        [0 for i in range(len(slab_field_names) - 1)])
        crust_field = None  # store the field of crust composition
        # todo_field
        if prepare_moho is not None:
            # convert to list
            if isinstance(prepare_moho, str):
                prepare_moho = [prepare_moho]
            assert isinstance(prepare_moho, list)
            # prepare the crust field
            crust_field = vtk_to_numpy(cell_point_data.GetArray(prepare_moho[0]))
            for i in range(1, len(prepare_moho)):
                crust_field += vtk_to_numpy(cell_point_data.GetArray(prepare_moho[i]))
        
        # Identify cells based on composition and radius, storing minimum radius for slab depth
        min_r = self.Ro
        for i in range(self.i_poly_data.GetNumberOfCells()):
            cell = self.i_poly_data.GetCell(i)
            id_list = cell.GetPointIds()  # list of point ids in this cell
            x = centers[i][0]
            y = centers[i][1]
            r = get_r(x, y, self.geometry)
            slab = slab_field[i]
            if slab > slab_threshold and ((self.Ro - r) > self.slab_shallow_cutoff):
                self.slab_cells.append(i)
                if r < min_r:
                    min_r = r
        
        # If moho preparation is requested, identify crustal cells similarly
        if prepare_moho is not None:
            # cells of the crustal composition
            for i in range(self.i_poly_data.GetNumberOfCells()):
                cell = self.i_poly_data.GetCell(i)
                id_list = cell.GetPointIds()  # list of point ids in this cell
                x = centers[i][0]
                y = centers[i][1]
                r = get_r(x, y, self.geometry)
                crust = crust_field[i]
                if crust > slab_threshold and ((self.Ro - r) > self.slab_shallow_cutoff):
                    self.crust_cells.append(i)
        
        # Calculate slab depth based on minimum radius
        self.slab_depth = self.Ro - min_r

        end=time.time()
        print("\tIdentifing cells takes %.2f s" % (end-start))
        start=time.time()

        # Group slab cells into envelop intervals based on radial depth
        total_en_interval = int((self.slab_depth - self.slab_shallow_cutoff) // self.slab_envelop_interval + 1)
        slab_en_cell_lists = [ [] for i in range(total_en_interval) ]
        for id in self.slab_cells:
            x = centers[id][0]  # first, separate cells into intervals
            y = centers[id][1]
            r = get_r(x, y, self.geometry)
            id_en =  int(np.floor(
                                  (self.Ro - r - self.slab_shallow_cutoff)/
                                  self.slab_envelop_interval))# id in the envelop list
            slab_en_cell_lists[id_en].append(id)
        
        end=time.time()
        print("\tGroup cells takes %.2f s" % (end-start))
        start=time.time()

        # Find angular boundaries (min and max theta) for each slab interval
        for id_en in range(len(slab_en_cell_lists)):
            theta_min = 0.0  # then, loop again by intervals to look for a
            theta_max = 0.0  # max theta and a min theta for each interval
            cell_list = slab_en_cell_lists[id_en]
            if len(cell_list) == 0:
                continue  # make sure we have some point
            is_first = True
            id_min = -1
            id_max = -1
            for id in cell_list:
                x = centers[id][0]
                y = centers[id][1]
                theta = get_theta(x, y, self.geometry)  # cart
                if is_first:
                    id_min = id
                    id_max = id
                    theta_min = theta
                    theta_max = theta
                    is_first = False
                else:
                    if theta < theta_min:
                        id_min = id
                        theta_min = theta
                    if theta > theta_max:
                        id_max = id
                        theta_max = theta
            self.slab_envelop_cell_list0.append(id_min)  # first half of the envelop
            self.slab_envelop_cell_list1.append(id_max)  # second half of the envelop
        
        end=time.time()
        print("\tFinding regular boundaries takes %.2f s" % (end-start))
        start=time.time()
        
        # Identify the trench position based on maximum angular position
        id_tr = self.slab_envelop_cell_list1[0] # point of the trench
        x_tr = centers[id_tr][0]  # first, separate cells into intervals
        y_tr = centers[id_tr][1]
        self.trench = get_theta(x_tr, y_tr, self.geometry)

        # Calculate dip angle at a specified depth lookup (e.g., 100 km)
        self.coord_100 = self.SlabSurfDepthLookup(depth_lookup)
        if self.geometry == "chunk":
            x100 = (self.Ro - depth_lookup) * np.cos(self.coord_100)
            y100 = (self.Ro - depth_lookup) * np.sin(self.coord_100)
        elif self.geometry == "box":
            x100 = self.coord_100
            y100 = self.Ro - depth_lookup
        r100 = get_r(x100, y100, self.geometry)
        theta100 = get_theta(x100, y100, self.geometry)
        self.dip_100 = get_dip(x_tr, y_tr, x100, y100, self.geometry)
        
        end=time.time()
        print("\tFinding trench and dip angle takes %.2f s" % (end-start))
        start=time.time()

        # If moho is being prepared, repeat envelope grouping and interval checks for crust cells
        if prepare_moho is not None:
            # get the crust envelop
            crust_en_cell_lists = [ [] for i in range(total_en_interval) ]
            for id in self.crust_cells:
                x = centers[id][0]  # first, separate cells into intervals
                y = centers[id][1]
                r = get_r(x, y, self.geometry)
                id_en =  int(np.floor(
                                    (self.Ro - r - self.slab_shallow_cutoff)/
                                    self.slab_envelop_interval))# id in the envelop list
                crust_en_cell_lists[id_en].append(id) 

            # Identify angular boundaries for crust cells to match slab envelope intervals
            for id_en in range(len(crust_en_cell_lists)):
                theta_min = 0.0  # then, loop again by intervals to look for a
                # theta_max = 0.0  # max theta and a min theta for each interval
                cell_list = crust_en_cell_lists[id_en]
                if len(cell_list) == 0:
                    if len(slab_en_cell_lists[id_en]) == 0:
                        pass
                    else:
                        # if there are points in the slab interval list
                        # I'll append some non-sense value here to make sure these 
                        # two have the same size
                        self.moho_envelop_cell_list.append(-1)
                    continue  # make sure we have some point
                is_first = True
                id_min = -1
                # id_max = -1
                for id in cell_list:
                    x = centers[id][0]
                    y = centers[id][1]
                    theta = get_theta(x, y, self.geometry)  # cart
                    if is_first:
                        id_min = id
                        # id_max = id
                        theta_min = theta
                        # theta_max = theta
                        is_first = False
                    else:
                        if theta < theta_min:
                            id_min = id
                            theta_min = theta
                        # if theta > theta_max:
                        #     id_max = id
                        #    theta_max = theta
                self.moho_envelop_cell_list.append(id_min)  # first half of the envelop

            # Ensure crust and slab envelopes have equal lengths 
            assert(len(self.moho_envelop_cell_list)==len(self.slab_envelop_cell_list1))
        
            end=time.time()
            print("\tPrepare moho envelop takes %.2f s" % (end-start))
            start=time.time()
    

    def PrepareSlabByDT(self, **kwargs):
        '''
        prepare slab composition by temperature difference to the reference adiabat
        Inputs:
            Tref_func: a function for the reference T profile.
        '''
        assert(self.include_cell_center)
        assert(self.Tref_func != None)
        slab_threshold = kwargs.get('slab_threshold', -100.0)
        points = vtk_to_numpy(self.i_poly_data.GetPoints().GetData())
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())
        point_data = self.i_poly_data.GetPointData()
        cell_point_data = self.c_poly_data.GetPointData()
        # the temperature field
        T_field = vtk_to_numpy(cell_point_data.GetArray("T"))
        # add cells by composition
        min_r = self.Ro
        for i in range(self.i_poly_data.GetNumberOfCells()):
            cell = self.i_poly_data.GetCell(i)
            id_list = cell.GetPointIds()  # list of point ids in this cell
            x = centers[i][0]
            y = centers[i][1]
            r = get_r(x, y, self.geometry)
            Tref = self.Tref_func(self.Ro - r)
            T = T_field[i]
            if T - Tref < slab_threshold and ((self.Ro - r) > self.slab_shallow_cutoff):
                # note on the "<": slab internal is cold
                self.slab_cells.append(i)
                if r < min_r:
                    min_r = r
        self.slab_depth = self.Ro - min_r  # cart
        # get slab envelops
        total_en_interval = int((self.slab_depth - self.slab_shallow_cutoff) // self.slab_envelop_interval + 1)
        slab_en_cell_lists = [ [] for i in range(total_en_interval) ]
        for id in self.slab_cells:
            x = centers[id][0]  # first, separate cells into intervals
            y = centers[id][1]
            r = get_r(x, y, self.geometry)
            id_en =  int(np.floor(
                                  (self.Ro - r - self.slab_shallow_cutoff)/
                                  self.slab_envelop_interval))# id in the envelop list
            slab_en_cell_lists[id_en].append(id)
        for id_en in range(len(slab_en_cell_lists)):
            theta_min = 0.0  # then, loop again by intervals to look for a
            theta_max = 0.0  # max theta and a min theta for each interval
            cell_list = slab_en_cell_lists[id_en]
            if len(cell_list) == 0:
                continue  # make sure we have some point
            is_first = True
            id_min = -1
            id_max = -1
            for id in cell_list:
                x = centers[id][0]
                y = centers[id][1]
                theta = get_theta(x, y, self.geometry)  # cart
                if is_first:
                    id_min = id
                    id_max = id
                    theta_min = theta
                    theta_max = theta
                    is_first = False
                else:
                    if theta < theta_min:
                        id_min = id
                        theta_min = theta
                    if theta > theta_max:
                        id_max = id
                        theta_max = theta
            self.slab_envelop_cell_list0.append(id_min)  # first half of the envelop
            self.slab_envelop_cell_list1.append(id_max)  # second half of the envelop
        # trench
        id_tr = self.slab_envelop_cell_list1[0] # point of the trench
        x_tr = centers[id_tr][0]  # first, separate cells into intervals
        y_tr = centers[id_tr][1]
        self.trench = get_theta(x_tr, y_tr, self.geometry)
        # 100 km dip angle
        depth_lookup = 100e3
        self.coord_100 = self.SlabSurfDepthLookup(depth_lookup)
        if self.geometry == "chunk":
            x100 = (self.Ro - depth_lookup) * np.cos(self.coord_100)
            y100 = (self.Ro - depth_lookup) * np.sin(self.coord_100)
        elif self.geometry == "box":
            x100 = self.coord_100
            y100 = self.Ro - depth_lookup
        r100 = get_r(x100, y100, self.geometry)
        theta100 = get_theta(x100, y100, self.geometry)
        self.dip_100 = get_dip(x_tr, y_tr, x100, y100, self.geometry)
        pass

    def GetDipAtDepth(self, depth_lookup, depth_interval):
        # 100 km dip angle
        self.coord_0 = self.SlabSurfDepthLookup(depth_lookup-depth_interval)
        self.coord_1 = self.SlabSurfDepthLookup(depth_lookup)
        x_0, y_0, x_1, y_1 = None, None, None, None
        if self.geometry == "chunk":
            x_0 = (self.Ro - depth_lookup +depth_interval) * np.cos(self.coord_0)
            y_0 = (self.Ro - depth_lookup +depth_interval) * np.sin(self.coord_0)
            x_1 = (self.Ro - depth_lookup) * np.cos(self.coord_1)
            y_1 = (self.Ro - depth_lookup) * np.sin(self.coord_1)
        elif self.geometry == "box":
            x_0 = self.coord_0
            y_0 = self.Ro - depth_lookup + depth_interval
            x_1 = self.coord_1
            y_1 = self.Ro - depth_lookup
        r_0 = get_r(x_0, y_0, self.geometry)
        theta_0 = get_theta(x_0, y_0, self.geometry)
        r_1 = get_r(x_1, y_1, self.geometry)
        theta_1 = get_theta(x_1, y_1, self.geometry)
        dip = get_dip(x_0, y_0, x_1, y_1, self.geometry)
        print("x_0, y_0: ", x_0, y_0) # debug
        print("x_1, y_1: ", x_1, y_1)
        return dip

    def ExportOvAthenProfile(self, depth_distant_lookup, **kwargs):
        '''
        query a profile ending at depth_distant_lookup that is 5 deg to the
        Inputs:
            depth_distant_lookup - depth to place this query point
        '''
        project_velocity = kwargs.get("project_velocity", True)
        n_sample = kwargs.get("n_sample", 100)
        
        assert((self.coord_100 is not None) and (self.coord_100 is not None))

        # query the poly_data 
        query_grid = np.zeros((n_sample,2))
        v_distant_profile = np.zeros((n_sample, 2))
        depths = np.zeros(n_sample)
        value_fixings = np.zeros(n_sample)
        for i in range(n_sample):
            x_distant_200 = None; y_distant_200 = None
            depth = depth_distant_lookup * (1.0 * i / (n_sample-1.0))
            depths[i] = depth
            if self.geometry == "chunk":
                x_distant_200 = (self.Ro - depth) * np.cos(self.coord_100 + 5.0 * np.pi / 180.0)
                y_distant_200 = (self.Ro - depth) * np.sin(self.coord_100 + 5.0 * np.pi / 180.0)
            elif self.geometry == "box":
                x_distant_200 = self.coord_100 + 5.0 * np.pi / 180.0 * self.Ro
                y_distant_200 = self.Ro - depth
            query_grid[i, 0] = x_distant_200
            query_grid[i, 1] = y_distant_200
        # interpolate 
        query_poly_data = InterpolateGrid(self.i_poly_data, query_grid, quiet=True)
        query_vs = vtk_to_numpy(query_poly_data.GetPointData().GetArray('velocity'))
        query_viscs = vtk_to_numpy(query_poly_data.GetPointData().GetArray('viscosity'))
        # fix_missing_data
        # reason: interpolation fails upon chaning resolution
        # strategy: perturb by 0.1 degree at a time
        for i in range(n_sample):
            while (abs(query_viscs[i])) < 1e-6:
                # data missin
                value_fixings[i] += 1.0
                if self.geometry == "chunk":
                    # perturb by 0.1 degress
                    x_distant_200_1 = (self.Ro - depths[i]) * np.cos(self.coord_100 + (5.0 - 0.1*value_fixings[i]) * np.pi / 180.0)
                    y_distant_200_1 = (self.Ro - depths[i]) * np.sin(self.coord_100 + (5.0 - 0.1*value_fixings[i]) * np.pi / 180.0)
                    x_distant_200_2 = (self.Ro - depths[i]) * np.cos(self.coord_100 + (5.0 + 0.1*value_fixings[i]) * np.pi / 180.0)
                    y_distant_200_2 = (self.Ro - depths[i]) * np.sin(self.coord_100 + (5.0 + 0.1*value_fixings[i]) * np.pi / 180.0)
                elif self.geometry == "box":
                    x_distant_200_1 = self.coord_100 + (5.0 - 0.1*value_fixings[i]) * np.pi / 180.0 * self.Ro
                    y_distant_200_1 = self.Ro - depths[i]
                    x_distant_200_2 = self.coord_100 + (5.0 + 0.1*value_fixings[i]) * np.pi / 180.0 * self.Ro
                    y_distant_200_2 = self.Ro - depths[i]
                query_grid1 = np.zeros((1, 2))
                query_grid1[0, 0] = x_distant_200_1
                query_grid1[0, 1] = y_distant_200_1
                query_grid2 = np.zeros((1, 2))
                query_grid1[0, 0] = x_distant_200_2
                query_grid1[0, 1] = y_distant_200_2
                query_poly_data1 = InterpolateGrid(self.i_poly_data, query_grid1, quiet=True)
                query_poly_data2 = InterpolateGrid(self.i_poly_data, query_grid2, quiet=True)
                query_v1 = vtk_to_numpy(query_poly_data1.GetPointData().GetArray('velocity'))
                query_visc1 = vtk_to_numpy(query_poly_data1.GetPointData().GetArray('viscosity'))
                query_v2 = vtk_to_numpy(query_poly_data2.GetPointData().GetArray('velocity'))
                query_visc2 = vtk_to_numpy(query_poly_data2.GetPointData().GetArray('viscosity'))
                if (abs(query_visc1)) > 1e-6:
                    # fixed
                    query_vs[i, :] = query_v1
                    query_viscs[i] = query_visc1
                    query_grid[i, :] = query_grid1
                    break
                elif (abs(query_visc2)) > 1e-6:
                    query_vs[i, :] = query_v2
                    query_viscs[i] = query_visc2
                    query_grid[i, :] = query_grid2
                    break
        # project the velocity if needed and get the viscosity
        if project_velocity:
            # project velocity to theta direction in a spherical geometry
            v_distant_profile[:, 0], v_distant_profile[:, 1] = ProjectVelocity(x_distant_200, y_distant_200, query_vs, self.geometry)
        else:
            v_distant_profile[:, 0], v_distant_profile[:, 1] = query_vs[:, 0], query_vs[:, 1]
        
        return query_grid, depths, v_distant_profile, query_viscs, value_fixings
    
    
    def ExportSlabInfo(self):
        '''
        Output slab information
        '''
        return self.trench, self.slab_depth, self.dip_100

    def ExportVelocity(self, **kwargs):
        '''
        Output sp and ov plate velocity
        Inputs:
            kwargs:
                project_velocity - whether the velocity is projected to the tangential direction
        '''
        project_velocity = kwargs.get('project_velocity', False)
        assert(self.trench is not None)
        if self.geometry == "chunk":
            r_sp_query = self.Ro - self.velocitw_query_depth
            # theta_sp_query = self.trench - self.velocitw_query_disl_to_trench / self.Ro
            theta_sp_query = self.trench / 2.0
            r_ov_query = self.Ro - self.velocitw_query_depth
            # theta_ov_query = self.trench + self.velocitw_query_disl_to_trench / self.Ro
            theta_ov_query = (self.trench + self.Xmax) / 2.0
            x_sp_query = r_sp_query * np.cos(theta_sp_query)
            y_sp_query = r_sp_query * np.sin(theta_sp_query)
            x_ov_query = r_ov_query * np.cos(theta_ov_query)
            y_ov_query = r_ov_query * np.sin(theta_ov_query)
        elif self.geometry == "box":
            # x_sp_query = self.trench - self.velocitw_query_disl_to_trench
            x_sp_query = self.trench / 2.0
            y_sp_query = self.Ro - self.velocitw_query_depth
            # x_ov_query = self.trench + self.velocitw_query_disl_to_trench
            x_ov_query = (self.trench + self.Xmax) / 2.0
            y_ov_query = self.Ro - self.velocitw_query_depth
        query_grid = np.zeros((2,2))
        query_grid[0, 0] = x_sp_query
        query_grid[0, 1] = y_sp_query
        query_grid[1, 0] = x_ov_query
        query_grid[1, 1] = y_ov_query
        query_poly_data = InterpolateGrid(self.i_poly_data, query_grid, quiet=True)
        query_vs = vtk_to_numpy(query_poly_data.GetPointData().GetArray('velocity'))
        if project_velocity:
            # project velocity to theta direction in a spherical geometry
            self.vsp, _ = ProjectVelocity(x_sp_query, y_sp_query, query_vs[0, :], self.geometry)
            self.vov, _ = ProjectVelocity(x_ov_query, y_ov_query, query_vs[1, :], self.geometry)
        else:
            self.vsp = query_vs[0, :]
            self.vov = query_vs[1, :]
        return self.vsp, self.vov

    def SlabSurfDepthLookup(self, depth_lkp):
        '''
        Get point from the surface of the slab by depth
        '''
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())
        assert(len(self.slab_envelop_cell_list1) > 0)
        assert(depth_lkp < self.slab_depth)
        is_first = True
        coord_last = 0.0
        depth_last = 0.0
        for id in self.slab_envelop_cell_list1:
            x = centers[id][0]
            y = centers[id][1]
            r = get_r(x, y, self.geometry)
            coord = get_theta(x, y, self.geometry)
            depth = self.Ro - r
            if depth_last < depth_lkp and depth_lkp <= depth:
                coord_lkp = coord * (depth_lkp - depth_last) / (depth - depth_last) +\
                            coord_last * (depth_lkp - depth) / (depth_last - depth)
                break
            coord_last = coord
            depth_last = depth
        return coord_lkp

    
    def ExportSlabInternal(self, output_xy=False):
        '''
        export slab internal points
        '''
        cell_source = vtk.vtkExtractCells()
        cell_source.SetInputData(self.i_poly_data)
        cell_source.SetCellList(NpIntToIdList(self.slab_cells))
        cell_source.Update()
        slab_cell_grid = cell_source.GetOutput()
        if output_xy:
            coords = vtk_to_numpy(slab_cell_grid.GetPoints().GetData())
            return coords
        else:
            return slab_cell_grid
    
    def ExportSlabEnvelopCoord(self, **kwargs):
        '''
        export slab envelop envelops,
        outputs:
            coordinates in slab envelop
        '''
        assert (len(self.slab_envelop_cell_list0) > 0 and\
            len(self.slab_envelop_cell_list1) > 0)  # assert we have slab internels
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())
        slab_envelop0 = []
        slab_envelop1 = []
        # envelop 0
        xs = []
        ys = []
        for id in self.slab_envelop_cell_list0:
            x = centers[id][0]
            y = centers[id][1]
            xs.append(x)
            ys.append(y)
        slab_envelop0 = np.array([xs, ys])
        # envelop 1
        xs = []
        ys = []
        for id in self.slab_envelop_cell_list1:
            x = centers[id][0]
            y = centers[id][1]
            xs.append(x)
            ys.append(y)
        slab_envelop1 = np.array([xs, ys])
        return slab_envelop0.T, slab_envelop1.T

    def ExportSlabmohoCoord(self, **kwargs):
        '''
        export slab core-mantle boundary envelop
        returns:
            coordinates of points on the core-mantle boundary
        '''
        assert (len(self.moho_envelop_cell_list) > 0)
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())
        xs = []
        ys = []
        for id in self.moho_envelop_cell_list:
            x = -1e31  # these initial value are none sense
            y = -1e31  # but if there are negative ids, just maintain these values
            if id > 0:
                x = centers[id][0]
                y = centers[id][1]
            xs.append(x)
            ys.append(y)
        moho_envelop = np.array([xs, ys])
        return moho_envelop.T

    def ExportWedgeT(self, **kwargs):
        '''
        export the temperature in the mantle wedge
        Inputs:
            kwargs:
                fileout - path for output temperature, if this is None,
                        then no output is generated
        '''
        fileout = kwargs.get('fileout', None)
        depth_lookup = 100e3
        min_depth = 0.0
        max_depth = 100e3
        n_points = 100  # points for vtk interpolation
        o_n_points = 200  # points for output
        depths = np.linspace(0.0, max_depth, n_points)
        o_depths = np.linspace(0.0, max_depth, o_n_points)
        o_Ts = np.zeros(o_n_points)
        # look up for the point on the slab that is 100 km deep
        coord_lookup = self.SlabSurfDepthLookup(depth_lookup)
        vProfile2D = self.VerticalProfile2D((self.Ro - max_depth, self.Ro),\
                                            coord_lookup, n_points, fix_point_value=True)
        Tfunc = vProfile2D.GetFunction('T')
        outputs = "# 1: x (m)\n# 2: y (m)\n# 3: T\n"
        is_first = True
        for i in range(o_n_points):
            depth = o_depths[i]
            if self.geometry == "chunk":
                radius = self.Ro - depth
                x = radius * np.cos(coord_lookup)
                y = radius * np.sin(coord_lookup)
            elif self.geometry == "box":
                x = coord_lookup
                y = self.Ro - depth
            T = Tfunc(self.Ro - depth)
            o_Ts[i] = T
            if is_first:
                is_first = False
            else:
                outputs += "\n"
            outputs += "%.4e %.4e %.4e" % (x, y, T)
        if fileout is not None:
            with open(fileout, 'w') as fout:
                fout.write(outputs)
            print("%s: write file %s" % (func_name(), fileout))  # screen output
        return o_depths, o_Ts

    def ExportTrenchT(self, **kwargs):
        '''
        export the temperature in the mantle wedge
        Inputs:
            trench: coordinate of trench position (theta or x)
            kwargs:
                fileout - path for output temperature, if this is None,
                        then no output is generated
        '''
        assert(self.trench is not None)  # assert that the trench posiiton is processed
        fileout = kwargs.get('fileout', None)
        distance_to_trench = 200e3
        # the point to look up needs to be on the the subducting plate
        if self.geometry == 'box':
            lookup = self.trench - distance_to_trench
        elif self.geometry == 'chunk':
            lookup = self.trench - distance_to_trench / self.Ro
        depth_lookup = 100e3
        min_depth = 0.0
        max_depth = 100e3
        n_points = 100  # points for vtk interpolation
        o_n_points = 200  # points for output
        depths = np.linspace(0.0, max_depth, n_points)
        o_depths = np.linspace(0.0, max_depth, o_n_points)
        o_Ts = np.zeros(o_n_points)
        # look up for the point on the slab that is 100 km deep
        vProfile2D = self.VerticalProfile2D((self.Ro - max_depth, self.Ro),\
                                            lookup, n_points, fix_point_value=True)
        Tfunc = vProfile2D.GetFunction('T')
        outputs = "# 1: x (m)\n# 2: y (m)\n# 3: depth (m)\n# 4: T (K)\n"
        is_first = True
        for i in range(o_n_points):
            depth = o_depths[i]
            if self.geometry == "chunk":
                radius = self.Ro - depth
                x = radius * np.cos(lookup)
                y = radius * np.sin(lookup)
            elif self.geometry == "box":
                x = lookup
                y = self.Ro - depth
            T = Tfunc(self.Ro - depth)
            o_Ts[i] = T
            if is_first:
                is_first = False
            else:
                outputs += "\n"
            outputs += "%.4e %.4e %.4e %.4e" % (x, y, depth, T)
        if fileout is not None:
            with open(fileout, 'w') as fout:
                fout.write(outputs)
            print("%s: write file %s" % (func_name(), fileout))  # screen output
        return o_depths, o_Ts

    def SlabBuoyancy(self, v_profile, depth_increment):
        '''
        Compute the slab buoyancy
        Inputs:
            v_profile: vertical profile containing the reference profile
        Outputs:
            total_buoyancy: the total buoyancy forces in N/m
            b_profile: the depths and buoyancies in np array, 
                    the depths serve as the center in [depth - depth_increment/2.0, depth + depth_increment/2.0]
                    and the buoyancies contain a correspondent value for each range

        '''
        grav_acc = 10.0
        assert(self.include_cell_center)
        assert(len(self.slab_cells) > 0)
        n_depth = int(np.ceil(self.slab_depth / depth_increment))
        buoyancies = np.zeros(n_depth) # save the values of buoyancy with ranges of depths
        depths = []  # construct depths, each serve as the center in [depth - depth_increment/2.0, depth + depth_increment/2.0]
        for i in range(n_depth):
            depth = (i + 0.5) * depth_increment
            depths.append(depth)
        depths = np.array(depths)
        centers = vtk_to_numpy(self.c_poly_data.GetPoints().GetData())  # note these are data mapped to cell center
        density_data = vtk_to_numpy(self.c_poly_data.GetPointData().GetArray('density'))
        density_ref_func = v_profile.GetFunction('density')
        # now compute the slab buoyancy
        total_buoyancy = 0.0
        for i in self.slab_cells:
            x = centers[i][0]
            y = centers[i][1]
            r = get_r(x, y, self.geometry)
            i_r = int(np.floor((self.Ro - r) / depth_increment))
            density = density_data[i]
            density_ref = density_ref_func(r)
            cell_size = self.cell_sizes[i]  # temp
            buoyancy = - grav_acc * (density - density_ref) * cell_size  # gravity
            buoyancies[i_r] += buoyancy
            total_buoyancy += buoyancy
        b_profile = np.zeros((n_depth, 2))
        b_profile[:, 0] = depths
        b_profile[:, 1] = buoyancies
        return total_buoyancy, b_profile

    def FindMDD(self, **kwargs):
        '''
        find the mechanical decoupling depth from the velocity field
        '''
        print("%s started" % func_name())
        start=time.time()
        dx0 = kwargs.get('dx0', 10e3)
        dx1 = kwargs.get('dx1', 10e3)
        tolerance = kwargs.get('tolerance', 0.05)
        indent = kwargs.get("indent", 0)  # indentation for outputs
        debug = kwargs.get("debug", False) # output debug messages
        extract_depths = kwargs.get("extract_depths", None) # extract additional depth
        slab_envelop0, slab_envelop1 = self.ExportSlabEnvelopCoord()
        query_grid = np.zeros((2,2))
        mdd = -1.0
        for i in range(slab_envelop1.shape[0]):
            x = slab_envelop1[i, 0]
            y = slab_envelop1[i, 1]
            r = get_r(x, y, self.geometry) 
            theta = get_theta(x, y, self.geometry)
            if self.geometry == "chunk":
                x0 = r * np.cos(theta - dx0/self.Ro) 
                y0 = r * np.sin(theta - dx0/self.Ro)
                x1 = r * np.cos(theta + dx1/self.Ro)
                y1 = r * np.sin(theta + dx1/self.Ro)
            elif self.geometry == "box":
                x0 = x - dx0
                y0 = y
                x1 = x + dx1
                y1 = y
            query_grid[0, 0] = x0
            query_grid[0, 1] = y0
            query_grid[1, 0] = x1
            query_grid[1, 1] = y1
            query_poly_data = InterpolateGrid(self.i_poly_data, query_grid, quiet=True)
            query_vs = vtk_to_numpy(query_poly_data.GetPointData().GetArray('velocity'))
            vi = query_vs[0, :]
            vi_mag = (query_vs[0, 0]**2.0 + query_vs[0, 1]**2.0)**0.5
            vi_theta = np.arctan2(query_vs[0,0], query_vs[0, 1])
            vo = query_vs[1, :]
            vo_mag = (query_vs[1, 0]**2.0 + query_vs[1, 1]**2.0)**0.5
            vo_theta = np.arctan2(query_vs[1,0], query_vs[1, 1])
            depth = (self.Ro - (x**2.0 + y**2.0)**0.5)
            if debug:
                print("%sx: %.4e, y: %.4e, depth: %.4e, vi: [%.4e, %.4e] (mag = %.4e, theta = %.4e), vo: [%.4e, %.4e] (mag = %.4e, theta = %.4e)"\
                % (indent*" ", x, y, depth, vi[0], vi[1], vi_mag, vi_theta, vo[0], vo[1], vo_mag, vo_theta))
            if (abs((vo_mag - vi_mag)/vi_mag) < tolerance) and (abs((vo_theta - vi_theta)/vi_theta) < tolerance):
                # mdd depth
                mdd = depth
                # extract a horizontal profile at this dept
                if self.geometry == "chunk":
                    query_grid, query_vs = self.extract_mdd_profile(r, theta)
                elif self.geometry == "box":
                    query_grid, query_vs = self.extract_mdd_profile(x, y)
                break
        print("\t%sfindmdd_tolerance = %.4e, dx0 = %.4e, dx1 = %.4e" % (indent*" ", tolerance, dx0, dx1))
        end = time.time()
        print("\tFinding mdd depth takes %.2f s" % (end-start))
        start = end

        extract_profiles_grid = []; extract_profiles_vs = []
        if extract_depths is not None:
            rs = (slab_envelop1[:, 0]**2.0 + slab_envelop1[:, 1]**2.0)**0.5
            nearest_interp_x = interp1d(rs, slab_envelop1[:, 0], kind='linear', fill_value='extrapolate')
            nearest_interp_y = interp1d(rs, slab_envelop1[:, 1], kind='linear', fill_value='extrapolate')
            for i, extract_depth in enumerate(extract_depths):
                extract_r = self.Ro - extract_depth
                xc = nearest_interp_x(extract_r)
                yc = nearest_interp_y(extract_r)
                if self.geometry == "chunk":
                    rc = get_r(xc, yc, self.geometry) 
                    theta_c = get_theta(xc, yc, self.geometry)
                    query_grid_foo, query_vs_foo = self.extract_mdd_profile(rc, theta_c)
                elif self.geometry == "box":
                    query_grid_foo, query_vs_foo = self.extract_mdd_profile(xc, yc)
                extract_profiles_grid.append(query_grid_foo)
                extract_profiles_vs.append(query_vs_foo)
            
        end = time.time()
        print("\tExtracting extra profiles takes %.2f s" % (end-start))

        if mdd > 0.0:
            print("%smdd = %.4e m" % (indent*" ", mdd))
            return mdd, query_grid, query_vs, extract_profiles_grid, extract_profiles_vs
        else:
            raise ValueError("FindMDD: a valid MDD has not been found, please considering changing the tolerance")
        pass

    def extract_mdd_profile(self, coord1, coord2):
        '''
        extract mdd profile at given depths
        '''
        # extract a horizontal profile at this dept
        n_query1 = 51
        query_grid1 = np.zeros((n_query1,2))
        dx2 = 10e3 # query distance of the profile
        for i in range(n_query1):
            if self.geometry == "chunk":
                theta_i = coord2 + ((n_query1-1.0-i) * (-1.0*dx2) + i*dx2)/(n_query1-1.0)/self.Ro
                xi = coord1 * np.cos(theta_i) 
                yi = coord1 * np.sin(theta_i)
            elif self.geometry == "box":
                xi = coord1 + ((n_query1-1.0-i) * (-1.0*dx2) + i*dx2)/(n_query1-1.0)
                yi = coord2
            else:
                raise NotImplementedError()
            query_grid1[i, 0] = xi
            query_grid1[i, 1] = yi
        query_poly_data1 = InterpolateGrid(self.i_poly_data, query_grid1, quiet=True)
        query_vs1 = vtk_to_numpy(query_poly_data1.GetPointData().GetArray('velocity'))
        return query_grid1, query_vs1

    def PrepareSZ(self, fileout, **kwargs):
        '''
        Get the geometry of the shear zone
        '''
        contour_poly_data = ExportContour(self.i_poly_data, 'spcrust', 0.9, fileout=fileout)
        contour_data = vtk_to_numpy(contour_poly_data.GetPoints().GetData())
        Dsz = kwargs.get("Dsz", 7.5e3)
        max_extent = 50e3
        min_extent = 2e3
        min_depth = 10e3
        max_depth = 200e3
        depth_interval = 2e3
        inf = 1e31 # define a vary big value
        small_depth_variation = 500 # a negalectable variation in depth
        theta_subgroup = np.pi / 10.0 # 1 degree
        n_group = int(np.floor(max_depth / depth_interval) + 1.0)
        idx_groups = [[] for i in range(n_group)]
        sz_depths  = []  # initial arrays
        sz_theta_mins = []
        sz_theta_maxs = []
        # step 1: a. assign points in groups, with an interval in depth
        #         b. append points on the surface
        for i in range(contour_data.shape[0]):
            x = contour_data[i, 0]
            y = contour_data[i, 1]
            r = get_r(x, y, self.geometry)
            # theta = get_theta(x, y, self.geometry) 
            depth = self.Ro - r
            if depth < max_depth and depth > min_depth:
                # assign points in groups, with an interval in depth
                i_group = int(np.floor(depth / depth_interval))
                idx_groups[i_group].append(i)
        # step 2: look for the points located near to a query point in depth
        for i_group in range(n_group-1):
            if len(idx_groups[i_group])==0:
                # skip groups that has no points
                continue
            query_depth = depth_interval * (i_group + 0.5)
            theta_min = np.pi
            theta_max = 0.0
            for i in (idx_groups[i_group]):
                # find the minimum and maximum theta in this group
                x = contour_data[i, 0]
                y = contour_data[i, 1]
                theta = get_theta(x, y, self.geometry) 
                if theta < theta_min and True:
                    theta_min = theta
                if theta > theta_max and True:
                    theta_max = theta
            extent = 0.0
            if self.geometry == "box":
                extent = theta_max - theta_min
            elif self.geometry == "chunk":
                extent = (self.Ro - depth) * (theta_max - theta_min)
            else:
                raise ValueError("Geometry must be either box or chunk")
            if extent < max_extent and extent > min_extent:
                # select the reasonable values of layer thickness
                sz_depths.append(query_depth)
                sz_theta_mins.append(theta_min)
                sz_theta_maxs.append(theta_max)
        # loop again, append some points
        sz_theta_min_start = sz_theta_mins[0]
        sz_depths_app = []
        sz_theta_mins_app = [] 
        sz_theta_maxs_app = []
        theta_max_moho = 0.0
        depth_at_theta_max_moho = 0.0
        found = False
        for i in range(contour_data.shape[0]):
            # find the points on the subducting plate moho, and this point has to be below the subducting
            # plate and near the trench
            x = contour_data[i, 0]
            y = contour_data[i, 1]
            r = get_r(x, y, self.geometry)
            theta = get_theta(x, y, self.geometry) 
            depth = self.Ro - r
            if abs(depth - Dsz) < small_depth_variation and theta < sz_theta_min_start and (theta-sz_theta_min_start)/sz_theta_min_start > -0.1:
                # print("theta: ",theta)
                found = True
                if (theta > theta_max_moho):
                    theta_max_moho = theta
                    depth_at_theta_max_moho = depth
        my_assert(found, ValueError, "PrepareSZ: No point found on moho near the trench.")
        sz_depths_surf_shallow = []
        sz_theta_mins_surf_shallow = []
        sz_theta_maxs_surf_shallow = []
        for i in range(contour_data.shape[0]):
            # find the points on the subducting surface and shallower than the initial depth
            x = contour_data[i, 0]
            y = contour_data[i, 1]
            r = get_r(x, y, self.geometry)
            theta = get_theta(x, y, self.geometry) 
            depth = self.Ro - r
            if depth > 0.0 and depth < depth_at_theta_max_moho and theta > theta_max_moho:
                pass
                sz_depths_surf_shallow.append(depth)
                sz_theta_mins_surf_shallow.append(-inf)
                sz_theta_maxs_surf_shallow.append(theta)
        sz_depths_app += sz_depths_surf_shallow # append the points on the subducting surface and shallower than the initial depth
        sz_theta_mins_app += sz_theta_mins_surf_shallow
        sz_theta_maxs_app += sz_theta_maxs_surf_shallow
        sz_depths_app.append(0.0) # append two points, one on the surface, another on the moho
        sz_theta_mins_app.append(-inf)
        sz_theta_maxs_app.append(theta_max_moho)
        sz_depths_app.append(depth_at_theta_max_moho)
        sz_theta_mins_app.append(theta_max_moho)
        sz_theta_maxs_app.append(inf)
        sz_depths = sz_depths_app + sz_depths # append to the front
        sz_theta_mins = sz_theta_mins_app + sz_theta_mins
        sz_theta_maxs = sz_theta_maxs_app + sz_theta_maxs
        self.sz_geometry = np.array([sz_depths, sz_theta_mins, sz_theta_maxs]).transpose()
        # write output file
        header = "# 1: depth (m)\n# 2: theta_min\n# 3: theta_max\n"
        with open(fileout, 'w') as fout:
            fout.write(header)
            np.savetxt(fout, self.sz_geometry)
        print("%s: file output: %s" % (func_name(), fileout))
                
####
# Utilities functions
####


def get_theta(x, y, geometry):
    '''
    Get theta (the second coordinate)
    Inputs:
        x - x coordinate
        y - y coordinate
        geometry - 'chunk' or 'box'
    '''
    if geometry == 'chunk':
        theta = np.arctan2(y, x)  # cart
    elif geometry == 'box':
        theta = x
    else:
        raise ValueError("not implemented")
    return theta


def get_dip(x0, y0, x1, y1, geometry):
    '''
    Get dip angle
    Inputs:
        x0, y0: coordinates of the first point
        x1, y1: coordinates of the second point
        geometry - 'chunk' or 'box'
    '''
    if geometry == 'chunk':
        # here, for the 2nd dimension, we need something multiple the change in theta,
        # and I pick (r1 + r0)/2.0 for it.
        theta0 = np.arctan2(y0, x0)  # cart
        theta1 = np.arctan2(y1, x1)  # cart
        dtheta = theta1 - theta0
        r0 = (x0*x0 + y0*y0)**0.5
        r1 = (x1*x1 + y1*y1)**0.5
        # dip = np.arctan2(r0-r1*np.cos(dtheta), r1*np.sin(dtheta))
        dip = np.arctan2(r0-r1, (r1 + r0)/2.0*dtheta)
    elif geometry == 'box':
        dip = np.arctan2(-(y1-y0), (x1-x0))
    else:
        raise ValueError("not implemented")
    return dip


class T_FIT_FUNC():
    '''Residual functions'''
    def __init__(self, depth_to_fit, T_to_fit, **kwargs):
        '''
        Inputs:
            depth_to_fit - an array of depth
            T_to_fit - an array of temperature to fit
            kwargs:
                potential_temperature - mantle potential temperature

        '''
        self.depth_to_fit = depth_to_fit
        self.T_fit = T_to_fit
        self.potential_temperature = kwargs.get('potential_temperature', 1673.0)

    def PlateModel(self, xs):
        '''
        Inputs:
            xs - an array of non-dimensioned variables
        '''
        Ts = []
        seconds_in_yr = 3600.0 * 24 * 365
        age = xs[0] * 40 * seconds_in_yr * 1e6
        for depth in self.depth_to_fit:
            Ts.append(plate_model_temperature(depth, age=age, potential_temperature=self.potential_temperature))
        return np.linalg.norm(Ts - self.T_fit, 2.0)


def plate_model_temperature(depth, **kwargs):
    '''
    Use plate model to compute the temperature, migrated from the world builder
    in order to generate consistent result with the world builder.
    Inputs:
        x - distance to ridge, in meter
        depth - depth under the surface
        kwargs:
            plate_velocity - velocity of the plate, if this value is given, then the
                            distance to the ridge is also needed.
            distance_ridge - distance to the ridge
            age - age of the plate
    '''
    plate_velocity = kwargs.get('plate_velocity', None)
    if plate_velocity is None:
        age = kwargs['age']
    else:
        distance_ridge = kwargs['distance_ridge']
        age = distance_ridge / plate_velocity
    max_depth = kwargs.get('max_depth', 150e3)
    potential_mantle_temperature = kwargs.get('potential_temperature', 1673.0)
    top_temperature = kwargs.get('top_temperature', 273.0)
    thermal_diffusivity = 1e-6
    thermal_expansion_coefficient = 3e-5
    gravity_norm = 10.0
    specific_heat = 1250.0
    sommation_number = 100 # same as in the World Builder
    bottom_temperature_local = potential_mantle_temperature *\
                            np.exp(((thermal_expansion_coefficient* gravity_norm) / specific_heat) * depth)

    temperature = top_temperature + (bottom_temperature_local - top_temperature) * (depth / max_depth)

    for i in range(1, sommation_number+1):
        # suming over the "sommation_number"
        # use a spreading ridge around the left corner and a constant age around the right corner 
        if plate_velocity is not None:
            temperature = temperature + (bottom_temperature_local - top_temperature) *\
                        ((2.0 / (i * np.pi)) * np.sin((i * np.pi * depth) / max_depth) *\
                         np.exp((((plate_velocity * max_depth)/(2 * thermal_diffusivity)) -\
                                   np.sqrt(((plate_velocity*plate_velocity*max_depth*max_depth) /\
                                              (4*thermal_diffusivity*thermal_diffusivity)) + i * i * np.pi * np.pi)) *\
                                  ((plate_velocity * age) / max_depth)))
        else:
            temperature = temperature + (bottom_temperature_local - top_temperature) *\
                        ((2.0 / (i * np.pi)) * np.sin((i * np.pi * depth) / max_depth) *\
                         np.exp(-1.0 * i * i * np.pi * np.pi * thermal_diffusivity * age / (max_depth * max_depth)))
    return temperature 

    

####
# stepwise functions
####
def PlotSlabForces(filein, fileout, **kwargs):
    '''
    Plot slab surface profile
    Inputs:
        filein (str): path to input
        fileout (str): path to output
        kwargs (dict):
    '''
    assert(os.path.isfile(filein))
    ## load data: forces
    data = np.loadtxt(filein)
    depths = data[:, 0]
    buoyancies = data[:, 1]
    total_buoyancy = np.linalg.norm(buoyancies, 1)  # total buoyancy
    buoyancie_gradients = data[:, 2]
    pressure_lower = data[:, 3]
    pressure_upper = data[:, 4]
    differiential_pressure = data[:, 5]
    differiential_pressure_v = data[:, 6]
    differiential_pressure_v_1 = data[:, 7]
    compensation = data[:, 8]
    dynamic_pressure_lower = data[:, 9]
    dynamic_pressure_upper = data[:, 10]
    differiential_dynamic_pressure = data[:, 11]
    differiential_dynamic_pressure_v = data[:, 12]
    v_zeros = np.zeros(data.shape[0])
    fig = plt.figure(tight_layout=True, figsize=(15, 15))
    gs = gridspec.GridSpec(3, 3) 
    # figure 1: show forces
    ax = fig.add_subplot(gs[0, 0]) 
    ax.plot(buoyancie_gradients, depths/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(pressure_upper, depths/1e3, 'c--', label='sigma_n (upper) (N/m2)')
    ax.plot(pressure_lower, depths/1e3, 'm--', label='sigma_n (lower) (N/m2)')
    ax.set_title('Buoyancy gradients and total pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.legend()
    ax.invert_yaxis()
    # figure 2: buoyancy and vertical pressure differences
    ax = fig.add_subplot(gs[0, 1]) 
    ax.plot(buoyancie_gradients, depths/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(differiential_pressure, depths/1e3, 'm--', label='Pressure differences (N/m2)')
    ax.plot(differiential_pressure_v, depths/1e3, 'r--', label='Vertical pressure differences (N/m2)')
    ax.plot(v_zeros, depths/1e3, 'k--')
    ax.invert_yaxis()
    ax.set_title('Buoyancy gradients and pressure differences')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Depth (km)')
    ax.legend()
    # figure 3: buoyancy and vertical pressure differences: difined otherwise
    ax = fig.add_subplot(gs[0, 2]) 
    ax.plot(buoyancie_gradients, depths/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(differiential_pressure_v_1, depths/1e3, '--', color=mcolors.CSS4_COLORS['lightcoral'], label='Vertical pressure differences 1 (N/m2)')
    ax.plot(v_zeros, depths/1e3, 'k--')
    ax.invert_yaxis()
    ax.set_title('Buoyancy gradients and pressure differences (defined otherwise)')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Depth (km)')
    # figure 4: field of compensation
    ax = fig.add_subplot(gs[1, 0]) 
    ax.plot(compensation, depths/1e3, 'k')
    ax.invert_yaxis()
    ax.set_title("Field of compensation")
    ax.set_xlim([-10, 10])
    ax.set_xlabel('Compensation')
    # figure 5: buoyancy and vertical pressure differences - with a 500 km limit
    mask = depths < 500e3
    ax = fig.add_subplot(gs[1, 1]) 
    ax.plot(buoyancie_gradients[mask], depths[mask]/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(differiential_pressure[mask], depths[mask]/1e3, 'm--', label='Pressure differences (N/m2)')
    ax.plot(differiential_pressure_v[mask], depths[mask]/1e3, 'r--', label='Vertical pressure differences (N/m2)')
    ax.plot(v_zeros[mask], depths[mask]/1e3, 'k--')
    ax.set_ylim([0, 500]) # set y limit
    ax.invert_yaxis()
    ax.set_title('Buoyancy gradients and pressure differences, depth in [0, 500] km')
    ax.set_xlabel('Pressure (Pa)')
    ax.set_ylabel('Depth (km)')
    ax.legend()
    # figure 6: dynamic pressure
    ax = fig.add_subplot(gs[1, 2]) 
    ax.plot(dynamic_pressure_upper, depths/1e3, 'c--', label='Dynamic P (upper) (N/m2)')
    ax.plot(dynamic_pressure_lower, depths/1e3, 'm--', label='Dynamic P (lower) (N/m2)')
    ax.set_title('Buoyancy gradients and dynamic pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.legend()
    ax.invert_yaxis()
    # figure 7: dynamic pressure differences
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(buoyancie_gradients, depths/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(differiential_dynamic_pressure, depths/1e3, 'm--', label='Dynamic P differences (N/m2)')
    ax.plot(differiential_dynamic_pressure_v, depths/1e3, 'r--', label='Vertical dynamic P differences (N/m2)')
    ax.plot(v_zeros, depths/1e3, 'k--')
    ax.set_title('Buoyancy gradients and differential dynamic pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.legend()
    ax.invert_yaxis() 
    # figure 8: dynamic pressure, in the upper 400 km
    mask = depths < 400e3
    ax = fig.add_subplot(gs[2, 1]) 
    ax.plot(buoyancie_gradients[mask], depths[mask]/1e3, 'b', label='Buoyancy gradients (N/m2)')
    ax.plot(dynamic_pressure_upper[mask], depths[mask]/1e3, 'c--', label='Dynamic P (upper) (N/m2)')
    ax.plot(dynamic_pressure_lower[mask], depths[mask]/1e3, 'm--', label='Dynamic P (lower) (N/m2)')
    ax.plot(differiential_dynamic_pressure[mask], depths[mask]/1e3, 'r--', label='Dynamic P differences (N/m2)')
    ax.plot(v_zeros[mask], depths[mask]/1e3, 'k--')
    ax.set_ylim([0, 400]) # set y limit
    ax.set_title('Buoyancy gradients and dynamic pressure')
    ax.set_xlabel('Pressure (Pa)')
    ax.legend()
    ax.invert_yaxis()
    fig.suptitle('Buoyancy (total %.4e N/m2)' % total_buoyancy)
    fig.tight_layout()
    plt.savefig(fileout)
    print("PlotSlabForces: plot figure", fileout)


def SlabMorphology_dual_mdd(case_dir, vtu_snapshot, **kwargs):
    '''
    Wrapper for using PVTK class to get slab morphology, uses two distinct mdd_dx1 value
    to get both the partial and the final coupling point.
    Inputs:
        case_dir (str): case directory
        vtu_snapshot (int): index of file in vtu outputs
        kwargs:
            project_velocity - whether the velocity is projected to the tangential direction
    '''
    indent = kwargs.get("indent", 0)  # indentation for outputs
    findmdd = kwargs.get("findmdd", False)
    output_ov_ath_profile = kwargs.get("output_ov_ath_profile", False)
    output_path = kwargs.get("output_path", os.path.join(case_dir, "vtk_outputs"))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    findmdd_tolerance = kwargs.get("findmdd_tolerance", 0.05)
    depth_distant_lookup = kwargs.get("depth_distant_lookup", 200e3)
    dip_angle_depth_lookup = kwargs.get("dip_angle_depth_lookup", None)
    dip_angle_depth_lookup_interval = kwargs.get("dip_angle_depth_lookup_interval", 60e3)
    project_velocity = kwargs.get('project_velocity', False)
    find_shallow_trench = kwargs.get("find_shallow_trench", False)
    print("find_shallow_trench: ", find_shallow_trench) # debug
    extract_depths = kwargs.get("extract_depths", None)
    mdd = -1.0 # an initial value
    print("%s%s: Start" % (indent*" ", func_name()))
    output_slab = kwargs.get('output_slab', None)
    # todo_crust
    n_crust = kwargs.get("n_crust", 1)
    if n_crust == 1:
        crust_fields = ['spcrust']
    elif n_crust == 2:
        crust_fields = ['spcrust_up', 'spcrust_low']
    filein = os.path.join(case_dir, "output", "solution", "solution-%05d.pvtu" % vtu_snapshot)
    if not os.path.isfile(filein):
        raise FileExistsError("input file (pvtu) doesn't exist: %s" % filein)
    else:
        print("SlabMorphology_dual_mdd: processing %s" % filein)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # vtk_option_path, _time, step = PrepareVTKOptions(VISIT_OPTIONS, case_dir, 'TwoDSubduction_SlabAnalysis',\
    # vtu_step=vtu_step, include_step_in_filename=True, generate_horiz=True)
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    _time, step = Visit_Options.get_time_and_step(vtu_step)
    geometry = Visit_Options.options['GEOMETRY']
    Ro =  Visit_Options.options['OUTER_RADIUS']
    if geometry == "chunk":
        Xmax = Visit_Options.options['XMAX'] * np.pi / 180.0
    else:
        Xmax = Visit_Options.options['XMAX']
    VtkP = VTKP(geometry=geometry, Ro=Ro, Xmax=Xmax)
    VtkP.ReadFile(filein)
    field_names = ['T', 'density', 'spharz', 'velocity', 'viscosity'] + crust_fields
    VtkP.ConstructPolyData(field_names, include_cell_center=True)
    VtkP.PrepareSlab(crust_fields + ['spharz'], prepare_slab_distant_properties=True, depth_distant_lookup=depth_distant_lookup)
    if find_shallow_trench:
        outputs = VtkP.PrepareSlabShallow(n_crust=n_crust)
        x, y, z = outputs["corrected"]["points"]
        _, _, trench_shallow = cart2sph(x, y, z)
    if findmdd:
        try:
            # mdd depths and horizontal profiles of velocity
            mdd1, query_grid1, query_vs1, extract_profiles_grid, extract_profiles_vs = \
                VtkP.FindMDD(tolerance=findmdd_tolerance, dx1=-Visit_Options.options["INITIAL_SHEAR_ZONE_THICKNESS"], 
                             extract_depths=extract_depths)
            mdd2, query_grid2, query_vs2, _, _ = VtkP.FindMDD(tolerance=findmdd_tolerance, dx1=10e3)
            # velocity profile at mdd1
            header = "# 1: x (m)\n# 2: y (m)\n# 3: velocity_x (m/s)\n# 4: velocity_y (m/s)\n"
            outputs1 = np.concatenate((query_grid1, query_vs1[:, 0:2]), axis=1)
            o_file1 = os.path.join(output_path, "mdd1_profile_%05d.txt" % (vtu_step))
            with open(o_file1, 'w') as fout:
                fout.write(header)  # output header
            with open(o_file1, 'a') as fout:
                np.savetxt(fout, outputs1, fmt="%20.8e")  # output data
            print("%s%s: write file %s" % (indent*" ", func_name(), o_file1))
            # velocity profile at mdd2
            outputs2 = np.concatenate((query_grid2, query_vs2[:, 0:2]), axis=1)
            o_file2 = os.path.join(output_path, "mdd2_profile_%05d.txt" % (vtu_step))
            with open(o_file2, 'w') as fout:
                fout.write(header)  # output header
            with open(o_file2, 'a') as fout:
                np.savetxt(fout, outputs2, fmt="%20.8e")  # output data
            print("%s%s: write file %s" % (indent*" ", func_name(), o_file2))
            # Extra velocity profiles
            if extract_depths is not None:
                for i_depth, depth in enumerate(extract_depths):
                    query_grid_foo = extract_profiles_grid[i_depth]
                    query_vs_foo = extract_profiles_vs[i_depth]
                    outputs2 = np.concatenate((query_grid_foo, query_vs_foo[:, 0:2]), axis=1)
                    o_file2 = os.path.join(output_path, "mdd_extract_profile_%05d_depth_%.2fkm.txt" % (vtu_step, depth/1e3))
                    with open(o_file2, 'w') as fout:
                        fout.write(header)  # output header
                    with open(o_file2, 'a') as fout:
                        np.savetxt(fout, outputs2, fmt="%20.8e")  # output data
                    print("%s%s: write file %s" % (indent*" ", func_name(), o_file2))
                
        except ValueError:
            mdd1 = - 1.0
            mdd2 = - 1.0
    # output slab profile
    if output_slab is not None:
        slab_envelop0, slab_envelop1 = VtkP.ExportSlabEnvelopCoord()
        slab_internal = VtkP.ExportSlabInternal(output_xy=True)
        if output_slab == "vtp":
            o_slab_env0 = os.path.join(case_dir,\
                "vtk_outputs", "slab_env0_%05d.vtp" % (vtu_step)) # envelop 0
            o_slab_env1 = os.path.join(case_dir,\
                "vtk_outputs", "slab_env1_%05d.vtp" % (vtu_step)) # envelop 1
            o_slab_in = os.path.join(case_dir,\
                "vtk_outputs", "slab_internal_%05d.txt" % (vtu_step)) # envelop 1
            ExportPolyDataFromRaw(slab_envelop0[:, 0], slab_envelop0[:, 1], None, None, o_slab_env0) # write the polydata
            # np.savetxt(o_slab_env0, slab_envelop0)
            print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_env0))
            ExportPolyDataFromRaw(slab_envelop1[:, 0], slab_envelop1[:, 1], None, None, o_slab_env1) # write the polydata
            print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_env1))
            np.savetxt(o_slab_in, slab_internal)
            print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_in))
        # todo_o_env
        if output_slab == "txt":
            o_slab_env =  os.path.join(case_dir, \
                "vtk_outputs", "slab_env_%05d.txt" % (vtu_step)) # envelop 1
            slab_env_outputs = np.concatenate([slab_envelop0, slab_envelop1], axis=1) 
            slab_env_output_header = "X0 Y0 X1 Y1"
            np.savetxt(o_slab_env, slab_env_outputs, header=slab_env_output_header)
            print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_env))
            o_slab_in = os.path.join(case_dir,\
                "vtk_outputs", "slab_internal_%05d.txt" % (vtu_step)) # envelop 1
            np.savetxt(o_slab_in, slab_internal)
            print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_in))
    # output a profile distant to the slab
    n_distant_profiel_sample = 100
    v_distant_profile = None; query_viscs = None
    if output_ov_ath_profile:
        header = "# 1: x (m)\n# 2: y (m)\n# 3: depth (m)\n# 4: velocity_h (m/s)\n# 5: velocity_r (m/s)\n# 6: viscosity (pa*s)\n# 7: fixing (by 0.1 deg)\n"
        coords, depths, v_distant_profile, query_viscs, value_fixing = VtkP.ExportOvAthenProfile(depth_distant_lookup, n_sample=n_distant_profiel_sample)
        outputs = np.concatenate((coords, depths.reshape((-1, 1)), v_distant_profile,\
            query_viscs.reshape((-1, 1)), value_fixing.reshape((-1, 1))), axis=1)
        o_file = os.path.join(output_path, "ov_ath_profile_%05d.txt" % (vtu_step))
        with open(o_file, 'w') as fout:
            fout.write(header)  # output header
        with open(o_file, 'a') as fout:
            np.savetxt(fout, outputs, fmt="%20.8e")  # output data
        print("%s%s: write file %s" % (indent*" ", func_name(), o_file))
    # process trench, slab depth, dip angle
    trench, slab_depth, dip_100 = VtkP.ExportSlabInfo()
    if project_velocity:
        vsp_magnitude, vov_magnitude = VtkP.ExportVelocity(project_velocity=True)
    else:
        vsp, vov = VtkP.ExportVelocity()
        vsp_magnitude = np.linalg.norm(vsp, 2)
        vov_magnitude = np.linalg.norm(vov, 2)
    # generate outputs
    outputs = "%-12s%-12d%-14.4e%-14.4e%-14.4e%-14.4e%-14.4e%-14.4e"\
    % (vtu_step, step, _time, trench, slab_depth, dip_100, vsp_magnitude, vov_magnitude)
    if findmdd:
        outputs += "%-14.4e %-14.4e" % (mdd1, mdd2)
    if output_ov_ath_profile:
        outputs += "%-14.4e %-14.4e" % (v_distant_profile[n_distant_profiel_sample-1, 0], query_viscs[n_distant_profiel_sample-1])
    if dip_angle_depth_lookup is not None:
        outputs += "%-14.4e" % (VtkP.GetDipAtDepth(dip_angle_depth_lookup, dip_angle_depth_lookup_interval))
    if find_shallow_trench:
        outputs += "%-14.4e" % (trench_shallow)
    
    outputs += "\n"
    print("%s%s" % (indent*" ", outputs)) # debug
    return vtu_step, outputs


def SlabAnalysis(case_dir, vtu_snapshot, o_file, **kwargs):
    '''
    Perform analysis on the slab, this would output a file including the
    buoyancy forces of the slab and the pressures on the slab surface.
    Inputs:
        kwargs(dict):
            output_slab - output slab file
            use_dT - use temperature difference as the criteria for the slab surface.
    '''
    # read in parameters
    indent = kwargs.get("indent", 0)  # indentation for outputs
    print("%s%s: Start" % (indent*" ", func_name()))
    output_slab = kwargs.get('output_slab', False)
    output_poly_data = kwargs.get('output_poly_data', True)
    use_dT = kwargs.get('use_dT', False)
    dT = kwargs.get('dT', -100.0)
    slab_envelop_interval = kwargs.get("slab_envelop_interval", 5e3)
    ha_file = os.path.join(case_dir, "output", "depth_average.txt")
    assert(os.path.isfile(ha_file))
    output_path = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    # look for the input file named as "output/solution/solution-%05d.pvtu"
    filein = os.path.join(case_dir, "output", "solution",\
         "solution-%05d.pvtu" % (vtu_snapshot))
    assert(os.path.isfile(filein))
    # get parameters
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    geometry = Visit_Options.options['GEOMETRY']
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    _time, step = Visit_Options.get_time_and_step(vtu_step)
    # initiate class
    VtkP = VTKP(ha_file=ha_file, time=_time, slab_envelop_interval=slab_envelop_interval)
    VtkP.ReadFile(filein)
    # fields to load
    field_names = ['T', 'p', 'density', 'spcrust', 'spharz']
    has_dynamic_pressure = int(Visit_Options.options['HAS_DYNAMIC_PRESSURE']) 
    if has_dynamic_pressure == 1:
        field_names += ['nonadiabatic_pressure']
    VtkP.ConstructPolyData(field_names, include_cell_center=True, construct_Tdiff=True)
    # include a v_profile
    r0_range = [6371e3 - 2890e3, 6371e3]
    Ro = 6371e3
    x1 = 0.01 
    n = 100
    v_profile = VtkP.VerticalProfile2D(r0_range, x1, n)
    # output poly data, debug
    if output_poly_data:
        file_out = os.path.join(output_path, "processed-%05d.vtp" % vtu_snapshot)
        ExportPolyData(VtkP.i_poly_data, file_out)
        file_out_1 = os.path.join(output_path, "processed_center-%05d.vtp" % vtu_snapshot)
        ExportPolyData(VtkP.c_poly_data, file_out_1)
    # slab envelop
    if use_dT:
        VtkP.PrepareSlabByDT(slab_threshold=dT)  # slab: differential temperature
    else:
        VtkP.PrepareSlab(['spcrust', 'spharz'])  # slab: composition
    # output slab profile
    if output_slab:
        slab_envelop0, slab_envelop1 = VtkP.ExportSlabEnvelopCoord()
        slab_internal = VtkP.ExportSlabInternal(output_xy=True)
        o_slab_env0 = os.path.join(case_dir,\
            "vtk_outputs", "slab_env0_%05d.vtp" % (vtu_step)) # envelop 0
        o_slab_env1 = os.path.join(case_dir,\
            "vtk_outputs", "slab_env1_%05d.vtp" % (vtu_step)) # envelop 1
        o_slab_in = os.path.join(case_dir,\
            "vtk_outputs", "slab_internal_%05d.txt" % (vtu_step)) # envelop 1
        ExportPolyDataFromRaw(slab_envelop0[:, 0], slab_envelop0[:, 1], None, None, o_slab_env0) # write the polydata
        ExportPolyDataFromRaw(slab_envelop1[:, 0], slab_envelop1[:, 1], None, None, o_slab_env1) # write the polydata
        np.savetxt(o_slab_in, slab_internal)
        print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_in))
    # buoyancy
    total_buoyancy, b_profile = VtkP.SlabBuoyancy(v_profile, 5e3)  # test 5e3, 50e3
    depths_o = b_profile[:, 0]  # use these depths to generate outputs
    buoyancies = b_profile[:, 1]
    buoyancy_gradients = buoyancies / (depths_o[1] - depths_o[0])  # gradient of buoyancy
    # pressure 
    slab_envelop0, slab_envelop1 = VtkP.ExportSlabEnvelopCoord()  # raw data on the envelop and output
    fileout = os.path.join(output_path, 'slab_pressures0_%05d.txt' % (vtu_step))
    depths0, thetas0, ps0= SlabPressures(VtkP, slab_envelop0, fileout=fileout, indent=4, has_dynamic_pressure=has_dynamic_pressure)  # depth, dip angle and pressure
    fileout = os.path.join(output_path, 'slab_pressures1_%05d.txt' % (vtu_step))
    depths1, thetas1, ps1 = SlabPressures(VtkP, slab_envelop1, fileout=fileout, indent=4, has_dynamic_pressure=has_dynamic_pressure)
    ps0_o = np.interp(depths_o, depths0, ps0[:, 0])  # interpolation to uniform interval
    thetas0_o = np.interp(depths_o, depths0, thetas0)
    ps0_d_o = np.interp(depths_o, depths0, ps0[:, 3])  # dynamic pressure
    ps1_o = np.interp(depths_o, depths1, ps1[:, 0])  # interpolation to uniform interval
    thetas1_o = np.interp(depths_o, depths1, thetas1)
    ps1_d_o = np.interp(depths_o, depths1, ps1[:, 3])  # dynamic pressure
    ps_o = ps0_o - ps1_o  # this has to be minus: sides of pressure are differnent on top or below.
    ps_d_o = ps0_d_o - ps1_d_o  # dynamic pressure difference
    # pvs_o = ps0_o * np.cos(thetas0_o)  - ps1_o * np.cos(thetas0_o)
    # pvs_o1 = ps0_o * np.cos(thetas0_o)  - ps1_o * np.cos(thetas1_o)  # here we cannot multiply thetas1_o, otherwise it will be zagged
    pvs_o = ps0_o / np.tan(thetas0_o)  - ps1_o / np.tan(thetas0_o)   # Right now, I am convinced this is the right way.
    pvs_d_o = ps0_d_o / np.tan(thetas0_o)  - ps1_d_o / np.tan(thetas0_o)   # vertical component of dynamic pressure differences
    pvs_o1 = ps0_o / np.tan(thetas0_o)  - ps1_o / np.tan(thetas1_o)  # here we cannot multiply thetas1_o, otherwise it will be zagged
    compensation = pvs_o / (-buoyancy_gradients)
    outputs = np.concatenate((b_profile, buoyancy_gradients.reshape((-1, 1)),\
    ps0_o.reshape((-1, 1)), ps1_o.reshape((-1, 1)),\
    ps_o.reshape((-1, 1)), pvs_o.reshape((-1, 1)), pvs_o1.reshape((-1, 1)),\
    compensation.reshape((-1, 1)), ps0_d_o.reshape((-1, 1)), ps1_d_o.reshape((-1, 1)),\
    ps_d_o.reshape((-1, 1)), pvs_d_o.reshape((-1, 1))), axis=1)
    # output data
    # all this data are outputed just to toy with the plot of buoyancy and pressure
    header = "# 1: depth (m)\n# 2: buoyancy (N/m)\n\
# 3: buoyancy gradient (Pa)\n# 4: pressure upper (Pa) \n# 5: pressure lower (Pa)\n\
# 6: differiential pressure (Pa)\n# 7: vertical differiential pressure\n\
# 8: vertical differiential pressure 1\n# 9: compensation\n\
# 10: dynamic pressure upper (Pa)\n# 11: dynamic pressure lower (Pa)\n\
# 12: differential dynamic pressure (Pa)\n# 13: vertical differential dynamic pressure (Pa)\n"
    with open(o_file, 'w') as fout:
        fout.write(header)  # output header
    with open(o_file, 'a') as fout:
        np.savetxt(fout, outputs, fmt="%20.8e")  # output data
    print("%s: write file %s" % (func_name(), o_file))


def SlabPressures(VtkP, slab_envelop, **kwargs):
    '''
    extract slab pressures, interpolated results onto regular grid is outputed,
    original data is returned
    Inputs:
        VtkP: VTKP class
        slab_envelop: slab envelop coordinates (x and y)
    returns:
        depths: depth of points
        thetas: dip angles
        ps: pressures
    '''
    Ro = 6371e3
    fileout = kwargs.get('fileout', None)
    indent = kwargs.get('indent', 0)
    has_dynamic_pressure = kwargs.get('has_dynamic_pressure', 0)
    rs_n = 5 # resample interval
    ip_interval = 1e3  # interval for interpolation
    # resample the original envelop dataset
    n_point = slab_envelop.shape[0]
    rs_idx = range(0, n_point, rs_n)
    slab_envelop_rs = slab_envelop[np.ix_(rs_idx, [0, 1])] # use np.ix to resample, check out numpy documentation
    slab_env_polydata = InterpolateGrid(VtkP.i_poly_data, slab_envelop_rs, quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
    temp_vtk_array = slab_env_polydata.GetPointData().GetArray('p')
    env_ps  = vtk_to_numpy(temp_vtk_array)
    # dynamic pressure is outputed -> read in
    # dynamic pressure is not outputed -> use pressure - static_pressure as an estimation
    if has_dynamic_pressure == 1:
        temp_vtk_array = slab_env_polydata.GetPointData().GetArray('nonadiabatic_pressure')
        env_dps  = vtk_to_numpy(temp_vtk_array)
        print("Read in the dynamic pressures")
    else:
        env_dps = None
    # import data onto selected points
    depths = np.zeros(slab_envelop_rs.shape[0]) # data on envelop0
    ps = np.zeros((slab_envelop_rs.shape[0], 4)) # pressure, horizontal & vertical components, dynamic pressure
    thetas = np.zeros((slab_envelop_rs.shape[0], 1))
    is_first = True
    for i in range(0, slab_envelop_rs.shape[0]):
        x = slab_envelop_rs[i, 0]
        y = slab_envelop_rs[i, 1]
        theta_xy = np.arctan2(y, x)
        r = (x*x + y*y)**0.5
        depth = Ro - r  # depth of this point
        p = env_ps[i]  # pressure of this point
        p_static = VtkP.StaticPressure([r, VtkP.Ro], theta_xy, 2000)
        if env_dps is not None:
            p_d = env_dps[i]
        else:
            p_d = p - p_static # dynamic pressure, read in or compute
        depths[i] = depth
        d1 = 0.0  # coordinate differences
        d2 = 0.0
        # here we first get a dip angle
        if is_first:
            xnext = slab_envelop_rs[i+1, 0]  # coordinates of this and the last point
            ynext = slab_envelop_rs[i+1, 1]
            theta = get_dip(x, y, xnext, ynext, VtkP.geometry)
            is_first = False
        else: 
            xlast = slab_envelop_rs[i-1, 0]  # coordinates of this and the last point
            ylast = slab_envelop_rs[i-1, 1]
            theta = get_dip(xlast, ylast, x, y, VtkP.geometry) 
        thetas[i, 0] = theta
        # then we project the pressure into vertical and horizontal
        p_v = p * np.cos(theta)
        p_h = p * np.sin(theta)
        ps[i, 0] = p
        ps[i, 1] = p_h
        ps[i, 2] = p_v
        ps[i, 3] = p_d
    temp = np.concatenate((slab_envelop_rs, thetas), axis=1)
    data_env0 = np.concatenate((temp, ps), axis=1)  # assemble all the data
    c_out = data_env0.shape[1]
    # interpolate data to regular grid & prepare outputs
    start = np.ceil(depths[0]/ip_interval) * ip_interval
    end = np.floor(depths[-1]/ip_interval) * ip_interval
    n_out = int((end-start) / ip_interval)
    data_env0_out = np.zeros((n_out, c_out+1))
    depths_out = np.arange(start, end, ip_interval)
    data_env0_out[:, 0] = depths_out
    for j in range(c_out):
        data_env0_out[:, j+1] = np.interp(depths_out, depths, data_env0[:, j]) # interpolation
        header = "# 1: depth (m)\n# 2: x (m)\n# 3: y (m)\n# 4: theta_v \n# 5: p (Pa)\n# 6: p_h (Pa) \n# 7: p_v (Pa)\n"
    with open(fileout, 'w') as fout:
        fout.write(header)
    with open(fileout, 'a') as fout: 
        np.savetxt(fout, data_env0_out, fmt='%.4e\t')
    print("%s%s: write output %s" % (' '*indent, func_name(), fileout))
    return depths, thetas[:, 0], ps  # return depths and pressures

class mohoExtractionIndexError(Exception):
    pass

def SlabTemperature(case_dir, vtu_snapshot, ofile=None, **kwargs):
    '''
    Perform analysis on the slab, this would output a file including the
    buoyancy forces of the slab and the pressures on the slab surface.
    Inputs:
        case_dir(str) - directory of the case
        vtu_snapshot - snapshots (step plus the level of the initial adaptive refinements)
        ofile - output file of the slab temperature profile. If this is None, then no outputs are generated
        kwargs(dict):
            output_slab - output slab file
            use_dT - use temperature difference as the criteria for the slab surface.
    '''
    # parameters
    debug = kwargs.get('debug', False)
    indent = kwargs.get("indent", 0)  # indentation for outputs
    print("%s%s: Start" % (indent*" ", func_name()))
    output_slab = kwargs.get('output_slab', False)
    output_poly_data = kwargs.get('output_poly_data', True)
    slab_envelop_interval = kwargs.get("slab_envelop_interval", 5e3)
    max_depth = kwargs.get("max_depth", 660e3)
    fix_shallow = kwargs.get("fix_shallow", False)
    ofile_surface = kwargs.get("ofile_surface", None)
    ofile_moho = kwargs.get("ofile_moho", None)
    n_crust = kwargs.get("n_crust", 1)
    output_path = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    filein = os.path.join(case_dir, "output", "solution",\
         "solution-%05d.pvtu" % (vtu_snapshot))
    assert(os.path.isfile(filein))
    # offsets
    offsets = kwargs.get("offsets", [])
    assert(isinstance(offsets, (list)))

    # get parameters
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    geometry = Visit_Options.options['GEOMETRY']
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    _time, step = Visit_Options.get_time_and_step(vtu_step)
    # initiate class
    VtkP = VTKP(time=_time, slab_envelop_interval=slab_envelop_interval, slab_shallow_cutoff=25e3)
    VtkP.ReadFile(filein)
    # fields to load
    # todo_field
    if n_crust == 1:
        crust_fields = ['spcrust']
    elif n_crust == 2:
        crust_fields = ['spcrust_up', 'spcrust_low']
    else:
        return NotImplementedError()
    field_names = ['T', 'p', 'density', 'spharz'] + crust_fields
    has_dynamic_pressure = int(Visit_Options.options['HAS_DYNAMIC_PRESSURE']) 
    if has_dynamic_pressure == 1:
        field_names += ['nonadiabatic_pressure']
    VtkP.ConstructPolyData(field_names, include_cell_center=True, construct_Tdiff=False)
   
    # include a v_profile
    r0_range = [6371e3 - 2890e3, 6371e3]
    Ro = 6371e3
    x1 = 0.01 
    n = 100
    v_profile = VtkP.VerticalProfile2D(r0_range, x1, n)
    
    # output poly data, debug
    if output_poly_data:
        file_out = os.path.join(output_path, "processed-%05d.vtp" % vtu_snapshot)
        ExportPolyData(VtkP.i_poly_data, file_out)
        file_out_1 = os.path.join(output_path, "processed_center-%05d.vtp" % vtu_snapshot)
        ExportPolyData(VtkP.c_poly_data, file_out_1)
    
    # slab envelop
    VtkP.PrepareSlab(crust_fields + ['spharz'], prepare_moho=crust_fields)  # slab: composition
    slab_envelop0, slab_envelop1 = VtkP.ExportSlabEnvelopCoord()
    moho_envelop = VtkP.ExportSlabmohoCoord()
    if output_slab:
        slab_internal = VtkP.ExportSlabInternal(output_xy=True)
        o_slab_env0 = os.path.join(case_dir,\
            "vtk_outputs", "slab_env0_%05d.vtp" % (vtu_step)) # envelop 0
        o_slab_env1 = os.path.join(case_dir,\
            "vtk_outputs", "slab_env1_%05d.vtp" % (vtu_step)) # envelop 1
        o_moho_env = os.path.join(case_dir,\
            "vtk_outputs", "moho_env_%05d.vtp" % (vtu_step)) # envelop 0
        o_slab_in = os.path.join(case_dir,\
            "vtk_outputs", "slab_internal_%05d.txt" % (vtu_step)) # envelop 1
        ExportPolyDataFromRaw(slab_envelop0[:, 0], slab_envelop0[:, 1], None, None, o_slab_env0) # write the polydata
        ExportPolyDataFromRaw(slab_envelop1[:, 0], slab_envelop1[:, 1], None, None, o_slab_env1) # write the polydata
        # export the envelop of the core-mantle boundary
        ExportPolyDataFromRaw(moho_envelop[:, 0], moho_envelop[:, 1], None, None, o_moho_env) # write the polydata
        np.savetxt(o_slab_in, slab_internal)
        print("%s%s: write file %s" % (indent*" ", func_name(), o_slab_in))
    rs_n = 5 # resample interval
    ip_interval = 1e3  # interval for interpolation

    # resample the origin slab surface
    n_point = slab_envelop1.shape[0]
    rs_idx = range(0, n_point, rs_n)
    slab_envelop_rs_raw = slab_envelop1[np.ix_(rs_idx, [0, 1])]  # for slab surface
    
    if fix_shallow:
        # append the shallow trench point
        outputs_shallow = VtkP.PrepareSlabShallow(n_crust=n_crust)
        slab_envelop_rs_raw = np.vstack((outputs_shallow["corrected"]["points"][0:2], slab_envelop_rs_raw)) 
    else:
        outputs_shallow = None
    
    depths_raw = Ro - (slab_envelop_rs_raw[:, 0]**2.0 + slab_envelop_rs_raw[:, 1]**2.0)**0.5
    id_max = np.where(depths_raw < max_depth)[0][-1]
    
    depths = depths_raw[0: id_max+1]
    slab_envelop_rs = slab_envelop_rs_raw[0: id_max+1, :]

    # resample the original moho
    if debug:
        print("moho_envelop: ")  # screen outputs
        print(moho_envelop)
    try:
        moho_envelop_rs_raw = moho_envelop[np.ix_(rs_idx, [0, 1])]
        if debug:
            print("moho_envelop_rs_raw: ")  # screen outputs
            print(moho_envelop_rs_raw)
    except IndexError as e:
        rs_idx_last = rs_idx[-1]
        moho_envelop_rs_raw_length = moho_envelop.shape[0]
        raise Exception("the last index to extract is %d, while the shape of the moho_envelop is %d"\
        % (rs_idx_last, moho_envelop_rs_raw_length)) from e
    
    if fix_shallow:
        # append the shallow trench point
        r, theta, phi = cart2sph(*outputs_shallow["corrected"]["points"])
        r1 = r - Visit_Options.options["INITIAL_SHEAR_ZONE_THICKNESS"]
        moho_shallow = np.array([r1 * np.sin(theta) * np.cos(phi), r1 * np.sin(theta) * np.sin(phi), 0])
        moho_envelop_rs_raw = np.vstack((moho_shallow[0:2], moho_envelop_rs_raw))
    
    
    depths_moho_raw = Ro - (moho_envelop_rs_raw[:, 0]**2.0 + moho_envelop_rs_raw[:, 1]**2.0)**0.5
    
    depths_moho = depths_moho_raw[0: id_max+1] # match the array for the surface
    moho_envelop_rs = moho_envelop_rs_raw[0: id_max+1, :]
    
    id_valid = np.where(depths_moho > 0.0)[0][-1] # reason: max depth could be shallower than the slab surface
    
    # interpolate the curve
    # start = np.ceil(depths[0]/ip_interval) * ip_interval
    start = np.ceil(depths[0]/ip_interval) * ip_interval
    end = np.floor(depths[-1]/ip_interval) * ip_interval
    n_out = int((end-start) / ip_interval)
    depths_out = np.arange(start, end, ip_interval)

    # interpolate T for surface
    slab_Xs = interp1d(depths, slab_envelop_rs[:, 0], kind='cubic')(depths_out)
    slab_Ys = interp1d(depths, slab_envelop_rs[:, 1], kind='cubic')(depths_out)
    
    slab_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((slab_Xs, slab_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
    env_Ttops  = vtk_to_numpy(slab_env_polydata.GetPointData().GetArray('T'))

    # interpolate T for moho
    mask_moho = ((depths_out > depths_moho[0]) & (depths_out < depths_moho[id_valid]))
    moho_Xs = np.zeros(depths_out.shape)
    moho_Ys = np.zeros(depths_out.shape)
    moho_Xs[mask_moho] = interp1d(depths_moho[0: id_valid+1], moho_envelop_rs[0: id_valid+1, 0], kind='cubic')(depths_out[mask_moho])
    moho_Ys[mask_moho] = interp1d(depths_moho[0: id_valid+1], moho_envelop_rs[0: id_valid+1, 1], kind='cubic')(depths_out[mask_moho])

    moho_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((moho_Xs, moho_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
    env_Tbots = vtk_to_numpy(moho_env_polydata.GetPointData().GetArray('T'))
    
    mask = (env_Tbots < 1.0) # fix the non-sense values
    env_Tbots[mask] = -np.finfo(np.float32).max
    if debug:
        print("env_Tbots")  # screen outputs
        print(env_Tbots)
    
    offset_Xs_array=[]; offset_Ys_array=[]; env_Toffsets_array = []
    # interpolate T for offest profiles
    for i, offset in enumerate(offsets):
        offset_Xs, offset_Ys = offset_profile(slab_Xs, slab_Ys, offset)
        offset_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((offset_Xs, offset_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
        env_Toffsets = vtk_to_numpy(offset_env_polydata.GetPointData().GetArray('T'))
    
        mask = (env_Toffsets < 1.0) # fix the non-sense values
        env_Toffsets[mask] = -np.finfo(np.float32).max

        offset_Xs_array.append(offset_Xs)
        offset_Ys_array.append(offset_Ys)
        env_Toffsets_array.append(env_Toffsets)

    # output 
    if ofile is not None:
        # write output if a valid path is given
        data_env0 = np.zeros((depths_out.size, 7+len(offsets)*3)) # output: x, y, Tbot, Ttop
        data_env0[:, 0] = depths_out
        # coordinates
        data_env0[:, 1] = slab_Xs
        data_env0[:, 2] = slab_Ys
        data_env0[:, 3] = moho_Xs
        data_env0[:, 4] = moho_Ys
        idx = 4
        idx1 = idx
        for i in range(len(offsets)):
            data_env0[:, idx+2*i+1] = offset_Xs_array[i]
            data_env0[:, idx+2*i+2] = offset_Ys_array[i]
        idx += len(offsets)*2
        idx2 = idx
        # temperatures
        data_env0[:, idx+1] = env_Tbots
        data_env0[:, idx+2] = env_Ttops
        idx += 2
        idx3 = idx
        for i in range(len(offsets)):
            data_env0[:, idx+i+1] = env_Toffsets_array[i]
        idx += len(offsets)

        # interpolate data to regular grid & prepare outputs
        # add additional headers if offset profiles are required
        header = "# 1: depth (m)\n# 2: x (m)\n# 3: y (m)\n# 4: x bot (m)\n# 5: y bot (m)\n"
        for i in range(len(offsets)):
            header += "# %d: x offset %d (m)\n# %d: y offset %d (m)\n" % (idx1+2*i+2,i,idx1+2*i+3,i)
        header += "# %d: Tbot (K)\n# %d: Ttop (K)\n" % (idx2+2, idx2+3)
        for i in range(len(offsets)):
            header += "# %d: Toffset %d (K)\n" % (idx3+i+2, i)
        with open(ofile, 'w') as fout:
            fout.write(header)
        with open(ofile, 'a') as fout: 
            np.savetxt(fout, data_env0, fmt='%.4e\t')
        print("%s%s: write output %s" % (' '*indent, func_name(), ofile))
    if ofile_surface is not None:
        # write output of surface and moho profile
        ExportPolyDataFromRaw(slab_Xs, slab_Ys, None, None, ofile_surface) # write the polydata
    if ofile_moho is not None:
        ExportPolyDataFromRaw(moho_Xs, moho_Ys, None, None, ofile_moho) # write the polydata
        
    return depths, env_Ttops, env_Tbots  # return depths and pressures


def SlabTemperature1(case_dir, vtu_snapshot, **kwargs):
    '''
    a wrapper for the SlabTemperature function
    '''
    output_path = os.path.join(case_dir, "vtk_outputs", "temperature")
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    geometry = Visit_Options.options['GEOMETRY']
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))

    if not os.path.isdir(os.path.dirname(output_path)):
        os.mkdir(os.path.dirname(output_path))
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    ofile = os.path.join(output_path, "slab_temperature_%05d.txt" % (vtu_step))
    ofile_surface = os.path.join(output_path, "slab_surface_%05d.vtp" % (vtu_step))
    ofile_moho = os.path.join(output_path, "slab_moho_%05d.vtp" % (vtu_step))
    
    kwargs["ofile_surface"] = ofile_surface
    kwargs["ofile_moho"] = ofile_moho
    kwargs['n_crust'] = Visit_Options.options["N_CRUST"]

    try: 
        SlabTemperature(case_dir, vtu_snapshot, ofile=ofile, **kwargs)
    except ValueError:
        warnings.warn("Generation of file for vtu_snapshot %d" % vtu_snapshot)


def WedgeT(case_dir, vtu_snapshot, **kwargs):
    '''
    export mantle temperature
    '''
    output_path = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    filein = os.path.join(case_dir, "output", "solution",\
         "solution-%05d.pvtu" % (vtu_snapshot))
    assert(os.path.isfile(filein))
    VtkP = VTKP()
    VtkP.ReadFile(filein)
    field_names = ['T', 'density', 'spcrust', 'spharz']
    VtkP.ConstructPolyData(field_names, include_cell_center=True)
    VtkP.PrepareSlab(['spcrust', 'spharz'])
    # test 1 output slab grid & envelop
    fileout = os.path.join(output_path, 'wedge_T100_%05d.txt' % (vtu_snapshot))
    VtkP.ExportWedgeT(fileout=fileout)
    assert(os.path.isfile(fileout))

def TrenchT(case_dir, vtu_snapshot, **kwargs):
    '''
    Export the trench temperature profiles
    '''
    filein = os.path.join(case_dir, "output", "solution",\
         "solution-%05d.pvtu" % (vtu_snapshot))
    output_path = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    fileout =  os.path.join(output_path, 'trench_T_%05d.txt' % (vtu_step))
    assert(os.path.isfile(filein))
    VtkP = VTKP()
    VtkP.ReadFile(filein)
    field_names = ['T', 'density', 'spcrust', 'spharz']
    VtkP.ConstructPolyData(field_names, include_cell_center=True)
    VtkP.PrepareSlab(['spcrust', 'spharz'])
    VtkP.ExportTrenchT(fileout=fileout)
    assert(os.path.isfile(fileout))


def ShearZoneGeometry(case_dir, vtu_snapshot, **kwargs):
    indent = kwargs.get("indent", 0)  # indentation for outputs
    assert(os.path.isdir(case_dir))
    # fix the output directory
    vtk_o_dir = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(vtk_o_dir):
        os.mkdir(vtk_o_dir)
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img_o_dir = os.path.join(img_dir, "shear_zone")
    if not os.path.isdir(img_o_dir):
        os.mkdir(img_o_dir)
    filein = os.path.join(case_dir, "output", "solution", "solution-%05d.pvtu" % vtu_snapshot)
    if not os.path.isfile(filein):
        raise FileExistsError("input file (pvtu) doesn't exist: %s" % filein)
    else:
        print("%sSlabMorphology: processing %s" % (indent*" ", filein))
    # prepare the slab
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    _time, step = Visit_Options.get_time_and_step(vtu_step)
    geometry = Visit_Options.options['GEOMETRY']
    Ro =  Visit_Options.options['OUTER_RADIUS']
    Xmax = Visit_Options.options['XMAX'] * np.pi / 180.0
    Dsz =  Visit_Options.options['INITIAL_SHEAR_ZONE_THICKNESS']
    VtkP = VTKP(geometry=geometry, Ro=Ro, Xmax=Xmax)
    VtkP.ReadFile(filein)
    field_names = ['T', 'density', 'spcrust', 'spharz', 'velocity']
    VtkP.ConstructPolyData(field_names, include_cell_center=True)
    VtkP.PrepareSlab(['spcrust', 'spharz'])
    # call the functions for the shear zone
    fileout = os.path.join(vtk_o_dir, "sz_%05d.txt" % vtu_step)
    VtkP.PrepareSZ(fileout, Dsz=Dsz)
    assert(os.path.isfile(fileout))  # assert file generation
    # plot
    fig_path = os.path.join(img_o_dir, "sz_thickness_%05d.png" % vtu_step) 
    fig, ax = plt.subplots()
    MorphPlotter = SLABPLOT("plot_slab")
    MorphPlotter.PlotShearZoneThickness(case_dir, axis=ax, filein=fileout, label='shear zone thickness')
    ax.legend()
    fig.savefig(fig_path)
    assert(os.path.isfile(fig_path))  # assert figure generation
    print("%s%s: figure generated %s" % (indent*" ", func_name(), fig_path))


def PlotSlabTemperature(case_dir, vtu_snapshot, **kwargs):
    '''
    Process slab envelops and plot the slab temperature
    '''
    indent = kwargs.get("indent", 0)  # indentation for outputs
    assert(os.path.isdir(case_dir))
    # read some options
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
    # fix the output directory
    vtk_o_dir = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(vtk_o_dir):
        os.mkdir(vtk_o_dir)
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img_o_dir = os.path.join(img_dir, "slab_temperature")
    if not os.path.isdir(img_o_dir):
        os.mkdir(img_o_dir)
    filein = os.path.join(case_dir, "output", "solution", "solution-%05d.pvtu" % vtu_snapshot)
    if not os.path.isfile(filein):
        raise FileExistsError("input file (pvtu) doesn't exist: %s" % filein)
    else:
        print("%s%s: processing %s" % (indent*" ",func_name(), filein))
    o_file = os.path.join(vtk_o_dir, "slab_temperature")
    if os.path.isfile(o_file):
        os.remove(o_file)
    assert(os.path.isdir(case_dir))
    _, _, _ = SlabTemperature(case_dir, vtu_snapshot, o_file, output_slab=True)
    assert(os.path.isfile(o_file))  # assert the outputs of slab and moho envelops
    # plot
    fig_path = os.path.join(img_o_dir, "slab_temperature_%05d.png" % vtu_step) 
    fig, ax = plt.subplots()
    MorphPlotter = SLABPLOT("plot_slab")
    MorphPlotter.PlotSlabT(case_dir, axis=ax, filein=o_file, label='slab temperature', xlims=[273.0, 1273.0], ylims=[25e3, 250e3])
    ax.legend()
    ax.invert_yaxis()
    fig.savefig(fig_path)
    # assert figure generation and screen outputs
    assert(os.path.isfile(fig_path))  # assert figure generation
    print("%s%s: figure generated %s" % (indent*" ", func_name(), fig_path))


def PlotSlabTemperatureCase(case_dir, **kwargs):
    '''
    Plot the slab temperature for the case
    '''
    indent = kwargs.get("indent", 0)  # indentation for outputs
    time_range = kwargs.get("time_range", None)
    plot_eclogite = kwargs.get("plot_eclogite", False)
    assert(os.path.isdir(case_dir))
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img_o_dir = os.path.join(img_dir, "slab_temperature")
    if not os.path.isdir(img_o_dir):
        os.mkdir(img_o_dir)
    # plot
    fig, ax = plt.subplots()
    MorphPlotter = SLABPLOT("plot_slab")
    # todo_eclogite
    if plot_eclogite:
        MorphPlotter.PlotEclogite(axis=ax)
    MorphPlotter.PlotSlabTCase(case_dir, axis=ax, label='slab temperature', xlims=[273.0, 1673.0], ylims=[25e3, 250e3], time_range=time_range)
    ax.legend()
    ax.invert_yaxis()
    if time_range is None:
        fig_path = os.path.join(img_o_dir, "slab_temperature.png") 
    else:
        fig_path = os.path.join(img_o_dir, "slab_temperature_%.4eMa_%.4eMa.png" % (time_range[0]/1e6, time_range[1]/1e6)) 
    fig.savefig(fig_path)
    # assert figure generation and screen outputs
    assert(os.path.isfile(fig_path))  # assert figure generation
    print("%s%s: figure generated %s" % (indent*" ", func_name(), fig_path))


def PlotWedgeTCase(case_dir, **kwargs):
    '''
    Plot the figure of the mantle wedge temperature
    kwargs:
        time_interval - the interval of time between two steps
    '''
    time_interval = kwargs.get("time_interval")
    ofile = os.path.join(case_dir, 'img', "wedge_T_100.png")
    SlabPlot = SLABPLOT('wedge_T')
    fig, ax = plt.subplots(figsize=(10, 4)) 
    ax, h = SlabPlot.PlotTWedge(case_dir, time_interval=time_interval, axis=ax)
    fig.colorbar(h, ax=ax, label='T (K)') 
    fig.savefig(ofile)
    assert(os.path.isfile(ofile))
    print("%s: output figure %s" % (func_name(), ofile))


####
# Case-wise functions
####
def SlabTemperatureCase(case_dir, **kwargs):
    '''
    run vtk and get outputs for every snapshots
    Inputs:
        kwargs:
            time_interval: the interval between two processing steps
    '''
    # get all available snapshots
    # the interval is choosen so there is no high frequency noises
    time_interval_for_slab_morphology = kwargs.get("time_interval", 0.5e6)
    run_parallel = kwargs.get("run_parallel", True)

    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
    available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval_for_slab_morphology)
    print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
    # get where previous session ends
    vtk_output_dir = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(vtk_output_dir):
        os.mkdir(vtk_output_dir)

    # todo_T
    # process the slab temperature
    options = {}
    options["if_rewrite"] = True
    options["assemble"] = False
    options["output_poly_data"] = False
    options["fix_shallow"] = True

    # Method 1: run in serial
    # Method 2: run in parallel
    if not run_parallel:
        for pvtu_snapshot in available_pvtu_snapshots:
            SlabTemperature1(case_dir, pvtu_snapshot, **options)
    else:
        ParallelWrapper = PARALLEL_WRAPPER_FOR_VTK('slab_temperature', SlabTemperature1, **options)
        ParallelWrapper.configure(case_dir)  # assign case directory
        # Remove previous file
        print("%s: Delete old slab_temperature.txt file." % func_name())
        ParallelWrapper.delete_temp_files(available_pvtu_snapshots)  # delete intermediate file if rewrite
        num_cores = multiprocessing.cpu_count()
        # loop for all the steps to plot
        Parallel(n_jobs=num_cores)(delayed(ParallelWrapper)(pvtu_snapshot)\
        for pvtu_snapshot in available_pvtu_snapshots)  # first run in parallel and get stepwise output
        ParallelWrapper.clear()

def SlabMorphologyCase(case_dir, **kwargs):
    '''
    run vtk and get outputs for every snapshots
    Inputs:
        kwargs:
            rewrite: if rewrite previous results
            project_velocity - whether the velocity is projected to the tangential direction
            file_tag - apply a tag to file name, default is false
    '''
    # todo_o_env
    findmdd = kwargs.get('findmdd', False)
    findmdd_tolerance = kwargs.get('findmdd_tolerance', 0.05)
    project_velocity = kwargs.get('project_velocity', False)
    find_shallow_trench = kwargs.get("find_shallow_trench", False)
    # todo_parallel
    use_parallel = kwargs.get('use_parallel', False)
    file_tag = kwargs.get('file_tag', False)
    output_ov_ath_profile = kwargs.get('output_ov_ath_profile', False)
    kwargs["if_rewrite"] = True
    # get all available snapshots
    # the interval is choosen so there is no high frequency noises
    time_interval_for_slab_morphology = kwargs.get("time_interval", 0.5e6)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
    available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval_for_slab_morphology)
    print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
    # available_pvtu_steps = [i - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']) for i in available_pvtu_snapshots]
    # get where previous session ends
    vtk_output_dir = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(vtk_output_dir):
        os.mkdir(vtk_output_dir)
    # file name
    if file_tag == 'interval' and abs(time_interval_for_slab_morphology - 5e5)/5e5 > 1e-6:
        slab_morph_file_name = 'slab_morph_t%.2e.txt' % time_interval_for_slab_morphology
    else:
        slab_morph_file_name = 'slab_morph.txt'
    slab_morph_file = os.path.join(vtk_output_dir, slab_morph_file_name)
    # Initiation Wrapper class for parallel computation
    # ParallelWrapper = PARALLEL_WRAPPER_FOR_VTK('slab_morph', SlabMorphology_dual_mdd, if_rewrite=True, findmdd=findmdd, project_velocity=project_velocity, findmdd_tolerance=findmdd_tolerance)
    # parse the number of crust from the Visit_Options variable
    kwargs['n_crust'] = Visit_Options.options["N_CRUST"]
    ParallelWrapper = PARALLEL_WRAPPER_FOR_VTK('slab_morph', SlabMorphology_dual_mdd, **kwargs)
    ParallelWrapper.configure(case_dir)  # assign case directory
    # Remove previous file
    if os.path.isfile(slab_morph_file):
        print("%s: Delete old slab_morph.txt file." % func_name())
        os.remove(slab_morph_file)  # delete slab morph file
    ParallelWrapper.delete_temp_files(available_pvtu_snapshots)  # delete intermediate file if rewrite
    ParallelWrapper.set_pvtu_steps(available_pvtu_snapshots)
    num_cores = multiprocessing.cpu_count()
    # loop for all the steps to plot, the parallel version doesn't work for now
    if use_parallel:
        # raise NotImplementedError("Parallel for the function %s is not properly implemented yet" % func_name())
        Parallel(n_jobs=num_cores)(delayed(ParallelWrapper)(pvtu_snapshot)\
        for pvtu_snapshot in available_pvtu_snapshots)  # first run in parallel and get stepwise output
        print("call assemble_parallel")  # debug
        pvtu_steps_o, outputs = ParallelWrapper.assemble_parallel()
    else:
        for pvtu_snapshot in available_pvtu_snapshots:  # then run in on cpu to assemble these results
            ParallelWrapper(pvtu_snapshot)
        pvtu_steps_o, outputs = ParallelWrapper.assemble()
    ParallelWrapper.clear()
    # last, output
    # header
    file_header = "# 1: pvtu_step\n# 2: step\n# 3: time (yr)\n# 4: trench (rad)\n# 5: slab depth (m)\n\
# 6: 100km dip (rad)\n# 7: subducting plate velocity (m/yr)\n# 8: overiding plate velocity (m/yr)\n"
    n_col = 8
    if findmdd:
        file_header += "# 9: mechanical decoupling depth1 (m)\n# 10: mechanical decoupling depth2 (m)\n"
        n_col += 2
    if output_ov_ath_profile:
        file_header += "# %d: athenosphere velocity (m/yr)\n# %d: athenosphere viscosity (pa*s)\n" % (n_col+1, n_col+2)
        n_col += 2
    if find_shallow_trench:
        file_header += "# %d: shallow trench (rad)\n" % (n_col+1)
        n_col += 1
    # output file name
    appendix = ""
    if abs(time_interval_for_slab_morphology - 0.5e6) / 0.5e6 > 1e-6:
        appendix = "_t%.2e" % time_interval_for_slab_morphology
    output_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph' + appendix + '.txt')
    # output data
    if not os.path.isfile(output_file):
        with open(output_file, 'w') as fout:
            fout.write(file_header)
            for output in outputs:
                fout.write("%s" % output)
        print('Created output: %s' % output_file)
    else:
        with open(output_file, 'a') as fout:
            for output in outputs:
                fout.write("%s" % output)
        print('Updated output: %s' % output_file)


def WedgeTCase(case_dir, **kwargs):
    '''
    run vtk and get outputs for every snapshots
    Inputs:
        kwargs:
            time_interval: the interval between two processing steps
    '''
    # get all available snapshots
    # the interval is choosen so there is no high frequency noises
    time_interval = kwargs.get("time_interval", 0.5e6)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
    available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval)
    print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
    # get where previous session ends
    vtk_output_dir = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(vtk_output_dir):
        os.mkdir(vtk_output_dir)
    # Initiation Wrapper class for parallel computation
    ParallelWrapper = PARALLEL_WRAPPER_FOR_VTK('wedgeT', WedgeT, if_rewrite=True, assemble=False, output_poly_data=False)
    ParallelWrapper.configure(case_dir)  # assign case directory
    # Remove previous file
    print("%s: Delete old slab_temperature.txt file." % func_name())
    ParallelWrapper.delete_temp_files(available_pvtu_snapshots)  # delete intermediate file if rewrite
    num_cores = multiprocessing.cpu_count()
    # loop for all the steps to plot
    Parallel(n_jobs=num_cores)(delayed(ParallelWrapper)(pvtu_snapshot)\
    for pvtu_snapshot in available_pvtu_snapshots)  # first run in parallel and get stepwise output
    ParallelWrapper.clear()
    # for pvtu_snapshot in available_pvtu_snapshots:  # then run in on cpu to assemble these results
    #    ParallelWrapper(pvtu_snapshot)
    # pvtu_steps_o, outputs = ParallelWrapper.assemble()


def TrenchTCase(case_dir, **kwargs):
    '''
    run vtk and get outputs of trench temperature profile for every snapshots
    Inputs:
        case_dir: the directory of case
        kwargs:
            time_interval: the interval between two processing steps
    '''
    # get all available snapshots
    # the interval is choosen so there is no high frequency noises
    time_interval = kwargs.get("time_interval", 0.5e6)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
    available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval)
    print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
    # get where previous session ends
    vtk_output_dir = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(vtk_output_dir):
        os.mkdir(vtk_output_dir)
    # Initiation Wrapper class for parallel computation
    ParallelWrapper = PARALLEL_WRAPPER_FOR_VTK('trenchT', TrenchT, if_rewrite=True, assemble=False, output_poly_data=False)
    ParallelWrapper.configure(case_dir)  # assign case directory
    # Remove previous file
    print("%s: Delete old slab_temperature.txt file." % func_name())
    ParallelWrapper.delete_temp_files(available_pvtu_snapshots)  # delete intermediate file if rewrite
    num_cores = multiprocessing.cpu_count()
    # loop for all the steps to plot
    Parallel(n_jobs=num_cores)(delayed(ParallelWrapper)(pvtu_snapshot)\
    for pvtu_snapshot in available_pvtu_snapshots)  # first run in parallel and get stepwise output
    ParallelWrapper.clear()
    # for pvtu_snapshot in available_pvtu_snapshots:  # then run in on cpu to assemble these results
    #    ParallelWrapper(pvtu_snapshot)
    # pvtu_steps_o, outputs = ParallelWrapper.assemble()
    

def PlotSlabForcesCase(case_dir, vtu_step, **kwargs):
    '''
    Inputs:
        case_dir (str): case directory
        step : step to plot
        kwargs(dict):
            output_slab - output slab file
    '''
    output_slab = kwargs.get('output_slab', False)
    assert(os.path.isdir(case_dir))
    Visit_Options = VISIT_OPTIONS(case_dir)
    # call function
    Visit_Options.Interpret()
    vtu_snapshot = int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']) + vtu_step
    vtk_output_dir = os.path.join(case_dir, 'vtk_outputs')
    if not os.path.isdir(vtk_output_dir):
        os.mkdir(vtk_output_dir)
    ofile = os.path.join(vtk_output_dir, "slab_forces_%05d" % vtu_step)
    SlabAnalysis(case_dir, vtu_snapshot, ofile, output_slab=output_slab)
    # plot figure
    img_dir = os.path.join(case_dir, 'img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    fig_ofile = os.path.join(img_dir, "slab_forces_%05d.png" % vtu_step)
    PlotSlabForces(ofile, fig_ofile)



def PlotSlabShape(in_dir, vtu_step):
    '''
    Plot the shape of the slab, debug usage
    Inputs:
        in_dir (str): directory containing the data file
            a. a "slab_env0_{vtu_step}.txt" and a "slab_env1_{vtu_step}.txt" file
            b. a "slab_internal_{vtu_step}.txt" file
        vtu_step: step in visualization.
    '''
    fig, ax = plt.subplots()
    file_env0 = os.path.join(in_dir, "slab_env0_%05d.txt" % vtu_step)
    file_env1 = os.path.join(in_dir, "slab_env1_%05d.txt" % vtu_step)
    file_inter = os.path.join(in_dir, "slab_internal_%05d.txt" % vtu_step)
    slab_env0 = np.loadtxt(file_env0)
    slab_env1 = np.loadtxt(file_env1)
    slab_inter = np.loadtxt(file_inter)
    ax.plot(slab_env0[:, 0], slab_env0[:, 1], 'b.')
    ax.plot(slab_env1[:, 0], slab_env1[:, 1], 'c.')
    ax.plot(slab_inter[:, 0], slab_inter[:, 1], 'r.')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_aspect('equal', adjustable='box')
    plt.show()


def ShearZoneGeometryCase(case_dir, **kwargs):
    '''
    plot the shear zone geometry for a single case
    '''
    indent = kwargs.get("indent", 0)  # indentation for outputs
    time_start = kwargs.get("time_start", 0.0)
    time_interval = kwargs.get("time_interval", 5e6)
    time_end = kwargs.get("time_end", 60e6)
    assert(os.path.isdir(case_dir))
    # fix the output directory
    vtk_o_dir = os.path.join(case_dir, "vtk_outputs")
    if not os.path.isdir(vtk_o_dir):
        os.mkdir(vtk_o_dir)
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img_o_dir = os.path.join(img_dir, "shear_zone")
    if not os.path.isdir(img_o_dir):
        os.mkdir(img_o_dir)
    # initialize the VTK object
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_start=time_start, time_interval=time_interval, time_end=time_end)
    geometry = Visit_Options.options['GEOMETRY']
    Ro =  Visit_Options.options['OUTER_RADIUS']
    Xmax = Visit_Options.options['XMAX'] * np.pi / 180.0
    Dsz =  Visit_Options.options['INITIAL_SHEAR_ZONE_THICKNESS']
    # initiate the function and object for the figure
    fig, ax = plt.subplots()
    MorphPlotter = SLABPLOT("plot_slab")
    length = len(available_pvtu_snapshots)
    normalizer = [ float(i)/(length-1) for i in range(length) ] 
    colors = cm.rainbow(normalizer) 
    i = 0
    for vtu_snapshot in available_pvtu_snapshots:
        # prepare the slab
        vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
        _time, step = Visit_Options.get_time_and_step(vtu_step)
        filein = os.path.join(case_dir, "output", "solution", "solution-%05d.pvtu" % vtu_snapshot)
        if not os.path.isfile(filein):
            raise FileExistsError("input file (pvtu) doesn't exist: %s" % filein)
        else:
            print("%s%s: processing %s" % (indent*" ",func_name(), filein))
        VtkP = VTKP(geometry=geometry, Ro=Ro, Xmax=Xmax)
        VtkP.ReadFile(filein)
        field_names = ['T', 'density', 'spcrust', 'spharz', 'velocity']
        VtkP.ConstructPolyData(field_names, include_cell_center=True)
        VtkP.PrepareSlab(['spcrust', 'spharz'])
        # call the functions for the shear zone
        fileout = os.path.join(vtk_o_dir, "sz_%05d.txt" % vtu_step)
        VtkP.PrepareSZ(fileout, Dsz=Dsz)
        assert(os.path.isfile(fileout))  # assert file generation
        plot_initial = False
        if i == 0:
            plot_initial = True
        MorphPlotter.PlotShearZoneThickness(case_dir, plot_initial, axis=ax, filein=fileout,\
                                            label="t = %.4e Ma" % (_time/1e6), color=colors[i])
        i += 1
    # plot
    ax.legend()
    fig_path = os.path.join(img_o_dir, "sz_thickness_combined_s%.1f_i%.1f_e%.1f.png"\
        % (time_start/1e6, time_interval/1e6, time_end/1e6)) 
    fig.savefig(fig_path)
    assert(os.path.isfile(fig_path))  # assert figure generation
    print("%s%s: figure generated %s" % (indent*" ", func_name(), fig_path))


class SLABPLOT(LINEARPLOT):
    '''
    Plot slab morphology
    Inputs:
        -
    Returns:
        -
    '''
    def __init__(self, _name):
        LINEARPLOT.__init__(self, _name)

    class MorphFileReadingError(Exception):
        pass
 
    def ReadWedgeT(self, case_dir, **kwargs):
        '''
        read the wedge_T100 files and rearrange data
        '''
        time_interval = kwargs.get("time_interval", 0.5e6)
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        initial_adaptive_refinement = int(self.prm['Mesh refinement']['Initial adaptive refinement'])
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        elif geometry == 'box':
            Ro = float(self.prm['Geometry model']['Box']['Y extent'])
        else:
            raise ValueError('Invalid geometry')
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
        available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval)
        i = 0
        for pvtu_step in available_pvtu_snapshots:
            file_in_path = os.path.join(case_dir, 'vtk_outputs', 'wedge_T100_%05d.txt' % pvtu_step)
            print("file_in_path: ", file_in_path)  # debug
            my_assert(os.access(file_in_path, os.R_OK), FileExistsError, "File %s doesn\'t exist" % file_in_path)
            self.ReadHeader(file_in_path)
            self.ReadData(file_in_path)
            col_x = self.header['x']['col']
            col_y = self.header['y']['col']
            col_T = self.header['T']['col']
            xs = self.data[:, col_x]
            ys = self.data[:, col_y]
            if i == 0: 
                rs = (xs**2.0 + ys**2.0)**0.5
                depthes = Ro - rs # compute depth
                # Ts = np.zeros((depthes.size, max_pvtu_step - min_pvtu_step + 1))
                Ts = np.zeros((depthes.size, len(available_pvtu_snapshots)))
            Ts[:, i] = self.data[:, col_T]
            i += 1
        return depthes, Ts

    def PlotMorph(self, case_dir, **kwargs):
        '''
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            defined but not used
        '''
        save_pdf = kwargs.get("save_pdf", False) 
        compare_shallow_trench = kwargs.get("compare_shallow_trench", False)
        # initialization
        findmdd = False
        # path
        img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        morph_dir = os.path.join(img_dir, 'morphology')
        if not os.path.isdir(morph_dir):
            os.mkdir(morph_dir)
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        assert(os.path.isfile(slab_morph_file))
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        time_interval = times[1] - times[0]

        try: 
            col_pvtu_shallow_trench = self.header['shallow_trench']['col']
        except KeyError:
            col_pvtu_shallow_trench = None
            shallow_trenches = None
        else:
            shallow_trenches = self.data[:, col_pvtu_shallow_trench]

        if time_interval < 0.5e6:
            warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
            if shallow_trenches is not None:
                shallow_trenches_migration_length = (shallow_trenches - shallow_trenches[0]) * Ro  # length of migration
            else:
                shallow_trenches_migration_length = None
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
            shallow_trenches_migration_length = None # not implemented yet
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        if shallow_trenches_migration_length is not None:
            shallow_trench_velocities = np.gradient(shallow_trenches_migration_length, times)
        else:
            shallow_trench_velocities = None
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]
        try:
            col_mdd1 = self.header['mechanical_decoupling_depth1']['col']
            col_mdd2 = self.header['mechanical_decoupling_depth2']['col']
        except KeyError:
            pass
        else:
            findmdd = True
            mdds1 = self.data[:, col_mdd1]
            mdds2 = self.data[:, col_mdd2]
        # trench velocity
        # start figure
        if findmdd:
            _size = (20, 10)
            gs = gridspec.GridSpec(4, 2) 
        else:
            _size = (15, 10)
            gs = gridspec.GridSpec(3, 2) 
        fig = plt.figure(tight_layout=True, figsize=(15, 10)) 
        fig.subplots_adjust(hspace=0)
        # 1: trench & slab movement
        ax = fig.add_subplot(gs[0, 0:2]) 
        ax_tx = ax.twinx()
        lns0 = ax.plot(times/1e6, trenches_migration_length/1e3, '-', color='tab:orange', label='trench position (km)')
        if shallow_trenches_migration_length is not None:
            lns0_1 = ax.plot(times/1e6, shallow_trenches_migration_length/1e3, '--', color='tab:orange')
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ax.set_ylabel('Trench Position (km)', color="tab:orange")
        ax.tick_params(axis='x', labelbottom=False) # labels along the bottom edge are off
        ax.tick_params(axis='y', labelcolor="tab:orange")
        ax.grid()
        lns1 = ax_tx.plot(times/1e6, slab_depthes/1e3, 'k-', label='slab depth (km)')
        ax_tx.set_ylabel('Slab Depth (km)')
        lns = lns0 + lns1
        labs = [I.get_label() for I in lns]
        # ax.legend(lns, labs)
        # 2: velocity
        ax = fig.add_subplot(gs[1, 0:2]) 
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        lns0 = ax.plot(times/1e6, trench_velocities*1e2, '-', color='tab:orange', label='trench velocity (cm/yr)')
        if shallow_trenches_migration_length is not None:
            lns0_1 = ax.plot(times/1e6, shallow_trench_velocities*1e2, '--', color='tab:orange')
        lns1 = ax.plot(times/1e6, sp_velocities*1e2, '-', color='tab:blue', label='subducting plate (cm/yr)')
        lns2 = ax.plot(times/1e6, ov_velocities*1e2, '-', color='tab:purple', label='overiding velocity (cm/yr)')
        ax.plot(times/1e6, sink_velocities*1e2, 'k-', label='sinking velocity (cm/yr)')
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ax.set_ylim((-10, 10))
        ax.set_ylabel('Velocity (cm/yr)')
        ax.set_xlabel('Times (Myr)')
        ax.grid()
        ax.legend()
        # 2.1: velocity smaller, no y limit, to show the whole curve
        ax = fig.add_subplot(gs[2, 0:2]) 
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        lns0 = ax.plot(times/1e6, trench_velocities*1e2, '-', color="tab:orange", label='trench velocity (cm/yr)')
        lns1 = ax.plot(times/1e6, sp_velocities*1e2, '-', color='tab:blue', label='subducting plate (cm/yr)')
        lns2 = ax.plot(times/1e6, ov_velocities*1e2, '-', color='tab:purple', label='overiding velocity (cm/yr)')
        ax.plot(times/1e6, sink_velocities*1e2, 'k-', label='trench velocity (cm/yr)')
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ax.set_ylabel('Velocity (whole, cm/yr)')
        ax.grid()
        # 3: the mechanical decoupling depth, only if the data is found
        if findmdd:
            ax = fig.add_subplot(gs[3, 0:2]) 
            ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
            lns3 = ax.plot(times/1e6, mdds1/1e3, '-', color="tab:blue", label='mdd1 (km)')
            lns4 = ax.plot(times/1e6, mdds2/1e3, '-', color="c", label='mdd2 (km)')
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
            ax.set_ylabel('Depth (km)')
            ax.grid()
            ax.legend()
        fig.tight_layout()
        # save figure
        if save_pdf:
            o_path = os.path.join(morph_dir, 'trench.pdf')
            plt.savefig(o_path, format="pdf", bbox_inches="tight")
        else:
            o_path = os.path.join(morph_dir, 'trench.png')
            plt.savefig(o_path)
        print("%s: figure %s generated" % (func_name(), o_path))
        if compare_shallow_trench:
            # compare shallow trench to trench
            assert(shallow_trenches_migration_length is not None)
            # Create a figure with 3 subplots arranged vertically
            fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6, 8))
            # plot location
            lns0 = ax0.plot(times/1e6, trenches, '-', color='tab:orange', label='trench')
            lns0_1 = ax0.plot(times/1e6, shallow_trenches, '--', color='tab:orange', label="shallow trench")
            ax0.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
            ax0.set_ylabel('Trench Position')
            ax0.grid()
            ax0.legend()
            lns1 = ax1.plot(times/1e6, trench_velocities*1e2, '-', color='tab:orange', label='trench velocity (cm/yr)')
            lns1_1 = ax1.plot(times/1e6, shallow_trench_velocities*1e2, '--', color='tab:orange')
            ax1.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
            ax1.set_ylabel('Velocity (whole, cm/yr)')
            ax1.set_xlabel('Times (Myr)')
            ax1.grid()
            o_path = os.path.join(morph_dir, 'shallow_trench_compare.pdf')
            plt.savefig(o_path)
            print("%s: figure %s generated" % (func_name(), o_path))
    
    def PlotMorphAnime(self, case_dir, **kwargs):
        '''
        Plot slab morphology for animation
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            time -a time to plot
        '''
        _time = kwargs.get('time', '0.0')
        # path
        img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        morph_dir = os.path.join(img_dir, 'morphology')
        if not os.path.isdir(morph_dir):
            os.mkdir(morph_dir)
        temp_dir = os.path.join(morph_dir, 'temp')
        if not os.path.isdir(temp_dir):
            os.mkdir(temp_dir)
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        assert(os.path.isfile(slab_morph_file))
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        time_interval = times[1] - times[0]
        if time_interval < 0.5e6:
            warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]

        # 1: trench & slab movement
        gs = gridspec.GridSpec(2, 1) 
        fig = plt.figure(tight_layout=True, figsize=(10, 10)) 
        ax = fig.add_subplot(gs[0, 0]) 
        ax_tx = ax.twinx()
        lns0 = ax.plot(times/1e6, trenches_migration_length/1e3, '-', color='tab:orange', label='trench position (km)')
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ylim0 = np.floor(np.min(trenches_migration_length)/1e5)*1e2
        ylim1 = np.floor(np.max(trenches_migration_length)/1e5)*1e2
        ax.set_ylim((ylim0, ylim1))
        ax.set_ylabel('Trench Position (km)', color="tab:orange")
        ax.set_xlabel('Time (Ma)')
        ax.tick_params(axis='y', labelcolor="tab:orange")
        ax.grid()
        temp_ts = _time * np.ones(100)
        temp_ys = np.linspace(ylim0, ylim1, 100)
        ax.plot(temp_ts/1e6, temp_ys, 'c--') # plot a vertical line
        lns1 = ax_tx.plot(times/1e6, slab_depthes/1e3, 'k-', label='slab depth (km)')
        ax_tx.set_ylabel('Slab Depth (km)')
        lns = lns0 + lns1
        labs = [I.get_label() for I in lns]
        ax.legend(lns, labs, loc='lower right')
        # ax1: velocity
        ax = fig.add_subplot(gs[1, 0]) 
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        lns0 = ax.plot(times/1e6, trench_velocities*1e2, '-', color="tab:orange", label='trench velocity (cm/yr)')
        lns1 = ax.plot(times/1e6, sp_velocities*1e2, '-', color='tab:blue', label='subducting plate (cm/yr)')
        ax.plot(times/1e6, sink_velocities*1e2, 'k-', label='trench velocity (cm/yr)')
        temp_ts = _time * np.ones(100)
        temp_ys = np.linspace(-10, 10, 100)
        ax.plot(temp_ts/1e6, temp_ys, 'c--') # plot a vertical line
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ax.set_ylim((-10, 10))
        ax.xaxis.set_ticks_position("top")
        ax.set_ylabel('Velocity (whole, cm/yr)')
        ax.grid()
        ax.legend(loc='lower right')
        # save figure
        o_path = os.path.join(temp_dir, 'trench_t%.4e.png' % _time)
        fig.savefig(o_path)
        print("%s: save figure %s" % (func_name(), o_path))
        plt.close()

    def PlotMorphPublication(self, case_dir, **kwargs):
        '''
        Plot slab morphology for publication
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            time -a time to plot
        '''
        time_interval = kwargs.get('time_interval', 5e6)
        time_range = kwargs.get('time_range', None)
        if time_range is not None:
            assert(len(time_range) == 2)
        time_markers = kwargs.get("time_markers", [])
        vlim = kwargs.get("vlim", [-10, 10])
        save_pdf = kwargs.get("save_pdf", False)
        assert(len(vlim) == 2)
        # path
        img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        morph_dir = os.path.join(img_dir, 'morphology')
        if not os.path.isdir(morph_dir):
            os.mkdir(morph_dir)
        # visit option class 
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        # input file name
        appendix = ""
        if abs(time_interval - 0.5e6) / 0.5e6 > 1e-6:
            appendix = "_t%.2e" % time_interval
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph' + appendix + '.txt')
        assert(os.path.isfile(slab_morph_file))
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        col_athenosphere_velocity = self.header['athenosphere_velocity']['col']
        col_athenosphere_viscosity = self.header['athenosphere_viscosity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        time_interval = times[1] - times[0]
        if time_interval < 0.5e6:
            warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]
        athenosphere_velocities = self.data[:, col_athenosphere_velocity]
        athenosphere_viscosities = self.data[:, col_athenosphere_viscosity]

        # 1: trench & slab movement
        gs = gridspec.GridSpec(4, 1) 
        fig = plt.figure(tight_layout=True, figsize=(20, 40)) 
        ax = fig.add_subplot(gs[0, 0]) 
        ax_tx = ax.twinx()
        lns0 = ax.plot(times/1e6, trenches_migration_length/1e3, '-', color='tab:orange', label='trench position (km)')
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ylim0 = np.floor(np.min(trenches_migration_length)/1e5)*1e2
        ylim1 = np.floor(np.max(trenches_migration_length)/1e5)*1e2
        ax.set_ylim((ylim0, ylim1))
        ax.set_ylabel('Trench Position (km)', color="tab:orange")
        ax.set_xlabel('Time (Ma)')
        ax.tick_params(axis='y', labelcolor="tab:orange")
        ax.grid()
        for _time in time_markers:
            temp_ts = _time * np.ones(100)
            temp_ys = np.linspace(ylim0, ylim1, 100)
            ax.plot(temp_ts/1e6, temp_ys, 'c--', dashes=(10, 10), alpha=0.7) # plot a vertical line
        lns1 = ax_tx.plot(times/1e6, slab_depthes/1e3, 'k-', label='slab depth (km)')
        if time_range is None:
            ax_tx.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax_tx.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ax_tx.set_ylabel('Slab Depth (km)')
        lns = lns0 + lns1
        labs = [I.get_label() for I in lns]
        ax.legend(lns, labs, loc='lower right')
        # ax1: velocity
        # ax1, part 1: velcoity
        ax = fig.add_subplot(gs[1, 0]) 
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        ln_v_tr = ax.plot(times/1e6, trench_velocities*1e2, '-', color="tab:orange", label='trench velocity (cm/yr)')
        ln_v_sub = ax.plot(times/1e6, sp_velocities*1e2, '-', color='tab:blue', label='subducting plate (cm/yr)')
        ln_v_ov = ax.plot(times/1e6, ov_velocities*1e2, '-', color='tab:purple', label='overiding plate (cm/yr)')
        ln_v_ath = ax.plot(times/1e6, athenosphere_velocities*1e2, '-', color='c', label='athenosphere (cm/yr)')
        ln_v_sink = ax.plot(times/1e6, sink_velocities*1e2, 'k-', label='sink velocity (cm/yr)')
        for _time in time_markers:
            temp_ts = _time * np.ones(200)
            temp_ys = np.linspace(-100, 100, 200)
            ax.plot(temp_ts/1e6, temp_ys, 'c--', dashes=(10, 10), alpha=0.7) # plot a vertical line
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ax.set_xlabel('Time (Ma)')
        ax.set_ylim((vlim[0], vlim[1]))
        ax.set_ylabel('Velocity (cm/yr)')
        ax.grid()
        # ax1, part 1: trench position
        ax_tx = ax.twinx()
        ln_tr = ax_tx.plot(times/1e6, trenches_migration_length/1e3, '*', color='tab:orange', label='trench position (km)')
        ax_tx.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ylim0 = np.floor(np.min(trenches_migration_length)/1e5)*1e2
        ylim1 = np.floor(np.max(trenches_migration_length)/1e5)*1e2
        if time_range is None:
            ax_tx.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax_tx.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ax_tx.set_ylim((ylim0, ylim1))
        ax_tx.set_ylabel('Trench Position (km)', color="tab:orange")
        ax_tx.tick_params(axis='y', labelcolor="tab:orange")
        lns = ln_tr + ln_v_tr + ln_v_sub + ln_v_sink + ln_v_ov + ln_v_ath
        labs = [I.get_label() for I in lns]
        ax.legend(lns, labs, loc='upper right')
        # read athenosphere dataset
        available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=float(time_interval))
        depth_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        time_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        viscosity_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        velocity_h_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        for i in range(len(available_pvtu_snapshots)):
            vtu_snapshot = available_pvtu_snapshots[i]
            _time, _ = Visit_Options.get_time_and_step_by_snapshot(vtu_snapshot)
            vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
            slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'ov_ath_profile_%.5d.txt' % vtu_step)
            assert(os.path.isfile(slab_morph_file))
            data = np.loadtxt(slab_morph_file)
            depths = data[:, 2]
            depth_mesh[i, :] = depths
            time_mesh[i, :] = np.ones(100) * _time
            velocities_h = data[:, 3]
            velocity_h_mesh[i, :] = velocities_h
            viscosities = data[:, 5]
            viscosity_mesh[i, :] = viscosities
        # plot the viscosity
        ax = fig.add_subplot(gs[2, 0])
        h = ax.pcolormesh(time_mesh/1e6, depth_mesh/1e3, np.log10(viscosity_mesh), cmap=ccm.roma,\
            vmin=np.log10(Visit_Options.options["ETA_MIN"]), vmax=np.log10(Visit_Options.options["ETA_MAX"]))
        ax.invert_yaxis()
        fig.colorbar(h, ax=ax, label='log(viscosity (Pa*s))', orientation="horizontal")
        ax.set_xlabel("Time (Ma)")
        ax.set_ylabel("Depth (km)")
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        # plot the velocity
        ax = fig.add_subplot(gs[3, 0])
        # todo_2dmorph
        vlim_for_ath = kwargs.get("vlim_for_ath", None)
        v_min = np.floor(velocity_h_mesh.min()*100.0 / 5.0) * 5.0
        v_max = np.ceil(velocity_h_mesh.max()*100.0 / 5.0) * 5.0
        if vlim_for_ath is not None:
            assert(type(vlim_for_ath) == list and len(vlim_for_ath) == 2)
            v_min = vlim_for_ath[0]
            v_max = vlim_for_ath[1]
        h = ax.pcolormesh(time_mesh/1e6, depth_mesh/1e3, velocity_h_mesh*100.0, cmap=ccm.vik, vmin=v_min, vmax=v_max)
        ax.invert_yaxis()
        fig.colorbar(h, ax=ax, label='velocity (cm/yr)', orientation="horizontal")
        ax.set_ylabel("Depth (km)")
        ax.set_xlabel("Time (Ma)")
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        # save figure
        o_path = os.path.join(morph_dir, 'trench_t%.2e' % time_interval)
        fig.savefig(o_path + '.png')
        print("%s: save figure %s" % (func_name(), o_path + '.png'))
        if save_pdf:
            fig.savefig(o_path + '.pdf')
            print("%s: save figure %s" % (func_name(), o_path + '.pdf'))
        plt.close()

    def PlotMorphPublicationBillen18(self, case_dir, **kwargs):
        '''
        Plot slab morphology for publication
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            time -a time to plot
        '''
        time_interval = kwargs.get('time_interval', 5e6)
        time_range = kwargs.get('time_range', None)
        time_markers = kwargs.get("time_markers", [])
        vlim = kwargs.get("vlim", [-10, 10])
        save_pdf = kwargs.get("save_pdf", False)
        assert(len(vlim) == 2)
        # path
        img_dir = os.path.join(case_dir, 'img')
        if not os.path.isdir(img_dir):
            os.mkdir(img_dir)
        morph_dir = os.path.join(img_dir, 'morphology')
        if not os.path.isdir(morph_dir):
            os.mkdir(morph_dir)
        # visit option class 
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        # input file name
        appendix = ""
        if abs(time_interval - 0.5e6) / 0.5e6 > 1e-6:
            appendix = "_t%.2e" % time_interval
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph' + appendix + '.txt')
        assert(os.path.isfile(slab_morph_file))
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        col_athenosphere_velocity = self.header['athenosphere_velocity']['col']
        col_athenosphere_viscosity = self.header['athenosphere_viscosity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        time_interval = times[1] - times[0]
        if time_interval < 0.5e6:
            warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]
        athenosphere_velocities = self.data[:, col_athenosphere_velocity]
        athenosphere_viscosities = self.data[:, col_athenosphere_viscosity]

        # start and end time
        time0, time1 = times[0]/1e6, times[-1]/1e6
        if time_range is not None:
            assert(len(time_range) == 2)
            time0, time1 = time_range[0] / 1e6, time_range[1] / 1e6

        # ax1, part 1: velcoity
        gs = gridspec.GridSpec(3, 1) 
        fig = plt.figure(tight_layout=True, figsize=(20, 30)) 
        ax = fig.add_subplot(gs[0, 0]) 
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        ln_v_tr = ax.plot(times/1e6, trench_velocities*1e2, '-', color="tab:orange", label='trench velocity (cm/yr)')
        ln_v_sub = ax.plot(times/1e6, sp_velocities*1e2, '-', color='tab:blue', label='subducting plate (cm/yr)')
        ln_v_ov = ax.plot(times/1e6, ov_velocities*1e2, '-', color='tab:purple', label='overiding plate (cm/yr)')
        ln_v_ath = ax.plot(times/1e6, athenosphere_velocities*1e2, '-', color='r', label='athenosphere (cm/yr)')
        for _time in time_markers:
            temp_ts = _time * np.ones(200)
            temp_ys = np.linspace(-100, 100, 200)
            ax.plot(temp_ts/1e6, temp_ys, 'c--', dashes=(10, 10), alpha=0.7) # plot a vertical line
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ax.set_ylim((vlim[0], vlim[1]))
        ax.set_ylabel('Velocity (cm/yr)')
        ax.tick_params(axis='x', which='both', direction='in', labelbottom=False)
        ax.grid()
        lns = ln_v_tr + ln_v_sub + ln_v_ov + ln_v_ath
        labs = [I.get_label() for I in lns]
        ax.legend(lns, labs, loc='upper right')
        # read athenosphere dataset
        available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=float(time_interval))
        depth_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        time_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        viscosity_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        velocity_h_mesh = np.zeros([len(available_pvtu_snapshots), 100])
        for i in range(len(available_pvtu_snapshots)):
            vtu_snapshot = available_pvtu_snapshots[i]
            _time, _ = Visit_Options.get_time_and_step_by_snapshot(vtu_snapshot)
            vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
            slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'ov_ath_profile_%.5d.txt' % vtu_step)
            # my_assert(os.path.isfile(slab_morph_file), FileExistsError, "%s: %s doesn't exist" % (func_name(), slab_morph_file))
            if os.path.isfile(slab_morph_file):
                data = np.loadtxt(slab_morph_file)
                depths = data[:, 2]
                depth_mesh[i, :] = depths
                time_mesh[i, :] = np.ones(100) * _time
                velocities_h = data[:, 3]
                velocity_h_mesh[i, :] = velocities_h
                viscosities = data[:, 5]
                viscosity_mesh[i, :] = viscosities
            else:
                # a method to fix invalid timestep:
                # make sure this plots outside of the figure
                depths = np.ones(100) * (-1)
                time_mesh[i, :] = np.ones(100) * _time
                velocities_h = data[:, 3]
        # plot the viscosity
        ax = fig.add_subplot(gs[1, 0])
        h = ax.pcolormesh(time_mesh/1e6, depth_mesh/1e3, np.log10(viscosity_mesh), cmap=ccm.roma,\
            vmin=np.log10(Visit_Options.options["ETA_MIN"]), vmax=np.log10(Visit_Options.options["ETA_MAX"]))
        ax.invert_yaxis()
        fig.colorbar(h, ax=ax, label='log(viscosity (Pa*s))', orientation="horizontal")
        ax.set_ylabel("Depth (km)")
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        ax.tick_params(axis='x', which='both', direction='in', labelbottom=False)
        # plot the velocity
        ax = fig.add_subplot(gs[2, 0])
        # todo_2dmorph
        vlim_for_ath = kwargs.get("vlim_for_ath", None)
        v_min = np.floor(velocity_h_mesh.min()*100.0 / 5.0) * 5.0
        v_max = np.ceil(velocity_h_mesh.max()*100.0 / 5.0) * 5.0
        if vlim_for_ath is not None:
            assert(type(vlim_for_ath) == list and len(vlim_for_ath) == 2)
            v_min = vlim_for_ath[0]
            v_max = vlim_for_ath[1]
        h = ax.pcolormesh(time_mesh/1e6, depth_mesh/1e3, velocity_h_mesh*100.0, cmap=ccm.vik, vmin=v_min, vmax=v_max)
        ax.invert_yaxis()
        fig.colorbar(h, ax=ax, label='velocity (cm/yr)', orientation="horizontal")
        ax.set_ylabel("Depth (km)")
        ax.set_xlabel("Time (Ma)")
        if time_range is None:
            ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        else:
            ax.set_xlim((time_range[0]/1e6, time_range[1]/1e6))  # set x limit
        fig.subplots_adjust(hspace=0)
        # save figure
        o_path = os.path.join(morph_dir, 'trench_b18_t%.2e' % time_interval)
        fig.savefig(o_path + '.png')
        print("%s: save figure %s" % (func_name(), o_path + '.png'))
        if save_pdf:
            fig.savefig(o_path + '.pdf')
            print("%s: save figure %s" % (func_name(), o_path + '.pdf'))
        plt.close()

    def PlotTWedge(self, case_dir, **kwargs):
        '''
        plot the mantle wedge temperature on top of the 100-km deep slab.
        '''
        time_interval = kwargs.get("time_interval", 0.5e6)
        ax = kwargs.get('axis', None)
        if ax == None:
            raise ValueError("Not implemented")
        depthes, Ts = self.ReadWedgeT(case_dir)
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval)
        print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
        times = []
        for snapshot in available_pvtu_snapshots:
            _time, _ = Visit_Options.get_time_and_step_by_snapshot(snapshot)
            times.append(_time)
        times = np.array(times)
        tt, dd = np.meshgrid(times, depthes)
        h = ax.pcolormesh(tt/1e6,dd/1e3,Ts, shading='gouraud') 
        ax.invert_yaxis()
        ax.set_xlim((times[0]/1e6, times[-1]/1e6))  # set x limit
        ax.set_xlabel('Times (Myr)')
        ax.set_ylabel('Depth (km)')
        return ax, h

    
    def PlotTrenchVelocity(self, case_dir, **kwargs):
        '''
        a variation of the PlotMorph function: used for combining results
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            defined but not used
        '''
        # initiate
        ax = kwargs.get('axis', None)
        if ax == None:
            raise ValueError("Not implemented")
        label_all = kwargs.get('label_all', False)
        color = kwargs.get('color', None)
        time_range = kwargs.get('time_range', [])
        v_range = kwargs.get('v_range', [])
        fix_v_range = kwargs.get('fix_v_range', False)
        if label_all:
            # if label_all, append labels, otherwise don't
            labels = ["trench velocity", "subducting plate", "overiding velocity", "sinking velocity"]
        else:
            labels = [None, None, None, None]
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        my_assert(os.path.isfile(slab_morph_file), FileExistsError, "%s doesn't exist" % slab_morph_file)
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]
        # trench velocity
        ax.plot(times/1e6, 0.0 * np.zeros(times.shape), 'k--')
        lns0 = ax.plot(times/1e6, trench_velocities*1e2, '-', color=color, label=labels[0])
        lns1 = ax.plot(times/1e6, sp_velocities*1e2, ':', color=color, label=labels[1])
        lns2 = ax.plot(times/1e6, ov_velocities*1e2, '-.', color=color, label=labels[2])
        ax.plot(times/1e6, sink_velocities*1e2, '--', color=color, label=labels[3])
        if time_range != []:
            xlims = time_range
        else:
            xlims = (np.min(times), np.max(times))
        ax.set_xlim(xlims[0]/1e6, xlims[1]/1e6)  # set x limit
        # for the limit of y, there are 3 options: a. fix_v_range would give a (-10, 20);
        # b. assigne a v_range will apply that value; c. by default, the min value of 
        # the trench velocity and the max value of the subducting velocity will be used.
        if v_range != []:
            ylims = v_range
        else:
            mask = (times > xlims[0]) & (times < xlims[1])
            ylims = [-0.15, np.max(sp_velocities[mask])]
        if fix_v_range:
            ax.set_ylim((-10, 20))
        else:
            ax.set_ylim((ylims[0]*1e2, ylims[1]*1e2))
        ax.set_ylabel('Velocity (cm/yr)')
        ax.set_xlabel('Times (Myr)')
        ax.grid()
        ax.legend()
        # lns = lns0 + lns1
        # labs = [I.get_label() for I in lns]
        # return lns, labs
    
    
    def PlotTrenchPosition(self, case_dir, **kwargs):
        '''
        a variation of the PlotMorph function: used for combining results
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            defined but not used
        '''
        # initiate
        ax = kwargs.get('axis', None)
        if ax == None:
            raise ValueError("Not implemented")
        label = kwargs.get('label', [None, None])
        assert(len(label) == 2)
        color = kwargs.get('color', None)
        time_range = kwargs.get('time_range', [])
        tp_range = kwargs.get('tp_range', [])
        sd_range = kwargs.get('sd_range', [])
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,\
        "prm file %s cannot be opened" % prm_file)
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        my_assert(os.path.isfile(slab_morph_file), FileExistsError, "%s doesn't exist" % slab_morph_file)
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        col_pvtu_sp_v = self.header['subducting_plate_velocity']['col']
        col_pvtu_ov_v = self.header['overiding_plate_velocity']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        sp_velocities = self.data[:, col_pvtu_sp_v]
        ov_velocities = self.data[:, col_pvtu_ov_v]
        # trench velocity
        # 1: trench & slab movement
        ax_tx = ax.twinx()
        lns0 = ax.plot(times/1e6, trenches_migration_length/1e3, '-', color=color, label=label[0])
        if time_range != []:
            xlims = time_range
        else:
            xlims = (np.min(times), np.max(times))
        ax.set_xlim(xlims[0]/1e6, xlims[1]/1e6)  # set x limit
        ax.set_xlabel("Time (Myr)")
        if tp_range != []:
            ylims = tp_range
        else:
            ylims = (np.min(trenches_migration_length), np.max(trenches_migration_length))
        ax.set_ylim(ylims[0]/1e3, ylims[1]/1e3)
        ax.set_ylabel('Trench Position (km)')
        ax.tick_params(axis='x') # labels along the bottom edge are off
        ax.tick_params(axis='y')
        ax.grid()
        lns1 = ax_tx.plot(times/1e6, slab_depthes/1e3, '--', color=color, label=label[1])
        if sd_range != []:
            ylims = sd_range
        else:
            ylims = (np.min(slab_depthes), np.max(slab_depthes))
        ax_tx.set_ylim(ylims[0]/1e3, ylims[1]/1e3)
        ax_tx.set_ylabel('Slab Depth (km)')
        lns = lns0 + lns1
        labs = [I.get_label() for I in lns]
        return lns, labs

    def PlotShearZoneThickness(self, case_dir, plot_initial=True, **kwargs):
        '''
        a variation of the PlotMorph function: used for combining results
        Inputs:
            case_dir (str): directory of case
        kwargs(dict):
            defined but not used
        '''
        # initiate
        ax = kwargs.get('axis', None)
        if ax == None:
            raise ValueError("Not implemented")
        filein = kwargs.get("filein", None)
        if filein is not None:
            sz_file = filein
        else:
            sz_file = os.path.join(case_dir, 'vtk_outputs', 'shear_zone.txt')
        my_assert(os.path.isfile(sz_file), FileExistsError, "%s doesn't exist" % sz_file)
        label = kwargs.get('label', None)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        _color = kwargs.get("color", "tab:blue")
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        wb_file = os.path.join(case_dir, 'case.wb')
        assert(os.access(wb_file, os.R_OK))
        # get the initial thickness of the shear zone
        with open(wb_file, 'r') as fin:
            wb_dict = json.load(fin)
        i0 = find_wb_feature(wb_dict, 'Subducting plate')
        sp_dict = wb_dict['features'][i0]
        initial_thickness = sp_dict["composition models"][0]["max depth"]
        # read data
        self.ReadHeader(sz_file)
        self.ReadData(sz_file)
        col_depth = self.header['depth']['col']
        col_theta_min = self.header['theta_min']['col']
        col_theta_max = self.header['theta_max']['col']
        depths = self.data[:, col_depth]
        theta_mins = self.data[:, col_theta_min]
        theta_maxs = self.data[:, col_theta_max]
        # convert to thickness along strike
        num = depths.size
        thicks = np.zeros(num)
        for i in range(num):
            r_min = Ro - depths[i]
            theta_min = theta_mins[i]
            theta_max = theta_maxs[0]
            thick = 0.0
            for j in range(0, num):
                r_max = Ro - depths[j]
                theta_max = theta_maxs[j]
                thick_temp = point2dist([theta_min, r_min], [theta_max, r_max], geometry)
                if j == 0:
                    thick = thick_temp
                if thick_temp < thick:
                    thick = thick_temp
            thicks[i] = thick 
        # plot
        mask = (depths > initial_thickness) & (theta_mins > 0.0) # points with theta min < 0.0 are those on the surface
        ax.plot(depths[mask]/1e3, thicks[mask]/1e3, label=label, color=_color) # debug
        if plot_initial:
            ax.plot(depths/1e3, initial_thickness*np.ones(depths.size)/1e3, 'k--')
        if xlims is not None:
            # set x limit
            assert(len(xlims) == 2)
            ax.set_xlim([xlims[0]/1e3, xlims[1]/1e3])
        if ylims is not None:
            # set y limit
            assert(len(ylims) == 2)
            ax.set_ylim([ylims[0]/1e3, ylims[1]/1e3])
        ax.set_xlabel("Depth (km)")
        ax.set_ylabel("Thickness (km)")

    def PlotSlabT(self, case_dir, **kwargs):
        '''
        Plot the slab temperature
        Inputs:
            case_dir (str) - directory of case
            kwargs(dict):
                axis - a matplotlib axis
        '''
        # initiate
        ax = kwargs.get('axis', None)
        if ax == None:
            raise ValueError("Not implemented")
        filein = kwargs.get("filein", None)
        if filein is not None:
            temp_file = filein
        else:
            temp_file = os.path.join(case_dir, 'vtk_outputs', 'shear_zone.txt')
        my_assert(os.path.isfile(temp_file), FileExistsError, "%s doesn't exist" % temp_file)
        label = kwargs.get('label', None)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        # read data
        self.ReadHeader(temp_file)
        self.ReadData(temp_file)
        col_depth = self.header['depth']['col']
        col_Tbot = self.header['Tbot']['col']
        col_Ttop = self.header['Ttop']['col']
        depths = self.data[:, col_depth]
        Tbots = self.data[:, col_Tbot]
        Ttops = self.data[:, col_Ttop]
        ax.plot(Ttops, depths/1e3, label=label, color="tab:blue")
        mask = (Tbots > 0.0)  # non-sense values are set to negative when these files are generated
        ax.plot(Tbots[mask], depths[mask]/1e3, label=label, color="tab:green")
        if xlims is not None:
            # set temperature limit
            assert(len(xlims) == 2)
            ax.set_xlim([xlims[0], xlims[1]])
        if ylims is not None:
            # set depth limit
            assert(len(ylims) == 2)
            ax.set_ylim([ylims[0]/1e3, ylims[1]/1e3])
        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel("Depth (km)")

    def PlotSlabTCase(self, case_dir, **kwargs):
        '''
        Plot the temperature profile for a case by assembling results
        from individual steps
            kwargs(dict):
                axis - a matplotlib axis
                debug - print debug message
                time_range - range of the time for plotting the temperature
        '''
        # initiate
        n_plot = 100 # points in the plot
        max_plot_depth = 250e3
        ax = kwargs.get('axis', None)
        debug = kwargs.get('debug', False)
        use_degree = kwargs.get('use_degree', False)
        if ax == None:
            raise ValueError("Not implemented")
        label = kwargs.get('label', None)
        xlims = kwargs.get('xlims', None)
        ylims = kwargs.get('ylims', None)
        time_range = kwargs.get('time_range', None)
        if time_range is not None:
            assert(len(time_range) == 2)
            assert(time_range[0] < time_range[1])
        # options for slab temperature outputs
        time_interval_for_slab_morphology = kwargs.get("time_interval", 0.5e6)
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
        available_pvtu_snapshots= Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval_for_slab_morphology)
        print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
        # assert all files exist
        for pvtu_snapshot in available_pvtu_snapshots:
            vtu_step = max(0, int(pvtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
            output_path = os.path.join(case_dir, "vtk_outputs")
            temp_file = os.path.join(output_path, "slab_temperature_%05d.txt" % (vtu_step))
            assert(os.access(temp_file, os.R_OK))
        pDepths = np.linspace(0, max_plot_depth, n_plot)
        # derive the range of the slab temperatures
        pTtops_min = np.ones(n_plot) * 1e31
        pTtops_max = np.ones(n_plot) * (-1e31)
        pTtops_med = np.zeros(n_plot)
        pTtops_wt = np.zeros(n_plot)
        pTbots_min = np.ones(n_plot) * 1e31
        pTbots_max = np.ones(n_plot) * (-1e31)
        pTbots_med = np.zeros(n_plot)
        pTbots_wt = np.zeros(n_plot)
        time_last = 0.0
        for pvtu_snapshot in available_pvtu_snapshots:
            vtu_step = max(0, int(pvtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
            _time, step = Visit_Options.get_time_and_step(vtu_step)
            if time_range is not None:
                if _time < time_range[0] or _time > time_range[1]:
                    # if the step is out of the time range, skip it.
                    continue
            dtime = (_time - time_last)
            time_last = _time
            output_path = os.path.join(case_dir, "vtk_outputs")
            temp_file = os.path.join(output_path, "slab_temperature_%05d.txt" % (vtu_step))
            # read data from file
            self.ReadHeader(temp_file)
            self.ReadData(temp_file)
            col_depth = self.header['depth']['col']
            col_Tbot = self.header['Tbot']['col']
            col_Ttop = self.header['Ttop']['col']
            depths = self.data[:, col_depth]
            Tbots = self.data[:, col_Tbot]
            Ttops = self.data[:, col_Ttop]
            Tbot_func = interp1d(depths, Tbots, assume_sorted=True) 
            Ttop_func = interp1d(depths, Ttops, assume_sorted=True) 
            for i in range(n_plot):
                pDepth = pDepths[i]
                if pDepth < depths[0] or pDepth > depths[-1]:
                    # the range in the slab temperature file
                    # could be limited, so that invalid values
                    # are skipped 
                    continue
                pTtop = Ttop_func(pDepth)
                if pTtop > 0.0:
                    if pTtop < pTtops_min[i]:
                        pTtops_min[i] = pTtop
                    if pTtop > pTtops_max[i]:
                        pTtops_max[i] = pTtop
                    pTtops_med[i] += pTtop * dtime
                    pTtops_wt[i] += dtime
                pTbot = Tbot_func(pDepth)
                if pTbot > 0.0:
                    # only deal with valid values
                    if pTbot < pTbots_min[i]:
                        pTbots_min[i] = pTbot
                    if pTbot > pTbots_max[i]:
                        pTbots_max[i] = pTbot
                    pTbots_med[i] += pTbot * dtime
                    pTbots_wt[i] += dtime
        pTtops_med /= pTtops_wt
        pTbots_med /= pTbots_wt
        if debug:
            print("pTbots_min: ")  # screen outputs
            print(pTbots_min)
            print("pTbots_max: ") 
            print(pTbots_max)
            print("pTbots_med: ")
            print(pTbots_med)
        # plot result of slab surface temperature
        mask = ((pTbots_min > 0.0) & (pTbots_min < 1e4))
        ax.plot(pTbots_min[mask], pDepths[mask]/1e3, "--", label="moho T", color="tab:green")
        mask = ((pTbots_max > 0.0) & (pTbots_max < 1e4))
        ax.plot(pTbots_max[mask], pDepths[mask]/1e3, "--", color="tab:green")
        mask = ((pTbots_med > 0.0) & (pTbots_med < 1e4))
        ax.plot(pTbots_med[mask], pDepths[mask]/1e3, "-",  color="tab:green")
        # plot result of moho temperature
        mask = ((pTtops_min > 0.0) & (pTtops_min < 1e4))
        ax.plot(pTtops_min[mask], pDepths[mask]/1e3, "--", label="surface T", color="tab:blue")
        mask = ((pTtops_max > 0.0) & (pTtops_max < 1e4))
        ax.plot(pTtops_max[mask], pDepths[mask]/1e3, "--", color="tab:blue")
        mask = ((pTtops_med > 0.0) & (pTtops_med < 1e4))
        ax.plot(pTtops_med[mask], pDepths[mask]/1e3, "-",  color="tab:blue")
        ax.set_xlim(xlims)
        ax.set_ylim([ylims[0]/1e3, ylims[1]/1e3])
        ax.set_ylabel("Depth (km)")
        if use_degree:
            ax.set_xlabel("Temperature (C)")
        else:
            ax.set_xlabel("Temperature (K)")

    # todo_eclogite
    def PlotEclogite(self, **kwargs):
        '''
        Plot the temperature profile for a case by assembling results
        from individual steps
            kwargs(dict):
                axis - a matplotlib axis
                debug - print debug message
                time_range - range of the time for plotting the temperature
        '''
        ax = kwargs.get('axis', None)
        use_degree = kwargs.get('use_degree', False)
        p_to_depth = 3.3/100 # 100 km 3.3 GPa
        file_stern_2001 = os.path.join(LEGACY_FILE_DIR, 'reference', 'eclogite_stern_2001.txt')
        assert(os.path.isfile(file_stern_2001))
        file_hu_2022 = os.path.join(LEGACY_FILE_DIR, 'reference', 'eclogite_hernandez-uribe_2022.txt')
        assert(os.path.isfile(file_hu_2022))
        # data from stern 2001
        self.ReadHeader(file_stern_2001)
        self.ReadData(file_stern_2001)
        col_T = self.header['temperature']['col']
        col_depth = self.header['depth']['col']
        Ts = self.data[:, col_T] # C
        depths = self.data[:, col_depth]
        if not use_degree:
            Ts += 273.0 # K
        ax.plot(Ts, depths, 'k-.', label='stern_2001')
        # data from the Hernandez-uribe_2022
        self.ReadHeader(file_hu_2022)
        self.ReadData(file_hu_2022)
        col_T = self.header['temperature']['col']
        col_P = self.header['pressure']['col']
        Ts = self.data[:, col_T] # C
        depths = self.data[:, col_P] / p_to_depth
        if not use_degree:
            Ts += 273.0 # K
        ax.plot(Ts, depths, 'c-.', label='Hernandez-uribe_2022')

    def FitTrenchT(self, case_dir, vtu_snapshot):
        '''
        fit the trench temperature
        '''
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,
                'BASH_OPTIONS.__init__: case prm file - %s cannot be read' % prm_file)
        with open(prm_file, 'r') as fin:
            idict = parse_parameters_to_dict(fin)
        potential_temperature = idict.get('Adiabatic surface temperature', 1673.0)
        # read the temperature and depth profile
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        vtu_step = max(0, int(vtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
        trench_T_file = os.path.join(case_dir, 'vtk_outputs', 'trench_T_%05d.txt' % (vtu_step))
        assert(os.path.isfile(trench_T_file))
        self.ReadHeader(trench_T_file)
        self.ReadData(trench_T_file)
        col_depth = self.header['depth']['col']
        col_T = self.header['T']['col']
        depths = self.data[:, col_depth]
        Ts = self.data[:, col_T]
        tFitFunc = T_FIT_FUNC(depths, Ts, potential_temperature=1573.0)
        x0 = [1.0]  # variables: age; scaling: 40 Ma
        res = minimize(tFitFunc.PlateModel, x0)
        age_myr = res.x[0] * 40.0 # Ma
        print('step: ', vtu_step, 'age_myr: ', age_myr)
        return age_myr

    def GetSlabMorph(self, case_dir):
        '''
        read the slab_morph file
        '''
        morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        assert(os.path.isfile(morph_file))
        self.ReadHeader(morph_file)
        self.ReadData(morph_file)
        col_sp_velocity = self.header['subducting_plate_velocity']['col']
        col_ov_velocity = self.header['overiding_plate_velocity']['col']
        col_dip = self.header['100km_dip']['col']
        col_time = self.header['time']['col']
        col_trench = self.header['trench']['col']
        times = self.data[:, col_time]
        sp_velocities = self.data[:, col_sp_velocity]
        ov_velocities = self.data[:, col_ov_velocity]
        trenches = self.data[:, col_trench]
        dips = self.data[:, col_dip]
        conv_velocities = sp_velocities - ov_velocities
        return times, sp_velocities, ov_velocities, dips, trenches

    def GetAgeTrench(self, case_dir, use_thermal=True, **kwargs):
        '''
        get the ages of the subducting plate at the trench
        Inputs:
            kwargs:
                time_interval - interval between steps
        '''
        time_interval = kwargs.get('time_interval', 0.5e6)
        Visit_Options = VISIT_OPTIONS(case_dir)
        Visit_Options.Interpret()
        if use_thermal:
            # use thermal option would fit the temperature at the trench for the individual steps
            # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
            available_pvtu_snapshots = Visit_Options.get_snaps_for_slab_morphology(time_interval=time_interval)
            print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug
            age_trenchs = []
            times = []
            for pvtu_snapshot in available_pvtu_snapshots:
                vtu_step = max(0, int(pvtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
                output_path = os.path.join(case_dir, "vtk_outputs")
                temp_file = os.path.join(output_path, "trench_T_%05d.txt" % (vtu_step))
                assert(os.access(temp_file, os.R_OK))
            for pvtu_snapshot in available_pvtu_snapshots:
                vtu_step = max(0, int(pvtu_snapshot) - int(Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT']))
                _time, _ = Visit_Options.get_time_and_step(vtu_step)
                times.append(_time)
                age_trench = self.FitTrenchT(case_dir, pvtu_snapshot)
                age_trenchs.append(age_trench)
            times = np.array(times)
            age_trenchs = np.array(age_trenchs)
        else:
            morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
            self.ReadHeader(morph_file)
            self.ReadData(morph_file)
            assert(os.path.isfile(morph_file))
            # read the geometry & Ro
            prm_file = os.path.join(case_dir, 'output', 'original.prm')
            assert(os.access(prm_file, os.R_OK))
            self.ReadPrm(prm_file)
            # read parameters
            geometry = self.prm['Geometry model']['Model name']
            if geometry == 'chunk':
                Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
            elif geometry == 'box':
                Ro =  float(self.prm['Geometry model']['box']["Y extent"])
            # read the spreading velocity 
            wb_file = os.path.join(case_dir, 'case.wb')
            assert(os.path.isfile(wb_file))
            with open(wb_file, 'r') as fin:
                wb_dict = json.load(fin)
            i0 = find_wb_feature(wb_dict, 'Subducting plate')
            sp_dict = wb_dict['features'][i0]
            trench_ini = sp_dict['coordinates'][2][0] * np.pi / 180.0
            spreading_velocity = sp_dict['temperature models'][0]['spreading velocity']
            # read data in slab_morph.txt
            col_time = self.header['time']['col']
            col_trench = self.header['trench']['col']
            col_sp_velocity = self.header['subducting_plate_velocity']['col']
            times = self.data[:, col_time]
            trenchs = self.data[:, col_trench]
            sp_velocities = self.data[:, col_sp_velocity]
            # correction on the trench coordinates
            trench_correction = trench_ini - trenchs[0]
            trenchs = trenchs + trench_correction
            age_trenchs = []
            is_first = True
            dist = 0.0
            time_last = 0.0
            for i in range(times.size):
                _time = times[i]
                sp_velocity = sp_velocities[i]
                trench = trenchs[i]
                dtime = _time - time_last 
                if is_first:
                    is_first = False
                else:
                    dist += sp_velocity * dtime
                time_last = _time
                # first part is the initial age of the material at the trench
                if geometry == 'box':
                    age = (trench - dist) / spreading_velocity + _time
                elif geometry == 'chunk':
                    age = (trench * Ro - dist) / spreading_velocity + _time
                age_trenchs.append(age)
            age_trenchs = np.array(age_trenchs)
        return times, age_trenchs


    def PlotTrenchAge(self, case_dir, **kwargs):
        '''
        plot the age of the trench
        Inputs:
            kwargs:
                time_interval - interval between steps
        '''
        time_interval = kwargs.get('time_interval', 0.5e6)
        ax = kwargs.get('axis', None)
        use_thermal = kwargs.get('use_thermal', True)
        # options for slab temperature outputs
        times, age_trenchs = self.GetAgeTrench(case_dir, use_thermal, time_interval=time_interval)
        if use_thermal:
            figure_label = "age of the trench (%s)" % "thermally interpreted"
        else:
            figure_label = "age of the trench (%s)" % "motion reconstruction"
        ax.plot(times/1e6, age_trenchs, label=figure_label)
        ax.set_xlabel("Time (Ma)")
        ax.set_ylabel("Trench Age (Ma)")
    
    def PlotThermalParameter(self, case_dir, **kwargs):
        '''
        plot the age of the trench
        Inputs:
            kwargs:
                time_interval - interval between steps
                time_stable - begining of the stable subduction
                plot_velocity - plot the velocity alongside the thermal parameter
        '''
        time_interval = kwargs.get('time_interval', 0.5e6)
        time_stable = kwargs.get('time_stable', None)
        ax = kwargs.get('axis', None)
        use_thermal = kwargs.get('use_thermal', True)
        plot_velocity = kwargs.get('plot_velocity', False)
        plot_dip = kwargs.get('plot_dip', False)
        # options for slab temperature outputs
        if use_thermal:
            # not implemented yet, the array of the age_trenchs imported this way could
            # be different in size from other arrays
            raise NotImplementedError()
        _, age_trenchs = self.GetAgeTrench(case_dir, use_thermal, time_interval=time_interval)
        times, sp_velocities, ov_velocities, dips, _ = self.GetSlabMorph(case_dir)
        conv_velocities = sp_velocities - ov_velocities
        thermal_parameters = age_trenchs * conv_velocities * np.sin(dips)
        if use_thermal:
            figure_label = "thermal parameter (%s)" % "thermally interpreted"
        else:
            figure_label = "thermal parameter (%s)" % "motion reconstruction"
        ax.plot(times/1e6, thermal_parameters/1e3, label=figure_label)
        ax.set_xlabel("Time (Ma)")
        ax.set_ylabel("Thermal Parameter (km)", color='tab:blue')
        if time_stable is not None:
            # focus on the range of thermal parameter in the stable regem
            mask = (times > time_stable)
            # the ValueError would be induced if the model hasn't reached
            # the stage of stable subduction
            try:
                ymax = np.ceil(np.max(thermal_parameters[mask]) / 1e6) * 1e6
                ymin = np.floor(np.min(thermal_parameters[mask]) / 1e6) * 1e6
                ax.set_ylim([ymin/1e3, ymax/1e3])
            except ValueError:
                pass
        # converging velocity
        if plot_velocity:
            ax1 = ax.twinx()
            ax1.plot(times/1e6, conv_velocities/1e3, '--', label="conv velocity", color='tab:green')
            ax1.set_ylabel("Velocity (m/yr)", color='tab:green')
            if time_stable is not None:
                # focus on the range of thermal parameter in the stable regem
                mask = (times > time_stable)
                try:
                    ymax1 = np.ceil(np.max(conv_velocities[mask]) / 1e-3) * 1e-3
                    ymin1 = np.floor(np.min(conv_velocities[mask]) / 1e-3) * 1e-3
                    ax1.set_ylim([ymin1/1e3, ymax1/1e3])
                except ValueError:
                    pass
        if plot_dip:
            ax2 = ax.twinx()
            ax2.plot(times/1e6, np.sin(dips), '--', label="sin(dip angle)", color='tab:red')
            ax2.set_ylabel("sin(dip angle)", color='tab:red')
            if time_stable is not None:
                # focus on the range of thermal parameter in the stable regem
                mask = (times > time_stable)
                try:
                    ymax2 = np.ceil(np.max(np.sin(dips[mask])) / 1e-2) * 1e-2
                    ymin2 = np.floor(np.min(np.sin(dips[mask])) / 1e-2) * 1e-2
                    ax2.set_ylim([ymin2, ymax2])
                except ValueError:
                    pass

    def GetTimeDepthTip(self, case_dir, query_depth, **kwargs):
        '''
        todo_t660
        Get the time the slab tip is at a certain depth
        Inputs:
            case_dir (str): case directory
        '''
        filename = kwargs.get("filename", "slab_morph.txt")

        assert(os.path.isdir(case_dir))
        morph_file = os.path.join(case_dir, 'vtk_outputs', filename)
        my_assert(os.path.isfile(morph_file), self.SlabMorphFileNotExistError, "%s is not a file." % morph_file)
        self.ReadHeader(morph_file)
        self.ReadData(morph_file)

        try: 
            col_time = self.header['time']['col']
            times = self.data[:, col_time]
            col_slab_depth = self.header['slab_depth']['col']
            slab_depths = self.data[:, col_slab_depth]
        except IndexError as e:
            # in case the file cannot be read, just return an invalid value
            return -1.0
            # raise SLABPLOT.MorphFileReadingError("Error while reading slab morphology file %s" % morph_file) from e
        query_time = -1.0
        for i in range(len(times)-1):
            _time = times[i]
            depth = slab_depths[i]
            next_depth = slab_depths[i+1]
            if depth < query_depth and next_depth > query_depth:
                next_time = times[i+1]
                query_time = (query_depth - depth) / (next_depth - depth) * next_time +\
                    (query_depth - next_depth) / (depth - next_depth) * _time
                
        return query_time

    def GetAverageVelocities(self, case_dir, t0, t1, **kwargs):
        '''
        Inputs:
            compute the average velocities between t0 and t1
        '''
        assert(os.path.isdir(case_dir))
        filename = kwargs.get("filename", "slab_morph.txt")

        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        my_assert(os.access(prm_file, os.R_OK), FileNotFoundError,\
        "prm file %s cannot be opened" % prm_file)
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        else:
            Ro = -1.0  # in this way, wrong is wrong
        
        # read morph file
        morph_file = os.path.join(case_dir, 'vtk_outputs', filename)
        my_assert(os.path.isfile(morph_file), self.SlabMorphFileNotExistError, "%s is not a file." % morph_file)
        self.ReadHeader(morph_file)
        self.ReadData(morph_file)
        
        # assign initial values 
        V_sink_avg = -1.0
        V_plate_avg = -1.0
        V_ov_plate_avg = -1.0
        V_trench_avg = -1.0

        try: 
            col_time = self.header['time']['col']
            times = self.data[:, col_time]
        except IndexError:
            return V_sink_avg, V_plate_avg, V_ov_plate_avg, V_trench_avg
            
        # if the time range is invalid (t1 should be bigger), return a state of -2.0 
        if (t1 <= t0):
            V_sink_avg = -2.0
            V_plate_avg = -2.0
            V_ov_plate_avg = -2.0
            V_trench_avg = -2.0
            return V_sink_avg, V_plate_avg, V_ov_plate_avg, V_trench_avg
        
        # calculate the velocity is suitable value of t1 is provided
        if t1 < times[times.size-1]:
            col_trench = self.header['trench']['col']
            trenches = self.data[:, col_trench]
            col_slab_depth = self.header['slab_depth']['col']
            slab_depths = self.data[:, col_slab_depth]
            col_sp_velocity = self.header['subducting_plate_velocity']['col']
            col_op_velocity = self.header['overiding_plate_velocity']['col']
            sp_velocities = self.data[:, col_sp_velocity]
            op_velocities = self.data[:, col_op_velocity]
    
            # trench migration 
            trench0 = np.interp(t0, times, trenches)
            trench1 = np.interp(t1, times, trenches)
            trenches_migration_length = 0.0
            if geometry == "chunk":
                trenches_migration_length = (trench1 - trench0) * Ro  # length of migration
            elif geometry == 'box':
                trenches_migration_length = trench1 - trench0
            V_trench_avg = trenches_migration_length / (t1 - t0)
    
            # slab depth 
            slab_depth0 = np.interp(t0, times, slab_depths)
            slab_depth1 = np.interp(t1, times, slab_depths)
            V_sink_avg = (slab_depth1 - slab_depth0) / (t1 - t0)
    
            # plate motion
            # first compute the velocity of the subducting plate
            v_temp = 0.0
            w_temp = 0.0
            for i in range(times.size-1):
                time0 = times[i]
                time1 = times[i+1]
                if time0 > t0 and time1 < t1:
                    v_temp += (time1 - time0) * (sp_velocities[i] + sp_velocities[i+1])/2.0
                    w_temp += (time1 - time0)
            V_plate_avg = v_temp / w_temp
            # then compute the velocity of the overiding plate 
            v_temp = 0.0
            w_temp = 0.0
            for i in range(times.size-1):
                time0 = times[i]
                time1 = times[i+1]
                if time0 > t0 and time1 < t1:
                    v_temp += (time1 - time0) * (op_velocities[i] + op_velocities[i+1])/2.0
                    w_temp += (time1 - time0)
            V_ov_plate_avg = v_temp / w_temp

        return V_sink_avg, V_plate_avg, V_ov_plate_avg, V_trench_avg


    class SlabMorphFileNotExistError(Exception):
        pass

    def write_csv(self, case_dir, **kwargs):
        '''
        using the pandas interface to convert to csv
        Inputs:
            case_dir (str): direction of the case
        '''
        # read data
        o_csv_path = kwargs.get("o_path", None)
        slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.txt')
        assert(os.path.isfile(slab_morph_file))
        self.ReadHeader(slab_morph_file)
        self.ReadData(slab_morph_file)
        if not self.HasData():
            print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
            return 1
        col_pvtu_step = self.header['pvtu_step']['col']
        col_pvtu_time = self.header['time']['col']
        col_pvtu_trench = self.header['trench']['col']
        col_pvtu_slab_depth = self.header['slab_depth']['col']
        pvtu_steps = self.data[:, col_pvtu_step]
        times = self.data[:, col_pvtu_time]
        trenches = self.data[:, col_pvtu_trench]
        # read the geometry & Ro
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
        elif geometry == 'box':
            Ro =  float(self.prm['Geometry model']['box']["Y extent"])
        # get the trench migration length 
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
        elif geometry == 'box':
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError('Invalid geometry')
        # collect data 
        slab_depthes = self.data[:, col_pvtu_slab_depth]
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)
        # assemble in an output
        o_csv_array = np.zeros([self.data.shape[0], 2])
        o_csv_array[:, 0] = times
        o_csv_array[:, 1] = trenches
        # uses a default path in vtk_outputs if no option is giving
        if o_csv_path is None:
            o_csv_path = os.path.join(case_dir, 'vtk_outputs', 'slab_morph.csv')
        # export
        # TODO: add field names and write R functions to parse the result
        df = pd.DataFrame(o_csv_array)
        df.to_csv(o_csv_path)


class SLABMATERIAL(LINEARPLOT): 
    '''
    A class defined to plot the Slab materials (crust and harzburgite)
    '''
    def __init__(self, _name):
        LINEARPLOT.__init__(self, _name)
        self.ha_reader =  DEPTH_AVERAGE_PLOT('DepthAverage') # reader of the horizontal average file

    def ReadFile(self, case_dir):
        '''
        Inputs:
        '''
        # read inputs
        prm_file = os.path.join(case_dir, 'output', 'original.prm')
        assert(os.access(prm_file, os.R_OK))
        self.ReadPrm(prm_file)
        # read parameters
        geometry = self.prm['Geometry model']['Model name']
        if geometry == 'chunk':
            self.Ro = float(self.prm['Geometry model']['Chunk']['Chunk outer radius'])
            self.geometry_extent = float(self.prm['Geometry model']['Chunk']['Chunk maximum longitude'])\
                - float(self.prm['Geometry model']['Chunk']["Chunk minimum longitude"])
            self.geometry_input = "cartesian" # this is the input that goes into the integretion function
        elif geometry == 'box':
            self.Ro =  float(self.prm['Geometry model']['box']["Y extent"])
            self.geometry_extent =  float(self.prm['Geometry model']['box']["X extent"])
            self.geometry_input = "spherical"
        else:
            return ValueError("%s: geometry must be chunk or box" % func_name())
        # read data
        case_output_dir = os.path.join(case_dir, 'output')
        depth_average_file = os.path.join(case_output_dir, 'depth_average.txt')
        assert(os.access(depth_average_file, os.R_OK))
        self.ha_reader.ReadHeader(depth_average_file)  # inteprate header information
        self.ha_reader.ReadData(depth_average_file)  # read data
        # self.ha_reader.ManageUnits()  # mange unit to output
        self.ha_reader.SplitTimeStep()  # split time step data
   
   
    def PlotSlabMaterial(self, _time, ax):
        '''
        plot the slab material
        Inputs:
            ax: an axis is passed in for plotting
        '''
        data_list0, step = self.ha_reader.ExportDataByTime(_time, ["depth", "spcrust", "spharz"])
        depths = data_list0[:, 0]
        spcrusts = data_list0[:, 1]
        spcrust_integretions, spcrust_segmentations = self.ha_reader.GetIntegrateArray(_time, "spcrust",\
            2, self.geometry_input, self.geometry_extent, Ro=self.Ro)
        # plot
        mask = (spcrusts > 0.0)
        # ax.semilogx(spcrusts[mask], depths[mask]/1e3, 'b', label="t = %.2f Myr" % (_time / 1e6)) # plot
        ax.semilogx(spcrust_segmentations, depths/1e3, 'b', label="t = %.2f Myr" % (_time / 1e6)) # plot
        ax.invert_yaxis()
        ax.set_xlabel("Crust Material in Segments (km^2)")
        ax.set_ylabel("Depth (km)")
        ax.set_xlim([spcrust_segmentations[0]/1e3, spcrust_segmentations[0]*5.0])

    
    def PlotMaterialRate(self, _time, ax, **kwargs):
        '''
        plot rate of tranform
        Inputs:
            ax: an axis is passed in for plotting
        '''
        # need results from two adjacent steps
        dt = kwargs.get('dt', 0.5e6)
        # read data from this step
        data_list, step = self.ha_reader.ExportDataByTime(_time, ["depth", "spcrust", "spharz"])
        depths = data_list[:, 0]
        # t_last: last step to compute the rate of material transformation
        if _time < dt:
            t_last = 0.0
        else:
            t_last = _time - dt
        spcrust_integretions, spcrust_segmentations = self.ha_reader.GetIntegrateArray(_time, "spcrust" ,\
            2, self.geometry_input, self.geometry_extent, Ro=self.Ro)
        spcrust_integretions_last, spcrust_segmentations_last = self.ha_reader.GetIntegrateArray(t_last, "spcrust",\
            2, self.geometry_input, self.geometry_extent, Ro=self.Ro)
        # derive the rate of material transformation
        spcrust_below_depths = spcrust_integretions[-1] - spcrust_integretions
        spcrust_below_depths_last = spcrust_integretions_last[-1] - spcrust_integretions_last
        spcrust_transform_rate = (spcrust_below_depths - spcrust_below_depths_last)/dt
        # plot
        ax.semilogx(spcrust_transform_rate, depths/1e3, 'b', label="t = %.2f Myr" % (_time / 1e6)) # plot
        ax.invert_yaxis()
        ax.set_xlabel("Crust Material transformed (km^2/yr)")
        ax.set_ylabel("Depth (km)")
        ax.set_xlim([spcrust_transform_rate[0]/1e3, spcrust_transform_rate[0]*5.0])
        return step


class PC_OPT_BASE(JSON_OPT):
    def __init__(self):
        '''
        initiation of the class
        '''
        JSON_OPT.__init__(self)
        self.add_key("Path of the root directory", str, ["case_root"], '', nick="case_root")
        self.add_key("Relative Path of case directorys", list, ["cases"], ['foo'], nick="case_relative_paths")
        self.add_key("Width of one subplot. This is set to -1.0 by default. By that, the width is determined with the \"anchor\" plot within the sequence",\
             float, ["width"], -1.0, nick="width")
        self.add_key("Use relative path for the output directory", int, ["output directory", "relative"],\
        0, nick="output_dir_relative")
        self.add_key("Path for the output directory", str, ["output directory", "path"], '', nick="output_dir_path")
        self.n_cases = len(self.values[1])  # number of cases
        self.add_key("Time range in yr", list, ["time range"], [], nick="time_range")
    
    def check(self):
        '''
        check to see if these values make sense
        '''
        # check existence of cases
        case_root = self.values[0]
        case_relative_paths = self.values[1]
        for case_relative_path in case_relative_paths:
            case_path = os.path.join(var_subs(case_root), case_relative_path)
            if not os.path.isdir(case_path):
                raise FileExistsError("Directory %s doesn't exist." % case_path)
        # make the output directory if not existing
        output_dir_relative = self.values[3]
        output_dir_path = self.values[4]
        if output_dir_relative == 1:
            output_dir = os.path.join(var_subs(case_root), output_dir_path)
        else:
            output_dir = var_subs(output_dir_path)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # assert the time range is valid
        time_range = self.values[5]
        print("time_range: ", time_range)  # debug
        if len(time_range)>0:
            assert(len(time_range) == 2 and time_range[0] < time_range[1])
    
    def get_case_absolute_paths(self):
        '''
        return the absolute paths of cases
        '''
        case_root = self.values[0]
        case_relative_paths = self.values[1]
        case_absolute_paths = []
        for case_relative_path in case_relative_paths:
            case_path = os.path.join(var_subs(case_root), case_relative_path)
            case_absolute_paths.append(case_path)
        return case_absolute_paths

    def get_case_names(self):
        '''
        return the names of cases
        '''
        case_names = []
        for case_relative_path in case_relative_paths:
            case_name = os.path.basename(case_relative_path)
            case_names.append(case_name)
        return case_names

    def get_output_dir(self):
        '''
        return the output dir
        '''
        case_root = self.values[0]
        case_relative_paths = self.values[1]
        output_dir_relative = self.values[3]
        output_dir_path = self.values[4]
        if output_dir_relative == 1:
            output_dir = os.path.join(var_subs(case_root), output_dir_path)
        else:
            output_dir = var_subs(output_dir_path)
        return output_dir
    
    def get_color_json_output_path(self):
        '''
        return a path to output the color scheme to a json file
        '''
        output_dir = self.get_output_dir()
        color_json_output_path = os.path.join(output_dir, 'color.json')
        return color_json_output_path


class PC_MORPH_OPT(PC_OPT_BASE):
    '''
    Define a class to work with json files
    '''
    def __init__(self):
        '''
        Initiation, first perform parental class's initiation,
        then perform daughter class's initiation.
        '''
        PC_OPT_BASE.__init__(self)
        self.start = self.number_of_keys()
        self.add_key("time range", list, ["time range"], [], nick="time_range")
        self.add_key("trench position range", list, ["trench position range"], [], nick="tp_range")
        self.add_key("slab depth range", list, ["slab depth range"], [], nick="sd_range")

    def to_init(self):
        '''
        interfaces to the __init__ function
        '''
        case_absolute_paths = self.get_case_absolute_paths()
        return case_absolute_paths

    def to_call(self):
        '''
        interfaces to the __call__ function
        '''
        width = self.values[2]
        output_dir = self.get_output_dir()
        time_range = self.values[self.start]
        tp_range = self.values[self.start + 1]
        sd_range = self.values[self.start + 2]
        return width, output_dir, time_range, tp_range, sd_range


class PLOT_COMBINE():
    '''
    Combine separate figures to a bigger one.
    Also combine figures from individual cases to a bigger one.
    Attributes:
        cases (list): list of case paths
        plots (2d list): plots to convert
        n_cases (int): number of cases
        n_plots (int): number of plots
        title_height (int): height of the title
    '''
    def __init__(self, case_paths):
        '''
        Initiation, read in a list of cases
        Inputs:
            case_paths (list): a list of cases
        '''
        self.cases = case_paths
        self.n_cases = len(case_paths)
        assert(self.n_cases > 0)
        self.plots = [[] for i in range(len(self.cases))]
        self.title_height = 200  # height of the title
        pass
    
    def add_case(self, case_path):
        '''
        Add one case
        Inputs:
            case_path (str): the path of a case
        '''
        self.cases.append(case_path)
        self.plots.append([])  # also add a new list of plots
        pass
    
    def set_plots(self, plots):
        '''
        Add one plot
        Inputs:
            plots (list): a list of plots
        '''
        self.plots = plots
        self.n_plots = len(plots[0])
        pass

    def configure(self):
        '''
        configuration
        '''
        pass
    
    def get_total_size(self, width, _title, **kwargs):
        '''
        get the size of new image
        Return:
            locations (list of 2 list): locations of subimages in the new combined image
                This is where the upper-left corner of each figure is located. Index 0 in
                the first dimension is along the horizontal direction and index 1 is along the
                vertical direction.
            width: fixed width of a subimage
            _title (str or None) - title
        '''
        anchor = kwargs.get('anchor', 0)
        if width < 0.0:
            first_figure_path = os.path.join(self.cases[0], 'img', self.plots[0][anchor])
            my_assert(os.path.isfile(first_figure_path), FileNotFoundError,\
             "%s: file doesn't exist (%s)" % (func_name(), first_figure_path))
            image = Image.open(first_figure_path)
            width = image.size[0]
        locations = [[], []]
        total_size = []
        locations[0] = [i*width for i in range(self.n_cases+1)]
        if _title is None:
            locations[1].append(0)  # first one, location along the length dimension is right at the start
        else:
            locations[1].append(self.title_height)  # if there is a title, leave a little space
        for j in range(self.n_plots):
            _path = '' # initiation
            find = False
            for i in range(self.n_cases):
                # find an existing file for me
                _path = os.path.join(self.cases[i], 'img', self.plots[i][j])
                if os.path.isfile(_path):
                    find = True
                    break
            if find == False:
                # we could choose from raising an error or allow this
                # raise FileNotFoundError("No existing figure %s in all cases" % self.plots[0][j])
                length = 500  # this is the length of a vacant plot
            else:
                image = Image.open(_path)
                width_old = image.size[0]
                length_old = image.size[1]
                length = int(length_old * width / width_old) # rescale this figure
            locations[1].append(locations[1][j] + length)   # later one, add the previous location
        total_size.append(width*self.n_cases)
        total_size.append(locations[-1])
        return locations, width
        
    def draw_title(self, image, _title, if_include_case_names, w_locations):
        '''
        Draw title at the top of the image
        Inputs:
            _title (str) - title
            if_include_case_names (0 or 1) - If we include case names
            w_locations (list of int): locations along the width
        '''
        individual_figure_width = w_locations[1] - w_locations[0]
        if False:
            fnt_size = 40
        else:
            fnt_size = 40
        fnt0 = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", fnt_size)  # get a font
        d = ImageDraw.Draw(image)
        d.text((10,10), _title, font=fnt0, fill=(0, 0, 0))  # anchor option doesn't workf
        if if_include_case_names:
            for i in range(self.n_cases):
                case_name = os.path.basename(self.cases[i])
                str_len = fnt0.getsize(case_name)
                if str_len[0] > 0.9 * individual_figure_width:
                    # resize font with longer case name
                    fnt_size_re = int(fnt_size * 0.9 * individual_figure_width // str_len[0])
                    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", fnt_size_re)
                else:
                    fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", fnt_size)
                w_location = w_locations[i] + 10
                h_location = self.title_height / 2 + 10
                d.text((w_location,h_location), case_name, font=fnt, fill=(0, 0, 0))  # anchor option doesn't work
        
    def initiate_combined_plotting(self, shape, color_method, dump_color_to_json, **kwargs):
        '''
        Initiate options for combining plots
        Inputs:
            color_method: generated, list or check_first
        '''
        multiple_size = kwargs.get("multiple_size", 1)
        n_color_max = 5
        ni = shape[0]
        nj = shape[1]
        fig = plt.figure(tight_layout=True, figsize=[7*nj*multiple_size, 5*ni*multiple_size])
        gs = gridspec.GridSpec(ni, nj)
        colors_dict = {}
        if color_method == 'generated':
            colors_dict['max'] = n_color_max
            if self.n_cases > n_color_max:
                raise ValueError("max number of colors must be bigger than the number of cases")
            normalizer = [ float(i)/(n_color_max) for i in range(self.n_cases) ]
            colors = cm.rainbow(normalizer)
            for i in range(self.n_cases):
                case_name = os.path.basename(self.cases[i])
                colors_dict[case_name] = list(colors[i])
        elif color_method == 'list':
            colors_dict['max'] = 8
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
            for i in range(self.n_cases):
                case_name = os.path.basename(self.cases[i])
                colors_dict[case_name] = list(colors[i])
        elif color_method == 'check_first':
            colors = []
            if dump_color_to_json is not None:
                if os.path.isfile(dump_color_to_json):
                    with open(dump_color_to_json, 'r') as fin:
                        colors_dict = json.load(fin)
                else:
                    colors_dict['max'] = n_color_max
            # first loop to get the number of colors in the json file
            n_color_in_json = 0
            for i in range(self.n_cases):
                case_name = os.path.basename(self.cases[i])
                if case_name in colors_dict:
                    n_color_in_json += 1
            normalizer = [ float(i)/(n_color_max) for i in range(n_color_in_json, self.n_cases) ]
            new_colors = cm.rainbow(normalizer)
            # second loop to assign colors to new cases
            j = 0
            for i in range(self.n_cases):
                case_name = os.path.basename(self.cases[i])
                try:
                    colors.append(colors_dict[case_name])
                except KeyError:
                    colors.append(new_colors[j])
                    colors_dict[case_name] = list(colors[i])
                    j += 1
        else:
            raise ValueError('Not implemented')
        # dump a color file
        if dump_color_to_json is not None:
            assert(os.path.isdir(os.path.dirname(dump_color_to_json)))
            with open(dump_color_to_json, 'w') as fout:
                json.dump(colors_dict, fout)
            print("%s: dump color options: %s" % (func_name(), dump_color_to_json))
        return fig, gs, colors
    
    def __call__(self, width, anchor, output_dir, _title, if_include_case_names, _name, **kwargs):
        '''
        perform combination
        Inputs:
            sizes: (list of 2) - size of the plot
            output_dir: directory to output to
            _title (str or None) - title
            if_include_case_names (0 or 1) - If we include case names
            _name (str) - name of the plot
        '''
        save_pdf = kwargs.get("save_pdf", False)
        assert(os.path.isdir(output_dir))
        locations, width = self.get_total_size(width, _title, anchor=anchor)  # width is the width of a subplot
        image_size = [locations[0][-1], locations[1][-1]]
        # initiate
        new_image = Image.new('RGB',image_size,(250,250,250))
        if _title is not None:
            self.draw_title(new_image, _title,if_include_case_names, locations[0])
        for i in range(self.n_cases):
            for j in range(self.n_plots):
                plot_path = os.path.join(self.cases[i], 'img', self.plots[i][j])
                if os.path.isfile(plot_path):
                    image = Image.open(plot_path)
                    image = image.resize((width, locations[1][j+1] - locations[1][j]))  # resize to fit the spot
                else:
                    image = Image.new('RGB', (width, 500), (250,250,250)) # append a blank one
                new_image.paste(image, (locations[0][i], locations[1][j])) # paste image in place
        new_image_path = os.path.join(output_dir, '%s.png' % _name)
        print("%s: save figure: %s" % (func_name(), new_image_path))
        if save_pdf:
            new_pdf_path = os.path.join(output_dir, '%s.pdf' % _name)
            print("%s: save figure: %s" % (func_name(), new_pdf_path))
        new_image.save(new_image_path)
        return new_image_path



class PLOT_COMBINE_SLAB_MORPH(PLOT_COMBINE):
    '''
    Combine results from slab morphology
    '''
    def __init__(self, case_paths):
        PLOT_COMBINE.__init__(self, case_paths)
        UnitConvert = UNITCONVERT()
        self.MorphPlotter = SLABPLOT("plot_slab")
        pass

    def __call__(self, width, output_dir, time_range, tp_range, sd_range, **kwargs):
        '''
        perform combination
        Inputs:
            sizes: (list of 2) - size of the plot
            output_dir: directory to output to
        kwargs:
            color_method: use a list of color or the generated values
        '''
        multiple_size = kwargs.get("multiple_size", 1) # get the multiple size factor
        _name = "combine_morphology"
        _title = "Comparing slab morphology results"
        color_method = kwargs.get('color_method', 'list')
        dump_color_to_json = kwargs.get('dump_color_to_json', None)
        save_pdf = kwargs.get('save_pdf', False)
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        # initiate
        ni = 3  # number of plots along 1st and 2nd dimension
        nj = 2
        fig, gs, colors = self.initiate_combined_plotting((ni, nj), color_method, dump_color_to_json, multiple_size=multiple_size)
        case_names = []  # names of cases
        for i in range(self.n_cases):
            case_name = os.path.basename(self.cases[i])
            case_names.append(case_name)
        # plot trench position
        ax = fig.add_subplot(gs[1, 0])
        lns = None
        labs = None
        for i in range(self.n_cases):
            if i == 0:
                label = ['Trench Position', 'Slab Depth']
            else:
                label = [None, None]
            case_dir = self.cases[i]
            case_name = os.path.basename(case_dir)
            # plot results and combine
            lns_temp, labs_temp = self.MorphPlotter.PlotTrenchPosition(case_dir, time_range=time_range,\
            tp_range=tp_range, sd_range=sd_range, axis=ax, color=colors[i], label=label)
            if i == 0:
                lns = lns_temp  # record the lables at the start
                labs = labs_temp
            pass
        ax.legend(lns, labs)
        # plot trench velocity
        ax = fig.add_subplot(gs[2, 0])
        lns = None
        labs = None
        for i in range(self.n_cases):
            if i == 0:
                label_all = True
            else:
                label_all = False
            case_dir = self.cases[i]
            case_name = os.path.basename(case_dir)
            # plot results and combine
            self.MorphPlotter.PlotTrenchVelocity(case_dir, time_range=time_range,\
            tp_range=tp_range, sd_range=sd_range, axis=ax, color=colors[i], label_all=label_all)
        ax.legend()
        # plot trench velocity, zoom in
        ax = fig.add_subplot(gs[2, 1])
        for i in range(self.n_cases):
            case_dir = self.cases[i]
            case_name = os.path.basename(case_dir)
            # plot results and combine
            self.MorphPlotter.PlotTrenchVelocity(case_dir, time_range=time_range,\
            tp_range=tp_range, sd_range=sd_range, axis=ax, color=colors[i], label_all=False, fix_v_range=True)
        # plot the color labels
        ax = fig.add_subplot(gs[0, 0])
        PlotColorLabels(ax, case_names, colors)
        # generate figures
        fig_path = os.path.join(output_dir, '%s.png' % _name)
        print("%s: save figure: %s" % (func_name(), fig_path))
        plt.savefig(fig_path)
        if save_pdf == True:
            pdf_path = os.path.join(output_dir, '%s.pdf' % _name)
            print("%s: save figure: %s" % (func_name(), pdf_path))
            plt.savefig(pdf_path)
        return fig_path
    
def PlotColorLabels(ax, case_names, colors): 
    '''
    plot the color labels used for different cases
    '''
    labels = []
    patches = []
    for case_name in case_names:
        labels.append(case_name)
    for _color in colors:
        patches.append(mpatches.Patch(color=_color))
    ax.legend(patches, labels, loc='center', frameon=False)
    ax.axis("off")


def PlotSlabMaterialTime(case_dir, _time):
    '''
    Plot slab material
    '''
    # mkdir directory
    img_dir = os.path.join(case_dir, 'img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    img_sm_dir = os.path.join(img_dir,'slab_material')
    if not os.path.isdir(img_sm_dir):
        os.mkdir(img_sm_dir)
    # initiate plotter
    plotter_material = SLABMATERIAL('slab')
    plotter_material.ReadFile(case_dir)
    fig = plt.figure(tight_layout=True, figsize=(5, 10))
    gs = gridspec.GridSpec(2, 1)
    # plot composition
    ax = fig.add_subplot(gs[0, :])
    step = plotter_material.PlotSlabMaterial(_time, ax)
    ax.legend()
    # plot rate of transformation
    ax = fig.add_subplot(gs[1, :])
    step = plotter_material.PlotMaterialRate(_time, ax)
    ax.legend()
    fileout = os.path.join(img_sm_dir, "s%06d_t%.4e.png" % (step, _time))
    fig.savefig(fileout)
    print("%s: %s generated" % (func_name(), fileout))
    pass

def PlotTrenchAgeFromT(case_dir, **kwargs):
    '''
    plot the age of the trench, from the result of the thermal interpretation of the trench temperature
    Inputs:
        kwargs:
            time_interval - interval between steps
    '''
    use_thermal = True  # twik options between thermally interpretation and motion reconstruction
    # let the user check these options
    if use_thermal:
        # use thermal option would fit the temperature at the trench for the individual steps
        # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
        _continue = input("This option will plot the data in the vtk_outputs/trench_T.txt file, \
but will not generarte that file. Make sure all these files are updated, proceed (y/n)?")
        if _continue != 'y':
            print('abort')
            exit(0)
    else:
        _continue = input("This option requires the data in the vtk_outputs/slab_morph.txt file, \
but will not generarte that file. Make sure all these files are updated, proceed (y/n)?")
        if _continue != 'y':
            print('abort')
            exit(0) 
    time_interval = kwargs.get('time_interval', 0.5e6)
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    fig = plt.figure(tight_layout=True, figsize=(5, 5))
    gs = gridspec.GridSpec(1, 1)
    # plot composition
    plotter = SLABPLOT('trench_age_from_T')
    # 0. age
    ax = fig.add_subplot(gs[0, 0])
    plotter.PlotTrenchAge(case_dir, axis=ax, time_interval=time_interval, use_thermal=use_thermal)
    fig_path = os.path.join(img_dir, "trench_age_from_T.png") 
    fig.savefig(fig_path)
    print("%s: save figure %s" % (func_name(), fig_path))

def PlotTrenchThermalState(case_dir, **kwargs):
    '''
    plot the age of the trench
    Inputs:
        kwargs:
            time_interval - interval between steps
            silent - function would not ask user input to progress
    '''
    use_thermal = False  # twik options between thermally interpretation and motion reconstruction
    silent = kwargs.get('silent', False)
    # let the user check these options
    if not silent:
        if use_thermal:
            # use thermal option would fit the temperature at the trench for the individual steps
            # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
            _continue = input("This option will plot the data in the vtk_outputs/trench_T.txt file, \
    but will not generarte that file. Make sure all these files are updated, proceed (y/n)?")
            if _continue != 'y':
                print('abort')
                exit(0)
        else:
            _continue = input("This option requires the data in the vtk_outputs/slab_morph.txt file, \
    but will not generarte that file. Make sure all these files are updated, proceed (y/n)?")
            if _continue != 'y':
                print('abort')
                exit(0) 
    time_interval = kwargs.get('time_interval', 0.5e6)
    img_dir = os.path.join(case_dir, "img")
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    fig = plt.figure(tight_layout=True, figsize=(10, 15))
    gs = gridspec.GridSpec(3, 2)
    # plot composition
    plotter = SLABPLOT('trench_thermal_state')
    # 0. age
    ax = fig.add_subplot(gs[0, 0])
    plotter.PlotTrenchAge(case_dir, axis=ax, time_interval=time_interval, use_thermal=use_thermal)
    ax.legend()
    # 1. thermal parameter
    ax = fig.add_subplot(gs[1, 0])
    plotter.PlotThermalParameter(case_dir, axis=ax, time_interval=time_interval, use_thermal=use_thermal)
    ax.legend()
    # 2. thermal parameter, focusing on the stable subduction regem
    ax = fig.add_subplot(gs[2, 0])
    plotter.PlotThermalParameter(case_dir, axis=ax, time_interval=time_interval, use_thermal=use_thermal, time_stable=10e6)
    ax.legend()
    # 3. thermal parameter with the subducting plate velocity
    ax = fig.add_subplot(gs[0, 1])
    plotter.PlotThermalParameter(case_dir, axis=ax, time_interval=time_interval,\
    use_thermal=use_thermal, time_stable=10e6, plot_velocity=True)
    ax.legend()
    # 4. thermal parameter with the dip angle
    ax = fig.add_subplot(gs[1, 1])
    plotter.PlotThermalParameter(case_dir, axis=ax, time_interval=time_interval,\
    use_thermal=use_thermal, time_stable=10e6, plot_dip=True)
    ax.legend()

    fig_path = os.path.join(img_dir, "trench_thermal_state.png") 
    fig.savefig(fig_path)
    print("%s: save figure %s" % (func_name(), fig_path))


def PlotMorphAnimeCombined(case_dir, **kwargs):
    '''
    plot slab morphology for making animation
    Inputs:
        case_dir: case directory
        kwargs:
            time_interval - time_interval of plotting
    '''
    # initiate
    time_interval = kwargs.get("time_interval", 5e5)
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret()
    # SLABPLOT object 
    SlabPlot = SLABPLOT('slab')
    
    # call get_snaps_for_slab_morphology, this prepare the snaps with a time interval in between.
    times= Visit_Options.get_times_for_slab_morphology(time_interval=time_interval)
    # print("available_pvtu_snapshots: ", available_pvtu_snapshots)  # debug 
    for _time in times:
        SlabPlot.PlotMorphAnime(case_dir, time=_time)


def PlotTrenchDifferences2dInter1Ma(SlabPlot, case_dir, **kwargs):
    '''
    plot the differences in the trench location since model started (trench migration)
    overlay the curve on an existing axis.
    This function is created for combining results with those from the 3d cases
    '''
    # initiate plot
    _color = kwargs.get('color', "c")
    ax = kwargs.get('axis', None) # for trench position
    ax_twinx = kwargs.get("axis_twinx", None) # for slab depth
    if ax == None:
        raise ValueError("Not implemented")
    # path
    img_dir = os.path.join(case_dir, 'img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    morph_dir = os.path.join(img_dir, 'morphology')
    if not os.path.isdir(morph_dir):
        os.mkdir(morph_dir)
    # read inputs
    prm_file = os.path.join(case_dir, 'output', 'original.prm')
    assert(os.access(prm_file, os.R_OK))
    SlabPlot.ReadPrm(prm_file)
    # read parameters
    geometry = SlabPlot.prm['Geometry model']['Model name']
    if geometry == 'chunk':
        Ro = float(SlabPlot.prm['Geometry model']['Chunk']['Chunk outer radius'])
    else:
        Ro = None
    # read data
    slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph_t1.00e+06.txt')
    assert(os.path.isfile(slab_morph_file))
    SlabPlot.ReadHeader(slab_morph_file)
    SlabPlot.ReadData(slab_morph_file)
    if not SlabPlot.HasData():
        print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
        return 1
    col_pvtu_step = SlabPlot.header['pvtu_step']['col']
    col_pvtu_time = SlabPlot.header['time']['col']
    col_pvtu_trench = SlabPlot.header['trench']['col']
    col_pvtu_slab_depth = SlabPlot.header['slab_depth']['col']
    col_pvtu_sp_v = SlabPlot.header['subducting_plate_velocity']['col']
    col_pvtu_ov_v = SlabPlot.header['overiding_plate_velocity']['col']
    pvtu_steps = SlabPlot.data[:, col_pvtu_step]
    times = SlabPlot.data[:, col_pvtu_time]
    trenches = SlabPlot.data[:, col_pvtu_trench]
    slab_depths = SlabPlot.data[:, col_pvtu_slab_depth]
    time_interval = times[1] - times[0]
    if time_interval < 0.5e6:
        warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)
    if geometry == "chunk":
        trenches_migration_length = (trenches - trenches[0]) * Ro  # length of migration
    elif geometry == 'box':
        trenches_migration_length = trenches - trenches[0]
    else:
        raise ValueError('Invalid geometry')
    # get_slab_dimensions_2(x, y, Ro, is_chunk)
    ax.plot(times/1e6, trenches_migration_length/1e3, color=_color, label = "2d")
    if ax_twinx is not None:
        ax_twinx.plot(times/1e6, slab_depths/1e3, '--', color=_color)


def PlotSlabDip100km2dInter1Ma(SlabPlot, case_dir, **kwargs):
    '''
    plot the differences in the trench location since model started (trench migration)
    overlay the curve on an existing axis.
    This function is created for combining results with those from the 3d cases
    '''
    # initiate plot
    _color = kwargs.get('color', "c")
    ax = kwargs.get('axis', None) # for trench position
    ax_twinx = kwargs.get("axis_twinx", None) # for slab depth
    if ax == None:
        raise ValueError("Not implemented")
    # path
    img_dir = os.path.join(case_dir, 'img')
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    morph_dir = os.path.join(img_dir, 'morphology')
    if not os.path.isdir(morph_dir):
        os.mkdir(morph_dir)
    # read inputs
    prm_file = os.path.join(case_dir, 'output', 'original.prm')
    assert(os.access(prm_file, os.R_OK))
    SlabPlot.ReadPrm(prm_file)
    # read parameters
    geometry = SlabPlot.prm['Geometry model']['Model name']
    # if geometry == 'chunk':
    #     Ro = float(SlabPlot.prm['Geometry model']['Chunk']['Chunk outer radius'])
    # else:
    #     Ro = None
    # read data
    slab_morph_file = os.path.join(case_dir, 'vtk_outputs', 'slab_morph_t1.00e+06.txt')
    assert(os.path.isfile(slab_morph_file))
    SlabPlot.ReadHeader(slab_morph_file)
    SlabPlot.ReadData(slab_morph_file)
    if not SlabPlot.HasData():
        print("PlotMorph: file %s doesn't contain data" % slab_morph_file)
        return 1
    col_pvtu_step = SlabPlot.header['pvtu_step']['col']
    col_pvtu_time = SlabPlot.header['time']['col']
    col_pvtu_trench = SlabPlot.header['trench']['col']
    col_pvtu_slab_depth = SlabPlot.header['slab_depth']['col']
    # col_pvtu_sp_v = SlabPlot.header['subducting_plate_velocity']['col']
    # col_pvtu_ov_v = SlabPlot.header['overiding_plate_velocity']['col']
    col_100km_dip = SlabPlot.header['100km_dip']['col']
    # pvtu_steps = SlabPlot.data[:, col_pvtu_step]
    times = SlabPlot.data[:, col_pvtu_time]
    trenches = SlabPlot.data[:, col_pvtu_trench]
    # slab_depths = SlabPlot.data[:, col_pvtu_slab_depth]
    dip_100kms = SlabPlot.data[:, col_100km_dip]
    time_interval = times[1] - times[0]
    if time_interval < 0.5e6:
        warnings.warn("Time intervals smaller than 0.5e6 may cause vabriation in the velocity (get %.4e)" % time_interval)

    # Apply a univariate spline to smooth the dip angles
    spline = UnivariateSpline(times / 1e6, dip_100kms, s=0)  # s=0 means interpolation without smoothing
    times_new = np.linspace(times.min(), times.max(), 1000)
    dips_splined = spline(times_new / 1e6)
    # get_slab_dimensions_2(x, y, Ro, is_chunk)
    ax.plot(times/1e6, dip_100kms * 180.0 / np.pi, color=_color, label = "2d")
    ax.plot(times_new/1e6, dips_splined * 180.0 / np.pi, "-.", color=_color, label = "2d")
    # if ax_twinx is not None:
    #     ax_twinx.plot(times/1e6, slab_depths/1e3, '--', color=_color)


def GetSlabDipAt660(case_dir, **kwargs):
    '''
    Get the slab dip angle when reaching 660
    '''
    IndexByValue = lambda array_1d, val: np.argmin(abs(array_1d - val))
    Resample1d = lambda array_1d, n: array_1d[np.ix_(range(0, array_1d.size, n))]

    query_depth = kwargs.get("query_depth", 660e3)
    dip_angle_depth_lookup = kwargs.get("dip_angle_depth_lookup", 660e3)
    dip_angle_depth_lookup_interval = kwargs.get("dip_angle_depth_lookup_interval", 60e3)
    
    Visit_Options = VISIT_OPTIONS(case_dir)
    Visit_Options.Interpret() 
    
    slab_morph_path = os.path.join(case_dir, "vtk_outputs", "slab_morph_t1.00e+05.txt")
    my_assert(os.path.isfile(slab_morph_path), SLABPLOT.SlabMorphFileNotExistError, "File %s doesn't exist" % slab_morph_path)
    
    data = np.loadtxt(slab_morph_path)
    steps = data[:, 1]
    times = data[:, 2]
    trenches = data[:, 3]
    slab_depths = data[:, 4]
    
    # time of slab tip reaching _depth km and the index in the list
    sfunc = interp1d(slab_depths, times, assume_sorted=True)
    t_depth = sfunc(query_depth)
    i_depth = IndexByValue(times, t_depth)
    step_depth = steps[i_depth]

    # figure out the snapshot to analyze 
    available_pvtu_snapshots = Visit_Options.get_snaps_for_slab_morphology(time_interval=0.1e6)
    available_pvtu_times = Visit_Options.get_times_for_slab_morphology(time_interval=0.1e6)
    # available_pvtu_times, available_pvtu_snapshots = Visit_Options.get_snaps_for_slab_morphology_outputs(time_interval=0.1e6)
    id = IndexByValue(available_pvtu_times, t_depth)
    vtu_snapshot = available_pvtu_snapshots[id]

    # get the dip angle at _depth km 
    vtu_step, outputs = SlabMorphology_dual_mdd(case_dir, vtu_snapshot, dip_angle_depth_lookup=dip_angle_depth_lookup, dip_angle_depth_lookup_interval=dip_angle_depth_lookup_interval)
    o_list = []
    for entry in outputs.split(' '):
        if entry not in ["", "\n"]:
            o_list.append(entry)
    dip_depth = float(o_list[-1])
    return dip_depth


def get_slab_dimensions_2(x, y, Ro, is_chunk):
    '''
    Derives the length along the three dimensions of a subducting slab.

    Inputs:
        x (float): x-coordinate of the slab point.
        y (float): y-coordinate of the slab point.
        z (float): z-coordinate of the slab point.
        Ro (float): Outer radius of the spherical domain.
        is_chunk (bool): Flag indicating whether the geometry is a spherical chunk.

    Returns:
        tuple: A tuple containing (r, w, l):
            - r (float): Radius or z-coordinate depending on whether the geometry is a chunk.
            - w (float): Width of the slab in the y-dimension, or converted width for chunk geometry.
            - l (float): Length of the slab in the x-dimension, or converted length for chunk geometry.
    
    Description:
        - For chunk geometries, converts Cartesian coordinates to spherical coordinates and calculates
          width and length using the outer radius Ro and spherical angles.
        - For non-chunk geometries, returns the z, x, and y coordinates directly as radius, length, and width.
    '''
    if is_chunk:
        # Convert Cartesian coordinates to spherical coordinates for chunk geometry
        r, th1, ph1 = cart2sph(x, y, 0.0)
        w = 0.0
        l = Ro * ph1  # Calculate length using the spherical angle ph1
    else:
        # For non-chunk geometry, use Cartesian coordinates directly
        r = y
        w = 0.0
        l = x 

    return r, w, l


# todo_o_env
def SlabEnvelopRetrivePoints(local_dir: str, _time: float, Visit_Options: object, depths: float, **kwargs: dict) -> tuple:
    '''
    Retrieves the point of the slab envelop at give depth

    Parameters:
        local_dir (str): Directory where output files are located.
        _time (float): The time at which to retrieve the heat flow profile.
        Visit_Options (object): An object containing options for retrieving data.
        depth: the depth of the point to retrive
        kwargs (dict): Additional keyword arguments; accepts 'phi_diff' (float, default: 5.0).

    Returns:
        tuple: A tuple containing two masked arrays: heat fluxes and corresponding phi values.
    '''
    Ro = Visit_Options.options["OUTER_RADIUS"]

    _time1, timestep, vtu_step = Visit_Options.get_timestep_by_time(_time)
    filein = os.path.join(local_dir, "vtk_outputs", "slab_env_%05d.txt" % vtu_step)
    my_assert(os.path.isfile(filein), FileExistsError, "%s: %s doesn't exist" % (func_name(), filein))

    # Retrieve slab envelops 
    # and interpolate the points based on the give depths
    data = np.loadtxt(filein)
    X1 = data[:, 2]
    Y1 = data[:, 3]

    # L1: box geometry - X dimension; chunk geometry - phi dimension
    if Visit_Options.options["GEOMETRY"] == "box":
        Ys = np.interp(Ro-depths, Y1,X1) 
        return Ys
    elif Visit_Options.options["GEOMETRY"] == "chunk":
        R0, _, Phi0 = cart2sph(X1,Y1,np.zeros(X1.shape))
        Phis = np.interp(depths, Ro-R0, Phi0)
        return Phis
    else:
        raise NotImplementedError

def minimum_distance_array(a, x0, y0, z0):
    """
    Calculate the minimum distance from a reference point (x0, y0, z0) 
    to the points in array 'a', where 'a' is a numpy array of shape (n, 3).
    
    Parameters:
        a (numpy.ndarray): Array of shape (n, 3), where each row is [x, y, z] coordinates.
        x0 (float): x-coordinate of the reference point.
        y0 (float): y-coordinate of the reference point.
        z0 (float): z-coordinate of the reference point.
    
    Returns:
        float: The minimum distance from (x0, y0, z0) to the points in 'a'.
    """
    # Calculate the squared distances to avoid unnecessary square roots
    squared_distances = (a[:, 0] - x0)**2 + (a[:, 1] - y0)**2 + (a[:, 2] - z0)**2
    
    # Find the minimum squared distance and take its square root to get the actual distance
    min_distance = np.sqrt(np.min(squared_distances))
    min_index = np.argmin(squared_distances)

    return min_index, min_distance



class PARALLEL_WRAPPER_FOR_VTK():
    '''
    a parallel wrapper for analyzing slab morphology
    Attributes:
        name(str): name of this plot
        case_dir (str): case directory
        module (a function): a function to use for plotting
        last_pvtu_step (str): restart from this step, as there was previous results
        if_rewrite (True or False): rewrite previous results if this is true
        pvtu_steps (list of int): record the steps
        outputs (list of str): outputs
    '''
    def __init__(self, name, module, **kwargs):
        '''
        Initiation
        Inputs:
            name(str): name of this plot
            module (a function): a function to use for plotting
            kwargs (dict)
                last_pvtu_step
                if_rewrite
        '''
        self.name = name
        self.module = module
        self.last_pvtu_step = kwargs.get('last_pvtu_step', -1)
        self.if_rewrite = kwargs.get('if_rewrite', False)
        self.do_assemble = kwargs.get('assemble', True)
        self.kwargs = kwargs
        self.pvtu_steps = []
        self.outputs = []
        pass

    def configure(self, case_dir):
        '''
        configure
        Inputs:
            case_dir (str): case diretory to assign
        '''
        os.path.isdir(case_dir)
        self.case_dir = case_dir
    
    def __call__(self, pvtu_step):
        '''
        call function
        Inputs:
            pvtu_step (int): the step to plot
        '''
        expect_result_file = os.path.join(self.case_dir, 'vtk_outputs', '%s_s%06d' % (self.name, pvtu_step))
        if pvtu_step <= self.last_pvtu_step and not self.if_rewrite:
            # skip existing steps
            return 0
        if os.path.isfile(expect_result_file) and not self.if_rewrite:
            # load file content
            print("%s: previous result exists(%s), load" % (func_name(), expect_result_file))
            with open(expect_result_file, 'r') as fin:
                pvtu_step = int(fin.readline())
                output = fin.readline()
        else:
            if self.do_assemble:    
                # here the outputs from individual steps are combined together
                if pvtu_step == 0:
                    # start new file with the 0th step
                    pvtu_step, output = self.module(self.case_dir, pvtu_step, new=True, **self.kwargs)
                else:
                    pvtu_step, output = self.module(self.case_dir, pvtu_step, **self.kwargs)
                with open(expect_result_file, 'w') as fout:
                    fout.write('%d\n' % pvtu_step)
                    fout.write(output)
                print("%s: pvtu_step - %d, output - %s" % (func_name(), pvtu_step, output))
                # self.pvtu_steps.append(pvtu_step) # append to data
                self.outputs.append(output)
            else: 
                # otherwise, just call the module for each steps
                if pvtu_step == 0:
                    # start new file with the 0th step
                    self.module(self.case_dir, pvtu_step, new=True, **self.kwargs)
                else:
                    self.module(self.case_dir, pvtu_step, **self.kwargs)
        return 0
    
    def assemble(self):
        '''
        Returns:
            pvtu_steps
            outputs
        '''
        assert(len(self.pvtu_steps) == len(self.outputs))
        length = len(self.pvtu_steps)
        # bubble sort
        for i in range(length):
            for j in range(i+1, length):
                if self.pvtu_steps[j] < self.pvtu_steps[i]:
                    temp = self.pvtu_steps[i]
                    self.pvtu_steps[i] = self.pvtu_steps[j]
                    self.pvtu_steps[j] = temp
                    temp = self.outputs[i]
                    self.outputs[i] = self.outputs[j]
                    self.outputs[j] = temp
        return self.pvtu_steps, self.outputs
    
    def assemble_parallel(self):
        '''
        Returns:
            pvtu_steps
            outputs
        '''
        for pvtu_step in self.pvtu_steps:
            expect_result_file = os.path.join(self.case_dir, 'vtk_outputs', '%s_s%06d' % (self.name, pvtu_step))
            assert(os.path.isfile(expect_result_file))
            with open(expect_result_file, 'r') as fin:
                fin.readline()
                output = fin.readline()
                self.outputs.append(output)
        return self.pvtu_steps, self.outputs
    
    def set_pvtu_steps(self, pvtu_steps):
        '''
        set_pvtu_steps
        Inputs:
            pvtu_steps(list of int): step to look for
        '''
        self.pvtu_steps = pvtu_steps
    
    def delete_temp_files(self, pvtu_steps):
        '''
        delete temp files
        Inputs:
            pvtu_steps(list of int): step to look for
        '''
        print('delete temp files')
        for pvtu_step in pvtu_steps:
            expect_result_file = os.path.join(self.case_dir, 'vtk_outputs', '%s_s%06d' % (self.name, pvtu_step))
            if os.path.isfile(expect_result_file):
                os.remove(expect_result_file)
    
    def clear(self):
        '''
        clear data
        '''
        self.pvtu_steps = []
        self.outputs = []
