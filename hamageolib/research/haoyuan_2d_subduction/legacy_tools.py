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
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageDraw, ImageFont
from cmcrameri import cm as ccm 
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.optimize import minimize
from joblib import Parallel, delayed
from shutil import rmtree, copy2, copytree
from difflib import unified_diff
from copy import deepcopy
from .legacy_utilities import JsonOptions, ReadHeader, CODESUB, cart2sph, SphBound, clamp, ggr2cart, point2dist, UNITCONVERT, ReadHeader2,\
ggr2cart, var_subs, JSON_OPT, string2list, re_neat_word, ReadDashOptions, insert_dict_after
from ...utils.exception_handler import my_assert
from ...utils.handy_shortcuts_haoyuan import func_name
from ...utils.dealii_param_parser import parse_parameters_to_dict, save_parameters_from_dict
from ...utils.world_builder_param_parser import find_wb_feature
from ...utils.geometry_utilities import offset_profile, compute_pairwise_distances

JSON_FILE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "legacy_json_files")
LEGACY_FILE_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "legacy_files")
RESULT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "dtemp", "rheology_results")
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")

if not os.path.isdir(RESULT_DIR):
    os.mkdir(RESULT_DIR)

# constants in the file    
R = 8.314
year = 365 * 24 * 3600.0  # yr to s

class WBFeatureNotFoundError(Exception):
    pass

def FindWBFeatures(Inputs_wb, key):
    '''
    find index of feature in a world builder inputs by its key
    Inputs:
        Inputs_wb (dict): World buider dictionary
        key (str): name of the feature
    '''
    assert(type(Inputs_wb) == dict)
    Features = Inputs_wb['features']
    i = 0
    for feature in Features:
        if feature['name'] == key:
            break
        i += 1
        if i == len(Features):  # not found
            raise WBFeatureNotFoundError("%s: There is no feature named %s" % (func_name(), key))
    return i


def RemoveWBFeatures(Inputs_wb, i):
    '''
    remove a feature in World builder dictionary with an index i
    Inputs:
        Inputs_wb (dict): World buider dictionary
        i (int): index of the feature
    '''
    assert(type(Inputs_wb) == dict)
    Outputs_wb = Inputs_wb.copy()
    try:
        Features = Inputs_wb['features']
    except KeyError:
        raise WBFeatureNotFoundError()
    Features.pop(i)
    Outputs_wb['features'] = Features
    return Outputs_wb


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

        # convert additional fields to string 
        self.options["ADDITIONAL_FIELDS"] = str(additional_fields)

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
        
        additional_fields = [] # initiate the additional fields list

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
        if "metastable" in self.idict["Compositional fields"]["Names of fields"]:
            self.options['INCLUDE_METASTABLE'] = "True"
            additional_fields.append("metastable")
        else:
            self.options['INCLUDE_METASTABLE'] = "False"
        
        self.options["ADDITIONAL_FIELDS"] = str(additional_fields) # prescribe additional fields

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
                # todo_fdata
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

# todo_vtp
def ExportPolyDataFromRaw(Xs, Ys, Zs, Fs, fileout, **kwargs):
    '''
    Export poly data from raw data
    '''
    field_name = kwargs.get("field_name", "foo")
    assert(Xs.size == Ys.size)
    if Zs != None:
        assert(Xs.size == Zs.size)
    # add points
    i_points = vtk.vtkPoints()
    for i in range(Xs.size):
        x = Xs[i]
        y = Ys[i]
        if Zs is not None:
            z = Zs[i]
        else:
            z = 0.0
        i_points.InsertNextPoint(x, y, z)
    # Add VERTS to render points in ParaView
    i_verts = vtk.vtkCellArray()
    for i in range(i_points.GetNumberOfPoints()):
        i_verts.InsertNextCell(1)
        i_verts.InsertCellPoint(i)
    i_poly_data = vtk.vtkPolyData()  # initiate poly daa
    i_poly_data.SetPoints(i_points) # insert points
    i_poly_data.SetVerts(i_verts)
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
    Filter = vtk.vtkProbeFilter()
    Filter.SetSourceData(poly_data)  # use the polydata
    Filter.SetInputData(grid_data) # Interpolate 'Source' at these points
    Filter.Update()
    o_poly_data = Filter.GetOutput()
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
    if n_crust == 1:
        VtkP.PrepareSlab(crust_fields + ['spharz'], prepare_slab_distant_properties=True, depth_distant_lookup=depth_distant_lookup)
    elif n_crust == 2:
        VtkP.PrepareSlab(crust_fields, prepare_slab_distant_properties=True, depth_distant_lookup=depth_distant_lookup)
    else:
        raise NotImplementedError()
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
    compute_crust_thickness = kwargs.get("compute_crust_thickness", False)
    output_path = os.path.join(case_dir, "vtk_outputs")
    slab_shallow_cutoff = kwargs.get("slab_shallow_cutoff", 25e3)
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
    VtkP = VTKP(time=_time, slab_envelop_interval=slab_envelop_interval, slab_shallow_cutoff=slab_shallow_cutoff)
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
    if n_crust == 1:
        VtkP.PrepareSlab(crust_fields + ['spharz'], prepare_moho=crust_fields)  # slab: composition
    if n_crust == 2:
        VtkP.PrepareSlab(crust_fields, prepare_moho=crust_fields)  # slab: composition
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

    rs_n = kwargs.get("rs_n", 5) # resample interval
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

    start = time.time() 
    # interpolate the curve
    # start = np.ceil(depths[0]/ip_interval) * ip_interval
    start_depth = np.ceil(depths[0]/ip_interval) * ip_interval
    end_depth = np.floor(depths[-1]/ip_interval) * ip_interval
    n_out = int((end_depth-start_depth) / ip_interval)
    depths_out = np.arange(start_depth, end_depth, ip_interval)

    # interpolate T for surface
    interp_kind = kwargs.get("interp_kind", "cubic")
    slab_Xs = interp1d(depths, slab_envelop_rs[:, 0], kind=interp_kind)(depths_out)
    slab_Ys = interp1d(depths, slab_envelop_rs[:, 1], kind=interp_kind)(depths_out)
    if output_slab:
        o_slab_env_interp = os.path.join(case_dir,\
            "vtk_outputs", "slab_env1_interpolated_%05d.vtp" % (vtu_step)) # envelop 1
        ExportPolyDataFromRaw(slab_Xs, slab_Ys, None, None, o_slab_env_interp) # write the polydata
    end = time.time()
    print("Prepare interpolate points takes %.2f s" % (end-start))
    start = end

    slab_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((slab_Xs, slab_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
    env_Ttops  = vtk_to_numpy(slab_env_polydata.GetPointData().GetArray('T'))
    # fix invalid 0.0 values
    env_Ttops = fix_profile_field_zero_values(slab_Xs, slab_Ys, None, env_Ttops)


    # interpolate T for moho
    mask_moho = ((depths_out > depths_moho[0]) & (depths_out < depths_moho[id_valid]))
    moho_Xs = np.zeros(depths_out.shape)
    moho_Ys = np.zeros(depths_out.shape)
    moho_Xs[mask_moho] = interp1d(depths_moho[0: id_valid+1], moho_envelop_rs[0: id_valid+1, 0], kind='cubic')(depths_out[mask_moho])
    moho_Ys[mask_moho] = interp1d(depths_moho[0: id_valid+1], moho_envelop_rs[0: id_valid+1, 1], kind='cubic')(depths_out[mask_moho])

    moho_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((moho_Xs, moho_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
    env_Tbots = vtk_to_numpy(moho_env_polydata.GetPointData().GetArray('T'))
    env_Tbots = fix_profile_field_zero_values(moho_Xs, moho_Ys, None, env_Tbots) # fix 0 values
    
    mask = (env_Tbots < 1.0) # fix the non-sense values
    env_Tbots[mask] = -np.finfo(np.float32).max
    if debug:
        print("env_Tbots")  # screen outputs
        print(env_Tbots)
    end = time.time()
    print("Interpolating main profiles takes %.2f" % (end - start))
    start = end
    
    offset_Xs_array=[]; offset_Ys_array=[]; env_Toffsets_array = []
    # interpolate T for offest profiles
    for i, offset in enumerate(offsets):
        offset_Xs, offset_Ys = offset_profile(slab_Xs, slab_Ys, offset)
        offset_env_polydata = InterpolateGrid(VtkP.i_poly_data, np.column_stack((offset_Xs, offset_Ys)), quiet=True) # note here VtkPp is module shilofue/VtkPp, while the VtkP is the class
        env_Toffsets = vtk_to_numpy(offset_env_polydata.GetPointData().GetArray('T'))
        env_Toffsets = fix_profile_field_zero_values(offset_Xs, offset_Ys, None, env_Toffsets) # fix 0 values
    
        mask = (env_Toffsets < 1.0) # fix the non-sense values
        env_Toffsets[mask] = -np.finfo(np.float32).max

        offset_Xs_array.append(offset_Xs)
        offset_Ys_array.append(offset_Ys)
        env_Toffsets_array.append(env_Toffsets)
    end = time.time()
    print("Interpolating offset profiles takes %.2f" % (end - start))
    start = end

    # crustal thickness
    if compute_crust_thickness:
        distance_matrix = compute_pairwise_distances(slab_Xs, slab_Ys, moho_Xs, moho_Ys)
        distances = np.min(distance_matrix, axis=1)

    
    # output 
    if ofile is not None:
        # write output if a valid path is given
        n_columns = 7+len(offsets)*3
        if compute_crust_thickness:
            n_columns += 1
        data_env0 = np.zeros((depths_out.size, n_columns)) # output: x, y, Tbot, Ttop
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
        idx4 = idx
        if compute_crust_thickness:
            data_env0[:, idx+1] = distances

        # interpolate data to regular grid & prepare outputs
        # add additional headers if offset profiles are required
        header = "# 1: depth (m)\n# 2: x (m)\n# 3: y (m)\n# 4: x bot (m)\n# 5: y bot (m)\n"
        for i in range(len(offsets)):
            header += "# %d: x offset %d (m)\n# %d: y offset %d (m)\n" % (idx1+2*i+2,i,idx1+2*i+3,i)
        header += "# %d: Tbot (K)\n# %d: Ttop (K)\n" % (idx2+2, idx2+3)
        for i in range(len(offsets)):
            header += "# %d: Toffset %d (K)\n" % (idx3+i+2, i)
        if compute_crust_thickness:
            header += "# %d: crustal thickness (m)\n" % (idx4+2)
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
    
    end = time.time()
    print("Writing outputs takes %.2f" % (end - start))
    start = end
        
    return depths, env_Ttops, env_Tbots  # return depths and pressures


def fix_profile_field_zero_values(xs, ys, zs, T):
    '''
    fix the invalid 0.0 values in a vtk interpolation product
    '''
    # Step 1: fix z value to allow None input
    if zs is None:
        assert(xs.shape == ys.shape)
        zs = np.zeros(xs.shape)
    else:
        assert(xs.shape == ys.shape and xs.shape == zs.shape)

    # Step 2: Define 1D path distance for parametric interpolation
    path = np.cumsum(np.sqrt(np.diff(xs, prepend=xs[0])**2 + np.diff(ys, prepend=ys[0])**2 + np.diff(zs, prepend=zs[0])**2))

    # Step 3: Identify valid and invalid values
    valid_mask = (T != 0.0)
    invalid_mask = ~valid_mask

    # Step 4: Create 1D interpolator along path using valid values
    interp_func = interp1d(path[valid_mask], T[valid_mask], kind='linear', fill_value="extrapolate")

    # Step 5: Interpolate missing values and replace
    T_fixed = T.copy()
    T_fixed[invalid_mask] = interp_func(path[invalid_mask])

    return T_fixed


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

    # modify options with number of crusts
    kwargs["n_crust"] = Visit_Options.options["N_CRUST"]

    _, _, _ = SlabTemperature(case_dir, vtu_snapshot, o_file, **kwargs)
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
    options["output_slab"] = True
    options["assemble"] = False
    options["output_poly_data"] = False
    options["fix_shallow"] = True
    options["offsets"] = [-5e3, -10e3]
    options["slab_shallow_cutoff"] = 25e3
    options["rs_n"] = 5
    options["interp_kind"] = "linear"
    options["compute_crust_thickness"] = True

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
        try:
            self.ReadData(morph_file)
        except ValueError as e:
            raise ValueError("Reading file fails: %s" % morph_file) from e

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

########################################
# Functions for cases
########################################
class SLURM_OPT(JSON_OPT):
    def __init__(self):
        '''
        Initiation, first perform parental class's initiation,
        then perform daughter class's initiation.
        '''
        JSON_OPT.__init__(self)
        self.add_key("Slurm file (inputs)", str, ["slurm file"], "slurm.sh", nick='slurm_base_file')
        self.add_key("Openmpi version", str, ["openmpi version"], "", nick='openmpi')
        self.add_key("build directory", str, ["build directory"], "", nick="build_directory")
        self.add_key("Flag", str, ["flag"], "", nick="flag")
        self.add_key("Tasks per node", int, ["tasks per node"], 32, nick='tasks_per_node')
        self.add_key("cpus", int, ["cpus"], 1, nick="cpus")
        self.add_key("Threads per cpu", int, ["threads per cpu"], 1, nick='threads_per_cpu')
        self.add_key("List of nodes", list, ["node list"], [], nick="nodelist")
        self.add_key("Path to the prm file", str, ["prm path"], "./case.prm", nick="prm_path")
        self.add_key("Output directory", str, ["output directory"], ".", nick="output_directory")
        self.add_key("Base directory", str, ["base directory"], ".", nick="base_directory")

    def check(self):
        slurm_base_path = self.values[0]
        os.path.isfile(slurm_base_path)
        prm_path = self.values[8]
        os.path.isfile(prm_path)
        base_directory = self.values[10]
        assert(os.path.isdir(base_directory))
        output_directory = self.values[9]
        if not os.path.isdir(output_directory):
            os.mkdir(output_directory)

    def get_base_path(self):
        '''
        get the path to the base file (i.e job_p-billen.sh)
        '''
        base_directory = self.values[10]
        slurm_base_file = self.values[0]
        slurm_base_path = os.path.join(base_directory, slurm_base_file)
        return slurm_base_path

    def to_set_affinity(self):
        tasks_per_node = self.values[4]
        cpus = self.values[5]
        threads_per_cpu = self.values[6]
        threads = int(cpus * threads_per_cpu)
        nnode = int(np.ceil(threads / tasks_per_node))
        return nnode, threads, threads_per_cpu

    def to_set_command(self):
        build_directory = self.values[2]
        prm_path = self.values[8]
        return build_directory, prm_path

    def get_job_name(self):
        '''
        use the basename of the output directory as the job name
        '''
        output_directory = self.values[9]
        job_name = os.path.basename(output_directory)
        return job_name
        pass

    def get_output_path(self):
        '''
        get the path of the output file (i.e. job_p-billen.sh)
        '''
        slurm_base_path = self.values[0]
        output_directory = self.values[9]
        output_path = os.path.join(output_directory, os.path.basename(slurm_base_path))
        return output_path

    def fix_base_dir(self, base_directory):
        self.values[10] = base_directory
    
    def fix_output_dir(self, output_directory):
        self.values[9] = output_directory
    
    def get_output_dir(self):
        output_directory = self.values[9]
        return output_directory

class CASE_OPT(JSON_OPT):
    '''
    Define a class to work with CASE
    List of keys:
    '''
    def __init__(self):
        '''
        Initiation, first perform parental class's initiation,
        then perform daughter class's initiation.
        '''
        JSON_OPT.__init__(self)
        self.add_key("Name of the case", str, ["name"], "foo", nick='name')
        self.add_key("Base directory (inputs)", str, ["base directory"], ".", nick='base_dir')
        self.add_key("Output directory", str, ["output directory"], ".", nick='o_dir')
        self.add_key("Geometry", str, ["geometry"], "chunk", nick='geometry')
        self.add_key("potential temperature of the mantle", float,\
            ["potential temperature"], 1673.0, nick='potential_T')
        self.add_key("include fast first step", int,\
            ["include fast first step"], 0, nick='if_fast_first_step')
        self.add_key("Additional files to include", list,\
            ["additional files"], [], nick='additional_files')
        self.add_key("Root level from the project root", int,\
         ["root level"], 1, nick="root_level")
        self.add_key("If use world builder", int, ['use world builder'], 0, nick='if_wb')
        self.add_key("Type of the case", str, ["type"], '', nick='_type')
        self.add_key("Material model to use", str,\
         ["material model"], 'visco plastic', nick="material_model")
        self.add_key("Linear solver toleracne", float,\
         ["stokes solver", "linear solver tolerance"], 0.1, nick="stokes_linear_tolerance")
        self.add_key("End time", float, ["end time"], 60e6, nick="end_time")
        self.add_key("Type of velocity boundary condition\n\
            available options in [all fs, bt fs side ns]", str,\
            ["boundary condition", "velocity", "type"], "all fs", nick='type_bd_v')
        self.add_key("Dimension", int, ['dimension'], 2, nick='dimension')
        self.add_key("Refinement level, note this is a summarized parameter of the refinement scheme assigned,\
it only takes effect if the input is positiveh",\
            int, ["refinement level"], -1, nick="refinement_level")
        self.add_key("Case Output directory", str, ["case output directory"], "output", nick='case_o_dir')
        self.add_key("mantle rheology", str, ['mantle rheology', 'scheme'], "HK03_wet_mod", nick='mantle_rheology_scheme')
        self.add_key("Stokes solver type", str,\
         ["stokes solver", "type"], "block AMG", nick="stokes_solver_type")
        self.add_features('Slurm options', ['slurm'], SLURM_OPT)
        self.add_key("partitions", list, ["partitions"], [], nick='partitions')
        self.add_key("if a test case is generated for the initial steps", int, ['test initial steps', 'number of outputs'], -1, nick='test_initial_n_outputs')
        self.add_key("interval of outputs for the initial steps", float, ['test initial steps', 'interval of outputs'], 1e5, nick='test_initial_outputs_interval')
        self.add_key("Version number", float, ["version"], 0.1, nick="version")
        self.add_key("Type of visualization software for post-process", str,\
         ["post process", "visualization software"], "visit", nick="visual_software")
        self.add_key("Type of composition method", str,\
         ["composition method", "scheme"], "field", nick="comp_method")
        self.add_key("Depth average inputs", str, ["depth average file"], "", nick='da_inputs')
        self.add_key("mantle rheology scenario (previous composed)", str, ['mantle rheology', 'known scenario'], "", nick='mantle_rheology_known_scenario')
        self.add_key("Use the new rheology module, default is 0 to keep backward consistency", int, ['use new rheology module'], 0, nick='use_new_rheology_module')
        self.add_key("Minimum number of particles per cell", int,\
         ["composition method", "minimum particles per cell"], 33, nick="minimum_particles_per_cell")
        self.add_key("Maximum number of particles per cell", int,\
         ["composition method", "maximum particles per cell"], 50, nick="maximum_particles_per_cell")
    
    def check(self):
        '''
        check to see if these values make sense
        '''
        # output and input dirs
        base_dir = var_subs(self.values[1])
        o_dir = var_subs(self.values[2])
        my_assert(os.path.isdir(base_dir), FileNotFoundError, "No such directory: %s" % base_dir)
        # in case this is "", we'll fix that later.
        my_assert(o_dir=="" or os.path.isdir(o_dir), FileNotFoundError, "No such directory: %s" % o_dir)
        # type of the stokes solver
        stokes_solver_type = self.values[18]
        assert (stokes_solver_type in ["block AMG", "block GMG"])
        # type of the visualization software
        visual_software = self.values[24] 
        assert (visual_software in ["paraview", "visit"])
        # type of the composition method
        comp_method = self.values[25]
        assert (comp_method in ['field', 'particle'])
        # check file depth_average exist
        da_inputs = var_subs(self.values[26])
        if da_inputs != "":
            assert(os.path.isfile(da_inputs))
        use_new_rheology_module = self.values[28]
        assert(use_new_rheology_module in [0, 1])
        minimum_particles_per_cell = self.values[29]
        maximum_particles_per_cell = self.values[30]

    def to_init(self):
        '''
        Interface to init
        '''
        _type = self.values[9]
        base_dir = self.values[1]
        if _type == '':
            base_name = 'case.prm'
        else:
            base_name = 'case_%s.prm' % _type
        inputs = os.path.join(base_dir, base_name)
        if_wb = self.values[8]
        return self.values[0], inputs, if_wb

    def to_configure_prm(self):
        '''
        Interface to configure_prm
        '''
        refinement_level = self.values[15]
        return refinement_level
    
    def to_configure_final(self):
        '''
        Interface to configure_final
        '''
        return "foo", "foo"

    def wb_inputs_path(self):
        '''
        Interface to wb_inputs
        '''
        _type = self.values[9]
        if _type == '':
            base_name = 'case.wb'
        else:
            base_name = 'case_%s.wb' % _type
        wb_inputs = os.path.join(self.values[1], base_name)
        return wb_inputs

    def da_inputs_path(self):
        '''
        Interface to da_inputs
        '''
        da_inputs = var_subs(self.values[26])
        return da_inputs
    
    def o_dir(self):
        '''
        Interface to output dir
        '''
        return var_subs(self.values[2])
    
    def case_name(self):
        '''
        Return name of the case
        Return:
            case name (str)
        '''
        return self.values[0]
    
    def get_additional_files(self):
        '''
        Interface to add_files
        '''
        files = []
        for additional_file in self.values[6]:
            _path = var_subs(os.path.join(self.values[1], additional_file))
            my_assert(os.access(_path, os.R_OK), FileNotFoundError,\
            "Additional file %s is not found" % _path)
            files.append(_path)
        return files

    def output_step_one_with_fast_first_step(self):
        '''
        If we generate a case with fast-first-step computation
        and output the 1st step as well
        '''
        if_fast_first_step = self.values[5]
        if if_fast_first_step:
            self.values[5] = 2
        return self.values[5]
    
    def if_fast_first_step(self):
        '''
        If we generate a case with fast-first-step computation
        '''
        return self.values[5]
        pass
    
    def test_initial_steps(self):
        '''
        options for generatign a test case for the initial steps
        '''
        test_initial_n_outputs = self.values[21]
        test_initial_outputs_interval = self.values[22]
        return (test_initial_n_outputs, test_initial_outputs_interval)

    def if_use_world_builder(self):
        '''
        if we use world builder
        '''
        if_wb = self.values[8]
        return  (if_wb==1)
    
    def fix_case_name(self, case_name):
        '''
        fix base dir with a new value
        '''
        self.values[0] = case_name

    def fix_base_dir(self, base_dir):
        '''
        fix base dir with a new value
        '''
        assert(os.path.isdir(base_dir))
        self.values[1] = base_dir
    
    def fix_output_dir(self, o_dir):
        '''
        fix directory to output
        '''
        self.values[2] = o_dir

    def reset_refinement(self, reset_refinement_level):
        '''
        reset refinement level
        '''
        self.values[15] = reset_refinement_level
        pass
    
    def fix_case_output_dir(self, case_o_dir):
        '''
        reset refinement level
        '''
        self.values[16] = case_o_dir

    def reset_stokes_solver_type(self, stokes_solver_type):
        '''
        reset stokes solver type
        '''
        assert(stokes_solver_type in ["block AMG", "block GMG"])
        self.values[18] = stokes_solver_type
    
    def get_slurm_opts(self):
        slurm_opts = self.values[19]
        o_dir = var_subs(self.values[2])
        _name = self.values[0]
        output_directory = os.path.join(o_dir, _name)
        for slurm_opt in slurm_opts:
            slurm_opt.fix_output_dir(output_directory)
        return slurm_opts

class RHEOLOGY_OPT(JSON_OPT):
    '''
    Define a complex class for using the json files for testing the rheologies.
    '''
    def __init__(self):
        '''
        initiation
        '''
        JSON_OPT.__init__(self)
        self.add_key("Type of diffusion creep", str, ["diffusion"], "", nick='diffusion')
        self.add_key("Type of dislocation creep", str, ["dislocation"], "", nick='dislocation')
        self.add_key("Differences in ratio of the prefactor for diffusion creep", float, ["diffusion prefactor difference ratio"], 1.0, nick='dA_diff_ratio')
        self.add_key("Differences of the activation energy for diffusion creep", float, ["diffusion activation energy difference"], 0.0, nick='dE_diff')
        self.add_key("Differences of the activation volume for diffusion creep", float, ["diffusion activation volume difference"], 0.0, nick='dV_diff')
        self.add_key("Differences in ratio of the prefactor for dislocation creep", float, ["dislocation prefactor difference ratio"], 1.0, nick='dA_disl_ratio')
        self.add_key("Differences of the activation energy for dislocation creep", float, ["dislocation activation energy difference"], 0.0, nick='dE_disl')
        self.add_key("Differences of the activation volume for dislocation creep", float, ["dislocation activation volume difference"], 0.0, nick='dV_disl')
        # todo_r_json
        self.add_key("Grain size in mu m", float, ["grain size"], 10000.0, nick='d')
        self.add_key("Coh in /10^6 Si", float, ["coh"], 1000.0, nick='coh')
        self.add_key("fh2o in MPa", float, ["fh2o"], -1.0, nick='fh2o')
    
    def check(self):
        '''
        check values are validate
        '''
        RheologyPrm = RHEOLOGY_PRM()
        diffusion = self.values[0]
        if diffusion != "":
            diffusion_creep_key = (diffusion + "_diff") 
            assert(hasattr(RheologyPrm, diffusion_creep_key))
        dislocation = self.values[1]
        if dislocation != "":
            dislocation_creep_key = (dislocation + "_disl") 
            assert(hasattr(RheologyPrm, dislocation_creep_key))

    def to_RheologyInputs(self):
        '''
        '''

        diffusion = self.values[0]
        dislocation = self.values[1]
        dA_diff_ratio = self.values[2]
        dE_diff = self.values[3]
        dV_diff = self.values[4]
        dA_disl_ratio = self.values[5]
        dE_disl = self.values[6]
        dV_disl = self.values[7]
        d = self.values[8]
        coh = self.values[9]
        fh2o = self.values[10]
        use_coh = True
        if fh2o > 0.0:
            use_coh = False
        return diffusion, dislocation, dA_diff_ratio, dE_diff, dV_diff,\
        dA_disl_ratio, dE_disl, dV_disl, d, coh, fh2o, use_coh

class RHEOLOGY_PRM():
    """
    class for rheologies
    components and units:
        A (the prefactor) - MPa^(-n-r)*um**p/s
        n (stress dependence) - 1
        p (grain size dependence) - 1
        r (power of C_{OH}) - 1
        E (activation energy) - J / mol
        V (activation volume) - m^3 / mol
    Notes on the choice of the units:
        The unit of E and V are much easier to convert to UI.
        But for A, this code will handle the convertion, so the
        user only need to take the value for published flow laws.
    """
    def __init__(self):
        '''
        Initiation, initiate rheology parameters
        '''
        self.HK03_dry_disl = \
            {
                "A": 1.1e5,
                "p": 0.0,
                "r": 0.0,
                "n": 3.5,
                "E": 530e3,
                "V": 12e-6,
            }
        
        # dry diffusion creep in Hirth & Kohlstedt 2003)
        # note the V (activation energy) value has a large variation, here I
        # picked up a value the same as the wet value.
        self.HK03_dry_diff = \
            {
                "A": 1.5e9,
                "p": 3.0,
                "r": 0.0,
                "n": 1.0,
                "E": 375e3,
                "V": 4e-6,
            }
        
        # dislocation creep in Hirth & Kohlstedt 2003
        # with constant fH2O
        self.HK03_f_disl = \
            {
                "A": 1600,
                "p": 0.0,
                "r": 1.2,
                "n": 3.5,
                "E": 520e3,
                "V": 22e-6,
                "use_f": 1,
                "wet": 1
            }

        # diffusion creep in Hirth & Kohlstedt 2003
        self.HK03_f_diff = \
            {
                "A" : 2.5e7,
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3,
                "V" : 10e-6,
                "use_f": 1,
                "wet": 1
            }

        # dislocation creep in Hirth & Kohlstedt 2003
        # with constant Coh
        self.HK03_disl = \
            {
                "A": 90,
                "p": 0.0,
                "r": 1.2,
                "n": 3.5,
                "E": 480e3,
                "V": 11e-6,
            }

        # diffusion creep in Hirth & Kohlstedt 2003
        self.HK03_diff = \
            {
                "A" : 1.0e6,
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 335e3,
                "V" : 4e-6,
            }
        
        # dislocation creep in Hirth & Kohlstedt 2003
        # with varied Coh
        self.HK03_w_disl = \
            {
                "A": 1600.0,
                "p": 0.0,
                "r": 1.2,
                "n": 3.5,
                "E": 520e3,
                "V": 22e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0
            }

        # diffusion creep in Hirth & Kohlstedt 2003
        self.HK03_w_diff = \
            {
                "A" : 2.5e7,
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3,
                "V" : 10e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0
            }
        
        # dislocation creep in Arredondo & Billen 2017
        # this is found in the supplementary material in the paper.
        # Note that the original value in the paper uses "Pa" for A,
        # while I converted it to "MPa" here.
        self.AB17_disl = \
            {
                "A": 25.7,
                "p": 0.0,
                "r": 1.2,
                "n": 3.5,
                "E": 496e3,
                "V": 11e-6,
                "d" : 1e4,
                "Coh" : 1000.0
            }
        
        # diffusion creep in Arredondo & Billen 2017
        self.AB17_diff = \
            {
                "A" : 2.85e5,  # note: their number in the 2017 appendix is wrong,
                "p" : 3.0, #  but it's right in the 2016 paper.
                "r" : 1.0,
                "n" : 1.0,
                "E" : 317e3,
                "V" : 4e-6,
                "d" : 1e4,
                "Coh" : 1000.0
            }
        
        # modified creep laws from Hirth & Kohlstedt 2003
        # for detail, refer to magali's explain_update_modHK03_rheology.pdf file
        # 'wet' indicates this has to applied with a rheology of water
        self.HK03_wet_mod_diff = \
            {
                "A" : 7.1768184e6,  
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3,
                "V" : 23e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0,  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
                "use_f": 1
            }

        self.HK03_wet_mod_disl = \
            {
                "A" : 457.142857143,
                "p" : 0.0,
                "r" : 1.2,
                "n" : 3.5,
                "E" : 520e3,
                "V" : 24e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet" : 1.0,
                "use_f": 1
            }
        
        # modified creep laws from Hirth & Kohlstedt 2003
        # for detail, refer to magali's explain_update_modHK03_rheology.pdf file
        # 'wet' indicates this has to applied with a rheology of water
        self.HK03_wet_mod_ln_diff = \
            {
                "A" : 7.1768184e6,  
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3,
                "V" : 23e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }

        self.HK03_wet_mod_ln_disl = \
            {
                "A" : 457.142857143,
                "p" : 0.0,
                "r" : 1.2,
                "n" : 3.5,
                "E" : 520e3,
                "V" : 24e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet" : 1.0
            }
        
        # modified creep laws from Hirth & Kohlstedt 2003
        # I bring the values to the limit of the range
        # for detail, refer to magali's explain_update_modHK03_rheology.pdf file
        self.HK03_wet_mod1_diff = \
            {
                # "A" : 10**6.9,  # MPa^(-n-r)*um**p/s
                "A" : 7.1768e6,  # MPa^(-n-r)*um**p/s
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3 - 25e3,
                "V" : 23e-6 -5.5e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }

        self.HK03_wet_mod1_disl = \
            {
                "A" : 10**2.65,
                "p" : 0.0,
                "r" : 1.0,
                "n" : 3.5,
                "E" : 520e3 + 40e3,
                "V" : 24e-6 + 4e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet" : 1.0
            }

         # modified creep laws from Hirth & Kohlstedt 2003
        # for detail, refer to magali's explain_update_modHK03_rheology.pdf file
        # 'wet' indicates this has to applied with a rheology of water
        # In the version, I modified the value of r, compared to the first version
        self.HK03_wet_mod2_diff = \
            {
                # "A" : 10**6.9,  # MPa^(-n-r)*um**p/s
                "A" : 7.1768e6,  # MPa^(-n-r)*um**p/s
                "p" : 3.0,
                "r" : 0.8, # 1.0 -> 0.8
                "n" : 1.0,
                "E" : 375e3,
                "V" : 23e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }

        self.HK03_wet_mod2_disl = \
            {
                "A" : 10**2.65,
                "p" : 0.0,
                "r" : 1.2,  # 1.0 -> 1.2
                "n" : 3.5,
                "E" : 520e3,
                "V" : 24e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet" : 1.0
            }
        
        
        # modified creep laws from Hirth & Kohlstedt 2003
        # I bring the values to the limit of the range
        # for detail, refer to magali's explain_update_modHK03_rheology.pdf file
        # this is specifically the one I used for the TwoD models.
        # Combined with the usage of function "MantleRheology", then same rheology
        # could be reproduced
        self.HK03_wet_mod_2d_diff = \
            {
                # "A" : 10**6.9,  # MPa^(-n-r)*um**p/s
                "A" : 7.1768e6,  # MPa^(-n-r)*um**p/s
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 375e3 - 40e3,
                "V" : 23e-6 -5.5e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }

        self.HK03_wet_mod_2d_disl = \
            {
                "A" : 10**2.65,
                "p" : 0.0,
                "r" : 1.0,
                "n" : 3.5,
                "E" : 520e3 + 20e3,
                "V" : 24e-6,
                "d" : 1e4,
                "Coh" : 1000.0,
                "wet" : 1.0
            }
        
        self.WarrenHansen23_disl =\
            {
                "A": 20,
                "p": 0.0,
                "r": 1.2,
                "n": 3.5,
                "E": 480e3,
                "V": 11e-6
            }

        # diffusion creep in Hirth & Kohlstedt 2003
        # Note I use the value of 4e-6 from the original experimental result
        # and apply a -2.1e-6 differences later to get the values 
        # in the Warren and Hansen 2023 paper
        self.WarrenHansen23_diff = \
            {
                "A" : 2.9e5,
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 335e3,
                "V" : 4e-6
            }
        
        
        self.water = \
            {
                "A" : 87.75,             # H/(10^6*Si)/MPa
                "E" : 50e3,                     # J/mol +/-2e3
                "V" : 10.6e-6                     # m^3/mol+/-1
            }

        # this is the values used in the ARCAY17 paper
        # note: their rheology is only stress dependent (dislocation creep)
        # their yielding criterion is stress dependent as well.
        self.ARCAY17_diff = None
        self.ARCAY17_disl = \
            {
                "A" : 339428.7,
                "p" : 0.0,
                "r" : 0.0,  # not dependent on the "Coh"
                "n" : 3.0,
                "E" : 465e3,
                "V" : 17e-6,
                "d" : 1e4, # not dependent on d
                "Coh" : 1000.0
            }
        self.ARCAY17_brittle = \
        {
            "friction" : 0.05,
            "cohesion": 1e6, # pa
            "n": 30.0,
            "ref strain rate" : 1.0e-14,
            "type": "stress dependent"
        }

        self.water = \
            {
                "A" : 87.75,             # H/(10^6*Si)/MPa
                "E" : 50e3,                     # J/mol +/-2e3
                "V" : 10.6e-6                     # m^3/mol+/-1
            }

        # todo_mineral
        # Basalt rheology from Shelton and Tullis 1981 
        # and Hacker and Christie 1990
        # the diffusion creep of this rheology is missing
        # in literatures
        self.ST1981_basalt_diff = None

        self.ST1981_basalt_disl = \
            {
                "A" : 1.0e-4,
                "p" : 0.0,
                "r" : 0.0,  # not dependent on the "Coh"
                "n" : 3.5,
                "E" : 250e3,
                "V" : 0.0,
                "d" : 1e4, # not dependent on d
            }

        # Quartz rheology from Ranalli and Murphy 1987.
        # the diffusion creep of this rheology is missing
        # in literatures
        self.ST1981_basalt_diff = None

        self.RM1987_quartz_disl = \
            {
                "A" : 6.8e-6,
                "p" : 0.0,
                "r" : 0.0,  # not dependent on the "Coh"
                "n" : 3,
                "E" : 156e3,
                "V" : 0.0
            }
        
        self.KK1987_quartz_disl = \
            {
                "A" : 3.2e-4,
                "p" : 0.0,
                "r" : 0.0,  # not dependent on the "Coh"
                "n" : 2.3,
                "E" : 154e3,
                "V" : 8e-6
            }
        
        self.Ranali_95_anorthite_75_diff = None
        
        self.Ranali_95_anorthite_75_disl = \
            {
                "A" : 3.3e-4,
                "p" : 0.0,
                "r" : 0.0,  # not dependent on the "Coh"
                "n" : 3.2,
                "E" : 238e3,
                "V" : 8e-6
            }

        self.Rybachi_06_anorthite_wet_diff = \
            {
                "A" : 0.2,  # note, 10^(-0.7), less than 1 digit accuracy, as 10^(0.1) = 1.25
                "p" : 3.0,
                "r" : 1.0,
                "n" : 1.0,
                "E" : 159e3,
                "V" : 38e-6,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }
        
        self.Rybachi_06_anorthite_wet_disl = \
            {
                "A" : 1.6,  # note, 10^(0.2), less than 1 digit accuracy, as 10^(0.1) = 1.25
                "p" : 0.0,
                "r" : 1.0,
                "n" : 3.0,
                "E" : 345e3,
                "V" : 38e-6,
                "wet": 1.0  # I use this to mark this is a wet rheology, so I need to account for V and E for water later.
            }
        
        self.Rybachi_06_anorthite_dry_diff = \
            {
                "A" : 1.26e12,  # note, 10^(12.1), less than 1 digit accuracy, as 10^(0.1) = 1.25
                "p" : 3.0,
                "r" : 0.0,  # dry, not dependent on fugacity
                "n" : 1.0,
                "E" : 460e3,
                "V" : 24e-6
            }
        
        self.Rybachi_06_anorthite_dry_disl = \
            {
                "A" : 5.01e12,  # note, 10^(12.7), less than 1 digit accuracy, as 10^(0.1) = 1.25
                "p" : 0.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 3.0,
                "E" : 641e3,
                "V" : 24e-6
            }
        
        self.Dimanov_Dresen_An50Di35D_wet_diff = \
        {
                # diffusion creep for a 35 mu*m grain size
                "A" : 5488000000.0,  #   1.28e-1 * (1e6) / (35)^(-3)
                "p" : 3.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 1.0,
                "E" : 316e3,
                "V" : 0.0  # not present in the table
        }
        
        self.Dimanov_Dresen_An50Di35D_wet_disl = \
        {
            # this is actually the An50DiD in table 3b
            # since the dislocation creep is not grain size sensitive
                "A" : 10174679.0993,  #  / 1.54e-17 * (1e6)^3.97
                "p" : 0.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 3.97,
                "E" : 556e3,
                "V" : 0.0  # not present in the table
        }

        self.Dimanov_Dresen_An50Di35D_dry_diff = \
        {
                # diffusion creep for a 35 mu*m grain size
                "A" : 5.1879e13,  #  1.21e3 * (1e6) / (35)^(-3)
                "p" : 3.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 1.0,
                "E" : 436e3,
                "V" : 0.0  # not present in the table
        }
        
        self.Dimanov_Dresen_An50Di35D_dry_disl = \
        {
            # this is actually the An50DiD in table 3b
            # since the dislocation creep is not grain size sensitive
                "A" : 8.1840692e+12,  #  / 2.71e-12 * (1e6)^4.08
                "p" : 0.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 4.08,
                "E" : 723e3,
                "V" : 0.0  # not present in the table
        }
        
        
        self.Dimanov_Dresen_An50Di45D_dry_diff = \
        {
                # diffusion creep for a 45 mu*m grain size
                "A" : 6.187e15,  # 6.79e4 * 1e6 / (45)^(-3.0)
                "p" : 3.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 1.0,
                "E" : 496e3,
                "V" : 0.0  # not present in the table
        }
        
        self.Dimanov_Dresen_An50Di45D_dry_disl = \
        {
            # this is actually the An50DiD in table 3b
            # since the dislocation creep is not grain size sensitive
                "A" : 8.1840692e+12,  #  / 2.71e-12 * (1e6)^4.08
                "p" : 0.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 4.08,
                "E" : 723e3,
                "V" : 0.0  # not present in the table
        }

        self.Rybachi_2000_An100_dry_diff = \
        {
                # diffusion creep, for the An100 in table 3
                # note that this is marked as dry, but there
                # is 640 ppm H/Si in the synthetic anorthite
                "A" : 1.258925e12, # 10^12.1
                "p" : 3.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 1.0,
                "E" : 467e3,
                "V" : 0.0  # not present in the table
        }
        
        self.Rybachi_2000_An100_dry_disl = \
        {
            # dislocation creep, for the An100 in table 3
                "A" : 5.01187e12,  # 10^12.7
                "p" : 0.0,
                "r" : 0.0, # dry, not dependent on fugacity
                "n" : 3.0,
                "E" : 648e3,
                "V" : 0.0  # not present in the table
        }

        # todo_peierls 
        self.MK10_peierls = \
        {
            'q': 1.0,
            'p': 0.5,
            'n': 2.0,
            'sigp0': 5.9e3,    				# MPa (+/- 0.2e3 Pa)
            'A': 1.4e-7,      # s^-1 MPa^-2
            'E': 320e3,      				# J/mol (+/-50e3 J/mol)
            'V' : 0.0,  # not dependent on the pressure
            "Tref" : 873.0, # reference temperature from the experiment
            "Pref" : 4.5e9 # Pa, reference pressure
        }
        
        self.Idrissi16_peierls = \
        {
            'q': 2.0,
            'p': 0.5,
            'n': 0.0,
            'sigp0': 3.8e3,    				# MPa (+/- 0.2e3 Pa)
            'A': 1e6,      # s^-1 MPa^-2
            'E': 566e3,      				# J/mol (+/-50e3 J/mol)
            'V' : 0.0  # not dependent on the pressure
        }


        self.Byerlee_brittle = \
        {
            "type": "Byerlee"
        }

    def get_rheology(self, _name, _type):
        '''
        read rheology parameters, and account for effects of water if it is a wet rheology
        '''
        assert(_type in ['diff', 'disl', 'brittle'])
        _attr = _name + "_" + _type
        if not hasattr(self, _attr):
            raise ValueError("RHEOLOGY_PRM object doesn't have attribute %s" % _attr)
        creep = getattr(self, _attr)
        if "wet" in creep:
            # foh enters explicitly, converting to use Coh
            assert(_type in ['diffusion', 'dislocation'])
            ### effects of water accounted, see Magali's file explain_update_modHK03_rheology eq(5)
            water_creep = getattr(self, "water")
            creep['A'] = creep['A'] / (water_creep['A'] ** creep['r'])
            creep['V'] = creep['V'] - water_creep['V'] * creep['r']
            creep['E'] = creep['E'] - water_creep['E'] * creep['r']
        return creep


def ReadAspectProfile(depth_average_path, **kwargs):
    """
    read a T,P profile from aspect's depth average file
    """
    # include options
    include_adiabatic_temperature = kwargs.get("include_adiabatic_temperature", False)
    interp = kwargs.get("interp", 0)
    # check file exist
    assert(os.access(depth_average_path, os.R_OK))
    # read that
    DepthAverage = DEPTH_AVERAGE_PLOT('DepthAverage')
    DepthAverage.ReadHeader(depth_average_path)
    DepthAverage.ReadData(depth_average_path)
    DepthAverage.SplitTimeStep()
    time_step = 0
    i0 = DepthAverage.time_step_indexes[time_step][-1] * DepthAverage.time_step_length
    if time_step == len(DepthAverage.time_step_times) - 1:
        # this is the last step
        i1 = DepthAverage.data.shape[0]
    else:
        i1 = DepthAverage.time_step_indexes[time_step + 1][0] * DepthAverage.time_step_length
    data = DepthAverage.data[i0:i1, :]
    col_depth = DepthAverage.header['depth']['col']
    col_P = DepthAverage.header['adiabatic_pressure']['col']
    col_T = DepthAverage.header['temperature']['col']
    col_Tad = DepthAverage.header['adiabatic_temperature']['col']
    depths = data[:, col_depth]
    pressures = data[:, col_P]
    temperatures = data[:, col_T]
    if include_adiabatic_temperature:
        adiabatic_temperatures = data[:, col_Tad]
    if interp > 0:
        depths_old = depths.copy()
        depths = np.linspace(depths[0], depths[-1], interp)
        pressures_old = pressures.copy()
        pressures = np.interp(depths, depths_old, pressures_old)
        temperatures_old = temperatures.copy()
        temperatures = np.interp(depths, depths_old, temperatures_old) 
        if include_adiabatic_temperature:
            adiabatic_temperatures_old = adiabatic_temperatures.copy()
            adiabatic_temperatures = np.interp(depths, depths_old, adiabatic_temperatures_old) 
    # return type is determined by whether there are included terms
    if include_adiabatic_temperature:
        return depths, pressures, temperatures, adiabatic_temperatures
    else:
        return depths, pressures, temperatures

def GetRheology(rheology, **kwargs):
    '''
    read rheology parameters, and account for effects of water if it is a wet rheology
    Inputs:
        kwargs:
            dEdiff - a difference between the activation energy and the medium value in experiment
                (dVdiff, dEdisl, dVdisl) are defined in the same way
            dAdiff_ratio - a ratio of (A / A_medium) for the prefactor of the diffusion creep
                dAdisl_ratio is defined in the same way.
            use_coh - whether use the Coh or Fh2O as input into the wet rheology
    '''
    # these options are for a differences from the central value
    dEdiff = kwargs.get('dEdiff', 0.0)  # numbers for the variation in the rheology
    dVdiff = kwargs.get('dVdiff', 0.0)
    dAdiff_ratio = kwargs.get("dAdiff_ratio", 1.0)
    dAdisl_ratio = kwargs.get("dAdisl_ratio", 1.0)
    dEdisl = kwargs.get('dEdisl', 0.0)
    dVdisl = kwargs.get('dVdisl', 0.0)
    use_coh = kwargs.get("use_coh", True)
    # initiate the class object
    RheologyPrm = RHEOLOGY_PRM()
    # if the diffusion creep flow law is specified, then include it here
    if hasattr(RheologyPrm, rheology + "_diff"):
        diffusion_creep = getattr(RheologyPrm, rheology + "_diff")
        ### if the rheology is formulated with the fugacity, convert it to using the Coh
        if diffusion_creep is not None:
            try:
                _ = diffusion_creep['wet']
            except KeyError:
                pass
            else:
                if use_coh:
                    water_creep = getattr(RheologyPrm, "water")
                    diffusion_creep['A'] = diffusion_creep['A'] / (water_creep['A'] ** diffusion_creep['r'])
                    diffusion_creep['V'] = diffusion_creep['V'] - water_creep['V'] * diffusion_creep['r']
                    diffusion_creep['E'] = diffusion_creep['E'] - water_creep['E'] * diffusion_creep['r']
            # apply the differences to the medium value
            diffusion_creep['A'] *= dAdiff_ratio
            diffusion_creep['E'] += dEdiff
            diffusion_creep['V'] += dVdiff
    else:
        diffusion_creep = None
    # if the dislocation creep flow law is specified, then include it here
    if hasattr(RheologyPrm, rheology + "_disl"):
        dislocation_creep = getattr(RheologyPrm, rheology + "_disl")
        ### if the rheology is formulated with the fugacity, convert it to using the Coh
        if dislocation_creep is not None:
            try:
                _ = dislocation_creep['wet']
            except KeyError:
                pass
            else:
                if use_coh:
                    water_creep = getattr(RheologyPrm, "water")
                    dislocation_creep['A'] = dislocation_creep['A'] / (water_creep['A'] ** dislocation_creep['r'])
                    dislocation_creep['V'] = dislocation_creep['V'] - water_creep['V'] * dislocation_creep['r']
                    dislocation_creep['E'] = dislocation_creep['E'] - water_creep['E'] * dislocation_creep['r']
            # apply the differences to the medium value
            dislocation_creep['A'] *= dAdisl_ratio
            dislocation_creep['E'] += dEdisl
            dislocation_creep['V'] += dVdisl
    else:
        dislocation_creep = None
    # return the rheology 
    return diffusion_creep, dislocation_creep

def GetPeierlsRheology(rheology):
    '''
    read the peierls rheology parameters
    Inputs:
        rheology: a string of the type of rheology to use.
    Returns:
        peierls_creep: a dict of the flow law variables for the peierls creep
    '''
    RheologyPrm = RHEOLOGY_PRM()
    my_assert(hasattr(RheologyPrm, rheology + "_peierls"), ValueError,\
    "The %s is not a valid option for the peierls rheology" % rheology)
    peierls_creep = getattr(RheologyPrm, rheology + "_peierls")
    return peierls_creep

def CreepStrainRate(creep, stress, P, T, d, Coh, **kwargs):
    """
    Calculate strain rate by flow law in form of 
        B * sigma^n * exp( - (E + P * V) / (R * T))
    Units:
     - P: Pa
     - T: K
     - d: mu m
     - stress: MPa
     - Coh: H / 10^6 Si
     - Return value: s^-1
    kwargs:
        use_effective_strain_rate - use the second invariant as input
    Pay attention to pass in the right value, this custom is inherited
    """
    A = creep['A']
    p = creep['p']
    r = creep['r']
    n = creep['n']
    E = creep['E']
    V = creep['V']
    # calculate B
    # compute F
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    f_by_factor = kwargs.get('f_by_factor', False)
    if use_effective_strain_rate:
        F = 3**((n+1)/2) / 2.0
    elif f_by_factor:
        F = kwargs['F']
    else:
        F = 1.0
    B = A * d**(-p) * Coh**r
    return F * B *stress**n * np.exp(-(E + P * V) / (R * T))

def CreepRheology(creep, strain_rate, P, T, d=1e4, Coh=1e3, **kwargs):
    """
    Calculate viscosity by flow law in form of (strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp((E + P * V) / (n * R * T))
    Previously, there is a typo in the F factor
    Units:
     - P: Pa
     - T: K
     - d: mu m
     - Coh: H / 10^6 Si
     - Return value: Pa*s
    Pay attention to pass in the right value, this custom is inherited
    """
    A = creep['A']
    p = creep['p']
    r = creep['r']
    n = creep['n']
    E = creep['E']
    V = creep['V']
    # compute value of F(pre factor)
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    f_by_factor = kwargs.get('f_by_factor', False)
    if use_effective_strain_rate:
        F = 1 / (2**((n-1)/n)*3**((n+1)/2/n)) * 2.0
    elif f_by_factor:
        F = kwargs['F']
    else:
        F = 1.0
    # calculate B
    B = A * d**(-p) * Coh**r
    eta = 1/2.0 * F * (strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp((E + P * V) / (n * R * T)) * 1e6

    return eta

def ComputeComposite(*Args):
    '''
    compute value of composite viscosity from value of diffusion creep and 
    dislocation creep. This will check that at least one entry is not None.
    If one of them is none, then the other entry will be directly returned
    '''
    i = 0
    indexes = []
    for Arg in Args:
        if Arg is not None:
            indexes.append(i)
        i += 1
    assert(len(indexes) > 0)  # check their is valid inputs
    if len(indexes) == 1:
        # if there is only 1 entry, just return it
        return Args[indexes[0]]
    else:
        reciprocal = 0.0
        for index in indexes:
            reciprocal += 1.0 / Args[index]
        eta_comp = 1.0 / reciprocal
        return eta_comp

def CreepComputeA(creep, strain_rate, P, T, eta, d=1e4, Coh=1e3, **kwargs):
    """
    Compute the prefactor in the rheology with other variables in a flow law (p, r, n, E, V).
    The viscosity is computed at condition of P, T and is constrained to be eta.
    Calculate viscosity by flow law in form of 0.5*(strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp((E + P * V) / (n * R * T))
    Units:
     - creep: flow law that contains p, r, n, E, V
     - strain_rate: the strain rate to compute viscosity with
     - P: The pressure to compute viscosity, unit is Pa
     - T: The temperature to compute viscosity, unit is K
     - d: The grain size to compute viscosity, unit is mu m
     - Coh: H / 10^6 Si
     - Return value: Pa*s
    Pay attention to pass in the right value, this custom is inherited
    Here I tried to input the right value for the F factor
    """
    p = creep['p']
    r = creep['r']
    n = creep['n']
    E = creep['E']
    V = creep['V']
    # compute value of F(pre factor)
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    if use_effective_strain_rate:
        F = 1 / (2**((n-1)/n)*3**((n+1)/2/n))
    else:
        F = 1.0
    # calculate B
    B = (0.5*F/eta)**n * strain_rate**(1-n) * np.exp((E+P*V)/(R*T)) * (1e6)**n
    A = B * d**p * Coh**(-r)
    return A

def Convert2AspectInput(creep, **kwargs):
    """
    Viscosity is calculated by flow law in form of (strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp((E + P * V) / (n * R * T)) * 1e6
    while in aspect, flow law in form of 0.5 * A**(-1.0 / n) * d**(m / n) * (strain_rate)**(1.0 / n - 1) * np.exp((E + P * V) / (n * R * T))
    In this version, I am trying to take care of the F factor correctly
    Inputs:
        kwargs:
            d: um, the grain size to use, default is 1e4
            Coh: H / 10^6 Si, default is 1000.0
    Original Units in creep:
     - P: Pa
     - T: K
    Converted units in aspect_creep:
     - P: Pa
     - T: K
     - d: m
    """
    # read in initial value
    A = creep['A']
    p = creep['p']
    r = creep['r']
    n = creep['n']
    E = creep['E']
    V = creep['V']
    d = kwargs.get('d', 1e4)
    Coh = kwargs.get('Coh', 1000.0)
    # compute value of F(pre factor)
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    if use_effective_strain_rate:
        F = 1 / (2**((n-1)/n)*3**((n+1)/2/n)) * 2.0
    else:
        F = 1.0
    # prepare values for aspect
    aspect_creep = {}
    # stress in the original equation is in Mpa, grain size is in um
    aspect_creep['A'] = 1e6**(-p) * (1e6)**(-n) * Coh**r * A / F**n  # F term: use effective strain rate
    aspect_creep['d'] = d / 1e6
    aspect_creep['n'] = n
    aspect_creep['m'] = p
    aspect_creep['E'] = E
    aspect_creep['V'] = V
    return aspect_creep

def CreepRheologyInAspectViscoPlastic(creep, strain_rate, P, T):
    """
    def CreepRheologyInAspectVisoPlastic(creep, strain_rate, P, T)

    Calculate viscosity by way of Visco Plastic module in aspect
    flow law in form of 0.5 * A**(-1.0 / n) * d**(m / n) * (strain_rate)**(1.0 / n - 1) * np.exp((E + P * V) / (n * R * T))
    Units:
     - P: Pa
     - T: K
     - d: m
     - Return value: Pa*s
    """
    A = creep['A']
    m = creep['m']
    n = creep['n']
    E = creep['E']
    V = creep['V']
    d = creep['d']
    # calculate B
    return 0.5 * A**(-1.0 / n) * d**(m / n) * (strain_rate)**(1.0 / n - 1) * np.exp((E + P * V) / (n * R * T))

def CreepComputeV(creep, strain_rate, P, T, eta, d=1e4, Coh=1e3, **kwargs):
    """
    Calculate V based on other parameters 
    Units:
     - P: Pa
     - T: K
     - d: mu m
     - Coh: H / 10^6 Si
     - Return value: Pa*s
    Pay attention to pass in the right value, this custom is inherited
    """
    A = creep['A']
    p = creep['p']
    r = creep['r']
    n = creep['n']
    E = creep['E']
    # compute value of F(pre factor)
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    if use_effective_strain_rate:
        F = 1 / (2**((n-1)/n)*3**((n+1)/2/n)) * 2.0
    else:
        F = 1.0
    # calculate B
    B = A * d**(-p) * Coh**r
    exponential = eta / (1/2.0 * F * (strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp(E / (n * R * T)) * 1e6)
    V = n * R * T * np.log(exponential) / P
    return V

def LowerMantleV(E, Tmean, Pmean, grad_T, grad_P):
    '''
    compute the value of activation volume for the lower mantle
    based on the criteria of a nearly constant viscosity
    '''    
    V = E * grad_T / (grad_P * Tmean - Pmean * grad_T)
    return V

class RHEOLOGY_OPR():
    '''
    rheology operation, do some complex staff
    Attributes:
        RheologyPrm: an initiation of the class RHEOLOGY_PRM
        depths(ndarray), pressures, temperatures: depth, pressure, temperature profile
            (all these 3 profiles are loaded from a depth_average outputs of ASPECT)
        peierls_type: a string, type of the peierls rheology
        peierls: the dictionary of variables to be used for the peierls rheology
    '''
    def __init__(self):
        '''
        Initiation
        '''
        self.RheologyPrm = RHEOLOGY_PRM()
        # set of variables for mantle profile
        self.depths = None
        self.pressures = None
        self.tempertures = None
        self.output_profile = None # for the figure plotted
        self.output_json = None
        self.output_aspect_json = None
        self.diff_type = None
        self.diff = None
        self.disl_type = None
        self.disl = None
        self.brittle_type = None
        self.brittle = None
        self.peierls_type = None
        self.peierls = None
        pass

    def SetRheology(self, **kwargs):
        '''
        set rheology type with instances of rheology (i.e. a dictionary)
        '''
        self.diff = kwargs.get('diff', None)
        self.disl = kwargs.get('disl', None)
        self.brittle = kwargs.get('brittle', None)
        self.peierls = kwargs.get('peierls', None)
        pass
    
    def SetRheologyByName(self, **kwargs):
        '''
        set rheology type with instances of rheology (i.e. a dictionary)
        '''
        diff_type = kwargs.get('diff', None)
        disl_type = kwargs.get('disl', None)
        brittle_type = kwargs.get('brittle', None)
        peierls_type = kwargs.get('peierls', None)
        self.diff_type = diff_type
        self.disl_type = disl_type
        self.brittle_type = brittle_type
        self.peierls_type = peierls_type
        if diff_type != None and disl_type != None:
            assert(diff_type == disl_type)  # doesn't make sense to have inconsistent flow laws
        if diff_type != None:
            diffusion_creep, _ = GetRheology(diff_type)
            self.diff = diffusion_creep
        if disl_type != None:
            _, dislocation_creep = GetRheology(disl_type)
            self.disl = dislocation_creep
        if brittle_type != None:
            self.brittle = self.RheologyPrm.get_rheology(brittle_type, 'brittle')
        if self.peierls_type != None:
            self.peierls = GetPeierlsRheology(peierls_type) 
    
    
    def ReadProfile(self, file_path):
        '''
        Read a depth, pressure and temperature profile from a depth_average output
        These values are saved as class variables
        '''
        self.depths, self.pressures, self.temperatures = ReadAspectProfile(file_path, interp=3000)


    # todo_HK03
    def VaryWithStress(self, P, T, d, Coh, stress_range, **kwargs):
        '''
        With a set of variables, compare to the data points reported
        in experimental publications.
        Inputs:
            stress_range - a range of stress for the strain rate - stress plot
            strain_rate_range - a range of strain_rate, only affects plot
            P,d,Coh - samples of variables for the strain rate - stress plot
            kwargs:
                ax - an axis for plot
                label - label for the curve
                color - the color used for plotting
                
        '''
        assert(self.diff is not None and self.disl is not None)
        stress_range = kwargs.get("stress_range", [10.0, 1000.0])  # Mpa
        strain_rate_range = kwargs.get("strain_rate_range", None)  # s^{-1}
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', None)
        _color = kwargs.get('color', 'b')
        use_effective_strain_rate = kwargs.get('use_effective_strain_rate', True)

        stresses = np.linspace(stress_range[0], stress_range[1], 1000)
        strain_rates_diff = CreepStrainRate(self.diff, stresses, P, T, d, Coh, use_effective_strain_rate=use_effective_strain_rate)
        strain_rates_disl = CreepStrainRate(self.disl, stresses, P, T, d, Coh, use_effective_strain_rate=use_effective_strain_rate)
        strain_rates = strain_rates_diff + strain_rates_disl  # apply the isostress model
        if ax is not None:
            ax.loglog(stresses, strain_rates, '-', color=_color, label=(label + "comp"))
            ax.loglog(stresses, strain_rates_diff, '--', color=_color, label=(label + "diff"))
            ax.loglog(stresses, strain_rates_disl, '-.', color=_color, label=(label + "disl"))
        ax.set_xlabel("Stress (Mpa)")
        ax.set_xlim(stress_range[0], stress_range[1])
        if strain_rate_range is not None:
            ax.set_ylim(strain_rate_range[0], strain_rate_range[1])
        ax.set_ylabel("Strain Rate (s^-1)")
        _title = "P = %.4e, T = %.4e, d = %.4e, Coh = %.4e" % (P/1e6, T, d, Coh)
        ax.set_title(_title)

    
    def VaryWithT(self, P, stress, d, Coh, T_range, **kwargs):
        '''
        With a set of variables, compare to the data points reported
        in experimental publications.
        Inputs:
            P - a pressure to compute the strain rate
            stress - a sample of stress for the strain rate - 10^4/T plot
            d - a sample of grain size for the strain rate - 10^4/T plot
            Coh - a sample of Coh for the strain rate - 10^4/T plot
            T_range - a range of T for the strain rate - 10^4/T plot
            kwargs:
                ax - an axis for plot
                label - label for the curve
                color - the color used for plotting
        '''
        assert(self.diff is not None and self.disl is not None)
        strain_rate_range = kwargs.get("strain_rate_range", None)  # s^{-1}
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', None)
        _color = kwargs.get('color', 'b')
        use_effective_strain_rate = kwargs.get('use_effective_strain_rate', True)
        
        Ts = np.linspace(T_range[0], T_range[1], 1000)
        strain_rates_diff = CreepStrainRate(self.diff, stress, P, Ts, d, Coh, use_effective_strain_rate=use_effective_strain_rate)
        strain_rates_disl = CreepStrainRate(self.disl, stress, P, Ts, d, Coh, use_effective_strain_rate=use_effective_strain_rate)
        strain_rates = strain_rates_diff + strain_rates_disl  # apply the isostress model
        if ax is not None:
            ax.semilogy(1e4/Ts, strain_rates, '-', color=_color, label=(label + "comp"))
            ax.semilogy(1e4/Ts, strain_rates_diff, '--', color=_color, label=(label + "diff"))
            ax.semilogy(1e4/Ts, strain_rates_disl, '-.', color=_color, label=(label + "disl"))
        ax.set_xlabel("10^4 / T (K^-1)")
        ax.set_xlim(1e4 / T_range[1], 1e4 / T_range[0])
        if strain_rate_range is not None:
            ax.set_ylim(strain_rate_range[0], strain_rate_range[1])
        ax.set_ylabel("Strain Rate (s^-1)")
        _title = "P = %.4e, Stress = %.4e, d = %.4e, Coh = %.4e" % (P/1e6, stress, d, Coh)
        ax.set_title(_title)

    def MantleRheology(self, **kwargs):
        '''
        Derive mantle rheology from an aspect profile
        In this version, I would use the F factor (second invariant) as the default for computing the viscosity.
        Inputs:
            kwargs:
                rheology - type of rheology to use
                strain_rate - the strain rate used for viscosity estimation
                dEdiff, dVdiff, dAdiff_ratio, dAdisl_ratio, dEdisl, dVdisl - these factors
                    would apply a differences to the medium value in the flow law.
                save_profile - if the mantle profile of viscosity is saved as plot.
                save_json - if the derived mantle rheology is saved as a json file
                fig_path - if the save_profile is true, then a path of figure could be given
                    otherwise, a default path will be adapted.
        '''
        strain_rate = kwargs.get('strain_rate', 1e-15)
        use_effective_strain_rate = kwargs.get('use_effective_strain_rate', True)
        eta = np.ones(self.depths.size) 
        # these options are for a differences from the central value
        dEdiff = float(kwargs.get('dEdiff', 0.0))  # numbers for the variation in the rheology
        dVdiff = float(kwargs.get('dVdiff', 0.0))
        dAdiff_ratio = float(kwargs.get("dAdiff_ratio", 1.0))
        dAdisl_ratio = float(kwargs.get("dAdisl_ratio", 1.0))
        dEdisl = float(kwargs.get('dEdisl', 0.0))
        dVdisl = float(kwargs.get('dVdisl', 0.0))
        save_pdf = kwargs.get("save_pdf", False)
        rheology = kwargs.get('rheology', 'HK03_wet_mod')
        save_profile = kwargs.get('save_profile', 0)
        save_json = kwargs.get('save_json', 0)
        debug = kwargs.get('debug', False)
        fig_path = kwargs.get("fig_path", None)
        Coh = kwargs.get("Coh", 1000.0)
        assign_rheology = kwargs.get("assign_rheology", False)
        ymax = kwargs.get("ymax", 2890.0) # km, only used for plots

        
        eta_diff = np.ones(self.depths.size)
        eta_disl = np.ones(self.depths.size)
        eta_disl13 = np.ones(self.depths.size)
        eta13 = np.ones(self.depths.size)

        # First, read in the flow law and apply the diffeorence to the medium value 
        if assign_rheology:
            diffusion_creep = kwargs["diffusion_creep"]
            dislocation_creep = kwargs['dislocation_creep']
        else:
            diffusion_creep, dislocation_creep = GetRheology(rheology)
            diffusion_creep['A'] *= dAdiff_ratio
            diffusion_creep['E'] += dEdiff
            dislocation_creep['A'] *= dAdisl_ratio
            dislocation_creep['E'] += dEdisl
            diffusion_creep['V'] += dVdiff
            dislocation_creep['V'] += dVdisl
        self.diff = diffusion_creep  # record these with the class variables
        self.disl = dislocation_creep

        strain_rate_diff_correction = kwargs.get("strain_rate_diff_correction", 1.0)
        strain_rate_disl_correction = kwargs.get("strain_rate_disl_correction", 1.0)
        eta_diff_correction = strain_rate_diff_correction ** (1 / diffusion_creep['n'])
        eta_disl_correction = strain_rate_disl_correction ** (1 / dislocation_creep['n'])

        # Then, convert T, P as function. The T_func and P_func are used
        # in the following code to get the values
        T_func = interp1d(self.depths, self.temperatures, assume_sorted=True)
        P_func = interp1d(self.depths, self.pressures, assume_sorted=True)

        # okay, we are ready
        # Start by getting the rheology < 410 km
        depth_up = 410e3
        depth_low = 660e3
        mask_up = (self.depths < depth_up)
        eta_diff[mask_up] = CreepRheology(diffusion_creep, strain_rate, self.pressures[mask_up],\
                                          self.temperatures[mask_up], d=1e4, Coh=Coh,\
                                            use_effective_strain_rate=use_effective_strain_rate)\
                                        * eta_diff_correction
        eta_disl[mask_up] = CreepRheology(dislocation_creep, strain_rate, self.pressures[mask_up],\
                                           self.temperatures[mask_up], d=1e4, Coh=Coh,\
                                              use_effective_strain_rate=use_effective_strain_rate)\
                                        * eta_disl_correction
        eta_disl13[mask_up] = CreepRheology(dislocation_creep, 1e-13, self.pressures[mask_up],\
                                            self.temperatures[mask_up], d=1e4, Coh=Coh,\
                                                use_effective_strain_rate=use_effective_strain_rate)\
                                        * eta_disl_correction
        eta[mask_up] = ComputeComposite(eta_diff[mask_up], eta_disl[mask_up])
        eta13[mask_up] = ComputeComposite(eta_diff[mask_up], eta_disl13[mask_up])


        # then, the rheology in the MTZ
        # Now there is no differences from the scenario we used in the upper mantle
        # in the future, more will be added.
        mask_mtz = (self.depths > depth_up) & (self.depths < depth_low)
        if True:
            # MTZ from olivine rheology
            eta_diff[mask_mtz] = CreepRheology(diffusion_creep, strain_rate, self.pressures[mask_mtz],\
                                               self.temperatures[mask_mtz], d=1e4, Coh=Coh,\
                                                use_effective_strain_rate=use_effective_strain_rate)\
                                            * eta_diff_correction
            eta_disl[mask_mtz] = CreepRheology(dislocation_creep, strain_rate, self.pressures[mask_mtz],\
                                                self.temperatures[mask_mtz], d=1e4, Coh=Coh,\
                                                    use_effective_strain_rate=use_effective_strain_rate)\
                                            * eta_disl_correction
            eta_disl13[mask_mtz] = CreepRheology(dislocation_creep, 1e-13, self.pressures[mask_mtz],\
                                                 self.temperatures[mask_mtz], d=1e4, Coh=Coh,\
                                                    use_effective_strain_rate=use_effective_strain_rate)\
                                            * eta_disl_correction
            eta[mask_mtz] = ComputeComposite(eta_diff[mask_mtz], eta_disl[mask_mtz])
            eta13[mask_mtz] = ComputeComposite(eta_diff[mask_mtz], eta_disl13[mask_mtz])
        
        # At last, the lower mantle
        # The diffusion creep is assumed to be the only activated mechanism in the lower mantle.
        mask_low = (self.depths > depth_low)
        jump_lower_mantle = kwargs.get('jump_lower_mantle', 30.0)
        # Computing V in the lower mantle.
        # For this, we need the T, P on the 660 boundary
        depth_lm = 660e3
        depth_max = self.depths[-1] - 10e3
        T660 = T_func(depth_lm)
        P660 = P_func(depth_lm)
        eta_diff660 = CreepRheology(diffusion_creep, strain_rate, P660, T660, d=1e4, Coh=Coh,\
                                    use_effective_strain_rate=use_effective_strain_rate)\
                                * eta_diff_correction
        # dislocation creep
        eta_disl660 = CreepRheology(dislocation_creep, strain_rate, P660, T660, d=1e4, Coh=Coh,\
                                    use_effective_strain_rate=use_effective_strain_rate)\
                                * eta_disl_correction
        eta660 = ComputeComposite(eta_diff660, eta_disl660)
        if debug:
            print("eta_diff660 = ", eta_diff660)
            print("eta_disl660 = ", eta_disl660)
            print("eta_comp660 = ", eta660)
        diff_lm = diffusion_creep.copy()
        diff_lm['V'] = 3e-6  # assign a value
        diff_lm['A'] = CreepComputeA(diff_lm, strain_rate, P660, T660, eta660*jump_lower_mantle, d=1e4, Coh=Coh, use_effective_strain_rate=use_effective_strain_rate)
        
        # dump json file 
        constrained_rheology = {'diffusion_creep': diffusion_creep, 'dislocation_creep': dislocation_creep, 'diffusion_lm': diff_lm}
        if debug:
            print("constrained_rheology: ")
            print(constrained_rheology)
        
        # convert aspect rheology
        if debug:
            print('diffusion_creep: ', diffusion_creep)
        diffusion_creep_aspect = Convert2AspectInput(diffusion_creep, Coh=Coh, use_effective_strain_rate=use_effective_strain_rate)
        diffusion_lm_aspect = Convert2AspectInput(diff_lm, Coh=Coh, use_effective_strain_rate=use_effective_strain_rate)
        dislocation_creep_aspect = Convert2AspectInput(dislocation_creep, Coh=Coh, use_effective_strain_rate=use_effective_strain_rate)
        constrained_rheology_aspect = {'diffusion_creep': diffusion_creep_aspect, 'dislocation_creep': dislocation_creep_aspect, 'diffusion_lm': diffusion_lm_aspect}
        constrained_viscosity_profile = {'T': None, 'P': None, 'diffusion': None, 'dislocation': None,\
                                        'composite': None, 'dislocation_13': None,\
                                        'composite_13': None, 'depth': None}
        
        
        # lower mnatle rheology
        eta_diff[mask_low] = CreepRheologyInAspectViscoPlastic(diffusion_lm_aspect, strain_rate, self.pressures[mask_low], self.temperatures[mask_low])
        eta_disl[mask_low] = None  # this is just for visualization
        eta_disl13[mask_low] = None  # this is just for visualization
        eta[mask_low] = eta_diff[mask_low]  # diffusion creep is activated in lower mantle
        eta13[mask_low] = eta_diff[mask_low]  # diffusion creep is activated in lower mantle
        
        # Next, we visit some constraints for whole manlte rheology 
        # to see whether we match them
        # The haskel constraint
        radius = 6371e3
        lith_depth = 100e3
        integral_depth = 1400e3
        mask_integral = (self.depths > lith_depth) & (self.depths < integral_depth)
        integral_cores = 4 * np.pi * (radius - self.depths)**2.0
        # upper mantle
        # use harmonic average
        # lower mantle
        integral = np.trapz(integral_cores[mask_integral] * np.log10(eta[mask_integral]), self.depths[mask_integral])
        volume = np.trapz(integral_cores[mask_integral], self.depths[mask_integral])
        average_log_eta = integral / volume
        if save_json == 1:
            json_path = os.path.join(RESULT_DIR, "mantle_profile_v1_%s_dEdiff%.4e_dEdisl%.4e_dVdiff%4e_dVdisl%.4e_dAdiff%.4e_dAdisl%.4e.json" % (rheology, dEdiff, dEdisl, dVdiff, dVdisl, dAdiff_ratio, dAdisl_ratio))
            json_path_aspect = os.path.join(RESULT_DIR, "mantle_profile_aspect_v1_%s_dEdiff%.4e_dEdisl%.4e_dVdiff%4e_dVdisl%.4e_dAdiff%.4e_dAdisl%.4e.json" % (rheology, dEdiff, dEdisl, dVdiff, dVdisl, dAdiff_ratio, dAdisl_ratio))
            with open(json_path, 'w') as fout:
                json.dump(constrained_rheology, fout)
            with open(json_path_aspect, 'w') as fout:
                json.dump(constrained_rheology_aspect, fout)
            print("New json: %s" % json_path)
            print("New json: %s" % json_path_aspect)
            self.output_json = json_path
            self.output_json_aspect = json_path_aspect

        # save the constrained viscosity profile
        constrained_viscosity_profile['depth'] = self.depths.copy() 
        constrained_viscosity_profile['T'] = self.temperatures.copy() 
        constrained_viscosity_profile['P'] = self.pressures.copy() 
        constrained_viscosity_profile['diffusion'] = eta_diff.copy()
        constrained_viscosity_profile['dislocation'] = eta_disl.copy()
        constrained_viscosity_profile['composite'] = eta.copy()
        constrained_viscosity_profile['composite_13'] = eta13.copy()
        constrained_viscosity_profile['dislocation_13'] = eta_disl13.copy()
        
        # plot the profile of viscosity if it's required
        if save_profile == 1:
            # plots
            ylim=[ymax, 0.0]
            masky = (self.depths/1e3 < ymax)
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            color = 'tab:blue'
            axs[0].plot(self.pressures/1e9, self.depths/1e3, color=color, label='pressure')
            axs[0].set_ylabel('Depth [km]') 
            axs[0].set_xlabel('Pressure [GPa]', color=color)
            # axs[0].set_xlabel('Pressure [GPa] P660: %.4e' % (P660), color=color)
            Pmax = np.ceil(np.max(self.pressures[masky]/1e9) / 10.0) *10.0
            axs[0].set_xlim([0.0, Pmax])
            # axs[0].invert_yaxis()
            axs[0].set_ylim(ylim)
            # ax2: temperature
            color = 'tab:red'
            ax2 = axs[0].twiny()
            ax2.set_ylim(ylim)
            ax2.plot(self.temperatures, self.depths/1e3, color=color, label='temperature')
            Tmax = np.ceil(np.max(self.temperatures[masky]) / 100.0) *100.0
            ax2.set_xlim([0.0, Tmax])
            ax2.set_xlabel('Temperature [K]', color=color) 
            # ax2.set_xlabel('Temperature [K] T660: %.4e' % (T660), color=color) 
            # second: viscosity
            #   upper mantle
            axs[1].semilogx(eta_diff, self.depths/1e3, 'c', label='diffusion creep')
            axs[1].semilogx(eta_disl, self.depths/1e3, 'g', label='dislocation creep(%.2e)' % strain_rate)
            axs[1].semilogx(eta, self.depths/1e3, 'r--', label='Composite')
            axs[1].set_xlim([1e18,1e25])
            axs[1].set_ylim(ylim)
            axs[1].grid()
            axs[1].set_ylabel('Depth [km]')
            axs[1].legend()
            # axs[1].set_title('%s_lowerV_%.4e_haskell%.2f' % (rheology, diffusion_lm_aspect['V'], average_log_eta))
            # third, viscosity at 1e-13 /s strain rate
            axs[2].semilogx(eta_diff, self.depths/1e3, 'c', label='diffusion creep')
            axs[2].semilogx(eta_disl13, self.depths/1e3, 'g', label='dislocation creep(%.2e)' % 1e-13)
            axs[2].semilogx(eta13, self.depths/1e3, 'r--', label='Composite')
            axs[2].set_xlim([1e18,1e25])
            axs[2].set_ylim(ylim)
            axs[2].grid()
            axs[2].set_ylabel('Depth [km]')
            axs[2].legend()
            # axs[2].set_title('strain_rate1.0e-13')
            fig.tight_layout()
            # save figure
            if fig_path == None:
                fig_path = os.path.join(RESULT_DIR,\
                    "mantle_profile_v1_%s_dEdiff%.4e_dEdisl%.4e_dVdiff%4e_dVdisl%.4e_dAdiff%.4e_dAdisl%.4e.png"\
                    % (rheology, dEdiff, dEdisl, dVdiff, dVdisl, dAdiff_ratio, dAdisl_ratio))
            fig.savefig(fig_path)
            print("New figure: %s" % fig_path)
            if save_pdf:
                fig_path = os.path.join(RESULT_DIR,\
                    "mantle_profile_v1_%s_dEdiff%.4e_dEdisl%.4e_dVdiff%4e_dVdisl%.4e_dAdiff%.4e_dAdisl%.4e.pdf"\
                    % (rheology, dEdiff, dEdisl, dVdiff, dVdisl, dAdiff_ratio, dAdisl_ratio))
                fig.savefig(fig_path)
                print("New figure: %s" % fig_path)
            plt.close()
            self.output_profile = fig_path
            pass
        return constrained_rheology_aspect, constrained_viscosity_profile
    
    def ConstrainRheology(self, **kwargs):
        '''
        varying around a give rheology with variation with applied constraints
        Version 1:
            0. use modified wet rheology see the file from Magali
            1. compute V with rheology at 250km depth, this is in turn, constraint in a range
            2. dislocation creep is bigger than diffusion creep at 300e3.  
            3. value of rheology on 660 within a range
            Lower mantle is then assigned so that: 
                1. only diffusion creep is activated. 
                2 viscosity is nearly constant by controlling V. 
                3. There is a 30 times jump between u/l boundary by controlling A. 
        inputs: 
            kwargs(dict):
                rheology: type of initial rheology
        '''
        constrained_rheologies = []
        constrained_ds = []
        constrained_Vdisls = []
        constrained_Edisls = []
        constrained_Vdiffs = []
        constrained_Ediffs = []
        rheology = kwargs.get('rheology', 'HK03_wet_mod')
        N = 1001
        
        # make a new directory
        fig_dir = os.path.join(RESULT_DIR, 'constrained_rheology_v1_%s_N%d' % (rheology, N))
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)
        else:
            rmtree(fig_dir)
            os.mkdir(fig_dir)
        
        # get rheology
        diffusion_creep, dislocation_creep = GetRheology(rheology)
        diff_orig = diffusion_creep.copy()
        disl_orig = dislocation_creep.copy()
        orig_rheology = {'diff': diff_orig, 'disl': disl_orig}
        json_path = os.path.join(fig_dir, "original_profile.json")
        with open(json_path, 'w') as fout:
            json.dump(orig_rheology, fout)
            print("New json: %s" % json_path)
        
        lm_diff = {}
        strain_rate = 1e-15
        radius = kwargs.get('radius', 6371e3)  # earth radius
        
        T_func = interp1d(self.depths, self.temperatures, assume_sorted=True)
        P_func = interp1d(self.depths, self.pressures, assume_sorted=True)

        # lower mantle
        depth_lm = 660e3
        T_lm_mean = T_func(1700e3)
        P_lm_mean = P_func(1700e3)
        depth_max = self.depths[-1] - 10e3
        lm_grad_T = (T_func(depth_max) - T_func(depth_lm)) / (depth_max - depth_lm)
        lm_grad_P = (P_func(depth_max) - P_func(depth_lm)) / (depth_max - depth_lm)

        # grain size
        d_mean = kwargs.get('d', 0.75e4)

        # range of sampling
        Vdiff_sigma = 5.5e-6
        Ediff_sigma = 40e3
        Vdisl_sigma = 4e-6 # 10e-6
        Edisl_sigma = 50e3 # 10e-6
        d_sigma = 5e3

        # random sampling
        Ediffs = np.random.normal(diffusion_creep['E'], Ediff_sigma, N)
        Edisls = np.random.normal(dislocation_creep['E'], Edisl_sigma, N)
        ds = np.random.normal(d_mean, d_sigma, N)
        include_lower_mantle = kwargs.get('include_lower_mantle', None)
        mask_um = (self.depths < depth_lm)  # a mask to get the components of the upper mantle
        mask_lm = (self.depths >= depth_lm)  # a mask to get the components of the lower mantle
        for i in range(N):
            Ediff = Ediffs[i]
            Edisl = Edisls[i]
            d = ds[i]
            if Ediff < 0.0 or Edisl < 0.0 or d <= 0.0:
                continue
            diffusion_creep['E'] = Ediff
            diffusion_creep['d'] = d
            dislocation_creep['E'] = Edisl
            dislocation_creep['d'] = d
            diffusion_creep['Edev'] = Ediff - diff_orig['E']
            dislocation_creep['Edev'] = Edisl - disl_orig['E']
            if (abs(diffusion_creep['Edev'] > Ediff_sigma) or abs(dislocation_creep['Edev'] > Edisl_sigma)):
                # check Edev is in range
                continue

            # 250km
            depth1 = 250e3
            eta250 = 5e19
            T1 = T_func(depth1)
            P1 = P_func(depth1)
            # diffusion creep
            # compute V, viscosities from both creeps are equal
            Vdiff = CreepComputeV(diffusion_creep, strain_rate, P1, T1, 2*eta250, d=d)
            diffusion_creep['V'] = Vdiff
            Vdisl = CreepComputeV(dislocation_creep, strain_rate, P1, T1, 2*eta250, d=d, use_effective_strain_rate=True)
            dislocation_creep['V'] = Vdisl
            if Vdiff < 0.0 or Vdisl < 0.0:
                continue
            diffusion_creep['Vdev'] = Vdiff - diff_orig['V']
            dislocation_creep['Vdev'] = Vdisl - disl_orig['V']

            # 660 km 
            # diffusion creep
            T660 = T_func(depth_lm)
            P660 = P_func(depth_lm)
            eta_diff660 = CreepRheology(diffusion_creep, strain_rate, P660, T660)
            # dislocation creep
            eta_disl660 = CreepRheology(dislocation_creep, strain_rate, P660, T660, use_effective_strain_rate=True)
            eta660 = ComputeComposite(eta_diff660, eta_disl660)
            
            # other constraints
            depth2 = 300e3
            Ttemp = T_func(depth2)
            Ptemp = P_func(depth2)
            # diffusion creep
            eta_diff300 = CreepRheology(diffusion_creep, strain_rate, Ptemp, Ttemp)
            # dislocation creep
            eta_disl300 = CreepRheology(dislocation_creep, strain_rate, Ptemp, Ttemp, use_effective_strain_rate=True)

            if include_lower_mantle is not None:
                # lower mantle rheology
                diff_lm = diffusion_creep.copy()
                diff_lm['V'] = LowerMantleV(diffusion_creep['E'], T_lm_mean, P_lm_mean, lm_grad_T, lm_grad_P)
                diff_lm['A'] = CreepComputeA(diff_lm, strain_rate, P660, T660, eta660*include_lower_mantle)

            
            # 1000km integral
            eta_diff = CreepRheology(diffusion_creep, strain_rate, self.pressures, self.temperatures)
            eta_disl = CreepRheology(dislocation_creep, strain_rate, self.pressures, self.temperatures, use_effective_strain_rate=True)
            eta = ComputeComposite(eta_diff, eta_disl)
            lith_depth = 100e3
            integral_depth = 1400e3
            mask_integral = (self.depths > lith_depth) & (self.depths < integral_depth)
            integral_cores = 4 * np.pi * (radius - self.depths)**2.0
            # upper mantle
            mask_um_integral = (mask_um & mask_integral)
            integral_um = np.trapz(eta[mask_um_integral] * integral_cores[mask_um_integral], self.depths[mask_um_integral])
            # lower mantle
            integral_lm = 0.0
            if include_lower_mantle is not None:
                mask_lm_integral = (mask_lm & mask_integral)
                eta_diff_lm = CreepRheology(diff_lm, strain_rate, self.pressures, self.temperatures)
                integral_lm = np.trapz(eta_diff_lm[mask_lm_integral] * integral_cores[mask_lm_integral], self.depths[mask_lm_integral])
            else:
                integral_lm = 4.0 / 3 * np.pi * ((radius - depth_lm)**3.0 - (radius - integral_depth)**3.0) * eta660 * 30.0 # assume 30 times jump
            volume = 4.0 / 3 * np.pi * ((radius - lith_depth)**3.0 - (radius - integral_depth)**3.0)
            average_eta = (integral_um + integral_lm) / volume

            # conditions:
            #   0: 300km
            #   1: 660km
            #   2: integral
            #   3: range of V
            eta660range = [4e20, 1e21]
            average_range = [0.65e21, 1.1e21]
            conds = [(eta_disl300 < eta_diff300),\
            (eta660 >= eta660range[0]) and (eta660 <= eta660range[1]),\
            (average_eta >= average_range[0]) and (average_eta <= average_range[1]),\
            abs(diffusion_creep['Vdev']) < Vdiff_sigma and abs(dislocation_creep['Vdev']) < Vdisl_sigma]
            # failed info
            # print(conds)
            condition_indexes = [0, 1, 3]
            cond_combined = True
            for i in condition_indexes:
                cond_combined = (cond_combined and conds[i])
            
            if cond_combined:
                constrained_ds.append(d)
                constrained_Vdisls.append(Vdisl)
                constrained_Vdiffs.append(Vdiff)
            else:
                constrained_rheologies.append({'diff': diffusion_creep.copy(), 'disl': dislocation_creep.copy(),\
                'average_upper_region': average_eta})


        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(constrained_Ediffs, constrained_Edisls, constrained_ds)
        ax.set_xlabel('Ediff [J/mol]')
        ax.set_ylabel('Edisl [J/mol]')
        ax.set_zlabel('d [um]')
        fig_name = 'constrained_rheology_%s_N%d.png' % (rheology, N)
        fig_path = os.path.join(fig_dir, fig_name)
        fig.savefig(fig_path)
        print("New figure: %s" % fig_path)
        plt.close()

        # plot profiles
        save_profile = kwargs.get('save_profile', 0)
        i = 0  # index
        for constrained_rheology in constrained_rheologies:
            # dump json
            Vdiff = constrained_rheology['diff']['V']
            Vdisl = constrained_rheology['disl']['V']
            Ediff = constrained_rheology['diff']['E']
            Edisl = constrained_rheology['disl']['E']
            d = constrained_rheology['disl']['d']
            json_path = os.path.join(fig_dir, "constrained_profile_Ediff%.4e_Edisl%.4e_Vdiff%.4e_Vdisl%.4e_d%.4e.json" % (Ediff, Edisl, Vdiff, Vdisl, d))
            with open(json_path, 'w') as fout:
                json.dump(constrained_rheology, fout)
                print("[%d / %d], New json: %s" % (i, len(constrained_rheologies), json_path))
            #  save profile
            if save_profile == 1:
                eta_diff = CreepRheology(constrained_rheology['diff'], strain_rate, self.pressures, self.temperatures)
                eta_disl = CreepRheology(constrained_rheology['disl'], strain_rate, self.pressures, self.temperatures, use_effective_strain_rate=True)
                eta = ComputeComposite(eta_diff, eta_disl)
                # plots
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                color = 'tab:blue'
                axs[0].plot(self.pressures/1e9, self.depths/1e3, color=color, label='pressure')
                axs[0].set_ylabel('Depth [km]') 
                axs[0].set_xlabel('Pressure [GPa]', color=color) 
                # axs[0].invert_yaxis()
                if include_lower_mantle is None:
                    ylim=[660.0, 0.0]
                else:
                    ylim=[2890, 0.0]
                axs[0].set_ylim(ylim)
                # ax2: temperature
                color = 'tab:red'
                ax2 = axs[0].twiny()
                ax2.set_ylim(ylim)
                ax2.plot(self.temperatures, self.depths/1e3, color=color, label='temperature')
                ax2.set_xlabel('Temperature [K]', color=color) 
                # second: viscosity
                #   upper mantle
                axs[1].semilogx(eta_diff[mask_um], self.depths[mask_um]/1e3, 'c', label='diffusion creep')
                axs[1].semilogx(eta_disl[mask_um], self.depths[mask_um]/1e3, 'g', label='dislocation creep(%.2e)' % strain_rate)
                axs[1].semilogx(eta[mask_um], self.depths[mask_um]/1e3, 'r--', label='Composite')
                axs[1].set_xlim([1e18,1e25])
                axs[1].set_ylim(ylim)
                # axs[1].invert_yaxis()
                axs[1].grid()
                axs[1].set_ylabel('Depth [km]')
                axs[1].legend()
                if include_lower_mantle:
                    # include lower mantle info as title:
                    diff_lm = constrained_rheology['diff_lm']
                    eta_diff_lm = CreepRheology(constrained_rheology['diff_lm'], strain_rate, self.pressures, self.temperatures)
                    axs[1].semilogx(eta_diff_lm[mask_lm], self.depths[mask_lm]/1e3, 'c')
                    title_str = "lm_jump%.2d_Vlm%.4e_avg%.4e" % (include_lower_mantle, diff_lm['V'], constrained_rheology['average_upper_region'])
                    axs[1].set_title(title_str)
                # save figure
                fig_path = os.path.join(fig_dir, "constrained_profile_Ediff%.4e_Edisl%.4e_Vdiff%.4e_Vdisl%.4e_d%.4e.png" % (Ediff, Edisl, Vdiff, Vdisl, d))
                fig.savefig(fig_path)
                print("[%d / %d], New figure: %s" % (i, len(constrained_rheologies), fig_path))
                plt.close()
            i = i + 1

def FastZeroStep(Inputs, output_first_step = False):
    '''
    Generate a prm file to run only the 0th step, and real fast
    Inputs:
        output_first_step: if true, then output the 1st step as well.
    '''
    Inputs['Nonlinear solver scheme'] = 'no Advection, no Stokes'
    if output_first_step:
        time_between_graphical_output = None
        try:
            time_between_graphical_output = Inputs['Postprocess']['Visualization']['Time between graphical output']
        except KeyError:
            raise KeyError("input file has to have \'Time between graphical output\' if the option of output_first_step is True")
        Inputs['End time'] = time_between_graphical_output
    else:
        Inputs['End time'] = '0' # end time is 0
        # don't solve it

def TestInitalSteps(Inputs, n_outputs, output_interval):
    '''
    options for generating a case to test the initial steps
    '''
    Inputs['End time'] = str(output_interval * n_outputs)
    # output timing information at every step
    Inputs["Timing output frequency"] = "0"
    # fix post-process section
    pp_dict = Inputs["Postprocess"]
    if "Depth average" in pp_dict:
        # output Depth average at every step
        pp_dict["Depth average"]["Time between graphical output"] = "0"
    if "Visualization" in pp_dict:
        pp_dict["Visualization"]["Time between graphical output"] = str(output_interval)
    if "Particles" in pp_dict:
        pp_dict["Particles"]["Time between data output"] = str(output_interval)
    # fix checkpointing - chekcpoitn eveyr step
    Inputs["Checkpointing"] = {
        "Steps between checkpoint": "1"
    }

def ParseFromSlurmBatchFile(fin):
    '''
    read options from a slurm batch file.
    Note I allow multiple " " in front of each line, this may or may not be a good option.
    '''
    inputs = {}
    inputs["header"] = []
    inputs["config"] = {}
    inputs["load"] = []
    inputs["unload"] = []
    inputs["command"] = []
    inputs["others"] = []
    line = fin.readline()
    while line != "":
        if re.match('^(\t| )*#!', line):
            line1 = re.sub('(\t| )*\n$', '', line)  # eliminate the \n at the end
            inputs["header"].append(line1)
        elif re.match('^(\t| )*#(\t| )*SBATCH', line):
            line1 = re.sub('^(\t| )*#(\t| )*SBATCH ', '', line, count=1)
            key, value = ReadDashOptions(line1)
            inputs["config"][key] = value
        elif re.match('(\t| )*module(\t| )*load', line):
            line1 = re.sub('(\t| )*module(\t| )*load(\t| )*', '', line, count=1)
            value = re.sub('(\t| )*\n$', '', line1) 
            inputs["load"].append(value)
        elif re.match('(\t| )*module(\t| )*unload', line):
            line1 = re.sub('(\t| )*module(\t| )*unload(\t| )*', '', line, count=1)
            value = re.sub('(\t| )*\n$', '', line1) 
            inputs["unload"].append(value)
        elif re.match('(\t| )*srun', line):
            temp = re.sub('(\t| )*srun(\t| )*', '', line, count=1)
            inputs["command"].append("srun")
            line1 = re.sub('(\t| )*\n$', '',temp)  # eliminate the \n at the end
            for temp in line1.split(' '):
                if not re.match("^(\t| )*$", temp):
                    # not ' '
                    temp1 = re.sub('^(\t| )*', '', temp) # eliminate ' '
                    value = re.sub('^(\t| )$', '', temp1)
                    inputs["command"].append(value)
        elif re.match('(\t| )*ibrun', line):
            temp = re.sub('(\t| )*ibrun(\t| )*', '', line, count=1)
            inputs["command"].append("ibrun")
            line1 = re.sub('(\t| )*\n$', '',temp)  # eliminate the \n at the end
            for temp in line1.split(' '):
                if not re.match("^(\t| )*$", temp):
                    # not ' '
                    temp1 = re.sub('^(\t| )*', '', temp) # eliminate ' '
                    value = re.sub('^(\t| )$', '', temp1)
                    inputs["command"].append(value)
        else:
            inputs["others"].append(re.sub('(\t| )*\n$', '', line))
            pass
        line = fin.readline()
    return inputs

def ParseToSlurmBatchFile(fout, outputs, **kwargs):
    '''
    export options to a slurm batch file.
    '''
    contents = ""
    # write header
    for header in outputs["header"]:
        contents += (header + "\n")
    # write config
    for key, value in outputs["config"].items():
        if re.match('^--', key):
            contents += ("#SBATCH" + " " + key + "=" + value + "\n")
        elif re.match('^-', key):
            contents += ("#SBATCH" + " " + key + " " + value + "\n")
        else:
            raise ValueError("The format of key (%s) is incorrect" % key)
    # load and unload 
    for module in outputs["unload"]:
        contents += ("module unload %s\n" % module)
    for module in outputs["load"]:
        contents += ("module load %s\n" % module)
    # others
    for line in outputs['others']:
        contents += (line + '\n')
    # command
    is_first = True
    for component in outputs["command"]:
        if is_first:
            is_first = False
        else:
            contents += " "
        contents += component
        if component == "mpirun":
            # append the cpu options after mpirun
            contents += " "
            contents += ("-n " + outputs['config']['-n'])
    # extra contents
    if 'extra' in outputs:
        if outputs['extra'] is not None:
            contents += outputs['extra']
            contents += '\n'
    fout.write(contents)
    pass

class SLURM_OPERATOR():
    '''
    Class for modifying slurm operation files
    Attributes:
        i_dict (dict): input options
        o_dict (dict): output options
    '''
    def __init__(self, slurm_base_file_path):
        my_assert(FileExistsError, os.path.isfile(slurm_base_file_path), "%s doesn't exist" % slurm_base_file_path)
        with open(slurm_base_file_path, 'r') as fin:
            self.i_dict = ParseFromSlurmBatchFile(fin)
        self.check() # call the check function to check the contents of the file
        self.o_dict = {}
    
    def check(self):
        '''
        check the options in inputs
        '''
        assert('-N' in self.i_dict['config'])
        assert('-n' in self.i_dict['config'])
        assert('--threads-per-core' in self.i_dict['config'])
        assert('--tasks-per-node' in self.i_dict['config'])
        assert('--partition' in self.i_dict['config'])

    def SetAffinity(self, nnode, nthread, nthreads_per_cpu, **kwargs):
        '''
        set options for affinities
        '''
        partition = kwargs.get('partition', None)
        use_mpirun = kwargs.get('use_mpirun', False)
        bind_to = kwargs.get('bind_to', None)
        self.o_dict = deepcopy(self.i_dict)
        self.o_dict['config']['-N'] = str(int(nnode))
        self.o_dict['config']['-n'] = str(nthread)
        self.o_dict['config']['--threads-per-core'] = str(nthreads_per_cpu)
        self.o_dict['config']['--tasks-per-node'] = str(int(nthread//nnode))
        if partition is not None:
             self.o_dict['config']['--partition'] = partition 
        if use_mpirun:
            self.o_dict['command'][0] = 'mpirun'
        if bind_to != None:
            assert(bind_to in ["socket", 'core'])
            if self.o_dict['command'][0] == "mpirun":
                self.o_dict['command'].insert(1, "--bind-to " + bind_to)
            if self.o_dict['command'][0] == "srun":
                self.o_dict['command'].insert(1, "--cpu-bind=" + bind_to + "s")

    def SetName(self, _name):
        '''
        set the name of the job
        '''
        self.o_dict['config']['--job-name'] = _name

    def SetModule(self, module_list, module_u_list=[]):
        '''
        set the list of module to load
        '''
        assert(type(module_list) == list)
        self.o_dict['load'] = module_list
        self.o_dict['unload'] = module_u_list

    def SetCommand(self, build_directory, prm_file):
        '''
        Set the command to use
        '''
        if build_directory != "":
            self.o_dict['command'][-2] = "${ASPECT_SOURCE_DIR}/build_%s/aspect" % build_directory
        else:
            self.o_dict['command'][-2] = "${ASPECT_SOURCE_DIR}/build/aspect"
        self.o_dict['command'][-1] = prm_file

    def ResetCommand(self, **kwargs):
        '''
        reset the command
        '''
        command_only = kwargs.get("command_only", False)
        self.o_dict['command'] = []
        if not command_only:
            self.o_dict['others'] = []

    def SetTimeByHour(self, hr):
        '''
        set the time limit
        '''
        self.o_dict['config']['-t'] = "%d:00:00" % hr
    
    def SetExtra(self, contents):
        '''
        set extra commands
        '''
        self.o_dict['extra'] = contents
    
    def GetOthers(self):
        '''
        get other commands
        '''
        contents = self.o_dict['others']
        return contents

    def __call__(self, slurm_file_path):
        with open(slurm_file_path, 'w') as fout:
            ParseToSlurmBatchFile(fout, self.o_dict)
        print("Slurm file created: ", slurm_file_path)

    pass

class CASE():
    '''
    class for a case
    Attributes:
        name(str):
            list for name of variables to change
        idict(dict):
            dictionary of parameters
        wb_dict(idit):
            dictionary of world builder options
        extra_files(array):
            an array of extra files of this case
        model_stages(int):]
            stages in a model: if > 0, then create multiple prm files
    '''
    # future: add interface for extra
    def __init__(self, case_name, inputs, if_wb, **kwargs):
        '''
        initiate from a dictionary
        Inputs:
            idict(dict):
                dictionary import from a base file
            if_wb(True or False):
                if use world builder
            kwargs:
                wb_inputs(dict or str):
                    inputs from a world builder file
        '''
        self.case_name = case_name
        self.extra_files = []
        self.wb_dict = {}
        self.model_stages = 1
        self.additional_idicts = []
        self.output_files = [] # for saving the path of files and images output from this class
        self.output_imgs = []
        self.particle_data = None
        if type(inputs)==dict:
            # direct read if dict is given
            print("    Read inputs from a dictionary")
            self.idict = deepcopy(inputs)
        elif type(inputs)==str:
            # read from file if file path is given. This has the virtual that the new dict is indepent of the previous one.
            print("    Read inputs from %s" % var_subs(inputs)) 
            with open(var_subs(inputs), 'r') as fin:
                self.idict = parse_parameters_to_dict(fin)
            pass
        else:
            raise TypeError("Inputs must be a dictionary or a string")
        # read world builder
        if if_wb:
            wb_inputs = kwargs.get('wb_inputs', {})
            if type(wb_inputs) == dict:
                # direct read if dict is given
                print("    Read world builder options from a dictionary")
                self.wb_dict = deepcopy(wb_inputs)
            elif type(wb_inputs)==str:
                # read from file if file path is given. This has the virtual that the new dict is indepent of the previous one.
                print("    Read world builder options from %s" % var_subs(wb_inputs))
                with open(var_subs(wb_inputs), 'r') as fin:
                    self.wb_dict = json.load(fin)
                pass
            else:
                raise TypeError("CASE:%s: wb_inputs must be a dictionary or a string" % func_name())
        # operator of rheology
        self.Rheology_Opr = RHEOLOGY_OPR()
        # profile read from the depth_average files
        da_inputs = kwargs.get('da_inputs', "")
        if da_inputs != "":
            depths, pressures, temperatures, adiabatic_temperatures =\
              ReadAspectProfile(da_inputs, include_adiabatic_temperature=True) 
            self.da_T_func = interp1d(depths, temperatures, assume_sorted=True)
            self.da_P_func = interp1d(depths, pressures, assume_sorted=True)
            self.da_Tad_func = interp1d(depths, adiabatic_temperatures, assume_sorted=True)

    
    def create(self, _root, **kwargs):
        '''
        create a new case
        Inputs:
            _root(str): a directory to put the new case
            **kwargs:
                "fast_first_step": generate another file for fast running the 0th step
                "slurm_opts": options for slurm files
        Return:
            case_dir(str): path to created case.
        '''
        is_reload = kwargs.get("is_reload", False)
        # folder
        is_tmp = kwargs.get("is_tmp", False)
        if is_tmp:
            case_dir = os.path.join(_root, "%s_tmp" % self.case_name)
        else:
            case_dir = os.path.join(_root, self.case_name)
        if os.path.isdir(case_dir):
           # remove old ones 
           rmtree(case_dir)
        os.mkdir(case_dir)
        output_files_dir = os.path.join(case_dir, 'configurations')
        os.mkdir(output_files_dir)
        img_dir = os.path.join(case_dir, 'img')
        os.mkdir(img_dir)
        output_img_dir = os.path.join(img_dir, 'initial_condition')
        os.mkdir(output_img_dir)
        # file output
        prm_out_path = os.path.join(case_dir, "case.prm")  # prm for running the case
        wb_out_path = os.path.join(case_dir, "case.wb")  # world builder file
        with open(prm_out_path, "w") as fout:
            save_parameters_from_dict(fout, self.idict)
        if self.model_stages > 1:
            assert(len(self.additional_idicts) == self.model_stages-1)
            for i in range(self.model_stages-1):
                prm_out_path = os.path.join(case_dir, "case_%d.prm" % (i+1))  # prm for running the case
                with open(prm_out_path, "w") as fout:
                    save_parameters_from_dict(fout, self.idict)
        # fast first step
        fast_first_step = kwargs.get('fast_first_step', 0) 
        if fast_first_step == 0:
            pass
        elif fast_first_step == 1:
            outputs = deepcopy(self.idict)
            prm_fast_out_path = os.path.join(case_dir, "case_f.prm")
            FastZeroStep(outputs)  # generate another file for fast running the 0th step
            with open(prm_fast_out_path, "w") as fout:
                save_parameters_from_dict(fout, outputs)
            
        elif fast_first_step == 2:
            outputs = deepcopy(self.idict)
            prm_fast_out_path = os.path.join(case_dir, "case_f.prm")
            FastZeroStep(outputs, True)  # generate another file for fast running the 0th step
            with open(prm_fast_out_path, "w") as fout:
                save_parameters_from_dict(fout, outputs)
        else:
            raise ValueError("The option for fast_first_step must by 0, 1, 2")

        # test initial steps
        test_initial_steps = kwargs.get('test_initial_steps', (-1, 0.0))
        my_assert(len(test_initial_steps)==2, ValueError, "test_initial_steps needs to have two components")
        test_initial_n_outputs = test_initial_steps[0]
        test_initial_outputs_interval = test_initial_steps[1]
        if test_initial_n_outputs > 0:
            outputs = deepcopy(self.idict)
            prm_fast_out_path = os.path.join(case_dir, "case_ini.prm")
            TestInitalSteps(outputs, test_initial_n_outputs, test_initial_outputs_interval)  # generate another file for fast running the 0th step
            with open(prm_fast_out_path, "w") as fout:
                save_parameters_from_dict(fout, outputs)
        # append extra files
        for path in self.extra_files:
            base_name = os.path.basename(path)
            path_out = os.path.join(case_dir, base_name)
            copy2(path, path_out)
        # world builder
        if self.wb_dict != {}:
            with open(wb_out_path, 'w') as fout:
                json.dump(self.wb_dict, fout, indent=2)
        # assign a particle.dat file that contains the coordinates of particles
        if self.particle_data is not None:
            particle_o_dir = os.path.join(case_dir, 'particle_file')
            if not os.path.isdir(particle_o_dir):
                os.mkdir(particle_o_dir)
            particle_file_path = os.path.join(particle_o_dir, 'particle.dat')
            with open(particle_file_path, 'w') as fout:
                output_particle_ascii(fout, self.particle_data)
        print("New case created: %s" % case_dir)
        # generate slurm files if options are included
        if is_reload:
            pass
        else: 
            slurm_opts = kwargs.get("slurm_opts", [])
            if len(slurm_opts) > 0:
                for slurm_opt in slurm_opts:
                    # SlurmOperator = SLURM_OPERATOR(self.slurm_base_path)
                    SlurmOperator = SLURM_OPERATOR(slurm_opt.get_base_path())
                    # SlurmOperator.SetAffinity(np.ceil(core_count/self.tasks_per_node), core_count, 1)
                    SlurmOperator.SetAffinity(*slurm_opt.to_set_affinity())
                    SlurmOperator.SetCommand(*slurm_opt.to_set_command())
                    SlurmOperator.SetName(slurm_opt.get_job_name())
                    appendix = ""
                    if is_tmp:
                        # append a marker if this is a tmp case
                        appendix = "_tmp"
                    SlurmOperator(os.path.join(os.path.dirname(slurm_opt.get_output_path()) + appendix, os.path.basename(slurm_opt.get_output_path())))
                    pass
        # copy paste files and figures generated
        for path in self.output_files:
            base_name = os.path.basename(path)
            path_out = os.path.join(output_files_dir, base_name)
            copy2(path, path_out)
        for path in self.output_imgs:
            base_name = os.path.basename(path)
            path_out = os.path.join(output_img_dir, base_name)
            copy2(path, path_out)
        return case_dir

    def configure(self, func, config, **kwargs):
        '''
        applies configuration for this case
        Inputs:
            func(a function), the form of it is:
                outputs = func(inputs, config)
        '''
        rename = kwargs.get('rename', None)
        if rename != None:
            # rename the case with the configuration
            self.idict, appendix = func(self.idict, config)
            self.case_name += appendix
        else:
            # just apply the configuration
            self.idict = func(self.idict, config)

    def configure_prm(self, refinement_level):
        '''
        configure the prm file
        '''
        o_dict = self.idict.copy()
        o_dict["Mesh refinement"]["Initial global refinement"] = str(refinement_level)
        self.idict = o_dict
        pass

    def configure_case_output_dir(self, case_o_dir):
        '''
        configure the output directory of the case
        '''
        o_dict = self.idict.copy()
        # directory to put outputs
        o_dict["Output directory"] = case_o_dir
        self.idict = o_dict
    
    def configure_wb(self):
        '''
        Configure world builder file
        '''
        pass
    
    def configure_final(self, _, __):
        '''
        finalize configuration
        '''
        pass

    def add_extra_file(self, path):
        '''
        add an extra file to list
        Inputs:
            path(str): an extra file
        '''
        self.extra_files.append(path)

    def set_end_step(self, end_step): 
        '''
        set the step to end the computation
        '''
        self.idict = SetEndStep(self.idict, end_step)


def create_case_with_json(json_opt, CASE, CASE_OPT, **kwargs):
    '''
    A wrapper for the CASES class
    Inputs:
        json_opt(str, dict): path or dict a json file
        kwargs (dict):
            update (bool): update existing cases?
    Returns:
        case_dir: return case directory
    '''
    print("%s: Creating case" % func_name())
    is_update = kwargs.get('update', True)  # this controls whether we update
    is_force_update = kwargs.get('force_update', False)  # when this is false, inputs is required to continue.
    update_flag = False # this marks whether there is impending update.
    fix_case_name = kwargs.get('fix_case_name', None)
    fix_base_dir = kwargs.get('fix_base_dir', None)
    fix_output_dir = kwargs.get('fix_output_dir', None)
    reset_refinement_level = kwargs.get('reset_refinement_level', None)
    fix_case_output_dir = kwargs.get('fix_case_output_dir', None)
    reset_stokes_solver_type = kwargs.get("reset_stokes_solver_type", None)
    end_step = kwargs.get("end_step", -1)
    is_reload = kwargs.get("is_reload", False)
    Case_Opt = CASE_OPT()
    # read in json options
    if type(json_opt) == str:
        if not os.access(json_opt, os.R_OK):
            raise FileNotFoundError("%s doesn't exist" % json_opt)
        Case_Opt.read_json(json_opt)
    elif type(json_opt) == dict:
        Case_Opt.import_options(json_opt)
    else:
        raise TypeError("Type of json_opt must by str or dict")
    # reset case properties
    if fix_case_name != None:
        Case_Opt.fix_case_name(fix_case_name)  # fix base dir, useful when creating a group of case from a folder
    if fix_base_dir != None:
        Case_Opt.fix_base_dir(fix_base_dir)  # fix base dir, useful when creating a group of case from a folder
    if fix_output_dir != None:
        Case_Opt.fix_output_dir(fix_output_dir)  # fix output dir, useful when doing tests
    if reset_refinement_level != None:
        Case_Opt.reset_refinement(reset_refinement_level)
    if fix_case_output_dir != None:
        Case_Opt.fix_case_output_dir(fix_case_output_dir)
    if reset_stokes_solver_type != None:
        Case_Opt.reset_stokes_solver_type(reset_stokes_solver_type)
    # check if the options make sense
    Case_Opt.check()
    # check if the case already exists. If so, only update if it is explicitly 
    # required
    case_dir_to_check = os.path.join(Case_Opt.o_dir(), Case_Opt.case_name())
    if os.path.isdir(case_dir_to_check):
        if is_update:
            update_flag = True
        else:
            print("Case %s already exists, aborting" % case_dir_to_check)
            return case_dir_to_check
    # Manage case files
    Case = CASE(*Case_Opt.to_init(), wb_inputs=Case_Opt.wb_inputs_path(), da_inputs=Case_Opt.da_inputs_path())
    if end_step > 0:
        # set end step
        Case.set_end_step(end_step)
    Case.configure_prm(*Case_Opt.to_configure_prm())
    if Case_Opt.if_use_world_builder():
        Case.configure_wb(*Case_Opt.to_configure_wb())
    # add extra files
    for _path in Case_Opt.get_additional_files():
        Case.add_extra_file(_path)
    # finalizing
    Case.configure_final(*(Case_Opt.to_configure_final()))
    # create new case
    if not is_reload:
        if update_flag:
            # update a previous case:
            # a. put new case into a dir - "case_tmp"
            # b. figure whether to update: if prm or wb files are different from the original case
            case_dir = os.path.join(Case_Opt.o_dir(), Case_Opt.case_name())
            case_dir_tmp = os.path.join(Case_Opt.o_dir(), "%s_tmp" % Case_Opt.case_name())
            if os.path.isdir(case_dir_tmp):
                rmtree(case_dir_tmp)
            Case.create(Case_Opt.o_dir(), fast_first_step=Case_Opt.if_fast_first_step(),\
                test_initial_steps=Case_Opt.test_initial_steps(), is_tmp=True,\
                    slurm_opts=Case_Opt.get_slurm_opts(), is_reload=is_reload)
            assert(os.path.isdir(case_dir_tmp))
            do_update = False # a flag to perform update, only true if the parameters are different
            # generate catalog: loop over files in the new folder and output the differences from
            contents = ""
            for _name in os.listdir(case_dir_tmp):
                file_newer = os.path.join(case_dir_tmp, _name)
                if not os.path.isfile(file_newer):
                    continue
                file_older = os.path.join(case_dir, _name)
                older_text = "" # if no older files are present, the text is vacant.
                if os.path.isfile(file_older):
                    with open(file_older, 'r') as fin0:
                        try:
                            older_text = fin0.readlines()
                        except UnicodeDecodeError:
                            continue
                with open(file_newer, 'r') as fin1:
                    try:
                        newer_text = fin1.readlines()
                    except Exception:
                        continue
                diff_results = unified_diff(older_text, newer_text, fromfile=file_older, tofile=file_newer, lineterm='')
                for line in diff_results:
                    contents += line
                    if line[-1] == "\n":
                        pass
                    else:
                        contents += "\n"
            cat_file = os.path.join(case_dir_tmp, 'change_log')
            with open(cat_file, 'w') as fout:
                fout.write(contents)
            do_update = (contents != "")  # if there are differences, do update
            # execute the changes
            if do_update:
                print("Case %s already exists and there are changes, updating" % case_dir_to_check)
                if is_force_update:
                    print("Force update")
                    pass
                else:
                    print("Please check the change log first before continue: %s" % cat_file)
                    entry = input("Proceed? (y/n)")
                    if entry != "y":
                        print("Not updating, removing tmp files")
                        rmtree(case_dir_tmp)
                        return
                # document older files: 
                # 0. change_log file
                # a. files in the directory.
                # b. the img/initial_condition folder.
                # c. the configurations folder
                index = 0
                while os.path.isdir(os.path.join(case_dir, "update_%02d" % index)):
                    # figure out how many previous updates have been there.
                    index += 1
                older_dir = os.path.join(case_dir, "update_%02d" % index)
                if os.path.isdir(older_dir):
                    rmtree(older_dir)
                os.mkdir(older_dir)
                cat_file = os.path.join(older_dir, 'change_log')
                with open(cat_file, 'w') as fout:
                    fout.write(contents)
                for subdir, _, files in os.walk(case_dir):
                    for filename in files:
                        file_ori = os.path.join(subdir, filename)
                        if os.path.dirname(file_ori) == case_dir:
                            # only choose the files on the toppest level
                            copy2(file_ori, older_dir)
                ini_img_dir = os.path.join(case_dir, "img", "initial_condition")
                if os.path.isdir(ini_img_dir):
                    copytree(ini_img_dir, os.path.join(older_dir, os.path.basename(ini_img_dir)))
                    rmtree(ini_img_dir) # remove images
                configure_dir = os.path.join(case_dir, "configurations")
                if os.path.isdir(configure_dir):
                    # c. remove old configuration files
                    rmtree(configure_dir)
                # copy new file
                # a. files directly in the folder
                # b. the img/initial_condition folder
                # c. the configuration folder
                for subdir, _, files in os.walk(case_dir_tmp):
                    for filename in files:
                        if filename == "change_log":
                            # skip the log file
                            continue
                        file_tmp = os.path.join(subdir, filename)
                        file_to = os.path.join(case_dir, filename)
                        if os.path.dirname(file_tmp) == case_dir_tmp:
                            if os.path.isfile(file_to):
                                os.remove(file_to)
                            copy2(file_tmp, file_to)
                ini_img_tmp_dir = os.path.join(case_dir_tmp, "img", "initial_condition")
                if os.path.isdir(ini_img_tmp_dir):
                    copytree(ini_img_tmp_dir, ini_img_dir)
                configure_dir_tmp = os.path.join(case_dir_tmp, "configurations")
                if os.path.isdir(configure_dir_tmp):
                    copytree(configure_dir_tmp, configure_dir)
            else:
                print("Case %s already exists but there is no change, aborting" % case_dir_to_check)
            rmtree(case_dir_tmp)
        else:
            case_dir = Case.create(Case_Opt.o_dir(), fast_first_step=Case_Opt.if_fast_first_step(),\
                test_initial_steps=Case_Opt.test_initial_steps(), slurm_opts=Case_Opt.get_slurm_opts())
        return case_dir
    else:
        return Case

def SetBcVelocity(bc_dict, dimension, type_bc_v):
    '''
    set the boundary condition for velocity
    Inputs:
        bc_dict: an input with entries for Boundary velocity model in a prm file
        type_bc_v: type of velocity bc
    '''
    fs_indicator=[]
    ns_indicator = []
    if dimension == 2 and type_bc_v == 'all fs':
        fs_indicator = [0, 1, 2, 3]
    elif dimension == 3 and type_bc_v == 'all fs':
        fs_indicator = [0, 1, 2, 3, 4, 5]
    elif dimension == 2 and type_bc_v == 'bt fs side ns':
        fs_indicator = [1, 3]
        ns_indicator = [0, 2]
    elif dimension == 3 and type_bc_v == 'bt fs side ns':
        fs_indicator = [4, 5]
        ns_indicator = [0, 1, 2, 3]
    else:
        raise NotImplementedError("This combination of dimension (%d) and \
velocity boundary (%s) is not implemented yet." % (dimension, type_bc_v))
    if len(fs_indicator) > 0:
        bc_dict["Tangential velocity boundary indicators"] = str(fs_indicator).strip(']').lstrip('[')  # free slip
    else:
        if "Tangential velocity boundary indicators" in bc_dict:
            _ = bc_dict.pop("Tangential velocity boundary indicators")
    if len(ns_indicator) > 0:
        bc_dict["Zero velocity boundary indicators"] = str(ns_indicator).strip(']').lstrip('[')  # no slip
    else:
        if "Zero velocity boundary indicators" in bc_dict:
            _ = bc_dict.pop("Zero velocity boundary indicators")
    return bc_dict
    pass


def SetNewtonSolver(o_dict):
    '''
    Settings for the newton solver, starting from a combination that works
    '''
    o_dict["Nonlinear solver scheme"] = "single Advection, iterated Newton Stokes"
    o_dict["Max nonlinear iterations"] = "100"
    o_dict["Max nonlinear iterations in pre-refinement"] = "0"
    o_dict["Nonlinear solver tolerance"] = "1e-6"
    o_dict["Solver parameters"] = {}
    o_dict["Solver parameters"]["Newton solver parameters"] = {
        "Max pre-Newton nonlinear iterations" :"20",\
        "Nonlinear Newton solver switch tolerance": "1e-3",\
        "Max Newton line search iterations": "0",\
        "Maximum linear Stokes solver tolerance": "0.9",\
        "Use Newton residual scaling method": "true",\
        "Use Newton failsafe": "true",\
        "Stabilization preconditioner": "SPD",\
        "Stabilization velocity block": "SPD",\
        "Use Eisenstat Walker method for Picard iterations": "true"
    }
    o_dict["Solver parameters"]["Stokes solver parameters"] = {
        "Maximum number of expensive Stokes solver steps": "5000",\
        "Number of cheap Stokes solver steps": "500",\
        "Linear solver tolerance": "1e-1",\
        "GMRES solver restart length": "100"
    }
    return o_dict


def SetEndStep(o_dict, end_step):
    '''
    set termination criteria by "End step"
    '''
    if "Termination criteria" in o_dict:
        if "End time" in o_dict["Termination criteria"]:
            # pop option for "End time"
            _ = o_dict["Termination criteria"].pop("End time")
        o_dict["Termination criteria"]["End step"] = str(end_step)
        o_dict["Termination criteria"]["Termination criteria"] = "end step"
    else:
        o_dict["Termination criteria"] = {"End step": str(end_step), "Termination criteria": "end step"}
    return o_dict


def output_particle_ascii(fout, particle_data):
    '''
    Output to a ascii file that contains Particle coordinates, containing the coordinates of each particle
    '''
    # header information
    _header = '# Ascii file for particle coordinates\n'
    # output particle file
    fout.write(_header)
    is_first = True
    for i in range(particle_data.shape[0]):
        if is_first:
            is_first = False
        else:
            fout.write('\n')
        _string = ''
        for j in range(particle_data.shape[1]):
            if j == particle_data.shape[1] - 1:
                _string += '%.4e' % particle_data[i, j]
            else:
                _string += '%.4e ' % particle_data[i, j]
        fout.write(_string)
    pass

class CASE_OPT_TWOD(CASE_OPT):
    '''
    Define a class to work with CASE
    List of keys:
    '''
    def __init__(self):
        '''
        Initiation, first perform parental class's initiation,
        then perform daughter class's initiation.
        see document (run with -h option) for detail
        '''
        CASE_OPT.__init__(self)
        self.start = self.number_of_keys()
        self.add_key("Age of the subducting plate at trench", float,\
            ['world builder', 'subducting plate','age trench'], 80e6, nick='sp_age_trench')
        self.add_key("Spreading rate of the subducting plate", float,\
            ['world builder', 'subducting plate', 'sp rate'], 0.05, nick='sp_rate')
        self.add_key("Age of the overiding plate", float,\
            ['world builder', "overiding plate", 'age'], 40e6, nick='ov_age')
        self.add_key("Age of the transit overiding plate", float,\
            ['world builder', "overiding plate", "transit", 'age'], -1.0, nick='ov_trans_age')
        self.add_key("Length of the transit overiding plate", float,\
            ['world builder', "overiding plate", "transit", 'length'], 300e3, nick='ov_trans_length')
        self.add_key("Type of boundary condition\n\
            available options in [all free slip, ]", str,\
            ["boundary condition", "model"], "all free slip", nick='type_of_bd')
        self.add_key("Width of the Box", float,\
            ["box width"], 6.783e6, nick='box_width')
        self.add_key("Method to use for prescribing temperature", str,\
         ["prescribe temperature method"], 'function', nick="prescribe_T_method")
        self.add_key("Method to use for adjusting plate age.\n\
        The default method \"by values\" is to assign values of box_width, sp_rate, and sp_age_trench.\n\
        The method \"adjust box width\" is to change box_width\
        by assuming a default box_width for a default sp_age_trench\
         and extend the box for an older sp_age_trench.",\
         str,\
         ["world builder", "plate age method"], 'by values', nick="plate_age_method")
        self.add_key("Include peierls creep", int, ['include peierls creep'], 0, nick='if_peierls')
        self.add_key("Coupling the eclogite phase to shear zone viscosity",\
         int, ["shear zone", 'coupling the eclogite phase to shear zone viscosity'], 0, nick='if_couple_eclogite_viscosity')
        self.add_key("Width of the Box before adjusting for the age of the trench.\
This is used with the option \"adjust box width\" for configuring plate age at the trench.\
This value is the width of box for a default age (i.e. 80Myr), while the width of box for a\
different age will be adjusted.",\
          float, ["world builder", "box width before adjusting"], 6.783e6, nick='box_width_pre_adjust')
        self.add_key("Model to use for mantle phase transitions", str,\
         ["phase transition model"], 'CDPT', nick="phase_model")
        self.add_key("Root directory for lookup tables", str,\
         ["HeFESTo model", "data directory"], '.', nick="HeFESTo_data_dir")
        self.add_key("Cutoff depth for the shear zone rheology",\
          float, ["shear zone", 'cutoff depth'], 100e3, nick='sz_cutoff_depth')
        self.add_key("Adjust the refinement of mesh with the size of the box", int,\
          ["world builder", "adjust mesh with box width"], 0, nick='adjust_mesh_with_width') 
        self.add_key("Thickness of the shear zone / crust", float, ["shear zone", 'thickness'], 7.5e3, nick='Dsz')
        self.add_key("Refinement scheme", str, ["refinement scheme"], "2d", nick='rf_scheme')
        self.add_key("peierls creep scheme", str, ['peierls creep', 'scheme'], "MK10", nick='peierls_scheme')
        self.add_key("peierls creep, create a 2 stage model. I want to do this because including peierls scheme in the\
intiation stage causes the slab to break in the middle",\
         float, ['peierls creep', 'two stage intial time'], -1.0, nick='peierls_two_stage_time')
        self.add_key("mantle rheology", str, ['mantle rheology', 'scheme'], "HK03_wet_mod_twod", nick='mantle_rheology_scheme')
        self.add_key("Scheme for shear zone viscosity", str, ["shear zone", 'viscous scheme'], "constant", nick='sz_viscous_scheme')
        self.add_key("cohesion", float, ['mantle rheology', 'cohesion'], 50e6, nick='cohesion')
        self.add_key("friction", float, ['mantle rheology', 'friction'], 25.0, nick='friction')
        self.add_key("cohesion in the shear zone", float, ['shear zone', 'cohesion'], 10e6, nick='crust_cohesion')
        self.add_key("friction in the shear zone", float, ['shear zone', 'friction'], 2.8624, nick='crust_friction')
        self.add_key("constant viscosity in the shear zone", float, ['shear zone', 'constant viscosity'], 1e20, nick='sz_constant_viscosity')
        self.add_key("use WB new ridge implementation", int, ['world builder', 'use new ridge implementation'], 0, nick='wb_new_ridge')
        self.add_key("branch", str, ['branch'], "", nick='branch')
        self.add_key("minimum viscosity in the shear zone", float, ['shear zone', 'minimum viscosity'], 1e18, nick='sz_minimum_viscosity')
        self.add_key("Use embeded fault implementation",\
          int, ["shear zone", 'use embeded fault'], 0, nick='use_embeded_fault')
        self.add_key("factor for the embeded fault that controls the maxmimum thickness of the layer", float,\
            ['shear zone', 'ef factor'], 1.9, nick='ef_factor')
        self.add_key("bury depth of the particles in the harzburgite layer", float,\
            ['shear zone', 'ef particle bury depth'], 5e3, nick='ef_Dbury')
        self.add_key("interval measured in meter between adjacent particles", float,\
            ['shear zone', 'ef particle interval'], 10e3, nick='ef_interval')
        self.add_key("Use embeded fault implementation with the implementation of world builder feature surface",\
          int, ["shear zone", 'use embeded fault with feature surface'], 0, nick='use_embeded_fault_feature_surface')
        self.add_key("transition distance at the trench for the kinematic boundary condition", float,\
            ["boundary condition", "trench transit distance"], 20e3, nick='delta_trench')
        self.add_key("Cold cutoff of the eclogite transition",\
         float, ["shear zone", 'Max pressure for eclogite transition'], 5e9, nick='eclogite_max_P')
        self.add_key("eclogite transition that matches the mineral phase boundaries",\
         int, ["shear zone", 'match the eclogite transition with phase diagram'], 0, nick='eclogite_match')
        self.add_key("number of layer in the crust", int, ["world builder", 'layers of crust'], 1, nick='n_crust_layer')
        self.add_key("Thickness of the upper crust", float, ["shear zone", 'upper crust thickness'], 3e3, nick='Duc')
        self.add_key("Rheology of the upper crust", str, ["shear zone", 'upper crust rheology scheme'], '', nick='upper_crust_rheology_scheme')
        self.add_key("Rheology of the lower crust", str, ["shear zone", 'lower crust rheology scheme'], '', nick='lower_crust_rheology_scheme')
        self.add_key("Distance of the subducting plate to the box side", float,\
            ['world builder', 'subducting plate', 'trailing length'], 0.0, nick='sp_trailing_length')
        self.add_key("Distance of the overiding plate to the box side", float,\
            ['world builder', 'overiding plate', 'trailing length'], 0.0, nick='ov_trailing_length')
        self.add_key("Viscosity in the slab core", float,\
            ['shear zone', "slab core viscosity"], -1.0, nick='slab_core_viscosity')
        self.add_key("Value of Coh to use in the rheology", float,\
            ['mantle rheology', "Coh"], 1000.0, nick='mantle_coh')
        self.add_key("Minimum viscosity", float,\
            ["minimum viscosity"], 1e18, nick='minimum_viscosity')
        self.add_key("automatically fix boundary temperature",\
            int, ['world builder', 'fix boudnary temperature auto'], 0, nick='fix_boudnary_temperature_auto')
        self.add_key("the maximum extent of a slice in the geometry refinement",\
            float, ['world builder', 'maximum repetition slice'], 1e31, nick='maximum_repetition_slice')
        self.add_key("Global refinement", int, ['refinement', 'global refinement'], 4, nick='global_refinement')
        self.add_key("Adaptive refinement", int, ['refinement', 'adaptive refinement'], 2, nick='adaptive_refinement')
        self.add_key("remove overiding plate composition", int, ['world builder', 'remove ov comp'], 0, nick='rm_ov_comp')
        self.add_key("peierls creep scheme", str, ['peierls creep', 'flow law'], "exact", nick='peierls_flow_law')
        self.add_key("reset density in the two corners", int, ["reset density"], 0, nick='reset_density')
        self.add_key("Maximum Peierls strain rate iterations", int, ['peierls creep', "maximum peierls iterations"], 40, nick='maximum_peierls_iterations')
        self.add_key("Type of CDPT Model to use for mantle phase transitions", str,\
         ["phase transition model CDPT type"], 'Billen2018_old', nick="CDPT_type")
        self.add_key("Fix the activation volume of the Peierls creep", str,\
         ['peierls creep', "fix peierls V as"], '', nick="fix_peierls_V_as")
        self.add_key("Width for prescribing temperature", float,\
         ["prescribe temperature width"], 2.75e5, nick="prescribe_T_width")
        self.add_key("Prescribing temperature with trailing edge present", int,\
         ["prescribe temperature with trailing edge"], 0, nick="prescribe_T_with_trailing_edge")
        self.add_key("Value of lower/upper mantle ratio to use in the rheology", float,\
            ['mantle rheology', "jump lower mantle"], 100.0, nick='jump_lower_mantle')
        self.add_key("use 3d depth average file", int,\
            ['mantle rheology', "use 3d da file"], 0, nick='use_3d_da_file')
        self.add_key("use lookup table morb", int, ["use lookup table morb"], 0, nick="use_lookup_table_morb")
        self.add_key("lookup table morb mixing, 1: iso stress (weakest), 2: iso strain (strongest), 3: log (intermediate)",\
                     int, ["lookup table morb", "mixing model"], 0, nick="use_lookup_table_morb")
        self.add_key("difference of activation volume in the diffusion creep rheology", float,\
            ['mantle rheology', "delta Vdiff"], -2.1e-6, nick='delta_Vdiff')
        self.add_key("Clapeyron slope of the 410 km", float,\
         ["CDPT", "slope 410"], 2e6, nick="slope_410")
        self.add_key("Clapeyron slope of the 660 km", float,\
         ["CDPT", "slope 660"], -1e6, nick="slope_660")
        self.add_key("Slab strengh", float, ["slab", "strength"], 500e6, nick="slab_strength")
        self.add_key("Height of the Box", float,\
            ["world builder", "box height"], 2890e3, nick='box_height')
        self.add_key("Refine Wedge", int,\
            ["refinement", "refine wedge"], 0, nick='refine_wedge')
        self.add_key("Output heat flux", int,\
            ["outputs", "heat flux"], 0, nick='output_heat_flux')
        self.add_key("difference of activation energy in the diffusion creep rheology", float,\
            ['mantle rheology', "delta Ediff"], 0.0, nick='delta_Ediff')
        self.add_key("difference of activation energy in the dislocation creep rheology", float,\
            ['mantle rheology', "delta Edisl"], 0.0, nick='delta_Edisl')
        self.add_key("difference of activation volume in the dislocation creep rheology", float,\
            ['mantle rheology', "delta Vdisl"], 3e-6, nick='delta_Vdisl')
        self.add_key("Include metastable transition", int,\
            ['metastable', 'include metastable'], 0, nick='include_meta')
    
    def check(self):
        '''
        check to see if these values make sense
        '''
        CASE_OPT.check(self)
        geometry = self.values[3]
        # geometry options
        my_assert(geometry in ['chunk', 'box'], ValueError,\
        "%s: The geometry for TwoDSubduction cases must be \"chunk\" or \"box\"" \
        % func_name())
        if self.values[3] == 'box':
            my_assert(self.values[8] == 1, ValueError,\
            "%s: When using the box geometry, world builder must be used for initial conditions" \
            % func_name())  # use box geometry, wb is mandatory
        # check the setting for adjust box width
        plate_age_method = self.values[self.start + 8] 
        assert(plate_age_method in ['by values', 'adjust box width', 'adjust box width only assigning age'])
        if plate_age_method == 'adjust box width':
            box_width = self.values[self.start + 6]
            if box_width != self.defaults[self.start + 6]:
                warnings.warn("By using \"adjust box width\" method for subduction plate age\
                box width will be automatically adjusted. Thus the given\
                value is not taken.")
            box_width_pre_adjust = self.values[self.start+11]
            sp_age_trench_default = self.defaults[self.start]  # default value for age at trench
            sp_rate_default = self.defaults[self.start+1]  # default value for spreading rate
            my_assert(box_width_pre_adjust > sp_age_trench_default * sp_rate_default, ValueError,\
            "For the \"adjust box width\" method to work, the box width before adjusting needs to be wider\
than the multiplication of the default values of \"sp rate\" and \"age trench\"")
        # check the option for refinement
        refinement_level = self.values[15]
        assert(refinement_level in [-1, 9, 10, 11, 12, 13])  # it's either not turned on or one of the options for the total refinement levels
        # check the option for the type of boundary conditions
        type_of_bd = self.values[self.start + 5]
        assert(type_of_bd in ["all free slip", "top prescribed", "top prescribed with bottom right open", "top prescribed with bottom left open"])
        # check the method to use for phase transition
        phase_model = self.values[self.start + 12]
        my_assert( phase_model in ["CDPT", "HeFESTo"], ValueError,\
        "%s: Models to use for phases must by CDPT or HeFESTo" \
        % func_name())
        # check the directory for HeFESTo
        o_dir = self.values[2]
        root_level = self.values[7]
        if phase_model == "HeFESTo":  # check the existence of Hefesto files
            HeFESTo_data_dir = self.values[self.start + 13]
            HeFESTo_data_dir_pull_path = os.path.join(o_dir, ".." * (root_level - 1), HeFESTo_data_dir)
            my_assert(os.path.isdir(HeFESTo_data_dir_pull_path),\
            FileNotFoundError, "%s is not a directory" % HeFESTo_data_dir_pull_path)
        # assert scheme to use for refinement
        rf_scheme = self.values[self.start + 17]
        assert(rf_scheme in ['2d', '3d consistent'])
        # assert scheme of peierls creep to use
        peierls_scheme = self.values[self.start + 18]
        assert(peierls_scheme in ['MK10', "MK10p", 'Idrissi16'])
        peierls_flow_law = self.values[self.start + 52]
        assert(peierls_flow_law in ["approximation", "exact"])
        # assert viscous scheme to use
        sz_viscous_scheme = self.values[self.start + 21]
        assert(sz_viscous_scheme in ["stress dependent", "constant"])
        friction = self.values[self.start + 23]
        assert(friction >= 0.0 and friction < 90.0)  # an angle between 0.0 and 90.0
        crust_friction = self.values[self.start + 25]
        assert(crust_friction >= 0.0 and crust_friction < 90.0)  # an angle between 0.0 and 90.0
        sz_constant_viscosity = self.values[self.start + 26]
        assert(sz_constant_viscosity > 0.0)
        wb_new_ridge = self.values[self.start + 27]
        assert(wb_new_ridge in [0, 1])  # use the new ridge implementation or not
        # assert there is either 1 or 2 layers in the crust
        n_crust_layer = self.values[self.start + 38]
        assert(n_crust_layer in [1, 2])
        # the use embeded fault method currently is inconsistent with the particle method
        use_embeded_fault = self.values[self.start + 30]
        comp_method = self.values[25] 
        if use_embeded_fault == 1:
            assert(comp_method == "field")
        # CDPT model type 
        CDPT_type = self.values[self.start + 55]
        fix_peierls_V_as = self.values[self.start + 56]
        assert(CDPT_type in ["Billen2018_old", "HeFESTo_consistent", "Billen2018"])
        assert(fix_peierls_V_as in ["", "diffusion", "dislocation"])
        # lookup table morb
        use_lookup_table_morb = self.values[self.start + 61]
        lookup_table_morb_mixing = self.values[self.start + 62]
        if use_lookup_table_morb:
            assert(lookup_table_morb_mixing in [0, 1, 2, 3]) # check the mixing model used
        refine_wedge = self.values[self.start + 68]
        if geometry == "box" and refine_wedge:
            raise NotImplementedError

    def to_configure_prm(self):
        if_wb = self.values[8]
        type_of_bd = self.values[self.start + 5]
        sp_rate = self.values[self.start + 1]
        ov_age = self.values[self.start + 2]
        potential_T = self.values[4]
        box_width = self.values[self.start + 6]
        geometry = self.values[3]
        prescribe_T_method = self.values[self.start + 7]
        plate_age_method = self.values[self.start + 8] 
        if plate_age_method == 'adjust box width':
            box_width = re_write_geometry_while_assigning_plate_age(
            *self.to_re_write_geometry_pa()
            ) # adjust box width
        if plate_age_method == 'adjust box width only assigning age':
            box_width = re_write_geometry_while_only_assigning_plate_age(
            *self.to_re_write_geometry_pa()
            ) # adjust Gox width
        if_peierls = self.values[self.start + 9]
        if_couple_eclogite_viscosity = self.values[self.start + 10]
        phase_model = self.values[self.start + 12]
        HeFESTo_data_dir = self.values[self.start + 13]
        root_level = self.values[7]
        HeFESTo_data_dir_relative_path = os.path.join("../"*root_level, HeFESTo_data_dir)
        sz_cutoff_depth = self.values[self.start+14]
        adjust_mesh_with_width = self.values[self.start+15]
        rf_scheme = self.values[self.start + 17]
        peierls_scheme = self.values[self.start + 18]
        peierls_two_stage_time = self.values[self.start + 19]
        mantle_rheology_scheme = self.values[self.start + 20]
        stokes_linear_tolerance = self.values[11]
        end_time = self.values[12]
        refinement_level = self.values[15]
        case_o_dir = self.values[16]
        sz_viscous_scheme = self.values[self.start + 21]
        cohesion = self.values[self.start + 22]
        friction = self.values[self.start + 23]
        crust_cohesion = self.values[self.start + 24]
        crust_friction = self.values[self.start + 25]
        sz_constant_viscosity = self.values[self.start + 26]
        branch = self.values[self.start + 28]
        partitions = self.values[20]
        sz_minimum_viscosity = self.values[self.start + 29]
        use_embeded_fault = self.values[self.start + 30]
        Dsz = self.values[self.start + 16]
        ef_factor = self.values[self.start + 31]
        ef_Dbury = self.values[self.start + 32]
        sp_age_trench = self.values[self.start]
        use_embeded_fault_feature_surface = self.values[self.start + 34]
        ef_particle_interval = self.values[self.start + 33]
        delta_trench = self.values[self.start + 35]
        eclogite_max_P = self.values[self.start + 36]
        eclogite_match = self.values[self.start + 37]
        version = self.values[23]
        n_crust_layer = self.values[self.start + 38]
        upper_crust_rheology_scheme = self.values[self.start + 40]
        lower_crust_rheology_scheme = self.values[self.start + 41]
        sp_trailing_length = self.values[self.start + 42]
        ov_trailing_length = self.values[self.start + 43]
        slab_core_viscosity = self.values[self.start + 44]
        mantle_coh = self.values[self.start + 45]
        minimum_viscosity = self.values[self.start + 46]
        fix_boudnary_temperature_auto = self.values[self.start + 47]
        maximum_repetition_slice = self.values[self.start + 48]
        global_refinement = self.values[self.start + 49]
        adaptive_refinement = self.values[self.start + 50]
        rm_ov_comp = self.values[self.start + 51]
        comp_method = self.values[25] 
        peierls_flow_law = self.values[self.start + 52]
        reset_density = self.values[self.start + 53]
        maximum_peierls_iterations = self.values[self.start + 54]
        CDPT_type = self.values[self.start + 55]
        use_new_rheology_module = self.values[28]
        fix_peierls_V_as = self.values[self.start + 56]
        prescribe_T_width = self.values[self.start + 57]
        prescribe_T_with_trailing_edge = self.values[self.start + 58]
        jump_lower_mantle = self.values[self.start + 59]
        use_3d_da_file = self.values[self.start + 60]
        use_lookup_table_morb = self.values[self.start + 61]
        lookup_table_morb_mixing = self.values[self.start + 62]
        delta_Vdiff = self.values[self.start + 63]
        slope_410 = self.values[self.start + 64]
        slope_660 = self.values[self.start + 65]
        slab_strength = self.values[self.start + 66]
        box_height = self.values[self.start + 67]
        minimum_particles_per_cell = self.values[29]
        maximum_particles_per_cell = self.values[30]
        refine_wedge = self.values[self.start + 68]
        output_heat_flux = self.values[self.start + 69]
        delta_Ediff = self.values[self.start + 70]
        delta_Edisl = self.values[self.start + 71]
        delta_Vdisl = self.values[self.start + 72]
        include_meta = self.values[self.start + 73]

        return if_wb, geometry, box_width, type_of_bd, potential_T, sp_rate,\
        ov_age, prescribe_T_method, if_peierls, if_couple_eclogite_viscosity, phase_model,\
        HeFESTo_data_dir_relative_path, sz_cutoff_depth, adjust_mesh_with_width, rf_scheme,\
        peierls_scheme, peierls_two_stage_time, mantle_rheology_scheme, stokes_linear_tolerance, end_time,\
        refinement_level, case_o_dir, sz_viscous_scheme, cohesion, friction, crust_cohesion, crust_friction, sz_constant_viscosity,\
        branch, partitions, sz_minimum_viscosity, use_embeded_fault, Dsz, ef_factor, ef_Dbury, sp_age_trench, use_embeded_fault_feature_surface,\
        ef_particle_interval, delta_trench, eclogite_max_P, eclogite_match, version, n_crust_layer,\
        upper_crust_rheology_scheme, lower_crust_rheology_scheme, sp_trailing_length, ov_trailing_length, slab_core_viscosity,\
        mantle_coh, minimum_viscosity, fix_boudnary_temperature_auto, maximum_repetition_slice, global_refinement, adaptive_refinement,\
        rm_ov_comp, comp_method, peierls_flow_law, reset_density, maximum_peierls_iterations, CDPT_type, use_new_rheology_module, fix_peierls_V_as,\
        prescribe_T_width, prescribe_T_with_trailing_edge, plate_age_method, jump_lower_mantle, use_3d_da_file, use_lookup_table_morb, lookup_table_morb_mixing,\
        delta_Vdiff, slope_410, slope_660, slab_strength, box_height, minimum_particles_per_cell, maximum_particles_per_cell, refine_wedge, output_heat_flux,\
        delta_Ediff, delta_Edisl, delta_Vdisl, include_meta

    def to_configure_wb(self):
        '''
        Interface to configure_wb
        '''
        if_wb = self.values[8]
        geometry = self.values[3]
        potential_T = self.values[4]
        sp_age_trench = self.values[self.start]
        sp_rate = self.values[self.start + 1]
        ov_age = self.values[self.start + 2]
        ov_trans_age = self.values[self.start + 3]
        ov_trans_length = self.values[self.start + 4]
        if self.values[self.start + 3] < 0.0:
            if_ov_trans = False
        else:
            if_ov_trans = True
        is_box_wider = self.is_box_wider()
        Dsz = self.values[self.start + 16]
        wb_new_ridge = self.values[self.start + 27]
        version = self.values[23]
        n_crust_layer = self.values[self.start + 38]
        Duc = self.values[self.start + 39]
        rm_ov_comp = self.values[self.start + 51]


        if n_crust_layer == 1:
            # number of total compositions
            n_comp = 4
        elif n_crust_layer == 2:
            n_comp = 6
        sp_trailing_length = self.values[self.start + 42]
        ov_trailing_length = self.values[self.start + 43]
        plate_age_method = self.values[self.start + 8] 
        box_width = self.values[self.start + 6]
        if plate_age_method == 'adjust box width':
            box_width = re_write_geometry_while_assigning_plate_age(
            *self.to_re_write_geometry_pa()
            ) # adjust box width
        elif plate_age_method == 'adjust box width only assigning age':
            box_width = re_write_geometry_while_only_assigning_plate_age(
            *self.to_re_write_geometry_pa()
            ) # adjust box width
        return if_wb, geometry, potential_T, sp_age_trench, sp_rate, ov_age,\
            if_ov_trans, ov_trans_age, ov_trans_length, is_box_wider, Dsz, wb_new_ridge, version,\
            n_crust_layer, Duc, n_comp, sp_trailing_length, ov_trailing_length, box_width, rm_ov_comp,\
            plate_age_method
    
    def to_configure_final(self):
        '''
        Interface to configure_final
        '''
        geometry = self.values[3]
        Dsz = self.values[self.start + 16]
        use_embeded_fault = self.values[self.start + 30]
        ef_Dbury = self.values[self.start + 32]
        ef_particle_interval = self.values[self.start + 33]
        use_embeded_fault_feature_surface = self.values[self.start + 34]
        return geometry, Dsz, use_embeded_fault, ef_Dbury, ef_particle_interval, use_embeded_fault_feature_surface
    
    def to_re_write_geometry_pa(self):
        '''
        Interface to re_write_geometry_pa
        '''
        box_width_pre_adjust = self.values[self.start+11]
        sp_trailing_length = self.values[self.start + 42]
        return box_width_pre_adjust, self.defaults[self.start],\
        self.values[self.start], self.values[self.start+1], sp_trailing_length, 0.0
    
    def is_box_wider(self):
        '''
        Return whether we should use a box wider than 90 degree in longtitude
        Return:
            True or False
        '''
        box_width = re_write_geometry_while_assigning_plate_age(
            *self.to_re_write_geometry_pa()
            ) # adjust box width
        if box_width > 1e7:
            return True
        else:
            return False

def RefitRheology(rheology, diff_correction, disl_correction, ref_state):
    '''
    Inputs:
        rheology (str or dict): rheology to start with
        diff_correction (dict): variation to the diffusion creep
            e.g. {'A': 1.0, 'p': 0.0, 'r': 0.0, 'n': 0.0, 'E': 0.0, 'V': -2.1e-6}
        disl_correction (dict): variation to the dislocation creep
            e.g. {'A': 1.0, 'p': 0.0, 'r': 0.0, 'n': 0.0, 'E': 0.0, 'V': 3e-6}
        ref_state (dict): reference state
    Return:
        rheology_dict (dict): refit rheology
    '''
    # read the parameters from HK 03.
    # water is in the water fugacity

    if type(rheology) == str: 
        rheology_prm_dict = RHEOLOGY_PRM()
        diffusion_creep_ori = getattr(rheology_prm_dict, rheology + "_diff")
        dislocation_creep_ori = getattr(rheology_prm_dict, rheology + "_disl")
        rheology_dict = {'diffusion': diffusion_creep_ori, 'dislocation': dislocation_creep_ori}
    elif type(rheology) == dict:
        assert('diffusion' in rheology and 'dislocation' in rheology)
        rheology_dict = rheology
        diffusion_creep_ori = rheology['diffusion']
        dislocation_creep_ori = rheology['dislocation']
    else:
        raise ValueError('rheology must be either str or dict')
    
    # apply variations to the original rheology
    assert('A' in diff_correction and 'p' in diff_correction and 'r' in diff_correction\
           and 'n' in diff_correction and 'E' in diff_correction and 'V' in diff_correction)
    assert('A' in disl_correction and 'p' in disl_correction and 'r' in disl_correction\
           and 'n' in disl_correction and 'E' in disl_correction and 'V' in disl_correction)

    # reference state
    assert('Coh' in ref_state and 'stress' in ref_state and 'P' in ref_state\
           and 'T' in ref_state and 'd' in ref_state) 
    Coh_ref = ref_state['Coh'] # H / 10^6 Si
    stress_ref = ref_state['stress'] # MPa
    P_ref = ref_state['P'] # Pa
    T_ref = ref_state['T'] # K
    d_ref = ref_state['d'] # mu m
    strain_rate_diff_ref = CreepStrainRate(diffusion_creep_ori, stress_ref, P_ref, T_ref, d_ref, Coh_ref)
    strain_rate_disl_ref = CreepStrainRate(dislocation_creep_ori, stress_ref, P_ref, T_ref, d_ref, Coh_ref)
    print("strain_rate_diff_ref: ", strain_rate_diff_ref)
    print("strain_rate_disl_ref: ", strain_rate_disl_ref)

    # reference pseudo "strain rate" for correction 
    strain_rate_diff_correction = CreepStrainRate(diff_correction, stress_ref, P_ref, T_ref, d_ref, Coh_ref)
    strain_rate_disl_correction = CreepStrainRate(disl_correction, stress_ref, P_ref, T_ref, d_ref, Coh_ref)

    # make a copy of the original rheology 
    diffusion_creep = diffusion_creep_ori.copy()
    dislocation_creep = dislocation_creep_ori.copy()
    
    # apply the correction
    # the only task to do is to devide the prefactor by the strain_rate correction
    diffusion_creep['E'] += diff_correction['E']
    diffusion_creep['V'] += diff_correction['V']
    dislocation_creep['E'] += disl_correction['E']
    dislocation_creep['V'] += disl_correction['V']
    diffusion_creep['A'] /= strain_rate_diff_correction
    dislocation_creep['A'] /= strain_rate_disl_correction
    
    # return values
    rheology_dict = {'diffusion': diffusion_creep, 'dislocation': dislocation_creep}
    return rheology_dict

class STRENGTH_PROFILE(RHEOLOGY_OPR):

    def __init__(self, **kwargs):
        RHEOLOGY_OPR.__init__(self)
        self.Sigs = None
        self.Sigs_brittle = None
        self.etas_brittle = None
        self.Sigs_viscous = None
        self.etas_viscous = None
        self.etas_peierls = None
        self.Zs = None
        self.Etas = None
        self.Computed = False
        # todo_peierls
        self.peierls_type = None
        self.max_depth = kwargs.get('max_depth', 80e3)
        self.T_type = kwargs.get("T_type", "hpc")
        self.SetRheologyByName(diff=kwargs.get("diff", None),\
                               disl=kwargs.get("disl", None),\
                               brittle=kwargs.get("brittle", None),\
                               peierls=kwargs.get("peierls", None))

    def Execute(self, **kwargs):
        '''
        Compute the strength profile
        '''
        year = 365 * 24 * 3600.0
        compute_second_invariant = kwargs.get('compute_second_invariant', False)
        brittle = self.brittle
        assert(self.diff_type != None or self.disl_type != None)
        assert(brittle != None)
        averaging = kwargs.get('averaging', 'harmonic')
        strain_rate = kwargs.get('strain_rate', 1e-14)
        # rheology_prm = RHEOLOGY_PRM()
        # self.plastic = rheology_prm.ARCAY17_plastic
        # self.dislocation_creep = rheology_prm.ARCAY17_disl
        Zs = np.linspace(0.0, self.max_depth, 100)
        if self.T_type == 'hpc':
            # use a half space cooling
            Ts = temperature_halfspace(Zs, 40e6*year, Tm=1573.0) # adiabatic temperature
        elif self.T_type == 'ARCAY17':
            Ts = 713 * Zs / 78.245e3  + 273.14# geotherm from Arcay 2017 pepi, figure 3d 2
        else:
            raise NotImplementedError
        Tliths = temperature_halfspace(Zs, 40e6*year, Tm=1573.0) # adiabatic temperature
        Ps = pressure_from_lithostatic(Zs, Tliths)
        # brittle self.brittle
        if brittle["type"] == "stress dependent":
            # note this is questionable, is this second order invariant
            self.Sigs_brittle = StressDependentYielding(Ps, brittle["cohesion"], brittle["friction"], brittle["ref strain rate"], brittle["n"], strain_rate)
        elif brittle["type"] == "Coulumb":
            self.Sigs_brittle = CoulumbYielding(Ps, brittle["cohesion"], brittle["friction"])
        elif brittle["type"] == "Byerlee":
            self.Sigs_brittle = Byerlee(Ps)
        else:
            raise NotImplementedError()
        self.etas_brittle = self.Sigs_brittle / 2.0 / strain_rate
        # viscous stress
        # Note on the d and coh:
        #      disl - not dependent on d;
        #      coh - the one in the Arcay paper doesn't depend on Coh
        etas_diff = None
        etas_disl = None
        etas_peierls = None
        if self.diff_type is not None:
            etas_diff = CreepRheology(self.diff, strain_rate, Ps, Ts,\
                                          1e4, 1000.0, use_effective_strain_rate=compute_second_invariant)
        if self.disl_type is not None:
            etas_disl = CreepRheology(self.disl, strain_rate, Ps, Ts,\
                                          1e4, 1000.0, use_effective_strain_rate=compute_second_invariant)
        if self.peierls_type is not None:
            self.etas_peierls = np.zeros(Ts.size)
            for i in range(Ts.size):
                P = Ps[i]
                T = Ts[i]
                self.etas_peierls[i] = PeierlsCreepRheology(self.peierls, strain_rate, P, T)
            self.Sigs_peierls = 2.0 * strain_rate * self.etas_peierls
        self.etas_viscous = ComputeComposite(etas_diff, etas_disl)
        self.Sigs_viscous = 2.0 * strain_rate * self.etas_viscous
        # Sigs_viscous = CreepStress(self.creep, strain_rate, Ps, Ts, 1e4, 1000.0) # change to UI
        if averaging == 'harmonic':
            self.Etas = ComputeComposite(self.etas_viscous, self.etas_brittle, self.etas_peierls)
        else:
            raise NotImplementedError()
        self.Sigs = 2 * strain_rate * self.Etas
        self.Zs = Zs
        self.computed = True

    def PlotStress(self, **kwargs):
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', None)
        label_components = kwargs.get('label_components', False)
        plot_stress_by_log = kwargs.get('plot_stress_by_log', False)
        if label_components:
            label_brittle = "brittle"
            label_viscous = "viscous"
            label_peierls = "peierls"
        else:
            label_brittle = None
            label_viscous = None
            label_peierls = None
        _color = kwargs.get('color', 'b')
        if ax == None:
            raise NotImplementedError()
        # make plots
        mask = (self.Etas > 1e-32) # get the reasonable values, the peierls creep may return inf values
        if plot_stress_by_log:
            # plot the log values of the stress on the x axis
            ax.semilogx(self.Sigs[mask]/1e6, self.Zs[mask]/1e3, color=_color, label=label)
            ax.semilogx(self.Sigs_brittle[mask]/1e6, self.Zs[mask]/1e3, '.', color=_color, label=label_brittle)
            ax.semilogx(self.Sigs_viscous[mask]/1e6, self.Zs[mask]/1e3, '--', color=_color, label=label_viscous)
            if self.peierls_type is not None:
                ax.semilogx(self.Sigs_peierls[mask]/1e6, self.Zs[mask]/1e3, '-.', color=_color, label=label_peierls)
            ax.set_xlim([0.0, 10**(3.5)])
        else:
            # plot the log values of the stress on the x axis
            ax.plot(self.Sigs[mask]/1e6, self.Zs[mask]/1e3, color=_color, label=label)
            ax.plot(self.Sigs_brittle[mask]/1e6, self.Zs[mask]/1e3, '.', color=_color, label=label_brittle)
            ax.plot(self.Sigs_viscous[mask]/1e6, self.Zs[mask]/1e3, '--', color=_color, label=label_viscous)
            if self.peierls_type is not None:
                ax.plot(self.Sigs_peierls[mask]/1e6, self.Zs[mask]/1e3, '-.', color=_color, label=label_peierls)
            x_max = np.ceil(np.max(self.Sigs[mask]/1e6) / 100.0) * 100.0
            ax.set_xlim([0.0, x_max])
        ax.set_xlabel("Second invariant of the stress tensor (MPa)")
        ax.set_ylabel("Depth (km)")
    
    def PlotViscosity(self, **kwargs):
        ax = kwargs.get('ax', None)
        label = kwargs.get('label', None)
        label_components = kwargs.get('label_components', False)
        if label_components:
            label_brittle = "brittle"
            label_viscous = "viscous"
            label_peierls = "peierls"
        else:
            label_brittle = None
            label_viscous = None
            label_peierls = None
        _color = kwargs.get('color', 'b')
        if ax == None:
            raise NotImplementedError()
        # plot viscosity
        mask = (self.Etas > 1e-32) # get the reasonable values, the peierls creep may return inf values
        ax.semilogx(self.Etas[mask], self.Zs[mask]/1e3, color=_color, label=label)
        ax.semilogx(self.etas_brittle[mask], self.Zs[mask]/1e3, '.', color=_color, label=label_brittle)
        ax.semilogx(self.etas_viscous[mask], self.Zs[mask]/1e3, '--', color=_color, label=label_viscous)
        if self.peierls_type is not None:
            ax.semilogx(self.etas_peierls[mask], self.Zs[mask]/1e3, '-.', color=_color, label=label_peierls)
        ax.set_xlabel("Viscosity (Pa * s)")
        ax.set_ylabel("Depth (km)")
        x_min = 1e18
        x_max = 1e24
        ax.set_xlim([x_min, x_max])

def ConvertFromAspectInput(aspect_creep, **kwargs):
    """
    Viscosity is calculated by flow law in form of (strain_rate)**(1.0 / n - 1) * (B)**(-1.0 / n) * np.exp((E + P * V) / (n * R * T)) * 1e6
    while in aspect, flow law in form of 0.5 * A**(-1.0 / n) * d**(m / n) * (strain_rate)**(1.0 / n - 1) * np.exp((E + P * V) / (n * R * T)).
    Here I convert backward from the flow law used in aspect
    Original Units:
     - P: Pa
     - T: K
     - d: mm
     - Coh: H / 10^6 Si
    Original Units:
     - P: Pa
     - T: K
     - d: m
    """
    # read in initial value
    A = aspect_creep['A']
    m = aspect_creep['m']
    n = aspect_creep['n']
    E = aspect_creep['E']
    V = aspect_creep['V']
    d = aspect_creep['d']
    # compute value of F(pre factor)
    use_effective_strain_rate = kwargs.get('use_effective_strain_rate', False)
    if use_effective_strain_rate:
        F = 1 / (2**((n-1)/n)*3**((n+1)/2/n)) * 2.0
    else:
        F = 1.0
    # prepare values for aspect
    creep = {}
    # stress in the original equation is in Mpa, grain size is in um
    creep['A'] = 1e6**m * 1e6**n * A * F**n  # F term: use effective strain rate
    creep['d'] = d * 1e6
    creep['n'] = n
    creep['p'] = m
    creep['E'] = E
    creep['V'] = V
    creep['r'] = 0.0  # assume this is not dependent on Coh
    creep['Coh'] = 1000.0
    return creep

def AssignAspectViscoPlasticPhaseRheology(visco_plastic_dict, key, idx, diffusion_creep, dislocation_creep, **kwargs):
    '''
    Inputs:
        visco_plastic_dict(dict): options for the viscoplastic module in the aspect material model
        key: name for the composition
        idx: index for the phase
        diffusion_creep: diffusion creep rheology
        dislocation_creep: dislocation creep rheology
        kwargs:
            no_convert: do not convert to aspect format (inputs are already aspect format)
    Return:
        visco_plastic_dict(dict): options for the viscoplastic module in the aspect material model after the change
    '''
    assert((diffusion_creep is not None) or (dislocation_creep is not None))
    no_convert = kwargs.get("no_convert", False)
    # diffusion creep
    if diffusion_creep is not None:
        if no_convert:
            diffusion_creep_aspect = diffusion_creep
        else:
            diffusion_creep_aspect = Convert2AspectInput(diffusion_creep, use_effective_strain_rate=True)
        # print("visco_plastic_dict: ", visco_plastic_dict) # debug
        visco_plastic_dict["Prefactors for diffusion creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Prefactors for diffusion creep"], key, idx, diffusion_creep_aspect['A'])
        visco_plastic_dict["Grain size exponents for diffusion creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Grain size exponents for diffusion creep"], key, idx, diffusion_creep_aspect['m'])
        visco_plastic_dict["Activation energies for diffusion creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Activation energies for diffusion creep"], key, idx, diffusion_creep_aspect['E'])
        visco_plastic_dict["Activation volumes for diffusion creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Activation volumes for diffusion creep"], key, idx, diffusion_creep_aspect['V'])
    else:
        visco_plastic_dict["Prefactors for diffusion creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Prefactors for diffusion creep"], key, idx, 1e-31)
    # dislocation creep
    if dislocation_creep is not None:
        if no_convert:
            dislocation_creep_aspect = dislocation_creep
        else:
            dislocation_creep_aspect = Convert2AspectInput(dislocation_creep, use_effective_strain_rate=True)
        visco_plastic_dict["Prefactors for dislocation creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Prefactors for dislocation creep"], key, idx, dislocation_creep_aspect['A'])
        visco_plastic_dict["Stress exponents for dislocation creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Stress exponents for dislocation creep"], key, idx, dislocation_creep_aspect['n'])
        visco_plastic_dict["Activation energies for dislocation creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Activation energies for dislocation creep"], key, idx, dislocation_creep_aspect['E'])
        visco_plastic_dict["Activation volumes for dislocation creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Activation volumes for dislocation creep"], key, idx, dislocation_creep_aspect['V'])
    else:
        visco_plastic_dict["Prefactors for dislocation creep"] = \
            ReplacePhaseOption(visco_plastic_dict["Prefactors for dislocation creep"], key, idx, 1e-31)
    return visco_plastic_dict


def GetPeierlsApproxVist(flv):
    '''
    export Peierls rheology for approximation
    '''
    mpa = 1e6  # MPa to Pa
    Peierls={}
    if flv == "MK10":
        Peierls['q'] = 1.0
        Peierls['p'] = 0.5
        n = 2.0
        Peierls['n'] =  n
        Peierls['sigp0'] = 5.9e9					# Pa (+/- 0.2e9 Pa)
        Peierls['A'] = 1.4e-7/np.power(mpa,n) 	# s^-1 Pa^-2
        Peierls['E'] = 320e3  					# J/mol (+/-50e3 J/mol)
    elif flv == "Idrissi16":
        Peierls['q'] = 2.0
        Peierls['p'] = 0.5
        n = 0.0
        Peierls['n'] =  n
        Peierls['sigp0'] = 3.8e9					# Pa (+/- 0.7e9 Pa)
        Peierls['A'] = 1e6 	# s^-1, note unit of A is related to n (here n = 0.0)
        Peierls['E'] = 566e3  					# J/mol (+/-74e3 J/mol)
    else:
        raise ValueError("flv must by \'MK10\'")
    return Peierls

def GetPeierlsStressPDependence(flv):
    '''
    export P dependence for the peierls creep
    '''
    G0 = 77.4*1e9 # GPa  
    Gp = 1.61 # GPa/GPa 
    return G0, Gp

def ReplacePhaseOption(str_in, key, idx, new_option):
    '''
    Replace the options for a phase
    Inputs:
        str_in: input string
        key: key of the designated composition
        idx: inded of the designated phase
        new_option: option to set
    '''
    has_comp = (len(str_in.split(',')) > 1)
    if has_comp:
        comp = COMPOSITION(str_in)
        comp.data[key][idx] = new_option
        str_out = comp.parse_back() 
    else:
        if key == "background":
            str_out = "%.4e" % new_option
        else:
            raise KeyError("No composition in str_in (%s) and the key is not background" % str_in)
    return str_out 
            
class CASE_TWOD(CASE):
    '''
    class for a case
    More Attributes:
    '''
    def configure_prm(self, if_wb, geometry, box_width, type_of_bd, potential_T,\
    sp_rate, ov_age, prescribe_T_method, if_peierls, if_couple_eclogite_viscosity, phase_model,\
    HeFESTo_data_dir, sz_cutoff_depth, adjust_mesh_with_width, rf_scheme, peierls_scheme,\
    peierls_two_stage_time, mantle_rheology_scheme, stokes_linear_tolerance, end_time,\
    refinement_level, case_o_dir, sz_viscous_scheme, cohesion, friction, crust_cohesion, crust_friction,\
    sz_constant_viscosity, branch, partitions, sz_minimum_viscosity, use_embeded_fault, Dsz, ef_factor, ef_Dbury,\
    sp_age_trench, use_embeded_fault_feature_surface, ef_particle_interval, delta_trench, eclogite_max_P, eclogite_match,\
    version, n_crust_layer, upper_crust_rheology_scheme, lower_crust_rheology_scheme, sp_trailing_length, ov_trailing_length,\
    slab_core_viscosity, mantle_coh, minimum_viscosity, fix_boudnary_temperature_auto, maximum_repetition_slice,\
    global_refinement, adaptive_refinement, rm_ov_comp, comp_method, peierls_flow_law, reset_density,\
    maximum_peierls_iterations, CDPT_type, use_new_rheology_module, fix_peierls_V_as, prescribe_T_width,\
    prescribe_T_with_trailing_edge, plate_age_method, jump_lower_mantle, use_3d_da_file, use_lookup_table_morb,\
    lookup_table_morb_mixing, delta_Vdiff, slope_410, slope_660, slab_strength, box_height, minimum_particles_per_cell, maximum_particles_per_cell,\
    refine_wedge, output_heat_flux,  delta_Ediff, delta_Edisl, delta_Vdisl, include_meta):
        Ro = 6371e3
        self.configure_case_output_dir(case_o_dir)
        o_dict = self.idict.copy()

        if plate_age_method == 'adjust box width only assigning age': 
            trench = get_trench_position_with_age(sp_age_trench, sp_rate, geometry, Ro)
        else:
            trench = get_trench_position(sp_age_trench, sp_rate, geometry, Ro, sp_trailing_length)

        # velocity boundaries
        if type_of_bd == "all free slip":  # boundary conditions
            pass
        elif type_of_bd == "top prescribed":
            # assign a 0.0 value for the overiding plate velocity
            # the subducting plate velocity is consistent with the value used in the worldbuilder
            bd_v_dict = prm_top_prescribed(trench, sp_rate, 0.0, refinement_level, delta_trench=delta_trench)
            o_dict["Boundary velocity model"] = bd_v_dict
        elif type_of_bd == "top prescribed with bottom right open":
            # assign a 0.0 value for the overiding plate velocity
            # the subducting plate velocity is consistent with the value used in the worldbuilder
            bd_v_dict, bd_t_dict = prm_top_prescribed_with_bottom_right_open(trench, sp_rate, 0.0, refinement_level, delta_trench=delta_trench)
            o_dict["Boundary velocity model"] = bd_v_dict
            o_dict["Boundary traction model"] = bd_t_dict
        elif type_of_bd == "top prescribed with bottom left open":
            # assign a 0.0 value for the overiding plate velocity
            # the subducting plate velocity is consistent with the value used in the worldbuilder
            bd_v_dict, bd_t_dict = prm_top_prescribed_with_bottom_left_open(trench, sp_rate, 0.0, refinement_level)
            o_dict["Boundary velocity model"] = bd_v_dict
            o_dict["Boundary traction model"] = bd_t_dict
        # directory to put outputs
        if branch != "":
            if branch == "master":
                branch_str = ""
            else:
                branch_str = "_%s" % branch
            o_dict["Additional shared libraries"] =  "$ASPECT_SOURCE_DIR/build%s/prescribe_field/libprescribed_temperature.so, \
$ASPECT_SOURCE_DIR/build%s/visco_plastic_TwoD/libvisco_plastic_TwoD.so, \
$ASPECT_SOURCE_DIR/build%s/isosurfaces_TwoD1/libisosurfaces_TwoD1.so" % (branch_str, branch_str, branch_str)
        # solver schemes
        if abs((stokes_linear_tolerance-0.1)/0.1) > 1e-6:
            # default is negative, thus do nothing
            o_dict["Solver parameters"]["Stokes solver parameters"]["Linear solver tolerance"] = str(stokes_linear_tolerance)
        # time of computation
        if abs((end_time - 60e6)/60e6) > 1e-6:
            o_dict["End time"] = str(end_time)
        # Adiabatic surface temperature
        o_dict["Adiabatic surface temperature"] = str(potential_T)
        # geometry model
        # repitition: figure this out by deviding the dimensions with a unit value of repitition_slice
        # The repitition_slice is defined by a max_repitition_slice and the minimum value compared with the dimensions
        repetition_slice = np.min(np.array([maximum_repetition_slice, 2.8900e6]))
        if geometry == 'chunk':
            max_phi = box_width / Ro * 180.0 / np.pi  # extent in term of phi
            if rf_scheme == "3d consistent":
                y_extent_str = "2.8900e6"
                if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
                    y_extent_str = "%.4e" % box_height
                if adjust_mesh_with_width:
                    x_repetitions = int(np.ceil(int((box_width/repetition_slice) * 2.0) / 2.0))
                    y_repetitions = int(np.ceil(int((box_height/repetition_slice) * 2.0) / 2.0))
                else:
                    x_repetitions = 2
                    y_repetitions = 1
                o_dict["Geometry model"] = {
                    "Model name": "chunk",
                    "Chunk": {
                        "Chunk inner radius": "%.4e" % (6371e3 - box_height),
                        "Chunk outer radius": "%.4e" % 6371e3,
                        "Chunk minimum longitude": "0.0",
                        "Chunk maximum longitude": "%.4e" % max_phi,
                        "Longitude repetitions": "%d" % x_repetitions,
                        "Radius repetitions": "%d" % y_repetitions
                    }
                }
            else:
                o_dict["Geometry model"] = prm_geometry_sph(max_phi, adjust_mesh_with_width=adjust_mesh_with_width)
        elif geometry == 'box':
            y_extent_str = "2.8900e6"
            if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
                y_extent_str = "%.4e" % box_height
            if adjust_mesh_with_width:
                x_repetitions = int(np.ceil(int((box_width/repetition_slice) * 2.0) / 2.0))
                y_repetitions = int(np.ceil(int((box_height/repetition_slice) * 2.0) / 2.0))
            else:
                x_repetitions = 2
                y_repetitions = 1
            o_dict["Geometry model"] = {
                "Model name": "box",
                "Box": {
                    "X extent": "%.4e" % box_width,
                    "Y extent": y_extent_str,
                    "X repetitions": "%d" % x_repetitions
                }
            }
            if y_repetitions > 1:
                o_dict["Geometry model"]["Box"]["Y repetitions"] = "%d" % y_repetitions
        # refinement
        if rf_scheme == "2d":
            if refinement_level > 0:
                # these options only take effects when refinement level is positive
                if refinement_level == 9:
                    # this is only an option if the input is positive
                    o_dict["Mesh refinement"]["Initial global refinement"] = "5"
                    o_dict["Mesh refinement"]["Initial adaptive refinement"] = "4"
                elif refinement_level == 10:
                    o_dict["Mesh refinement"]["Initial global refinement"] = "5"
                    o_dict["Mesh refinement"]["Initial adaptive refinement"] = "5"
                    pass
                elif refinement_level == 11:
                    o_dict["Mesh refinement"]["Initial global refinement"] = "6"
                    o_dict["Mesh refinement"]["Initial adaptive refinement"] = "5"
                elif refinement_level == 12:
                    o_dict["Mesh refinement"]["Initial global refinement"] = "6"
                    o_dict["Mesh refinement"]["Initial adaptive refinement"] = "6"
                elif refinement_level == 13:
                    o_dict["Mesh refinement"]["Initial global refinement"] = "7"
                    o_dict["Mesh refinement"]["Initial adaptive refinement"] = "6"
                else:
                    raise NotImplementedError()
                o_dict["Mesh refinement"]["Minimum refinement level"] = o_dict["Mesh refinement"]["Initial global refinement"]
                if geometry == 'chunk':
                    if plate_age_method == 'adjust box width only assigning age': 
                        trench = get_trench_position_with_age(sp_age_trench, sp_rate, geometry, Ro)
                    else:
                        trench = get_trench_position(sp_age_trench, sp_rate, geometry, Ro, sp_trailing_length)
                    o_dict["Mesh refinement"]['Minimum refinement function'] = prm_minimum_refinement_sph(refinement_level=refinement_level, refine_wedge=refine_wedge, trench=trench)
                elif geometry == 'box':
                    o_dict["Mesh refinement"]['Minimum refinement function'] = prm_minimum_refinement_cart(refinement_level=refinement_level)
            else:
                if geometry == 'chunk':
                    o_dict["Mesh refinement"]['Minimum refinement function'] = prm_minimum_refinement_sph()
                elif geometry == 'box':
                    o_dict["Mesh refinement"]['Minimum refinement function'] = prm_minimum_refinement_cart()
        # adjust refinement with different schemes
        elif rf_scheme == "3d consistent":
            # 3d consistent scheme:
            #   the global and adaptive refinement are set up by two variables separately
            #   remove additional inputs
            o_dict["Mesh refinement"]["Initial global refinement"] = "%d" % global_refinement
            o_dict["Mesh refinement"]["Initial adaptive refinement"] = "%d" % adaptive_refinement
            o_dict["Mesh refinement"]["Minimum refinement level"] = "%d" % global_refinement
            o_dict["Mesh refinement"]["IsosurfacesTwoD1"].pop("Depth for coarsening the lower mantle")
            o_dict["Mesh refinement"]["IsosurfacesTwoD1"].pop("Level for coarsening the lower mantle")
            if geometry == "box":
                o_dict["Mesh refinement"]["Minimum refinement function"]["Coordinate system"] = "cartesian"
                o_dict["Mesh refinement"]["Minimum refinement function"]["Variable names"] = "x, y, t"
                Do_str = "2.8900e+06" # strong for box height
                if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
                    Do_str = "%.4e" % box_height
                o_dict["Mesh refinement"]["Minimum refinement function"]["Function expression"] = \
                    "(Do-y<UM)?\\\n\
                                        ((Do-y<Dp+50e3)? Rd: Rum)\\\n\
                                        :0"
                o_dict["Mesh refinement"]["Minimum refinement function"]["Function constants"] = \
                    "Do=%s, UM=670e3, Dp=100e3, Rd=%d, Rum=%d" %\
                        (Do_str, global_refinement+adaptive_refinement-1, global_refinement+adaptive_refinement-2)
            else:
                o_dict["Mesh refinement"]["Minimum refinement function"]["Function constants"] = \
                    "Ro=6371e3, UM=670e3, Dp=100e3, Rd=%d, Rum=%d" %\
                        (global_refinement+adaptive_refinement-1, global_refinement+adaptive_refinement-2)
                o_dict["Mesh refinement"]["Minimum refinement function"]["Function expression"] = \
                    "(Ro-r<UM)?\\\n\
                                        ((Ro-r<Dp+50e3)? Rd: Rum)\\\n\
                                        :0"
        # boundary temperature model
        # 1. assign the option from sph and cart model, respectively
        # 2. check if we want to automatically fix the boundary temperature
        if geometry == 'chunk':
            o_dict['Boundary temperature model'] = prm_boundary_temperature_sph()
        elif geometry == 'box':
            o_dict['Boundary temperature model'] = prm_boundary_temperature_cart()
        else:
            pass
        if fix_boudnary_temperature_auto:
            # boudnary temperature: figure this out from the depth average profile
            assert(self.da_Tad_func is not None)
            try:
                Tad_bot = self.da_Tad_func(box_height) # bottom adiabatic temperature
            except ValueError:
                # in case this is above the given range of depth in the depth_average
                # file, apply a slight variation and try again
                Tad_bot = self.da_Tad_func(box_height - 50e3)
            if geometry == "box":
                o_dict['Boundary temperature model']['Box']['Bottom temperature'] = "%.4e" % Tad_bot
            elif geometry == "chunk":
                o_dict['Boundary temperature model']['Spherical constant']['Inner temperature'] = "%.4e" % Tad_bot
       
        # compositional fields
        # note the options for additional compositions and less compositions are handled later
        # options for using the particle method
        if comp_method == "particle":
            o_dict = change_field_to_particle(o_dict, minimum_particles_per_cell=minimum_particles_per_cell, maximum_particles_per_cell=maximum_particles_per_cell)
        
        # set up subsection reset viscosity function
        visco_plastic_twoD = self.idict['Material model']['Visco Plastic TwoD']
        if geometry == 'chunk':
            o_dict['Material model']['Visco Plastic TwoD'] =\
              prm_visco_plastic_TwoD_sph(visco_plastic_twoD, max_phi, type_of_bd, sp_trailing_length, ov_trailing_length)
        elif geometry == 'box':
            o_dict['Material model']['Visco Plastic TwoD'] =\
              prm_visco_plastic_TwoD_cart(visco_plastic_twoD, box_width, box_height, type_of_bd, sp_trailing_length,\
                                           ov_trailing_length, Dsz, reset_density=reset_density)
       
        # set up subsection Prescribed temperatures
        if geometry == 'chunk':
            if prescribe_T_method not in ['plate model 1', 'default']:
                prescribe_T_method = "default" # reset to default
            o_dict['Prescribed temperatures'] =\
                prm_prescribed_temperature_sph(max_phi, potential_T, sp_rate, ov_age, model_name=prescribe_T_method, area_width=prescribe_T_width)
            if type_of_bd == "all free slip":
                o_dict["Prescribe internal temperatures"] = "true"
        elif geometry == 'box':
            if prescribe_T_method == 'function':
                o_dict['Prescribed temperatures'] =\
                    prm_prescribed_temperature_cart(box_width, potential_T, sp_rate, ov_age)
            elif prescribe_T_method == 'plate model':
                o_dict['Prescribed temperatures'] =\
                    prm_prescribed_temperature_cart_plate_model(box_width, potential_T, sp_rate, ov_age)
            elif prescribe_T_method == 'plate model 1':
                o_dict['Prescribed temperatures'] =\
                    prm_prescribed_temperature_cart_plate_model_1(box_width, box_height, potential_T, sp_rate, ov_age, area_width=prescribe_T_width)
                o_dict["Prescribe internal temperatures"] = "true"

        if type_of_bd in ["top prescribed with bottom right open", "top prescribed with bottom left open", "top prescribed"]:
            # in this case, I want to keep the options for prescribing temperature but to turn it off at the start
            o_dict["Prescribe internal temperatures"] = "false" # reset this to false as it doesn't work for now
        if prescribe_T_with_trailing_edge == 0:
            if sp_trailing_length > 1e-6 or ov_trailing_length > 1e-6:
                # in case the trailing edge doesn't touch the box, reset this
                # this doesn't work for now
                o_dict["Prescribe internal temperatures"] = "false"
        
        # Material model
        if use_3d_da_file:
            da_file = os.path.join(LEGACY_FILE_DIR, 'reference_ThD', "depth_average.txt")
        else:
            da_file = os.path.join(LEGACY_FILE_DIR, 'reference_TwoD', "depth_average.txt")
        assert(os.path.isfile(da_file))

        # CDPT model
        if phase_model == "CDPT":
            CDPT_set_parameters(o_dict, CDPT_type, slope_410=slope_410, slope_660=slope_660)

        if use_new_rheology_module == 1:
            Operator = RHEOLOGY_OPR()
        else:
            raise NotImplementedError("Need to include Rheology_old_Dec_2023.RHEOLOGY_OPR")
            # Operator = Rheology_old_Dec_2023.RHEOLOGY_OPR()

        # mantle rheology
        Operator.ReadProfile(da_file)
        rheology = {}
        if mantle_rheology_scheme == "HK03_wet_mod_twod":  # get the type of rheology
            # note that the jump on 660 is about 15.0 in magnitude
            # deprecated
            rheology,viscosity_profile = Operator.MantleRheology(rheology="HK03_wet_mod", dEdiff=-40e3, dEdisl=30e3,\
    dVdiff=-5.5e-6, dVdisl=2.12e-6, save_profile=1, dAdiff_ratio=0.33333333333, dAdisl_ratio=1.040297619, save_json=1,\
    jump_lower_mantle=15.0, Coh=mantle_coh)
            if sz_viscous_scheme == "constant" and\
                abs(sz_constant_viscosity - 1e20)/1e20 < 1e-6 and slab_core_viscosity < 0.0:  # assign the rheology
                if abs(minimum_viscosity - 1e18) / 1e18 > 1e-6:
                    # modify the minimum viscosity if a different value is given
                    o_dict['Material model']['Visco Plastic TwoD']['Minimum viscosity'] = str(minimum_viscosity)
            else:
                CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
                sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03_wet_mod_twod1":  # get the type of rheology
            # note that the jump on 660 is about 15.0 in magnitude
            # fix the issue that in the previous scheme, the rheolog is not assigned to the prm.
            rheology, viscosity_profile = Operator.MantleRheology(rheology="HK03_wet_mod", dEdiff=-40e3, dEdisl=30e3,\
    dVdiff=-5.5e-6, dVdisl=2.12e-6, save_profile=1, dAdiff_ratio=0.33333333333, dAdisl_ratio=1.040297619, save_json=1,\
    jump_lower_mantle=15.0, Coh=mantle_coh)
            if sz_viscous_scheme == "constant" and\
                abs(sz_constant_viscosity - 1e20)/1e20 < 1e-6 and slab_core_viscosity < 0.0:  # assign the rheology
                if abs(minimum_viscosity - 1e18) / 1e18 > 1e-6:
                    # modify the minimum viscosity if a different value is given
                    o_dict['Material model']['Visco Plastic TwoD']['Minimum viscosity'] = str(minimum_viscosity)
                print("mantle_coh: ", mantle_coh)  # debug
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03_wet_mod_weakest_diffusion":
            # in this rheology, I maintained the prefactors from the derivation of the "HK03_wet_mod" rheology
            rheology, viscosity_profile = Operator.MantleRheology(rheology="HK03_wet_mod", dEdiff=-40e3, dEdisl=20e3,\
    dVdiff=-5.5e-6, dVdisl=-1.2e-6, save_profile=1, save_json=1, jump_lower_mantle=15.0, Coh=mantle_coh)
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03":
            # in this one, I don't include F because of the issue related to pressure calibration
            rheology, viscosity_profile = Operator.MantleRheology(rheology=mantle_rheology_scheme, use_effective_strain_rate=False, save_profile=1, save_json=1,\
    jump_lower_mantle=15.0)
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03_const":
            # use the const coh = 1000.0 rheology in HK03
            rheology, viscosity_profile = Operator.MantleRheology(rheology="HK03", use_effective_strain_rate=True, save_profile=1, save_json=1,\
                        jump_lower_mantle=15.0)
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03_dry":
            # use dry olivine rheology
            rheology, viscosity_profile = Operator.MantleRheology(rheology="HK03_dry", use_effective_strain_rate=True, save_profile=1, save_json=1,\
                        jump_lower_mantle=15.0)
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        elif mantle_rheology_scheme == "HK03_WarrenHansen23":
            # read in the original rheology
            rheology_name = "WarrenHansen23"
            rheology_prm_dict = RHEOLOGY_PRM()
            diffusion_creep_ori = getattr(rheology_prm_dict, rheology_name + "_diff")
            dislocation_creep_ori = getattr(rheology_prm_dict, rheology_name + "_disl")
            rheology_dict = {'diffusion': diffusion_creep_ori, 'dislocation': dislocation_creep_ori}
            # prescribe the correction
            diff_correction = {'A': 1.0, 'p': 0.0, 'r': 0.0, 'n': 0.0, 'E': delta_Ediff, 'V': delta_Vdiff}
            disl_correction = {'A': 1.0, 'p': 0.0, 'r': 0.0, 'n': 0.0, 'E': delta_Edisl, 'V': delta_Vdisl}
            # prescribe the reference state
            ref_state = {}
            ref_state["Coh"] = mantle_coh # H / 10^6 Si
            ref_state["stress"] = 50.0 # MPa
            ref_state["P"] = 100.0e6 # Pa
            ref_state["T"] = 1250.0 + 273.15 # K
            ref_state["d"] = 15.0 # mu m
            # refit rheology
            print("refit rheology") # debug
            rheology_dict_refit = RefitRheology(rheology_dict, diff_correction, disl_correction, ref_state)
            # derive mantle rheology
            rheology, viscosity_profile = Operator.MantleRheology(assign_rheology=True, diffusion_creep=rheology_dict_refit['diffusion'],\
                                                        dislocation_creep=rheology_dict_refit['dislocation'], save_profile=1,\
                                                        use_effective_strain_rate=True, save_json=1, Coh=mantle_coh,\
                                                        jump_lower_mantle=jump_lower_mantle)
            print("rheology_dict_refit: ", rheology_dict_refit) # debug
            print("rheology: ", rheology) # debug
            # assign to the prm file
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        else:
            # default is to fix F
            rheology, viscosity_profile = Operator.MantleRheology(rheology=mantle_rheology_scheme, save_profile=1, save_json=1)
            CDPT_assign_mantle_rheology(o_dict, rheology, sz_viscous_scheme=sz_viscous_scheme, sz_constant_viscosity=sz_constant_viscosity,\
            sz_minimum_viscosity=sz_minimum_viscosity, slab_core_viscosity=slab_core_viscosity, minimum_viscosity=minimum_viscosity)
        # record the upper mantle rheology
        um_diffusion_creep = rheology['diffusion_creep']
        um_dislocation_creep = rheology['dislocation_creep']
        self.viscosity_profile=viscosity_profile

        # these files are generated with the rheology variables
        self.output_files.append(Operator.output_json)
        self.output_files.append(Operator.output_json_aspect)
        self.output_imgs.append(Operator.output_profile) # append plot of initial conition to figures
        # yielding criteria
        if sz_viscous_scheme == "stress dependent":
            CDPT_assign_yielding(o_dict, cohesion, friction, crust_cohesion=crust_cohesion, crust_friction=crust_friction\
            , if_couple_eclogite_viscosity=if_couple_eclogite_viscosity)
        else:
            CDPT_assign_yielding(o_dict, cohesion, friction)
        # append to initial condition output
        if sz_viscous_scheme == "stress dependent":
            brittle_yielding = {}
            brittle_yielding['cohesion'] = crust_cohesion
            brittle_yielding['friction'] = np.tan(crust_friction * np.pi / 180.0)
            brittle_yielding['type'] = 'Coulumb'
            Operator_Sp = STRENGTH_PROFILE()
            rheology_experiment_dislocation = ConvertFromAspectInput(rheology['dislocation_creep'])
            Operator_Sp.SetRheology(disl=rheology_experiment_dislocation, brittle=brittle_yielding)
            # save a strength figure: uncomment
            # fig_path = os.path.join(ASPECT_LAB_DIR, "results", "shear_zone_strength.png")
            # PlotShearZoneStrengh(Operator_Sp, fig_path) # deprecated
            # self.output_imgs.append(fig_path)
        # Change the slab strength by the maximum yield stress, the default value is 500 Mpa
        if abs(slab_strength - 500e6) / 500e6 > 1e-6:
            o_dict['Material model']['Visco Plastic TwoD']["Maximum yield stress"] = "%.4e" % slab_strength
#        if slab_core_viscosity > 0.0:
#            # assign a strong core inside the slab
#            o_dict['Material model']['Visco Plastic TwoD']['Minimum viscosity'] =\
#                "background: %.4e, spcrust: %.4e, spharz: %.4e, opcrust: %.4e, opcrust: %.4e" %\
#                    (minimum_viscosity, minimum_viscosity, slab_core_viscosity, minimum_viscosity, minimum_viscosity)

        # Include peierls rheology
        if if_peierls:
            # The inputs are taken care of based on the scheme to use (in the following if blocks)
            try:
                temp = o_dict['Material model']['Visco Plastic TwoD']['Peierls fitting parameters']
            except KeyError as e:
                raise KeyError('The options use Peierls rheology by there are missing parameters in the prm file') from e
            o_dict['Material model']['Visco Plastic TwoD']['Include Peierls creep'] = 'true'
            if peierls_scheme in ["MK10", "MK10p"]: 
                Peierls = GetPeierlsApproxVist('MK10')
                # fix peierls activation volume as mantle rheology
                o_dict['Material model']['Visco Plastic TwoD']['Peierls glide parameters p'] = str(Peierls['p'])
                o_dict['Material model']['Visco Plastic TwoD']['Peierls glide parameters q'] = str(Peierls['q'])
                o_dict['Material model']['Visco Plastic TwoD']['Stress exponents for Peierls creep'] = str(Peierls['n'])
                o_dict['Material model']['Visco Plastic TwoD']['Peierls stresses'] = '%.4e' % Peierls['sigp0']
                o_dict['Material model']['Visco Plastic TwoD']['Activation energies for Peierls creep'] = '%.4e' % Peierls['E']
                o_dict['Material model']['Visco Plastic TwoD']['Activation volumes for Peierls creep'] = '0.0'
                if fix_peierls_V_as is not "":
                    o_dict['Material model']['Visco Plastic TwoD']['Activation volume differences for Peierls creep'] =\
                        '%.4e' % rheology[fix_peierls_V_as + '_creep']['V']
                A = Peierls['A']
                if phase_model == "CDPT":
                    # note that this part contains the different choices of phases
                    # in order to set up for the lower mantle compositions
                    # a future implementation could indicate in the phases which are lower mantle compositions
                    if peierls_flow_law == "exact":
                        o_dict['Material model']['Visco Plastic TwoD']['Prefactors for Peierls creep'] = "%.4e" % A
                                        # o_dict['Material model']['Visco Plastic TwoD']['Peierls strain rate residual tolerance'] = '%.4e' % 1e-22
                        o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                            'Peierls strain rate residual tolerance', '%.4e' % 1e-22,'Peierls shear modulus derivative')
                        # o_dict['Material model']['Visco Plastic TwoD']['Maximum Peierls strain rate iterations'] = '%d' % 40
                        o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                            'Maximum Peierls strain rate iterations', '%d' % maximum_peierls_iterations, 'Peierls strain rate residual tolerance')
                        # note that this part contains the different choices of phases
                        # in order to set up for the Idrissi flow law
                        # an additional parameter of Cutoff pressure is needed
                        o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                                                                            'Cutoff pressures for Peierls creep',\
                                                                            "background: 2.5e+10|2.5e+10|2.5e+10|2.5e+10|0.0|0.0|0.0|0.0, spcrust: 0.0|2.5e+10|0.0|0.0, spharz: 2.5e+10|2.5e+10|2.5e+10|2.5e+10|0.0|0.0|0.0|0.0, opcrust: 2.5e+10, opharz: 2.5e+10", \
                                                                            "Maximum Peierls strain rate iterations")

                    else:
                        o_dict['Material model']['Visco Plastic TwoD']['Prefactors for Peierls creep'] = \
                        "background: %.4e|%.4e|%.4e|%.4e|1e-31|1e-31|1e-31|1e-31,\
    spcrust: %.4e|%.4e|1e-31|1e-31,\
    spharz: %.4e|%.4e|%.4e|%.4e|1e-31|1e-31|1e-31|1e-31,\
    opcrust: %.4e, opharz: %.4e" % (A, A, A, A, A, A, A, A, A, A, A, A)
                else:
                    pass  # not implemented
                if peierls_scheme == "MK10p":
                    # add p dependence in the peierls stress
                    G0, Gp = GetPeierlsStressPDependence()
                    o_dict['Material model']['Visco Plastic TwoD']['Peierls shear modulus'] = "%.4e" % G0
                    o_dict['Material model']['Visco Plastic TwoD']['Peierls shear modulus derivative'] =  "%.4e" % Gp
                else:
                    # no p dependence, note: sigp = sigp0*(1 + (Gp/G0)*P1), here we want to set Gp = 0
                    o_dict['Material model']['Visco Plastic TwoD']['Peierls shear modulus derivative'] = "0.0"
                    pass
            elif peierls_scheme == "Idrissi16":
                # Idrissi 16 flow law
                # unneeded variables (e.g. shear modulus), and new to add in (e.g. Peierls strain rate residual tolerance)
                # pay attention to the converting of the prefactor (depending on the n value. Here is 0.0)
                Peierls = GetPeierlsApproxVist('Idrissi16')
                o_dict['Material model']['Visco Plastic TwoD'].pop('Peierls shear modulus') # don't need this one
                o_dict['Material model']['Visco Plastic TwoD'].pop('Peierls shear modulus derivative') # don't need this one
                o_dict['Material model']['Visco Plastic TwoD']['Peierls creep flow law'] = 'exact'
                o_dict['Material model']['Visco Plastic TwoD']['Peierls glide parameters p'] = str(Peierls['p'])
                o_dict['Material model']['Visco Plastic TwoD']['Peierls glide parameters q'] = str(Peierls['q'])
                o_dict['Material model']['Visco Plastic TwoD']['Stress exponents for Peierls creep'] = str(Peierls['n'])
                o_dict['Material model']['Visco Plastic TwoD']['Peierls stresses'] = '%.4e' % Peierls['sigp0']
                o_dict['Material model']['Visco Plastic TwoD']['Activation energies for Peierls creep'] = '%.4e' % Peierls['E']
                o_dict['Material model']['Visco Plastic TwoD']['Activation volumes for Peierls creep'] = '0.0'
                o_dict['Material model']['Visco Plastic TwoD']['Peierls fitting parameters'] = '0.15'

                # o_dict['Material model']['Visco Plastic TwoD']['Peierls strain rate residual tolerance'] = '%.4e' % 1e-22
                o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                    'Peierls strain rate residual tolerance', '%.4e' % 1e-22,'Activation volumes for Peierls creep')
                # o_dict['Material model']['Visco Plastic TwoD']['Maximum Peierls strain rate iterations'] = '%d' % 40
                o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                    'Maximum Peierls strain rate iterations', '%d' % 40, 'Peierls strain rate residual tolerance')
                # o_dict['Material model']['Visco Plastic TwoD']['Cutoff stresses for Peierls creep'] = '%.4e' % 2e7
                o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                    'Cutoff stresses for Peierls creep', '%.4e' % 2e7, 'Maximum Peierls strain rate iterations')
                # o_dict['Material model']['Visco Plastic TwoD']['Apply strict stress cutoff for Peierls creep'] = 'true'
                o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                    'Apply strict stress cutoff for Peierls creep', 'true', 'Cutoff stresses for Peierls creep')
                A = Peierls['A']
                if phase_model == "CDPT":
                    # note that this part contains the different choices of phases
                    # in order to set up for the Idrissi flow law
                    # an additional parameter of Cutoff pressure is needed
                    o_dict['Material model']['Visco Plastic TwoD']['Prefactors for Peierls creep'] = "1.0e6"
                    o_dict['Material model']['Visco Plastic TwoD'] = insert_dict_after(o_dict['Material model']['Visco Plastic TwoD'],\
                                                                        'Cutoff pressures for Peierls creep',\
                                                                        "background: 1e+31|1e+31|1e+31|1e+31|0.0|0.0|0.0|0.0, \
spcrust: 1e+31|1e+31|0.0|0.0, \
spharz: 1e+31|1e+31|1e+31|1e+31|0.0|0.0|0.0|0.0, \
opcrust: 1e+31, opharz: 1e+31", \
                                                                        "Apply strict stress cutoff for Peierls creep")
        else:
            o_dict['Material model']['Visco Plastic TwoD']['Include Peierls creep'] = 'false'
        # eclogite transition
        if eclogite_match > 0:
            # assign a set of variables that matches the mineral phase transtions
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite transition"]["Pressure for eclogite transition"] = "1.5e9"
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite transition"]["Pressure slope for eclogite transition"] = "2.17e6"
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite transition"]["Pressure width for eclogite transition"] = "0.5e9"
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite transition"]["Average phase functions for eclogite transition"] = 'false'
        if abs(eclogite_max_P - 5e9)/5e9 > 1e-6:
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite transition"]["Max pressure for eclogite transition"] =\
            "%.4e" % eclogite_max_P # assign the max pressure for the eclogite transition
        # Couple eclogite viscosity
        if if_couple_eclogite_viscosity:
            o_dict['Material model']['Visco Plastic TwoD']["Decoupling eclogite viscosity"] = 'false'
        else:
            o_dict['Material model']['Visco Plastic TwoD']["Decoupling eclogite viscosity"] = 'true'
            o_dict['Material model']['Visco Plastic TwoD']["Eclogite decoupled viscosity"] =\
                {
                    "Decoupled depth": str(sz_cutoff_depth),
                    "Decoupled depth width": '10e3'
                }
            if n_crust_layer == 2:
                o_dict['Material model']['Visco Plastic TwoD']["Eclogite decoupled viscosity"]["Crust index"] = "3"
        # phase model
        if phase_model == "HeFESTo":
            o_dict['Material model']['Visco Plastic TwoD']["Use lookup table"] = 'true'
            o_dict['Material model']['Visco Plastic TwoD']["Lookup table"]["Data directory"] = HeFESTo_data_dir
            pass
        elif phase_model == "CDPT":
            o_dict['Material model']['Visco Plastic TwoD'].pop("Use lookup table", "Foo")
        # post-process
        # assign the options for the embeded-fault implementation of shear zone:
        #   1. add a section in the material model
        #   2. set up particles
        if use_embeded_fault:
            o_dict['Material model']['Visco Plastic TwoD']["Sz from embeded fault"] = 'true'
            o_dict['Material model']['Visco Plastic TwoD']["Sz embeded fault"] =\
            {
                "Sz composition index" : '0',\
                "Sz thickness minimum" : str(Dsz),\
                "Sz thickness maximum" :  str(ef_factor * Dsz),\
                "Sz depth" : str(sz_cutoff_depth),\
                "Sz particle bury depth" : str(ef_Dbury)
            }
            pp_dict = o_dict['Postprocess']
            # add particles in this section
            pp_dict["List of postprocessors"] += ', particles'
            pp_dict['Particles'] = {\
                "Data output format" : "vtu",\
                "List of particle properties" : "initial position",\
                "Time between data output": "0.1e6"\
            }
            o_dict['Postprocess'] = pp_dict

        # expand the layer of the crust to multiple layers
        if n_crust_layer == 1:
            # 1 layer in the crust, this is the default option
            pass
        elif n_crust_layer == 2:
            # 2 layers, expand the option of 'sp_crust' to 2 different layers
            o_dict = expand_multi_composition(o_dict, 'spcrust', ['spcrust_up', 'spcrust_low'])
            o_dict = expand_multi_composition(o_dict, 'opcrust', ['opcrust_up', 'opcrust_low'])
            # assign the rheology
            visco_plastic_dict = o_dict['Material model']['Visco Plastic TwoD']
            if use_new_rheology_module == 1:
                GetFunc = GetRheology
                AssignFunc = AssignAspectViscoPlasticPhaseRheology
            else:
                raise NotImplementedError()
                # GetFunc = Rheology_old_Dec_2023.GetRheology
                # AssignFunc = Rheology_old_Dec_2023.AssignAspectViscoPlasticPhaseRheology
            if upper_crust_rheology_scheme != "":
                diffusion_creep, dislocation_creep = GetFunc(upper_crust_rheology_scheme)
                visco_plastic_dict = AssignFunc(visco_plastic_dict, 'spcrust_up', 0, diffusion_creep, dislocation_creep)
            if lower_crust_rheology_scheme == "":
                pass
            elif lower_crust_rheology_scheme == "mantle":
                visco_plastic_dict = AssignFunc(visco_plastic_dict, 'spcrust_low', 0, um_diffusion_creep, um_dislocation_creep, no_convert=True)
            else:
                diffusion_creep, dislocation_creep = GetFunc(lower_crust_rheology_scheme)
                visco_plastic_dict = AssignFunc(visco_plastic_dict, 'spcrust_low', 0, diffusion_creep, dislocation_creep)
            o_dict['Material model']['Visco Plastic TwoD'] = visco_plastic_dict
        else:
            raise NotImplementedError()

        if include_meta:
            # expand composition fields
            o_dict = expand_multi_composition_isosurfaces(o_dict, 'opharz', ["opharz", "metastable", "meta_x0", "meta_x1", "meta_x2", "meta_x3", "meta_is", "meta_rate"])
            o_dict = expand_multi_composition_composition_field(o_dict, 'opharz', ["opharz", "metastable", "meta_x0", "meta_x1", "meta_x2", "meta_x3", "meta_is", "meta_rate"])
            o_dict["Compositional fields"]["Mapped particle properties"] = \
                  "spcrust:initial spcrust, spharz:initial spharz, opcrust:initial opcrust, opharz:initial opharz, metastable: kinetic metastable, meta_x0: kinetic meta_x0, meta_x1: kinetic meta_x1, meta_x2: kinetic meta_x2, meta_x3: kinetic meta_x3, meta_is: kinetic meta_is, meta_rate: kinetic meta_rate"

            # fix the partical properties
            particle_options = o_dict["Postprocess"]["Particles"]
            particle_visualization_options = {}
            particle_visualization_options["Data output format"] = particle_options.pop("Data output format")
            particle_visualization_options["Time between data output"] = particle_options.pop("Time between data output")
            o_dict["Postprocess"]["Particles"] = particle_visualization_options
            n_particles = particle_options.pop("Number of particles")
            particle_options["Generator"] = {"Random uniform": {"Number of particles": n_particles}}
            o_dict["Particles"] = particle_options

        
        # crustal phase transition
        if use_lookup_table_morb:
            visco_plastic_dict = o_dict['Material model']['Visco Plastic TwoD']
            # first reset the manual method
            visco_plastic_dict["Manually define phase method crust"] = "0.0"
            visco_plastic_dict["Decoupling eclogite viscosity"] = "false"
            if "Eclogite transition" in visco_plastic_dict:
                visco_plastic_dict.pop("Eclogite transition")
            if "Eclogite decoupled viscosity" in visco_plastic_dict:
                visco_plastic_dict.pop("Eclogite decoupled viscosity")
            if "Use lookup table" in visco_plastic_dict:
                visco_plastic_dict["Use lookup table"] = "false"
            if "Lookup table" in visco_plastic_dict:
                visco_plastic_dict.pop("Lookup table")

            visco_plastic_dict["Use lookup table morb"] = "true"
            visco_plastic_dict["Use phase rheology mixing"] = "true"
            visco_plastic_dict["Phase rheology mixing models"] = "0, %d, 0, 0, 0" % lookup_table_morb_mixing
            if n_crust_layer == 2:
                visco_plastic_dict["Phase rheology mixing models"] = "0, 0, 0, %d, 0, 0, 0" % lookup_table_morb_mixing
            morb_index = "1"
            if n_crust_layer == 2:
                morb_index = "4"
            visco_plastic_dict["Lookup table morb"] = {
                "Data directory": "$ASPECT_SOURCE_DIR/lookup_tables/",
                "Material file names": "perplex_morb_test.txt",
                "Morb composition index": morb_index,
                "Cutoff eclogite phase below": "0.25",
                "Bilinear interpolation": "true",
                "Cutoff eclogite phase above T1": "1673.0",
                "Cutoff eclogite phase above P1": "3.6e+09",
                "Cutoff eclogite phase above T2": "0.0",
                "Cutoff eclogite phase above P2": "7.8e+09",
                "Rewrite morb density": "false",
                "Eclogite phase divisor": "0.8"
            }
            o_dict['Material model']['Visco Plastic TwoD'] = visco_plastic_dict
        
        # remove the overiding plate composition
        if rm_ov_comp:
            o_dict = remove_composition(o_dict, 'opcrust')
            o_dict = remove_composition(o_dict, 'opharz')
            # modify options for the rheology

        # apply the changes 
        self.idict = o_dict

        # create a multi-stage model
        if if_peierls and (peierls_two_stage_time > 0):
            o_dict1 = deepcopy(o_dict)
            # for stage 1
            o_dict['Material model']['Visco Plastic TwoD']['Include Peierls creep'] = 'false'
            o_dict['End time'] = '%.4e' % peierls_two_stage_time
            # for stage 2
            o_dict1['Resume computation'] = 'true'
            self.model_stages = 2
            self.additional_idicts.append(o_dict1)

        # additional outputs 
        # adjust list of output for different versions
        if version >= 3.0:
            o_dict['Postprocess']["Visualization"]["List of output variables"] = 'material properties, named additional outputs, nonadiabatic pressure, strain rate, stress, heating'
        # heat flux outputs
        if output_heat_flux:
            o_dict['Postprocess']["List of postprocessors"] += ', heat flux map'
            o_dict['Postprocess']["Visualization"]["List of output variables"] += ', heat flux map'
            o_dict['Postprocess']["Visualization"]["Heat flux map"] = {"Output point wise heat flux": "true"}

    def configure_wb(self, if_wb, geometry, potential_T, sp_age_trench, sp_rate, ov_ag,\
        if_ov_trans, ov_trans_age, ov_trans_length, is_box_wider, Dsz, wb_new_ridge, version,\
        n_crust_layer, Duc, n_comp, sp_trailing_length, ov_trailing_length, box_width, rm_ov_comp,\
        plate_age_method):
        '''
        Configure world builder file
        Inputs:
            see description of CASE_OPT
        '''
        if not if_wb:
            # check first if we use wb file for this one
            return
        # potential T
        self.wb_dict['potential mantle temperature'] = potential_T

        # todo_version
        # adjust world builder version
        if version >= 3.0:
            self.wb_dict["version"] = "1.1"

        # geometry
        if geometry == 'chunk':
            # fix the depth method:
            #   a update with the "begin at end segment" method to use
            # fix the width of box
            self.wb_dict["coordinate system"] = {"model": "spherical", "depth method": "begin segment"}
            if version < 1.0:
                pass
            elif version < 2.0:
                self.wb_dict["coordinate system"]["depth method"] = "begin at end segment"
            elif version >= 3.0:
                self.wb_dict["coordinate system"]["depth method"] = "begin at end segment"
            else:
                raise NotImplementedError
            if is_box_wider:
                self.wb_dict["cross section"] = [[0, 0], [360.0, 0.0]]
            else:
                self.wb_dict["cross section"] = [[0, 0], [180.0, 0.0]]
        elif geometry == 'box':
            # delete the depth method
            #   it is not allowed in the cartesion geometry
            # fix the width of box
            self.wb_dict["coordinate system"]["model"] = "cartesian"
            self.wb_dict["coordinate system"].pop("depth method")  # remove depth method in this case
            if is_box_wider:
                self.wb_dict["cross section"] = [[0, 0], [1e7, 0.0]]
            else:
                self.wb_dict["cross section"] = [[0, 0], [2e7, 0.0]]
        else:
            raise ValueError('%s: geometry must by one of \"chunk\" or \"box\"' % func_name())
        # plates
        if geometry == 'chunk':
            if is_box_wider:
                max_sph = 360.0
            else:
                max_sph = 180.0
            Ro = float(self.idict['Geometry model']['Chunk']['Chunk outer radius'])
            # sz_thickness
            self.wb_dict = wb_configure_plates(self.wb_dict, sp_age_trench,\
            sp_rate, ov_ag, wb_new_ridge, version, n_crust_layer, Duc, plate_age_method, Ro=Ro, if_ov_trans=if_ov_trans, ov_trans_age=ov_trans_age,\
            ov_trans_length=ov_trans_length, geometry=geometry, max_sph=max_sph, sz_thickness=Dsz, n_comp=n_comp,\
            sp_trailing_length=sp_trailing_length, ov_trailing_length=ov_trailing_length, box_width=box_width,\
            rm_ov_comp=rm_ov_comp)
        elif geometry == 'box':
            if is_box_wider:
                Xmax = 2e7
            else:
                Xmax = 1e7  # lateral extent of the box
            Ro = float(self.idict['Geometry model']['Box']['Y extent'])
            self.wb_dict = wb_configure_plates(self.wb_dict, sp_age_trench,\
            sp_rate, ov_ag, wb_new_ridge, version, n_crust_layer, Duc, plate_age_method, Xmax=Xmax, if_ov_trans=if_ov_trans, ov_trans_age=ov_trans_age,\
            ov_trans_length=ov_trans_length, geometry=geometry, sz_thickness=Dsz, n_comp=n_comp, sp_trailing_length=sp_trailing_length,\
            ov_trailing_length=ov_trailing_length, box_width=box_width, rm_ov_comp=rm_ov_comp) # plates
        else:
            raise ValueError('%s: geometry must by one of \"chunk\" or \"box\"' % func_name())
    
    def configure_final(self, geometry, Dsz, use_embeded_fault, ef_Dbury, ef_particle_interval, use_embeded_fault_feature_surface):
        '''
        final step of configurations.
        1. fix the options for the embeded fault method
        '''
        # use the embeded fault implementation, here we need to
        # assign particle positions with world builder configuration
        if geometry == 'chunk':
            Ro = float(self.idict['Geometry model']['Chunk']['Chunk outer radius'])
        elif geometry == 'box':
            Ro = float(self.idict['Geometry model']['Box']['Y extent'])
        else:
            raise ValueError('%s: geometry must by one of \"chunk\" or \"box\"' % func_name())
        # find information of the slab
        i0 = FindWBFeatures(self.wb_dict, 'Slab')  # find trench position
        s_dict = self.wb_dict['features'][i0]
        trench = s_dict["coordinates"][0][0]
        p0 = np.array([trench, Ro]) # starting point of the slab, theta needs to be in radian
        segments = s_dict["segments"]  # find slab lengths and slab_dips
        slab_lengths = []
        slab_dips = []
        for i in range(len(segments)-1):
            # the last one is a ghost component for tapering, thus get rid of it
            segment = segments[i]
            slab_dips.append(segment["angle"])
            slab_lengths.append(segment["length"])
        # fix options for the embeded fault method
        if use_embeded_fault:
            if use_embeded_fault_feature_surface:
                # if the feature surface from the WorldBuilder, no need to generate particles manually
                # set up the variables instead
                n_particles_on_plate = int(Ro * trench * np.pi / 180.0 // ef_particle_interval)
                n_particles_on_slab = 0
                for slab_length in slab_lengths:
                    n_particles_on_slab += int(slab_length//ef_particle_interval)
                n_particles = n_particles_on_plate + n_particles_on_slab
                self.idict['Postprocess']['Particles']["Number of particles"] = str(n_particles)
                self.idict['Postprocess']['Particles']["Particle generator name"] = "world builder feature surface"
                self.idict['Postprocess']['Particles']["Generator"] = {\
                    "World builder feature surface":\
                    {\
                        "Number of particles on the slab": str(n_particles_on_slab),\
                        "Feature surface distance": "%.4e" % (Dsz + ef_Dbury),\
                        "Maximum radius": "%.4e" % Ro,\
                        "Minimum radius" : "%.4e" % (Ro - 200e3),\
                        "Feature start": "%.4e" % (trench * np.pi / 180.0),\
                        "Search start": "%.4e" % (trench * np.pi / 180.0),\
                        "Search length": "0.00174",\
                        "Search max step": "100"\
                    }\
                }
            else:
                self.idict['Postprocess']['Particles']["Particle generator name"] = "ascii file"
                self.idict['Postprocess']['Particles']["Generator"] = {
                    "Ascii file": {\
                        "Data directory": "./particle_file/",\
                        "Data file name": "particle.dat"\
                    }
                }
                # if not using the feature surface from the WorldBuilder, generate particles manually
                self.particle_data = particle_positions_ef(geometry, Ro, trench, Dsz, ef_Dbury, p0, slab_lengths, slab_dips, interval=ef_particle_interval)


def change_field_to_particle(i_dict, **kwargs):
    '''
    change field method to particle method.
    This function will automatically substitute all
    the options with the particle method
    Inputs:
        i_dict: a dictionary containing parameters of a case
    '''
    minimum_particles_per_cell = kwargs.get("minimum_particles_per_cell", 33)
    maximum_particles_per_cell = kwargs.get("maximum_particles_per_cell", 50)
    o_dict = deepcopy(i_dict)
    comp_dict = o_dict["Compositional fields"]
    nof = int(o_dict["Compositional fields"]["Number of fields"])
    # construct the new Compositional field methods
    comp_method_expression = ""
    is_first = True
    for i in range(nof):
        if is_first:
            is_first = False
        else:
            comp_method_expression += ", "
        comp_method_expression += "particles"
    comp_dict["Compositional field methods"] = comp_method_expression
    # map to particles
    mapped_properties_expression = ""
    field_name_exppression = comp_dict["Names of fields"]
    field_name_options = field_name_exppression.split(',')
    is_first = True
    for _option in field_name_options:
        if is_first:
            is_first = False
        else:
            mapped_properties_expression += ", "
        field_name = re_neat_word(_option) 
        mapped_properties_expression += "%s: initial %s" % (field_name, field_name)
    comp_dict["Mapped particle properties"] = mapped_properties_expression
    # parse back
    o_dict["Compositional fields"] = comp_dict
    # deal with the Postprocess section
    pp_dict = o_dict['Postprocess']
    pp_dict["List of postprocessors"] += ', particles'
    pp_dict['Particles'] = {\
        "Number of particles": "5e7",\
        "Minimum particles per cell": "%d" % minimum_particles_per_cell,\
        "Maximum particles per cell": "%d" % maximum_particles_per_cell,\
        "Load balancing strategy": "remove and add particles",\
        "Interpolation scheme": "cell average",\
        "Update ghost particles": "true",\
        "Particle generator name": "random uniform",\
        "Data output format" : "vtu",\
        "List of particle properties" : "initial composition",\
        "Time between data output": "0.1e6",\
        "Allow cells without particles": "true"
    }
    o_dict['Postprocess'] = pp_dict
    return o_dict


def expand_multi_composition(i_dict, comp0, comps):
    '''
    expand one composition to multiple compositions in the dictionary
    Inputs:
        i_dict: dictionary of inputs
        comp0: composition of inputs
        comps: compositions (list) to expand to
    '''

    temp_dict = expand_multi_composition_options(i_dict, comp0, comps)
    temp_dict = expand_multi_composition_isosurfaces(temp_dict, comp0, comps)
    o_dict  = expand_multi_composition_composition_field(temp_dict, comp0, comps)
    return o_dict

def duplicate_composition_option(str_in, comp0, comp1, **kwargs):
    '''
    duplicate the composition option,  parse to a new string
    Inputs:
        str_in: an input of string
        comp0, comp1: duplicate the option with comp0 to a new one with comp1
    Return:
        str_out: a new string for inputs of composition
    '''
    is_mapped_particle_properties = kwargs.get("is_mapped_particle_properties", False)
    # read in original options
    comp_in = COMPOSITION(str_in)
    var = comp_in.data[comp0]
    if is_mapped_particle_properties:
        # replace substring by new composition in value
        var = ["initial %s" % comp1]
    # duplicate option with a new composition
    comp_out = COMPOSITION(comp_in)
    comp_out.data[comp1] = var
    # convert to a new string & return
    str_out = comp_out.parse_back() 
    return str_out

def remove_composition_option(str_in, comp0):
    '''
    remove the composition option
    Inputs:
        str_in: an input of string
        comp0, comp1: remove the option with comp0
    Return:
        str_out: a new string for inputs of composition
    '''
    # read in original options
    comp_in = COMPOSITION(str_in)
    # move option to a new composition
    comp_out = COMPOSITION(comp_in)
    _ = comp_out.data.pop(comp0)
    # convert to a new string & return
    str_out = comp_out.parse_back() 
    return str_out

def expand_multi_composition_options(i_dict, comp0, comps):
    '''
    expand one composition to multiple compositions in the dictionary
    Inputs:
        i_dict: dictionary of inputs
        comp0: composition of inputs
        comps: compositions (list) to expand to
    '''
    o_dict = deepcopy(i_dict)

    for key, value in o_dict.items():
        if type(value) == dict:
            # in case of dictionary: iterate
            o_dict[key] = expand_multi_composition_options(value, comp0, comps)
        elif type(value) == str:
            # in case of string, try matching it with the composition
            if re.match('.*' + comp0, value) and re.match('.*:', value) and not re.match('.*;', value):
                try:
                    temp = value
                    for comp in comps:
                        if key == "Mapped particle properties":
                            temp = duplicate_composition_option(temp, comp0, comp, is_mapped_particle_properties=True)
                        else:
                            temp = duplicate_composition_option(temp, comp0, comp)
                    o_dict[key] = remove_composition_option(temp, comp0)
                except KeyError:
                    # this is not a string with options of compositions, skip
                    pass
        else:
            raise TypeError("value must be either dict or str")
    return o_dict


def expand_composition_array(old_str, id_comp0, n_comp):
    '''
    expand the composition options
    Inputs:
        old_str: the old option
        id_comp0: index of the option for the original composition
        n_comp: number of new compositions
    '''
    options = old_str.split(',')
    removed_option = options.pop(id_comp0)
    # compile new option
    new_option = ""
    for i in range(n_comp):
        options.append(removed_option)
    is_first = True
    for _option in options:
        if is_first:
            is_first = False
        else:
            new_option += ', '
        new_option += _option
    return new_option 


def expand_multi_composition_composition_field(i_dict, comp0, comps):
    '''
    expand one composition to multiple compositions in the dictionary
    Inputs:
        i_dict: dictionary of inputs
        comp0: composition of inputs
        comps: compositions (list) to expand to
    '''
    n_comp = len(comps)
    o_dict = deepcopy(i_dict)
    # composition field
    # find the index of the original field to be id0
    str_name = o_dict["Compositional fields"]["Names of fields"]
    str_name_options = str_name.split(',')
    id = 0
    found = False
    for _option in str_name_options:
        comp = re_neat_word(_option) 
        if comp == comp0:
            id0 = id
            found = True
            str_name_options.remove(comp)
        id += 1
    if not found:
        raise ValueError("the option of \"Names of fields\" doesn't have option of %s" % comp0)
    # add the new compositions to the original list of compositions
    for comp in comps:
        str_name_options.append(comp)
    str_name_new = ""
    is_first = True
    for _option in str_name_options:
        if is_first:
            is_first = False
        else:
            str_name_new += ","
        str_name_new += _option
    o_dict["Compositional fields"]["Names of fields"] = str_name_new
    # number of composition
    nof = int(o_dict["Compositional fields"]["Number of fields"])
    new_nof = nof + n_comp-1
    o_dict["Compositional fields"]["Number of fields"] = str(new_nof)
    # field method
    str_f_methods = o_dict["Compositional fields"]["Compositional field methods"]
    f_method = str_f_methods.split(',')[0]
    o_dict["Compositional fields"]["Compositional field methods"] += ("," + f_method) * (n_comp -1)
    # discretization option
    find_discretization_option = False
    try:
        discretization_option = o_dict["Discretization"]["Stabilization parameters"]['Global composition maximum']
        find_discretization_option = True
    except:
        pass
    if find_discretization_option:
        o_dict["Discretization"]["Stabilization parameters"]['Global composition maximum'] = expand_composition_array(discretization_option, id0, n_comp)
    find_discretization_option = False
    try:
        discretization_option = o_dict["Discretization"]["Stabilization parameters"]['Global composition minimum']
        find_discretization_option = True
    except:
        pass
    if find_discretization_option:
        o_dict["Discretization"]["Stabilization parameters"]['Global composition minimum'] = expand_composition_array(discretization_option, id0, n_comp)
    # option in the adiabatic temperature
    try:
        _ = o_dict["Initial temperature model"]["Adiabatic"]["Function"]["Function expression"]
        new_adiabatic_functiion_expression = ""
        is_first = True
        for i in range(new_nof):
            if is_first:
                is_first = False
            else:
                new_adiabatic_functiion_expression += "; "
            new_adiabatic_functiion_expression += "0.0"
        o_dict["Initial temperature model"]["Adiabatic"]["Function"]["Function expression"] = new_adiabatic_functiion_expression
    except KeyError:
        pass
    # option in the look up table
    try:
        look_up_index_option = o_dict["Material model"]["Visco Plastic TwoD"]["Lookup table"]["Material lookup indexes"]
        # note there is an entry for the background in this option, so I have to use id0 + 1 
        o_dict["Material model"]["Visco Plastic TwoD"]["Lookup table"]["Material lookup indexes"] = expand_composition_array(look_up_index_option, id0+1, n_comp)
    except KeyError:
        pass

    return o_dict
    

def expand_multi_composition_isosurfaces(i_dict, comp0, comps):
    '''
    expand one composition to multiple compositions in the dictionary
    Inputs:
        i_dict: dictionary of inputs
        comp0: composition of inputs
        comps: compositions (list) to expand to
    '''
    found = False
    # find the right option of isosurface
    try:
        isosurface_option = i_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"]
        isosurface_option_type = 1
        found = True
    except KeyError:
        try:
            isosurface_option = i_dict["Mesh refinement"]["IsosurfacesTwoD1"]["Isosurfaces"]
            isosurface_option_type = 2
            found = True
        except KeyError:
            pass
    # change the options of isosurface
    if found:
        # check that the target composition is in the string
        if re.match('.*' + comp0, isosurface_option):
            options = isosurface_option.split(";")
            has_composition = False
            for _option in options:
                if re.match('.*' + comp0, _option):
                    comp_option = _option
                    options.remove(_option)
                    has_composition = True
            # expand the composition
            if has_composition:
                for comp in comps:
                    options.append(re.sub(comp0, comp, comp_option))
                new_isosurface_option = ""
                is_first = True
                for _option in options:
                    if is_first:
                        is_first = False
                    else:
                        new_isosurface_option += ";"
                    new_isosurface_option += _option
                if isosurface_option_type == 1:
                    i_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] = new_isosurface_option
                elif isosurface_option_type == 2:
                    i_dict["Mesh refinement"]["IsosurfacesTwoD1"]["Isosurfaces"] = new_isosurface_option
    return i_dict

def remove_composition(i_dict, comp):
    '''
    remove one composition to multiple compositions in the dictionary
    Inputs:
        i_dict: dictionary of inputs
        comp: composition to remove
    '''
    temp_dict = remove_composition_options(i_dict, comp)
    temp_dict = remove_composition_isosurfaces(temp_dict, comp)
    o_dict  = remove_composition_composition_field(temp_dict, comp)
    return o_dict


def remove_composition_options(i_dict, comp):
    '''
    remove the options for one composition in the prm file
    Inputs:
        i_dict: dictionary of inputs
        comp: composition to remove
    '''
    o_dict = deepcopy(i_dict)

    for key, value in o_dict.items():
        if type(value) == dict:
            # in case of dictionary: iterate
            o_dict[key] = remove_composition_options(value, comp)
        elif type(value) == str:
            # in case of string, try matching it with the composition
            if re.match('.*' + comp, value) and re.match('.*:', value) and not re.match('.*;', value):
                try:
                    o_dict[key] = remove_composition_option(value, comp)
                except KeyError:
                    # this is not a string with options of compositions, skip
                    pass
        else:
            raise TypeError("value must be either dict or str")
    return o_dict


def remove_composition_isosurfaces(i_dict, comp):
    '''
    remove the isosurface options for one composition in the prm file
    Inputs:
        i_dict: dictionary of inputs
        comp: composition to remove
    '''
    found = False
    # find the right option of isosurface
    try:
        isosurface_option = i_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"]
        isosurface_option_type = 1
        found = True
    except KeyError:
        try:
            isosurface_option = i_dict["Mesh refinement"]["IsosurfacesTwoD1"]["Isosurfaces"]
            isosurface_option_type = 2
            found = True
        except KeyError:
            pass
    # change the options of isosurface
    if found:
        # check that the target composition is in the string
        # then remove it
        if re.match('.*' + comp, isosurface_option):
            options = isosurface_option.split(";")
            for _option in options:
                if re.match('.*' + comp, _option):
                    options.remove(_option)
                new_isosurface_option = ""
                is_first = True
                for _option in options:
                    if is_first:
                        is_first = False
                    else:
                        new_isosurface_option += ";"
                    new_isosurface_option += _option
                if isosurface_option_type == 1:
                    i_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] = new_isosurface_option
                elif isosurface_option_type == 2:
                    i_dict["Mesh refinement"]["IsosurfacesTwoD1"]["Isosurfaces"] = new_isosurface_option
    o_dict = i_dict
    return o_dict


def remove_composition_array(old_str, id_comp0):
    '''
    expand the composition options
    Inputs:
        old_str: the old option
        id_comp0: index of the option for the original composition
        n_comp: number of new compositions
    '''
    options = old_str.split(',')
    removed_option = options.pop(id_comp0)
    # compile new option
    new_option = ""
    is_first = True
    for _option in options:
        if is_first:
            is_first = False
        else:
            new_option += ', '
        new_option += _option
    return new_option 


def remove_composition_composition_field(i_dict, comp0):
    '''
    remove the isosurface options for one composition in the prm file
    Inputs:
        i_dict: dictionary of inputs
        comp: composition to remove
    '''
    o_dict = deepcopy(i_dict)
    # composition field
    # find the index of the original field to be id0
    str_name = o_dict["Compositional fields"]["Names of fields"]
    str_name_options = str_name.split(',')
    id = 0
    found = False
    for _option in str_name_options:
        comp = re_neat_word(_option) 
        if comp == comp0:
            id0 = id
            found = True
            str_name_options.remove(comp)
        id += 1
    if not found:
        raise ValueError("the option of \"Names of fields\" doesn't have option of %s" % comp0)
    # construct the new string for Names of fields
    str_name_new = ""
    is_first = True
    for _option in str_name_options:
        if is_first:
            is_first = False
        else:
            str_name_new += ","
        str_name_new += _option
    o_dict["Compositional fields"]["Names of fields"] = str_name_new
    # number of composition
    nof = int(o_dict["Compositional fields"]["Number of fields"])
    new_nof = nof - 1
    o_dict["Compositional fields"]["Number of fields"] = str(new_nof)
    # field method
    str_f_methods = o_dict["Compositional fields"]["Compositional field methods"]
    f_method = str_f_methods.split(',')[0]
    # construct the new Compositional field methods
    comp_method_expression = ""
    is_first = True
    for i in range(new_nof):
        if is_first:
            is_first = False
        else:
            comp_method_expression += ", "
        comp_method_expression += f_method
    o_dict["Compositional fields"]["Compositional field methods"] = comp_method_expression

    # discretization option
    find_discretization_option = False
    try:
        discretization_option = o_dict["Discretization"]["Stabilization parameters"]['Global composition maximum']
        find_discretization_option = True
    except:
        pass
    if find_discretization_option:
        o_dict["Discretization"]["Stabilization parameters"]['Global composition maximum'] = remove_composition_array(discretization_option, id0)
    find_discretization_option = False
    try:
        discretization_option = o_dict["Discretization"]["Stabilization parameters"]['Global composition minimum']
        find_discretization_option = True
    except:
        pass
    if find_discretization_option:
        o_dict["Discretization"]["Stabilization parameters"]['Global composition minimum'] = remove_composition_array(discretization_option, id0)
    # option in the adiabatic temperature
    try:
        _ = o_dict["Initial temperature model"]["Adiabatic"]["Function"]["Function expression"]
        new_adiabatic_functiion_expression = ""
        is_first = True
        for i in range(new_nof):
            if is_first:
                is_first = False
            else:
                new_adiabatic_functiion_expression += "; "
            new_adiabatic_functiion_expression += "0.0"
        o_dict["Initial temperature model"]["Adiabatic"]["Function"]["Function expression"] = new_adiabatic_functiion_expression
    except KeyError:
        pass
    # option in the look up table
    try:
        look_up_index_option = o_dict["Material model"]["Visco Plastic TwoD"]["Lookup table"]["Material lookup indexes"]
        # note there is an entry for the background in this option, so I have to use id0 + 1 
        o_dict["Material model"]["Visco Plastic TwoD"]["Lookup table"]["Material lookup indexes"] = remove_composition_array(look_up_index_option, id0+1)
    except KeyError:
        pass
    return o_dict
    

def wb_configure_plates(wb_dict, sp_age_trench, sp_rate, ov_age, wb_new_ridge, version, n_crust_layer, Duc, plate_age_method, **kwargs):
    '''
    configure plate in world builder
    '''
    Ro = kwargs.get('Ro', 6371e3)
    Xmax = kwargs.get('Xmax', 7e6)
    max_sph = kwargs.get("max_sph", 180.0)
    geometry = kwargs.get('geometry', 'chunk')
    Dsz = kwargs.get("sz_thickness", None)
    n_comp = kwargs.get("n_comp", 4)
    sp_trailing_length = kwargs.get("sp_trailing_length", 0.0)
    ov_trailing_length = kwargs.get("ov_trailing_length", 0.0)
    box_width = kwargs.get("box_width", None)
    rm_ov_comp = kwargs.get("rm_ov_comp", 0)
    D2C_ratio = 35.2e3 / 7.5e3 # ratio of depleted / crust layer
    o_dict = wb_dict.copy()
    max_cart = 2 * Xmax
    side_angle = 5.0  # side angle to creat features in the 3rd dimension
    side_dist = 1e3
    if geometry == 'chunk':
        _side = side_angle
        _max = max_sph
    elif geometry == 'box':
        _side = side_dist
        _max = max_cart
    if plate_age_method == 'adjust box width only assigning age': 
        trench = get_trench_position_with_age(sp_age_trench, sp_rate, geometry, Ro)
    else:
        trench = get_trench_position(sp_age_trench, sp_rate, geometry, Ro, sp_trailing_length)
    if wb_new_ridge == 1:
        sp_ridge_coords = [[[0, -_side], [0, _side]]]
    else:
        sp_ridge_coords = [[0, -_side], [0, _side]]
    # the index of layers
    if n_crust_layer == 1:
        i_uc = -1
        i_lc = 0
        i_hz = 1
    elif n_crust_layer == 2:
        i_uc = 0
        i_lc = 1
        i_hz = 2
    else:
        raise NotImplementedError()
    # Overiding plate
    if_ov_trans = kwargs.get('if_ov_trans', False)  # transit to another age
    if if_ov_trans and ov_age > (1e6 + kwargs['ov_trans_age']):  # only transfer to younger age
        i0 = FindWBFeatures(o_dict, 'Overiding plate 1')
        ov_trans_dict, ov =\
            wb_configure_transit_ov_plates(wb_dict['features'][i0], trench,\
                ov_age, kwargs['ov_trans_age'], kwargs['ov_trans_length'], wb_new_ridge,\
                Dsz, D2C_ratio,\
                Ro=Ro, geometry=geometry)
        # options with multiple crustal layers
        sample_composiiton_model =  ov_trans_dict["composition models"][0].copy()
        if n_crust_layer == 1:
            pass
        elif n_crust_layer == 2:
            ov_trans_dict["composition models"].append(sample_composiiton_model)
            ov_trans_dict["composition models"][i_uc]["min depth"] = 0.0
            ov_trans_dict["composition models"][i_uc]["max depth"] = Duc
            ov_trans_dict["composition models"][i_uc]["compositions"] = [n_comp - 2]
            ov_trans_dict["composition models"][i_lc]["min depth"] = Duc
            ov_trans_dict["composition models"][i_lc]["compositions"] = [n_comp - 1]
            ov_trans_dict["composition models"][i_hz]["compositions"] = [0]
        else:
            raise NotImplementedError()
        ov_trans_dict["composition models"][i_lc]["max depth"] = Dsz
        ov_trans_dict["composition models"][i_hz]["min depth"] = Dsz
        ov_trans_dict["composition models"][i_hz]["max depth"] = Dsz * D2C_ratio
        if rm_ov_comp:
            # options for removing overiding plate compositions
            ov_trans_dict.pop("composition models")
        o_dict['features'][i0] = ov_trans_dict
    else:
        # if no using transit plate, remove the feature
        try:
            i0 = FindWBFeatures(o_dict, 'Overiding plate 1')
        except KeyError:
            pass
        o_dict = RemoveWBFeatures(o_dict, i0)
        ov = trench
    # options for the overiding plate
    i0 = FindWBFeatures(o_dict, 'Overiding plate')
    op_dict = o_dict['features'][i0]
    if ov_trailing_length > 0.0:
        assert(box_width is not None)
        # migrate the trailing edge from the box side
        if geometry == "box":
            end_point =  box_width - ov_trailing_length
        elif geometry == "chunk":
            end_point =  (box_width - ov_trailing_length) / Ro * 180.0 / np.pi
        op_dict["coordinates"] = [[ov, -_side], [ov, _side],\
            [end_point, _side], [end_point, -_side]] # trench position
    else:
        op_dict["coordinates"] = [[ov, -_side], [ov, _side],\
            [_max, _side], [_max, -_side]] # trench position
    op_dict["temperature models"][0]["plate age"] = ov_age  # age of overiding plate
    # options for multiple crustal layers
    sample_composiiton_model =  op_dict["composition models"][0].copy()
    if n_crust_layer == 1:
        pass
    elif n_crust_layer == 2:
        # options with multiple crustal layer
        op_dict["composition models"].append(sample_composiiton_model)
        op_dict["composition models"][i_uc]["min depth"] = 0.0
        op_dict["composition models"][i_uc]["max depth"] = Duc
        op_dict["composition models"][i_uc]["compositions"] = [n_comp - 2]
        op_dict["composition models"][i_lc]["min depth"] = Duc
        op_dict["composition models"][i_lc]["compositions"] = [n_comp - 1]
        op_dict["composition models"][i_hz]["compositions"] = [1]
    else:
        raise NotImplementedError()
    op_dict["composition models"][i_lc]["max depth"] = Dsz
    op_dict["composition models"][i_hz]["min depth"] = Dsz
    op_dict["composition models"][i_hz]["max depth"] = Dsz * D2C_ratio
    if rm_ov_comp:
        # options for removing overiding plate compositions
        op_dict.pop("composition models")
    o_dict['features'][i0] = op_dict
    # Subducting plate
    # 1. change to the starting point sp_trailing_length and fix the geometry
    i0 = FindWBFeatures(o_dict, 'Subducting plate')
    sp_dict = o_dict['features'][i0]
    if geometry == "box":
        start_point = sp_trailing_length
    elif geometry == "chunk":
        start_point = sp_trailing_length / Ro * 180.0 / np.pi
    sp_dict["coordinates"] = [[start_point, -_side], [start_point, _side],\
        [trench, _side], [trench, -_side]] # trench position
    sp_dict["temperature models"][0]["spreading velocity"] = sp_rate
    sp_dict["temperature models"][0]["ridge coordinates"] = sp_ridge_coords
    # options for multiple crustal layers
    sample_composiiton_model =  sp_dict["composition models"][0].copy()
    if n_crust_layer == 1:
        pass
    elif n_crust_layer == 2:
        # options with multiple crustal layer
        sp_dict["composition models"].append(sample_composiiton_model)
        sp_dict["composition models"][i_uc]["min depth"] = 0.0
        sp_dict["composition models"][i_uc]["max depth"] = Duc
        sp_dict["composition models"][i_uc]["compositions"] = [n_comp - 4]
        sp_dict["composition models"][i_lc]["min depth"] = Duc
        sp_dict["composition models"][i_lc]["compositions"] = [n_comp - 3]
        sp_dict["composition models"][i_hz]["compositions"] = [0]
    else:
        raise NotImplementedError()
    sp_dict["composition models"][i_lc]["max depth"] = Dsz
    sp_dict["composition models"][i_hz]["min depth"] = Dsz
    sp_dict["composition models"][i_hz]["max depth"] = Dsz * D2C_ratio
    o_dict['features'][i0] = sp_dict

    # Slab
    i0 = FindWBFeatures(o_dict, 'Slab')
    s_dict = o_dict['features'][i0]
    s_dict["coordinates"] = [[trench, -_side], [trench, _side]] 
    s_dict["dip point"] = [_max, 0.0]
    s_dict["temperature models"][0]["ridge coordinates"] = sp_ridge_coords
    # todo_version
    # temperature model
    if version >= 3.0:
        if "plate velocity" in s_dict["temperature models"][0]:
            s_dict["temperature models"][0].pop("plate velocity")
        if "shallow dip" in s_dict["temperature models"][0]:
            s_dict["temperature models"][0].pop("shallow dip")
        s_dict["temperature models"][0]["spreading velocity"] = sp_rate
        s_dict["temperature models"][0]["subducting velocity"] = sp_rate
    else:
        s_dict["temperature models"][0]["plate velocity"] = sp_rate
    if version < 1.0:
        if sp_age_trench > 100e6:
            # in this case, I'll use the plate model
            s_dict["temperature models"][0]["use plate model as reference"] = True
            s_dict["temperature models"][0]["max distance slab top"] = 150e3
            s_dict["temperature models"][0]["artificial heat factor"] = 0.5
    else:
            s_dict["temperature models"][0]["reference model name"] = "plate model"
            s_dict["temperature models"][0]["max distance slab top"] = 150e3
    for i in range(len(s_dict["segments"])-1):
        
        # thickness of crust, last segment is a ghost, so skip
        sample_composiiton_model =  s_dict["segments"][i]["composition models"][0].copy()
        if n_crust_layer == 1:
            pass
        elif n_crust_layer == 2:
            # options with multiple crustal layer
            s_dict["segments"][i]["composition models"].append(sample_composiiton_model)
            s_dict["segments"][i]["composition models"][i_uc]["min distance slab top"] = 0.0
            s_dict["segments"][i]["composition models"][i_uc]["max distance slab top"] = Duc
            s_dict["segments"][i]["composition models"][i_uc]["compositions"] = [n_comp - 4]
            s_dict["segments"][i]["composition models"][i_lc]["min distance slab top"] = Duc
            s_dict["segments"][i]["composition models"][i_lc]["compositions"] = [n_comp - 3]
            s_dict["segments"][i]["composition models"][i_hz]["compositions"] = [0]
        else:
            raise NotImplementedError()

        s_dict["segments"][i]["composition models"][i_lc]["max distance slab top"] = Dsz
        s_dict["segments"][i]["composition models"][i_hz]["min distance slab top"] = Dsz
        s_dict["segments"][i]["composition models"][i_hz]["max distance slab top"] = Dsz * D2C_ratio
        pass
    o_dict['features'][i0] = s_dict
    # mantle for substracting adiabat
    i0 = FindWBFeatures(o_dict, 'mantle to substract')
    m_dict = o_dict['features'][i0]
    m_dict["coordinates"] =[[0.0, -_side], [0.0, _side],\
        [_max, _side], [_max, -_side]]
    o_dict['features'][i0] = m_dict
    return o_dict

def wb_configure_transit_ov_plates(i_feature, trench, ov_age,\
    ov_trans_age, ov_trans_length, wb_new_ridge, Dsz, D2C_ratio, **kwargs):
    '''
    Transit overiding plate to a younger age at the trench
    See descriptions of the interface to_configure_wb
    '''
    geometry = kwargs.get('geometry', 'chunk')
    side_angle = 5.0  # side angle to creat features in the 3rd dimension
    side_dist = 1e3
    Ro = kwargs.get("Ro", 6371e3)
    o_feature = i_feature.copy()
    trans_angle = ov_trans_length / Ro / np.pi * 180.0
    if geometry == 'chunk':
        ov = trench  + trans_angle  # new ending point of the default overiding plage
        side = side_angle
        ridge = trench - trans_angle * ov_trans_age / (ov_age - ov_trans_age)
    elif geometry == 'box':
        ov = trench + ov_trans_length
        side = side_dist
        ridge = trench - ov_trans_length * ov_trans_age / (ov_age - ov_trans_age)
    else:
        pass
    v = ov_trans_length / (ov_age - ov_trans_age)
    o_feature["temperature models"][0]["spreading velocity"] = v
    o_feature["coordinates"] = [[trench, -side], [trench, side],\
        [ov, side], [ov, -side]]
    if wb_new_ridge == 1:
        o_feature["temperature models"][0]["ridge coordinates"] =\
            [[[ridge, -side], [ridge, side]]]
    else:
        o_feature["temperature models"][0]["ridge coordinates"] =\
            [[ridge, -side], [ridge, side]]
    o_feature["composition models"][0]["max depth"] = Dsz
    o_feature["composition models"][1]["min depth"] = Dsz
    o_feature["composition models"][1]["max depth"] = Dsz * D2C_ratio
    return o_feature, ov


def prm_geometry_sph(max_phi, **kwargs):
    '''
    reset geometry for chunk geometry
    '''
    adjust_mesh_with_width = kwargs.get("adjust_mesh_with_width")
    inner_radius = 3.481e6
    outer_radius = 6.371e6
    if adjust_mesh_with_width:
        longitude_repetitions = int(outer_radius * max_phi / 180.0 * np.pi / (outer_radius - inner_radius))
    else:
        longitude_repetitions = 2
    o_dict = {
        "Model name": "chunk",
        "Chunk": {
            "Chunk inner radius": "3.481e6",\
            "Chunk outer radius": "6.371e6",\
            "Chunk maximum longitude": "%.4e" % max_phi,\
            "Chunk minimum longitude": "0.0",\
            "Longitude repetitions": "%d" % longitude_repetitions
        }
    }
    return o_dict


def prm_minimum_refinement_sph(**kwargs):
    """
    minimum refinement function for spherical geometry
    """
    Ro = kwargs.get('Ro', 6371e3)
    refine_wedge = kwargs.get('refine_wedge', False)
    trench = kwargs.get('trench', 0.62784)
    refinement_level = kwargs.get("refinement_level", 10)
    if refinement_level == 9:
        R_UM = 6
        R_LS = 7
        pass
    elif refinement_level == 10:
        R_UM = 6
        R_LS = 8
    elif refinement_level == 11:
        R_UM = 7
        R_LS = 9
    elif refinement_level == 12:
        R_UM = 7
        R_LS = 9
    elif refinement_level == 13:
        R_UM = 7
        R_LS = 10
    else:
        raise ValueError("Wrong value %d for the \"refinement_level\"" % refinement_level)
    o_dict = {
      "Coordinate system": "spherical",
      "Variable names": "r,phi,t"
    }
    if refine_wedge:
        o_dict["Function constants"] = "Ro=%.4e, UM=670e3, DD=200e3, phiT=%.4f, dphi=%.4f" % (Ro, trench*np.pi/180.0, 10.0*np.pi/180.0)
        o_dict["Function expression"] =  "((Ro-r<UM)? \\\n                                   ((Ro-r<DD)? (((phi-phiT>-dphi)&&(phi-phiT<dphi))? %d: %d): %d): 0.0)"\
            % (refinement_level, R_LS, R_UM)
    else:
        o_dict["Function constants"] = "Ro=%.4e, UM=670e3, DD=100e3" % Ro
        o_dict["Function expression"] =  "((Ro-r<UM)? \\\n                                   ((Ro-r<DD)? %d: %d): 0.0)" % (R_LS, R_UM)
    print(o_dict) # debug
    return o_dict


def prm_minimum_refinement_cart(**kwargs):
    """
    minimum refinement function for cartesian geometry
    """
    Do = kwargs.get('Do', 2890e3)
    refinement_level = kwargs.get("refinement_level", 10)
    if refinement_level == 9:
        R_UM = 6
        R_LS = 7
        pass
    elif refinement_level == 10:
        R_UM = 6
        R_LS = 8
    elif refinement_level == 11:
        R_UM = 7
        R_LS = 9
    else:
        raise ValueError("Wrong value for the \"refinement_level\"")
    o_dict = {
      "Coordinate system": "cartesian",
      "Variable names": "x, y, t",
      "Function constants": "Do=%.4e, UM=670e3, DD=100e3" % Do,
      "Function expression": "((Do-y<UM)? \\\n                                   ((Do-y<DD)? %d: %d): 0.0)" % (R_LS, R_UM)
    }
    return o_dict


def prm_boundary_temperature_sph():
    '''
    boundary temperature model in spherical geometry
    '''
    o_dict = {
        "Fixed temperature boundary indicators": "bottom, top",
        "List of model names": "spherical constant",
        "Spherical constant": {
            "Inner temperature": "3500", 
            "Outer temperature": "273"
        }
    }
    return o_dict


def prm_boundary_temperature_cart():
    '''
    boundary temperature model in cartesian geometry
    '''
    o_dict = {
        "Fixed temperature boundary indicators": "bottom, top",
        "List of model names": "box",
        "Box": {
            "Bottom temperature": "3500",
            "Top temperature": "273"
            }
    }
    return o_dict


def prm_visco_plastic_TwoD_sph(visco_plastic_twoD, max_phi, type_of_bd, sp_trailing_length, ov_trailing_length, **kwargs):
    '''
    reset subsection Visco Plastic TwoD
    Inputs:
        visco_plastic_twoD (dict): inputs for the "subsection Visco Plastic TwoD"
        part in a prm file
        kwargs(dict):
    '''
    o_dict = visco_plastic_twoD.copy()
    o_dict['Reset viscosity function'] =\
        prm_reset_viscosity_function_sph(max_phi, sp_trailing_length, ov_trailing_length)
    if type_of_bd in ["all free slip"]:
        o_dict["Reaction mor"] = 'true'
    else:
        o_dict["Reaction mor"] = 'false'
    o_dict["Reaction mor function"] =\
        prm_reaction_mor_function_sph(max_phi, sp_trailing_length, ov_trailing_length)
    if type_of_bd == "all free slip":
        # use free slip on both sides, set ridges on both sides
        o_dict['Reset viscosity'] = 'true'
    else:
        o_dict['Reset viscosity'] = 'false'
    return o_dict


def prm_visco_plastic_TwoD_cart(visco_plastic_twoD, box_width, box_height, type_of_bd, sp_trailing_length,\
                                ov_trailing_length, Dsz, **kwargs):
    '''
    reset subsection Visco Plastic TwoD
    Inputs:
        visco_plastic_twoD (dict): inputs for the "subsection Visco Plastic TwoD"
        part in a prm file
        kwargs(dict):
    '''
    reset_density = kwargs.get("reset_density", False)
    o_dict = visco_plastic_twoD.copy()
    o_dict['Reset viscosity function'] =\
        prm_reset_viscosity_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length)
    if type_of_bd in ["all free slip"]:
        o_dict["Reaction mor"] = 'true'
    else:
        o_dict["Reaction mor"] = 'false'
    print("Dsz: ", Dsz)  # debug
    o_dict["Reaction mor function"] =\
        prm_reaction_mor_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length, Dsz)
    if type_of_bd in ["all free slip"]:
        # use free slip on both sides, set ridges on both sides
        o_dict['Reset viscosity'] = 'true'
    else:
        o_dict['Reset viscosity'] = 'false'
    # reset density
    # if reset_density is 1, then this option will show up in the prm file
    if reset_density:
        o_dict['Reset density'] = 'true'
        o_dict['Reset density function'] = \
            prm_reaction_density_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length)

    return o_dict


def prm_reset_viscosity_function_sph(max_phi, sp_trailing_length, ov_trailing_length):
    '''
    Default setting for Reset viscosity function in spherical geometry
    Inputs:
        sp_trailing_length: trailing length of the subducting plate
        ov_trailing_length: trailing length of the overiding plate
    '''
    max_phi_in_rad = max_phi * np.pi / 180.0
    if sp_trailing_length < 1e-6 and ov_trailing_length < 1e-6:
        function_constants_str = "Depth=1.45e5, Width=2.75e5, Ro=6.371e6, PHIM=%.4e, CV=1e20" % max_phi_in_rad
        function_expression_str =  "(((r > Ro - Depth) && ((Ro*phi < Width) || (Ro*(PHIM-phi) < Width)))? CV: -1.0)"
    elif sp_trailing_length > 1e-6 and ov_trailing_length > 1e-6:
        function_constants_str = "Depth=1.45e5, SPTL=%.4e, OPTL=%.4e, Ro=6.371e6, PHIM=%.4e, CV=1e20" % \
        (sp_trailing_length, ov_trailing_length, max_phi_in_rad)
        function_expression_str =  "(((r > Ro - Depth) && ((Ro*phi < SPTL) || (Ro*(PHIM-phi) < OPTL)))? CV: -1.0)"
    else:
        return NotImplementedError()
    odict = {
        "Coordinate system": "spherical",
        "Variable names": "r, phi",
        "Function constants": function_constants_str,
        "Function expression": function_expression_str
      }
    return odict


def prm_reset_viscosity_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length):
    '''
    Default setting for Reset viscosity function in cartesian geometry
    Inputs:
        sp_trailing_length: trailing length of the subducting plate
        ov_trailing_length: trailing length of the overiding plate
    '''
    Do_str = "2.890e6"
    if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
        Do_str = "%.4e" % box_height
    if sp_trailing_length < 1e-6 and ov_trailing_length < 1e-6:
        function_constants_str =  "Depth=1.45e5, Width=2.75e5, Do=%s, xm=%.4e, CV=1e20" % (Do_str, box_width)
        function_expression_str =  "(((y > Do - Depth) && ((x < Width) || (xm-x < Width)))? CV: -1.0)"
    elif sp_trailing_length > 1e-6 and ov_trailing_length > 1e-6:
        function_constants_str = "Depth=1.45e5, SPTL=%.4e, OPTL=%.4e, Do=%s, xm=%.4e, CV=1e20" % \
        (sp_trailing_length, ov_trailing_length, Do_str, box_width)
        function_expression_str =  "(((y > Do - Depth) && ((x < SPTL) || (xm-x < OPTL)))? CV: -1.0)"
    else:
        raise ValueError("Must set the trailing edges of the subducting plate and the overiding palte as the same time.")
    odict = {
        "Coordinate system": "cartesian",
        "Variable names": "x, y",
        "Function constants": function_constants_str,
        "Function expression": function_expression_str
    }
    return odict


def prm_reaction_mor_function_sph(max_phi, sp_trailing_length, ov_trailing_length):
    '''
    Default setting for Reaction mor function in cartesian geometry
    '''
    max_phi_in_rad = max_phi * np.pi / 180.0
    if sp_trailing_length < 1e-6 and ov_trailing_length < 1e-6:
        function_constants_str = "Width=2.75e5, Ro=6.371e6, PHIM=%.4e, DCS=7.500e+03, DHS=3.520e+04" % max_phi_in_rad
        function_expression_str =  "((r > Ro - DCS) && (Ro*phi < Width)) ? 0:\
\\\n                                        ((r < Ro - DCS) && (r > Ro - DHS) && (Ro*phi < Width)) ? 1:\
\\\n                                        ((r > Ro - DCS) && (Ro*(PHIM - phi) < Width)) ? 2:\
\\\n                                        ((r < Ro - DCS) && (r > Ro - DHS) && (Ro*(PHIM - phi) < Width)) ? 3: -1"
    elif sp_trailing_length > 1e-6 and ov_trailing_length > 1e-6:
        function_constants_str = "Width=2.75e5, SPTL=%.4e, OPTL=%.4e, Ro=6.371e6, PHIM=%.4e, DCS=7.500e+03, DHS=3.520e+04" % \
        (sp_trailing_length, ov_trailing_length, max_phi_in_rad)
        function_expression_str =  "((r > Ro - DCS) && (Ro*phi > SPTL) && (Ro*phi < Width + SPTL)) ? 0:\
\\\n                                        ((r < Ro - DCS) && (r > Ro - DHS) && (Ro*phi > SPTL) && (Ro*phi < Width + SPTL)) ? 1:\
\\\n                                        ((r > Ro - DCS) && (Ro*(PHIM - phi) > OPTL) && (Ro*(PHIM - phi) < Width + OPTL)) ? 2:\
\\\n                                        ((r < Ro - DCS) && (r > Ro - DHS) && (Ro*(PHIM - phi) > OPTL) && (Ro*(PHIM - phi) < Width + OPTL)) ? 3: -1"
    else:
        return NotImplementedError()
    odict = {
        "Coordinate system": "spherical",\
        "Variable names": "r, phi",\
        # "Function constants": function_constants_str,\
        # debug
        "Function constants": function_constants_str,\
        "Function expression":  function_expression_str
      }
    return odict


def prm_reaction_mor_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length, Dsz):
    '''
    Default setting for Reaction mor function in cartesian geometry
    '''
    Do_str = "2.890e6"
    if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
        Do_str = "%.4e" % box_height
    if sp_trailing_length < 1e-6 and ov_trailing_length < 1e-6:
        function_constants_str = "Width=2.75e5, Do=%s, xm=%.4e, DCS=%.4e, DHS=%.4e" % (Do_str, box_width, Dsz, Dsz*35.2/7.5)
        function_expression_str = "((y > Do - DCS) && (x < Width)) ? 0:\
\\\n                                        ((y < Do - DCS) && (y > Do - DHS) && (x < Width)) ? 1:\
\\\n                                        ((y > Do - DCS) && (xm - x < Width)) ? 2:\
\\\n                                        ((y < Do - DCS) && (y > Do - DHS) && (xm - x < Width)) ? 3: -1" 
    elif sp_trailing_length > 1e-6 and ov_trailing_length > 1e-6:
        function_constants_str = "Width=2.75e5, SPTL=%.4e, OPTL=%.4e, Do=%s, xm=%.4e, DCS=%.4e, DHS=%.4e" % \
        (sp_trailing_length, ov_trailing_length, Do_str, box_width, Dsz, Dsz*35.2/7.5)
        function_expression_str = "((y > Do - DCS) && (x > SPTL) && (x < Width + SPTL)) ? 0:\
\\\n                                        ((y < Do - DCS) && (y > Do - DHS) && (x > SPTL) && (x < Width + SPTL)) ? 1:\
\\\n                                        ((y > Do - DCS) && (xm - x > OPTL) && (xm - x < Width + OPTL)) ? 2:\
\\\n                                        ((y < Do - DCS) && (y > Do - DHS) && (xm - x > OPTL) && (xm - x < Width + OPTL)) ? 3: -1"
    else:
        return NotImplementedError()
    odict = {
        "Coordinate system": "cartesian",\
        "Variable names": "x, y",\
        "Function constants": function_constants_str,\
        "Function expression": function_expression_str
    }
    return odict


def prm_reaction_density_function_cart(box_width, box_height, sp_trailing_length, ov_trailing_length):
    '''
    Default setting for Reaction mor function in cartesian geometry
    '''
    Do_str = "2.890e6"
    if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
        Do_str = "%.4e" % box_height
    function_constants_str = "Depth=1.45e5, SPTL=%.4e, OPTL=%.4e, Do=%s, xm=%.4e, CD=3300.0" % (sp_trailing_length, ov_trailing_length, Do_str, box_width)
    function_expression_str = "(((y > Do - Depth) && ((x < SPTL) || (xm-x < OPTL)))? CD: -1.0)"
    odict = {
        "Coordinate system": "cartesian",\
        "Variable names": "x, y",\
        "Function constants": function_constants_str,\
        "Function expression": function_expression_str
    }
    return odict

def prm_prescribed_temperature_sph(max_phi, potential_T, sp_rate, ov_age, **kwargs):
    '''
    Default setting for Prescribed temperatures in spherical geometry
    Inputs:
        kwargs:
            model_name: name of model to use
            area_width: width of resetting
    '''
    model_name = kwargs.get('model_name', 'default')
    max_phi_in_rad = max_phi * np.pi / 180.0
    area_width = kwargs.get('area_width', 2.75e5) 
    if model_name == 'default':
        odict = {
            "Indicator function": {
              "Coordinate system": "spherical",\
              "Variable names": "r, phi",\
              "Function constants": "Depth=1.45e5, Width=2.75e5, Ro=6.371e6, PHIM=%.4e" % max_phi_in_rad,\
              "Function expression": "(((r>Ro-Depth)&&((r*phi<Width)||(r*(PHIM-phi)<Width))) ? 1:0)"\
            },\
            "Temperature function": {
              "Coordinate system": "spherical",\
              "Variable names": "r, phi",\
              "Function constants": "Depth=1.45e5, Width=2.75e5, Ro=6.371e6, PHIM=%.4e,\\\n                             AGEOP=%.4e, TS=2.730e+02, TM=%.4e, K=1.000e-06, VSUB=%.4e, PHILIM=1e-6" %\
                  (max_phi_in_rad, ov_age * year, potential_T, sp_rate / year),\
              "Function expression": "((r*(PHIM-phi)<Width) ? TS+(TM-TS)*(1-erfc(abs(Ro-r)/(2*sqrt(K*AGEOP)))):\\\n\t(phi > PHILIM)? (TS+(TM-TS)*(1-erfc(abs(Ro-r)/(2*sqrt((K*Ro*phi)/VSUB))))): TM)"
            }
        }
    elif model_name == 'plate model 1':
        odict = {
            "Model name": "plate model 1",
            "Indicator function": {
              "Coordinate system": "spherical",
              "Variable names": "r, phi",
              "Function constants": "Depth=1.45e5, Width=%.4e, Ro=6.371e6, PHIM=%.4e" % (area_width, max_phi_in_rad),
              "Function expression": "(((r>Ro-Depth)&&((Ro*phi<Width)||(Ro*(PHIM-phi)<Width))) ? 1:0)"
            },
            "Plate model 1": {
                "Area width": "%.4e" % area_width,
                "Subducting plate velocity": "%.4e" % (sp_rate / year),
                "Overiding plate age": "%.4e" % (ov_age * year),
                "Overiding area width": "%.4e" % area_width,
                "Top temperature": "273.0"
            }
        }
        pass
    else:
        raise ValueError('model name is either \"default\" or \"plate model 1\"')
    return odict


def prm_prescribed_temperature_cart(box_width, potential_T, sp_rate, ov_age):
    '''
    Default setting for Prescribed temperatures in cartesian geometry
    '''
    odict = {
        "Indicator function": {
          "Coordinate system": "cartesian",
          "Variable names": "x, y",
          "Function constants": "Depth=1.45e5, Width=2.75e5, Do=2.890e6, xm=%.4e" % box_width,
          "Function expression": "(((y>Do-Depth)&&((x<Width)||(xm-x<Width))) ? 1:0)"
        },
        "Temperature function": {
          "Coordinate system": "cartesian",
          "Variable names": "x, y",
          "Function constants": "Depth=1.45e5, Width=2.75e5, Do=2.890e6, xm=%.4e,\\\n                             AGEOP=%.4e, TS=2.730e+02, TM=%.4e, K=1.000e-06, VSUB=%.4e, XLIM=6" %\
                (box_width, ov_age * year, potential_T, sp_rate / year),\
          "Function expression": "(xm-x<Width) ? TS+(TM-TS)*(1-erfc(abs(Do-y)/(2*sqrt(K*AGEOP)))):\\\n\t((x > XLIM)? (TS+(TM-TS)*(1-erfc(abs(Do-y)/(2*sqrt((K*x)/VSUB))))): TM)"
        }
    }
    return odict


def prm_prescribed_temperature_cart_plate_model(box_width, potential_T, sp_rate, ov_age):
    '''
    Default setting for Prescribed temperatures in cartesian geometry using the plate model
    '''
    odict = {
        "Model name": "plate model",
        "Indicator function": {
          "Coordinate system": "cartesian",
          "Variable names": "x, y",
          "Function constants": "Depth=1.45e5, Width=2.75e5, Do=2.890e6, xm=%.4e" % box_width,
          "Function expression": "(((y>Do-Depth)&&((x<Width)||(xm-x<Width))) ? 1:0)"
        },
        "Plate model":{
            "Subducting plate velocity" : "%.4e" % (sp_rate/year)
        }
    }
    return odict


def prm_prescribed_temperature_cart_plate_model_1(box_width, box_height, potential_T, sp_rate, ov_age, **kwargs):
    '''
    Default setting for Prescribed temperatures in cartesian geometry using the plate model
    Inputs:
        area_width: width of the reseting area
    '''
    area_width = kwargs.get('area_width', 2.75e5)
    Do_str = "2.890e6"
    if abs(box_height - 2.89e6) / 2.89e6 > 1e-6:
        Do_str = "%.4e" % box_height
    odict = {
        "Model name": "plate model 1",
        "Indicator function": {
          "Coordinate system": "cartesian",
          "Variable names": "x, y",
          "Function constants": "Depth=1.45e5, Width=%.4e, Do=%s, xm=%.4e" % (area_width, Do_str, box_width),
          "Function expression": "(((y>Do-Depth)&&((x<Width)||(xm-x<Width))) ? 1:0)"
        },
        "Plate model 1": {
            "Area width": "%.4e" % area_width,
            "Subducting plate velocity": "%.4e" % (sp_rate / year),
            "Overiding plate age": "%.4e" % (ov_age * year),
            "Overiding area width": "%.4e" % area_width,
            "Top temperature": "273.0"
        }
    }
    return odict


###
# velocity boundary conditions
###

def prm_prescribed_velocity_function(trench, delta_trench, sp_rate, ov_rate):
    '''
    the "Function" subsection in the "" subsection
    Inputs:
        trench: position of the trench
        delta_trench: the transition distance where the velocity 
                      varies continously from the subducting plate to the overiding plate
        sp_rate: prescribed rate of the subducting plate
        ov_rate: prescribed rate of the overidding plate
    Returns:
        func_dict: a dictionary storing the settings
    '''
    func_dict = \
    {
        "Function constants": "u0=0.03, x0=10000",\
        "Variable names": "x,y",\
        "Function constants": "xtr=%.4e, dtr=%.4e, usp=%.4e, uov=%.4e, xrd=100e3" % (trench, delta_trench,sp_rate, ov_rate),\
        "Function expression": "((x < xrd)? (x/xrd*usp):\\\n%s\
 ((x < (xtr - dtr/2.0))? usp:\\\n%s\
 ((x < (xtr + dtr/2.0))?(usp + (uov - usp)*(x-xtr+dtr/2.0)/dtr): uov)))\\\n%s; 0.0" %\
        (36*" ", 36*" ", 34*" ")
    }
    return func_dict


def prm_top_prescribed(trench, sp_rate, ov_rate, refinement_level, **kwargs):
    '''
    Inputs:
        trench: position of the trench
        sp_rate: prescribed rate of the subducting plate
        ov_rate: prescribed rate of the overidding plate
        refinement_level: total levele of refinement, for figuring out the number of integration points
        kwargs:
            delta_trench: the transition distance where the velocity 
                          varies continously from the subducting plate to the overiding plate
    '''
    delta_trench = kwargs.get("delta_trench", 20e3)
    prescribed_velocity_function =  prm_prescribed_velocity_function(trench, delta_trench,sp_rate, ov_rate)
    bd_v_dict = {
        "Prescribed velocity boundary indicators": "3:function",\
        "Tangential velocity boundary indicators": "0, 1, 2",\
        "Function": prescribed_velocity_function
    }
    return bd_v_dict
    pass



def prm_top_prescribed_with_bottom_right_open(trench, sp_rate, ov_rate, refinement_level, **kwargs):
    '''
    Inputs:
        trench: position of the trench
        sp_rate: prescribed rate of the subducting plate
        ov_rate: prescribed rate of the overidding plate
        refinement_level: total levele of refinement, for figuring out the number of integration points
        kwargs:
            delta_trench: the transition distance where the velocity 
                          varies continously from the subducting plate to the overiding plate
    '''
    delta_trench = kwargs.get("delta_trench", 20e3)
    prescribed_velocity_function =  prm_prescribed_velocity_function(trench, delta_trench,sp_rate, ov_rate)
    bd_v_dict = {
        "Prescribed velocity boundary indicators": "3:function",\
        "Tangential velocity boundary indicators": "0",\
        "Function": prescribed_velocity_function
    }
    # fix the number of integretion points
    n_integration_points = 2048
    if refinement_level > 0:
        n_integration_points = int(2**(refinement_level+1))
    bd_t_dict = {
        "Prescribed traction boundary indicators": "1:initial lithostatic pressure, 2:initial lithostatic pressure",\
        "Initial lithostatic pressure":{
            "Representative point": "100000.0, 100000.0",\
            "Number of integration points": "%d" % n_integration_points
        }
    }
    return bd_v_dict, bd_t_dict


def prm_top_prescribed_with_bottom_left_open(trench, sp_rate, ov_rate, refinement_level, **kwargs):
    '''
    Inputs:
        trench: position of the trench
        sp_rate: prescribed rate of the subducting plate
        ov_rate: prescribed rate of the overidding plate
        refinement_level: total levele of refinement, for figuring out the number of integration points
    '''
    delta_trench = kwargs.get("delta_trench", 20e3)
    prescribed_velocity_function =  prm_prescribed_velocity_function(trench, delta_trench,sp_rate, ov_rate)
    bd_v_dict = {
        "Prescribed velocity boundary indicators": "3:function",\
        "Tangential velocity boundary indicators": "1",\
        "Function": prescribed_velocity_function
    }
    # fix the number of integretion points
    n_integration_points = 2048
    if refinement_level > 0:
        n_integration_points = int(2**(refinement_level+1))
    bd_t_dict = {
        "Prescribed traction boundary indicators": "0:initial lithostatic pressure, 2:initial lithostatic pressure",\
        "Initial lithostatic pressure":{
            "Representative point": "100000.0, 100000.0",\
            "Number of integration points": "%d" % n_integration_points
        }
    }
    return bd_v_dict, bd_t_dict


def re_write_geometry_while_only_assigning_plate_age(box_width0, sp_age0, sp_age, sp_rate, sp_trailing_length, ov_trailing_length):
    '''
    adjust box width with assigned spreading rate of subducting plate and subducting plate age
    Inputs:
        box_width0: default box width
        sp_age0: default plate age
        sp_age: plate age
        sp_rate: spreading rate of the subducting plate
    '''
    box_width = box_width0 + (sp_age - sp_age0) * sp_rate
    return box_width


def re_write_geometry_while_assigning_plate_age(box_width0, sp_age0, sp_age, sp_rate, sp_trailing_length, ov_trailing_length):
    '''
    adjust box width with assigned spreading rate of subducting plate and subducting plate age
    Inputs:
        box_width0: default box width
        sp_age0: default plate age
        sp_age: plate age
        sp_rate: spreading rate of the subducting plate
        sp_trailing_length: trailing length of the sp plate
    '''
    box_width = box_width0 + (sp_age - sp_age0) * sp_rate + sp_trailing_length + ov_trailing_length
    return box_width


def CDPT_set_parameters(o_dict, CDPT_type, **kwargs):
    '''
    set parameters for the CDPT model
    '''
    slope_410 = kwargs.get("slope_410", 2e6)
    slope_660 = kwargs.get("slope_660", -1e6)
    sz_different_composition = kwargs.get("sz_different_composition", 1)
    print("slope_660: ", slope_660) # debug
    if CDPT_type == 'HeFESTo_consistent':
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition depths'] = \
        'background:410e3|520e3|560e3|660e3|660e3|660e3|660e3, spcrust: 80e3|665e3|720e3, spharz: 410e3|520e3|560e3|660e3|660e3|660e3|660e3'
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition widths'] = \
        'background:13e3|25e3|60e3|5e3|5e3|5e3|5e3, spcrust: 5e3|60e3|5e3, spharz: 13e3|25e3|60e3|5e3|5e3|5e3|5e3'
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition temperatures'] = \
        'background:1780.0|1850.0|1870.0|1910.0|2000.0|2000.0|2000.0, spcrust: 1173.0|1870.0|2000.0, spharz: 1780.0|1850.0|1870.0|1910.0|2000.0|2000.0|2000.0'
        if abs((slope_410 - 2e6) / 2e6) > 1e-6 or abs((slope_660 - (-1e6)) / 1e6) > 1e-6: 
            o_dict['Material model']['Visco Plastic TwoD']['Phase transition Clapeyron slopes'] = \
            'background:%.2e|4.1e6|4e6|%.2e|0|%.2e|2e6, spcrust: 0.0|4e6|2e6, spharz: %.2e|4.1e6|4e6|%.2e|0|%.2e|2e6' % \
                (slope_410, slope_660, slope_660, slope_410, slope_660, slope_660)
        else:
            o_dict['Material model']['Visco Plastic TwoD']['Phase transition Clapeyron slopes'] = \
            'background:2e6|4.1e6|4e6|-1e6|0|-1e6|2e6, spcrust: 0.0|4e6|2e6, spharz: 2e6|4.1e6|4e6|-1e6|0|-1e6|2e6'
    elif CDPT_type == 'Billen2018_old':
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition depths'] = \
        'background:410e3|520e3|560e3|670e3|670e3|670e3|670e3, spcrust: 80e3|665e3|720e3, spharz: 410e3|520e3|560e3|670e3|670e3|670e3|670e3'
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition widths'] = \
        "background:5e3|5e3|5e3|10e3|5e3|5e3|5e3, spcrust: 5e3|5e3|5e3, spharz: 5e3|5e3|5e3|10e3|5e3|5e3|5e3"
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition temperatures'] = \
        "background:1662.0|1662.0|1662.0|1662.0|1662.0|1662.0|1662.0, spcrust: 1173.0|1662.0|1662.0, spharz: 1662.0|1662.0|1662.0|1662.0|1662.0|1662.0|1662.0"
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition Clapeyron slopes'] = \
        "background:4e6|4.1e6|4e6|-2e6|4e6|-3.1e6|1.3e6, spcrust: 0.0|4e6|1.3e6, spharz: 4e6|4.1e6|4e6|-2e6|4e6|-3.1e6|1.3e6"
    elif CDPT_type == 'Billen2018':
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition depths'] = \
        'background:410e3|520e3|560e3|670e3|670e3|670e3|670e3, spcrust: 80e3|665e3|720e3, spharz: 410e3|520e3|560e3|670e3|670e3|670e3|670e3'
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition widths'] = \
        "background:5e3|5e3|5e3|10e3|5e3|5e3|5e3, spcrust: 5e3|5e3|5e3, spharz: 5e3|5e3|5e3|10e3|5e3|5e3|5e3"
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition temperatures'] = \
        "background:1800.0|1800.0|1800.0|1800.0|1800.0|1800.0|1800.0, spcrust: 1173.0|1800.0|1800.0, spharz: 1800.0|1800.0|1800.0|1800.0|1800.0|1800.0|1800.0"
        o_dict['Material model']['Visco Plastic TwoD']['Phase transition Clapeyron slopes'] = \
        "background:4e6|4.1e6|4e6|-2e6|4e6|-3.1e6|1.3e6, spcrust: 0.0|4e6|1.3e6, spharz: 4e6|4.1e6|4e6|-2e6|4e6|-3.1e6|1.3e6"




def CDPT_assign_mantle_rheology(o_dict, rheology, **kwargs):
    '''
    Assign mantle rheology in the CDPT model
    ''' 
    diffusion_creep = rheology['diffusion_creep']
    dislocation_creep = rheology['dislocation_creep']
    diffusion_creep_lm = rheology['diffusion_lm']
    sz_viscous_scheme = kwargs.get("sz_viscous_scheme", "constant")
    sz_constant_viscosity = kwargs.get("sz_constant_viscosity", 1e20)
    sz_minimum_viscosity = kwargs.get("sz_minimum_viscosity", 1e18)
    slab_core_viscosity = kwargs.get("slab_core_viscosity", -1.0)
    minimum_viscosity = kwargs.get('minimum_viscosity', 1e18)
    if sz_viscous_scheme == "constant":
        diff_crust_A = 1.0 / 2.0 / sz_constant_viscosity
        diff_crust_m = 0.0
        diff_crust_E = 0.0
        diff_crust_V = 0.0
        disl_crust_A = 5e-32
        disl_crust_n = 1.0
        disl_crust_E = 0.0
        disl_crust_V = 0.0
    elif sz_viscous_scheme == "stress dependent":
        diff_crust_A = 5e-32
        diff_crust_m = 0.0
        diff_crust_E = 0.0
        diff_crust_V = 0.0
        disl_crust_A = dislocation_creep['A']
        disl_crust_n = dislocation_creep['n']
        disl_crust_E = dislocation_creep['E']
        disl_crust_V = dislocation_creep['V']
    diff_A = diffusion_creep['A']
    diff_m = diffusion_creep['m']
    diff_n = diffusion_creep['n']
    diff_E = diffusion_creep['E']
    diff_V = diffusion_creep['V']
    diff_d = diffusion_creep['d']
    disl_A = dislocation_creep['A']
    disl_m = dislocation_creep['m']
    disl_n = dislocation_creep['n']
    disl_E = dislocation_creep['E']
    disl_V = dislocation_creep['V']
    disl_d = dislocation_creep['d']
    diff_A_lm = diffusion_creep_lm['A']
    diff_m_lm = diffusion_creep_lm['m']
    diff_n_lm = diffusion_creep_lm['n']
    diff_E_lm = diffusion_creep_lm['E']
    diff_V_lm = diffusion_creep_lm['V']
    diff_d_lm = diffusion_creep_lm['d']
    o_dict['Material model']['Visco Plastic TwoD']['Prefactors for diffusion creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
spcrust: %.4e|%.4e|%.4e|%.4e,\
spharz: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
opcrust: %.4e, opharz: %.4e" % (diff_A, diff_A, diff_A, diff_A, diff_A_lm, diff_A_lm,\
diff_A_lm, diff_A_lm, diff_crust_A, diff_A, diff_A_lm, diff_A_lm,\
diff_A, diff_A, diff_A, diff_A, diff_A_lm, diff_A_lm, diff_A_lm, diff_A_lm,\
diff_A, diff_A)
    o_dict['Material model']['Visco Plastic TwoD']['Grain size exponents for diffusion creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
spcrust: %.4e|%.4e|%.4e|%.4e,\
spharz: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
opcrust: %.4e, opharz: %.4e" % (diff_m, diff_m, diff_m, diff_m, diff_m_lm, diff_m_lm,\
diff_m_lm, diff_m_lm, diff_crust_m, diff_m, diff_m_lm, diff_m_lm,\
diff_m, diff_m, diff_m, diff_m, diff_m_lm, diff_m_lm, diff_m_lm, diff_m_lm,\
diff_m, diff_m)
    o_dict['Material model']['Visco Plastic TwoD']['Activation energies for diffusion creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
spcrust: %.4e|%.4e|%.4e|%.4e,\
spharz: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
opcrust: %.4e, opharz: %.4e" % (diff_E, diff_E, diff_E, diff_E, diff_E_lm, diff_E_lm,\
diff_E_lm, diff_E_lm, diff_crust_E, diff_E, diff_E_lm, diff_E_lm,\
diff_E, diff_E, diff_E, diff_E, diff_E_lm, diff_E_lm, diff_E_lm, diff_E_lm,\
diff_E, diff_E)
    o_dict['Material model']['Visco Plastic TwoD']['Activation volumes for diffusion creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
spcrust: %.4e|%.4e|%.4e|%.4e,\
spharz: %.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e|%.4e,\
opcrust: %.4e, opharz: %.4e" % (diff_V, diff_V, diff_V, diff_V, diff_V_lm, diff_V_lm,\
diff_V_lm, diff_V_lm, diff_crust_V, diff_V, diff_V_lm, diff_V_lm,\
diff_V, diff_V, diff_V, diff_V, diff_V_lm, diff_V_lm, diff_V_lm, diff_V_lm,\
diff_V, diff_V)
    o_dict['Material model']['Visco Plastic TwoD']['Prefactors for dislocation creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|5.0000e-32|5.0000e-32|5.0000e-32|5.0000e-32,\
spcrust: %.4e|%.4e|5.0000e-32|5.0000e-32,\
spharz: %.4e|%.4e|%.4e|%.4e|5.0000e-32|5.0000e-32|5.0000e-32|5.0000e-32,\
opcrust: %.4e, opharz: %.4e" % (disl_A, disl_A, disl_A, disl_A,\
disl_crust_A, disl_A, disl_A, disl_A, disl_A, disl_A, disl_A, disl_A)
    o_dict['Material model']['Visco Plastic TwoD']['Stress exponents for dislocation creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|1.0000e+00|1.0000e+00|1.0000e+00|1.0000e+00,\
spcrust: %.4e|%.4e|1.0000e+00|1.0000e+00,\
spharz: %.4e|%.4e|%.4e|%.4e|1.0000e+00|1.0000e+00|1.0000e+00|1.0000e+00,\
opcrust: %.4e, opharz: %.4e" % (disl_n, disl_n, disl_n, disl_n,\
disl_crust_n, disl_n, disl_n, disl_n, disl_n, disl_n, disl_n, disl_n)
    o_dict['Material model']['Visco Plastic TwoD']['Activation energies for dislocation creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|0.0000e+00|0.0000e+00|0.0000e+00|0.0000e+00,\
spcrust: %.4e|%.4e|0.0000e+00|0.0000e+00,\
spharz: %.4e|%.4e|%.4e|%.4e|0.0000e+00|0.0000e+00|0.0000e+00|0.0000e+00,\
opcrust: %.4e, opharz: %.4e" % (disl_E, disl_E, disl_E, disl_E,\
disl_crust_E, disl_E, disl_E, disl_E, disl_E, disl_E, disl_E, disl_E)
    o_dict['Material model']['Visco Plastic TwoD']['Activation volumes for dislocation creep'] = \
        "background: %.4e|%.4e|%.4e|%.4e|0.0000e+00|0.0000e+00|0.0000e+00|0.0000e+00,\
spcrust: %.4e|%.4e|0.0000e+00|0.0000e+00,\
spharz: %.4e|%.4e|%.4e|%.4e|0.0000e+00|0.0000e+00|0.0000e+00|0.0000e+00,\
opcrust: %.4e, opharz: %.4e" % (disl_V, disl_V, disl_V, disl_V,\
disl_crust_V, disl_V, disl_V, disl_V, disl_V, disl_V, disl_V, disl_V)
    if sz_minimum_viscosity > minimum_viscosity:
        spcrust_value = sz_minimum_viscosity
    else:
        spcrust_value = str(minimum_viscosity)
    if slab_core_viscosity > 0.0:
        spharz_value = slab_core_viscosity
    else:
        spharz_value = str(minimum_viscosity)
    if sz_minimum_viscosity > minimum_viscosity or slab_core_viscosity > 0.0:
        # modify the minimum viscosity for non-linear rheology in the shear zone
        o_dict['Material model']['Visco Plastic TwoD']['Minimum viscosity'] = \
        'background: %s, spcrust: %s, spharz: %s, opcrust: %s, opharz: %s' % \
        (str(minimum_viscosity), spcrust_value, spharz_value, str(minimum_viscosity), str(minimum_viscosity))


def CDPT_assign_yielding(o_dict, cohesion, friction, **kwargs):
    '''
    Assign mantle rheology in the CDPT model
    Inputs:
        kwargs:
            if_couple_eclogite_viscosity - if the viscosity is coupled with the eclogite transition
    ''' 
    crust_cohesion = kwargs.get("crust_cohesion", cohesion)
    crust_friction = kwargs.get("crust_friction", friction)
    if_couple_eclogite_viscosity = kwargs.get("if_couple_eclogite_viscosity", False)
    if abs(cohesion  - 50e6)/50e6 < 1e-6 and abs(friction - 25.0)/25.0 < 1e-6\
    and abs(crust_cohesion  - 50e6)/50e6 < 1e-6 and  abs(crust_friction - 25.0)/25.0 < 1e-6:
        pass  # default conditions
    else:
        if if_couple_eclogite_viscosity:
            # take care of the different phases if the viscosity change is coupled to the eclogite transition
            spcrust_friction_str = "spcrust: %.4e|%.4e|%.4e|%.4e" % (crust_friction, friction, friction, friction)
            spcrust_cohesion_str = "spcrust: %.4e|%.4e|%.4e|%.4e" % (crust_cohesion, cohesion, cohesion, cohesion)
        else:
            spcrust_friction_str = "spcrust: %.4e" % crust_friction
            spcrust_cohesion_str = "spcrust: %.4e" % crust_cohesion
        o_dict['Material model']['Visco Plastic TwoD']["Angles of internal friction"] = "background: %.4e" % friction + ", "\
         + spcrust_friction_str + ", " + "spharz: %.4e, opcrust: %.4e, opharz: %.4e" % (friction, friction, friction)
        o_dict['Material model']['Visco Plastic TwoD']["Cohesions"] = "background: %.4e" % cohesion + ", "\
         + spcrust_cohesion_str + ", " + "spharz: %.4e, opcrust: %.4e, opharz: %.4e" % (cohesion, cohesion, cohesion)


def get_trench_position(sp_age_trench, sp_rate, geometry, Ro, sp_trailing_length):
    '''
    Inputs:
        sp_trainling_length: distance of the trailing end of the subduction plate to the
        side wall.
    Returns:
        trench: coordinate of the trench
    '''
    trench_sph = (sp_age_trench * sp_rate + sp_trailing_length) / Ro * 180.0 / np.pi
    trench_cart = sp_age_trench * sp_rate + sp_trailing_length
    if geometry == "chunk":
        trench = trench_sph
    elif geometry == "box":
        trench = trench_cart
    return trench


def get_trench_position_with_age(sp_age_trench, sp_rate, geometry, Ro):
    '''
    Inputs:
        sp_trainling_length: distance of the trailing end of the subduction plate to the
        side wall.
    Returns:
        trench: coordinate of the trench
    '''
    trench_sph = sp_age_trench * sp_rate / Ro * 180.0 / np.pi
    trench_cart = sp_age_trench * sp_rate
    if geometry == "chunk":
        trench = trench_sph
    elif geometry == "box":
        trench = trench_cart
    return trench


def particle_positions_ef(geometry, Ro, trench0, Dsz, Dbury, p0, slab_lengths, slab_dips, **kwargs):
    '''
    figure out particle positions for the ef method
    Inputs:
        geometry: geometry of the model: box or chunk
        Ro: y/r extent of the geometry
        trench: position of the trench
        Dsz: thickness of the shear zone
        Dbury: bury depth of the particle
    '''
    # assert that equal number of sections are given in slab_lengths and slab_dips
    # and that the slab_dips contains ranges of slab dip angles (2 componets each)
    assert(len(slab_lengths) == len(slab_dips))
    for slab_dip in slab_dips:
        assert(len(slab_dip) == 2)
    # initiation
    if geometry == "chunk":
        trench = Ro * trench0 * np.pi / 180.0  # convert to radian
    elif geometry == "box":
        trench = trench0
    interval = kwargs.get("interval", 10e3)
    num = int(trench//interval)  # figure out the total number of point
    for slab_length in slab_lengths:
        num += int(slab_length//interval)
    particle_data = np.zeros((num, 2))
    for i in range(int(trench//interval)):
        if geometry == "box":
            x = interval * i
            y = Ro - Dsz - Dbury
            pass
        elif geometry == "chunk":
            theta = (interval * i)/Ro
            x = (Ro - Dsz - Dbury) * np.cos(theta)
            y = (Ro - Dsz - Dbury) * np.sin(theta)
            pass
        else:
            pass
        particle_data[i][0] = x
        particle_data[i][1] = y 
    # particles entrained in the slab
    i = int(trench//interval)
    total_slab_length = 0.0
    i_sect = 0
    l1_last = Ro - Dsz - Dbury
    if geometry == "box":
        l2_last = trench
    elif geometry == "chunk":
        l2_last = (Ro - Dsz - Dbury)/Ro * trench
    else:
        pass
    # call a predefined function to get the coordinates in cartesian
    if geometry == "box":
        ps = slab_surface_profile(p0, slab_lengths, slab_dips, "cartesian", num=(num - i))
        xs = ps[:, 0]
        ys = ps[:, 1] -  Dsz - Dbury
    elif geometry == "chunk":
        ps = slab_surface_profile(p0, slab_lengths, slab_dips, "spherical", num=(num - i))
        thetas = ps[:, 0]
        rs = ps[:, 1] -  Dsz - Dbury
        xs = rs*np.cos(thetas)
        ys = rs*np.sin(thetas)
    particle_data[i: num, 0] = xs
    particle_data[i: num, 1] = ys
    return particle_data

def slab_surface_profile(p0_in, slab_lengths_in, slab_dips_in, coordinate_system, **kwargs):
    '''
    descriptions
    Inputs:
        (note: all the angles are passed in with degree, following the World Builder)
        p0 - a start point, in cartesion or spherical coordinates (radian)
        slab_lengths - segment lengths
        slab_dips - segment dip angles
        coordinate_system: "cartesian" or "spherical"
    Returns:
        p1 - an end point of the segment, in cartesion or spherical coordinates (radian)
    '''
    assert(p0_in.size == 2)
    assert(len(slab_lengths_in) == len(slab_dips_in))
    assert(coordinate_system in ["cartesian", "spherical"])
    # num = len(slab_lengths) + 1
    num = kwargs.get("num", 20)
    ps = np.zeros((num,2))
    if coordinate_system == "cartesian":
        p0 = p0_in
    elif coordinate_system == "spherical":
        p0 = p0_in
        p0[0] *= np.pi / 180.0
    ps[0, :] = p0
    slab_total_length = 0.0
    slab_accumulate_lengths = [0.0]
    slab_dips_at_input_points = [slab_dips_in[0][0] * np.pi / 180.0]
    for i in range(len(slab_lengths_in)):
        length = slab_lengths_in[i]
        slab_total_length += length
        slab_accumulate_lengths.append(slab_total_length)
        slab_dips_at_input_points.append(slab_dips_in[i][1] * np.pi / 180.0)
    intv_slab_length = slab_total_length / (num - 1)
    slab_accumulate_lengths_interpolated = np.linspace(0.0, slab_total_length, num)
    slab_dips_interpolated = np.interp(slab_accumulate_lengths_interpolated, slab_accumulate_lengths, slab_dips_at_input_points)
    for i in range(1, num):
        slab_dip = slab_dips_interpolated[i-1]
        if coordinate_system == "cartesian":
            x0 = ps[i-1, 0]
            y0 = ps[i-1, 1]
            ps[i, 0] = x0 + intv_slab_length * np.cos(slab_dip)
            ps[i, 1] = y0 - intv_slab_length * np.sin(slab_dip)
        elif coordinate_system == "spherical":
            theta0 = ps[i-1, 0]
            r0 = ps[i-1, 1]
            ps[i, 0] = theta0 + np.arcsin(np.cos(slab_dip) / (1 + r0**2.0/intv_slab_length**2.0 - 2 * r0/intv_slab_length * np.sin(slab_dip))**0.5)
            ps[i, 1] = (intv_slab_length**2.0 + r0**2.0 - 2 * r0 * intv_slab_length * np.sin(slab_dip))**0.5
    return ps

def PlotCaseRun(case_path, **kwargs):
    '''
    Plot case run result
    Inputs:
        case_path(str): path to the case
        kwargs:
            time_range
            step(int): if this is given as an int, only plot this step
            visualization (str): visualization software, visit or paraview.
            last_step: number of last steps to plot
    Returns:
        -
    '''
    run_visual = kwargs.get('run_visual', 0)
    step = kwargs.get('step', None)
    time_interval = kwargs.get('time_interval', None)
    visualization = kwargs.get('visualization', 'visit')
    plot_axis = kwargs.get('plot_axis', False)
    last_step = kwargs.get('last_step', 3)
    max_velocity = kwargs.get('max_velocity', -1.0)
    plot_types = kwargs.get("plot_types", ["upper_mantle"])
    rotation_plus = kwargs.get("rotation_plus", 0.0)
    # todo_velo
    assert(visualization in ["paraview", "visit", "pygmt"])
    print("PlotCaseRun in TwoDSubduction0: operating")
    # get case parameters
    prm_path = os.path.join(case_path, 'output', 'original.prm')

    # steps to plot: here I use the keys in kwargs to allow different
    # options: by steps, a single step, or the last step
    if type(step) == int:
        kwargs["steps"] = [step]
    elif type(step) == list:
        kwargs["steps"] = step
    elif type(step) == str:
        kwargs["steps"] = step
    else:
        kwargs["last_step"] = last_step

    # Inititiate the class and intepret the options
    # Note that all the options defined by kwargs is passed to the interpret function
    Visit_Options = VISIT_OPTIONS(case_path)
    Visit_Options.Interpret(**kwargs)

    # todo_pexport
    # generate scripts base on the method of plotting
    if visualization == 'visit':
        odir = os.path.join(case_path, 'visit_scripts')
        if not os.path.isdir(odir):
            os.mkdir(odir)
        print("Generating visit scripts")
        py_script = 'slab.py'
        ofile = os.path.join(odir, py_script)
        visit_script = os.path.join(SCRIPT_DIR, 'visit_scripts', 'TwoDSubduction', py_script)
        visit_script_base = os.path.join(SCRIPT_DIR, 'visit_scripts', 'base.py')
        Visit_Options.read_contents(visit_script_base, visit_script)  # combine these two scripts
        Visit_Options.substitute()
    elif visualization == 'paraview':
        odir = os.path.join(case_path, 'paraview_scripts')
        if not os.path.isdir(odir):
            os.mkdir(odir)
        print("Generating paraview scripts")
        py_script = 'slab.py'
        ofile = os.path.join(odir, py_script)
        paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'TwoDSubduction', py_script)
        paraview_script_base = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')
        Visit_Options.read_contents(paraview_script_base, paraview_script)  # combine these two scripts
        # todo_split
        Visit_Options.substitute()
    elif visualization == 'pygmt':
        odir = os.path.join(case_path, 'pygmt_scripts')
        if not os.path.isdir(odir):
            os.mkdir(odir)
        print("Generating pygmt scripts")
        py_script = 'make_lateral_flow_fig.py'
        ofile = os.path.join(odir, py_script)
        pygmt_script = os.path.join(SCRIPT_DIR, 'pygmt_scripts', 'TwoDSubduction', py_script)
        pygmt_script_base = os.path.join(SCRIPT_DIR, 'pygmt_scripts', 'aspect_plotting_util.py')
        Visit_Options.read_contents(pygmt_script_base, pygmt_script)  # combine these two scripts
        Visit_Options.substitute()

    ofile_path = Visit_Options.save(ofile, relative=True)
    # if run_visual == 1:
    #     print("Visualizing using visit")
    #     RunScripts(ofile_path)  # run scripts

    # return the Visit_Options for later usage
    return Visit_Options