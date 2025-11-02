import os
import numpy as np
from hamageolib.utils.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import COMPOSITION

class CASE_OPTIONS_TWOD(CASE_OPTIONS_BASE):
    '''
    Class: CASE_OPTIONS_TWOD
    Purpose:
        Parse `.prm` files and convert model parameters into a bash-readable option format.
    Inheritance:
        VISIT_OPTIONS_BASE, CASE_OPTIONS_BASE — this class extends both parent parsers.
    Attributes:
        case_dir (str): Directory path to the simulation case folder; passed to parent constructors.
    Notes:
        This class is designed to unify options from legacy VISIT-based workflow and the modern case-options handler.
    '''

    def __init__(self, case_dir, **kwargs):
        '''
        Initialize CASE_OPTIONS object.
        Parameters:
            case_dir (str): Path to the case directory containing parameter files.
        Returns:
            None — initializes parent classes.
        '''
        CASE_OPTIONS_BASE.__init__(self, case_dir, **kwargs)    # initialize base case option structure

    def Interpret(self, **kwargs):
        '''
        Interpret the provided keyword options and apply parsing logic.
        Parameters:
            **kwargs (dict):
                last_step (list): If provided, indicate the last few timesteps for plotting.
        Returns:
            None — updates internal option state.
        Notes:
            Designed to be overridden or extended in child classes.
        '''
        # additional inputs (placeholder) — extend here as needed

        # call functions from parent classes to process options
        CASE_OPTIONS_BASE.Interpret(self, **kwargs)

        # add additional options below

        # todo_collision
        eta_min_inputs =\
            COMPOSITION(self.idict['Material model']['Visco Plastic']['Minimum viscosity']) 
        self.options['ETA_MIN'] = eta_min_inputs.data['background'][0] # use phases
            
        eta_max_inputs =\
            COMPOSITION(self.idict['Material model']['Visco Plastic']['Maximum viscosity']) 
        self.options['ETA_MAX'] = eta_max_inputs.data['background'][0] # use phases
