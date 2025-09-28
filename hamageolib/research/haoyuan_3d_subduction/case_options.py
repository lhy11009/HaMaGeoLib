
import os
import numpy as np
import pandas as pd
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import VISIT_OPTIONS_BASE, FindWBFeatures, WBFeatureNotFoundError, COMPOSITION, GetSnapsSteps
from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import ggr2cart, string2list, func_name, my_assert
from hamageolib.utils.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


class CASE_OPTIONS(VISIT_OPTIONS_BASE, CASE_OPTIONS_BASE):
    """
    parse .prm file to a option file that bash can easily read
    """
    def __init__(self, case_dir):
        '''
        class initiation
        '''
        VISIT_OPTIONS_BASE.__init__(self, case_dir)
        CASE_OPTIONS_BASE.__init__(self, case_dir)

    def Interpret(self, **kwargs):
        """
        Interpret the inputs, to be reloaded in children
        kwargs: options
            last_step(list): plot the last few steps
        """
        # additional inputs
        rotation_plus = kwargs.get("rotation_plus", 0.0) # additional rotation
        
        to_rad = np.pi / 180.0
        # call function from parent
        VISIT_OPTIONS_BASE.Interpret(self, **kwargs)
        CASE_OPTIONS_BASE.Interpret(self, **kwargs)
        idx = FindWBFeatures(self.wb_dict, "Subducting plate")

        # Geometry
        sub_plate_feature = self.wb_dict["features"][idx]
        # this is point marking the extension of plate
        sub_plate_extends = sub_plate_feature['coordinates'][1]
        box_width = -1.0
        Ro = -1.0
        sp_age = np.nan
        ov_age = np.nan
        self.options['ROTATION_ANGLE'] = 0.0
        if self.options["GEOMETRY"] == "box":
            self.options['BOX_LENGTH'] = self.idict["Geometry model"]["Box"]["X extent"]
            box_width = self.idict["Geometry model"]["Box"]["Y extent"]
            self.options['TRENCH_EDGE_Y'] = sub_plate_extends[1] * 0.75
            self.options['SLAB_EXTENTS_FULL'] = sub_plate_extends[1]
            self.options['BOX_WIDTH'] = box_width
            self.options['BOX_THICKNESS'] = self.idict["Geometry model"]["Box"]["Z extent"]
            try:
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
            except WBFeatureNotFoundError:
                pass
            else:
                feature_sp = self.wb_dict['features'][index]
                trench_x = feature_sp["coordinates"][2][0]
                self.options['THETA_REF_TRENCH'] = trench_x
                try:
                    spreading_velocity = feature_sp["temperature models"][0]["spreading velocity"]
                except KeyError:
                    pass
                else:
                    sp_age = trench_x / spreading_velocity 
            try: 
                idx2 = FindWBFeatures(self.wb_dict, "Overiding plate")
            except WBFeatureNotFoundError:
                pass
            else:
                ov_plate_feature = self.wb_dict["features"][idx2]
                ov_age = ov_plate_feature["temperature models"][0]["plate age"]
            self.options['ROTATION_ANGLE'] = 0.0
        elif self.options["GEOMETRY"] == "chunk":
            
            # in chunk geometry, the coordinate is read in as the latitude, and it's in
            # degree
            Ro = float(self.idict["Geometry model"]["Chunk"]["Chunk outer radius"])
            self.options['TRENCH_EDGE_Y'] = sub_plate_extends[1] * np.pi / 180.0 * Ro * 0.75
            self.options['SLAB_EXTENTS_FULL'] = sub_plate_extends[1] * np.pi / 180.0 * Ro
            self.options['TRENCH_EDGE_LAT_FULL'] = sub_plate_extends[1]
            self.options["CHUNK_RIDGE_CENTER_X"] = Ro
            self.options["CHUNK_RIDGE_CENTER_Z"] = 0.0
            # convert to x, y, z with long = 0.75 * plate_extent, lat = 0.0, r = Ro
            ridge_edge_x, ridge_edge_y, ridge_edge_z = ggr2cart(sub_plate_extends[1]*to_rad*0.75, 0.0, Ro)
            self.options["CHUNK_RIDGE_EDGE_X"] = ridge_edge_x
            self.options["CHUNK_RIDGE_EDGE_Z"] = ridge_edge_z
            self.options["MAX_LATITUDE"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum latitude']) * np.pi / 180.0
            self.options["MAX_LONGITUDE"] = float(self.idict['Geometry model']['Chunk']['Chunk maximum longitude']) * np.pi / 180.0
            
            self.options['BOX_WIDTH'] = -1.0
            try:
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
            except WBFeatureNotFoundError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                rotation_angle = 52.0 + rotation_plus
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                trench_phi = feature_sp["coordinates"][2][0]
                self.options['THETA_REF_TRENCH'] = trench_phi * np.pi/180.0
                rotation_angle = 90.0 - trench_phi - 2.0 + rotation_plus
                try:
                    spreading_velocity = feature_sp["temperature models"][0]["spreading velocity"]
                except KeyError:
                    pass
                else:
                    sp_age = trench_phi * np.pi / 180.0 * Ro/ spreading_velocity 
            try: 
                idx2 = FindWBFeatures(self.wb_dict, "Overiding plate")
            except WBFeatureNotFoundError:
                pass
            else:
                ov_plate_feature = self.wb_dict["features"][idx2]
                ov_age = ov_plate_feature["temperature models"][0]["plate age"]
                self.options['ROTATION_ANGLE'] = rotation_angle
        else:
            raise ValueError("geometry must by either box or chunk")


        # viscosity
        # self.options['ETA_MIN'] = self.idict['Material model']['Visco Plastic TwoD']['Minimum viscosity']
        # self.options['ETA_MAX'] = self.idict['Material model']['Visco Plastic TwoD']['Maximum viscosity']
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

        # slab
        # TRENCH_INI_DERIVED - we should derive this later
        idx1 = FindWBFeatures(self.wb_dict, "Slab")
        slab_feature = self.wb_dict["features"][idx1]
        self.options['TRENCH_INITIAL'] = slab_feature['coordinates'][1][0]
        self.options['TRENCH_INI_DERIVED'] = -1.0
        self.options['INITIAL_SHEAR_ZONE_THICKNESS'] = slab_feature['segments'][0]["composition models"][0]["max distance slab top"]
        
        
        # yield stress
        try:
            self.options["MAXIMUM_YIELD_STRESS"] = float(self.idict['Material model']['Visco Plastic TwoD']["Maximum yield stress"])
        except KeyError:
            self.options["MAXIMUM_YIELD_STRESS"] = 1e9

        # peierls rheology
        try:
            include_peierls_rheology = self.idict['Material model']['Visco Plastic TwoD']['Include Peierls creep']
            if include_peierls_rheology == 'true':
                self.options['INCLUDE_PEIERLS_RHEOLOGY'] = True
            else:
                self.options['INCLUDE_PEIERLS_RHEOLOGY'] = False
        except KeyError:
            self.options['INCLUDE_PEIERLS_RHEOLOGY'] = False

        # todo_center
        # age
        self.options["SP_AGE"] = sp_age
        self.options["OV_AGE"] = ov_age

        # MOW
        names_of_compositional_fields_str = self.idict["Compositional fields"]["Names of fields"]
        if "metastable" in names_of_compositional_fields_str:
            self.options["MODEL_TYPE"] = "mow"

        if self.options["MODEL_TYPE"] == "mow":
            default_dict = {
                "Phase transition Clapeyron slope": 2e6,
                "Phase transition depth": 410e3,
                "Phase transition temperature": 1740.0
            }
            metastable_dict = self.idict["Material model"].get("metastable", default_dict)
            self.options["CL_PT_EQ"] = metastable_dict.get("Phase transition Clapeyron slope", 2e6)
            self.options["DEPTH_PT_EQ"] = metastable_dict.get("Phase transition depth", 410e3)
            self.options["P_PT_EQ"] = 1.34829e+10
            self.options["T_PT_EQ"] = metastable_dict.get("Phase transition temperature", 1740.0)
        
        self.options["MAX_PLOT_DEPTH_IN_SLICE"]  = 1300e3

    def vtk_options(self, **kwargs):
        '''
        options of vtk scripts
        '''
        # call function from parent
        VISIT_OPTIONS_BASE.vtk_options(self, **kwargs)
    
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
            center_profile_file_path = os.path.join(self._case_dir, "vtk_outputs", "center_profile_%05d.txt" % snap)
            if os.path.isfile(center_profile_file_path):
                # append if the file is found
                last_time = time
                psnaps.append(snap)
                ptimes.append(time)
        return ptimes, psnaps
    
    def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_BASE.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        new_columns = ["Slab depth", "Trench (center)", "Trench (center 50km)", "Dip 100 (center)"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan


class CASE_OPTIONS_TWOD1(VISIT_OPTIONS_BASE, CASE_OPTIONS_BASE):

    def __init__(self, case_dir):
        '''
        class initiation
        '''
        VISIT_OPTIONS_BASE.__init__(self, case_dir)
        CASE_OPTIONS_BASE.__init__(self, case_dir) 
    
    def Interpret(self, **kwargs):
        """
        Interpret the inputs, to be reloaded in children.
        This class is different from the class for the 2d cases
        kwargs: options
            last_step(list): plot the last few steps
        """
        # call function from parent
        CASE_OPTIONS_BASE.Interpret(self, **kwargs)
        VISIT_OPTIONS_BASE.Interpret(self, **kwargs)

        # Rotation angles 
        self.options['ROTATION_ANGLE'] = 0.0
        rotation_plus = kwargs.get("rotation_plus", 0.0) # additional rotation

        # Maximum and minimum viscosity
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

        # Domain and rotation
        if self.options['GEOMETRY'] == 'chunk':
            try:
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
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
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
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

        # plate setup
        if self.options['GEOMETRY'] == 'chunk':
            sp_age = -1.0
            ov_age = -1.0
            Ro = 6371e3
            try:
                index = FindWBFeatures(self.wb_dict, "Subducting plate")
                index1 = FindWBFeatures(self.wb_dict, "Overiding plate")
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
            self.options['BOX_LENGTH'] = self.idict["Geometry model"]["Box"]["X extent"]
            self.options['BOX_THICKNESS'] = self.idict["Geometry model"]["Box"]["Y extent"]

            sp_age = -1.0
            ov_age = -1.0
            try:
                
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
                index1 = FindWBFeatures(self.wb_dict, "Overiding plate")
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
        
        # Shear Zone configuration
        index = FindWBFeatures(self.wb_dict, 'Subducting plate')
        feature_sp = self.wb_dict['features'][index]
        self.options["INITIAL_SHEAR_ZONE_THICKNESS"] = feature_sp["composition models"][0]["max depth"]
        decoupling_eclogite_viscosity = self.idict['Material model']['Visco Plastic TwoD'].get('Decoupling eclogite viscosity', 'false')
        if decoupling_eclogite_viscosity == 'true':
            self.options["SHEAR_ZONE_CUTOFF_DEPTH"] = float(self.idict['Material model']['Visco Plastic TwoD']["Eclogite decoupled viscosity"]["Decoupled depth"])
        else:
            self.options["SHEAR_ZONE_CUTOFF_DEPTH"] = -1.0

        A_diff_inputs = COMPOSITION(self.idict['Material model']['Visco Plastic TwoD']['Prefactors for diffusion creep'])
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
        # TRENCH_INI_DERIVED - we should derive this later
        if self.options['GEOMETRY'] == 'chunk':
            try:
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
            except KeyError:
                # either there is no wb file found, or the feature 'Subducting plate' is not defined
                theta_ref_trench = 0.63
            else:
                # rotate to center on the slab
                feature_sp = self.wb_dict['features'][index]
                theta_ref_trench = feature_sp["coordinates"][2][0] / 180.0 * np.pi
        elif self.options['GEOMETRY'] == 'box':
            try:
                index = FindWBFeatures(self.wb_dict, 'Subducting plate')
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
        self.options['TRENCH_INITIAL'] = theta_ref_trench
        self.options['TRENCH_INI_DERIVED'] = -1.0

        self.options["MAX_PLOT_DEPTH_IN_SLICE"]  = 1300e3

    def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_BASE.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        new_columns = ["Slab depth", "Trench", "Dip 100", "Trench (50 km)"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan