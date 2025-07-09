# todo_data
import os
import numpy as np
import pandas as pd
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import VISIT_OPTIONS_BASE, VISIT_OPTIONS_TWOD, FindWBFeatures, WBFeatureNotFoundError, COMPOSITION, GetSnapsSteps
from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import ggr2cart, string2list, func_name
from hamageolib.utils.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


class CASE_OPTIONS(VISIT_OPTIONS_TWOD, CASE_OPTIONS_BASE):
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
        CASE_OPTIONS_BASE.Interpret(self, **kwargs)
        VISIT_OPTIONS_BASE.Interpret(self, **kwargs)
        idx = FindWBFeatures(self.wb_dict, "Subducting plate")
        idx1 = FindWBFeatures(self.wb_dict, "Slab")

        # Geometry
        sub_plate_feature = self.wb_dict["features"][idx]
        slab_feature = self.wb_dict["features"][idx1]
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
            self.options['TRENCH_EDGE_Y_FULL'] = sub_plate_extends[1]
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
            # todo_3d_chunk
            # in chunk geometry, the coordinate is read in as the latitude, and it's in
            # degree
            Ro = float(self.idict["Geometry model"]["Chunk"]["Chunk outer radius"])
            self.options['TRENCH_EDGE_Y'] = sub_plate_extends[1] * np.pi / 180.0 * Ro * 0.75
            self.options['TRENCH_EDGE_Y_FULL'] = sub_plate_extends[1] * np.pi / 180.0 * Ro
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
        self.options['TRENCH_INITIAL'] = slab_feature['coordinates'][1][0] 
        
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

        # age
        self.options["SP_AGE"] = sp_age
        self.options["OV_AGE"] = ov_age

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
    
    def SummaryCaseVtuStep(self, ofile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_BASE.SummaryCaseVtuStep(self)

        # Here we want to reset it to not mess up old results
        summary_df_foo = self.summary_df.copy(deep=True)

        # start from old file
        if os.path.isfile(ofile):
            # rewrite the results from file
            self.summary_df = pd.read_csv(ofile)

            # Identify new rows in summary_df_foo
            old_vtu_steps = set(self.summary_df["Vtu step"])
            new_rows_mask = ~summary_df_foo["Vtu step"].isin(old_vtu_steps)
            new_rows = summary_df_foo[new_rows_mask]

            # Concatenate new rows to the existing summary
            self.summary_df = pd.concat([self.summary_df, new_rows], ignore_index=True)

            # Sort by "Vtu step"
            self.summary_df.sort_values("Vtu step", inplace=True)
            self.summary_df.reset_index(drop=True, inplace=True)

        # Add new columns you want to add
        new_columns = ["Slab depth", "Trench (center)"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan





