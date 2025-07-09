# todo_data
import os
import numpy as np
from hamageolib.utils.case_options import CASE_OPTIONS
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import VISIT_OPTIONS_BASE, FindWBFeatures, WBFeatureNotFoundError, COMPOSITION, GetSnapsSteps
from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import ggr2cart, string2list
from hamageolib.research.haoyuan_3d_subduction.post_process import ProcessVtuFileThDStep
from hamageolib.utils.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


# class CASE_OPTIONS_THD(CASE_OPTIONS):
#     def __init__(self, case_dir):
#         CASE_OPTIONS.__init__(self, case_dir)
    
#     def interpret(self):
#         return CASE_OPTIONS.interpret(self)

#     def SummaryCase(self):
#         """
#         Generate Case Summary
#         """
#         CASE_OPTIONS.SummaryCase(self)
#         # add specific columns
#         self.summary_df["trench"] = np.nan


class CASE_OPTIONS(VISIT_OPTIONS_BASE, CASE_OPTIONS_BASE):
    """
    parse .prm file to a option file that bash can easily read
    """
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


class PLOT_CASE_RUN_THD():
    '''
    Plot case run result
    Attributes:
        case_path(str): path to the case
        Visit_Options: options for case
        kwargs:
            time_range
            step(int): if this is given as an int, only plot this step

    Returns:
        -
    '''
    def __init__(self, case_path, **kwargs):
        '''
        Initiation
        Inputs:
            case_path - full path to a 3-d case
            kwargs
        '''

        self.case_path = case_path
        self.kwargs = kwargs
        print("PlotCaseRun in ThDSubduction: operating")
        step = kwargs.get('step', None)
        last_step = kwargs.get('last_step', 3)

        # steps to plot: here I use the keys in kwargs to allow different
        # options: by steps, a single step, or the last step
        if type(step) == int:
            self.kwargs["steps"] = [step]
        elif type(step) == list:
            self.kwargs["steps"] = step
        elif type(step) == str:
            self.kwargs["steps"] = step
        else:
            self.kwargs["last_step"] = last_step

        # get case parameters
        # prm_path = os.path.join(self.case_path, 'output', 'original.prm')
        
        # initiate options
        self.Visit_Options = CASE_OPTIONS(self.case_path)
        self.Visit_Options.Interpret(**self.kwargs)

    def ProcessPyvista(self):
        '''
        pyvista processing
        '''
        for vtu_step in self.Visit_Options.options['GRAPHICAL_STEPS']:
            pvtu_step = vtu_step + int(self.Visit_Options.options['INITIAL_ADAPTIVE_REFINEMENT'])
            ProcessVtuFileThDStep(self.case_path, pvtu_step, self.Visit_Options)
        return
            

    def GenerateParaviewScript(self, ofile_list, additional_options):
        '''
        generate paraview script
        Inputs:
            ofile_list - a list of file to include in paraview
            additional_options - options to append
        '''
        require_base = self.kwargs.get('require_base', True)
        for ofile_base in ofile_list:
            ofile = os.path.join(self.case_path, 'paraview_scripts', ofile_base)
            paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts',"ThDSubduction", ofile_base)
            if require_base:
                paraview_base_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')  # base.py : base file
                self.Visit_Options.read_contents(paraview_base_script, paraview_script)  # this part combines two scripts
            else:
                self.Visit_Options.read_contents(paraview_script)  # this part combines two scripts
            self.Visit_Options.options.update(additional_options)
            self.Visit_Options.substitute()  # substitute keys in these combined file with values determined by Interpret() function
            ofile_path = self.Visit_Options.save(ofile, relative=False)  # save the altered script
            print("\t File generated: %s" % ofile_path)


