import os
import numpy as np
from hamageolib.utils.case_options import CASE_OPTIONS as CASE_OPTIONS_BASE
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import COMPOSITION
from hamageolib.utils.aspect.prm_wb_utils import find_WB_feature_by_name


def shallowest_suture_x(x, depths):
    """Return X (meters) at the shallowest finite profile point, or NaN."""
    if x is None or depths is None:
        return np.nan
    x, depths = np.asarray(x), np.asarray(depths)
    valid = np.isfinite(x) & np.isfinite(depths)
    if not np.any(valid):
        return np.nan
    return float(x[valid][np.argmin(depths[valid])])


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

        # Constant sea level relative to the undeformed ASPECT surface (m).
        marine = (self.idict.get("Mesh deformation", {})
                  .get("Fastscape", {}).get("Marine parameters", {}))
        self.options["SEA_LEVEL"] = float(marine.get("Sea level", 0.0))

        # Keep sedimentation positive/red; select only the erosion-rate range.
        # False: +/-3e-4 m/yr; True: legacy +/-1e-2 m/yr.
        self.options["USE_LEGACY_EROSION_RATE_RANGE"] = "False"

        # add additional options below

        # plate_setup
        idx_feature, feature = find_WB_feature_by_name(self.wb_dict, "Subducting Plate")

        self.options["PLATE_START_POINT"] = feature["coordinates"][-1][0]
        self.options["SLAB_HINGE_POINT"] = feature["coordinates"][0][0]

        # todo_collision
        eta_min_inputs =\
            COMPOSITION(self.idict['Material model']['Visco Plastic']['Minimum viscosity']) 
        self.options['ETA_MIN'] = eta_min_inputs.data['background'][0] # use phases
            
        eta_max_inputs =\
            COMPOSITION(self.idict['Material model']['Visco Plastic']['Maximum viscosity']) 
        self.options['ETA_MAX'] = eta_max_inputs.data['background'][0] # use phases

        # compositions
        self.options["HAS_HARZBURGITE"] = False
        if "harzburgite" in self.options['COMPOSITION_FIELDS']:
            self.options["HAS_HARZBURGITE"] = True

        self.options["CONTINENT_COMPOSITIONS"] = ['crust_upper', 'crust_lower']
        self.options["OCEANIC_CRUST_COMPOSITIONS"] = ['gabbro', 'MORB']

    def retrieve_visualization_file(self, visualization_snapshot_step):
        """Select raw and processed inputs from the snapshot's recorded stage.

        Tagged runs inherit snapshots from output/ even when their times exceed
        the requested resume time. Their PyVista inputs belong to the original
        stage as well; generated scripts and images retain their tagged paths.
        """
        filename = super().retrieve_visualization_file(visualization_snapshot_step)
        if getattr(self, "second_stage_tag", None) is not None:
            source_output = os.path.normpath(os.path.join(
                self.case_dir, os.path.dirname(filename), ".."))
            original_output = os.path.normpath(os.path.join(self.case_dir, "output"))
            self.options["PYVISTA_DIRECTORY"] = (
                os.path.join(self.pp_dir, "pyvista_outputs")
                if source_output == original_output else self.pyvista_dir
            )
        return filename

    def SummaryCaseVtuStep(self, ifile=None):
        '''
        Summary case result
        ofile (str): if this provided, import old results
        '''
        CASE_OPTIONS_BASE.SummaryCaseVtuStep(self, ifile)

        # Add new columns you want to add
        # todo_pin
        new_columns = ["slab_depth", "dip_100", "dip_300", "trench_center", "trench_center_50", 
                       "sp_velocity", "ov_velocity", "lower_plate_deformation", "upper_plate_deformation",
                       "trench_initial_X", "trench_50_initial_X", "average_topography",
                       "suture_shallow_x"]

        for col in new_columns:
            if col not in self.summary_df.columns:
                self.summary_df[col] = np.nan    

    def backfill_suture_shallow_x(self, *, export=False):
        """Fill missing summary values from existing suture VTP profiles.

        Uses snapshot numbers and recorded stage paths, including inherited
        first-stage outputs in tagged cases. Existing finite values are kept;
        absent or wholly invalid profiles leave NaN. Depth is taken from the
        saved ``depth`` field, or from vertical coordinates for older files.
        No simulation VTU or particle files are read. Returns the number filled.

        After prepare_case_option_2d(...), call
        ``options.backfill_suture_shallow_x(export=True)`` to save the selected
        stage's summary CSV; the default updates only the in-memory summary.
        """
        import pyvista as pv

        if "suture_shallow_x" not in self.summary_df:
            self.summary_df["suture_shallow_x"] = np.nan
        filled = 0
        for index, row in self.summary_df.iterrows():
            if np.isfinite(row["suture_shallow_x"]):
                continue
            snapshot = int(row["Vtu snapshot"])
            directory = self.pyvista_dir
            if getattr(self, "second_stage_tag", None) is not None:
                self.retrieve_visualization_file(snapshot)
                directory = self.options["PYVISTA_DIRECTORY"]
            filename = os.path.join(directory, f"{snapshot:05d}", f"suture_{snapshot:05d}.vtp")
            if not os.path.isfile(filename):
                continue
            profile = pv.read(filename)
            depths = profile["depth"] if "depth" in profile.point_data else -profile.points[:, 1]
            value = shallowest_suture_x(profile.points[:, 0], depths)
            if np.isfinite(value):
                self.summary_df.at[index, "suture_shallow_x"] = value
                filled += 1
        if export:
            self.SummaryCaseVtuStepExport(self.summary_file)
        return filled
