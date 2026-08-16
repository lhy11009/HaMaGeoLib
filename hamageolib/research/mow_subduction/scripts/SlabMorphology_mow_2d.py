import os, sys

HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

from hamageolib.research.mow_subduction.case_options import CASE_OPTIONS_TWOD
from hamageolib.research.haoyuan_3d_subduction.post_process import PlotCaseRunTwoD1, ProcessVtuFileTwoDStep
from hamageolib.research.haoyuan_2d_subduction.workflow_scripts import run_2d_subduction_visualization

# options
# todo_by
# one_graphical_step - If one graphical step is given, then only this step is executed.
#   Otherwise loop for all the steps with vtu outputs
#   Note this step is time / 0.1 Ma
local_dir_2d = "/mnt/lochy/ASPECT_DATA/MOW/mow01/C_mow_h2890.0_M_gr3_ar4_gz_2"
one_graphical_step = None # None or int number


assert(local_dir_2d is not None)

# case options 
Case_Options_2d = CASE_OPTIONS_TWOD(local_dir_2d)
Case_Options_2d.Interpret()
Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

# steps to post-process
graphical_steps_np = Case_Options_2d.summary_df["Vtu step"].to_numpy()
graphical_steps = [int(step) for step in graphical_steps_np]
if one_graphical_step is not None:
    graphical_steps = [one_graphical_step]

# Processing pyvista
# summarize additional metastable properties:
#   area of the MOW area
#   area of the MOW area in the cold slab
for step in graphical_steps:

    pvtu_step = step + int(Case_Options_2d.options['INITIAL_ADAPTIVE_REFINEMENT'])
    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)
    print("output_dict: ", output_dict) # debug
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab depth", step, output_dict["slab_depth"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Trench", step, output_dict["trench_center"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Trench (50 km)", step, output_dict["trench_center_50km"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Dip 100", step, output_dict["dip_100"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Dip 400", step, output_dict["dip_400"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab 400", step, output_dict["slab_400"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Dip 100 400", step, output_dict["dip_100_400"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Sp velocity", step, output_dict["sp_velocity"])
    if Case_Options_2d.options["MODEL_TYPE"] == "mow":
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Mow area", step, output_dict["metastable_area"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Mow area cold", step, output_dict["metastable_area_cold"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Mow area cold depth", step, output_dict["metastable_area_cold_depth"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("T depth 973.15", step, output_dict["T_depth_973.15"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("T depth 923.15", step, output_dict["T_depth_923.15"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("T depth 1023.15", step, output_dict["T_depth_1023.15"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy thermal", step, output_dict["slab_buoyancy_thermal"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy thermal MTZ", step, output_dict["slab_buoyancy_thermal_MTZ"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy MOW area cold", step, output_dict["slab_buoyancy_metastable_area_cold"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy beta MOW area cold", step, output_dict["slab_buoyancy_beta_metastable_area_cold"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy equilibrium area cold", step, output_dict["slab_buoyancy_equilibrium_area_cold"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy beta equilibrium area cold", step, output_dict["slab_buoyancy_beta_equilibrium_area_cold"])
        Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab buoyancy pv equilibrium area cold", step, output_dict["slab_buoyancy_pv_equilibrium_area_cold"])
    # break # debug

Case_Options_2d.SummaryCaseVtuStepExport(os.path.join(local_dir_2d, "summary.csv"))