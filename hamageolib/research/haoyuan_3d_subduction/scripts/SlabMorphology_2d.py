import os, sys

HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS_TWOD1
from hamageolib.research.haoyuan_3d_subduction.post_process import PlotCaseRunTwoD1, ProcessVtuFileTwoDStep
from hamageolib.research.haoyuan_2d_subduction.workflow_scripts import run_2d_subduction_visualization


local_dir_2d = "/mnt/lochz/ASPECT_DATA/TwoDSubduction/EBA_CDPT_3dconsistent_13/eba_cdpt_coh300_SA80.0_OA40.0_width80_ss300.0"

assert(local_dir_2d is not None)

# case options 
Case_Options_2d = CASE_OPTIONS_TWOD1(local_dir_2d)
Case_Options_2d.Interpret()
Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

graphical_steps_np = Case_Options_2d.summary_df["Vtu step"].to_numpy()
graphical_steps = [int(step) for step in graphical_steps_np]

# Processing pyvista
for step in graphical_steps:
    pvtu_step = step + int(Case_Options_2d.options['INITIAL_ADAPTIVE_REFINEMENT'])
    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)
    print("output_dict: ", output_dict) # debug
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Slab depth", step, output_dict["slab_depth"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Trench", step, output_dict["trench_center"])
    Case_Options_2d.SummaryCaseVtuStepUpdateValue("Dip 100", step, output_dict["dip_100"])
    # break # debug

Case_Options_2d.SummaryCaseVtuStepExport(os.path.join(local_dir_2d, "summary.csv"))