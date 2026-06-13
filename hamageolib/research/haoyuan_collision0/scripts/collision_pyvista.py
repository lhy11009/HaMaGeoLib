import os, sys, shutil, math
import numpy as np

# Include this pakage
HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

from hamageolib.research.haoyuan_collision0.post_process import ProcessVtuFileTwoDStep
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD

# Working directories
local_Collision_dir = "/mnt/lochy/ASPECT_DATA/Collision0" # data directory
assert(os.path.isdir(local_Collision_dir))

# Options
local_dir_2d = os.path.join(local_Collision_dir, "collision_setup23/C_WLM_SA50.0")
# option for 1 stage
is_process_second_stage = False; second_stage_outputs = None
prm_basename_2d = "case.prm"; wb_basename_2d = "case.wb"
# option for 2 stage
# is_process_second_stage = True; second_stage_outputs = "output_re"
# prm_basename_2d = "case_re.prm"; wb_basename_2d = "case.wb"

# case option object 
# Decide which part to process
Case_Options_p = None
if is_process_second_stage:
    assert(os.path.isdir(os.path.join(local_dir_2d, second_stage_outputs)))
    Case_Options_p = CASE_OPTIONS_TWOD(local_dir_2d, 
                                          case_file=prm_basename_2d, 
                                          wb_basename=wb_basename_2d, 
                                          output_directory=second_stage_outputs,
                                          pyvista_basename="pyvista_outputs_1",
                                          image_directory="img_1")
    Case_Options_p.Interpret()
    Case_Options_p.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary_1.csv"))
else:
    Case_Options_p = CASE_OPTIONS_TWOD(local_dir_2d, case_file=prm_basename_2d, wb_basename=wb_basename_2d, output_directory="output")
    Case_Options_p.Interpret()
    Case_Options_p.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

include_particles = True  # include particles in post-processing

graphical_steps_np = Case_Options_p.summary_df["Vtu step"].to_numpy()
graphical_steps = [int(step) for step in graphical_steps_np]

# Processing pyvista
for step in graphical_steps:
# while True: # debug
    # step = 104

    pvtu_step = Case_Options_p.get_pvtu_step(step)
    outputs = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_p, 
                                     include_particles=include_particles)
    # print("outputs: ", outputs) # debug

    for key, value in outputs.items():
        Case_Options_p.SummaryCaseVtuStepUpdateValue(key, step, value)
    # break # debug

file_name = "summary.csv"
if is_process_second_stage:
    file_name = "summary_1.csv"
Case_Options_p.SummaryCaseVtuStepExport(os.path.join(local_dir_2d, file_name))