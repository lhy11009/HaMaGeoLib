import os, sys, shutil, math
import numpy as np

# Include this pakage
HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

from hamageolib.research.haoyuan_collision0.post_process import process_all_vtu_steps, prepare_case_option_2d
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD

# Working directories
local_Collision_dir = "/mnt/lochy/ASPECT_DATA/Collision0" # data directory
assert(os.path.isdir(local_Collision_dir))

# Options
# one_vtu_step - if this option is not None, only execute one step
local_dir_2d = os.path.join(local_Collision_dir, 
                            "collision_setup27/C_ar5_WLM_WLF5.0e-02_SA50.0_FS"
                            )
# prm_basename_2d = "case.prm"; wb_basename_2d = "case.wb"; output_directory="output" # normal
prm_basename_2d = "case.prm"; wb_basename_2d = "case.wb"; output_directory="output" # in case we use a different name
# if min and max steps are given, then only perform analysis for steps in between,
# otherwise, loop for all the visualization steps
graphical_step_min = 70
graphical_step_max = None

# if this is set to None, then loop all steps
# if this is set to a number, only execute one step
one_vtu_step = None # None

include_particles = False  # include particles in post-processing
include_topography = True # include topography in post-processing
analyze_shortening = False # include shortening in post-processing
# option for 1 stage
is_process_second_stage = False; second_stage_outputs = None
# option for 2 stage
# is_process_second_stage = True; second_stage_outputs = "output_re"
# prm_basename_2d = "case_re.prm"; wb_basename_2d = "case.wb"

# Prepare case options
# The second option of the function is is_process_second_stage, this is not needed
# if not processing a dual-stage model 
Case_Options_2d = prepare_case_option_2d(local_dir_2d, is_process_second_stage,
                                        prm_basename_2d=prm_basename_2d, 
                                        wb_basename_2d=wb_basename_2d, 
                                        output_directory=output_directory, 
                                        second_stage_outputs="output_re")

process_all_vtu_steps(local_dir_2d, Case_Options_2d,
                      graphical_step_min=graphical_step_min,
                      graphical_step_max=graphical_step_max,
                      one_vtu_step=one_vtu_step,
                      include_particles=include_particles,
                      include_topography=include_topography,
                      analyze_shortening=analyze_shortening)