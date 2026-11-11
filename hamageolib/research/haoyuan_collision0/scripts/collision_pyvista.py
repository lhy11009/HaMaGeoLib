import os, sys, shutil, math
import numpy as np

# Include this pakage
HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

from hamageolib.research.haoyuan_collision0.post_process import process_all_vtu_steps, prepare_case_option_2d

# Working directories
local_Collision_dir = "/mnt/lochy/ASPECT_DATA/Collision0" # data directory
assert(os.path.isdir(local_Collision_dir))

# Options
# one_vtu_step - if this option is not None, only execute one step
local_dir_2d = os.path.join(local_Collision_dir,
                            "collision_setup39/C_PE5.00e+06_WLF2.0e-02_SA100.0_CA_Kf3.00e-07_marine_SeaL1.00e+03_GN"
                            )
# Select the stage with this option alone; do not include the "output_" prefix.
# A tag reads case_two_stage_{tag}.prm and output_{tag}, using the shared case.wb.
# Results go to pyvista_outputs_{tag} and summary_{tag}.csv.
second_stage_tag = None  # Read case.prm and output for the original stage.
# second_stage_tag = "St2.0Ma_Cv4.00e-02_Sp2.00e-02"

# if min and max steps are given, then only perform analysis for steps in between,
# otherwise, loop for all the visualization steps
graphical_step_min = None
graphical_step_max = None

# if this is set to None, then loop all steps
# if this is set to a number, only execute one step
one_vtu_step = None

include_particles = False  # include particles in post-processing
include_fastscape = True # include fastscape files
include_topography = True # include topography in post-processing
analyze_shortening = False # include shortening in post-processing
Case_Options_2d = prepare_case_option_2d(
    local_dir_2d, second_stage_tag=second_stage_tag,
)

process_all_vtu_steps(local_dir_2d, Case_Options_2d,
                      graphical_step_min=graphical_step_min,
                      graphical_step_max=graphical_step_max,
                      one_vtu_step=one_vtu_step,
                      include_particles=include_particles,
                      include_topography=include_topography,
                      include_fastscape=include_fastscape,
                      analyze_shortening=analyze_shortening)
