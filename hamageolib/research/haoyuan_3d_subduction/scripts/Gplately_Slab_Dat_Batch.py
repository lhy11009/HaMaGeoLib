
# todo_gp
import os,sys
import numpy as np

HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))
    
from hamageolib.research.haoyuan_3d_subduction.gplately_utilities import GPLATE_PROCESS

def main():

    # parameters     
    model_name = "Muller2019"
    anchor_plate_id = 0 # anchor plate id: 0 - Africa

    start_time = 0.0
    restart_time = None # invoke this options to restart from a previous fun
    end_time = 61.0
    interval = 1.0

    only_one_pid = None  # None - process all subductions; a number - only process with this subducting plid
    
    arc_length_edge = 2.0; arc_length_resample_section = 2.0
    
    case_dir = "/mnt/lochy/ASPECT_DATA/ThDSubduction/gplate_dataset_09202025"

    # initiate
    reconstruction_times = np.arange(start_time, end_time, interval)

    GplateP = GPLATE_PROCESS(case_dir)

    # loop first to get pid and region dict
    pid_dict = {} # record existing pid in timestep
    region_dict = {} # record plot region for each subduction
    
    for i, reconstruction_time in enumerate(reconstruction_times):
        reconstruction_time = int(reconstruction_time)
        
        # Import reconstruction dataset
        GplateP.reconstruct(model_name, reconstruction_time, anchor_plate_id)

        # Add age
        GplateP.add_age_raster()

        # Resample the subductions
        GplateP.resample_subduction(arc_length_edge, arc_length_resample_section)

        # update the pid dict
        pid_dict = GplateP.update_unique_pid_dict(True, pid_dict)
        
        # update the region dict
        region_dict = GplateP.update_region_dict(True, region_dict)

    # loop again to generate plots
    color_dict = {} # record color options
    for i, reconstruction_time in enumerate(reconstruction_times):
        if restart_time is not None and reconstruction_time < restart_time:
            # skip initial steps if restarting
            continue

        reconstruction_time = int(reconstruction_time)
        
        # Import reconstruction dataset
        GplateP.reconstruct(model_name, reconstruction_time, anchor_plate_id)

        # Add age
        GplateP.add_age_raster()
        GplateP.export_csv("subduction_data", "ori.csv")

        # Inspect and save results of every subduction
        # color_dict = GplateP.save_results_ori(True, only_one_pid=only_one_pid, color_dict=color_dict)

        # Resample the subductions
        GplateP.resample_subduction(arc_length_edge, arc_length_resample_section)
        
        # Inspect and save results of resampled dataset
        color_dict = GplateP.save_results_resampled(True, only_one_pid=only_one_pid, color_dict=color_dict, region_dict=region_dict)

main()