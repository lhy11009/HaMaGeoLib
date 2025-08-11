import os, sys

HaMaGeoLib_DIR = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLib_DIR) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLib_DIR))

# Load package modules
from hamageolib.research.haoyuan_3d_subduction.post_process import get_trench_position_from_file, get_slab_depth_from_file,\
    get_slab_dip_angle_from_file, ProcessVtuFileThDStep
from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS

# Whether to generate pyvista outputs
prepare_pyvista = True

# Case path
local_dir = "/mnt/lochy2/ASPECT_DATA/ThDSubduction/EBA_2d_consistent_8_6/eba3d_width80_c22_AR4_yd100"

# Initiate the case option class
Case_Options = CASE_OPTIONS(local_dir)
Case_Options.Interpret()
Case_Options.SummaryCaseVtuStep(os.path.join(local_dir, "summary.csv"))

graphical_steps_np = Case_Options.summary_df["Vtu step"].to_numpy()
graphical_steps = [int(step) for step in graphical_steps_np]

config = {"threshold_lower": 0.8} # options for processing vtu file
# Generate paraview script
for i, step in enumerate(graphical_steps):

    # get trench center
    pvtu_step = step + int(Case_Options.options['INITIAL_ADAPTIVE_REFINEMENT']) 
    pyvista_outdir = os.path.join(local_dir, "pyvista_outputs", "%05d" % pvtu_step)

    # processing pyvista
    try:
        if prepare_pyvista:
            _, outputs = ProcessVtuFileThDStep(local_dir, pvtu_step, Case_Options)

        trench_center = get_trench_position_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'])
        slab_depth = get_slab_depth_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'], float(Case_Options.options['OUTER_RADIUS']), "sp_lower")
        dip_100_center = get_slab_dip_angle_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'], float(Case_Options.options['OUTER_RADIUS']), "sp_upper", 0.0, 100e3)
        
        Case_Options.SummaryCaseVtuStepUpdateValue("File found", step, True)
    except FileNotFoundError:
        Case_Options.SummaryCaseVtuStepUpdateValue("File found", step, False)
    else:
        # update value in sumamry
        Case_Options.SummaryCaseVtuStepUpdateValue("Slab depth", step, slab_depth)
        Case_Options.SummaryCaseVtuStepUpdateValue("Trench (center)", step, trench_center)
        Case_Options.SummaryCaseVtuStepUpdateValue("Dip 100 (center)", step, dip_100_center)
    
    # break # debug

Case_Options.SummaryCaseVtuStepExport(os.path.join(local_dir, "summary.csv"))