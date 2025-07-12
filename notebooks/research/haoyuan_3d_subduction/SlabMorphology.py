
# Initiate the case option class
from hamageolib.research.haoyuan_3d_subduction.post_process import get_trench_position_from_file, get_slab_depth_from_file, PLOT_CASE_RUN_THD
from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS
import os

local_dir = "/mnt/lochy/ASPECT_DATA/ThDSubduction/chunk_geometry1/eba3d_width80_bw4000_sw1000_yd500.0_AR4"

Case_Options = CASE_OPTIONS(local_dir)
Case_Options.Interpret()
Case_Options.SummaryCaseVtuStep(os.path.join(local_dir, "summary.csv"))

graphical_steps_np = Case_Options.summary_df["Vtu step"].to_numpy()
graphical_steps = [int(step) for step in graphical_steps_np]

# Initiate plotting class
time_range = None
time_interval = None
plot_axis = False
rotation_plus = 0.47 # rotation of the frame along the lon when making plot
ofile_list = ["slab1.py"]; require_base=True
PlotCaseRunThD = PLOT_CASE_RUN_THD(local_dir, time_range=time_range, run_visual=False,\
        time_interval=time_interval, visualization="paraview", step=graphical_steps, plot_axis=plot_axis, max_velocity=max_velocity,\
                rotation_plus=rotation_plus, ofile_list=ofile_list, require_base=require_base)

# Processing pyvista
PlotCaseRunThD.ProcessPyvista()

# Generate paraview script
for step in graphical_steps:
    # get trench center
    pvtu_step = step + int(Case_Options.options['INITIAL_ADAPTIVE_REFINEMENT']) 
    pyvista_outdir = os.path.join(local_dir, "pyvista_outputs", "%05d" % pvtu_step)
    trench_center = get_trench_position_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'])
    slab_depth = get_slab_depth_from_file(pyvista_outdir, pvtu_step, Case_Options.options['GEOMETRY'], float(Case_Options.options['OUTER_RADIUS']), "sp_lower")
    # generate paraview script
    addtional_options = {"TRENCH_CENTER": trench_center}
    PlotCaseRunThD.GenerateParaviewScript(ofile_list, addtional_options)
    # update value in sumamry
    Case_Options.SummaryCaseVtuStepUpdateValue("Slab depth", step, slab_depth)
    Case_Options.SummaryCaseVtuStepUpdateValue("Trench (center)", step, trench_center)

    break # debug

Case_Options.SummaryCaseVtuStepExport(os.path.join(local_dir, "summary.csv"))