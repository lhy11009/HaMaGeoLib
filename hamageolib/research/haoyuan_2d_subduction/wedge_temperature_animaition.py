
####################################
# preliminaries
# execute in terminal to set display as local:
#   export DISPLAY=:0
# execute the command:
#   python dtemp/wegde_temperature_animaition.py
####################################

import os, sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import subprocess
import datetime
from scipy.interpolate import UnivariateSpline

####################################
# options
# uncomment the options needed for operation
####################################
# case_name = "EBA_CDPT26/eba_cdpt_coh500_SA80.0_OA40.0_du2.3_sv19.5_gr12"
case_name = "EBA_CDPT_test_gr_1/eba_cdpt_PTWL_SA80.0_cd7.5_gr11"

time_interval = 5e5  # time interval in the frames

# default values (trivial case)
generate_visualization_paraview = False; generate_slab_morphology_results = False
generate_visualization_paraview_stepwise = False; generate_slab_morphology_plots = False
generate_slab_temperature_results = False; generate_slab_temperature_plots = False
generate_surface_heat_flux_plots = False
generate_animation = False; plot_without_heat_flux = True

# generate visualization of upper mantle
generate_visualization_paraview = False
# generate results of slab morphology, run this before later steps,
# as the following steps make use of the file generated
# generate_slab_morphology_results = True
# generate visualization of mantle wedge corner (by running paraview stepwise)
generate_visualization_paraview_stepwise = True
# generate plots of slab morphology
generate_slab_morphology_plots = True
# generate results of slab temperature
# generate_slab_temperature_results = True
# generate plots of slab temperature
generate_slab_temperature_plots = True
# # generate plots of surface heat flux
# generate_surface_heat_flux_plots = True
# generate animation
# turn on "plot_without_heat_flux" if heat flux data is missing
generate_animation = True; plot_without_heat_flux = True


####################################
# date
####################################
today_date = datetime.datetime.today().strftime("%Y-%m-%d") # Get today's date in YYYY-MM-DD format

####################################
# paths
####################################
local_TwoDSubduction_dir = "/mnt/lochz/ASPECT_DATA/TwoDSubduction"
ASPECT_LAB_DIR = os.environ['ASPECT_LAB_DIR']
if os.path.abspath(ASPECT_LAB_DIR) not in sys.path:
    sys.path.append(os.path.abspath(ASPECT_LAB_DIR))
RESULT_DIR = os.path.join(ASPECT_LAB_DIR, 'results')
os.makedirs(RESULT_DIR, exist_ok=True) # Ensure the directory exists
py_temp_dir = os.path.join(ASPECT_LAB_DIR, "py_temp_files")
os.makedirs(py_temp_dir, exist_ok=True) # Ensure the directory exists
py_temp_file = os.path.join(py_temp_dir, f"py_temp_{today_date}.sh")
HaMaGeoLibDir = "/home/lochy/ASPECT_PROJECT/HaMaGeoLib"
if os.path.abspath(HaMaGeoLibDir) not in sys.path:
    sys.path.append(os.path.abspath(HaMaGeoLibDir))

import shilofue.PlotCase as PlotCase
import shilofue.TwoDSubduction0.PlotCase as TwoDPlotCase
from shilofue.PlotHeatFlow import HeatFlowRetriveProfile

from hamageolib.research.haoyuan_2d_subduction.post_process import CASE_OPTIONS
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import SlabEnvelopRetrivePoints
from hamageolib.research.haoyuan_2d_subduction.workflow_scripts import \
  run_2d_subduction_visualization, plot_slab_morphology_series, plot_temperature_profiles_steps,\
  finalize_visualization_2d_12172024, finalize_visualization_2d_wedge_02122025, finalize_visualization_2d_wedge_small_03282025,\
    create_avi_from_images
from hamageolib.utils.file_reader  import read_aspect_header_file
import hamageolib.utils.plot_helper as plot_helper


####################################
# Start of scripts
####################################
# create path and classes

local_dir = os.path.join(local_TwoDSubduction_dir, case_name)
assert(os.path.isdir(local_dir))

CaseOptions = CASE_OPTIONS(local_dir)

resampled_df = CaseOptions.resample_visualization_df(time_interval)

# basic analysis 
# define a function for round values
round_values = lambda values: [round(x) for x in values]

# prepare the graphical_steps
graphical_steps = round_values(resampled_df["Time"].values / float(resampled_df.attrs["Time between graphical output"]))
slices = None

# types of plot to include
# The available types of plots are
plot_types = ["upper_mantle"]; rotation_plus = 0.47 # for plotting the upper mantle

# Additional fields in case of two crustal layers
# additional_fields = ["spcrust_up", "spcrust_low"]  # in case of two crustal layers
additional_fields = [] # in case of one crustal layer

config = {
    "RESULT_DIR": RESULT_DIR,                   # directory to write output .txt
    "py_temp_file": py_temp_file,          # where to write pvpython script
    "PlotCase": PlotCase,                               # your PlotCase module
    "TwoDPlotCase": TwoDPlotCase,                       # your TwoDPlotCase module

    # ---
    # Visualization and plotting options
    # True: save a complete result
    # False: prepare for figures in a paper
    # ---
    "plot_axis": False,
    "graphical_steps": graphical_steps,
    "slices": None,
    "max_velocity": -1.0,
    "plot_types": plot_types,
    "rotation_plus": rotation_plus,
    "additional_fields": [],
    "CaseOptions": CaseOptions
}

Visit_Options = run_2d_subduction_visualization(local_dir, config)

# generate visualization of upper mantle
if generate_visualization_paraview:
    # run subprocess
    command = ["pvpython", "%s/paraview_scripts/slab.py" % local_dir]
    subprocess.run(command)

# generate results of slab morphology
if generate_slab_morphology_results:

    # # run with interval 0.1Ma 
    command = ["python", "-m", "shilofue.TwoDSubduction0.VtkPp", "morph_case", "-i", local_dir, "-ti", "%.4e" % (1e5)]
    
    slab_morph_file = os.path.join(local_dir, 'vtk_outputs', 'slab_morph_t1.00e+05.txt')
    if os.path.isfile(slab_morph_file):
        # move old file
        subprocess.run(["mv", slab_morph_file, os.path.join(local_dir, 'vtk_outputs', 'slab_morph_t1.00e+05.old.txt')])

    subprocess.run(command)
    
    # run with interval 0.5Ma 
    command = ["python", "-m", "shilofue.TwoDSubduction0.VtkPp", "morph_case", "-i", local_dir, "-ti", "%.4e" % (5e5)]
    
    slab_morph_file = os.path.join(local_dir, 'vtk_outputs', 'slab_morph.txt')
    if os.path.isfile(slab_morph_file):
        # move old file
        subprocess.run(["mv", slab_morph_file, os.path.join(local_dir, 'vtk_outputs', 'slab_morph.old.txt')])
   
    subprocess.run(command)
  

# generate visualization of mantle wedge corner (by running paraview stepwise)
if generate_visualization_paraview_stepwise:

    # options in plot

    # step
    resampled_df = CaseOptions.resample_visualization_df(time_interval)
    graphical_steps = round_values(resampled_df["Time"].values / float(resampled_df.attrs["Time between graphical output"]))

    # Assert existence of the slab morphology file
    slab_morph_file = os.path.join(local_dir, 'vtk_outputs', 'slab_morph_t1.00e+05.txt')
    if not os.path.isfile(slab_morph_file):
        raise FileNotFoundError(f"The file '{slab_morph_file}' does not exist.")

    # Read simulation log data
    pd_data = read_aspect_header_file(slab_morph_file)
    pvtu_steps = pd_data["pvtu_step"]
    times = pd_data["time"]
    trenches = pd_data["trench"]
    shallow_trenches = pd_data["shallow trench"]
    slab_depthes = pd_data["slab depth"]
    sp_velocities = pd_data["subducting plate velocity"]
    ov_velocities = pd_data["overiding plate velocity"]

    # Make the directory to hold the scripts
    print("Generating paraview scripts")
    odir = os.path.join(local_dir, 'paraview_scripts', "stepwise")
    if not os.path.isdir(odir):
        os.mkdir(odir)

    # generate paraview scripts
    for i, graphical_step in enumerate(graphical_steps):        
        print("i: ", i)
        print("graphical_step:", graphical_step)
        Visit_Options.options['GRAPHICAL_STEPS'] = [graphical_step]

        _time = float(resampled_df.attrs["Time between graphical output"]) * graphical_step
        i1 = (np.abs(times - _time)).argmin() # index nearest to _time
        trench = shallow_trenches[i1]
        trench_deg = trench * 180.0 / np.pi

        paraview_script = os.path.join(ASPECT_LAB_DIR, 'paraview_scripts', 'TwoDSubduction', "slab.py")
        paraview_script_base = os.path.join(ASPECT_LAB_DIR, 'paraview_scripts', 'base.py')

        # first plot: mantle wedge bigger
        diff_angle = 1.5 # deg, to match the frame we use
        Visit_Options.options["PLOT_TYPES"] = ["wedge_bigger"]
        Visit_Options.options['ROTATION_ANGLE'] = 90.0 - trench_deg - diff_angle
        print("ROTATION_ANGLE:", Visit_Options.options['ROTATION_ANGLE'])
        Visit_Options.read_contents(paraview_script_base, paraview_script)  # combine these two scripts
        Visit_Options.substitute()
        ofile = os.path.join(odir, 'slab_%d.py' % (graphical_step))
        Visit_Options.save(ofile)
        print("Generate %s" % ofile)
        
        # second plot: mantle wedge small
        diff_angle = 1.0 # deg, to match the frame we use
        Visit_Options.options["PLOT_TYPES"] = ["wedge_02252025"]
        Visit_Options.options['ROTATION_ANGLE'] = 90.0 - trench_deg - diff_angle
        print("ROTATION_ANGLE:", Visit_Options.options['ROTATION_ANGLE'])
        Visit_Options.read_contents(paraview_script_base, paraview_script)  # combine these two scripts
        Visit_Options.substitute()
        ofile1 = os.path.join(odir, 'slab1_%d.py' % (graphical_step))
        Visit_Options.save(ofile1)
        print("Generate %s" % ofile1)

    # append command to bash script
    with open(py_temp_file, 'w') as fout:
        
        for i, graphical_step in enumerate(graphical_steps):
        
            ofile = os.path.join(odir, 'slab_%d.py' % (graphical_step))
            ofile1 = os.path.join(odir, 'slab1_%d.py' % (graphical_step))

            fout.write("# Run slab morphology analysis\n")
            fout.write("pvpython %s\n" % ofile)
            fout.write("pvpython %s\n" % ofile1)

    # run subprocess
    command = ["chmod", "+x", py_temp_file]
    subprocess.run(command)
    command = ["bash", os.path.abspath(py_temp_file)]
    subprocess.run(command)



# generate plots of slab morphology
if generate_slab_morphology_plots:
  
    config = {
    "resampled_df": resampled_df,
    "Visit_Options": Visit_Options,
    }

    plot_slab_morphology_series(local_dir, config)

# generate results of slab temperature
if generate_slab_temperature_results:
    
    command = ["python", "-m", "shilofue.TwoDSubduction0.VtkPp", "slab_temperature_case", "-i", local_dir, "-ti", "%.4e" % (time_interval), "-rp", "0"]
    subprocess.run(command)

# generate plots of slab temperature
# current mdd doesn't work for thin shear zone
if generate_slab_temperature_plots:

    config = {
    "Visit_Options": Visit_Options,
    "plot_helper": plot_helper,
    # "times": [3e6], # 1. one timestep
    "times": resampled_df["Time"].values, # 2. all timesteps with assigned interval
    "with_legend": False,
    "with_mdd": True
    }
  
    plot_temperature_profiles_steps(local_dir, config)

# generate surface heat flux 
if generate_surface_heat_flux_plots:
    
    from matplotlib import rcdefaults
    from matplotlib.ticker import MultipleLocator


    # options
    use_shallow_trench = True
    times = resampled_df["Time"].values # 1. use all time steps
    time_steps = resampled_df["Time step number"].values # use values
    Ro = 6371e3

    hf_dir = os.path.join(local_dir, "img", "heat_flux")
    if not os.path.isdir(hf_dir):
        os.mkdir(hf_dir)


    # Retrieve the default color cycle
    default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    # factors for scaling
    scaling_factor = 1.75  # scale factor of plot
    font_scaling_multiplier = 3.0 # extra scaling multiplier for font
    legend_font_scaling_multiplier = 0.75
    line_width_scaling_multiplier = 2.0 # extra scaling multiplier for lines
    x_lim = (3.0, -1.0) # degree
    x_tick_interval = 1.0  # tick interval along x
    y_lim = (0.0, 120.0)
    y_tick_interval = 20.0  # tick interval along y
    n_minor_ticks = 4  # number of minor ticks between two major ones

    # scale the matplotlib params
    plot_helper.scale_matplotlib_params(scaling_factor, font_scaling_multiplier=font_scaling_multiplier,\
                            legend_font_scaling_multiplier=legend_font_scaling_multiplier,
                            line_width_scaling_multiplier=line_width_scaling_multiplier)

    # Update font settings for compatibility with publishing tools like Illustrator.
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    for i, _time in enumerate(times):

        fig, ax = plt.subplots(figsize=(8*scaling_factor, 5*scaling_factor))
        
        ax1 = ax.twinx()
    
        # Retrieves the heat flow profile based on the given parameters.
        hfs_masked, Phis_masked, mdds, trench_40km, shallow_trench = HeatFlowRetriveProfile(local_dir, _time, time_steps[i], Visit_Options)
        if shallow_trench is None:
            raise ValueError("shallow trench value is not accessible for case %s" % local_dir)

        phi0, phi1 = Phis_masked[0], Phis_masked[-1]
        mdd1_depth, mdd2_depth = mdds[0], mdds[1]
        mdd_Ls = SlabEnvelopRetrivePoints(local_dir, _time, Visit_Options, np.array(mdds))

        # Fit a smoothing spline to the data (s=0 gives an exact fit, increase 's' to smooth more)
        # Then calculate the first and second derivative of heat flow with respect to longitude
        hfs_spline = UnivariateSpline(Phis_masked, hfs_masked, s=0)
        dhf_dphi = hfs_spline.derivative(n=1) # unit: mw / rad

        # migrate the value of phi to center on the trench
        if use_shallow_trench:
            trench = shallow_trench
        else:
            trench = trench_40km
        ax.plot((Phis_masked - trench) * 180.0 / np.pi, hfs_masked * 1000.0, color=default_colors[0])
        ax1.plot((Phis_masked - trench) * 180.0 / np.pi, dhf_dphi(Phis_masked)/Ro*1e6, linestyle='-.', color=default_colors[0]) # unit: 1e6 * mw / m^3
        print("mdd1_depth: %.2f, mdd_Ls[0] - trench: %.2f" % (mdd1_depth, (mdd_Ls[0] - trench) * 180.0 / np.pi)) # debug
        ax.axvline((mdd_Ls[0] - trench) * 180.0 / np.pi, linestyle="--", color=default_colors[0])
        # ax.axvline((mdd_Ls[1] - trench) * 180.0 / np.pi, linestyle="--")

        ax.set_xlabel("Location to trench (degree)")
        ax.set_ylabel("Heat flux (mw / m^2)")
        ax1.set_ylabel("Gradient (per km)")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax1.set_ylim((-1.5, 1.5))

        ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
        ax.xaxis.set_minor_locator(MultipleLocator(x_tick_interval/(n_minor_ticks+1)))
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
        ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval/(n_minor_ticks+1)))
        
        # ax.set_xlabel("Trench Distance (Km)")
        # ax.set_ylabel("Heat Flux (mW/m^2)")
        # ax1.set_ylabel("Heat Flux Derivative (mw/m^3)")

        ax.grid()

        ax.legend()
        fig.tight_layout()
        
        # Adjust spine thickness for this plot
        for spine in ax.spines.values():
            spine.set_linewidth(0.5 * scaling_factor * line_width_scaling_multiplier)

        # output file name    
        ofile_basename = "heat_flux_top"
            
        if use_shallow_trench:
            ofile_basename += "_shallow_trench"
        else:
            pass
    
        ofile_basename += "_t%.4e" % _time

        ofile = os.path.join(hf_dir, ofile_basename + ".pdf")
        ofile_png = os.path.join(hf_dir, ofile_basename + ".png")
        
        fig.savefig(ofile)
        fig.savefig(ofile_png)

        print("Save output file %s" % (ofile))
        print("Save output file %s" % (ofile_png))
    pass

# Run and generate animation
if generate_animation:

    from IPython.display import Image, display

    file_paths = []

    for i in range(len(resampled_df["Time"].values)):
        
        # todo_cbar
        # # debug run step 0
        # if i > 0:
        #     break
        
        _time = resampled_df["Time"].values[i]

        time_rounded = round(_time / float(resampled_df.attrs["Time between graphical output"]))\
              * float(resampled_df.attrs["Time between graphical output"]) 
        
        # Options to add time stamp
        add_time = True

        # Paths to output files
        ani_dir = os.path.join(local_dir, "img", "animation")
        if not os.path.isdir(ani_dir):
            os.mkdir(ani_dir)
        
        prep_file_dir = os.path.join(local_dir, "img", "animation", 'prep')
        if not os.path.isdir(prep_file_dir):
            os.mkdir(prep_file_dir)

        # Inputs
        # slab temperatrue
        # heat flux
        # slab morphology
        # colorbar for viscosity
        slab_temperature_png_file_name = os.path.join(local_dir, "img", "temperature", "slab_temperature_combined2_t%.4e.png" % (_time))
        if (not os.path.isfile(slab_temperature_png_file_name)):
            raise FileNotFoundError(f"PNG file doesn't exist: {slab_temperature_png_file_name}")
        
        heat_flux_png_file_name = os.path.join(local_dir, "img", "heat_flux", "heat_flux_top_shallow_trench_t%.4e.png" % (_time))
        if not plot_without_heat_flux:
            if (not os.path.isfile(heat_flux_png_file_name)):
                raise FileNotFoundError(f"PNG file doesn't exist: {heat_flux_png_file_name}")

        slab_morph_file_name = os.path.join(prep_file_dir, "combined_morphology_t%.4e.png" % _time)
        if (not os.path.isfile(slab_morph_file_name)):
            raise FileNotFoundError(f"PNG file doesn't exist: {slab_morph_file_name}")
        
        
        colorbar_viscosity_png_file = "/home/lochy/Documents/papers/documented_files/TwoDSubduction/color_tables/viscosity_18_24_0524_2025.png"
        if (not os.path.isfile(colorbar_viscosity_png_file)):
            raise FileNotFoundError(f"PNG file doesn't exist: {colorbar_viscosity_png_file}")
        
        colorbar_temperature_png_file = "/home/lochy/Documents/papers/documented_files/TwoDSubduction/color_tables/temperature_0524_2025.png"
        if (not os.path.isfile(colorbar_viscosity_png_file)):
            raise FileNotFoundError(f"PNG file doesn't exist: {colorbar_viscosity_png_file}")

        # Options

        base_name = "animation_wedge1"
        target_file_name = os.path.join(local_dir, "img", "pv_outputs", "viscosity_t%.4e.png" % time_rounded)

        if (not os.path.isfile(target_file_name)):
            raise FileNotFoundError(f"PNG file doesn't exist: {target_file_name}")

        # make the figure of upper mantle 
        frame_png_file_with_ticks = "/home/lochy/Documents/papers/documented_files/TwoDSubduction/upper_mantle_frame/upper_mantle_frame_12172024_trans_with_frame-01.png"
        target_file_name_framed = finalize_visualization_2d_12172024(local_dir, "viscosity", time_rounded, frame_png_file_with_ticks,\
                                                                     add_time=False, canvas_size=((996, 625)))

        # make the figure of mantle wedge
        frame_png_file_with_ticks = "/home/lochy/Documents/papers/documented_files/TwoDSubduction/upper_mantle_frame/wedge_frame_11272024_trans_with_trench_frame.png"
        target_file_name_framed_1 = finalize_visualization_2d_wedge_02122025(local_dir, "viscosity_wedge_bigger", time_rounded, frame_png_file_with_ticks, add_time=False)
        
        # make the figure of mantle wedge small
        frame_png_file_with_ticks = "/home/lochy/Documents/papers/documented_files/TwoDSubduction/upper_mantle_frame/wedge_small_frame_03272025_trans_with_frame-01.png"
        target_file_name_framed_2 = finalize_visualization_2d_wedge_small_03282025(local_dir, "viscosity_wedge_small", time_rounded, frame_png_file_with_ticks, add_time=False)


        # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
        canvas_size = (1200, 1700) 
        output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (base_name, _time))
        if os.path.isfile(output_image_file):
            # Remove existing output image to ensure a clean overlay
            os.remove(output_image_file)
        if plot_without_heat_flux:
            # List of image file paths to overlay
            image_files=[target_file_name_framed, target_file_name_framed_1, target_file_name_framed_2, slab_temperature_png_file_name, slab_morph_file_name,  colorbar_viscosity_png_file, colorbar_temperature_png_file]
            # Positions of each image on the canvas
            image_positions=[(0, 100), (600, 100), (600, 500),(0, 1000), (0, 1350), (100, 925), (650, 925)]
            # Optional cropping regions for the images
            cropping_regions=[None, None, None, None, None, None, None]
            # Scaling factors for resizing the images
            image_scale_factors=[0.6, 0.5, 0.5, 0.4, 0.4, 1.0, 1.0]
        else:
            # List of image file paths to overlay
            image_files=[target_file_name_framed, target_file_name_framed_1, target_file_name_framed_2, slab_temperature_png_file_name, heat_flux_png_file_name, slab_morph_file_name,  colorbar_viscosity_png_file, colorbar_temperature_png_file]
            # Positions of each image on the canvas
            image_positions=[(0, 100), (600, 100), (600, 500),(0, 1000), (500, 1000), (0, 1350), (100, 925), (650, 925)]
            # Optional cropping regions for the images
            cropping_regions=[None, None, None, None, None, None, None, None]
            # Scaling factors for resizing the images
            image_scale_factors=[0.6, 0.5, 0.5, 0.4, 0.4, 0.4, 1.0, 1.0]
            
        plot_helper.overlay_images_on_blank_canvas(
            canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
            image_files=image_files,  
            image_positions=image_positions,
            cropping_regions=cropping_regions,
            image_scale_factors=image_scale_factors,
            output_image_file=output_image_file  # Path to save the final combined image
        )

        

        # Example Usage, add_text_to_image
        # image_path = "your_image.png"  # Replace with the path to your PNG file
        # output_path = "output_image_with_text.png"  # Path to save the output image
        if add_time:
            text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
            position = (25, 25)  # Replace with the desired text position (x, y)
            font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
            font_size = 56

            plot_helper.add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)


        # display(Image(filename=output_image_file))
        file_paths.append(output_image_file)

    # create animation
    output_file = os.path.join(local_dir, "img", "animation", "%s.avi" % base_name)
    create_avi_from_images(file_paths, output_file, 1)