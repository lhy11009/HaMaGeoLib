# ==============================================================================
# MIT License
# 
# Copyright (c) 2025 Haoyuan Li
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ==============================================================================

"""
File: workflow_scripts.py

Author: Haoyuan Li

Description:
    This module contains a collection of utility scripts and procedural functions
    used to facilitate and automate the workflow of 2D subduction geodynamic modeling.

    It is part of the research workspace located at:
    hamageolib/research/haoyuan_2d_subduction/

    Contents may include:
    - File preparation and post-processing routines
    - Automation of simulation runs or result collation
    - Workflow helpers for batch jobs or parameter sweeps
    - Custom data transformations tied to specific modeling experiments

Note:
    These scripts are research-use-focused and may not follow full package-wide
    standards. Use within the context of research reproducibility and customization.
"""
# Modules ..
import os
import numpy as np
import re

# matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

# scipy
from scipy.interpolate import interp1d

# Utility functions
from ...utils.file_reader  import read_aspect_header_file
from ...utils.plot_helper import convert_eps_to_pdf, extract_image_by_size, overlay_images_on_blank_canvas,\
    add_text_to_image, scale_matplotlib_params
from ...utils.case_options  import parse_log_file_for_solver_info

# Core functions
from ...core.melt import ROCK_MELTING  # Replace with actual import path


def plot_slab_morphology_series(local_dir, config):
    '''
    Generate slab morphology plots for a series of time steps from a resampled DataFrame.

    Parameters
    ----------
    local_dir : str
        Path to the case directory.
    config : dict
        Dictionary of configuration options. Must include:
            - resampled_df : pd.DataFrame
            - Visit_Options : object
            - plot_helper : module with scale_matplotlib_params(...)
            - SlabAnalysisPlotter : class to instantiate the plotter
    '''

    # Fixed plotting parameters
    scaling_factor = 1.75
    font_scaling_multiplier = 3.0
    legend_font_scaling_multiplier = 0.75
    line_width_scaling_multiplier = 2.0
    x_tick_interval = 1.0
    y_tick_interval = 20.0
    n_minor_ticks = 4

    # Extract from config
    resampled_df = config["resampled_df"]
    Visit_Options = config["Visit_Options"]

    # Get default matplotlib color cycle
    default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    # Style settings
    scale_matplotlib_params(
        scaling_factor,
        font_scaling_multiplier=font_scaling_multiplier,
        legend_font_scaling_multiplier=legend_font_scaling_multiplier,
        line_width_scaling_multiplier=line_width_scaling_multiplier
    )
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    # Prepare output folders
    ani_dir = os.path.join(local_dir, "img", "animation")
    os.makedirs(ani_dir, exist_ok=True)
    prep_file_dir = os.path.join(ani_dir, "prep")
    os.makedirs(prep_file_dir, exist_ok=True)

    # Initialize plotter and set options
    plotter = SlabAnalysisPlotter()
    plotter.plot_options.update({
        "time_range": (0, 15e6),
        "time_major_tick_interval": 5e6,
        "trench_range": (-500, 100),
        "trench_major_tick_interval": 100,
        "depth_range": (0, 1200),
        "depth_major_tick_interval": 200,
        "velocity_range": (-15, 15),
        "velocity_major_tick_interval": 5,
    })

    for _time in resampled_df["Time"].values:
        fig, axes = plt.subplots(1, 2, figsize=(15 * scaling_factor, 5 * scaling_factor))
        
        plotter.plot_slab_analysis(
            axes, local_dir, Visit_Options,
            color=default_colors[0],
            include_additional_label=True,
            time=_time
        )

        # Adjust spine thickness
        for ax in fig.get_axes():
            ax.grid()
            for spine in ax.spines.values():
                spine.set_linewidth(0.5 * scaling_factor * line_width_scaling_multiplier)

        fig.tight_layout()

        ofile = os.path.join(prep_file_dir, f"combined_morphology_t{_time:.4e}.png")
        fig.savefig(ofile)
        print(f"Saved output file: {ofile}")

    mpl.rcParams.update(mpl.rcParamsDefault)


def plot_temperature_profiles_steps(local_dir, config):
    '''
    Plot slab surface and Moho temperature profiles at specific times, with solidus curves.

    Parameters
    ----------
    local_dir : str
        Path to the simulation case directory.
    config : dict
        Must contain:
            - Visit_Options : object
            - plot_helper : module or object with .scale_matplotlib_params(...)
            - times : list of float
            - with_legend : bool
    '''

    # Fixed parameters (now internal to the function)
    max_depth = 150e3
    scaling_factor = 1.6
    font_scaling_multiplier = 3.0
    legend_font_scaling_multiplier = 0.5
    line_width_scaling_multiplier = 2.0
    x_tick_interval = 400.0
    y_tick_interval = 50.0
    n_minor_ticks = 4

    # From config
    times = config["times"]
    with_legend = config["with_legend"]
    Visit_Options = config["Visit_Options"]
    with_mdd = config.get("with_mdd", True)

    default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
    scale_matplotlib_params(
        scaling_factor,
        font_scaling_multiplier=font_scaling_multiplier,
        legend_font_scaling_multiplier=legend_font_scaling_multiplier,
        line_width_scaling_multiplier=line_width_scaling_multiplier
    )
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'pdf.fonttype': 42,
        'ps.fonttype': 42
    })

    slab_morph_file = os.path.join(local_dir, 'vtk_outputs', 'slab_morph.txt')
    if not os.path.isfile(slab_morph_file):
        raise FileNotFoundError(f"The file '{slab_morph_file}' does not exist.")
    
    pd_data = read_aspect_header_file(slab_morph_file)
    mtimes = pd_data["time"]
    mdd1s = pd_data["mechanical decoupling depth1"]
    mdd2s = pd_data["mechanical decoupling depth2"]

    for time_inspect in times:
        if with_legend:
            fig, axes = plt.subplots(nrows=2, figsize=(8 * scaling_factor, 7 * scaling_factor), tight_layout=True)
            ax, ax1 = axes
        else:
            fig, ax = plt.subplots(figsize=(8 * scaling_factor, 5 * scaling_factor), tight_layout=True)

        lines = []

        idx_m = np.argmin(np.abs(mtimes - time_inspect))
        mdd1 = mdd1s[idx_m]
        mdd2 = mdd2s[idx_m]
        print("mdd1 = %.1f, mdd2 = %.1f" % (mdd1, mdd2))

        color0 = default_colors[0]
        _, _, vtu_step = Visit_Options.get_timestep_by_time(time_inspect)

        slab_T_file_path = os.path.join(local_dir, "vtk_outputs", "temperature", f"slab_temperature_{vtu_step:05d}.txt")
        if not os.access(slab_T_file_path, os.R_OK):
            raise FileNotFoundError(f"The file at '{slab_T_file_path}' is not accessible or does not exist.")
        
        pd_data = read_aspect_header_file(slab_T_file_path)

        Tbot_func = interp1d(pd_data["depth"], pd_data["Tbot"], assume_sorted=True) 
        Ttop_func = interp1d(pd_data["depth"], pd_data["Ttop"], assume_sorted=True) 
        p_depths = np.arange(pd_data["depth"][0], max_depth, 1e3)
        p_Tbots = Tbot_func(p_depths)
        p_Ttops = Ttop_func(p_depths)

        line0 = ax.plot(p_Ttops - 273.15, p_depths/1e3, label=f"Surface T, {time_inspect/1e6:.1f} Ma", color=color0, linewidth=2)
        line1 = ax.plot(p_Tbots - 273.15, p_depths/1e3, label="Moho T", color=color0, linewidth=4)

        if with_mdd:
            ax.hlines(mdd1/1e3, 0.0, 3000.0, linestyles="dashdot", color=color0)
            ax.hlines(mdd2/1e3, 0.0, 3000.0, linestyles="dotted", color=color0)

        lines += line0 + line1

        rock_melting = ROCK_MELTING()
        pressure_ranges = {
            "dry_peridotite": np.linspace(0, 10e9, 100),
            "water_saturated_peridotite_low_pressure": np.linspace(0, 6e9, 100),
            "water_saturated_peridotite_high_pressure": np.linspace(6e9, 12e9, 100),
            "eclogite": np.linspace(3e9, 7.5e9, 100),
            "peridotite_aH2O_0.1": np.linspace(0, 10e9, 100),
            "peridotite_aH2O_0.3": np.linspace(0, 10e9, 100),
            "peridotite_aH2O_0.5": np.linspace(0, 10e9, 100),
            "peridotite_aH2O_0.7": np.linspace(0, 10e9, 100),
            "basalt_aH2O_0_3": np.linspace(0.078e9, 3.665e9, 100),
            "basalt_aH2O_0_8": np.linspace(0.078e9, 3.665e9, 100),
            "basalt_aH2O_1_3": np.linspace(0.078e9, 3.665e9, 100),
            "basalt_aH2O_1_8": np.linspace(0.078e9, 3.665e9, 100),
        }
        styles = {
            "dry_peridotite": ("k", "-"),
            "water_saturated_peridotite_low_pressure": (default_colors[3], "--"),
            "water_saturated_peridotite_high_pressure": (default_colors[3], "--"),
            "eclogite": ("r", "-"),
            "peridotite_aH2O_0.1": ("c", "-"),
            "peridotite_aH2O_0.3": ("c", "--"),
            "peridotite_aH2O_0.5": ("g", "-"),
            "peridotite_aH2O_0.7": (default_colors[4], "--"),
            "basalt_aH2O_0_3": (default_colors[1], "--"),
            "basalt_aH2O_0_8": ("m", "--"),
            "basalt_aH2O_1_3": ("y", "--"),
            "basalt_aH2O_1_8": ("y", "--"),
        }
        solidus_list = [
            "basalt_aH2O_0_3", "basalt_aH2O_1_8",
            "water_saturated_peridotite_low_pressure",
            "water_saturated_peridotite_high_pressure",
            "peridotite_aH2O_0.7"
        ]

        for name in solidus_list:
            func = rock_melting.solidus_data[name]
            P_Pa = pressure_ranges[name]
            T_K = func(P_Pa)
            depth_km = P_Pa / 33e6
            color, linestyle = styles[name]
            line = ax.plot(T_K - 273.15, depth_km, color=color, linestyle=linestyle, label=name.replace("_", " ").title())
            lines += line

        ax.set_xlim([0.0, 1200.0])
        ax.set_ylim([max_depth/1e3, 0])
        ax.set_xlabel("T (°C)")
        ax.set_ylabel("z (km)")
        ax.xaxis.set_major_locator(MultipleLocator(x_tick_interval))
        ax.xaxis.set_minor_locator(MultipleLocator(x_tick_interval / (n_minor_ticks + 1)))
        ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
        ax.yaxis.set_minor_locator(MultipleLocator(y_tick_interval / (n_minor_ticks + 1)))
        ax.grid()

        for ax_sub in fig.get_axes():
            for spine in ax_sub.spines.values():
                spine.set_linewidth(0.5 * scaling_factor * line_width_scaling_multiplier)

        if with_legend:
            ax1.legend(handles=lines, loc='center')
            ax1.axis('off')

        o_path = os.path.join(local_dir, "img", "temperature", f"slab_temperature_combined2_t{time_inspect:.4e}.pdf")
        o_path_png = os.path.join(local_dir, "img", "temperature", f"slab_temperature_combined2_t{time_inspect:.4e}.png")
        os.makedirs(os.path.dirname(o_path), exist_ok=True)
        fig.savefig(o_path)
        print("Saved figure:", o_path)
        fig.savefig(o_path_png)
        print("Saved figure:", o_path_png)

    mpl.rcParams.update(mpl.rcParamsDefault)


def finalize_visualization_2d_wedge_small_03282025(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)

    canvas_size = kwargs.get("canvas_size", (1243, 755))

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    #If pdf is not provide, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert(os.path.isfile(pdf_file))

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(-100, 15), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[1.0, 1.0],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    # image_path = "your_image.png"  # Replace with the path to your PNG file
    # output_path = "output_image_with_text.png"  # Path to save the output image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file

def run_2d_subduction_visualization(local_dir, config):
    '''
    Run the 2D subduction post-processing and visualization pipeline.

    Parameters
    ----------
    local_dir : str
        Path to the case folder containing simulation output and scripts.
    config : dict
        Dictionary of configuration options. Must include:
        - RESULT_DIR : str
        - py_temp_file : str
        - PlotCase : module or object with .PlotCaseRun()
        - TwoDPlotCase : module or object with .PlotCaseRun()
        - plot_axis : bool
        - graphical_steps : list[int]
        - slices : int or None
        - max_velocity : float
        - plot_types : list[str]
        - rotation_plus : float
        - additional_fields : list[str]

    Returns
    -------
    Visit_Options : Any
        The result returned from TwoDPlotCase.PlotCaseRun
    '''

    RESULT_DIR = config["RESULT_DIR"]
    py_temp_file = config["py_temp_file"]
    PlotCaseRun_base = config["PlotCaseRun_base"]
    PlotCaseRun_project = config["PlotCaseRun_project"]
    plot_axis = config["plot_axis"]
    graphical_steps = config["graphical_steps"]
    slices = config["slices"]
    max_velocity = config["max_velocity"]
    plot_types = config["plot_types"]
    rotation_plus = config["rotation_plus"]
    additional_fields = config["additional_fields"]
    CaseOptions = config["CaseOptions"]
    plot_slab_interface_points = config["plot_slab_interface_points"]
    plot_mdd_extract_profile_points = config["plot_mdd_extract_profile_points"]
    

    # Determine ASPECT major version
    if re.match("^2", CaseOptions.aspect_version):
        major_version = 2
    elif re.match("^3", CaseOptions.aspect_version):
        major_version = 3
    else:
        raise ValueError("Unrecognized ASPECT version: %s" % CaseOptions.aspect_version)

    # Parse solver output
    file_path = os.path.join(local_dir, "output", "log.txt")
    assert os.path.isfile(file_path), f"Missing log file: {file_path}"
    output_path = os.path.join(RESULT_DIR, 'run_time_output_newton')
    parse_log_file_for_solver_info(file_path, output_path, major_version=major_version)

    # Basic runtime plot
    if PlotCaseRun_base is not None:
        PlotCaseRun_base(
            local_dir,
            time_range=None,
            run_visual=False,
            time_interval=None,
            visualization="paraview",
            step=graphical_steps
        )
        plt.close()

    # todo_thin
    # Full visual plotting
    Visit_Options = PlotCaseRun_project(
        local_dir,
        time_range=None,
        run_visual=False,
        time_interval=None,
        visualization="paraview",
        step=graphical_steps,
        plot_axis=plot_axis,
        max_velocity=max_velocity,
        plot_types=plot_types,
        rotation_plus=rotation_plus,
        additional_fields=additional_fields,
        slices=slices,
        plot_slab_interface_points=plot_slab_interface_points,
        plot_mdd_extract_profile_points=plot_mdd_extract_profile_points
    )
    plt.close()

    # Append paraview script
    paraview_script = os.path.join(local_dir, "paraview_scripts", "slab.py")
    assert os.path.isfile(paraview_script), f"Missing paraview script: {paraview_script}"
    with open(py_temp_file, 'a') as fout:
        fout.write("# Run paraview script\n")
        fout.write("pvpython %s\n" % paraview_script)

    return Visit_Options


def create_avi_from_images(file_paths, output_file, frame_rate=30):
    """
    Converts a list of .png files into an .avi animation.

    Args:
        file_paths (list): List of file paths to .png images.
        output_file (str): The output .avi file path.
        frame_rate (int): The frame rate of the output animation. Defaults to 30.

    Returns:
        None
    """
    import cv2

    # Check if the file_paths list is empty
    if not file_paths:
        raise ValueError("The file_paths list is empty.")

    # Read the first image to get the frame size
    first_image = cv2.imread(file_paths[0])
    if first_image is None:
        raise ValueError(f"Cannot read the first image: {file_paths[0]}")

    height, width, _ = first_image.shape
    frame_size = (width, height)

    # Define the codec and create the VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    # Write each image to the video
    for file_path in file_paths:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Warning: Skipping {file_path} (cannot read file).")
            continue
        resized_image = cv2.resize(image, frame_size)  # Ensure consistent size
        out.write(resized_image)

    # Release the VideoWriter object
    out.release()
    print(f"AVI animation created: {output_file}")


class SlabAnalysisPlotter:
    """
    A class for plotting slab analysis results with default plotting options.

    Attributes:
    - plot_options (dict): Default options for plotting, including time, trench, depth, and velocity ranges and intervals.
    """

    def __init__(self):
        """
        Initialize the SlabAnalysisPlotter with default plot options.
        """
        self.plot_options = {
            "time_range": (0, 15e6),
            "time_major_tick_interval": 5e6,
            "trench_range": (-500, 100),
            "trench_major_tick_interval": 100,
            "depth_range": (0, 1200),
            "depth_major_tick_interval": 200,
            "velocity_range": (-5, 10),
            "velocity_major_tick_interval": 5,
            "dip_range" : (30, 90),
            "dip_major_tick_interval": 10.0
        }

    def reset_plot_option(self, key, value):
        """
        Reset a specific plot option.

        Parameters:
        - key: str, The key in the plot_options dictionary to modify.
        - value: Any, The new value to set for the given key.

        Returns:
        - None
        """
        if key in self.plot_options:
            self.plot_options[key] = value
        else:
            raise KeyError(f"Invalid key '{key}'. Valid keys are: {list(self.plot_options.keys())}")

    def plot_slab_analysis(self, axes, local_comp_dir, Visit_Options, **kwargs):
        """
        Plot slab analysis results using the class's plot_options.

        Parameters:
        - axes: List of matplotlib Axes, length must be 2, with subplots for plotting.
        - base_comp_dir: str, Base directory of the computation case.
        - case_comp_name: str, Name of the specific case computation directory.
        - Visit_Options: Object, Configuration options for the simulation (including geometry details).

        Returns:
        - None
        """
        # Get additional options
        _color = kwargs.get("color", 'b')
        include_additional_label = kwargs.get("include_additional_label", False)
        _name = kwargs.get("name", None)
        marker_time = kwargs.get("time", None)

        # Assert that axes has the correct structure
        assert len(axes) == 2, "The 'axes' input must contain exactly two subplots."
        
        # Extract geometry and set outer radius
        geometry = Visit_Options.options["GEOMETRY"]
        if geometry == 'chunk':
            Ro = float(Visit_Options.options["OUTER_RADIUS"])
        else:
            Ro = -1.0  # Default value to indicate invalid geometry

        # Assert existence of the slab morphology file
        slab_morph_file = os.path.join(local_comp_dir, 'vtk_outputs', 'slab_morph.txt')
        if not os.path.isfile(slab_morph_file):
            raise FileNotFoundError(f"The file '{slab_morph_file}' does not exist.")

        # Read simulation log data
        pd_data = read_aspect_header_file(slab_morph_file)
        pvtu_steps = pd_data["pvtu_step"]
        times = pd_data["time"]
        trenches = pd_data["trench"]
        slab_depthes = pd_data["slab depth"]
        sp_velocities = pd_data["subducting plate velocity"]
        ov_velocities = pd_data["overiding plate velocity"]
        dip_100s = pd_data["100km dip"]

        # Extract plot options
        time_range = self.plot_options['time_range']
        time_major_tick_interval = self.plot_options['time_major_tick_interval']
        time_minor_tick_interval = time_major_tick_interval / 5.0
        trench_range = self.plot_options['trench_range']
        trench_major_tick_interval = self.plot_options['trench_major_tick_interval']
        trench_minor_tick_interval = trench_major_tick_interval / 5.0
        depth_range = self.plot_options['depth_range']
        depth_major_tick_interval = self.plot_options['depth_major_tick_interval']
        depth_minor_tick_interval = depth_major_tick_interval / 5.0
        velocity_range = self.plot_options['velocity_range']
        velocity_major_tick_interval = self.plot_options['velocity_major_tick_interval']
        velocity_minor_tick_interval = velocity_major_tick_interval / 5.0
        dip_range = self.plot_options["dip_range"]
        dip_major_tick_interval = self.plot_options["dip_major_tick_interval"]
        dip_minor_tick_interval = dip_major_tick_interval / 5.0

        # Calculate trench migration length
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro
        elif geometry == "box":
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError("Invalid geometry specified.")

        # Compute velocities
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)

        # Plot "trench position" vs "time" on primary y-axis
        _label = 'Trench Position'
        if _name is not None:
            _label += f" ({_name})"
        line1, = axes[0].plot(times / 1e6, trenches_migration_length / 1e3, label=_label, linestyle='-', color=_color)
        axes[0].set_xlim(time_range[0] / 1e6, time_range[1] / 1e6)
        axes[0].set_xticks(np.arange(0, time_range[1] / 1e6 + 1, time_major_tick_interval / 1e6))
        axes[0].xaxis.set_minor_locator(MultipleLocator(time_minor_tick_interval / 1e6))
        axes[0].set_ylim(trench_range[0], trench_range[1])
        axes[0].set_yticks(np.arange(trench_range[0], trench_range[1] + 1, trench_major_tick_interval))
        axes[0].yaxis.set_minor_locator(MultipleLocator(trench_minor_tick_interval))
        axes[0].set_xlabel("Time (Ma)")
        axes[0].set_ylabel("Trench Position (km)")
        axes[0].tick_params(axis='y')
        # axes[0].grid()

        if marker_time is not None:
            axes[0].vlines(marker_time/1e6, trench_range[0], trench_range[1], color="gray", linestyle="dotted")

        # Plot "slab depth" vs "time" on twin y-axis
        ax_twin = axes[0].twinx()
        if include_additional_label:
            _label = 'Slab Depth'
        else:
            _label = None
        line2, = ax_twin.plot(times / 1e6, slab_depthes / 1e3, label=_label, linestyle='-.', color=_color)
        ax_twin.set_ylim(depth_range[0], depth_range[1])
        ax_twin.set_yticks(np.arange(depth_range[0], depth_range[1] + 1, depth_major_tick_interval))
        ax_twin.yaxis.set_minor_locator(MultipleLocator(depth_minor_tick_interval))
        ax_twin.set_ylabel("Slab Depth (km)")
        ax_twin.tick_params(axis='y')

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        # axes[0].legend(lines, labels, loc="lower left") 

        # Plot velocities vs time on the second subplot
        if include_additional_label:
            _label = 'Trench Velocity'
        else:
            _label = None
        line2_0, = axes[1].plot(times / 1e6, trench_velocities * 1e2, label=_label, linestyle='-', color=_color)
        if include_additional_label:
            _label = 'Subducting Plate Velocity'
        else:
            _label = None
        line2_1, = axes[1].plot(times / 1e6, sp_velocities * 1e2, label=_label, linestyle='--', color=_color)
        
        if marker_time is not None:
            axes[1].vlines(marker_time/1e6, velocity_range[0], velocity_range[1], color="gray", linestyle="dotted")
        
        axes[1].set_xlim(time_range[0] / 1e6, time_range[1] / 1e6)
        axes[1].set_xticks(np.arange(0, time_range[1] / 1e6 + 1, time_major_tick_interval / 1e6))
        axes[1].xaxis.set_minor_locator(MultipleLocator(time_minor_tick_interval / 1e6))
        axes[1].set_ylim(velocity_range[0], velocity_range[1])
        axes[1].set_yticks(np.arange(velocity_range[0], velocity_range[1] + 1, velocity_major_tick_interval))
        axes[1].yaxis.set_minor_locator(MultipleLocator(velocity_minor_tick_interval))
        axes[1].set_xlabel("Time (Ma)")
        axes[1].set_ylabel("Velocity (cm/yr)")
        # axes[1].legend()
        # axes[1].grid()

        ax1_twin = axes[1].twinx()
        ax1_twin.set_ylim(dip_range[0], dip_range[1])
        line2_2, = ax1_twin.plot(times / 1e6, dip_100s * 180.0 / np.pi, label="Dip Angle", linestyle='dotted', color=_color)
        ax1_twin.set_yticks(np.arange(dip_range[0], dip_range[1] + 1, dip_major_tick_interval))
        ax1_twin.yaxis.set_minor_locator(MultipleLocator(dip_minor_tick_interval))
        ax1_twin.set_ylabel("Dip Angle (degree)")
        ax1_twin.tick_params(axis='y')

        lines = [line2_0, line2_1, line2_2]
        labels = [line.get_label() for line in lines]
        # axes[1].legend(lines, labels, loc="upper right") 

    def plot_slab_mdds(self, axes, local_comp_dir, Visit_Options, **kwargs):
        """
        Plot slab analysis results using the class's plot_options.

        Parameters:
        - axes: List of matplotlib Axes, length must be 2, with subplots for plotting.
        - base_comp_dir: str, Base directory of the computation case.
        - case_comp_name: str, Name of the specific case computation directory.
        - Visit_Options: Object, Configuration options for the simulation (including geometry details).

        Returns:
        - None
        """
        # Get additional options
        _color = kwargs.get("color", 'b')
        include_additional_label = kwargs.get("include_additional_label", False)
        _name = kwargs.get("name", None)
        marker_time = kwargs.get("time", None)

        # Assert that axes has the correct structure
        # assert len(axes) == 2, "The 'axes' input must contain exactly two subplots."
        
        # Extract geometry and set outer radius
        geometry = Visit_Options.options["GEOMETRY"]
        if geometry == 'chunk':
            Ro = float(Visit_Options.options["OUTER_RADIUS"])
        else:
            Ro = -1.0  # Default value to indicate invalid geometry

        # Assert existence of the slab morphology file
        # slab_morph_file = os.path.join(local_comp_dir, 'vtk_outputs', 'slab_morph.txt')
        slab_morph_file = os.path.join(local_comp_dir, 'vtk_outputs', 'slab_morph_t1.00e+05.txt')
        if not os.path.isfile(slab_morph_file):
            raise FileNotFoundError(f"The file '{slab_morph_file}' does not exist.")

        # Read simulation log data
        pd_data = read_aspect_header_file(slab_morph_file)
        pvtu_steps = pd_data["pvtu_step"]
        times = pd_data["time"]
        trenches = pd_data["trench"]
        slab_depthes = pd_data["slab depth"]
        sp_velocities = pd_data["subducting plate velocity"]
        ov_velocities = pd_data["overiding plate velocity"]
        dip_100s = pd_data["100km dip"]
        mdd1s = pd_data["mechanical decoupling depth1"]
        mdd2s = pd_data["mechanical decoupling depth2"]

        # Extract plot options
        time_range = self.plot_options['time_range']
        time_major_tick_interval = self.plot_options['time_major_tick_interval']
        time_minor_tick_interval = time_major_tick_interval / 5.0
        trench_range = self.plot_options['trench_range']
        trench_major_tick_interval = self.plot_options['trench_major_tick_interval']
        trench_minor_tick_interval = trench_major_tick_interval / 5.0
        depth_range = self.plot_options['depth_range']
        depth_major_tick_interval = self.plot_options['depth_major_tick_interval']
        depth_minor_tick_interval = depth_major_tick_interval / 5.0
        velocity_range = self.plot_options['velocity_range']
        velocity_major_tick_interval = self.plot_options['velocity_major_tick_interval']
        velocity_minor_tick_interval = velocity_major_tick_interval / 5.0
        dip_range = (0, 90.0)
        mdd_range = (0, 150)
        mdd_major_tick_interval = 25.0
        mdd_minor_tick_interval = mdd_major_tick_interval / 5.0

        # Calculate trench migration length
        if geometry == "chunk":
            trenches_migration_length = (trenches - trenches[0]) * Ro
        elif geometry == "box":
            trenches_migration_length = trenches - trenches[0]
        else:
            raise ValueError("Invalid geometry specified.")

        # Compute velocities
        trench_velocities = np.gradient(trenches_migration_length, times)
        sink_velocities = np.gradient(slab_depthes, times)

        # Plot "trench position" vs "time" on primary y-axis
        _label = 'Trench Position'
        if _name is not None:
            _label += f" ({_name})"
        ax0 = axes[0,0]
        line1, = ax0.plot(times / 1e6, trenches_migration_length / 1e3, label=_label, linestyle='-', color=_color)
        ax0.set_xlim(time_range[0] / 1e6, time_range[1] / 1e6)
        ax0.set_xticks(np.arange(0, time_range[1] / 1e6 + 1, time_major_tick_interval / 1e6))
        ax0.xaxis.set_minor_locator(MultipleLocator(time_minor_tick_interval / 1e6))
        ax0.set_ylim(trench_range[0], trench_range[1])
        ax0.set_yticks(np.arange(trench_range[0], trench_range[1] + 1, trench_major_tick_interval))
        ax0.yaxis.set_minor_locator(MultipleLocator(trench_minor_tick_interval))
        ax0.set_xlabel("Time (Ma)")
        ax0.set_ylabel("Trench Position (km)")
        ax0.tick_params(axis='y')
        # ax0].grid()

        if marker_time is not None:
            ax0.vlines(marker_time/1e6, trench_range[0], trench_range[1], color="gray", linestyle="dotted")

        # Plot "slab depth" vs "time" on twin y-axis
        ax_twin = ax0.twinx()
        line2, = ax_twin.plot(times / 1e6, slab_depthes / 1e3, linestyle='-.', color=_color)
        ax_twin.set_ylim(depth_range[0], depth_range[1])
        ax_twin.set_yticks(np.arange(depth_range[0], depth_range[1] + 1, depth_major_tick_interval))
        ax_twin.yaxis.set_minor_locator(MultipleLocator(depth_minor_tick_interval))
        ax_twin.set_ylabel("Slab Depth (km)")
        ax_twin.tick_params(axis='y')

        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        # axes[0].legend(lines, labels, loc="lower left") 

        ax1 = axes[0, 1]
        
        # line2_2, = ax1.plot(times / 1e6, dip_100s * 180.0 / np.pi, label="Dip Angle", linestyle='dotted', color=_color)
        if include_additional_label:
            label1 = "Decoupling Depth 1"; label2 = "Decoupling Depth 2"
        else:
            label1 = None; label2 = None
        line2_2, = ax1.plot(times / 1e6, mdd1s/1e3, label=label1, linewidth=4, color=_color)
        line2_3, = ax1.plot(times / 1e6, mdd2s/1e3, label=label2, linewidth=2, color=_color)
        ax1.set_yticks(np.arange(mdd_range[0], mdd_range[1] + 1, mdd_major_tick_interval))
        ax1.yaxis.set_minor_locator(MultipleLocator(mdd_minor_tick_interval))
        ax1.set_ylabel("Decoupling depth (km)")
        ax1.tick_params(axis='y')
        
        ax1.set_ylim(mdd_range[0], mdd_range[1])

        lines = [line2_2, line2_3]
        labels = [line.get_label() for line in lines]

        ax2 = axes[1, 0]
        # Plot velocities vs time on the second subplot
        if include_additional_label:
            _label = 'Convergence Rate'
        else:
            _label = None
        line2_0, = ax2.plot(times / 1e6, (sp_velocities - trench_velocities) * 1e2, label=_label, linestyle='-', color=_color)
        
        if marker_time is not None:
            ax2.vlines(marker_time/1e6, velocity_range[0], velocity_range[1], color="gray", linestyle="dotted")
        
        ax2.set_xlim(time_range[0] / 1e6, time_range[1] / 1e6)
        ax2.set_xticks(np.arange(0, time_range[1] / 1e6 + 1, time_major_tick_interval / 1e6))
        ax2.xaxis.set_minor_locator(MultipleLocator(time_minor_tick_interval / 1e6))
        ax2.set_ylim(velocity_range[0], velocity_range[1])
        ax2.set_yticks(np.arange(velocity_range[0], velocity_range[1] + 1, velocity_major_tick_interval))
        ax2.yaxis.set_minor_locator(MultipleLocator(velocity_minor_tick_interval))
        ax2.set_xlabel("Time (Ma)")
        ax2.set_ylabel("Velocity (cm/yr)")

        ax3 = axes[1, 1]
        
        # line3_1, = ax1.plot(times / 1e6, dip_100s * 180.0 / np.pi, label="Dip Angle", linestyle='dotted', color=_color)
        line3_2, = ax3.plot(times / 1e6, (mdd2s - mdd1s)/1e3, linewidth=4, color=_color)
        ax3.set_xlim(time_range[0] / 1e6, time_range[1] / 1e6)
        ax3.set_xticks(np.arange(0, time_range[1] / 1e6 + 1, time_major_tick_interval / 1e6))
        ax3.xaxis.set_minor_locator(MultipleLocator(time_minor_tick_interval / 1e6))
        ax3.set_yticks(np.arange(0, mdd_range[1] - mdd_range[0] + 1, mdd_major_tick_interval))
        ax3.yaxis.set_minor_locator(MultipleLocator(mdd_minor_tick_interval))
        ax3.set_ylabel("Decoupling Depth Difference (km)")
        ax3.tick_params(axis='y')
        ax3.set_ylim(mdd_range[0], mdd_range[1])

        lines = [line2_2, line2_3]
        labels = [line.get_label() for line in lines]

        


def plot_pressure_temperature(ax1, ax1_twin, depth, P, T):
    """
    Create a subplot to plot Depth vs Pressure with a twin axis for Temperature.
    
    Parameters:
    ax1 : matplotlib.axes.Axes
        Main axis for plotting pressure vs depth.
    ax1_twin : matplotlib.axes.Axes
        Twin axis for plotting temperature vs depth.
    depth : array-like
        Depth values (in meters).
    P : array-like
        Pressure values (in Pascals).
    T : array-like
        Temperature values (in Kelvin).
    """
    # Hard-coded limits and tick intervals
    depth_limit = (0, 3000)  # Depth axis range (in kilometers).
    depth_tick_interval = 500  # Tick interval for depth axis.
    P_limit = (0, 150)  # Pressure axis range (in GPa).
    P_tick_interval = 50.0  # Tick interval for pressure axis.
    T_limit = (0, 3000)  # Temperature axis range (in Kelvin).
    T_tick_interval = 1000.0  # Tick interval for temperature axis.

    # Plot Pressure vs Depth
    ax1.plot(P / 1e9, depth / 1e3, label="P", color="blue")  # Pressure plot in GPa and km.
    lines1, labels1 = ax1.get_legend_handles_labels()  # Gather legend entries.

    ax1.set_xlim(P_limit)
    ax1.xaxis.set_major_locator(MultipleLocator(P_tick_interval))
    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax1.set_ylim(depth_limit)
    ax1.yaxis.set_major_locator(MultipleLocator(depth_tick_interval))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax1.set_xlabel("Pressure (GPa)")
    ax1.set_ylabel("Depth (km)")
    ax1.invert_yaxis()  # Depth increases downward.

    ax1.grid()

    # Plot Temperature vs Depth on the twin axis
    ax1_twin.plot(T, depth / 1e3, label="T", color="red")  # Temperature plot in Kelvin.
    lines2, labels2 = ax1_twin.get_legend_handles_labels()

    ax1_twin.set_xlim(T_limit)
    ax1_twin.xaxis.set_major_locator(MultipleLocator(T_tick_interval))

    ax1_twin.set_xlabel("Temperature (K)")

    # Combine legends for Pressure and Temperature plots
    combined_lines = lines1 + lines2
    combined_labels = labels1 + labels2
    ax1.legend(combined_lines, combined_labels, loc="lower left")


def plot_viscosity_components(ax2, depth, diffusion, dislocation, composite):
    """
    Create a subplot to plot Depth vs Viscosity components (diffusion, dislocation, and composite creep).
    
    Parameters:
    ax2 : matplotlib.axes.Axes
        Axis for plotting viscosity components vs depth.
    depth : array-like
        Depth values (in meters).
    diffusion : array-like
        Diffusion viscosity values.
    dislocation : array-like
        Dislocation viscosity values.
    composite : array-like
        Composite viscosity values.
    """
    # Hard-coded limits and tick intervals
    depth_limit = (0, 3000)  # Depth axis range (in kilometers).
    depth_tick_interval = 500  # Tick interval for depth axis.
    viscosity_limit = (1e18, 1e24)  # Viscosity axis range (in Pa·s, logarithmic scale).
    viscosity_ticks = [1e18, 1e19, 1e20, 1e21, 1e22, 1e23, 1e24]  # Tick values for viscosity axis.

    # Plot viscosity components vs Depth
    if diffusion is not None:
        ax2.plot(diffusion, depth / 1e3, label="Diffusion", color="c")  # Diffusion creep.
    if dislocation is not None:
        ax2.plot(dislocation, depth / 1e3, label="Dislocation", color="green")  # Dislocation creep.
    if composite is not None:
        ax2.plot(composite, depth / 1e3, linestyle="--", label="Composite", color="red")  # Composite creep.

    ax2.set_xscale("log")  # Logarithmic scale for viscosity.

    ax2.set_xlim(viscosity_limit)
    ax2.set_xticks(viscosity_ticks)

    ax2.set_ylim(depth_limit)
    ax2.yaxis.set_major_locator(MultipleLocator(depth_tick_interval))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax2.set_xlabel("Viscosity (Pa·s)")
    ax2.set_ylabel("Depth (km)")
    ax2.invert_yaxis()  # Depth increases downward.

    ax2.grid()
    ax2.legend()  # Add legend for the viscosity components.


def finalize_visualization_2d(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    #If pdf is not provide, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert(os.path.isfile(pdf_file))

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=(1700, 800),  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(-75, 64), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[1.20518518519, None],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    # image_path = "your_image.png"  # Replace with the path to your PNG file
    # output_path = "output_image_with_text.png"  # Path to save the output image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file


def finalize_visualization_2d_12172024(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)
    canvas_size = kwargs.get("canvas_size", (996, 568))

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    #If pdf is not provide, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert(os.path.isfile(pdf_file))

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    fig_pos = (-102, -9)
    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[fig_pos, (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[0.81185, None],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    # image_path = "your_image.png"  # Replace with the path to your PNG file
    # output_path = "output_image_with_text.png"  # Path to save the output image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file

def finalize_visualization_2d_wedge_12202024(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)

    canvas_size = kwargs.get("canvas_size", (650, 550))

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    #If pdf is not provide, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert(os.path.isfile(pdf_file))

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(-90, 5), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[0.661481, None],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    # image_path = "your_image.png"  # Replace with the path to your PNG file
    # output_path = "output_image_with_text.png"  # Path to save the output image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file



def finalize_visualization_2d_wedge_02122025(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):
    # Options
    add_time = kwargs.get("add_time", True)

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files
    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    # If pdf is not provided, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert os.path.isfile(pdf_file)

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=(1050, 770),  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(-33, 13), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[0.9370, None],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file

def finalize_visualization_2d_wedge_02252025(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)

    canvas_size = kwargs.get("canvas_size", (800, 450))

    # Inputs
    eps_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, "img", "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, "img", "prep")
    if not os.path.isdir(prep_file_dir):
        os.mkdir(prep_file_dir)

    output_image_file = os.path.join(prep_file_dir, "%s_t%.4e.png" % (file_name, _time))
    if os.path.isfile(output_image_file):
        # Remove existing output image to ensure a clean overlay
        os.remove(output_image_file)

    #If pdf is not provide, converts an EPS file to a PDF format using the plot_helper module.
    if not os.path.isfile(pdf_file):
        convert_eps_to_pdf(eps_file, pdf_file)
    assert(os.path.isfile(pdf_file))

    # Extracts an image from a PDF file with specific dimensions and an optional crop box.
    target_size = (1350, 704)  # Desired image dimensions in pixels
    crop_box = (200, 100, 1000, 700)  # Optional crop box to define the region of interest

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, "img"), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(45, -5), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[0.542963, 1.0],  # Scaling factors for resizing the images
        output_image_file=output_image_file  # Path to save the final combined image
    )

    # Example Usage, add_text_to_image
    # image_path = "your_image.png"  # Replace with the path to your PNG file
    # output_path = "output_image_with_text.png"  # Path to save the output image
    if add_time:
        text = "t = %.1f Ma" % (_time / 1e6)  # Replace with the text you want to add
        position = (25, 25)  # Replace with the desired text position (x, y)
        font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"  # Path to Times New Roman font
        font_size = 72

        add_text_to_image(output_image_file, output_image_file, text, position, font_path, font_size)

    return output_image_file