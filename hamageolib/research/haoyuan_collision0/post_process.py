import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from hamageolib.core.post_process import PYVISTA_PROCESS, PYVISTA_PROCESS_WORKFLOW_ERROR

SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


class PYVISTA_PROCESS_COLLISION(PYVISTA_PROCESS):

    def __init__(self, data_dir, options, *,
                 pyvista_outdir=None):

        PYVISTA_PROCESS.__init__(self, data_dir, 
                            pyvista_outdir=pyvista_outdir)
        
        self.Min0 = options["BOTTOM"]
        self.Max0 = options["TOP"]
        self.Min2 = options["LEFT"]
        self.Max2 = options["RIGHT"]

        # placeholder for class variables
        self.slab_surface_points = None

    def read(self, pvtu_step):

        # read dataset
        self.pvtu_step = pvtu_step
        PYVISTA_PROCESS.read(self, pvtu_step)

        # total slab composition
        self.grid["sp_total"] = self.grid["gabbro"] + self.grid["MORB"] + self.grid["sediment"]

    def extract_slab(self, *,
                     threshold=0.5,
                     d0=5e3,
                     dr=0.001,
                     output_surfuce=False):

        self.slab_grid = self.grid.threshold(value=threshold, scalars="sp_total", invert=False)

        slab_points = self.slab_grid.points
        slab_point_data = self.slab_grid.point_data

        # Extract slab surface points
        # First construct a KDTree from data
        # Then figure out the max phi value
        # Record the trench position, slab depth and save slab surface points
        # Here I inherited the structure with the 3D case to delineate dimensions
        # into 0, 1, 2.
        vals0 = np.arange(self.Min0, self.Max0, d0)
        vals1 = np.pi/2.0
        vals2 = np.full(vals0.size, np.nan)

        rt_upper = np.vstack([slab_points[:, 1]/self.Max0]).T
        v2_u = slab_points[:, 0]
        rt_tree = cKDTree(rt_upper)

        for i, v0 in enumerate(vals0):
            query_pt = np.array([v0/self.Max0])
            idxs = rt_tree.query_ball_point(query_pt, r=dr)

            if not idxs:
                continue

            v2s = v2_u[idxs]
            max_v2 = np.max(v2s)

            vals2[i] = max_v2
        
        mask = ~np.isnan(vals2)

        v0_surf = vals0[mask]
        v2_surf = vals2[mask]

        x = v2_surf
        y = v0_surf
        z = np.zeros(v0_surf.shape)

        self.slab_surface_points = np.vstack([x, y, z]).T

        if output_surfuce:
            point_cloud = pv.PolyData(self.slab_surface_points)
            filename = "slab_surface_%05d.vtp" % self.pvtu_step
            filepath = os.path.join(self.pyvista_outdir, filename)
            point_cloud.save(filepath)
            print("Save file %s" % filepath)

    def analyze_slab(self):

        if self.slab_surface_points is None:
            raise PYVISTA_PROCESS_WORKFLOW_ERROR("Needs to run extract_slab first.")

        # extract the x, y coordinates
        l0 = self.slab_surface_points[:, 1] 
        l2 = self.slab_surface_points[:, 0]
        
        # slab depth 
        slab_depth = self.Max0 - np.min(l0)

        # shallowest point
        i0 = np.argmax(l0) 
        depth0 = self.Max0 - l0[i0] 
        l2_0 = l2[i0]
        
        # dip angle - 100 km
        depth1 = 100e3
        l2_1 = np.interp(self.Max0 - depth1, l0, l2)
        dip_angle_100 = np.arctan2(depth1 - depth0, l2_1 - l2_0)

        # dip angle - 300 km
        depth1 = 300e3
        l2_1 = np.interp(self.Max0 - depth1, l0, l2)
        dip_angle_300 = np.arctan2(depth1 - depth0, l2_1 - l2_0)

        # trench position
        trench_center = l2[i0]
        trench_center_50km = np.interp(self.Max0 - 50e3, l0, l2)

        # record results of slab depth, dip angle, trench position, etc
        outputs = {} 
        outputs["slab_depth"] = slab_depth
        outputs["dip_100"] = dip_angle_100
        outputs["dip_300"] = dip_angle_300
        outputs["trench_center"] = trench_center
        outputs["trench_center_50"] = trench_center_50km

        return outputs


def ProcessVtuFileTwoDStep(case_path, pvtu_step, Case_Options, *,
                           pyvista_outdir=None):
    '''
    Process with pyvsita for a single step
    Inputs:
        case_path - full path of a 3-d case
        pvtu_step - pvtu_step of vtu output files
        Case_Options - options for the case
        kwargs
            threshold_lower - threshold for lower slab composition
    '''
    # time step and index
    idx = Case_Options.summary_df["Vtu snapshot"] == pvtu_step
    try:
      _time = Case_Options.summary_df.loc[idx, "Time"].values[0]
    except IndexError:
        raise IndexError("The pvtu_step %d doesn't seem to exist in this case" % pvtu_step)
    
    # output directory
    if pyvista_outdir is None:
        if not os.path.isdir(os.path.join(case_path, "pyvista_outputs")):
            os.mkdir(os.path.join(case_path, "pyvista_outputs"))
        pyvista_outdir = os.path.join(case_path, "pyvista_outputs", "%05d" % pvtu_step)

    if not os.path.isdir(pyvista_outdir):
        os.mkdir(pyvista_outdir)
    
    # dict for saving outputs 
    outputs = {}

    # initiate the processing class
    ProcessCollision = PYVISTA_PROCESS_COLLISION(os.path.join(case_path, "output", "solution"), Case_Options.options,
                                                 pyvista_outdir=pyvista_outdir)

    # read file
    ProcessCollision.read(pvtu_step)

    # extract slab
    ProcessCollision.extract_slab(output_surfuce=True)

    # analyze slab
    outputs1 = ProcessCollision.analyze_slab()
    outputs.update(**outputs1)

    return outputs


def GenerateParaviewScript(case_path, Case_Options, ofile_list, additional_options={}, **kwargs):
    '''
    generate paraview script
    Inputs:
        ofile_list - a list of file to include in paraview
        additional_options - options to append
    '''
    animation = kwargs.get("animation", False)
    require_base = kwargs.get('require_base', True)
    for ofile_base in ofile_list:
        # Different file name if make animation
        if animation:
            snapshot = Case_Options.get_pvtu_step(kwargs["steps"][0])
            odir = os.path.join(case_path, 'paraview_scripts', "%05d" % snapshot)
            if not os.path.isdir(odir):
                os.mkdir(odir)
            ofile = os.path.join(case_path, 'paraview_scripts', "%05d" % snapshot, ofile_base)
        else:
            ofile = os.path.join(case_path, 'paraview_scripts', ofile_base)
        # Read base file
        paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts',"Collision0", ofile_base)
        if require_base:
            paraview_base_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')  # base.py : base file
            Case_Options.read_contents(paraview_base_script, paraview_script)  # this part combines two scripts
        else:
            Case_Options.read_contents(paraview_script)  # this part combines two scripts
        # Update additional options
        Case_Options.options.update(additional_options)
        if animation:
            Case_Options.options["ANIMATION"] = "True"
        # Generate scripts
        Case_Options.substitute()  # substitute keys in these combined file with values determined by Interpret() function
        ofile_path = Case_Options.save(ofile)  # save the altered script
        print("\t File generated: %s" % ofile_path)


def finalize_visualization_2d_11022025(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    from ...utils.plot_helper import convert_eps_to_pdf, extract_image_by_size, overlay_images_on_blank_canvas,\
    add_text_to_image

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

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(-210, -160), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[None, None],  # Scaling factors for resizing the images
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


def finalize_visualization_2d_03132026(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    from hamageolib.utils.plot_helper import convert_eps_to_pdf, extract_image_by_size, overlay_images_on_blank_canvas,\
    add_text_to_image

    # Options
    add_time = kwargs.get("add_time", True)
    canvas_size = kwargs.get("canvas_size", (1500, 700))

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
        image_positions=[(51, -53), (0, 0)],  # Positions of each image on the canvas
        cropping_regions=[None, None],  # Optional cropping regions for the images
        image_scale_factors=[None, None],  # Scaling factors for resizing the images
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
