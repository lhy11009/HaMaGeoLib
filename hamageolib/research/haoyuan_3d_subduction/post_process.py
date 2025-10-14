import os
import math
import time
import psutil
from pathlib import Path
import pyvista as pv
import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import NearestNDInterpolator
from vtk import VTK_QUAD
from hamageolib.utils.geometry_utilities import cartesian_to_spherical, spherical_to_cartesian, cartesian_to_spherical_2d, PUnified
from hamageolib.utils.vtk_utilities import get_pyvista_extension
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from hamageolib.utils.exception_handler import my_assert
from hamageolib.utils.interp_utilities import KNNInterpolatorND
from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS, CASE_OPTIONS_TWOD1
from hamageolib.utils.plot_helper import convert_eps_to_pdf, extract_image_by_size, overlay_images_on_blank_canvas,\
    add_text_to_image, scale_matplotlib_params
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


class PYVISTA_PROCESS_THD_WORKFLOW_ERROR(Exception):
    pass

class PYVISTA_PROCESS_THD():

    """
    Class for post-processing spherical shell data from geodynamic simulations using PyVista.
    
    Author: Haoyuan Li
    License: MIT
    
    This class is designed for chunk geometry (spherical shell) simulations using 
    pyvista-readable `.pvtu` output. It provides tools to:
      - Read and augment simulation data with derived fields (e.g., radius)
      - Generate slices at the center, surface, or custom depth
      - Extract iso-volumes for fields like sp_upper, sp_lower, and plate_edge
      - Identify and export slab surface and plate edge surfaces
      - Apply geometric filters to extract regions of interest (e.g., large phi, small theta)
      - Generate bounding surfaces for visualization
    
    Attributes:
        geometry (str): Simulation geometry, currently only "chunk" is supported.
        pyvista_outdir (str): Output directory for saving generated VTK files.
        Max0 (float): Outer radius / box thickness in meters.
        Min0 (float): Inner radius in meters.
        Max1 (float): Maximum colatitude in radians (default: π/2).
        Min1 (float): Minimum colatitude in radians (default: 0.0).
        Max2 (float): Maximum longitude in radians (default: π/2).
        Min2 (float): Minimum longitude in radians (default: 0.0).
        clip (list): contains, 3 smaller list, Range of coordinates to clip
        phi_max (float): Maximum longitude in radians (default: 140°).
        pvtu_step (int): Timestep index associated with current file.
        pvtu_filepath (str): Path to the .pvtu file being processed.
        grid (pv.UnstructuredGrid): Loaded PyVista grid from .pvtu file.
        iso_volume_upper (pv.UnstructuredGrid): Iso-volume of 'sp_upper' above threshold.
        iso_volume_lower (pv.UnstructuredGrid): Iso-volume of 'sp_lower' above threshold.
        iso_plate_edge (pv.UnstructuredGrid): Iso-volume of 'plate_edge' above threshold.
        iso_volume_lower_filtered_pe (pv.UnstructuredGrid): Iso-volume of 'sp_lower' above threshold,
            with redundent points outside of the slab filtered.
        slab_surface_points (np.ndarray): Extracted (x, y, z) coordinates of slab surface.
        pe_edge_points (np.ndarray): Extracted (x, y, z) coordinates of plate edge surface.
    """

    class PyVistaObjectIsNoneError(Exception):
        pass

    def __init__(self, data_dir, config, **kwargs):
        # data_directory
        self.data_dir = data_dir

        # Required settings and geometry assumption
        self.geometry = config["geometry"]
        if self.geometry == "chunk":
            self.is_spherical = True
            self.Ro = 6371e3 # radius of the Earth
        elif self.geometry == "box":
            self.is_spherical = False
            self.Ro = None
        else:
            raise ValueError("geometry needs to be \"chunk\" or \"box\".")

        # Min and Max limit along each dimension
        self.Max0 = config["Max0"]
        self.Min0 = config["Min0"]
        self.Max1 = config["Max1"]
        self.Min1 = 0.0
        self.Max2 = config["Max2"]
        self.Min2 = 0.0
        self.time = config["time"]

        # Scaling along each dimension
        self.scale0 = 1000e3
        if self.is_spherical:
            self.scale1 = self.scale0/self.Ro
            self.scale2 = self.scale0/self.Ro
        else:
            self.scale1 = self.scale0
            self.scale2 = self.scale0
        
        # additional settings
        self.pyvista_outdir = kwargs.get("pyvista_outdir", os.path.join(self.data_dir, "..", "pyvista_outputs"))
        if not os.path.isdir(self.pyvista_outdir):
            os.mkdir(self.pyvista_outdir)
        self.clip = kwargs.get("clip", None)
        self.n_pieces = kwargs.get("n_pieces", None)
        if self.n_pieces is not None:
            my_assert(isinstance(self.n_pieces, int) and self.n_pieces > 0, TypeError, "n_pieces must be positive integar")
        
        # Initialize global variables 
        self.pvtu_step = None
        self.grid = None

        # Initialize runtime variables
        # We use the "combined" dict to append piecewise outputs into lists
        # and then merged to class attributes one by one
        self.keys = ["sliced", "sliced_u", "sliced_shell", "sliced_depth", "iso_volume_upper", "iso_volume_lower", "iso_volume_lower_filtered_pe",
                     "grid_metastable", "grid_metastable_slab", "iso_plate_edge", "slab_surface_points", "pe_edge_points", "trench_points", "additional_trench_points"]
        self.combined = {}
        for key in self.keys:
            setattr(self, key, None)

        # Initialize slab morphology variables
        # self.trench_points = None
        self.slab_surface_points = None
        self.slab_moho_points = None
        self.trench_center = None
        self.slab_depth = None
        self.dip_100_center = None
        self.additional_trench_center = None
        self.metastable_volume = None
        self.metastable_area_center = None
        self.metastable_area_cold_center = None

        # Initialize interpolators
        self.sp_upper_KDTree = None
        self.sp_lower_KDTree = None
        self.slab_surface_interp_func = None
        self.slab_moho_interp_func = None

        # Values for threshold
        self.threshold_upper = None
        self.threshold_lower = None
        self.threshold_edge = None
        self.threshold_metastable = None


    def read(self, pvtu_step, **kwargs):
        """
        Read a .pvtu file and initialize the PyVista grid object.

        Parameters:
            pvtu_step (int): Timestep index corresponding to the .pvtu file.
            pvtu_filepath (str): Full path to the .pvtu file to load.

        This method:
            - Loads the VTK mesh file into memory as a PyVista grid.
            - Computes the radius for each point and adds it as a new scalar field.
            - Stores the grid and file metadata in instance variables.
        """
        start = time.time()
        self.pvtu_step = pvtu_step
        piece = kwargs.get("piece", None)

        # check path of data
        if piece is None:
            filepath = os.path.join(self.data_dir, "solution-%05d.pvtu" % self.pvtu_step)
            my_assert(os.path.isfile(filepath), FileNotFoundError, "File %s is not found" % filepath)
        else:
            filepath = os.path.join(self.data_dir, "solution-%05d.%04d.vtu" % (self.pvtu_step, piece))
            my_assert(isinstance(piece, int) and piece >= 0, TypeError, "piece must be non-negative integar.")
            my_assert(os.path.isfile(filepath), FileNotFoundError, "File %s is not found" % filepath)
        print("PYVISTA_PROCESS_THD:\n\tRead file %s" % (filepath))

        # read data
        self.grid = pv.read(filepath)

        # append a "radius" field 
        points = self.grid.points
        if self.geometry == "chunk":
            radius = np.linalg.norm(points, axis=1)
        else:
            radius = points[:, 2]
        self.grid["radius"] = radius

        # clip data if needed
        # the order of coordinates are in l0, l1, l2
        # in cartesian, this is z, y, x
        # in spherical, this is phi, theta, r
        if self.clip is not None:
            if self.geometry == "chunk":
                self.grid = clip_grid_by_spherical_range(self.grid, r_range=self.clip[0],\
                     theta_range=[np.pi/2.0 - self.clip[1][1], np.pi/2.0 - self.clip[1][0]], phi_range=self.clip[2])
            else:
                self.grid = clip_grid_by_xyz_range(self.grid, x_range=self.clip[2], y_range=self.clip[1], z_range=self.clip[0])

        end = time.time()
        print("\ttakes %.1f s" % (end - start))

    def process_piecewise(self, func_name, keys, **kwargs):
        '''
        Process the data piecewise. Put them into the dict of 
        self.combined to be merged later
        Inputs:
            func_name - name of the member function
            keys - key of the data structure
        '''
        # Set method to use 
        method = getattr(self, func_name)

        # Set options to method
        kwargs["save_file"] = False
        kwargs["update_attributes"] = False

        outputs = method(**kwargs)
        if isinstance(outputs, tuple):
            # multiple outputs
            assert(len(outputs) == len(keys))
            for i, key in enumerate(keys):
                assert(isinstance(outputs[i], pv.DataSet))
                if key not in self.combined:
                    self.combined[key] = []
                self.combined[key].append(outputs[i])
        elif isinstance(outputs, pv.DataSet):
            # only one output
            assert(len(keys)==1)
            if keys[0] not in self.combined:
                self.combined[keys[0]] = []
            self.combined[keys[0]].append(outputs)
        else:
            raise TypeError("Return value from function " + str(method) + " should be either tuple or pyvista.DataSet")

    def export_piecewise(self, i_piece, **kwargs):
        '''
        Export the results from a single piece
        Inputs:
            i_piece - number of piece
        '''
        from hamageolib.utils.vtk_utilities import pyvista_safe_save
        
        # Addtional options
        indent = 4

        # Save file
        start = time.time()

        for key, value in self.combined.items():
            assert(isinstance(value, list) and len(value)==1)
            extension = get_pyvista_extension(value[0])
            odir = os.path.join(self.pyvista_outdir, "p%02d" % i_piece)
            if not os.path.isdir(odir):
                os.mkdir(odir)
            ofile = os.path.join(odir, "%s_%05d.%s" % (key, self.pvtu_step, extension))
            pyvista_safe_save(value[0], ofile)
            print("%ssaved file for piece %d: %s" % (indent*" ", i_piece, ofile))
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def import_piecewise(self, i_piece, **kwargs):
        '''
        Import the results from a single piece
        Inputs:
            i_piece - number of piece
        '''
        indent = 4

        # import
        start = time.time()
        for key, value in self.__dict__.items():
            if value is None or isinstance(value, pv.DataSet):
                odir = os.path.join(self.pyvista_outdir, "p%02d" % i_piece)
                assert(os.path.isdir(odir))

                ofile_base = "%s_%05d" % (key, self.pvtu_step)
                ofile = find_file_by_stem(odir, ofile_base)
                if ofile is not None:
                    pv_obj = pv.read(ofile)

                    if key not in self.combined:
                        self.combined[key] = []
                    self.combined[key].append(pv_obj)
                    print("%sload attr " % (indent*" ") + key + " from piece " + str(i_piece))

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))
    
    def combine_pieces(self):
        '''
        Merge the pv objects in self.combined
        '''
        indent = 4

        # merge objects in combined (dict)
        start = time.time()
        for key, value in self.combined.items():
            if len(value) == 0:
                merged = None
            else:
                merged = pv.merge(value, merge_points=True)
            setattr(self, key, merged)

        # reset self.combined
        self.combined = {}

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def reset_attrs(self, keys):
        '''
        reset attrs to None
        '''
        import gc
        for key in keys:
            setattr(self, key, None) 
    
    def write_key_to_file(self, key, filename_base, filetype):
        '''
        Write a single class attribute to file
        Inputs:
            key - name of the class attribute
            filename_base - basename of output file
            filetype - type of the file
        '''
        target = getattr(self, key)
        assert(target is not None)
        self.write_object_to_file(target, filename_base, filetype)

    def write_object_to_file(self, target, filename_base, filetype, **kwargs):
        '''
        Write a single class object to file
        Inputs:
            target - the target to write
            filename_base - basename of output file
            filetype - type of the file
        '''
        from hamageolib.utils.vtk_utilities import pyvista_safe_save

        indent = 4
        # check inputs
        assert(filetype in ["vtp", "vtu"])

        # export
        start = time.time()
        filename = "%s_%05d.%s" % (filename_base, self.pvtu_step, filetype)
        filepath = os.path.join(self.pyvista_outdir, filename)

        pyvista_safe_save(target, filepath)
        assert(os.path.isfile(filepath))
        print("%ssaved file %s" % (indent*" ", filepath))
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))


    def slice_center(self, **kwargs):
        """
        Extract a 2D slice at the center of the domain in the x-y plane.

        This method:
            - Uses a horizontal slicing plane (z-normal, through origin at radius Ro).
            - Projects the 'velocity' field onto the plane.
            - Saves the resulting slice and velocity projection as a .vtp file.
        """
        # Additional options
        boundary_range = kwargs.get("boundary_range", None)
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)

        start = time.time()
        assert self.grid is not None

        origin = (self.Max0, 0.0, 0.0)

        if self.geometry == "chunk":
            normal = (0.0, 0.0, -1.0)
        else:
            normal = (0.0, -1.0, 0.0)

        sliced = self.grid.slice(normal=normal, origin=origin, generate_triangles=True)

        V = sliced["velocity"]
        dot_products = np.dot(V, normal)
        V_proj = V - np.outer(dot_products, normal)

        if V_proj.shape[0] > 0:
            sliced["velocity_slice"] = V_proj

        # filter data with the given boundary
        if boundary_range is not None:
            points = sliced.points

            if self.geometry == "chunk":
                r_min = boundary_range[0][0]
                r_max = boundary_range[0][1]
                lat_min = boundary_range[1][0]
                lat_max = boundary_range[1][1]
                lon_min = boundary_range[2][0]
                lon_max = boundary_range[2][1]
                r, theta, phi = cartesian_to_spherical(*points.T)
                
                # Check which points are inside the bounds
                in_r = (r >= r_min) & (r <= r_max)
                in_theta = (theta >= np.pi/2.0 - lat_max) & (theta <= np.pi/2.0 - lat_min)
                in_phi = (phi >= lon_min) & (phi <= lon_max)
                is_in_bounds = in_r & in_theta & in_phi
            else:
                x_min = boundary_range[2][0]
                x_max = boundary_range[2][1]
                y_min = boundary_range[1][0]
                y_max = boundary_range[1][1]
                z_min = boundary_range[0][0]
                z_max = boundary_range[0][1]

                # Check which points are inside the bounds
                is_in_x = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
                is_in_y = (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
                is_in_z = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
                is_in_bounds = is_in_x & is_in_y & is_in_z
                
            cell_mask = np.ones(sliced.n_cells, dtype=bool)
                
            # Use PyVista's connectivity and cell arrays if available
            for cid in range(sliced.n_cells):
                pt_ids = sliced.get_cell(cid).point_ids
                if np.any(~is_in_bounds[pt_ids]):
                    cell_mask[cid] = False

            sliced_u = sliced.extract_cells(cell_mask)
        else:
            sliced_u = sliced.cast_to_unstructured_grid()

        # update class attributes and save files
        if update_attributes:
            self.sliced = sliced
            self.sliced_u = sliced_u
        if save_file:
            self.write_object_to_file(sliced, "slice_center_unbounded", "vtp")
            self.write_object_to_file(sliced_u, "slice_center", "vtu")

        end = time.time()
        print("\tslice_center takes %.1f s" % (end - start))

        return sliced, sliced_u
    
    def slice_surface(self, **kwargs):
        """
        Extract a thin shell at the planetary surface by thresholding radius.

        This method:
            - Applies a narrow threshold around the outer radius (Ro ± 5 km).
            - Extracts a thin surface shell of data near the planetary surface.
            - Saves the result as a .vtk file.
        """
        start = time.time()

        # Additional options 
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
       
        indent = 4

        r_slice = self.Max0
        r_diff = 5e3

        sliced_shell = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

        filename = "slice_outer_%05d.vtu" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        sliced_shell.save(filepath)

        if update_attributes:
            self.sliced_shell = sliced_shell
        if save_file:
            self.write_object_to_file(sliced_shell, "slice_outer", "vtu")

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_surface takes %.1f s" % (indent * " ", end - start))

        return sliced_shell

    def slice_at_depth(self, **kwargs):
        """
        Extract a horizontal shell at a specified depth below the surface.

        Parameters:
            kwargs:
                r_diff: range of radius / z values to clip
                depth (float): Depth in meters from the planetary surface.

        This method:
            - Applies a radius threshold around Ro - depth with ±10 km tolerance.
            - Extracts a shell of data at the given depth.
            - Saves the result as a .vtk file.
        """
        start = time.time()

        # additional options 
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
        indent = 4

        r_diff = kwargs.get("r_diff", 50e3)
        depth = kwargs.get("depth", 200e3)
        r_slice = self.Max0 - depth

        if self.geometry == 'chunk':
            slice_clip = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

            # Project velocity onto the surface
            # Get point coordinates and velocity vectors
            # Compute normalized radial direction vectors at each point
            # Project velocity onto tangent plane of sphere: v_tangent = v - (v · r̂) r̂
            # Store new vector field

            # interpolate the slice
            # Create the new mesh surface_points
            # Build KDTree and interpolate
            # Interpolate velocity field
            resolution_radian = 0.1*np.pi/180.0
            if self.clip is not None:
                theta = np.arange(np.pi/2.0 - self.clip[1][1], np.pi/2.0 - self.clip[1][0], resolution_radian)
                phi = np.arange(self.clip[2][0], self.clip[2][1], resolution_radian)
            else:
                theta = np.arange(np.pi/2.0 - self.Max1, np.pi/2.0 - self.Min1, resolution_radian)
                phi = np.arange(self.Min2, self.Max2, resolution_radian)
            n_theta = theta.size
            n_phi = phi.size

            theta_grid, phi_grid = np.meshgrid(theta, phi)
            x, y, z = spherical_to_cartesian(r_slice, theta_grid, phi_grid)
            surface_points = np.c_[x.ravel(), y.ravel(), z.ravel()]

            cells = []
            cell_types = []

            for j in range(n_phi - 1):
                for i in range(n_theta - 1):
                    # Index in flat array
                    p0 = j * n_theta + i
                    p1 = p0 + 1
                    p2 = p0 + n_theta + 1
                    p3 = p0 + n_theta

                    # Each cell is prefixed with number of points (4 for quad)
                    cells.append([4, p0, p1, p2, p3])
                    cell_types.append(VTK_QUAD)

            cells = np.array(cells, dtype=np.int64).flatten()
            cell_types = np.array(cell_types, dtype=np.uint8)

            tree = cKDTree(slice_clip.points)
            _, idx = tree.query(surface_points)

            sliced_depth = pv.UnstructuredGrid(cells, cell_types, surface_points)
            sliced_depth["velocity"] = slice_clip.point_data["velocity"][idx, :]
            sliced_depth["radius"] = slice_clip.point_data["radius"][idx]

            # take the clip as the slice
            # slice_at_depth = slice_clip

            # project the velocity
            points = sliced_depth.points
            velocities = sliced_depth.point_data["velocity"]
            radial_dirs = points / np.linalg.norm(points, axis=1)[:, np.newaxis]  # (N, 3)

            v_dot_r = np.sum(velocities * radial_dirs, axis=1)  # (N,)
            v_radial = (v_dot_r[:, np.newaxis]) * radial_dirs   # (N, 3)
            v_tangent = velocities - v_radial                   # (N, 3)

            sliced_depth.point_data["velocity_slice"] = v_tangent
        else:
            origin = (0, 0, r_slice)
            normal = (0, 0, 1.0)

            sliced_depth = self.grid.slice(normal=normal, origin=origin, generate_triangles=True)

            V = sliced_depth["velocity"]
            dot_products = np.dot(V, normal)
            V_proj = V - np.outer(dot_products, normal)

            sliced_depth["velocity_slice"] = V_proj

            sliced_depth = sliced_depth.cast_to_unstructured_grid()
        
        if update_attributes:
            self.sliced_depth = sliced_depth
        if save_file:
            self.write_object_to_file(sliced_depth, "slice_depth_%.1fkm" % (depth/1e3), "vtu")

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_at_depth takes %.1f s" % (indent * " ", end - start))
        return sliced_depth

    def extract_iso_volume_upper(self, **kwargs):
        """
        Extract the iso-volume of the 'sp_upper' composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for 'sp_upper'.

        This method:
            - Filters the grid for regions where sp_upper >= threshold.
            - Stores the result in `self.iso_volume_upper`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4

        # additional options 
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
        threshold = kwargs.get("threshold", 0.8)

        iso_volume_upper = self.grid.threshold(value=threshold, scalars="sp_upper", invert=False)
        
        if update_attributes:
            self.iso_volume_upper = iso_volume_upper
        if save_file:
            self.write_object_to_file(iso_volume_upper, "sp_upper_above_%.2f" % (threshold), "vtu")

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

        self.threshold_upper = threshold

        return iso_volume_upper

    def extract_iso_volume_lower(self, **kwargs):
        """
        Extract the iso-volume of the 'sp_lower' composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for 'sp_lower'.

        This method:
            - Filters the grid for regions where sp_lower >= threshold.
            - Stores the result in `self.iso_volume_lower`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4
        
        # additional options 
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
        threshold = kwargs.get("threshold", 0.8)

        # Iso volume of slab lower composition
        iso_volume_lower = self.grid.threshold(value=threshold, scalars="sp_lower", invert=False)
        
        if update_attributes:
            self.iso_volume_lower = iso_volume_lower
        if save_file:
            self.write_object_to_file(iso_volume_lower, "sp_lower_above_%.2f" % (threshold), "vtu")
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

        self.threshold_lower = threshold

        return iso_volume_lower

    def extract_iso_volume_metastable(self, **kwargs):
        """
        Extract the iso-volume of the 'metastable' composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for 'metastable'.

        This method:
            - Filters the grid for regions where metastable >= threshold.
            - Stores the result in `self.iso_volume_metastable`.
            - Saves the extracted volume as a .vtu file.
        """
        from hamageolib.utils.vtk_utilities import _make_empty_vtu

        start = time.time()
        indent = 4
        
        # additional options 
        T_slab = 900 + 273.15 # K

        options = kwargs["options"] # required, but need to preserve the interface
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
        threshold = kwargs.get("threshold", 0.8)

        # Iso volume of slab lower composition
        grid_metastable_raw = self.grid.threshold(0.5, scalars="metastable", invert=True)
        grid_metastable_c = grid_metastable_raw.point_data_to_cell_data(pass_point_data=False, categorical=False)
        cell_data_P_eq =  (grid_metastable_c.cell_data['T'] - options["T_PT_EQ"]) * options["CL_PT_EQ"] + options["P_PT_EQ"]
        mask = np.flatnonzero(grid_metastable_c.cell_data['p'] > cell_data_P_eq)
        grid_metastable = grid_metastable_c.extract_cells(mask)

        if grid_metastable.n_cells == 0:
            grid_metastable_slab = _make_empty_vtu()
        else:
            # extract the cold region 
            mask = np.flatnonzero(grid_metastable.cell_data['T'] < T_slab)
            grid_metastable_slab = grid_metastable.extract_cells(mask)

        if update_attributes:
            self.grid_metastable = grid_metastable
            self.grid_metastable_slab = grid_metastable_slab

        if save_file:
            self.write_object_to_file(grid_metastable, "metastable_below_%.2f" % (threshold), "vtu")
            self.write_object_to_file(grid_metastable_slab, "metastable_slab_below_%.2f" % (threshold), "vtu")

        self.threshold_metastable = threshold
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

        return grid_metastable, grid_metastable_slab



    def extract_plate_edge(self, **kwargs):
        """
        Extract the iso-volume of the 'plate_edge' field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for 'plate_edge'.

        This method:
            - Filters the grid for regions where plate_edge >= threshold.
            - Stores the result in `self.iso_plate_edge`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4

        # additional options 
        save_file = kwargs.get("save_file", True)
        update_attributes = kwargs.get("update_attributes", True)
        threshold = kwargs.get("threshold", 0.8)

        iso_plate_edge = self.grid.threshold(value=threshold, scalars="plate_edge", invert=False)

        if update_attributes:
            self.iso_plate_edge = iso_plate_edge
        if save_file:
            self.write_object_to_file(iso_plate_edge, "plate_edge_above_%.2f" % (threshold), "vtu")

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

        self.threshold_edge = threshold

        return iso_plate_edge

    def extract_slab_interface(self, field_name, **kwargs):
        """
        Extract the 3D surface of the subducting slab from the sp_upper iso-volume.

        Inputs:
            field_name (str): extract from this field
            extract_trench (bool): whether to extract trench position

        This method:
            - Creates a (r, θ) mesh and queries the maximum φ values for each node using KDTree.
            - Reconstructs the 3D coordinates of the slab interface as a surface in spherical coordinates.
            - Stores the result in `self.slab_surface_points` and exports it as a `.vtp` file.
        """
        start = time.time()

        # additional optoins
        N0 = kwargs.get("N0", 2000)
        N1 = kwargs.get("N1", 1000)
        dr = kwargs.get("dr", 0.001)

        # checking
        if field_name == "sp_upper":
            my_assert(self.iso_volume_upper is not None, PYVISTA_PROCESS_THD_WORKFLOW_ERROR, "Needs to run extract_iso_volume_upper first")
            source = self.iso_volume_upper

        elif field_name == "sp_lower":
            my_assert(self.iso_volume_lower_filtered_pe is not None, PYVISTA_PROCESS_THD_WORKFLOW_ERROR, "Needs to run filter_slab_lower_points first")
            source = self.iso_volume_lower_filtered_pe 
        else:
            raise NotImplementedError()

        # initiation
        indent = 4

        # build the KDTREE for surface points
        # with converted unified coordinates
        vals0 = np.linspace(0, self.Max0, N0)
        vals1 = np.linspace(self.Min1, self.Max1, N1)
        V2 = np.full((N0, N1), np.nan)
        vals2_tr = np.full(N1, np.nan)
        v0_u, v2_u, v1_u = PUnified.points2unified3(source.points, self.is_spherical, False)
        rt_upper = np.vstack([v0_u/self.Max0, v1_u/self.Max1]).T
        rt_tree = cKDTree(rt_upper)


        # extract slab surface points
        # V0, V1 are meshed grid on vals0, vals1
        # V2 are the near-neighbor value of the second dimension on the surface
        for i, v0 in enumerate(vals0):
            for j, v1 in enumerate(vals1):
                query_pt = np.array([v0/self.Max0, v1/self.Max1])
                idxs = rt_tree.query_ball_point(query_pt, r=dr)

                if not idxs:
                    continue

                v2s = v2_u[idxs]
                max_v2 = np.max(v2s)

                V2[i, j] = max_v2


        V0, V1 = np.meshgrid(vals0, vals1, indexing='ij')
        mask = ~np.isnan(V2)

        # Summarize results:
        # 1. slab surface points
        # 2. a k-nearneigbor interpolator (takes unified coordinate v0 and v1, returns v2 on the surface)
        #   k = 1 : using 1 near neighbor (>1 value uses mutliple near neighbors and weights the results)
        #   max_distance : max distance of near neighbor give by scale0*max_distance (e.g. 1000km * 0.05 = 50 km)
        #       Any points without neighbors in this range will be assigned a nan value as v2
        x, y, z = PUnified.unified2points3(np.vstack((V0[mask], V2[mask], V1[mask])).T, self.is_spherical, False)
        if field_name == "sp_upper":
            self.slab_surface_points = np.vstack([x, y, z]).T
            surface_points = self.slab_surface_points
            self.sp_upper_KDTree = rt_tree
            self.slab_surface_interp_func = KNNInterpolatorND(np.vstack((V0.ravel()/self.scale0, V1.ravel()/self.scale1)).T,\
                                                          V2.ravel(), k=1, max_distance=0.05)
        elif field_name == "sp_lower":
            self.slab_moho_points = np.vstack([x, y, z]).T
            surface_points = self.slab_moho_points
            self.sp_lower_KDTree = rt_tree
            self.slab_moho_interp_func = KNNInterpolatorND(np.vstack((V0.ravel()/self.scale0, V1.ravel()/self.scale1)).T,\
                                                          V2.ravel(), k=1, max_distance=0.05)

        # save slab surface points
        point_cloud = pv.PolyData(surface_points)
        filename = "%s_surface_%05d.vtp" % (field_name, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud.save(filepath)

        print("Save file %s" % filepath)

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))


    def extract_plate_edge_surface(self):
        """
        Extract the 3D surface of the plate edge from the plate_edge iso-volume.

        This method:
            - Creates a (r, φ) mesh and queries the minimum θ values for each node using KDTree.
            - Reconstructs the plate edge surface in 3D spherical coordinates.
            - Stores the result in `self.pe_edge_points` and exports it as a `.vtp` file.
        """
        # Extracting parameters
        # N0 - number along radius
        # N2 - number along theta
        # Note these two are chosen so that resolutions along the radius and phi
        # have roughly the same value.
        # min0 - starting value of extraction along radius. This should be
        # deeper than the lower-theta boundary of the subducting slab.
        # dr - normalized tolerance. This correlates to a dimensional value of
        # self.Max0 * dr
        min0 =  self.Max0 - 200e3; N0 = 40
        N2 = 3000
        dr = 0.01
        
        # Build KDTree with (r, theta)
        # convert to unified coordinates
        plate_edge_points = self.iso_plate_edge.points
        
        v0_pe, v2_pe, v1_pe = PUnified.points2unified3(plate_edge_points, self.is_spherical, False)
        rt_pe = np.vstack([v0_pe/self.scale0, v2_pe/self.scale2]).T
        rt_tree_pe = cKDTree(rt_pe)

        v0_vals = np.linspace(min0, self.Max0, N0)
        v2_vals = np.linspace(0.0, self.Max2, N2)
        V1_pe = np.full((N0, N2), np.nan)

        # Loop over each (r, phi), get theta
        # 1. get indexes of adjacent points in the plate_edge composition
        # 2. get minimum theta values within the adjacent points
        for i, v0 in enumerate(v0_vals):
            for j, v2 in enumerate(v2_vals):
                # query_pt = np.array([r/self.Max0, phi/self.Max2])
                query_pt = np.array([v0/self.scale0, v2/self.scale2])
                idxs = rt_tree_pe.query_ball_point(query_pt, r=dr)  # tolerance

                if not idxs:
                    continue

                # get all φ values at this (r, θ)
                v1_s = v1_pe[idxs]
                max1 = np.max(v1_s)

                # assign to value of v1
                V1_pe[i, j] = max1

        # Recover the 3D coordinates from unified coordinates
        V0_pe, V2_pe = np.meshgrid(v0_vals, v2_vals, indexing='ij')

        mask = ~np.isnan(V1_pe)

        x_pe_surf, y_pe_surf, z_pe_surf = PUnified.unified2points3(np.vstack((V0_pe[mask], V2_pe[mask], V1_pe[mask])).T,\
                                                                   self.is_spherical, False)

        # export by pyvista
        self.pe_edge_points = np.vstack([x_pe_surf, y_pe_surf, z_pe_surf]).T
        point_cloud_pe = pv.PolyData(self.pe_edge_points)

        filename = "plate_edge_surface_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_pe.save(filepath)

        print("Save file %s" % filepath)

    def filter_slab_lower_points(self):
        """
        Filter sp_lower iso-volume based on slab surface and plate edge surface geometry.

        This method:
            - Uses KDTree to match each lower point against the slab surface in (r, θ) space,
              filtering out points with φ greater than slab surface.
            - Repeats filtering against plate edge surface in (r, φ) space,
              removing points with θ less than the plate edge.
            - Exports both filtered point clouds and the final cleaned iso-volume.
        """

        # initiation
        my_assert(self.slab_surface_points is not None, self.PyVistaObjectIsNoneError, "slab_surface_points cannot be None")
        my_assert(self.pe_edge_points is not None, self.PyVistaObjectIsNoneError, "pe_edge_points cannot be None")
        start = time.time()
        indent = 4

        # filter parameters
        d_upper_bound = 0.01

        # mesh the slab surface points by kd tree
        v0_s, v2_s, v1_s = PUnified.points2unified3(self.slab_surface_points, self.is_spherical, False)
        normalized_rt = np.vstack([v0_s/self.scale0, v1_s/self.scale1]).T  # shape (N, 2)
        rt_tree_slab_surface = cKDTree(normalized_rt)
        
        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, theta) within slab surface points
        lower_points = self.iso_volume_lower.points
        v0_l, v2_l, v1_l = PUnified.points2unified3(lower_points, self.is_spherical, False)
        query_points = np.vstack([v0_l/self.scale0, v1_l/self.scale1]).T

        distances, indices = rt_tree_slab_surface.query(query_points, k=1, distance_upper_bound=d_upper_bound)
        valid_mask = (indices != rt_tree_slab_surface.n)

        indices_valid = indices[valid_mask]
        lower_points_valid = lower_points[valid_mask]
        v2_valid = v2_l[valid_mask]

        # get mask for points that has higher phi value
        # than their matching points in the slab surface points
        large_v2_mask = v2_valid > v2_s[indices_valid]

        # also get points that has absolutely large phi value,
        # larger than the maximum in slab surface points
        large_v2_abs_mask = v2_l > np.max(v2_s)

        combined_mask = np.zeros(valid_mask.shape, dtype=bool)
        combined_mask[valid_mask] = large_v2_mask
        large_v2_points = lower_points[combined_mask | large_v2_abs_mask]

        # export by pyvista
        point_cloud_large_v2 = pv.PolyData(large_v2_points)

        filename = "sp_lower_large_l2_points_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_large_v2.save(filepath)
        print("Save file %s" % filepath)
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))

        # Derive the indices of large phi points within the original iso_volume_lower
        is_large_v2_point = np.zeros(self.iso_volume_lower.n_points, dtype=bool)

        idx1 = np.flatnonzero(valid_mask)
        large_v2_point_indices = idx1[large_v2_mask]
        is_large_v2_point[large_v2_point_indices] = True
        is_large_v2_point[large_v2_abs_mask] = True

        # Initiate cell_mask with all True value (every cell is included)
        cell_mask = np.ones(self.iso_volume_lower.n_cells, dtype=bool)

        # Use PyVista's connectivity and cell arrays if available
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            # pt_ids = self.iso_volume_lower.Get_Cell(cid).GetPointIds()
            if np.any(is_large_v2_point[pt_ids]):
                cell_mask[cid] = False

        _ = self.iso_volume_lower.extract_cells(cell_mask)

        # mesh a slab edge point by kd tree
        v0_pe, v2_pe, v1_pe = PUnified.points2unified3(self.pe_edge_points, self.is_spherical, False)
        normalized_pe_rt = np.vstack([v0_pe/self.scale0, v2_pe/self.scale2]).T  # shape (N, 2)
        
        rt_tree_pe_surface = cKDTree(normalized_pe_rt)

        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, phi) within slab edge points
        lower_points = self.iso_volume_lower.points

        v0_l, v2_l, v1_l = PUnified.points2unified3(lower_points, self.is_spherical, False)
        query_points = np.vstack([v0_l/self.scale0, v2_l/self.scale2]).T

        # distances, indices = rt_tree_pe_surface.query(query_points_1, k=1, distance_upper_bound=d_upper_bound)
        distances, indices = rt_tree_pe_surface.query(query_points, k=1, distance_upper_bound=d_upper_bound)
        valid_pe_mask = (indices != rt_tree_pe_surface.n)

        indices_pe_valid = indices[valid_pe_mask]
        lower_points_pe_valid = lower_points[valid_pe_mask]
        v1_l_pe_valid = v1_l[valid_pe_mask]

        # get mask for points that has smaller theta value
        # than their matching points in the plate edge surface points
        big_v1_mask = v1_l_pe_valid > v1_pe[indices_pe_valid]

        big_l1_points = lower_points_pe_valid[big_v1_mask]

        # export by pyvista
        point_cloud_big_l1 = pv.PolyData(big_l1_points)

        filename = "sp_lower_big_l1_points_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_big_l1.save(filepath)
        print("Save file %s" % filepath)

        # filter out cells that has small theta values
        # Derive the indices of large theta points within the original iso_volume_lower
        is_big_l1_point = np.zeros(self.iso_volume_lower.n_points, dtype=bool)

        idx1 = np.flatnonzero(valid_pe_mask)
        big_l1_point_indices = idx1[big_v1_mask]
        is_big_l1_point[big_l1_point_indices] = True

        # Use PyVista's connectivity and cell arrays if available
        # Note the cell_mask continues from the previous step
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            if np.any(is_big_l1_point[pt_ids]):
                cell_mask[cid] = False

        # Extract the final filtered points
        self.iso_volume_lower_filtered_pe = self.iso_volume_lower.extract_cells(cell_mask)

        # Save to file for ParaView or future use
        filename = "sp_lower_above_%.2f_filtered_pe_%05d.vtu" % (self.threshold_lower,self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_volume_lower_filtered_pe.save(filepath)

        print("Save file %s" % filepath)

    def extract_slab_dip_angle_by_depth(self, v0_0, v1_0, v0_1, v1_1):
        my_assert(self.slab_moho_interp_func is not None, PYVISTA_PROCESS_THD_WORKFLOW_ERROR,\
                  "Needs to run extract_slab_interface for slab moho first")
        my_assert(v0_0 > v0_1, ValueError, "To extract dip angle, depth1 must be bigger than depth0")

        # derive the surface point from slab moho
        v2_0 = self.slab_moho_interp_func(v0_0/self.scale0, v1_0/self.scale1)
        v2_1 = self.slab_moho_interp_func(v0_1/self.scale0, v1_1/self.scale1)


        # for comparision: get surface point from slab surface
        # this approach has problem with later times (crustal material are not well distributed)
        # v2_0 = self.slab_surface_interp_func(v0_0/self.scale0, v1_0/self.scale1)
        # v2_1 = self.slab_surface_interp_func(v0_1/self.scale0, v1_1/self.scale1)

        if self.geometry == "chunk":
            v0_mean = (v0_0 + v0_1)/2.0
            dip_angle = np.arctan2(v0_0 - v0_1, v0_mean * (v2_1 - v2_0))
        else:
            dip_angle = np.arctan2(v0_0 - v0_1, v2_1 - v2_0)

        return dip_angle
    
    def extract_slab_dip_angle(self):
        '''
        extract slab dip angle
        '''
        self.dip_100_center = self.extract_slab_dip_angle_by_depth(self.Max0-15e3, 0.0, self.Max0-100e3, 0.0)
        

    def extract_slab_dip_angle_deprecated_0(self):
        '''
        extract slab dip angle
        '''
        assert(self.slab_surface_points is not None)
        indent = 4
        
        start = time.time()
        self.dip_100_center = get_slab_dip_angle_deprecated_0(self.slab_surface_points, self.geometry, self.Max0, 0.0, 100e3)
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def extract_slab_trench(self):
        '''
        extract slab trench
        '''
        indent = 4

        # trench points at surface
        start = time.time()
        trench_points = self.extract_trench_profile(0.0)
        _, self.trench_center, _ = PUnified.points2unified3(trench_points[0, :], self.is_spherical, False)

        # trench position at 50 km 
        trench_points = self.extract_trench_profile(50e3)
        _, self.trench_center_50km, _ = PUnified.points2unified3(trench_points[0, :], self.is_spherical, False)
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def extract_trench_profile(self, depth, **kwargs):
        '''
        Parameters:
            depth - the depth of the trench profile (e.g. 0.0 - surface trench position)
            kwargs:
                N1 - number of points along dimension 1
                file_type - type of file outputs (default: output vtp file, txt: output txt file)
        '''
        N1 = kwargs.get("N1", 1000)
        file_type = kwargs.get("file_type", "default")

        # K-nearneighbor interpolation
        val1s = np.linspace(self.Min1, self.Max1, N1)
        val0s = np.full(val1s.shape, self.Max0 - depth)
        val2s = self.slab_surface_interp_func(val0s/self.scale0, val1s/self.scale1)

        # Convert to ordinary coordinates
        mask = ~np.isnan(val2s)
        x, y, z = PUnified.unified2points3(np.vstack((val0s[mask], val2s[mask], val1s[mask])).T, self.is_spherical, False)
        trench_points = np.vstack([x, y, z]).T

        # Export to files
        if file_type == "default":
            point_cloud_tr = pv.PolyData(trench_points)
            filename = "trench_d%.2fkm_%05d.vtp" % (depth/1e3, self.pvtu_step)
            filepath = os.path.join(self.pyvista_outdir, filename)
            point_cloud_tr.save(filepath)
            print("saved file %s" % filepath)
        elif file_type == "txt":
            filename = "trench_d%.2fkm_%05d.txt" % (depth/1e3, self.pvtu_step)
            filepath = os.path.join(self.pyvista_outdir, filename)
            np.savetxt(filepath, trench_points, fmt="%.6f", delimiter="\t", header="X\tY\tZ", comments="")
            print("saved file %s" % filepath)
        else:
            raise ValueError("file_type needs to be default or txt")
        
        return trench_points
    
    def extract_sp_velocity(self):
        # todo_plate
        from hamageolib.utils.geometry_utilities import rotate_vec_cart_to_sph

        my_assert(self.sp_lower_KDTree is not None, PYVISTA_PROCESS_THD_WORKFLOW_ERROR\
                  , "Needs to run extract_slab_interface for sp_lower first")
        
        start = time.time()

        query_plate_velocity_depth = 40e3 # depth to query m
        query_plate_velocity_dist = 500e3 # distance from trench to query
        query_plate_velocity_dist_span = 200e3 # range of distance from trench to query

        query_v0 = self.Max0 - query_plate_velocity_depth
        query_v1 = 0.0 
        if self.is_spherical:
            query_v2_range = [self.trench_center_50km - (query_plate_velocity_dist + query_plate_velocity_dist_span)/self.Max0,\
                            self.trench_center_50km - query_plate_velocity_dist/self.Max0]
        else:
            query_v2_range = [self.trench_center_50km - query_plate_velocity_dist - query_plate_velocity_dist_span,\
                            self.trench_center_50km - query_plate_velocity_dist]

        # source of data
        points = self.iso_volume_lower_filtered_pe.points
        point_data = self.iso_volume_lower_filtered_pe.point_data
        
        # query points within a radius range
        dr = 0.003 
        query_pt = np.array([query_v0/self.Max0, query_v1/self.Max1])
        idxs = self.sp_lower_KDTree.query_ball_point(query_pt, r=dr)
        idx_np = np.array(idxs)
        _, v2s, _ = PUnified.points2unified3(points[idx_np], self.is_spherical, False)
 
        # apply mask for points in v2 range (distance to the trench)
        mask = ((v2s > query_v2_range[0]) & (v2s < query_v2_range[1]))
        v2s = v2s[mask]
        idxs = idx_np[mask]

        point_np = points[idxs]
        velocity_np = point_data["velocity"][idxs]

        # project to coordinate
        if self.is_spherical:
            _, _, velocity_2s = rotate_vec_cart_to_sph(point_np[:, 0], point_np[:, 1], point_np[:, 2],\
                                                    velocity_np[:, 0], velocity_np[:, 1], velocity_np[:, 2])
        else:
            velocity_2s = velocity_np[:, 0]
        self.sp_velocity = np.mean(velocity_2s)

        end = time.time()
        print("%s: Extract plate movement takes %.1f s" % (func_name(), end - start))

    def extract_metastable_area(self, options, **kwargs):
    
        T_slab = 900 + 273.15 # K
        threshold = kwargs.get("threshold", 0.5)

        # Extract metastable volume
        if self.grid_metastable.n_cells == 0:
            vol_selected = 0.0
        else:
            self.grid_metastable = self.grid_metastable.compute_cell_sizes(length=False, area=False, volume=True)
            vol_selected = float(self.grid_metastable.cell_data["Volume"].sum())

        if self.grid_metastable_slab.n_cells == 0:
            vol_selected_slab = 0.0
        else:
            self.grid_metastable_slab = self.grid_metastable_slab.compute_cell_sizes(length=False, area=False, volume=True)
            vol_selected_slab = float(self.grid_metastable_slab.cell_data["Volume"].sum())

        self.metastable_volume = vol_selected
        self.metastable_volume_cold = vol_selected_slab

        # Extract metastable area of slice at the center
        sliced_c = self.sliced.point_data_to_cell_data(pass_point_data=False, categorical=False)

        grid_metastable_c = sliced_c.threshold(threshold, scalars="metastable", invert=True)
        cell_data_P_eq_c =  (grid_metastable_c.cell_data['T'] - options["T_PT_EQ"]) * options["CL_PT_EQ"] + options["P_PT_EQ"]
        mask = np.flatnonzero(grid_metastable_c.cell_data['p'] > cell_data_P_eq_c)
        grid_metastable_c = grid_metastable_c.extract_cells(mask)

        # if no cell presents, assign trivial values
        # else compute the total cell area
        if grid_metastable_c.n_cells == 0:
            self.metastable_area_center = 0.0
            self.metastable_area_cold_center = 0.0
        else:
            sized = grid_metastable_c.compute_cell_sizes(length=False, area=True, volume=False)
            self.metastable_area_center = float(sized.cell_data["Area"].sum())
        
            # now derive the cold area
            mask = np.flatnonzero(grid_metastable_c.cell_data['T'] < T_slab)
            grid_metastable_slab_c = grid_metastable_c.extract_cells(mask)
            if grid_metastable_slab_c.n_cells == 0:
                self.metastable_area_cold_center = 0.0
            else:
                sized_cold = grid_metastable_slab_c.compute_cell_sizes(length=False, area=True, volume=False)
                self.metastable_area_cold_center = float(sized_cold.cell_data["Area"].sum())

            self.write_object_to_file(grid_metastable_c, "metastable_sliced_%.2f" % (threshold), "vtu")
            self.write_object_to_file(grid_metastable_slab_c, "metastable_sliced_slab_%.2f" % (threshold), "vtu")

    def extract_slice_ref_da_profile(self, **kwargs):
        '''
        Summary:
            Extract and compute the reference profile of non-adiabatic (dynamic) pressure along a specified distance 
            from the trench within a sliced PyVista dataset. The function constructs a KD-tree interpolator 
            from point data, queries points along a specified transect, averages interpolated pressure values, 
            and saves the resulting 1-D profile to a text file.

        Parameters:
            **kwargs:
                query_dist (float, optional): Reference distance from the trench in meters (default: 500e3).
                query_dist_span (float, optional): Distance span over which to average dynamic pressure in meters (default: 200e3).

        Returns:
            None

        Implementation Notes:
            - Builds a KD-tree interpolator from the sliced dataset's point data (`nonadiabatic_pressure`).
            - Generates query points along the v0 (horizontal) direction.
            - For each v0, interpolates over a range of v2 (vertical) positions near the trench.
            - Computes the mean non-adiabatic pressure for each horizontal position.
            - Saves the resulting (v0, pressure) pairs to a file named `da_profile_<step>.txt`.
        '''
        query_dist = kwargs.get("query_dist", 500e3)
        query_dist_span = kwargs.get("query_dist_span", 200e3)
        
        # construct the kd tree
        points = self.sliced_u.points
        point_data = self.sliced_u.point_data

        v0, v2, _ = PUnified.points2unified3(points, self.is_spherical, False)
        da_func = KNNInterpolatorND(np.vstack((v0/self.Max0, v2/self.Max2)).T, point_data["nonadiabatic_pressure"], k=1, max_distance=0.05)

        # construct the query points
        query_v0s = np.linspace(self.Min0, self.Max0, 1001)

        if self.is_spherical:
            query_v2_range = [self.trench_center_50km - (query_dist + query_dist_span)/self.Max0,\
                            self.trench_center_50km - query_dist/self.Max0]
        else:
            query_v2_range = [self.trench_center_50km - query_dist - query_dist_span,\
                            self.trench_center_50km - query_dist]
            
        query_v2s = np.linspace(query_v2_range[0], query_v2_range[1], 11)

        # loop and interpolate along v0
        # take the average dynamic pressure value
        query_das = np.zeros(query_v0s.size)
        for i, query_v0 in enumerate(query_v0s):
            da_array = da_func(np.full(query_v2s.shape, query_v0)/self.Max0, query_v2s/self.Max2)
            query_das[i] = np.mean(da_array)

        # export to file
        filename = "da_profile_%05d.txt" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        np.savetxt(filepath, np.column_stack((query_v0s, query_das)), fmt="%.6f", delimiter="\t")
        print("saved file: %s" % filepath)
        

    def get_slab_depth(self):
        # Get slab depth
        assert(self.iso_volume_lower is not None)
        points = self.iso_volume_lower.points
        if self.geometry == "chunk":
            l0, _, _ = cartesian_to_spherical(points[:, 0], points[:, 1], points[:, 2])
        else:
            l0 = points[:, 2]
        slab_depth = self.Max0 - np.min(l0)
        self.slab_depth  = slab_depth

        return slab_depth





    def make_boundary_cartesian(self, **kwargs):
        """
        Generate and save the six boundary surfaces for a cartesian box model domain.

        This method:
            - Constructs structured surface:
            - Combines them into a single surface and exports to a `.vtu` file.
        """
        marker_coordinates = kwargs.get("marker_coordinates", None)
        boundary_range = kwargs.get("boundary_range",\
            [[self.Min0, self.Max0], [self.Min1, self.Max1], [self.Min2, self.Max2]])

        # Chunk parameters
        x_min = boundary_range[2][0]
        x_max = boundary_range[2][1]
        y_min = boundary_range[1][0]
        y_max = boundary_range[1][1]
        z_min = boundary_range[0][0]
        z_max = boundary_range[0][1]

        n_x = 2
        n_y = 2
        n_z = 2

        # Get the maker positions
        marker_x = None; marker_y = None; marker_z = None
        if marker_coordinates is not None:
            assert(isinstance(marker_coordinates, dict))

            marker_x = marker_coordinates['x']
            marker_y = marker_coordinates['y']
            marker_z = marker_coordinates['z']

        # Faces and marker points:
        surfaces = []
        if marker_coordinates is not None:
            marker_points = []

        # x walls (x = min, max)
        x_edges = [x_min, x_max]
        for x_edge in x_edges:
            y_vals = np.linspace(y_min, y_max, n_x)
            z_vals = np.linspace(z_min, z_max, n_z)
            y_grid, z_grid = np.meshgrid(y_vals, z_vals)
            x_edge_grid = np.full(y_grid.shape, x_edge)
            surfaces.append(pv.StructuredGrid(x_edge_grid, y_grid, z_grid))

        # y walls (y = min, max)
        y_edges = [y_min, y_max]
        for y_edge in y_edges:
            x_vals = np.linspace(x_min, x_max, n_x)
            z_vals = np.linspace(z_min, z_max, n_z)
            x_grid, z_grid = np.meshgrid(x_vals, z_vals)
            y_edge_grid = np.full(x_grid.shape, y_edge)
            surfaces.append(pv.StructuredGrid(x_grid, y_edge_grid, z_grid))
        
        # x walls (x = min, max)
        z_edges = [z_min, z_max]
        for z_edge in z_edges:
            x_vals = np.linspace(x_min, x_max, n_x)
            y_vals = np.linspace(y_min, y_max, n_x)
            x_grid, y_grid = np.meshgrid(x_vals, y_vals)
            z_edge_grid = np.full(x_grid.shape, z_edge)
            surfaces.append(pv.StructuredGrid(x_grid, y_grid, z_edge_grid))
        
        # x axis markers 
        if marker_coordinates is not None:
            points_array = np.array([[], [], []]).T
            y_grid_marker, z_grid_marker = np.meshgrid((y_min, y_max), (z_min, z_max))
            for x in marker_x:
                x_grid_marker = np.full(y_grid_marker.shape, x)
                new_points_array = np.column_stack([
                     x_grid_marker.ravel(),
                     y_grid_marker.ravel(),
                     z_grid_marker.ravel()
                ])
                points_array = np.concatenate([points_array, new_points_array], axis=0)

            marker_points.append(pv.PolyData(points_array))

        # y axis markers
        if marker_coordinates is not None:
            points_array = np.array([[], [], []]).T
            x_grid_marker, z_grid_marker = np.meshgrid((x_min, x_max), (z_min, z_max))
            for y in marker_y:
                y_grid_marker = np.full(x_grid_marker.shape, y)
                new_points_array = np.column_stack([
                     x_grid_marker.ravel(),
                     y_grid_marker.ravel(),
                     z_grid_marker.ravel()
                ])
                points_array = np.concatenate([points_array, new_points_array], axis=0)

            marker_points.append(pv.PolyData(points_array))
        
        # z axis markers
        if marker_coordinates is not None:
            points_array = np.array([[], [], []]).T
            x_grid_marker, y_grid_marker = np.meshgrid((x_min, x_max), (y_min, y_max))
            for z in marker_z:
                z_grid_marker = np.full(x_grid_marker.shape, z)
                new_points_array = np.column_stack([
                     x_grid_marker.ravel(),
                     y_grid_marker.ravel(),
                     z_grid_marker.ravel()
                ])
                points_array = np.concatenate([points_array, new_points_array], axis=0)

            marker_points.append(pv.PolyData(points_array))

        # Combine all surfaces and marker points
        full_surface = surfaces[0]
        for s in surfaces[1:]:
            full_surface = full_surface.merge(s)

        full_marker_point = None
        if marker_coordinates is not None: 
            full_marker_point = marker_points[0]
            for p in marker_points[1:]:
                full_marker_point = full_marker_point.merge(p)

        # Save to file
        filename = "model_boundary.vtu"
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        full_surface.save(filepath)
        print("saved file: %s" % filepath)

        # deal with the annotation points
        if marker_coordinates is not None: 
            filename = "model_boundary_marker_points.vtp"
            filepath = os.path.join(self.pyvista_outdir, "..", filename)
            full_marker_point.save(filepath)
            print("saved file: %s" % filepath)

    def make_boundary_spherical(self, **kwargs):
        """
        Generate and save the six boundary surfaces for a spherical shell model domain.

        This method:
            - Constructs structured surfaces for:
                - outer sphere (r = Ro),
                - inner sphere (r = core-mantle boundary),
                - latitudinal walls (lat = min and max),
                - longitudinal walls (lon = min and max).
            - Combines them into a single surface and exports to a `.vtu` file.
        """

        marker_coordinates = kwargs.get("marker_coordinates", None)
        boundary_range = kwargs.get("boundary_range",\
            [[self.Min0, self.Max0], [0.0, 35.972864236749224 * np.pi / 180.0], [self.Min2, self.Max2]])

        # Chunk parameters
        r_inner = boundary_range[0][0]
        r_outer = boundary_range[0][1]
        lat_min = boundary_range[1][0]
        lat_max = boundary_range[1][1]
        lon_min = boundary_range[2][0]
        lon_max = boundary_range[2][1]

        # Resolution (adjust for finer mesh)
        n_r = 2
        n_lat = 100
        n_lon = 100

        # Get the maker positions
        marker_r = None; marker_lon = None; marker_lat = None
        if marker_coordinates is not None:
            assert(isinstance(marker_coordinates, dict))

            marker_r = marker_coordinates['r']
            marker_lon = marker_coordinates['lon']
            marker_lat = marker_coordinates['lat']

        # Helper function: spherical to Cartesian
        def sph2cart(r, lat, lon):
            x = r * np.cos(lat) * np.cos(lon)
            y = r * np.cos(lat) * np.sin(lon)
            z = r * np.sin(lat)
            return x, y, z

        # Function to create a surface patch between two r values, sweeping lat/lon
        def make_sphere_shell(r, lat_range, lon_range, n_lat, n_lon):
            lat = np.linspace(*lat_range, n_lat)
            lon = np.linspace(*lon_range, n_lon)
            lon_grid, lat_grid = np.meshgrid(lon, lat)
            x, y, z = sph2cart(r, lat_grid, lon_grid)
            return pv.StructuredGrid(x, y, z)
        
        # Function to create a surface patch between two r values, sweeping lat/lon
        def make_sphere_shell_marker_points(r, lat_range, lon_range, m_lat, m_lon):
            lon_grid_lon_edge, lat_grid_lon_edge = np.meshgrid(lon_range, m_lat)
            lon_grid_lat_edge, lat_grid_lat_edge = np.meshgrid(m_lon, lat_range)
            x_lon_edge, y_lon_edge, z_lon_edge = sph2cart(r, lat_grid_lon_edge, lon_grid_lon_edge)
            x_lat_edge, y_lat_edge, z_lat_edge = sph2cart(r, lat_grid_lat_edge, lon_grid_lat_edge)

            # create the point cloud
            points_array = np.column_stack([
                x_lon_edge.ravel(),
                y_lon_edge.ravel(),
                z_lon_edge.ravel()
            ])
            point_cloud = pv.PolyData(points_array)
            
            points_array = np.column_stack([
                x_lat_edge.ravel(),
                y_lat_edge.ravel(),
                z_lat_edge.ravel()
            ])
            point_cloud = point_cloud.merge(pv.PolyData(points_array))
            
            return point_cloud

        # Faces and marker points:
        surfaces = []
        if marker_coordinates is not None:
            marker_points = []

        # Outer surface (r = outer)
        surfaces.append(make_sphere_shell(r_outer, (lat_min, lat_max), (lon_min, lon_max), n_lat, n_lon))
        if marker_coordinates is not None:
            marker_points.append(make_sphere_shell_marker_points(r_outer, (lat_min, lat_max), (lon_min, lon_max), marker_lat, marker_lon))

        # Inner surface (r = inner)
        surfaces.append(make_sphere_shell(r_inner, (lat_min, lat_max), (lon_min, lon_max), n_lat, n_lon))
        if marker_coordinates is not None:
            marker_points.append(make_sphere_shell_marker_points(r_inner, (lat_min, lat_max), (lon_min, lon_max), marker_lat, marker_lon))

        # Latitudinal walls (lat = min, max)
        lat_edges = [lat_min, lat_max]
        for lat_edge in lat_edges:
            r = np.linspace(r_inner, r_outer, n_r)
            lon = np.linspace(lon_min, lon_max, n_lon)
            lon_grid, r_grid = np.meshgrid(lon, r)
            x, y, z = sph2cart(r_grid, lat_edge, lon_grid)
            surfaces.append(pv.StructuredGrid(x, y, z))

        # Longitudinal walls (lon = min, max)
        lon_edges = [lon_min, lon_max]
        for lon_edge in lon_edges:
            r = np.linspace(r_inner, r_outer, n_r)
            lat = np.linspace(lat_min, lat_max, n_lat)
            lat_grid, r_grid = np.meshgrid(lat, r)
            x, y, z = sph2cart(r_grid, lat_grid, lon_edge)
            surfaces.append(pv.StructuredGrid(x, y, z))
        
        if marker_coordinates is not None:
            points_array = np.array([[], [], []]).T
            lon_grid_marker, lat_grid_marker = np.meshgrid((lon_min, lon_max), (lat_min, lat_max))
            for r in marker_r:
                x, y, z = sph2cart(r, lat_grid_marker, lon_grid_marker)
                new_points_array = np.column_stack([
                     x.ravel(),
                     y.ravel(),
                     z.ravel()
                ])
                points_array = np.concatenate([points_array, new_points_array], axis=0)

            marker_points.append(pv.PolyData(points_array))

        # Combine all surfaces and marker points
        full_surface = surfaces[0]
        for s in surfaces[1:]:
            full_surface = full_surface.merge(s)

        full_marker_point = None
        if marker_coordinates is not None: 
            full_marker_point = marker_points[0]
            for p in marker_points[1:]:
                full_marker_point = full_marker_point.merge(p)

        # Save to file
        filename = "model_boundary.vtu"
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        full_surface.save(filepath)
        print("saved file: %s" % filepath)

        # deal with the annotation points
        if marker_coordinates is not None: 
            filename = "model_boundary_marker_points.vtp"
            filepath = os.path.join(self.pyvista_outdir, "..", filename)
            full_marker_point.save(filepath)
            print("saved file: %s" % filepath)


    def create_cartesian_plane(self, depth, resolution=(50, 50), **kwargs):
        """
        Create a rectangular plane at a constant z-depth.

        Parameters:
            depth: constant depth for the plane
            resolution: (nx, ny) number of subdivisions in x and y
            filename: output file name (.vtp)
        """
        boundary_range = kwargs.get("boundary_range",\
            [[self.Min0, self.Max0], [self.Min1, self.Max1], [self.Min2, self.Max2]])
        # Create linspace
        x = np.linspace(boundary_range[2][0], boundary_range[2][1], resolution[0])
        y = np.linspace(boundary_range[1][0], boundary_range[1][1], resolution[1])
        x_grid, y_grid = np.meshgrid(x, y)

        # Constant z
        z_grid = np.full_like(x_grid, self.Max0 - depth)

        # Combine to points
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (resolution[0], resolution[1], 1)

        # cast to unstructured
        unstructured = grid.cast_to_unstructured_grid()

        # Save and return
        filename="plane_%.1fkm.vtu" % (depth/1e3)
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        unstructured.save(filepath)
        print("saved file: %s" % filepath)


    def create_spherical_plane(self, depth, resolution=(50, 50), **kwargs):
        """
        Create a spherical plane at a constant radius.

        Parameters:
            depth: offset from base radius (final radius = radius + depth)
            resolution: (n_theta, n_phi) grid resolution
        """
        boundary_range = kwargs.get("boundary_range",\
            [[self.Min0, self.Max0], [self.Min1, self.Max1], [self.Min2, self.Max2]])

        # Compute actual surface radius
        surface_radius = self.Max0 - depth

        # Create theta and phi arrays (already in radians)
        theta = np.linspace(np.pi/2.0-boundary_range[1][1], np.pi/2.0-boundary_range[1][0], resolution[0])
        phi = np.linspace(boundary_range[2][0], boundary_range[2][1], resolution[1])
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Spherical to Cartesian conversion
        x = surface_radius * np.sin(theta_grid) * np.cos(phi_grid)
        y = surface_radius * np.sin(theta_grid) * np.sin(phi_grid)
        z = surface_radius * np.cos(theta_grid)

        # Combine to points
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (resolution[0], resolution[1], 1)
        
        # cast to unstructured
        unstructured = grid.cast_to_unstructured_grid()
        
        # Save and return
        filename="plane_%.1fkm.vtu" % (depth/1e3)
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        unstructured.save(filepath)
        print("saved file: %s" % filepath)


def get_trench_position_from_file(pyvista_outdir, pvtu_step, geometry, **kwargs):
    '''
    Retrieve the trench center position from a previously generated VTK PolyData (.vtp) file.

    Parameters:
        pyvista_outdir (str): 
            Path to the directory containing PyVista-generated output files.
        pvtu_step (int): 
            The simulation step index used to construct the filename.
        geometry (str): 
            Simulation geometry type. If "chunk", coordinates are treated as spherical; 
            otherwise, they are treated as Cartesian.
        **kwargs: 
            trench_depth (float, optional): Depth of the trench in km. Default is 0.0.

    Returns:
        float:
            The unified coordinate value representing the trench center position. 
            Computed from the first trench point in the VTK file.
    '''
    # set variables for geometry
    if geometry == "chunk":
        is_spherical = True
    else:
        is_spherical = False

    trench_depth = kwargs.get("trench_depth", 0.0)
    filename = "trench_d%.2fkm_%05d.vtp" % (trench_depth/1e3, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)

    point_cloud_tr = pv.read(filepath)
    trench_points = point_cloud_tr.points
    _, trench_center, _ = PUnified.points2unified3(trench_points[0, :], is_spherical, False)
    return trench_center


def get_slab_depth_from_file(pyvista_outdir, pvtu_step, geometry, Max0, field_name):
    '''
    Retrieve the slab depth from a previously generated VTK PolyData (.vtp) file.

    Parameters:
        pyvista_outdir (str): 
            Path to the directory containing PyVista-generated output files.
        pvtu_step (int): 
            The simulation step index used to construct the filename.
        geometry (str): 
            Simulation geometry type. If "chunk", coordinates are treated as spherical; 
            otherwise, they are treated as Cartesian.
        Max0 (float): 
            The outer radius or box thickness of the model domain (in meters).
        field_name (str): 
            The prefix for the slab surface file, e.g., "sp_crust" or "sp_mantle". 
            The filename is constructed as "{field_name}_surface_{step:05d}.vtp".

    Returns:
        float:
            The depth of the slab (in meters), computed as `Max0 - min(r_slab)`, 
            where `r_slab` is the radial coordinate of the slab surface.
    '''
    # set variables for geometry
    if geometry == "chunk":
        is_spherical = True
    else:
        is_spherical = False
    
    filename = "%s_surface_%05d.vtp" % (field_name, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    point_cloud_tr = pv.read(filepath)
    points = point_cloud_tr.points
    
    r_slab, _, _ = PUnified.points2unified3(points[0, :], is_spherical, False)
    slab_depth = Max0 - np.min(r_slab)
    return slab_depth


def get_slab_dip_angle_from_file(pyvista_outdir, pvtu_step, geometry, Max0, field_name, depth0, depth1):
    '''
    Get the position of trench from a file generated previously
    Inputs:
        depth0, depth1 - get slab dip angle between these depth
    '''
    filename = "%s_surface_%05d.vtp" % (field_name, pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    point_cloud_tr = pv.read(filepath)
    points = point_cloud_tr.points

    dip_angle = get_slab_dip_angle(points, geometry, Max0, depth0, depth1)

    return dip_angle


def get_slab_dip_angle_deprecated_0(points, geometry, Max0, depth0, depth1):

    d0 = 5e3 # tolerance along dimention 0
    if geometry == "chunk":
        d1 = 0.1 * np.pi / 180.0  # tolerance along dimention 1
        l0_mean = Max0 - (depth0 + depth1)/2.0
        l0, theta, l2 = cartesian_to_spherical(points[:, 0], points[:, 1], points[:, 2])
        l1 = np.pi/2.0 - theta
        mask0 = ((np.abs((Max0 - l0) - depth0) < d0) & (l1 < d1))
        l2_mean_0 = np.average(l2[mask0])
        mask1 = ((np.abs((Max0 - l0) - depth1) < d0) & (l1 < d1))
        l2_mean_1 = np.average(l2[mask1])
        dip_angle = np.arctan2(depth1 - depth0, l0_mean * (l2_mean_1 - l2_mean_0))
    else:
        d1 = 10e3
        l0 = points[:, 2]; l1 = points[:, 1]; l2 = points[:, 0]
        mask0 = ((np.abs((Max0 - l0) - depth0) < d0) & (l1 < d1))
        l2_mean_0 = np.average(l2[mask0])
        mask1 = ((np.abs((Max0 - l0) - depth1) < d0) & (l1 < d1))
        l2_mean_1 = np.average(l2[mask1])
        dip_angle = np.arctan2(depth1 - depth0, l2_mean_1 - l2_mean_0)

    return dip_angle


class PLOT_CASE_RUN_THD():
    '''
    Plot case run result
    Attributes:
        case_path(str): path to the case
        Case_Options: options for case
        kwargs:
            time_range
            step(int): if this is given as an int, only plot this step

    Returns:
        -
    '''
    def __init__(self, case_path, **kwargs):
        '''
        Initiation
        Inputs:
            case_path - full path to a 3-d case
            kwargs
        '''

        self.case_path = case_path
        self.kwargs = kwargs
        print("PlotCaseRun in ThDSubduction: operating")
        step = kwargs.get('step', None)
        last_step = kwargs.get('last_step', 3)

        # initiate options
        self.Case_Options = CASE_OPTIONS(self.case_path)
        self.Case_Options.Interpret(**self.kwargs)
        self.Case_Options.SummaryCaseVtuStep(os.path.join(self.case_path, "summary.csv"))

    def ProcessPyvista(self, **kwargs):
        '''
        pyvista processing
        '''
        # use a list to record whether files are found
        file_found_list = []
        for vtu_step in self.Case_Options.options['GRAPHICAL_STEPS']:
            pvtu_step = vtu_step + int(self.Case_Options.options['INITIAL_ADAPTIVE_REFINEMENT'])
            try:
                ProcessVtuFileThDStep(self.case_path, pvtu_step, self.Case_Options, **kwargs)
                file_found_list.append(True)
            except FileNotFoundError:
                file_found_list.append(False)
                pass
        return file_found_list
            

    def GenerateParaviewScript(self, ofile_list, additional_options, **kwargs):
        '''
        generate paraview script
        Inputs:
            ofile_list - a list of file to include in paraview
            additional_options - options to append
        '''
        animation = kwargs.get("animation", False)
        require_base = self.kwargs.get('require_base', True)
        for ofile_base in ofile_list:
            # Different file name if make animation
            if animation:
                snapshot = self.kwargs["steps"][0] + int(self.Case_Options.options['INITIAL_ADAPTIVE_REFINEMENT'])
                odir = os.path.join(self.case_path, 'paraview_scripts', "%05d" % snapshot)
                if not os.path.isdir(odir):
                    os.mkdir(odir)
                ofile = os.path.join(self.case_path, 'paraview_scripts', "%05d" % snapshot, ofile_base)
            else:
                ofile = os.path.join(self.case_path, 'paraview_scripts', ofile_base)
            # Read base file
            paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts',"ThDSubduction", ofile_base)
            if require_base:
                paraview_base_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')  # base.py : base file
                self.Case_Options.read_contents(paraview_base_script, paraview_script)  # this part combines two scripts
            else:
                self.Case_Options.read_contents(paraview_script)  # this part combines two scripts
            # Update additional options
            self.Case_Options.options.update(additional_options)
            if animation:
                self.Case_Options.options["ANIMATION"] = "True"
            # Generate scripts
            self.Case_Options.substitute()  # substitute keys in these combined file with values determined by Interpret() function
            ofile_path = self.Case_Options.save(ofile, relative=False)  # save the altered script
            print("\t File generated: %s" % ofile_path)


def clip_grid_by_xyz_range(grid, x_range=None, y_range=None, z_range=None):
    '''
    Clip the grid by given x, y and z range
    '''
    points = grid.points
    mask = np.ones(points.shape[0], dtype=bool)

    if x_range:
        mask &= (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
    if y_range:
        mask &= (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    if z_range:
        mask &= (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])

    return grid.extract_points(mask, adjacent_cells=True)


def clip_grid_by_spherical_range(grid, r_range=None, theta_range=None, phi_range=None):
    '''
    Clip the grid by given r, theta and phi range
    '''
    points = grid.points

    # Convert Cartesian to spherical
    r, theta, phi = cartesian_to_spherical(points[:, 0], points[:, 1], points[:, 2])

    mask = np.ones(points.shape[0], dtype=bool)

    if r_range:
        mask &= (r >= r_range[0]) & (r <= r_range[1])
    if theta_range:
        mask &= (theta >= theta_range[0]) & (theta <= theta_range[1])
    if phi_range:
        mask &= (phi >= phi_range[0]) & (phi <= phi_range[1])

    return grid.extract_points(mask, adjacent_cells=True)


def ProcessVtuFileThDStep(case_path, pvtu_step, Case_Options, **kwargs):
    '''
    Process with pyvsita for a single step
    Inputs:
        case_path - full path of a 3-d case
        pvtu_step - pvtu_step of vtu output files
        Case_Options - options for the case
        kwargs
            threshold_lower - threshold for lower slab composition
    '''
    # case options
    Case_Options = CASE_OPTIONS(case_path)
    Case_Options.Interpret()
    Case_Options.SummaryCaseVtuStep(os.path.join(case_path, "summary.csv"))

    geometry = Case_Options.options["GEOMETRY"]

    # time step and index
    idx = Case_Options.summary_df["Vtu snapshot"] == pvtu_step
    _time = Case_Options.summary_df.loc[idx, "Time"].values[0]

    # additional options
    threshold_upper = kwargs.get("threshold_upper", 0.8)
    threshold_lower = kwargs.get("threshold_lower", 0.8)
    threshold_edge = kwargs.get("threshold_edge", 0.8)
    threshold_metastable = kwargs.get("threshold_metastable", 0.8)
    # Whether to clip the domain and the depth of the clip
    do_clip = kwargs.get("do_clip", False)
    clip_depth = kwargs.get("clip_depth", 1000e3)
    extract_trench_at_additional_depths=  kwargs.get("extract_trench_at_additional_depths", None)
    n_pieces = kwargs.get("n_pieces", None)
    i_piece = kwargs.get("i_piece", None)
    odir = kwargs.get("odir", os.path.join(case_path, "pyvista_outputs"))
    only_initialization = kwargs.get("only_initialization", False)
    use_old_dip_angle_method = kwargs.get("use_old_dip_angle_method", False)
    
    outputs = {}

    # geometry information
    if geometry == "chunk":
        marker_intervals = [500e3, 10.0*np.pi/180.0, 10.0*np.pi/180.0]
    else:
        marker_intervals = [500e3, 1000e3, 1000e3]

    if geometry == "chunk":
        Max0 = float(Case_Options.options["OUTER_RADIUS"])
        Min0 = float(Case_Options.options["INNER_RADIUS"])
        Max1 = float(Case_Options.options["MAX_LATITUDE"])
        # Max1 = np.pi / 2.0
        Max2 = float(Case_Options.options["MAX_LONGITUDE"])
        trench_initial = float(Case_Options.options["TRENCH_INITIAL"]) * np.pi/180.0
    else:
        Max0 = float(Case_Options.options["BOX_THICKNESS"])
        Min0 = 0.0
        Max1 = float(Case_Options.options["BOX_WIDTH"])
        Max2 = float(Case_Options.options["BOX_LENGTH"])
        trench_initial = float(Case_Options.options["TRENCH_INITIAL"])

    
    # output directory
    if not os.path.isdir(os.path.join(case_path, "pyvista_outputs")):
        os.mkdir(os.path.join(case_path, "pyvista_outputs"))

    slice_depth = 200e3

    # initialize the class
    # fix the meaning of Max1 - latitude
    if geometry == "chunk":
        l0_section = clip_depth
        l1_section = 10.0 * np.pi / 180.0
        l2_section = 10.0 * np.pi / 180.0
    else:
        l0_section = clip_depth
        l1_section = np.max([float(Case_Options.options["SLAB_EXTENTS_FULL"]), 1000e3])
        l2_section = 1000e3
    tolerance = 1e-6
    clip_l0_min = Max0-l0_section
    clip_l0_max = Max0
    clip_l1_min = 0.0
    clip_l1_max = l1_section*2.0
    clip_l2_min = max(np.ceil(trench_initial / l2_section) * l2_section - 2 * l2_section, 0.0)
    clip_l2_max = min(np.ceil(trench_initial / l2_section) * l2_section + 2 * l2_section, Max2)
        
    if do_clip:
        clip = [[clip_l0_min-tolerance, clip_l0_max+tolerance] , [clip_l1_min-tolerance, clip_l1_max+tolerance], [clip_l2_min-tolerance, clip_l2_max+tolerance]]
        print("\tclip: ", clip)
    else:
        clip = None
    boundary_range=[[clip_l0_min, clip_l0_max] , [clip_l1_min, clip_l1_max], [clip_l2_min, clip_l2_max]]

    
    # initiate PYVISTA_PROCESS_THD class
    config = {"geometry": geometry, "Max0": Max0, "Min0": Min0, "Max1": Max1, "Max2": Max2, "time": _time}
    kwargs = {"clip": clip, "pyvista_outdir": os.path.join(odir, "%05d" % pvtu_step)}
    PprocessThD = PYVISTA_PROCESS_THD(os.path.join(case_path, "output", "solution"), config, **kwargs)

    # only do intialization of the class 
    if only_initialization:
        return PprocessThD, outputs

    # make domain boundary
    do_output = (n_pieces is None or i_piece is None or i_piece < 0)
    if do_output:
        if geometry == "chunk":
            # p_marker_coordinates = {"r": 6371e3 - np.arange(0, 6000e3, 1000e3), "lon": np.arange(0, 90, 10)*np.pi/180.0, "lat": np.arange(0, 90.0, 10.0)*np.pi/180.0}
            # d1 = np.floor((Max0 - Min0) / marker_intervals[0]) * marker_intervals[0]
            # lat1 = np.floor(Max1 / marker_intervals[1]) * marker_intervals[1]
            # lon1 = np.floor(Max2 / marker_intervals[2]) * marker_intervals[2]

            p_marker_coordinates = {"r": Max0 - np.arange(0, Max0 - Min0, marker_intervals[0]),\
                    "lat": np.arange(0, Max1, marker_intervals[1]),\
                "lon": np.arange(0, Max2, marker_intervals[2])}
            PprocessThD.make_boundary_spherical(marker_coordinates=p_marker_coordinates,\
                boundary_range=boundary_range)
            PprocessThD.create_spherical_plane(660e3, boundary_range=boundary_range)
        else:
            p_marker_coordinates = {"x": np.arange(0, Max2, marker_intervals[2]),\
                    "y": np.arange(0, Max1, marker_intervals[1]),\
                "z": Max0 - np.arange(0, Max0 - Min0, marker_intervals[0])}
            PprocessThD.make_boundary_cartesian(marker_coordinates=p_marker_coordinates,\
                boundary_range=boundary_range)
            PprocessThD.create_cartesian_plane(660e3, boundary_range=boundary_range)

    if n_pieces is None:
        # read vtu file
        PprocessThD.read(pvtu_step)
        # slice at center
        PprocessThD.slice_center()
        PprocessThD.slice_center(boundary_range=boundary_range)
        # slice at surface
        PprocessThD.slice_surface()
        # slice at depth
        PprocessThD.slice_at_depth(depth=slice_depth, rdiff=40e3)
        # extract sp_upper composition beyond a threshold
        PprocessThD.extract_iso_volume_upper(threshold=threshold_upper)
        # extract sp_lower composition beyond a threshold
        PprocessThD.extract_iso_volume_lower(threshold=threshold_lower)
        if Case_Options.options["MODEL_TYPE"] == "mow":
            PprocessThD.extract_iso_volume_metastable(options=Case_Options.options.copy(), threshold=threshold_metastable)
        # extract plate_edge composition beyond a threshold
        PprocessThD.extract_plate_edge(threshold=threshold_edge)
        process = psutil.Process(os.getpid())
        print("Memory Usage: ", process.memory_info().rss / 1024**2, "MB")
    else:
        if i_piece is None:
        # process piecewise
            for piece in range(n_pieces):
                PprocessThD.read(pvtu_step, piece=piece)
                PprocessThD.process_piecewise("slice_center", ["sliced", "sliced_u"])
                PprocessThD.process_piecewise("slice_surface", ["sliced_shell"])
                PprocessThD.process_piecewise("slice_at_depth", ["sliced_depth"], depth=slice_depth, r_diff=40e3)
                PprocessThD.process_piecewise("extract_iso_volume_upper", ["iso_volume_upper"], threshold=threshold_upper)
                PprocessThD.process_piecewise("extract_iso_volume_lower", ["iso_volume_lower"], threshold=threshold_lower)
                PprocessThD.process_piecewise("extract_plate_edge", ["iso_plate_edge"], threshold=threshold_edge)
                if Case_Options.options["MODEL_TYPE"] == "mow":
                    PprocessThD.process_piecewise("extract_iso_volume_metastable", ["grid_metastable", "grid_metastable_slab"],\
                                              options=Case_Options.options.copy(), threshold=threshold_metastable)
                process = psutil.Process(os.getpid())
                print("Memory Usage piecewise %d: " % piece, process.memory_info().rss / 1024**2, "MB")
        else:
            if i_piece >= 0:
                # process a single piece at a time.
                # at the end, a single-piece output will be exported
                PprocessThD.read(pvtu_step, piece=i_piece)
                PprocessThD.process_piecewise("slice_center", ["sliced", "sliced_u"])
                PprocessThD.process_piecewise("slice_surface", ["sliced_shell"])
                PprocessThD.process_piecewise("slice_at_depth", ["sliced_depth"], depth=slice_depth, r_diff=40e3)
                PprocessThD.process_piecewise("extract_iso_volume_upper", ["iso_volume_upper"], threshold=threshold_upper)
                PprocessThD.process_piecewise("extract_iso_volume_lower", ["iso_volume_lower"], threshold=threshold_lower)
                PprocessThD.process_piecewise("extract_plate_edge", ["iso_plate_edge"], threshold=threshold_edge)
                if Case_Options.options["MODEL_TYPE"] == "mow":
                    PprocessThD.process_piecewise("extract_iso_volume_metastable", ["grid_metastable", "grid_metastable_slab"],\
                                              options=Case_Options.options.copy(), threshold=threshold_metastable)
                process = psutil.Process(os.getpid())
                print("Memory Usage piecewise %d: " % i_piece, process.memory_info().rss / 1024**2, "MB")
                PprocessThD.export_piecewise(i_piece)
            
                # only continue if we pass a negative i_piece
                return PprocessThD, {}

        if i_piece < 0: 
            # Set class attributes
            PprocessThD.pvtu_step = pvtu_step
            PprocessThD.threshold_lower = threshold_lower
            PprocessThD.threshold_upper = threshold_upper
            PprocessThD.threshold_edge = threshold_edge
            if Case_Options.options["MODEL_TYPE"] == "mow":
                PprocessThD.threshold_metastable = threshold_metastable
            
            # Load the pieces
            for piece in range(n_pieces):
                PprocessThD.import_piecewise(piece)

        process = psutil.Process(os.getpid())
        print("Memory Usage (after processing files): ", process.memory_info().rss / 1024**2, "MB")
        if do_output:
            PprocessThD.combine_pieces()
            PprocessThD.write_key_to_file("sliced", "slice_center_unbounded", "vtp")
            PprocessThD.write_key_to_file("sliced_u", "slice_center", "vtu")
            PprocessThD.write_key_to_file("sliced_shell", "slice_outer", "vtu")
            PprocessThD.write_key_to_file("sliced_depth", "slice_depth_%.1fkm" % (slice_depth/1e3), "vtu")
            PprocessThD.write_key_to_file("iso_volume_upper", "sp_upper_above_%.2f" % (threshold_upper), "vtu")
            PprocessThD.write_key_to_file("iso_volume_lower", "sp_lower_above_%.2f" % (threshold_lower), "vtu")
            PprocessThD.write_key_to_file("iso_plate_edge", "plate_edge_above_%.2f" % (threshold_edge), "vtu")
            if Case_Options.options["MODEL_TYPE"] == "mow":
                PprocessThD.write_key_to_file("grid_metastable", "metastable_below_%.2f" % (threshold_metastable), "vtu")
                PprocessThD.write_key_to_file("grid_metastable_slab", "metastable_slab_below_%.2f" % (threshold_metastable), "vtu")
            process = psutil.Process(os.getpid())
            print("Memory Usage (after writing files): ", process.memory_info().rss / 1024**2, "MB")



    # extract slab surface and trench position, using "sp_upper" composition
    PprocessThD.extract_slab_interface("sp_upper")
    # extract slab edge
    PprocessThD.extract_plate_edge_surface()
    # filter the slab lower points
    PprocessThD.filter_slab_lower_points()
    # extract slab surface using "sp_lower" composition
    PprocessThD.extract_slab_interface("sp_lower")
    if Case_Options.options["MODEL_TYPE"] == "mow":
        PprocessThD.extract_metastable_area(Case_Options.options.copy(), threshold=threshold_metastable)

    # analysis
    # get slab depth, trenc position and slab dip angle
    PprocessThD.get_slab_depth()
    if use_old_dip_angle_method:
        PprocessThD.extract_slab_dip_angle_deprecated_0()
    else:
        PprocessThD.extract_slab_dip_angle()
    PprocessThD.extract_slab_trench()
    PprocessThD.extract_sp_velocity()

    # extract reference dynamic profile
    if Case_Options.options["HAS_DYNAMIC_PRESSURE"] == '1':
        PprocessThD.extract_slice_ref_da_profile()

    # extract outputs
    assert(PprocessThD.trench_center is not None)
    outputs["trench_center"] = PprocessThD.trench_center
    assert(PprocessThD.trench_center_50km is not None)
    outputs["trench_center_50km"] = PprocessThD.trench_center_50km
    assert(PprocessThD.slab_depth is not None)
    outputs["slab_depth"] = PprocessThD.slab_depth

    assert(PprocessThD.dip_100_center is not None)
    outputs["dip_100_center"] = PprocessThD.dip_100_center
    assert(PprocessThD.sp_velocity is not None)
    outputs["Sp velocity"] = PprocessThD.sp_velocity
    if Case_Options.options["MODEL_TYPE"] == "mow":
        assert(PprocessThD.metastable_volume is not None)
        outputs["Mow volume"] = PprocessThD.metastable_volume
        assert(PprocessThD.metastable_volume_cold is not None)
        outputs["Mow volume slab"] = PprocessThD.metastable_volume_cold
        assert(PprocessThD.metastable_area_center is not None)
        outputs["Mow area center"] = PprocessThD.metastable_area_center
        assert(PprocessThD.metastable_area_cold_center is not None)
        outputs["Mow area slab center"] = PprocessThD.metastable_area_cold_center

    return PprocessThD, outputs


def PlotCaseRunTwoD1(case_path, **kwargs):
    '''
    Plot case run result
    Inputs:
        case_path(str): path to the case
        kwargs:
            step(int): if this is given as an int, only plot this step
            last_step: number of last steps to plot
    Returns:
        -
    '''
    step = kwargs.get('step', None)
    last_step = kwargs.get('last_step', 3)
    rotation_plus = kwargs.get("rotation_plus", 0.0)
    additional_options = kwargs.get("additional_options", {})
    
    print("%s: start" % func_name())
    # get case parameters
    prm_path = os.path.join(case_path, 'output', 'original.prm')

    # steps to plot: here I use the keys in kwargs to allow different
    # options: by steps, a single step, or the last step
    if type(step) == int:
        kwargs["steps"] = [step]
    elif type(step) == list:
        kwargs["steps"] = step
    elif type(step) == str:
        kwargs["steps"] = step
    else:
        kwargs["last_step"] = last_step

    # Inititiate the class and intepret the options
    # Note that all the options defined by kwargs is passed to the interpret function
    Case_Options_2d = CASE_OPTIONS_TWOD1(case_path)
    Case_Options_2d.Interpret(**kwargs)

    for key, value in additional_options.items():
        Case_Options_2d.options[key] = value

    # generate scripts base on the method of plotting
    odir = os.path.join(case_path, 'paraview_scripts')
    if not os.path.isdir(odir):
        os.mkdir(odir)
    print("Generating paraview scripts")
    py_script = 'slab1.py'
    ofile = os.path.join(odir, py_script)
    paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'ThDSubduction', py_script)
    paraview_script_base = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')
    Case_Options_2d.read_contents(paraview_script_base, paraview_script)  # combine these two scripts
    Case_Options_2d.substitute()

    ofile_path = Case_Options_2d.save(ofile, relative=True)

    return Case_Options_2d

def ProcessVtuFileTwoDStep(case_path, pvtu_step, Case_Options, **kwargs):
    """
    Inputs:
        case_path - full path of a 3-d case
        pvtu_step - pvtu_step of vtu output files
        Case_Options - options for the case
    """
    from hamageolib.utils.geometry_utilities import rotate_vec_cart_to_sph

    output_dict = {}
    geometry = Case_Options.options["GEOMETRY"]
    Min0 = Case_Options.options["INNER_RADIUS"]
    Max0 = Case_Options.options["OUTER_RADIUS"]
    Min2 = 0.0
    Max2 = Case_Options.options["XMAX"] * np.pi/180.0 # longtitude in radian

    idx = Case_Options.summary_df["Vtu snapshot"] == pvtu_step
    _time = Case_Options.summary_df.loc[idx, "Time"].values[0]

    # constant values
    # T_slab - value used to extract metastable region in cold slab
    T_slab = 900 + 273.15 # K
    
    # output directory
    if not os.path.isdir(os.path.join(case_path, "pyvista_outputs")):
        os.mkdir(os.path.join(case_path, "pyvista_outputs"))
    pyvista_outdir = os.path.join(case_path, "pyvista_outputs", "%05d" % pvtu_step)
    if not os.path.isdir(pyvista_outdir):
        os.mkdir(pyvista_outdir)

    # Read data
    # append a "radius" field and a "sp_total" field
    start = time.time()
    pvtu_step = pvtu_step
    
    pvtu_filepath = os.path.join(case_path, "output", "solution", "solution-%05d.pvtu" % pvtu_step)
    my_assert(os.path.isfile(pvtu_filepath), FileNotFoundError, "File %s is not found" % pvtu_filepath)

    grid = pv.read(pvtu_filepath)

    points = grid.points
    if geometry == "chunk":
        radius = np.linalg.norm(points, axis=1)
    else:
        radius = points[:, 2]
    grid["radius"] = radius
    grid["sp_total"] = grid["spcrust"] + grid["spharz"]
    
    end = time.time()
    print("%s: Read file takes %.1f s" % (func_name(), end - start))

    # append a cell data grid if the model type requires doing so
    grid_c = None
    if Case_Options.options["MODEL_TYPE"] == "mow":
        start = time.time()
        grid_c = grid.point_data_to_cell_data(pass_point_data=False, categorical=False)
        end = time.time()
        print("%s: Map cell data takes %.1f s" % (func_name(), end - start))
    
    # take iso-volumes
    start = time.time()

    threshold = 0.5
    
    iso_volume_total = grid.threshold(value=threshold, scalars="sp_total", invert=False)
    
    end = time.time()
    print("%s: Making iso-volumes takes %.1f s" % (func_name(), end - start))

    # Extract slab surface points
    # First construct a KDTree from data
    # Then figure out the max phi value
    # Record the trench position, slab depth and save slab surface points
    start = time.time()
    d0 = 5e3
    vals0 = np.arange(Min0, Max0, d0)
    vals1 = np.pi/2.0
    vals2 = np.full(vals0.size, np.nan)

    slab_points = iso_volume_total.points
    slab_point_data = iso_volume_total.point_data

    if geometry == "chunk":
        # v0_u, _, v2_u = cartesian_to_spherical(*slab_points.T)
        v0_u, v2_u = cartesian_to_spherical_2d(slab_points[:, 0], slab_points[:, 1])
        rt_upper = np.vstack([v0_u/Max0]).T
    else:
        rt_upper = np.vstack([slab_points[:, 1]/Max0]).T
        v2_u = slab_points[:, 0]
    rt_tree = cKDTree(rt_upper)

    dr = 0.001
    for i, v0 in enumerate(vals0):
        query_pt = np.array([v0/Max0])
        idxs = rt_tree.query_ball_point(query_pt, r=dr)

        if not idxs:
            continue

        v2s = v2_u[idxs]
        max_v2 = np.max(v2s)

        if np.all(v2s <= max_v2):
            vals2[i] = max_v2
    
    mask = ~np.isnan(vals2)

    v0_surf = vals0[mask]
    v2_surf = vals2[mask]

    if geometry == "chunk":
        x = v0_surf * np.cos(v2_surf)
        y = v0_surf * np.sin(v2_surf)
        z = np.zeros(v0_surf.shape)
    else:
        x = v2_surf
        y = v0_surf
        z = np.zeros(v0_surf.shape)

    slab_surface_points = np.vstack([x, y, z]).T


    # derive trench center, slab depth and dip angle
    depth0 = 0.0; depth1 = 100e3
    if geometry == "chunk": 
        l0_mean = Max0 - (depth0 + depth1)/2.0
        l0, l2 = cartesian_to_spherical_2d(slab_surface_points[:, 0], slab_surface_points[:, 1])
        l2_1 = np.interp(Max0 - depth1, l0, l2)
        l2_0 = np.interp(Max0 - depth0, l0, l2)
        trench_center = l2[-1]
        trench_center_50km = np.interp(Max0 - 50e3, l0, l2)
        slab_depth = Max0 - np.min(l0)
        # dip angle
        mask0 = ((np.abs((Max0 - l0) - depth0) < d0))
        l2_mean_0 = np.average(l2[mask0])
        dip_angle = np.arctan2(depth1 - depth0, l0_mean * (l2_1 - l2_0))
    else:
        slab_depth = Max0 - np.min(slab_surface_points[:, 1])
        # dip angle
        l0 = slab_surface_points[:, 1]; l2 = slab_surface_points[:, 0]
        l2_1 = np.interp(Max0 - depth1, l0, l2)
        l2_0 = np.interp(Max0 - depth0, l0, l2)
        trench_center = slab_surface_points[-1, 0]
        trench_center_50km = np.interp(Max0 - 50e3, l0, l2)
        dip_angle = np.arctan2(depth1 - depth0, l2_1 - l2_0)
    
    output_dict["trench_center"] = trench_center
    output_dict["trench_center_50km"] = trench_center_50km
    output_dict["slab_depth"] = slab_depth
    output_dict["dip_100"] = dip_angle

    point_cloud = pv.PolyData(slab_surface_points)
    filename = "spcrust_surface_%05d.vtp" % pvtu_step
    filepath = os.path.join(pyvista_outdir, filename)
    point_cloud.save(filepath)
    print("Save file %s" % filepath)
    
    end = time.time()
    print("%s: Extract trench center takes %.1f s" % (func_name(), end - start))
    
    # plate movement, reuse the Kdtree in the last step
    # query the points base on a depth, and select points within a distance to the trench.
    # get the velocity and project to the phi / x direction
    # last, take the average
    start = time.time()

    query_plate_velocity_depth = 20e3 # depth to query m
    query_plate_velocity_dist = 500e3 # distance from trench to query
    query_plate_velocity_dist_span = 200e3 # range of distance from trench to query

    query_v0 = Max0 - query_plate_velocity_depth
    if geometry == "box":
        query_v2_range = [trench_center - query_plate_velocity_dist - query_plate_velocity_dist_span,\
                          trench_center - query_plate_velocity_dist]
    elif geometry == "chunk":
        query_v2_range = [trench_center - (query_plate_velocity_dist + query_plate_velocity_dist_span)/Max0,\
                          trench_center - query_plate_velocity_dist/Max0]

    query_pt = np.array([query_v0/Max0])
    idxs = rt_tree.query_ball_point(query_pt, r=dr)
    idx_np = np.array(idxs)
    v2s = v2_u[idxs]
    mask = ((v2s > query_v2_range[0]) & (v2s < query_v2_range[1]))
    v2s = v2s[mask]
    idxs = idx_np[mask]

    try:
        point_np = slab_points[idxs]
        velocity_np = slab_point_data["velocity"][idxs]

        if geometry == "chunk":
            _, _, velocity_2s = rotate_vec_cart_to_sph(point_np[:, 0], point_np[:, 1], point_np[:, 2],\
                                                    velocity_np[:, 0], velocity_np[:, 1], velocity_np[:, 2])
        elif geometry == "box":
            velocity_2s = velocity_np[:, 0]
        sp_velocity = np.mean(velocity_2s)
    except IndexError:
        sp_velocity = np.nan
        
    output_dict["sp_velocity"] = sp_velocity

    end = time.time()
    print("%s: Extract plate movement takes %.1f s" % (func_name(), end - start))

    # Process the dynamic pressure
    # similar to extract_slice_ref_da_profile function in the 3-d case
    query_dist = kwargs.get("query_dist", 500e3)
    query_dist_span = kwargs.get("query_dist_span", 200e3)
    
    # construct the kd tree
    v0, v2, _ = PUnified.points2unified3(points, (geometry == "chunk"), False)
    da_func = KNNInterpolatorND(np.vstack((v0/Max0, v2/Max2)).T, grid.point_data["nonadiabatic_pressure"], k=1, max_distance=0.05)

    # construct the query points
    query_v0s = np.linspace(Min0, Max0, 1001)

    if geometry == "chunk":
        query_v2_range = [trench_center_50km - (query_dist + query_dist_span)/Max0,\
                        trench_center_50km - query_dist/Max0]
    else:
        query_v2_range = [trench_center_50km - query_dist - query_dist_span,\
                        trench_center_50km - query_dist]
        
    query_v2s = np.linspace(query_v2_range[0], query_v2_range[1], 11)

    # loop and interpolate along v0
    # take the average dynamic pressure value
    query_das = np.zeros(query_v0s.size)
    for i, query_v0 in enumerate(query_v0s):
        da_array = da_func(np.full(query_v2s.shape, query_v0)/Max0, query_v2s/Max2)
        query_das[i] = np.mean(da_array)

    # export to file
    filename = "da_profile_%05d.txt" % (pvtu_step)
    filepath = os.path.join(pyvista_outdir, filename)
    np.savetxt(filepath, np.column_stack((query_v0s, query_das)), fmt="%.6f", delimiter="\t")
    print("saved file: %s" % filepath)
        

    
    # Process the region of slab metastablility, compute the area and output a vtu file
    # In addition, separately derive the area below a certain slab temperature (e.g 900 C)
    if Case_Options.options["MODEL_TYPE"] == "mow":
        start = time.time()
        # filter the grid base on cell data of "metastable"
        grid_metastable = grid_c.threshold(0.5, scalars="metastable", invert=True)
        cell_data_P_eq =  (grid_metastable.cell_data['T'] - Case_Options.options["T_PT_EQ"]) * Case_Options.options["CL_PT_EQ"] + Case_Options.options["P_PT_EQ"]
        mask = np.flatnonzero(grid_metastable.cell_data['p'] > cell_data_P_eq)
        grid_metastable = grid_metastable.extract_cells(mask)

        # if no cell presents, assign trivial values
        # else compute the total cell area
        if grid_metastable.n_cells == 0:
            metastable_area = 0.0
            metastable_area_cold = 0.0
        else:
            sized = grid_metastable.compute_cell_sizes(length=False, area=True, volume=False)
            metastable_area = float(sized.cell_data["Area"].sum())
            filename = "metastable_region_%05d.vtu" % pvtu_step
            filepath = os.path.join(pyvista_outdir, filename)
            grid_metastable.save(filepath)
            print("Save file %s" % filepath)

            # now derive the cold area
            mask = np.flatnonzero(grid_metastable.cell_data['T'] < T_slab)
            grid_metastable_slab = grid_metastable.extract_cells(mask)
            if grid_metastable_slab.n_cells == 0:
                metastable_area_cold = 0.0
            else:
                sized_cold = grid_metastable_slab.compute_cell_sizes(length=False, area=True, volume=False)
                metastable_area_cold = float(sized_cold.cell_data["Area"].sum())
                filename = "metastable_region_slab_%05d.vtu" % pvtu_step
                filepath = os.path.join(pyvista_outdir, filename)
                grid_metastable_slab.save(filepath)
                print("Save file %s" % filepath)


        output_dict["metastable_area"] = metastable_area
        output_dict["metastable_area_cold"] = metastable_area_cold
    
        end = time.time()
        print("%s: Compute metastable area takes %.1f s" % (func_name(), end - start))

    return output_dict


def PlotSlabMorphology(local_dir, local_dir_2d, **kwargs):

    import matplotlib.pyplot as plt
    from matplotlib import gridspec
    from matplotlib.ticker import MultipleLocator 
    from matplotlib import rcdefaults

    import hamageolib.utils.plot_helper as plot_helper

    time_marker = kwargs.get("time_marker", None)
    factor_3d = kwargs.get("factor_3d", 10)
    factor_2d = kwargs.get("factor_2d", 5)
    odir = kwargs.get("odir", os.path.join(local_dir, "img"))
    
    if not os.path.isdir(odir):
        os.mkdir(odir)
    
    # Retrieve the default color cycle
    default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    # Example usage
    # Rule of thumbs:
    # 1. Set the limit to something like 5.0, 10.0 or 50.0, 100.0 
    # 2. Set five major ticks for each axis
    scaling_factor = 1.0  # scale factor of plot
    font_scaling_multiplier = 1.5 # extra scaling multiplier for font
    legend_font_scaling_multiplier = 0.5
    line_width_scaling_multiplier = 2.0 # extra scaling multiplier for lines
    t_lim = (0.0, 20.0)
    t_tick_interval = 5.0   # tick interval along x
    y_lim = (-5.0, 5.0)
    y_tick_interval = 100.0  # tick interval along y
    v_lim = (-1.5, 1.5)
    v_level = 50  # number of levels in contourf plot
    v_tick_interval = 0.5  # tick interval along v
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

    # Initiate figure
    fig = plt.figure(figsize=(10*scaling_factor, 3.5*scaling_factor), tight_layout=True)
    gs = gridspec.GridSpec(1, 2)

    # Initiate case options
    Case_Options = CASE_OPTIONS(local_dir)

    Case_Options.Interpret()
    geometry = Case_Options.options["GEOMETRY"]
    Ro = Case_Options.options["OUTER_RADIUS"]

    Case_Options.SummaryCaseVtuStep(os.path.join(local_dir, "summary.csv"))

    time_3d = Case_Options.summary_df["Time"].to_numpy()
    trench_center_3d = Case_Options.summary_df["Trench (center)"].to_numpy()
    slab_depth_3d = Case_Options.summary_df["Slab depth"].to_numpy()
    dip_angle_center_3d = Case_Options.summary_df["Dip 100 (center)"].to_numpy()
    if geometry == "chunk":
        trench_center_3d *= Ro

    Case_Options_2d = CASE_OPTIONS_TWOD1(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    time_2d = Case_Options_2d.summary_df["Time"].to_numpy()
    trench_center_2d = Case_Options_2d.summary_df["Trench"].to_numpy()
    slab_depth_2d = Case_Options_2d.summary_df["Slab depth"].to_numpy()
    dip_angle_2d = Case_Options_2d.summary_df["Dip 100"].to_numpy()
    if geometry == "chunk":
        trench_center_2d *= Ro

    # plot
    ax = fig.add_subplot(gs[0, 0])
    ax_twin = ax.twinx()

    Xs_3d = time_3d/1e6
    Ys_3d = (trench_center_3d - trench_center_3d[0])/1e3
    Ys_3d_1 = slab_depth_3d/1e3
    dx_dy_3d = np.gradient(Ys_3d[::factor_3d], Xs_3d[::factor_3d]) / 1e3 * 1e2
    dx_dy_3d_1 = np.gradient(Ys_3d_1[::factor_3d], Xs_3d[::factor_3d]) / 1e3 * 1e2
    ax.plot(Xs_3d[::factor_3d],  Ys_3d[::factor_3d], label="Trench (center)", color=default_colors[0])
    ax_twin.plot(Xs_3d[::factor_3d],  Ys_3d_1[::factor_3d], linestyle="-.", label="Slab Depth", color=default_colors[0])

    Xs_2d = time_2d/1e6
    Ys_2d = (trench_center_2d - trench_center_2d[0])/1e3
    Ys_2d_1 = slab_depth_2d/1e3
    dx_dy_2d = np.gradient(Ys_2d[::factor_2d], Xs_2d[::factor_2d]) / 1e3 * 1e2
    dx_dy_2d_1 = np.gradient(Ys_2d_1[::factor_2d], Xs_2d[::factor_2d]) / 1e3 * 1e2
    ax.plot(Xs_2d[::factor_2d],  Ys_2d[::factor_2d], label="Trench 2d", color=default_colors[1])
    if time_marker is not None:
        ax.vlines(time_marker/1e6, linestyle="--", color="k", ymin=-150.0, ymax=100.0, linewidth=1)
    ax_twin.plot(Xs_2d[::factor_2d],  Ys_2d_1[::factor_2d], linestyle="-.", label="Slab Depth 2d", color=default_colors[1])

    ax.set_xlim(t_lim)
    ax.set_ylim([-150.0, 100.0])
    ax_twin.set_ylim([0, 1000.0])
    ax.set_xlabel("Time (Ma)")
    ax.set_ylabel("Trench (km)")

    # ax.legend()
    ax.grid()

    # Adjust spine thickness for this plot
    for spine in ax.spines.values():
        spine.set_linewidth(0.5 * scaling_factor * line_width_scaling_multiplier)

    ax.xaxis.set_major_locator(MultipleLocator(t_tick_interval))
    ax.xaxis.set_minor_locator(MultipleLocator(t_tick_interval/(n_minor_ticks+1)))
    ax.yaxis.set_major_locator(MultipleLocator(50.0))
    ax.yaxis.set_minor_locator(MultipleLocator(50.0/(n_minor_ticks+1)))
    ax_twin.yaxis.set_major_locator(MultipleLocator(200.0))
    ax_twin.yaxis.set_minor_locator(MultipleLocator(200.0/(n_minor_ticks+1)))

    ax1 = fig.add_subplot(gs[0, 1])

    ax1.plot(Xs_3d[::factor_3d], dx_dy_3d, label="Trench Velocity (center)", color=default_colors[0])
    ax1.plot(Xs_3d[::factor_3d], dx_dy_3d_1, linestyle="-.", label="Sinking Velocity (center)", color=default_colors[0])
    ax1.plot(Xs_2d[::factor_2d], dx_dy_2d, label="Trench Velocity 2d", color=default_colors[1])
    ax1.plot(Xs_2d[::factor_2d], dx_dy_2d_1, linestyle="-.", label="Sinking Velocity 2d", color=default_colors[1])
    if time_marker is not None:
        ax.vlines(time_marker/1e6, linestyle="--", color="k", ymin=-5.0, ymax=15.0, linewidth=1)

    ax1.set_xlim(t_lim)
    ax1.set_ylim([-5.0, 15.0])
    ax1.set_xlabel("Time (Ma)")
    ax1.set_ylabel("Velocity (cm/yr)")

    ax1.grid()
    # ax1.legend()

    ax1.xaxis.set_major_locator(MultipleLocator(t_tick_interval))
    ax1.xaxis.set_minor_locator(MultipleLocator(t_tick_interval/(n_minor_ticks+1)))
    ax1.yaxis.set_major_locator(MultipleLocator(5.0))
    ax1.yaxis.set_minor_locator(MultipleLocator(5.0/(n_minor_ticks+1)))
    
    ax1_twinx = ax1.twinx()
    ax1_twinx.plot(Xs_3d[::factor_3d], dip_angle_center_3d[::factor_3d]*180.0/np.pi, label="Dip 100 (center)", linestyle="--", color=default_colors[0])
    ax1_twinx.plot(Xs_2d[::factor_3d], dip_angle_2d[::factor_3d]*180.0/np.pi, label="Dip 100 2d", linestyle="--", color=default_colors[1])
    ax1_twinx.set_ylim([20.0, 60.0])
    ax1_twinx.yaxis.set_major_locator(MultipleLocator(10.0))
    ax1_twinx.yaxis.set_minor_locator(MultipleLocator(10.0/(n_minor_ticks+1))) 

    # save figure
    filepath = os.path.join(odir, "slab_morphology.pdf")
    fig.savefig(filepath)
    print("Saved figure: ", filepath)
    filepath_png = os.path.join(odir, "slab_morphology.png")
    fig.savefig(filepath_png)
    print("Saved figure: ", filepath_png)

    # Reset rcParams to defaults
    rcdefaults()


# todo_clip
def finalize_visualization_2d_07222025_box(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    # Options
    add_time = kwargs.get("add_time", True)
    canvas_size = kwargs.get("canvas_size", (996, 568))
    pos_v_diff = kwargs.get("pos_v_diff", 0)
    pos_v_diff_frame = kwargs.get("pos_v_diff_frame", 0)

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
        image_positions=[(-187, -50+pos_v_diff), (-39, -10+pos_v_diff_frame)],  # Positions of each image on the canvas
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


def fix_shallow_trench(upper_points, lower_points):
    '''
    fix the shallow trench points based on profiles of slab surface and moho, based on the function fix_shallow_trench_2d
    '''
    pass


def fix_shallow_trench_2d(upper_points, lower_points, thickness, is_spherical, Ro, **kwargs):
    '''
    fix the shallow trench points based on profiles of slab surface and moho
    Inputs:
        upper_points:
            points on the slab surface
        lower_points:
            points on the slab moho
        thickness:
            thickness of the initial crust
        is_spherical:
            if in spherical geometry
        Ro:
            vertial extent (outer radius or box height)
        kwargs:
            n - lookup stepping
            factor - factor * thickness is the distance to determine shallow trench points
    Returns:
        shallow_trench: trench position
    '''
    # free parameters
    # n - lookup stepping
    n = kwargs.get("n", 20)
    factor = kwargs.get("factor", 1.1)

        
    lower_l0 = lower_points[:, 1]
    lower_l2 = lower_points[:, 0]
    id = np.argmax(lower_points[:, 1])
    lower_end_l2 = lower_points[id, 0]
    mask0 = (Ro - lower_l2 < thickness)
    lower_l0_masked = lower_l0[mask0]
    lower_l2_masked = lower_l2[mask0]

    id = np.argmax(upper_points[:, 1])
    trench_center = upper_points[id, 0]


    # look for the shallow trench point
    found = -1
    shallow_trench = None
    for i in range(n, -1, -1):
        l2 = (i*trench_center + (n-i)* lower_end_l2)/n
        if is_spherical:
            dist_array = ((Ro - lower_l0_masked)**2.0 + (Ro*l2 - Ro*lower_l2_masked)**2.0)**0.5
        else:
            dist_array = ((Ro - lower_l0_masked)**2.0 + (l2-lower_l2_masked)**2.0)**0.5
        id_min = np.argmin(dist_array)
        dist = dist_array[id_min]
        print("i: ", i)
        print("id_min: ", id_min)
        print("dist: ", dist)
        if dist < factor * thickness:
            found = i
            shallow_trench = l2
            break

    my_assert(found >= 0, FixShallowTrenchError, "%s fail to find a shallow trench point." % func_name())

    return shallow_trench

class FixShallowTrenchError(Exception):
    pass

# Utilities
def find_file_by_stem(directory, name_without_ext):
    directory = Path(directory)
    for file in directory.iterdir():
        if file.is_file() and file.stem == name_without_ext:
            return str(file.resolve())  # full absolute path
    return None