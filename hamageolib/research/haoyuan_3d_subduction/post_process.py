import os
import math
import time
import pyvista as pv
import numpy as np
from hamageolib.utils.geometry_utilities import cartesian_to_spherical
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from scipy.spatial import cKDTree

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
        Ro (float): Planetary radius in meters.
        pyvista_outdir (str): Output directory for saving generated VTK files.
        max1 (float): Maximum colatitude in radians (default: π/2).
        max2 (float): Maximum longitude in radians (default: 140°).
        pvtu_step (int): Timestep index associated with current file.
        pvtu_filepath (str): Path to the .pvtu file being processed.
        grid (pv.UnstructuredGrid): Loaded PyVista grid from .pvtu file.
        iso_volume_upper (pv.UnstructuredGrid): Iso-volume of 'sp_upper' above threshold.
        iso_volume_lower (pv.UnstructuredGrid): Iso-volume of 'sp_lower' above threshold.
        iso_plate_edge (pv.UnstructuredGrid): Iso-volume of 'plate_edge' above threshold.
        slab_surface_points (np.ndarray): Extracted (x, y, z) coordinates of slab surface.
        pe_edge_points (np.ndarray): Extracted (x, y, z) coordinates of plate edge surface.
        trench_points (np.ndarray): Extracted (x, y, z) coordinates of trench.
        trench_center (np.ndarray): Extracted (x, y, z) point of the center at the model center.
    """

    def __init__(self, **kwargs):
        # Required settings and geometry assumption
        self.geometry = "chunk"
        self.Ro = 6371e3
        self.pyvista_outdir = kwargs.get("pyvista_outdir", ".")
        self.max1 = np.pi / 2.0
        self.max2 = 140.0 * np.pi / 180.0

        # Initialize runtime variables
        self.pvtu_step = None
        self.pvtu_filepath = None
        self.grid = None
        self.iso_volume_upper = None
        self.iso_volume_lower = None
        self.iso_plate_edge = None
        self.slab_surface_points = None
        self.pe_edge_points = None

        # Validate geometry
        if self.geometry != "chunk":
            raise NotImplementedError("Only 'chunk' geometry is supported.") 
        
    def read(self, pvtu_step, pvtu_filepath):
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
        self.pvtu_filepath = pvtu_filepath
        assert(os.path.isfile(self.pvtu_filepath))

        self.grid = pv.read(self.pvtu_filepath)

        points = self.grid.points
        if self.geometry == "chunk":
            radius = np.linalg.norm(points, axis=1)
        else:
            radius = points[:, 3]
        self.grid["radius"] = radius

        end = time.time()
        print("PYVISTA_PROCESS_THD: Read file takes %.1f s" % (end - start))

    def slice_center(self):
        """
        Extract a 2D slice at the center of the domain in the x-y plane.

        This method:
            - Uses a horizontal slicing plane (z-normal, through origin at radius Ro).
            - Projects the 'velocity' field onto the plane.
            - Saves the resulting slice and velocity projection as a .vtp file.
        """
        start = time.time()
        assert self.grid is not None

        origin = (self.Ro, 0.0, 0.0)
        normal = (0.0, 0.0, -1.0)

        sliced = self.grid.slice(normal=normal, origin=origin, generate_triangles=True)

        V = sliced["velocity"]
        dot_products = np.dot(V, normal)
        V_proj = V - np.outer(dot_products, normal)

        sliced["velocity_slice"] = V_proj

        filename = "slice_center_%05d.vtp" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        sliced.save(filepath)

        print("saved file %s" % filepath)
        end = time.time()
        print("PYVISTA_PROCESS_THD: slice_center takes %.1f s" % (end - start))

    def slice_surface(self):
        """
        Extract a thin shell at the planetary surface by thresholding radius.

        This method:
            - Applies a narrow threshold around the outer radius (Ro ± 5 km).
            - Extracts a thin surface shell of data near the planetary surface.
            - Saves the result as a .vtu file.
        """
        start = time.time()
        indent = 4

        r_slice = self.Ro
        r_diff = 5e3

        slice_shell = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

        filename = "slice_outer_%05d.vtu" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_shell.save(filepath)

        print("%ssaved file: %s" % (indent * " ", filepath))
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_surface takes %.1f s" % (indent * " ", end - start))

    def slice_at_depth(self, depth):
        """
        Extract a horizontal shell at a specified depth below the surface.

        Parameters:
            depth (float): Depth in meters from the planetary surface.

        This method:
            - Applies a radius threshold around Ro - depth with ±10 km tolerance.
            - Extracts a shell of data at the given depth.
            - Saves the result as a .vtu file.
        """
        start = time.time()
        indent = 4

        r_slice = self.Ro - depth
        r_diff = 10e3

        # Get slice shell
        slice_shell = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

        # Project velocity onto the surface
        # Get point coordinates and velocity vectors
        # Compute normalized radial direction vectors at each point
        # Project velocity onto tangent plane of sphere: v_tangent = v - (v · r̂) r̂
        # Store new vector field
        points = slice_shell.points                   # (N, 3)
        velocities = slice_shell.point_data["velocity"]  # (N, 3)

        if self.geometry == 'chunk':
            radial_dirs = points / np.linalg.norm(points, axis=1)[:, np.newaxis]  # (N, 3)
        else:
            radial_dirs = np.array([0, 0, 1.0])

        v_dot_r = np.sum(velocities * radial_dirs, axis=1)  # (N,)
        v_radial = (v_dot_r[:, np.newaxis]) * radial_dirs   # (N, 3)
        v_tangent = velocities - v_radial                   # (N, 3)

        slice_shell.point_data["velocity_slice"] = v_tangent

        # Export file
        filename = "slice_depth_%.1fkm_%05d.vtu" % (depth / 1e3, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_shell.save(filepath)
        print("%ssaved file: %s" % (indent * " ", filepath))

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_at_depth takes %.1f s" % (indent * " ", end - start))

    def extract_iso_volume_upper(self, threshold):
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

        self.iso_volume_upper = self.grid.threshold(value=threshold, scalars="sp_upper", invert=False)

        filename = "sp_upper_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_volume_upper.save(filepath)
        print("Save file %s" % filepath)

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def extract_iso_volume_lower(self, threshold):
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

        self.iso_volume_lower = self.grid.threshold(value=threshold, scalars="sp_lower", invert=False)

        filename = "sp_lower_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_volume_lower.save(filepath)
        print("Save file %s" % filepath)

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def extract_plate_edge(self, threshold):
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

        self.iso_plate_edge = self.grid.threshold(value=threshold, scalars="plate_edge", invert=False)

        filename = "plate_edge_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_plate_edge.save(filepath)
        print("Save file %s" % filepath)

        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent * " ", func_name(), end - start))

    def extract_slab_surface(self):
        """
        Extract the 3D surface of the subducting slab from the sp_upper iso-volume.

        This method:
            - Creates a (r, θ) mesh and queries the maximum φ values for each node using KDTree.
            - Reconstructs the 3D coordinates of the slab interface as a surface in spherical coordinates.
            - Stores the result in `self.slab_surface_points` and exports it as a `.vtp` file.
        """
        start = time.time()
        assert self.iso_volume_upper is not None
        indent = 4

        N0 = 1000

        N1 = 200
        if self.geometry == "chunk":
            min1 = 70.0 * np.pi / 180.0
        else:
            raise NotImplementedError()
            # min1 = 

        dr = 0.001

        # build the KDTREE
        vals0 = np.linspace(0, self.Ro, N0)
        vals1 = np.linspace(min1, self.max1, N1)
        vals2 = np.full((N0, N1), np.nan)
        vals2_tr = np.full(N1, np.nan)

        upper_points = self.iso_volume_upper.points
        v0_u, v1_u, v2_u = cartesian_to_spherical(*upper_points.T)
        rt_upper = np.vstack([v0_u/self.Ro, v1_u/self.max1]).T
        rt_tree = cKDTree(rt_upper)

        # extract slab surface points
        for i, v0 in enumerate(vals0):
            for j, v1 in enumerate(vals1):
                query_pt = np.array([v0/self.Ro, v1/self.max1])
                idxs = rt_tree.query_ball_point(query_pt, r=dr)

                if not idxs:
                    continue

                v2s = v2_u[idxs]
                max_v2 = np.max(v2s)

                if np.all(v2s <= max_v2):
                    vals2[i, j] = max_v2
                    if i == N0 - 1:
                        vals2_tr[j] = max_v2

        V0, V1 = np.meshgrid(vals0, vals1, indexing='ij')
        mask = ~np.isnan(vals2)

        v0_surf = V0[mask]
        v1_surf = V1[mask]
        v2_surf = vals2[mask]

        if self.geometry == "chunk":
            x = v0_surf * np.sin(v1_surf) * np.cos(v2_surf)
            y = v0_surf * np.sin(v1_surf) * np.sin(v2_surf)
            z = v0_surf * np.cos(v1_surf)
        else:
            x = v2_surf
            y = v1_surf
            z = v0_surf

        self.slab_surface_points = np.vstack([x, y, z]).T

        # save slab surface points
        point_cloud = pv.PolyData(self.slab_surface_points)
        filename = "sp_upper_surface_%05d.vtp" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud.save(filepath)

        print("Save file %s" % filepath)

        # extract trench points
        mask_tr = ~np.isnan(vals2_tr)
        v1_tr = vals1[mask_tr]
        v2_tr = vals2_tr[mask_tr]

        if self.geometry == "chunk": 
            x_tr = self.Ro * np.sin(v1_tr) * np.cos(v2_tr)
            y_tr = self.Ro * np.sin(v1_tr) * np.sin(v2_tr)
            z_tr = self.Ro * np.cos(v1_tr)
        else:
            x_tr = v2_tr
            y_tr = v1_tr
            z_tr = self.Ro
        
        self.trench_points = np.vstack([x_tr, y_tr, z_tr]).T
        _, _, self.trench_center = cartesian_to_spherical(self.trench_points[-1, 0], self.trench_points[-1, 1], self.trench_points[-1, 2])

        # save trench points
        point_cloud_tr = pv.PolyData(self.trench_points)
        filename = "trench_%05d.vtp" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_tr.save(filepath)

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
        # N2_1 - number along theta
        # Note these two are chosen so that resolutions along the radius and phi
        # have roughly the same value.
        # R_min_1 - starting value of extraction along radius. This should be
        # deeper than the lower-theta boundary of the subducting slab.
        # dr - normalized tolerance. This correlates to a dimensional value of
        # self.Ro * dr
        min0 =  self.Ro - 200e3; N0 = 40
        N2_1 = 3000
        dr = 0.001
        
        # Build KDTree with (r, theta)
        plate_edge_points = self.iso_plate_edge.points
        r_pe, theta_pe, phi_pe = cartesian_to_spherical(*plate_edge_points.T)
        rt_pe = np.vstack([r_pe/self.Ro, phi_pe/self.max2]).T
        rt_tree_pe = cKDTree(rt_pe)

        # Create a mesh on r, phi
        r_vals_1 = np.linspace(min0, self.Ro, N0)
        phi_vals_1 = np.linspace(0.0, self.max2, N2_1)
        theta_field_pe = np.full((N0, N2_1), np.nan)

        # Loop over each (r, phi), get theta
        # 1. get indexes of adjacent points in the plate_edge composition
        # 2. get minimum theta values within the adjacent points
        for i, r in enumerate(r_vals_1):
            for j, phi in enumerate(phi_vals_1):
                query_pt = np.array([r/self.Ro, phi/self.max2])
                idxs = rt_tree_pe.query_ball_point(query_pt, r=dr)  # tolerance

                if not idxs:
                    continue

                # get all φ values at this (r, θ)
                theta_s = theta_pe[idxs]
                min_theta = np.min(theta_s)

                # Check: is max_theta unique (i.e., no greater theta exists)?
                if np.all(theta_s >= min_theta):
                    theta_field_pe[i, j] = min_theta

        # Recover the 3D coordinates from (r, theta, phi_field)
        R_pe, Phi_pe = np.meshgrid(r_vals_1, phi_vals_1, indexing='ij')

        mask = ~np.isnan(theta_field_pe)

        r_pe_surf = R_pe[mask]
        phi_pe_surf = Phi_pe[mask]
        theta_pe_surf = theta_field_pe[mask]

        x_pe_surf = r_pe_surf * np.sin(theta_pe_surf) * np.cos(phi_pe_surf)
        y_pe_surf = r_pe_surf * np.sin(theta_pe_surf) * np.sin(phi_pe_surf)
        z_pe_surf = r_pe_surf * np.cos(theta_pe_surf)

        self.pe_edge_points = np.vstack([x_pe_surf, y_pe_surf, z_pe_surf]).T

        # export by pyvista
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
        assert(self.slab_surface_points is not None and self.pe_edge_points is not None)
        start = time.time()
        indent = 4

        # filter parameters
        d_upper_bound = 0.01

        # mesh the slab surface points by kd tree
        r_surf, theta_surf, _ = cartesian_to_spherical(*self.slab_surface_points.T)
        normalized_rt = np.vstack([r_surf/self.Ro, theta_surf/self.max1]).T  # shape (N, 2)
        rt_tree_slab_surface = cKDTree(normalized_rt)
        
        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, theta) within slab surface points
        lower_points = self.iso_volume_lower.points
        r_l, theta_l, phi_l = cartesian_to_spherical(*lower_points.T)
        query_points = np.vstack([r_l/self.Ro, theta_l/self.max1]).T

        distances, indices = rt_tree_slab_surface.query(query_points, k=1, distance_upper_bound=d_upper_bound)
        valid_mask = (indices != rt_tree_slab_surface.n)

        indices_valid = indices[valid_mask]
        lower_points_valid = lower_points[valid_mask]
        phi_l_valid = phi_l[valid_mask]

        # get mask for points that has higher phi value
        # than their matching points in the slab surface points
        _, _, phi_surf = cartesian_to_spherical(*self.slab_surface_points.T)
        large_phi_mask = phi_l_valid > phi_surf[indices_valid]

        large_phi_points = lower_points_valid[large_phi_mask]

        # export by pyvista
        point_cloud_large_phi = pv.PolyData(large_phi_points)

        filename = "sp_lower_large_phi_points_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_large_phi.save(filepath)
        print("Save file %s" % filepath)
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))


        # Derive the indices of large phi points within the original iso_volume_lower
        is_large_phi_point = np.zeros(self.iso_volume_lower.n_points, dtype=bool)

        idx1 = np.flatnonzero(valid_mask)
        large_phi_point_indices = idx1[large_phi_mask]
        is_large_phi_point[large_phi_point_indices] = True

        # Initiate cell_mask with all True value (every cell is included)
        cell_mask = np.ones(self.iso_volume_lower.n_cells, dtype=bool)

        # Use PyVista's connectivity and cell arrays if available
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            # pt_ids = self.iso_volume_lower.Get_Cell(cid).GetPointIds()
            if np.any(is_large_phi_point[pt_ids]):
                cell_mask[cid] = False

        filtered = self.iso_volume_lower.extract_cells(cell_mask)

        # Save to file for ParaView or future use
        # filename = "sp_lower_above_0.8_filtered_%05d.vtu" % (pvtu_step)
        # filepath = os.path.join(pyvista_outdir, filename)
        # filtered.save(filepath)

        # print("Save file %s" % filepath)

        # mesh a slab edge point by kd tree
        r_pe_surf, theta_pe_surf, phi_pe_surf = cartesian_to_spherical(*self.pe_edge_points.T)
        normalized_pe_rt = np.vstack([r_pe_surf/self.Ro, phi_pe_surf/self.max2]).T  # shape (N, 2)
        rt_tree_pe_surface = cKDTree(normalized_pe_rt)

        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, phi) within slab edge points
        lower_points = self.iso_volume_lower.points
        r_l, theta_l, phi_l = cartesian_to_spherical(*lower_points.T)
        query_points_1 = np.vstack([r_l/self.Ro, phi_l/self.max2]).T

        distances, indices = rt_tree_pe_surface.query(query_points_1, k=1, distance_upper_bound=d_upper_bound)
        valid_pe_mask = (indices != rt_tree_pe_surface.n)

        indices_pe_valid = indices[valid_pe_mask]
        lower_points_pe_valid = lower_points[valid_pe_mask]
        theta_l_pe_valid = theta_l[valid_pe_mask]

        # get mask for points that has smaller theta value
        # than their matching points in the plate edge surface points
        small_theta_mask = theta_l_pe_valid < theta_pe_surf[indices_pe_valid]

        small_theta_points = lower_points_pe_valid[small_theta_mask]

        # export by pyvista
        point_cloud_small_theta = pv.PolyData(small_theta_points)

        filename = "sp_lower_small_theta_points_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_small_theta.save(filepath)
        print("Save file %s" % filepath)


        # filter out cells that has small theta values
        # Derive the indices of large theta points within the original iso_volume_lower
        is_small_theta_point = np.zeros(self.iso_volume_lower.n_points, dtype=bool)

        idx1 = np.flatnonzero(valid_pe_mask)
        small_theta_point_indices = idx1[small_theta_mask]
        is_small_theta_point[small_theta_point_indices] = True

        # Use PyVista's connectivity and cell arrays if available
        # Note the cell_mask continues from the previous step
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            if np.any(is_small_theta_point[pt_ids]):
                cell_mask[cid] = False

        # Extract the final filtered points
        filtered_pe = self.iso_volume_lower.extract_cells(cell_mask)

        # Save to file for ParaView or future use
        filename = "sp_lower_above_0.8_filtered_pe_%05d.vtu" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        filtered_pe.save(filepath)

        print("Save file %s" % filepath)

    def make_boundary_cartesian(self, length_x, length_y, length_z, **kwargs):
        """
        Generate and save the six boundary surfaces for a cartesian box model domain.

        This method:
            - Constructs structured surface:
            - Combines them into a single surface and exports to a `.vtu` file.
        """
        marker_coordinates = kwargs.get("marker_coordinates", None)
        
        # Box parameters
        x_min = 0.0; x_max = length_x
        y_min = 0.0; y_max = length_y
        z_min = 0.0; z_max = length_z

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

        # y walls (y = min, max)
        y_edges = [y_min, y_max]
        for y_edge in y_edges:
            x_vals = np.linspace(x_min, x_max, n_x)
            z_vals = np.linspace(z_min, z_max, n_z)
            x_grid, z_grid = np.meshgrid(x_vals, z_vals)
            y_edge_grid = np.full(x_grid.shape, y_edge)
            surfaces.append(pv.StructuredGrid(x_grid, y_edge_grid, z_grid))

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
        filepath = os.path.join(self.pyvista_outdir, filename)
        full_surface.save(filepath)
        print("saved file: %s" % filepath)

        # deal with the annotation points
        if marker_coordinates is not None: 
            filename = "model_boundary_marker_points.vtp"
            filepath = os.path.join(self.pyvista_outdir, filename)
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

        # Chunk parameters
        r_inner = 3.481e6
        r_outer = self.Ro
        lon_min = 0.0
        lon_max = 80.00365006253027 * np.pi / 180.0
        lat_min = 0.0
        lat_max = 71.94572847349845 * np.pi / 180.0

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
        filepath = os.path.join(self.pyvista_outdir, filename)
        full_surface.save(filepath)
        print("saved file: %s" % filepath)

        # deal with the annotation points
        if marker_coordinates is not None: 
            filename = "model_boundary_marker_points.vtp"
            filepath = os.path.join(self.pyvista_outdir, filename)
            full_marker_point.save(filepath)
            print("saved file: %s" % filepath)

