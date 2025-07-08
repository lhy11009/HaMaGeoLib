import os
import math
import time
import pyvista as pv
import numpy as np
from hamageolib.utils.geometry_utilities import cartesian_to_spherical
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from hamageolib.utils.exception_handler import my_assert
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
        pyvista_outdir (str): Output directory for saving generated VTK files.
        Max0 (float): Outer radius / box thickness in meters.
        Min0 (float): Inner radius in meters.
        Max1 (float): Maximum colatitude in radians (default: π/2).
        Min1 (float): Minimum colatitude in radians (default: 0.0).
        Max2 (float): Maximum longitude in radians (default: π/2).
        Min2 (float): Minimum longitude in radians (default: 0.0).
        phi_max (float): Maximum longitude in radians (default: 140°).
        pvtu_step (int): Timestep index associated with current file.
        pvtu_filepath (str): Path to the .pvtu file being processed.
        grid (pv.UnstructuredGrid): Loaded PyVista grid from .pvtu file.
        iso_volume_upper (pv.UnstructuredGrid): Iso-volume of 'sp_upper' above threshold.
        iso_volume_lower (pv.UnstructuredGrid): Iso-volume of 'sp_lower' above threshold.
        iso_plate_edge (pv.UnstructuredGrid): Iso-volume of 'plate_edge' above threshold.
        slab_surface_points (np.ndarray): Extracted (x, y, z) coordinates of slab surface.
        pe_edge_points (np.ndarray): Extracted (x, y, z) coordinates of plate edge surface.
    """

    # todo_3d_visual
    def __init__(self, config, **kwargs):
        # Required settings and geometry assumption
        self.geometry = config["geometry"]
        self.Max0 = config["Max0"]
        self.Min0 = config["Min0"]
        self.Max1 = config["Max1"]
        # self.Min1 = 0.0
        self.Max2 = config["Max2"]
        self.Min2 = 0.0
        self.pyvista_outdir = kwargs.get("pyvista_outdir", ".")

        # Initialize runtime variables
        self.pvtu_step = None
        self.pvtu_filepath = None
        self.grid = None
        self.iso_volume_upper = None
        self.iso_volume_lower = None
        self.iso_plate_edge = None
        self.slab_surface_points = None
        self.pe_edge_points = None

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
        my_assert(os.path.isfile(self.pvtu_filepath), FileNotFoundError, "File %s is not found" % self.pvtu_filepath)

        self.grid = pv.read(self.pvtu_filepath)
        
        points = self.grid.points
        if self.geometry == "chunk":
            radius = np.linalg.norm(points, axis=1)
        else:
            radius = points[:, 2]
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

        origin = (self.Max0, 0.0, 0.0)

        if self.geometry == "chunk":
            normal = (0.0, 0.0, -1.0)
        else:
            normal = (0.0, -1.0, 0.0)

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
            - Saves the result as a .vtk file.
        """
        start = time.time()
        indent = 4

        r_slice = self.Max0
        r_diff = 5e3

        slice_shell = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

        filename = "slice_outer_%05d.vtu" % self.pvtu_step
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_shell.save(filepath)

        print("%ssaved file: %s" % (indent * " ", filepath))
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_surface takes %.1f s" % (indent * " ", end - start))

    def slice_at_depth(self, depth, **kwargs):
        """
        Extract a horizontal shell at a specified depth below the surface.

        Parameters:
            depth (float): Depth in meters from the planetary surface.
            kwargs:
                r_diff: range of radius / z values to clip

        This method:
            - Applies a radius threshold around Ro - depth with ±10 km tolerance.
            - Extracts a shell of data at the given depth.
            - Saves the result as a .vtk file.
        """
        start = time.time()
        indent = 4

        r_slice = self.Max0 - depth
        r_diff = kwargs.get("r_diff", 10e3)

        if self.geometry == 'chunk':
            slice_at_depth = self.grid.threshold([r_slice - r_diff, r_slice + r_diff], scalars="radius")

            # Project velocity onto the surface
            # Get point coordinates and velocity vectors
            # Compute normalized radial direction vectors at each point
            # Project velocity onto tangent plane of sphere: v_tangent = v - (v · r̂) r̂
            # Store new vector field
            points = slice_at_depth.points                   # (N, 3)
            velocities = slice_at_depth.point_data["velocity"]  # (N, 3)

            radial_dirs = points / np.linalg.norm(points, axis=1)[:, np.newaxis]  # (N, 3)

            v_dot_r = np.sum(velocities * radial_dirs, axis=1)  # (N,)
            v_radial = (v_dot_r[:, np.newaxis]) * radial_dirs   # (N, 3)
            v_tangent = velocities - v_radial                   # (N, 3)

            slice_at_depth.point_data["velocity_slice"] = v_tangent
            filename = "slice_depth_%.1fkm_%05d.vtu" % (depth / 1e3, self.pvtu_step)
        else:
            origin = (0, 0, r_slice)
            normal = (0, 0, 1.0)

            slice_at_depth = self.grid.slice(normal=normal, origin=origin, generate_triangles=True)

            V = slice_at_depth["velocity"]
            dot_products = np.dot(V, normal)
            V_proj = V - np.outer(dot_products, normal)

            slice_at_depth["velocity_slice"] = V_proj
            filename = "slice_depth_%.1fkm_%05d.vtp" % (depth / 1e3, self.pvtu_step)

        # Export results
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_at_depth.save(filepath)
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

        N0 = 2000
        N1 = 1000
        if self.geometry == "chunk":
            min1 = 70.0 * np.pi / 180.0
        else:
            min1 = 0.0

        dr = 0.001

        # build the KDTREE
        vals0 = np.linspace(0, self.Max0, N0)
        vals1 = np.linspace(min1, self.Max1, N1)
        vals2 = np.full((N0, N1), np.nan)
        vals2_tr = np.full(N1, np.nan)

        upper_points = self.iso_volume_upper.points
        
        if self.geometry == "chunk":
            v0_u, v1_u, v2_u = cartesian_to_spherical(*upper_points.T)
            rt_upper = np.vstack([v0_u/self.Max0, v1_u/self.Max1]).T
        else:
            rt_upper = np.vstack([upper_points[:, 2]/self.Max0, upper_points[:, 1]/self.Max1]).T
            v2_u = upper_points[:, 0]
        rt_tree = cKDTree(rt_upper)

        # extract slab surface points
        for i, v0 in enumerate(vals0):
            for j, v1 in enumerate(vals1):
                query_pt = np.array([v0/self.Max0, v1/self.Max1])
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
            x_tr = self.Max0 * np.sin(v1_tr) * np.cos(v2_tr)
            y_tr = self.Max0 * np.sin(v1_tr) * np.sin(v2_tr)
            z_tr = self.Max0 * np.cos(v1_tr)
        else:
            x_tr = v2_tr
            y_tr = v1_tr
            z_tr = self.Max0 * np.ones(v1_tr.shape)
        
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
        # N2 - number along theta
        # Note these two are chosen so that resolutions along the radius and phi
        # have roughly the same value.
        # min0 - starting value of extraction along radius. This should be
        # deeper than the lower-theta boundary of the subducting slab.
        # dr - normalized tolerance. This correlates to a dimensional value of
        # self.Max0 * dr
        min0 =  self.Max0 - 200e3; N0 = 40
        N2 = 3000
        dr = 0.001
        
        # Build KDTree with (r, theta)
        plate_edge_points = self.iso_plate_edge.points
        if self.geometry == "chunk":
            l0_pe, theta_pe, l2_pe = cartesian_to_spherical(*plate_edge_points.T)
            l1_pe = np.pi/2.0 - theta_pe
        else:
            l0_pe, l1_pe, l2_pe = plate_edge_points[:, 2], plate_edge_points[:, 1], plate_edge_points[:, 0]
        rt_pe = np.vstack([l0_pe/self.Max0, l2_pe/self.Max2]).T
        rt_tree_pe = cKDTree(rt_pe)

        # Create a mesh on r, phi
        l0_vals = np.linspace(min0, self.Max0, N0)
        l2_vals = np.linspace(0.0, self.Max2, N2)
        l1_field_pe = np.full((N0, N2), np.nan)

        # Loop over each (r, phi), get theta
        # 1. get indexes of adjacent points in the plate_edge composition
        # 2. get minimum theta values within the adjacent points
        for i, r in enumerate(l0_vals):
            for j, phi in enumerate(l2_vals):
                query_pt = np.array([r/self.Max0, phi/self.Max2])
                idxs = rt_tree_pe.query_ball_point(query_pt, r=dr)  # tolerance

                if not idxs:
                    continue

                # get all φ values at this (r, θ)
                l1_s = l1_pe[idxs]
                max1 = np.max(l1_s)

                # Check: is max_theta unique (i.e., no greater theta exists)?
                if np.all(l1_s <= max1):
                    l1_field_pe[i, j] = max1

        # Recover the 3D coordinates from (r, theta, phi_field)
        L0_pe, L2_pe = np.meshgrid(l0_vals, l2_vals, indexing='ij')

        mask = ~np.isnan(l1_field_pe)

        l0_pe_surf = L0_pe[mask]
        l2_pe_surf = L2_pe[mask]
        l1_pe_surf = l1_field_pe[mask]

        if self.geometry == "chunk":
            x_pe_surf = l0_pe_surf * np.sin(np.pi/2.0 - l1_pe_surf) * np.cos(l2_pe_surf)
            y_pe_surf = l0_pe_surf * np.sin(np.pi/2.0 - l1_pe_surf) * np.sin(l2_pe_surf)
            z_pe_surf = l0_pe_surf * np.cos(np.pi/2.0 - l1_pe_surf)
        else:
            x_pe_surf = l2_pe_surf
            y_pe_surf = l1_pe_surf
            z_pe_surf = l0_pe_surf

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
        if self.geometry == "chunk":
            r_surf, theta_surf, _ = cartesian_to_spherical(*self.slab_surface_points.T)
            normalized_rt = np.vstack([r_surf/self.Max0, theta_surf/self.Max1]).T  # shape (N, 2)
        else:
            normalized_rt = np.vstack([self.slab_surface_points[:, 2]/self.Max0, self.slab_surface_points[:, 1]/self.Max1]).T  # shape (N, 2)
        
        rt_tree_slab_surface = cKDTree(normalized_rt)
        
        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, theta) within slab surface points
        lower_points = self.iso_volume_lower.points

        if self.geometry == "chunk":
            r_l, theta_l, l2_l = cartesian_to_spherical(*lower_points.T)
            query_points = np.vstack([r_l/self.Max0, theta_l/self.Max1]).T
        else:
            query_points = np.vstack([lower_points[:, 2]/self.Max0, lower_points[:, 1]/self.Max1]).T
            l2_l = lower_points[:, 0]

        distances, indices = rt_tree_slab_surface.query(query_points, k=1, distance_upper_bound=d_upper_bound)
        valid_mask = (indices != rt_tree_slab_surface.n)

        indices_valid = indices[valid_mask]
        lower_points_valid = lower_points[valid_mask]
        l2_valid = l2_l[valid_mask]

        # get mask for points that has higher phi value
        # than their matching points in the slab surface points
        if self.geometry == "chunk":
            _, _, l2_surf = cartesian_to_spherical(*self.slab_surface_points.T)
        else:
            l2_surf = self.slab_surface_points[:, 0]
        large_l2_mask = l2_valid > l2_surf[indices_valid]

        # also get points that has absolutely large phi value,
        # larger than the maximum in slab surface points
        large_l2_abs_mask = l2_l > np.max(l2_surf) # debug

        combined_mask = np.zeros(valid_mask.shape, dtype=bool)
        combined_mask[valid_mask] = large_l2_mask
        large_l2_points = lower_points[combined_mask | large_l2_abs_mask]

        # export by pyvista
        point_cloud_large_l2 = pv.PolyData(large_l2_points)

        filename = "sp_lower_large_l2_points_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud_large_l2.save(filepath)
        print("Save file %s" % filepath)
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))


        # Derive the indices of large phi points within the original iso_volume_lower
        is_large_l2_point = np.zeros(self.iso_volume_lower.n_points, dtype=bool)

        idx1 = np.flatnonzero(valid_mask)
        large_l2_point_indices = idx1[large_l2_mask]
        is_large_l2_point[large_l2_point_indices] = True
        is_large_l2_point[large_l2_abs_mask] = True

        # Initiate cell_mask with all True value (every cell is included)
        cell_mask = np.ones(self.iso_volume_lower.n_cells, dtype=bool)

        # Use PyVista's connectivity and cell arrays if available
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            # pt_ids = self.iso_volume_lower.Get_Cell(cid).GetPointIds()
            if np.any(is_large_l2_point[pt_ids]):
                cell_mask[cid] = False

        filtered = self.iso_volume_lower.extract_cells(cell_mask)

        # Save to file for ParaView or future use
        # filename = "sp_lower_above_0.8_filtered_%05d.vtu" % (pvtu_step)
        # filepath = os.path.join(pyvista_outdir, filename)
        # filtered.save(filepath)

        # print("Save file %s" % filepath)

        # mesh a slab edge point by kd tree
        if self.geometry == "chunk":
            r_pe_surf, theta_pe_surf, phi_pe_surf = cartesian_to_spherical(*self.pe_edge_points.T)
            l1_pe_surf = np.pi/2.0 - theta_pe_surf
            normalized_pe_rt = np.vstack([r_pe_surf/self.Max0, phi_pe_surf/self.Max2]).T  # shape (N, 2)
        else:
            l1_pe_surf = self.pe_edge_points[:, 1]
            normalized_pe_rt = np.vstack([self.pe_edge_points[:, 2]/self.Max0, self.pe_edge_points[:, 0]/self.Max2]).T  # shape (N, 2)
        
        rt_tree_pe_surface = cKDTree(normalized_pe_rt)

        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, phi) within slab edge points
        lower_points = self.iso_volume_lower.points
        if self.geometry == "chunk":
            r_l, theta_l, phi_l = cartesian_to_spherical(*lower_points.T)
            l1_l = np.pi / 2.0 - theta_l
            query_points_1 = np.vstack([r_l/self.Max0, phi_l/self.Max2]).T
        else:
            query_points_1 = np.vstack([lower_points[:, 2]/self.Max0, lower_points[:, 0]/self.Max2]).T
            l1_l = lower_points[:, 1]

        distances, indices = rt_tree_pe_surface.query(query_points_1, k=1, distance_upper_bound=d_upper_bound)
        valid_pe_mask = (indices != rt_tree_pe_surface.n)

        indices_pe_valid = indices[valid_pe_mask]
        lower_points_pe_valid = lower_points[valid_pe_mask]
        l1_l_pe_valid = l1_l[valid_pe_mask]

        # get mask for points that has smaller theta value
        # than their matching points in the plate edge surface points
        big_l1_mask = l1_l_pe_valid > l1_pe_surf[indices_pe_valid]

        big_l1_points = lower_points_pe_valid[big_l1_mask]

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
        big_l1_point_indices = idx1[big_l1_mask]
        is_big_l1_point[big_l1_point_indices] = True

        # Use PyVista's connectivity and cell arrays if available
        # Note the cell_mask continues from the previous step
        for cid in range(self.iso_volume_lower.n_cells):
            pt_ids = self.iso_volume_lower.get_cell(cid).point_ids
            if np.any(is_big_l1_point[pt_ids]):
                cell_mask[cid] = False

        # Extract the final filtered points
        filtered_pe = self.iso_volume_lower.extract_cells(cell_mask)

        # Save to file for ParaView or future use
        filename = "sp_lower_above_0.8_filtered_pe_%05d.vtu" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        filtered_pe.save(filepath)

        print("Save file %s" % filepath)

    def make_boundary_cartesian(self, **kwargs):
        """
        Generate and save the six boundary surfaces for a cartesian box model domain.

        This method:
            - Constructs structured surface:
            - Combines them into a single surface and exports to a `.vtu` file.
        """
        marker_coordinates = kwargs.get("marker_coordinates", None)
        
        # Box parameters
        x_min = 0.0; x_max = self.Max2
        y_min = 0.0; y_max = self.Max1
        z_min = 0.0; z_max = self.Max0

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

        # Chunk parameters
        r_inner = self.Min0
        r_outer = self.Max0
        lat_min = 0.0
        lat_max = 35.972864236749224 * np.pi / 180.0
        lon_min = self.Min2
        lon_max = self.Max2

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


    def create_cartesian_plane(self, depth, resolution=(50, 50)):
        """
        Create a rectangular plane at a constant z-depth.

        Parameters:
            depth: constant depth for the plane
            resolution: (nx, ny) number of subdivisions in x and y
            filename: output file name (.vtp)
        """
        # Create linspace
        x = np.linspace(0.0, self.Max2, resolution[0])
        y = np.linspace(0.0, self.Max1, resolution[1])
        x_grid, y_grid = np.meshgrid(x, y)

        # Constant z
        z_grid = np.full_like(x_grid, self.Max0 - depth)

        # Combine to points
        points = np.column_stack((x_grid.ravel(), y_grid.ravel(), z_grid.ravel()))

        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (resolution[0], resolution[1], 1)

        # Save and return
        filename="plane_%.1fkm.vtk" % (depth/1e3)
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        grid.save(filepath)
        print("saved file: %s" % filepath)


    def create_spherical_plane(self, depth, resolution=(50, 50)):
        """
        Create a spherical plane at a constant radius.

        Parameters:
            depth: offset from base radius (final radius = radius + depth)
            resolution: (n_theta, n_phi) grid resolution
        """
        # Compute actual surface radius
        surface_radius = self.Max0 - depth

        # Create theta and phi arrays (already in radians)
        theta = np.linspace(self.Min1, self.Max1, resolution[0])
        phi = np.linspace(self.Min2, self.Max2, resolution[1])
        theta_grid, phi_grid = np.meshgrid(theta, phi)

        # Spherical to Cartesian conversion
        x = surface_radius * np.sin(phi_grid) * np.cos(theta_grid)
        y = surface_radius * np.sin(phi_grid) * np.sin(theta_grid)
        z = surface_radius * np.cos(phi_grid)

        # Combine to points
        points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

        # Create structured grid
        grid = pv.StructuredGrid()
        grid.points = points
        grid.dimensions = (resolution[0], resolution[1], 1)
        
        # Save and return
        filename="plane_%.1fkm.vtk" % (depth/1e3)
        filepath = os.path.join(self.pyvista_outdir, "..", filename)
        grid.save(filepath)
        print("saved file: %s" % filepath)


def get_trench_position_from_file(pyvista_outdir, pvtu_step):
    '''
    Get the position of trench from a file generated previously
    '''
    filename = "trench_%05d.vtp" % pvtu_step
    filepath = os.path.join(pyvista_outdir, filename)
    point_cloud_tr = pv.read(filepath)
    points = point_cloud_tr.points
    _, _, trench_center = cartesian_to_spherical(points[-1, 0], points[-1, 1], points[-1, 2])
    return trench_center