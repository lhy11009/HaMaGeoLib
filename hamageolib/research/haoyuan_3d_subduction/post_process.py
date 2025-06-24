import os
import math
import time
import pyvista as pv
import numpy as np
from hamageolib.utils.geometry_utilities import cartesian_to_spherical
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from scipy.spatial import cKDTree

class PYVISTA_PROCESS_THD():
    
    def __init__(self, **kwargs):

        # These are required variables
        self.geometry = "chunk"
        self.Ro = 6371e3
        self.pyvista_outdir = kwargs.get("pyvista_outdir", ".")
        self.theta_max = np.pi/2.0
        self.phi_max = 140.0 * np.pi / 180.0

        # There are generated variables
        self.pvtu_step = None
        self.pvtu_filepath = None
        self.grid = None
        self.iso_volume_upper = None
        self.iso_volume_lower = None
        self.iso_plate_edge = None
        self.slab_surface_points = None
        self.pe_edge_points = None

        # Addtional check
        if self.geometry != "chunk":
            raise NotImplementedError("Only allow chunk geometry")
        pass
        

    def read(self, pvtu_step, pvtu_filepath):

        # initiation 
        start = time.time()
        self.pvtu_step = pvtu_step
        self.pvtu_filepath = pvtu_filepath
        assert(os.path.isfile(self.pvtu_filepath))

        # read file
        self.grid = pv.read(self.pvtu_filepath)

        # add radius as an additional field 
        points = self.grid.points
        radius = np.linalg.norm(points, axis=1)
        self.grid["radius"] = radius

        # wrap up
        end = time.time()
        print("PYVISTA_PROCESS_THD: Read file takes %.1f s" % (end-start))

    def slice_center(self):

        start = time.time()
        assert(self.grid is not None)

        # Define slice parameters
        origin = (self.Ro, 0.0, 0.0)
        normal = (0.0, 0.0, -1.0)

        # Perform slicing
        sliced = self.grid.slice(normal=normal, origin=origin, generate_triangles=True)

        # Extract vector field from the sliced grid
        # 1. Project each vector onto the plane: V_proj = V - (V ⋅ n) * n;
        # 2. Add the projected vector field to slice;
        V = sliced["velocity"]  # shape: (N, 3)

        dot_products = np.dot(V, normal)  # (N,)
        V_proj = V - np.outer(dot_products, normal)  # shape: (N, 3)

        sliced["velocity_slice"] = V_proj

        # Save the slice to a new VTK file
        filename = "slice_center_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        sliced.save(filepath)  # You can also use .vtp or .vtu if preferred

        print("saved file %s" % filepath)
        end = time.time()
        
        print("PYVISTA_PROCESS_THD: slice_center takes %.1f s" % (end-start))


    def slice_surface(self):

        # Initiation
        start = time.time()
        indent = 4

        # Define slice parameters
        r_slice = self.Ro
        r_diff = 5e3

        # Threshold close to r = r_slice (e.g. 6.371e6) with some tolerance r_diff (e.g., ±5 km)
        slice_shell = self.grid.threshold([r_slice-r_diff, r_slice+r_diff], scalars="radius")

        # Save as .vtu
        filename = "slice_outer_%05d.vtk" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_shell.save(filepath)
        print("%ssaved file: %s" % (indent*" ", filepath))

        # Wrap up
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_surface takes %.1f s" % (indent*" ", end-start))

    def slice_at_depth(self, depth):

        # Initiation
        start = time.time()
        indent = 4

        # Slice parameters
        r_slice = self.Ro - depth
        r_diff = 10e3

        # Threshold close to r = r_slice (e.g. 6.371e6) with some tolerance r_diff (e.g., ±5 km)
        slice_shell = self.grid.threshold([r_slice-r_diff, r_slice+r_diff], scalars="radius")

        # Save as .vtu
        filename = "slice_depth_%.1fkm_%05d.vtk" % (depth/1e3, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        slice_shell.save(filepath)
        print("%ssaved file: %s" % (indent*" ", filepath))

        # Wrap up 
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: slice_at_depth takes %.1f s" % (indent*" ", end-start))

    def extract_iso_volume_upper(self, threshold):

        # Initiation
        start = time.time()
        indent = 4

        # Extract the iso-volume of sp_upper
        self.iso_volume_upper = self.grid.threshold(value=threshold, scalars="sp_upper", invert=False)

        # Save to file for ParaView or future use
        filename = "sp_upper_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_volume_upper.save(filepath)
        print("Save file %s" % filepath)

        # Wrap up
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))

    def extract_iso_volume_lower(self, threshold):

        # Initiation 
        start = time.time()
        indent = 4

        # Extract the iso-volume of sp_lower
        self.iso_volume_lower = self.grid.threshold(value=threshold, scalars="sp_lower", invert=False)

        # Save to file for ParaView or future use
        filename = "sp_lower_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_volume_lower.save(filepath)
        print("Save file %s" % filepath)

        # Wrap up
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))


    def extract_plate_edge(self, threshold):

        # Initiation
        start = time.time()
        indent = 4

        # Extract iso-volume
        self.iso_plate_edge = self.grid.threshold(value=threshold, scalars="plate_edge", invert=False)

        # Save to file for ParaView or future use
        filename = "plate_edge_above_%.2f_%05d.vtu" % (threshold, self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        self.iso_plate_edge.save(filepath)
        print("Save file %s" % filepath)

        # Wrap up 
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))

    def extract_slab_surface(self):

        # Initiation
        start = time.time()
        assert(self.iso_volume_upper is not None)
        indent = 4

        # Extracting parameters
        # Nr - number along radius
        # Ntheta - number along theta
        # Note these two are chosen so that resolutions along the radius and theta
        # have roughly the same value.
        # theta_min - starting value of extraction along theta. This should roughly
        # match the lower-theta boundary of the subducting slab.
        # dr - normalized tolerance. This correlates to a dimensional value of
        # self.Ro * dr
        Nr = 1000
        theta_min = 70.0 * np.pi / 180.0;  Ntheta = 200
        dr = 0.001

        # Make a r-theta mesh
        # Initiate phi values as nan
        r_vals = np.linspace(0, self.Ro, Nr)
        theta_vals = np.linspace(theta_min, self.theta_max, Ntheta)
        phi_field = np.full((Nr, Ntheta), np.nan)

        # Build KDTree with (r, theta)
        upper_points = self.iso_volume_upper.points
        r_u, theta_u, phi_u = cartesian_to_spherical(*upper_points.T)
        rt_upper = np.vstack([r_u/self.Ro, theta_u/self.theta_max]).T
        rt_tree = cKDTree(rt_upper)

        # Loop over each (r, theta), get phi
        # 1. get indexes of adjacent points in the sp_upper composition
        # 2. get maximum phi values within the adjacent points
        for i, r in enumerate(r_vals):
            for j, theta in enumerate(theta_vals):
                query_pt = np.array([r/self.Ro, theta/self.theta_max])
                idxs = rt_tree.query_ball_point(query_pt, r=dr)  # tolerance

                if not idxs:
                    continue

                phis = phi_u[idxs]
                max_phi = np.max(phis)

                if np.all(phis <= max_phi):
                    phi_field[i, j] = max_phi

        # Recover the 3D coordinates from (r, theta, phi_field)
        R, Theta = np.meshgrid(r_vals, theta_vals, indexing='ij')

        mask = ~np.isnan(phi_field)

        r_surf = R[mask]
        theta_surf = Theta[mask]
        phi_surf = phi_field[mask]

        x = r_surf * np.sin(theta_surf) * np.cos(phi_surf)
        y = r_surf * np.sin(theta_surf) * np.sin(phi_surf)
        z = r_surf * np.cos(theta_surf)

        self.slab_surface_points = np.vstack([x, y, z]).T

        # export by pyvista
        point_cloud = pv.PolyData(self.slab_surface_points)

        filename = "sp_upper_surface_%05d.vtp" % (self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        point_cloud.save(filepath)

        print("Save file %s" % filepath) 
        
        end = time.time()
        print("%sPYVISTA_PROCESS_THD: %s takes %.1f s" % (indent*" ", func_name(), end-start))
        
    def extract_plate_edge_surface(self):

        # Extracting parameters
        # Nr_1 - number along radius
        # Nphi_1 - number along theta
        # Note these two are chosen so that resolutions along the radius and phi
        # have roughly the same value.
        # R_min_1 - starting value of extraction along radius. This should be
        # deeper than the lower-theta boundary of the subducting slab.
        # dr - normalized tolerance. This correlates to a dimensional value of
        # self.Ro * dr
        R_min_1 =  self.Ro - 200e3; Nr_1 = 40
        Nphi_1 = 3000
        dr = 0.001
        
        # Build KDTree with (r, theta)
        plate_edge_points = self.iso_plate_edge.points
        r_pe, theta_pe, phi_pe = cartesian_to_spherical(*plate_edge_points.T)
        rt_pe = np.vstack([r_pe/self.Ro, phi_pe/self.phi_max]).T
        rt_tree_pe = cKDTree(rt_pe)

        # Create a mesh on r, phi
        r_vals_1 = np.linspace(R_min_1, self.Ro, Nr_1)
        phi_vals_1 = np.linspace(0.0, self.phi_max, Nphi_1)
        theta_field_pe = np.full((Nr_1, Nphi_1), np.nan)

        # Loop over each (r, phi), get theta
        # 1. get indexes of adjacent points in the plate_edge composition
        # 2. get minimum theta values within the adjacent points
        for i, r in enumerate(r_vals_1):
            for j, phi in enumerate(phi_vals_1):
                query_pt = np.array([r/self.Ro, phi/self.phi_max])
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

        # initiation
        assert(self.slab_surface_points is not None and self.pe_edge_points is not None)
        start = time.time()
        indent = 4

        # filter parameters
        d_upper_bound = 0.01

        # mesh the slab surface points by kd tree
        r_surf, theta_surf, _ = cartesian_to_spherical(*self.slab_surface_points.T)
        normalized_rt = np.vstack([r_surf/self.Ro, theta_surf/self.theta_max]).T  # shape (N, 2)
        rt_tree_slab_surface = cKDTree(normalized_rt)
        
        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, theta) within slab surface points
        lower_points = self.iso_volume_lower.points
        r_l, theta_l, phi_l = cartesian_to_spherical(*lower_points.T)
        query_points = np.vstack([r_l/self.Ro, theta_l/self.theta_max]).T

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
        normalized_pe_rt = np.vstack([r_pe_surf/self.Ro, phi_pe_surf/self.phi_max]).T  # shape (N, 2)
        rt_tree_pe_surface = cKDTree(normalized_pe_rt)

        # query points in the sp_lower iso-volume and get points that has 
        # matching pair in (r, phi) within slab edge points
        lower_points = self.iso_volume_lower.points
        r_l, theta_l, phi_l = cartesian_to_spherical(*lower_points.T)
        query_points_1 = np.vstack([r_l/self.Ro, phi_l/self.phi_max]).T

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

    def make_boundary(self):

        # Chunk parameters
        r_inner = 3.481e6
        r_outer = self.Ro
        lon_min = 0.0
        lon_max = 80.00365006253027
        lat_min = 0.0
        lat_max = 71.94572847349845

        # Resolution (adjust for finer mesh)
        n_r = 2
        n_lat = 100
        n_lon = 100

        # Helper function: spherical to Cartesian
        def sph2cart(r, lat_deg, lon_deg):
            lat = np.radians(lat_deg)
            lon = np.radians(lon_deg)
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

        # Faces:
        surfaces = []

        # Outer surface (r = outer)
        surfaces.append(make_sphere_shell(r_outer, (lat_min, lat_max), (lon_min, lon_max), n_lat, n_lon))

        # Inner surface (r = inner)
        surfaces.append(make_sphere_shell(r_inner, (lat_min, lat_max), (lon_min, lon_max), n_lat, n_lon))

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

        # Combine all surfaces
        full_surface = surfaces[0]
        for s in surfaces[1:]:
            full_surface = full_surface.merge(s)

        # Save to file
        filename = "model_boundary.vtu"
        filepath = os.path.join(pyvista_outdir, filename)
        full_surface.save(filepath)
