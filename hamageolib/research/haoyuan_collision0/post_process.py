import os
import numpy as np
import time
import math
import pyvista as pv
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.interpolate import interp1d, NearestNDInterpolator
from scipy.spatial import cKDTree
from hamageolib.core.post_process import PYVISTA_PROCESS, PYVISTA_PROCESS_WORKFLOW_ERROR
from hamageolib.utils.exception_handler import my_assert
from hamageolib.utils.interp_utilities import KNNInterpolatorND
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD
from hamageolib.utils.pyvista_utilities import get_corner_point_ids


SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")


class PYVISTA_PROCESS_COLLISION(PYVISTA_PROCESS):

    def __init__(self, data_dir, options, *,
                 pyvista_outdir=None,
                 include_particles=False):

        PYVISTA_PROCESS.__init__(self, data_dir, 
                            pyvista_outdir=pyvista_outdir,
                            include_particles=include_particles)

        # model geometry 
        self.Min0 = options["BOTTOM"]
        self.Max0 = options["TOP"]
        self.Min2 = options["LEFT"]
        self.Max2 = options["RIGHT"]

        # plate setup
        self.plate_start_point = options["PLATE_START_POINT"]
        self.slab_hinge_point = options["SLAB_HINGE_POINT"]
        
        # composition names 
        self.composition_names = [
            'sediment', 'gabbro', 'MORB',
            'crust_upper', 'crust_lower'
        ]

        # lithospheric temperature
        self.lithospheric_T = 1300 + 273.15 # K

        # placeholder for class variables
        self.slab_surface_points = None
        self.iso_volume_dict = {}

        # placeholder for interpolation functions
        self.particle_ul_func = None

        # placeholder for topography functions
        self.topography_func = None

        # placeholder for suture position     
        self.suture_profile_depths = None
        self.suture_profile_x = None

        # placeholder for dimention ratio function
        self.dimention_ratio_func = None


    def read(self, pvtu_step):

        # read dataset
        self.pvtu_step = pvtu_step
        PYVISTA_PROCESS.read(self, pvtu_step)

        # total slab composition
        self.grid["sp_total"] = self.grid["gabbro"] + self.grid["MORB"] + self.grid["sediment"]

        # background composition 
        self.add_background()

    def load_topograph(self, time_step):

        start = time.time()

        topography_file = os.path.join(self.data_dir, "..", "topography", "topography.%05d" % time_step)
    
        my_assert(os.path.isfile(topography_file), FileExistsError, "File %s doesn't exist." % topography_file)

        # Extract data
        data = np.loadtxt(topography_file, comments="#")

        x = data[:, 0]
        topography = data[:, 2]

        self.topography_func = interp1d(
            x/self.Max0,
            topography,
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))


    def extract_slab(self, *,
                     threshold=0.5,
                     d0=5e3,
                     dr=0.001,
                     output_surfuce=False):

        start = time.time()
        print("PYVISTA_PROCESS:\n\t%s" % func_name())

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
            print("\tSave file %s" % filepath)
        
        end = time.time()
        print("\ttakes %.1f s" % (end - start))

    def analyze_slab(self):
        
        start = time.time()
        print("PYVISTA_PROCESS:\n\t%s" % func_name())

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
        
        end = time.time()
        print("\ttakes %.1f s" % (end - start))

        return outputs
    
    def analyze_velocity(self, trench_center, *, 
                         query_depth=50e3,
                         dist_range=[500e3, 1000e3]):

        start = time.time()
        print("PYVISTA_PROCESS:\n\t%s" % func_name())

        outputs = {}

        # sample points 
        N = 10

        # sp velcoity
        vx_sum = 0.0
        for dist in np.linspace(dist_range[0], dist_range[1], N):

            x = trench_center - dist
            idx = self.grid.find_closest_point(
                (x, self.Max0 - query_depth, 0.0)
            )
            velocity = self.grid["velocity"][idx]
            vx, vy = velocity[:2]
            vx_sum += vx

        outputs["sp_velocity"] = vx_sum / N

        # ov velocity
        vx_sum = 0.0
        for dist in np.linspace(dist_range[0], dist_range[1], N):

            x = trench_center + dist
            idx = self.grid.find_closest_point(
                (x, self.Max0 - query_depth, 0.0)
            )
            velocity = self.grid["velocity"][idx]
            vx, vy = velocity[:2]
            vx_sum += vx

        outputs["ov_velocity"] = vx_sum / N

        end = time.time()
        print("\ttakes %.1f s" % (end - start))
        
        return outputs
    
    def extract_topography(self, *, 
                           dx=5e3, dr=0.001,
                           interp_dx=None, output_surface=False):
        '''
        Extract topography (surface elevation) from the grid.
    
        Steps:
            1. Sample along x-direction using KDTree
            2. Map to real mesh points (Option 2)
            3. Remove duplicate (x, y) pairs
            4. Sort by x
            5. Interpolate to regular grid (optional)
    
        Inputs:
            dx - sampling spacing for KDTree queries
            dr - normalized search radius
            interp_dx - spacing for interpolated regular grid (None = skip interpolation)
            output_surface - whether to save point cloud
    
        Outputs (stored as attributes):
            self.topography_profile : interpolated (x, y) if interp_dx is not None
        '''
        start = time.time()
        print("PYVISTA_PROCESS:\n\t%s" % func_name())

        my_assert(self.topography_func is None, PYVISTA_PROCESS_WORKFLOW_ERROR, "%s should not be executed if a topography_func already exists" % func_name())
    
        if self.grid is None:
            raise RuntimeError("Grid not loaded. Run read() first.")
    
        # --- extract raw coordinates ---
        points = self.grid.points
        x_all = points[:, 0]
        y_all = points[:, 1]
    
        # --- sampling positions ---
        xs_raw = np.arange(self.Min2, self.Max2, dx)
        xs = np.full(xs_raw.size, np.nan)
        ys = np.full(xs_raw.size, np.nan)
    
        # --- build KDTree in x-direction ---
        x_norm = (x_all / self.Max0).reshape(-1, 1)
        tree = cKDTree(x_norm)
    
        # --- sample surface ---
        for i, x in enumerate(xs_raw):
            query_pt = np.array([x / self.Max0])
            idxs = tree.query_ball_point(query_pt, r=dr)
    
            if not idxs:
                continue
    
            local_idxs = np.array(idxs)
            imax = np.argmax(y_all[local_idxs])
    
            # assign real mesh point (Option 2)
            xs[i] = x_all[local_idxs][imax]
            ys[i] = y_all[local_idxs][imax]
    
        # --- remove NaNs ---
        mask = ~np.isnan(ys)
        x_surf = xs[mask]
        y_surf = ys[mask]
    
        # --- remove duplicate (x, y) pairs ---
        pts2d = np.column_stack((x_surf, y_surf))
        _, unique_idx = np.unique(pts2d, axis=0, return_index=True)
        pts2d = pts2d[unique_idx]
    
        # --- sort by x ---
        pts2d = pts2d[np.argsort(pts2d[:, 0])]
        x_surf = pts2d[:, 0]
        y_surf = pts2d[:, 1]

        # --- generate the topography function ---
        self.topography_func = interp1d(
            x_surf/self.Max0,
            (y_surf-self.Max0),
            kind="linear",
            bounds_error=False,
            fill_value=np.nan
        )
    
        # --- optional interpolation to regular grid ---
        x_reg = x_surf
        y_reg = y_surf
        z_reg = np.zeros_like(x_surf)
        topo_reg = y_surf - self.Max0
        if interp_dx is not None:
            x_reg = np.arange(x_surf[0], x_surf[-1], interp_dx)
            topo_reg = self.topography_func(x_reg/self.Max0)
            y_reg = topo_reg + self.Max0
            z_reg = np.zeros_like(x_reg)
    
        topography_points = np.vstack([x_reg, y_reg, z_reg]).T
    
        # --- outputs ---
        if output_surface:
            point_cloud = pv.PolyData(topography_points)
            filename = "topography_%05d.vtp" % self.pvtu_step
            filepath = os.path.join(self.pyvista_outdir, filename)
            point_cloud.save(filepath)
            print(f"\tSave file {filepath}")

        data_out = np.vstack([x_reg, y_reg, topo_reg]).T
        filename_txt = "topography_%05d.txt" % self.pvtu_step
        filepath_txt = os.path.join(self.pyvista_outdir, filename_txt)
        np.savetxt(
            filepath_txt,
            data_out,
            header="# x y topography",
            fmt="%.4e"
        )
        print(f"\tSave file {filepath_txt}")
        
        end = time.time()
        print("\ttakes %.1f s" % (end - start))
    

    def extract_continent_crust_iso_volumes(self, *, 
                            save_file=True,
                            threshold=0.8,
                            fields=["crust_upper", "crust_lower"]):
        """
        Extract the iso-volume of the composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for compositions.

        This method:
            - Filters the grid for regions where sp_lower >= threshold.
            - Stores the result in `self.iso_volume_lower`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4
        
        # additional options 
        # Iso volume of slab lower composition
        for field in fields:
            iso_volume = self.grid.threshold(value=threshold, scalars=field, invert=False)
        
            self.iso_volume_dict[field] = iso_volume
        
            if save_file:
                self.write_object_to_file(iso_volume, "%s_above_%.2f" % (field, threshold), "vtu")
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
    
        print("\ttakes %.1f s" % (end - start))

    def extract_oceanic_plate_iso_volumes(self, *, 
                            save_file=True,
                            threshold=0.8,
                            crust_fields=["gabbro", "MORB"],
                            include_sediment=True,
                            include_harzburgite=True):
        """
        Extract the iso-volume of the composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for compositions.

        This method:
            - Filters the grid for regions where sp_lower >= threshold.
            - Stores the result in `self.iso_volume_lower`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4
        
        # Iso volume of oceanic sediment composition
        if include_sediment:
            field = "sediment"

            iso_volume = self.grid.threshold(value=threshold, scalars=field, invert=False)
        
            self.iso_volume_dict[field] = iso_volume
        
            if save_file:
                self.write_object_to_file(iso_volume, "%s_above_%.2f" % (field, threshold), "vtu")
        
        # Iso volume of oceanic crust composition
        # combine crust fields into a new scalar field
        oceanic_crust = self.grid[crust_fields[0]].copy()

        for field in crust_fields[1:]:
            oceanic_crust += self.grid[field]

        self.grid["oceanic_crust"] = oceanic_crust

        iso_volume = self.grid.threshold(
            value=threshold,
            scalars="oceanic_crust",
            invert=False
        )
        
        self.iso_volume_dict["oceanic_crust"] = iso_volume
        
        if save_file:
            self.write_object_to_file(iso_volume, "oceanic_crust_above_%.2f" % threshold, "vtu")

        # Iso volume of oceanic harzburgite composition
        if include_harzburgite:
            field = "harzburgite"

            iso_volume = self.grid.threshold(value=threshold, scalars=field, invert=False)
        
            self.iso_volume_dict[field] = iso_volume
        
            if save_file:
                self.write_object_to_file(iso_volume, "%s_above_%.2f" % (field, threshold), "vtu")
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
    
        print("\ttakes %.1f s" % (end - start))

    def extract_continent_lithosphere_iso_volumes(self, fields, *, 
                            save_file=True,
                            threshold=0.2,
                            temperature_threshold=1300+273.15):
        """
        Extract the iso-volume of the composition field above a threshold.

        Parameters:
            threshold (float): Scalar threshold for compositions.

        This method:
            - Filters the grid for regions where sp_lower >= threshold.
            - Stores the result in `self.iso_volume_lower`.
            - Saves the extracted volume as a .vtu file.
        """
        start = time.time()
        indent = 4
        
        # additional options 
        # Iso volume of slab lower composition
        all_composition = self.grid[fields[0]].copy()

        for field in fields[1:]:
            all_composition += self.grid[field]

        self.grid["all_composition"] = all_composition

        iso_volume = self.grid.threshold(
            value=[0.0, threshold],
            scalars="all_composition",
            invert=False
        )

        iso_volume_c = iso_volume.point_data_to_cell_data(pass_point_data=False, categorical=False)
        mask = np.flatnonzero(iso_volume_c.cell_data['T'] < temperature_threshold)
        grid_lithosphere = iso_volume_c.extract_cells(mask)
        
        if save_file:
            self.write_object_to_file(grid_lithosphere, 
                                      "lithosphere_below_%.2f_colder_%.2f" % (threshold, temperature_threshold), "vtu")
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
    
        print("\ttakes %.1f s" % (end - start))

    class InitialParticlePositionException(Exception):
        pass

    def process_particles(self):
        """
        Process particle data and construct an interpolator for initial particle positions.
    
        This function:
        - Extracts particle coordinates (x, y)
        - Retrieves the initial X position stored on particles
        - Builds a KNN interpolator that maps spatial coordinates → initial X
        - Stores the interpolator for later use on the grid
    
        Attributes used:
            self.particles : PyVista particle dataset
            self.Max0 : normalization factor for coordinates
    
        Attributes created:
            self.particle_initial_X_func : callable interpolator
        """
    
        start = time.time()
    
        assert(self.include_particles)
    
        points = self.particles.points
        x_p = points[:, 0]
        y_p = points[:, 1]

        try: 
            initial_X = self.particles["initial position"][:, 0]
        except KeyError:
            raise self.InitialParticlePositionException()

        self.particle_initial_X_func = KNNInterpolatorND(
            np.vstack((x_p/self.Max0, y_p/self.Max0)).T,
            initial_X.ravel(),
            k=1,
            max_distance=0.01
        )
    
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

    def analyze_shortening_by_cell(self, *,
                           threshold=0.8):
        
        start = time.time()

        my_assert((self.topography_func is not None) and 
                  (self.particle_initial_X_func is not None) and 
                  (self.suture_profile_x is not None), 
                  PYVISTA_PROCESS_WORKFLOW_ERROR,
                  "%s requires the topography function, the particle results and the suture profile" % func_name())

        # Get the crust field 
        crust_upper_comps = self.grid['crust_upper']
        crust_lower_comps = self.grid['crust_lower']
        initial_Xs = self.grid['initial_X']
        
        # Initialize cell data field
        self.grid.cell_data["dimention_ratio"] = np.full(
            self.grid.n_cells,
            np.nan,
            dtype=float
        )

        cells = self.grid.cells.reshape((-1, 5))[:, 1:]
        points = self.grid.points
        
        dimention_ratio_array = np.full(self.grid.n_cells, np.nan)
        
        for cell_id, point_ids in enumerate(cells):
        
            cell_points = points[point_ids]
        
            sorted_by_y = np.argsort(cell_points[:, 1])
        
            bottom_local = sorted_by_y[:2]
            top_local = sorted_by_y[-2:]
        
            top_points = cell_points[top_local]
            top_ids = point_ids[top_local]
        
            top_left_id = top_ids[np.argmin(top_points[:, 0])]
            top_right_id = top_ids[np.argmax(top_points[:, 0])]
        
            top_right_initial_X = initial_Xs[top_right_id]
            top_left_initial_X = initial_Xs[top_left_id]
        
            top_right_crust = (
                crust_upper_comps[top_right_id]
                + crust_lower_comps[top_right_id]
            )
            top_left_crust = (
                crust_upper_comps[top_left_id]
                + crust_lower_comps[top_left_id]
            )
        
            denominator = top_right_initial_X - top_left_initial_X
        
            if (
                np.isfinite(top_left_initial_X)
                and np.isfinite(top_right_initial_X)
                and abs(denominator) > 1e-16
                and top_right_crust > threshold
                and top_left_crust > threshold
            ):
                dimention_ratio_array[cell_id] = (
                    top_points[np.argmax(top_points[:, 0]), 0]
                    - top_points[np.argmin(top_points[:, 0]), 0]
                ) / denominator
        
        self.grid.cell_data["dimention_ratio"] = dimention_ratio_array

        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

    def analyze_shortening_by_bin(self, *,
                           threshold=0.8,
                           bin_size=10e3,
                           profile_half_size=500e3,
                           profile_bin_size=5e3,
                           profile_bury_depth=1e3):
        
        start = time.time()

        my_assert((self.topography_func is not None) and 
                  (self.particle_initial_X_func is not None) and 
                  (self.suture_profile_x is not None), 
                  PYVISTA_PROCESS_WORKFLOW_ERROR,
                  "%s requires the topography function, the particle results and the suture profile" % func_name())

        # dict for storing outputs 
        outputs = {}

        # Get the crust field 
        crust_upper_comps = self.grid['crust_upper']
        crust_lower_comps = self.grid['crust_lower']
        initial_Xs = self.grid['initial_X']
        
        # Initialize cell data field
        self.grid["dimention_ratio_bin"] = np.full(
            self.grid.n_points,
            np.nan,
            dtype=float
        )
        
        points = self.grid.points

        # mask points with continent compositions
        mask_cr = ((crust_upper_comps + crust_lower_comps) > threshold)
        continent_crust_points = points[mask_cr]

        # bin-left and bin-right points
        continent_crust_points_left = continent_crust_points.copy()
        continent_crust_points_left[:, 0] -= bin_size/2.0
        continent_crust_points_right = continent_crust_points.copy()
        continent_crust_points_right[:, 0] += bin_size/2.0

        initial_X_left = self.particle_initial_X_func(continent_crust_points_left[:, 0]/self.Max0, continent_crust_points_left[:, 1]/self.Max0)
        initial_X_right = self.particle_initial_X_func(continent_crust_points_right[:, 0]/self.Max0, continent_crust_points_right[:, 1]/self.Max0)
        mask_ini = (np.isfinite(initial_X_left) & np.isfinite(initial_X_right))

        # combined mask
        mask = mask_cr.copy()
        mask[mask_cr] = mask_ini 

        # compute dimention_ratio for valid points
        continent_crust_points_left_ini_valid = continent_crust_points_left[mask_ini]
        continent_crust_points_right_ini_valid = continent_crust_points_right[mask_ini]
        initial_X_left_valid = initial_X_left[mask_ini]
        initial_X_right_valid = initial_X_right[mask_ini]
        points_valid = points[mask]
        self.grid["dimention_ratio_bin"][mask] = (continent_crust_points_right_ini_valid[:, 0] - continent_crust_points_left_ini_valid[:, 0])/\
                                                (initial_X_right_valid - initial_X_left_valid)
        
        # interpolate a dimential_ratio function
        self.dimention_ratio_func = KNNInterpolatorND(
            np.vstack((points_valid[:, 0]/self.Max0, points_valid[:, 1]/self.Max0)).T,
            self.grid["dimention_ratio_bin"][mask].ravel(),
            k=1,
            max_distance=0.01
        )

        # derive and save a profile of dimention ratio
        # this profile is buried below the local topography by bury_depth
        suture_shallow_x = self.suture_profile_x[0]
        n_bins_foo = profile_half_size / profile_bin_size
        n_bins = int(round(n_bins_foo))

        assert math.isclose(n_bins_foo, n_bins, rel_tol=0.0, abs_tol=1e-12), (
            f"profile_half_size/profile_bin_size must be an integer. "
            f"Got {profile_half_size}/{profile_bin_size} = {n_bins_foo}"
        )

        profile_x = np.arange(suture_shallow_x - profile_half_size, 
                              suture_shallow_x + profile_half_size + 0.1, 
                              profile_bin_size)
        
        profile_y = self.Max0 + self.topography_func(profile_x/self.Max0) - profile_bury_depth

        profile_dimention_ratio = self.dimention_ratio_func(profile_x/self.Max0, profile_y/self.Max0)

        profile_bin_length0 = np.ones(profile_x.shape) * profile_bin_size
        profile_bin_length = profile_dimention_ratio * profile_bin_size
        profile_bin_deformation = (profile_dimention_ratio - 1.0) * profile_bin_size

        data = np.column_stack([profile_x, profile_y, 
                                profile_bin_length0, profile_bin_length,
                                profile_dimention_ratio, profile_bin_deformation])

        filename = "%s_%05d.txt" % ("dimention_ratio", self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)

        np.savetxt(
            filepath,
            data,
            header="# x y size0 size dimention_ratio deformation",
            comments="# suture x at %.2e m, with profiles extend %.2e m on either side.\n" % (suture_shallow_x, profile_half_size),
            fmt="%.4e"
        )

        print("\tsaved file %s" % filepath)

        # Derive shortening on the upper plate and lower plate
        lower_profile_subarray = profile_bin_deformation[0:n_bins]
        lower_valid_proflie_subarray = lower_profile_subarray[np.isfinite(lower_profile_subarray)]
        upper_profile_subarray = profile_bin_deformation[n_bins:]
        upper_valid_proflie_subarray = upper_profile_subarray[np.isfinite(upper_profile_subarray)]
        outputs["lower_plate_deformation"] = lower_valid_proflie_subarray.sum()
        outputs["upper_plate_deformation"] = upper_valid_proflie_subarray.sum()

        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

        return outputs
    
    def get_active_deforming_region_near_suture(self, percentile, *,
        depth_bin=10e3,
        x_half_dist=500e3,
        max_depth=None):
        """
        Identify active deforming regions based on the percentile of
        log10(strain_rate) within each depth bin.
    
        The threshold for each depth bin is computed from the specified
        x-range and then applied to the entire depth bin.
    
        Parameters
        ----------
        percentile : float
            Percentile (0-100) used to define the threshold.
        depth_bin : float, optional
            Thickness of depth bins in meters. Default is 10 km.
        x_half_dist : float, optional
            Determine the region to inspect, relative to the suture point
        max_depth : float, optional
            Maximum depth (m) to analyze.
            If None, use the deepest point in the dataset.
    
        Returns
        -------
        active_mask : ndarray of bool
            Boolean mask identifying active deforming regions.
        """
        my_assert(
            (self.suture_profile_x is not None) and (self.suture_profile_depths is not None),
            PYVISTA_PROCESS_WORKFLOW_ERROR,
            "Needs to first process suture position")

        start = time.time()

        # Get the coordinates.
        # Also parse option of x range, relative to suture position
        x = self.grid.points[:, 0]
        y = self.grid.points[:, 1]
        
        suture_shallow_x = self.suture_profile_x[0]
        x_range = [suture_shallow_x - x_half_dist, suture_shallow_x + x_half_dist]
        
        # Depth below the local surface
        # Also parse option of depth and create edges of bins
        surface_y = self.Max0 + self.topography_func(x / self.Max0)
    
        depth = surface_y - y
    
        if max_depth is None:
            max_depth = depth.max()
        
        depth_edges = np.arange(0.0, max_depth + depth_bin, depth_bin)

        # Get the strain rate field
        # Also derive a valid mask for positive values
        strain_rate = self.grid["strain_rate"]
    
        valid = (strain_rate > 0.0)

        # Initate the active_mask as all False
        active_mask = np.zeros(strain_rate.shape, dtype=bool)
    
        # For each depth bin wrapped by edges:
        # Look for a percentile for threshold of log values
        # Then select strain rate beyond this percentile
        for d0, d1 in zip(depth_edges[:-1], depth_edges[1:]):
    
            # Region used to compute the threshold
            sample = (
                valid &
                (x >= x_range[0]) &
                (x <= x_range[1]) &
                (depth >= d0) &
                (depth < d1)
            )
    
            if np.count_nonzero(sample) == 0:
                continue
    
            log_threshold = np.percentile(
                np.log10(strain_rate[sample]),
                percentile
            )
    
            # Apply threshold to the entire depth interval
            apply = (
                valid &
                (depth >= d0) &
                (depth < d1)
            )
    
            active_mask[apply] = (
                np.log10(strain_rate[apply]) > log_threshold
            )
    
        self.grid["active_deforming"] = active_mask.astype(np.uint8)

        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))
    

    def pin_vertical_profiles(self,
                              pin_dist_x=10e3,
                              pin_dist_y=2e3,
                              pin_max_depth=30e3):

        # This requires that particles are included
        # and that the topography is first processed.
        assert(self.include_particles)
        my_assert((self.topography_func is not None), PYVISTA_PROCESS_WORKFLOW_ERROR,
                  "%s requires the topography function, the particle results and the suture profile" % func_name())
        
        start = time.time()

        # Current particle positions and
        # Initial particle positions
        points = self.particles.points
        x_p = points[:, 0]
        y_p = points[:, 1]

        try:
            initial_position = self.particles["initial position"]
        except KeyError:
            raise RuntimeError("Particle field 'initial position' is not available.")

        initial_X = initial_position[:, 0]
        initial_Y = initial_position[:, 1]

        # Create pin grid
        pin_x = np.arange(0.0, self.Max2 + pin_dist_x, pin_dist_x)

        pin_y_max = self.Max0 + self.topography_func(pin_x/self.Max0)
        pin_depth = np.arange(0.0, pin_max_depth+pin_dist_y, pin_dist_y)

        # pin_X, pin_Y = np.meshgrid(pin_x, pin_y, indexing="xy")
        # Create the pin grid
        pin_X = np.tile(pin_x, (len(pin_depth), 1))
        pin_Y = pin_y_max[None, :] - pin_depth[:, None]

        pin_points = np.column_stack((
            pin_X.ravel(),
            pin_Y.ravel()
        ))

        # Build KD-tree using initial particle positions
        tree = cKDTree(np.column_stack((initial_X, initial_Y)))

        # Find nearest particle for every pin
        # and Reshape to match the pin grid
        pin_distance, pin_particle_idx = tree.query(pin_points)

        pin_particle_idx = pin_particle_idx.reshape(pin_X.shape)
        pin_distance = pin_distance.reshape(pin_X.shape)


        # Pinned points: Current locations 
        # and Initial locations (for reference)
        pin_current_X = x_p[pin_particle_idx]
        pin_current_Y = y_p[pin_particle_idx]

        pin_initial_X = initial_X[pin_particle_idx]
        pin_initial_Y = initial_Y[pin_particle_idx]

        # Connect pinned points into lines at every
        # x position, and stored in a pyvista object
        n_depth, n_x = pin_particle_idx.shape

        profiles = []
        profile_lengths = []
        local_lengths = []

        for j in range(n_x):

            # Current coordinates of this profile
            pts = np.column_stack((
                pin_current_X[:, j],
                pin_current_Y[:, j],
                np.zeros(n_depth)
            ))

            # Compute total profile length
            segment_lengths = np.linalg.norm(
                np.diff(pts, axis=0),
                axis=1
            )
            profile_length = segment_lengths.sum()
            profile_lengths.append(profile_length)

            # Local length record the distance of a
            # point on the profile to the next and ends
            # with 0.0 for the deepest point.
            local_length = np.append(segment_lengths, 0.0)
            local_lengths.append(local_length)

            # Create a polyline
            poly = pv.lines_from_points(pts)

            my_assert(pts.shape[0] == poly.n_points, ValueError, 
                      "pts.shape[0] is %d, while poly.n_points is %d" % (pts.shape[0], poly.n_points))

            profiles.append(poly)


        pin_lines = pv.merge(profiles, merge_points=False) 

        # Add proflie length, and local length to pin_lines
        profile_lengths_point = np.repeat(profile_lengths, n_depth)

        pin_lines["profile_length"] = profile_lengths_point

        pin_lines["local_length"] = np.concatenate(local_lengths)

        # Add current depth to pin_lines
        pin_y_max = self.Max0 + self.topography_func(pin_x / self.Max0)

        current_surface_y = np.repeat(pin_y_max, n_depth)

        current_depth = current_surface_y - pin_current_Y.ravel(order="F")

        pin_lines["depth"] = current_depth
        
        # Add initial depth to pin_lines
        initial_y_max = np.full(pin_x.shape, self.Max0)

        iniital_surface_y = np.repeat(initial_y_max, n_depth)

        initial_depth = iniital_surface_y - pin_initial_Y.ravel(order="F")

        pin_lines["initial_depth"] = initial_depth

        # output pyvista object
        filename = "%s_%05d.vtp" % ("pin_lines", self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        pin_lines.save(filepath)
        
        print("%ssaved file %s" % (4*" ", filepath))

        # Interpolate back the value of profile length to each grid point
        interp = NearestNDInterpolator(
            np.column_stack((pin_current_X.ravel(order="F"), pin_current_Y.ravel(order="F"))),
            profile_lengths_point)
        
        profile_length_interp = interp(
            self.grid.points[:, 0],
            self.grid.points[:, 1]
        )

        self.grid["pin_profile_length"] = profile_length_interp

        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

    # todo_pin
    def get_accreted_region(self, *,
                            profile_length_threshold=80e3,
                            sp_composition_threshold=0.5,
                            depth_threshold=40e3):
        """
        Get the accreted_region
        """

        assert(self.include_particles)
        my_assert((self.topography_func is not None), PYVISTA_PROCESS_WORKFLOW_ERROR,
                  "%s requires the topography function, the particle results and the suture profile" % func_name())

        # get depth at points 
        points = self.grid.points
        x_p = points[:, 0]
        y_p = points[:, 1]

        depth_p = self.Max0 + self.topography_func(x_p/self.Max0) - y_p

        # create mask bese on value of pined profile length, total subduction composition
        # and depth of points
        accreted_mask = (self.grid["pin_profile_length"] > profile_length_threshold) & \
        (self.grid["sp_total"] > sp_composition_threshold) & \
        (depth_p < depth_threshold)
        
        self.grid["accreted"] = accreted_mask.astype(np.uint8)


    def extract_additionals(self, *,
                      upper_lower_plate=True,
                      threshold=0.8,
                      suture_max_depth = 100e3,
                      suture_depth_interval = 5e3
                      ):
        """
        Interpolate particle-derived quantities onto the grid and construct final compositional fields.
    
        This function:
        - Interpolates initial particle X positions onto grid points
        - Splits crust into subducting plate (sp) and overriding plate (ov)
        - Adds derived compositional fields
        - Generates a composition indicator field
        - Writes the final grid to file
    
        Requirements:
            self.particle_initial_X_func must be initialized (via process_particles)
    
        Attributes modified:
            self.grid : updated with new scalar fields
        """
    
        start = time.time()

        if upper_lower_plate:
            # distinguish upper / lower mantle composition 
            my_assert(
                self.particle_initial_X_func is not None,
                PYVISTA_PROCESS_WORKFLOW_ERROR,
                "Needs to first process upper/lower late"
            )
        
            points = self.grid.points
            x_all = points[:, 0]
            y_all = points[:, 1]
        
            # interpolate initial X from particles → grid
            initial_X = self.particle_initial_X_func(x_all/self.Max0, y_all/self.Max0)
            self.grid['initial_X'] = initial_X
        
            # distinguish subducting plate (sp) and overriding plate (ov) crust
            crust_upper_comps = self.grid['crust_upper']
            sp_crust_upper_comps = crust_upper_comps * (initial_X < self.plate_start_point * 1.01)
            ov_crust_upper_comps = crust_upper_comps * (initial_X > self.slab_hinge_point * 0.99)
            self.grid['sp_crust_upper'] = sp_crust_upper_comps
            self.grid['ov_crust_upper'] = ov_crust_upper_comps

            crust_lower_comps = self.grid['crust_lower']
            sp_crust_lower_comps = crust_lower_comps * (initial_X < self.plate_start_point * 1.01)
            ov_crust_lower_comps = crust_lower_comps * (initial_X > self.slab_hinge_point * 0.99)
            self.grid['sp_crust_lower'] = sp_crust_lower_comps
            self.grid['ov_crust_lower'] = ov_crust_lower_comps

            # process suture profile

            self.suture_profile_depths = np.arange(
                0.0,
                suture_max_depth + suture_depth_interval,
                suture_depth_interval
            )

            self.suture_profile_x = np.full(
                self.suture_profile_depths.shape,
                np.nan
            )

            for i, depth in enumerate(self.suture_profile_depths):

                y_top = self.Max0 - depth
                y_bottom = y_top - suture_depth_interval

                mask = (
                    (y_all <= y_top)
                    & (y_all > y_bottom)
                    & ((ov_crust_upper_comps+ov_crust_lower_comps) > threshold)
                )

                if np.any(mask):
                    self.suture_profile_x[i] = x_all[mask].min()

            self.export_suture_profile()

    
        # compute composition indicator
        self.add_composition_indicator(upper_lower_plate=upper_lower_plate)
    
    
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

    def export_suture_profile(self):

        start = time.time()
        
        my_assert(
            (self.suture_profile_x is not None) and (self.suture_profile_depths is not None),
            PYVISTA_PROCESS_WORKFLOW_ERROR,
            "Needs to first process suture position")

        valid_foo = ~np.isnan(self.suture_profile_x)

        my_assert(np.sum(valid_foo) >= 2,
                ValueError,
                "Need at least two valid suture profile points.")

        x_foo = self.suture_profile_x[valid_foo]
        depth_foo = self.suture_profile_depths[valid_foo]

        points_foo = np.column_stack([
            x_foo,
            self.Max0 - depth_foo,
            np.zeros_like(x_foo)
        ])

        poly_foo = pv.PolyData(points_foo)

        poly_foo.lines = np.hstack([
            [len(points_foo)],
            np.arange(len(points_foo))
        ])

        poly_foo["depth"] = depth_foo

        filename = "%s_%05d.vtp" % ("suture", self.pvtu_step)
        filepath = os.path.join(self.pyvista_outdir, filename)
        poly_foo.save(filepath)
        
        print("%ssaved file %s" % (4*" ", filepath))
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))
    
    
    def add_background(self):
        """
        Compute and add a 'background' composition field.
    
        This function:
        - Stacks all composition fields
        - Computes residual composition: background = 1 - sum(compositions)
        - Clamps values to [0, 1] for numerical stability
    
        Attributes used:
            self.composition_names : list of composition field names
    
        Attributes added:
            self.grid["background"]
        """
    
        # stack compositions → shape (n_points, N)
        arrays = [self.grid[f] for f in self.composition_names]
        comp_data = np.column_stack(arrays)
    
        # compute residual background
        background = 1.0 - np.sum(comp_data, axis=1)
    
        # clamp to valid range
        background = np.clip(background, 0.0, 1.0)
    
        self.grid["background"] = background
    
    
    def add_composition_indicator(self, *,
                                  upper_lower_plate=True):
        """
        Construct a discrete composition indicator field based on dominant component.
    
        This function:
        - Builds derived fields (oceanic crust, asthenosphere, lithosphere)
        - Stacks all candidate composition fields
        - Assigns each point an integer label based on the maximum component
    
        Indicator mapping (by index):
            0 : asthenosphere (background, high T)
            1 : subducting plate upper crust
            2 : subducting plate lower crust
            3 : oceanic crust (gabbro + MORB)
            4 : sediment
            5 : lithosphere (background, low T)
            6 : overriding plate upper crust
            7 : overriding plate lower crust
    
        Attributes added:
            self.grid.point_data['composition_indicator']
        """
    
        # combine oceanic crust components
        oceanic_crust_field_names = ['gabbro', 'MORB']
        oceanic_crust_fields = np.column_stack([self.grid[f] for f in oceanic_crust_field_names])
        oceanic_crust_field = np.sum(oceanic_crust_fields, axis=1)
    
        # split background into asthenosphere vs lithosphere using temperature
        T = self.grid["T"]
        asthenosphere_field = self.grid['background'] * (T > self.lithospheric_T)
        lithosphere_field = self.grid['background'] * (T <= self.lithospheric_T)
    
        # stack all fields → shape (n_points, 8)
        if upper_lower_plate:
            arrays = [
                asthenosphere_field,
                self.grid['sp_crust_upper'],
                self.grid['sp_crust_lower'],
                oceanic_crust_field,
                self.grid['sediment'],
                lithosphere_field,
                self.grid['ov_crust_upper'],
                self.grid['ov_crust_lower']
            ]
        else:
            arrays = [
                asthenosphere_field,
                self.grid['crust_upper'],
                self.grid['crust_lower'],
                oceanic_crust_field,
                self.grid['sediment'],
                lithosphere_field
            ]
    
        data = np.column_stack(arrays)
    
        # determine dominant composition index
        indicator = np.argmax(data, axis=1)
    
        # store as integer field
        self.grid.point_data['composition_indicator'] = indicator.astype(np.int32)


def ProcessVtuFileTwoDStep(case_path, pvtu_step, Case_Options, *,
                           pyvista_outdir=None,
                           include_particles=False,
                           include_topography=False,
                           analyze_shortening=False,
                           analyze_deformation=False):
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
      time_step = Case_Options.summary_df.loc[idx, "Time step number"].values[0]
    except IndexError:
        raise IndexError("The pvtu_step %d doesn't seem to exist in this case" % pvtu_step)
    
    # output directory
    if pyvista_outdir is None:
        pyvista_outdir = os.path.join(Case_Options.pyvista_dir, "%05d" % pvtu_step)
    
    if not os.path.isdir(pyvista_outdir):
        os.mkdir(pyvista_outdir)
    
    # dict for saving outputs 
    outputs = {}

    # initiate the processing class
    # retrive the visualization file in case we perform a two-stage model
    # and the outputs are distributed in different folders
    file_name = Case_Options.retrieve_visualization_file(pvtu_step)
    ProcessCollision = PYVISTA_PROCESS_COLLISION(os.path.join(Case_Options.case_dir, os.path.dirname(file_name)), Case_Options.options,
                                                 pyvista_outdir=pyvista_outdir,
                                                 include_particles=include_particles)

    # read file
    ProcessCollision.read(pvtu_step)

    # read topography data
    if include_topography:
        try:
            ProcessCollision.load_topograph(time_step)
        except FileExistsError:
            ProcessCollision.extract_topography(dx=5e3, dr=0.001, interp_dx=5e3, output_surface=True)

    # extract slab
    ProcessCollision.extract_slab(output_surfuce=True)

    # analyze slab
    outputs1 = ProcessCollision.analyze_slab()
    outputs.update(**outputs1) 
    
    outputs2 = ProcessCollision.analyze_velocity(outputs["trench_center_50"])
    outputs.update(**outputs2) 

    # extract iso-volume objects
    # ProcessCollision.extract_continent_crust_iso_volumes()
    # ProcessCollision.extract_continent_lithosphere_iso_volumes(fields=Case_Options.options["COMPOSITION_FIELDS"])
    # ProcessCollision.extract_oceanic_plate_iso_volumes(include_harzburgite=Case_Options.options["HAS_HARZBURGITE"])

    # particles workflow
    if include_particles:

        # process the particle file and see if the initial particle
        # position presents. If so, differentiate the upper/lower 
        # plate with it.
        upper_lower_plate = True
        try:
            ProcessCollision.process_particles()
        except PYVISTA_PROCESS_COLLISION.InitialParticlePositionException:
            upper_lower_plate = False
    
    # extract the final output file
    if include_particles:
        ProcessCollision.extract_additionals(upper_lower_plate=upper_lower_plate)

    # analyze shortening in continental crust
    if analyze_shortening:
        my_assert (include_topography and include_particles, PYVISTA_PROCESS_WORKFLOW_ERROR, 
                   "To perform analyze_shortening, include_topography and include_particles must be true")

        # pin vertial profiles
        ProcessCollision.pin_vertical_profiles()

        # get accreted region
        ProcessCollision.get_accreted_region()       

        # analyze shortening by bins
        outputs_foo = ProcessCollision.analyze_shortening_by_bin()
        outputs.update(**outputs_foo)

    # analyze deforming region by percentile (first entry, e.g. 95)
    # near the suture by half-distance
    if analyze_deformation:
        ProcessCollision.get_active_deforming_region_near_suture(95,
        depth_bin=10e3,
        x_half_dist=500e3,
        max_depth=100e3)
    
    # write final output
    ProcessCollision.write_object_to_file(ProcessCollision.grid, "final", "vtu")
    

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
    img_dir = kwargs.get("img_dir", "img")

    # Inputs
    eps_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, img_dir, "prep")
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

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, img_dir), crop_box)

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
    img_dir = kwargs.get("img_dir", "img")

    # Inputs
    eps_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, img_dir, "prep")
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

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, img_dir), crop_box)

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

def finalize_visualization_2d_05012026(local_dir, file_name, _time, frame_png_file_with_ticks, **kwargs):

    from hamageolib.utils.plot_helper import convert_eps_to_pdf, extract_image_by_size, overlay_images_on_blank_canvas,\
    add_text_to_image

    # Options
    add_time = kwargs.get("add_time", True)
    canvas_size = kwargs.get("canvas_size", (1500, 700))
    img_dir = kwargs.get("img_dir", "img")

    # Inputs
    eps_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.eps" % (file_name, _time))
    pdf_file = os.path.join(local_dir, img_dir, "pv_outputs", "%s_t%.4e.pdf" % (file_name, _time))

    if (not os.path.isfile(eps_file)) and (not os.path.isfile(pdf_file)):
        raise FileNotFoundError(f"Neither the EPS nor pdf exists: {eps_file}, {pdf_file}")

    if not os.path.isfile(frame_png_file_with_ticks):
        raise FileNotFoundError(f"The PNG file with ticks does not exist: {frame_png_file_with_ticks}")

    # Outputs
    # Paths to output files

    prep_file_dir = os.path.join(local_dir, img_dir, "prep")
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

    full_image_path = extract_image_by_size(pdf_file, target_size, os.path.join(local_dir, img_dir), crop_box)

    # Overlays multiple images on a blank canvas with specified sizes, positions, cropping, and scaling.
    overlay_images_on_blank_canvas(
        canvas_size=canvas_size,  # Size of the blank canvas in pixels (width, height)
        image_files=[full_image_path, frame_png_file_with_ticks],  # List of image file paths to overlay
        image_positions=[(100, 0), (100, -47)],  # Positions of each image on the canvas
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


def read_topography_data(local_dir_2d, Case_Options_2d, plot_time_p, *,
                         time_interval=1e5):
    """
    Read topography data from an ASPECT topography output file at a specified time.

    The function locates the simulation output time step closest to
    ``plot_time_p``, verifies that the match is within a tolerance of
    1e4 years, reads the corresponding topography file, and returns the
    horizontal coordinate and topography arrays.

    Parameters
    ----------
    local_dir_2d : str
        Path to the local ASPECT case directory containing the
        ``output/topography`` folder.

    Case_Options_2d : CASE_OPTIONS
        Case options object that contains the simulation summary data and
        provides the ``resample_visualization_df`` method.

    plot_time_p : float
        Target model time (years) for which topography data should be
        extracted.

    time_interval : float, optional
        Time interval (years) used when resampling the visualization
        output table. Default is ``1e5``.

    Returns
    -------
    x : numpy.ndarray
        Horizontal coordinate values from the topography file.

    topography : numpy.ndarray
        Surface elevation values corresponding to ``x``.

    Raises
    ------
    ValueError
        If no simulation output time is found within 1e4 years of
        ``plot_time_p``.

    AssertionError
        If the corresponding topography file does not exist.

    Notes
    -----
    The topography file is expected to have three columns:

    - Column 0: horizontal coordinate (x)
    - Column 1: vertical coordinate (y)
    - Column 2: topography

    The file is read using ``numpy.loadtxt`` with lines beginning with
    ``#`` treated as comments.
    """
    
    resampled_df = Case_Options_2d.resample_visualization_df(time_interval)

    graphical_steps = resampled_df["Vtu step"].values

    idx = (Case_Options_2d.summary_df["Time"] - plot_time_p).abs().idxmin()

    _time = Case_Options_2d.summary_df.loc[idx, "Time"]
    my_assert(np.isclose(_time, plot_time_p, atol=1e4), ValueError, "Time %.1e is not found for %s" % (plot_time_p, local_dir_2d))

    time_step = Case_Options_2d.summary_df.loc[idx, "Time step number"]
    
    topography_file = os.path.join(local_dir_2d, "output/topography", "topography.%05d" % time_step)

    assert(os.path.isfile(topography_file))

    # Extract data
    data = np.loadtxt(topography_file, comments="#")

    x = data[:, 0]
    topography = data[:, 2]

    return x, topography


def prepare_case_option_2d(_dir_2d, is_process_second_stage, *, 
                           prm_basename_2d="case.prm", 
                           wb_basename_2d = "case.wb", 
                           output_directory="output", 
                           second_stage_outputs="output_re", 
                           pp_directory=None):
    """
    Initialize and prepare a 2-D case for post-processing.

    This helper function constructs a ``CASE_OPTIONS_TWOD`` object using the
    appropriate directories and filenames for either the first-stage or
    second-stage simulation outputs. It then interprets the case configuration
    and generates a summary of available VTU output steps.

    Parameters
    ----------
    _dir_2d : str
        Path to the case directory.

    is_process_second_stage : bool
        Whether to process the second-stage simulation outputs. If ``True``,
        the function uses the second-stage output, image, and PyVista
        directories (e.g., ``output_re``, ``img_1``,
        ``pyvista_outputs_1``). Otherwise, the default first-stage
        directories are used.

    prm_basename_2d : str, optional
        Name of the parameter file. Default is ``"case.prm"``.

    wb_basename_2d : str, optional
        Name of the World Builder input file. Default is ``"case.wb"``.

    output_directory : str, optional
        Name of the primary simulation output directory. Default is
        ``"output"``.

    second_stage_outputs : str, optional
        Name of the second-stage output directory. Used only when
        ``is_process_second_stage`` is ``True``. Default is ``"output_re"``.

    pp_directory : str, optional
        Directory containing post-processing results. If ``None``, the
        default location defined by ``CASE_OPTIONS_TWOD`` is used.

    Returns
    -------
    Case_Options_2d : CASE_OPTIONS_TWOD
        Initialized and interpreted case object.

    Notes
    -----
    This function calls

    - ``Interpret()`` to read and process the case configuration.
    - ``SummaryCaseVtuStep()`` to generate a summary of available VTU output
      files for subsequent post-processing.
    """
    Case_Options_2d = None
    if is_process_second_stage:
        Case_Options_2d = CASE_OPTIONS_TWOD(_dir_2d, 
                                          case_file=prm_basename_2d, 
                                          wb_basename=wb_basename_2d, 
                                          output_directory=second_stage_outputs,
                                          pyvista_basename="pyvista_outputs_1",
                                          image_directory="img_1",
                                          summary_filename="summary_1.csv",
                                          pp_directory=pp_directory)
    else:
        Case_Options_2d = CASE_OPTIONS_TWOD(_dir_2d,
                                          case_file=prm_basename_2d, 
                                          wb_basename=wb_basename_2d, 
                                          output_directory=output_directory,
                                          pp_directory=pp_directory)

    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(Case_Options_2d.summary_file)

    return Case_Options_2d




def plot_run_time_combined(dirs_2d, Case_Options_2d_array, *,
                           use_time_mask=False,
                           start_time=None,
                           end_time=None):
    """
    Generate comparison plots of runtime statistics for multiple ASPECT cases.

    This function compares selected runtime quantities from several simulation
    cases on a common figure. Currently, it plots

    - Time step number
    - Corrected wall-clock time

    as functions of model time. The data are interpolated onto a uniform
    0.1 Myr time grid to facilitate direct comparison between simulations with
    different output intervals.

    Parameters
    ----------
    dirs_2d : list of str
        List of directories containing the simulation results. The directory
        names are also used as legend labels.

    Case_Options_2d_array : list
        List of ``Case_Options`` objects corresponding to ``dirs_2d``.
        Each object must contain the attributes

        - ``statistic_df`` (pandas.DataFrame)
        - ``time_df`` (pandas.DataFrame)

        with a common ``Time`` column.

    use_time_mask : bool, optional
        Whether to restrict the plotted time range using ``start_time`` and
        ``end_time``. Default is ``False``.

    start_time : float, optional
        Beginning of the plotted time interval (model time, in years).
        If ``None``, plotting starts from the beginning of the simulation.
        Note that this and the end_time just limit the plot range, if not other wise be 
        required by use_time_mask

    end_time : float, optional
        End of the plotted time interval (model time, in years).
        If ``None``, plotting continues until the end of the simulation.

    is_process_second_stage : bool, optional
        If ``True``, save figures to the ``img_1/runtime_plots`` directory.
        Otherwise, save them to ``img/runtime_plots``. Default is ``False``.

    Notes
    -----
    - Runtime data are interpolated to a uniform spacing of 0.1 Myr before
      plotting.
    - Wall-clock time is converted from seconds to hours.
    - Figures are saved in both PNG and PDF formats as
      ``assembled.png`` and ``assembled.pdf``.
    - Matplotlib plotting parameters are temporarily modified for publication-
      quality output and restored to their defaults before returning.

    Returns
    -------
    None
    """

    import hamageolib.utils.plot_helper as plot_helper
    from matplotlib import rcdefaults
    from copy import deepcopy

    # Retrieve the default color cycle
    default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

    # Example usage
    # Rule of thumbs:
    # 1. Set the limit to something like 5.0, 10.0 or 50.0, 100.0 
    # 2. Set five major ticks for each axis
    scaling_factor = 1.0  # scale factor of plot
    font_scaling_multiplier = 2.0 # extra scaling multiplier for font
    legend_font_scaling_multiplier = 0.5
    line_width_scaling_multiplier = 2.0 # extra scaling multiplier for lines

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

    # plot_attr_types - which variable in the Case_Options object the attribute belongs to
    # plot_attrs - the names of the attribute to plot
    hr = 3600.0 # s in hr
    plot_attr_types = ["statistic_df", "time_df"]
    plot_attrs = ["Time step number", "Corrected Wall Clock"]
    scalings = [1.0, 1.0/hr]
    units = [None, "hr"]

    n_col = 2
    n_row = int(np.ceil(float(len(plot_attrs)) / n_col))
    fig = plt.figure(figsize=(10*n_col*scaling_factor, 6*n_row*scaling_factor), tight_layout=True)
    gs = gridspec.GridSpec(n_row, n_col)

    # Loop for 2d cases
    for j, plot_attr in enumerate(plot_attrs):
        i_row = j // n_col
        j_col = j % n_col
        ax = fig.add_subplot(gs[i_row, j_col])
        for i, _dir_2d in enumerate(dirs_2d):

            Case_Options_2d = Case_Options_2d_array[i]

            # raw data: append 0.0 at the start
            attr_df = getattr(Case_Options_2d, plot_attr_types[j])
            xs = deepcopy(attr_df.Time.to_numpy(dtype=float))
            xs = np.insert(xs, 0, 0.0)

            ys = deepcopy(attr_df[plot_attr].to_numpy(dtype=float))
            ys = np.insert(ys, 0, 0.0)

            ys*=scalings[j]

            xs_p = np.arange(0.0, np.max(xs)+0.1e6, 0.1e6)
            ys_p = np.interp(xs_p, xs, ys)

            # plotting mask, if either start_time or end_time
            # is given.
            # otherwise mask is all true
            mask = np.ones_like(xs_p, dtype=bool)

            if use_time_mask:
                if start_time is not None:
                    mask &= (xs_p >= start_time)
                if end_time is not None:
                    mask &= (xs_p <= end_time)


            # plot
            ax.plot(xs_p[mask], ys_p[mask], color=default_colors[i], label=os.path.basename(_dir_2d))
        ax.set_xlabel("Time")
        ax.set_xlim(left=start_time, right=end_time)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("%s (%s)" % (plot_attr, str(units[j])))
        ax.grid()
        if j == 0:
            ax.legend()


    output_dir = os.path.join(Case_Options_2d_array[0].img_dir, "runtime_plots")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # save both png and pdf file
    # save both png and pdf file
    file_path = os.path.join(output_dir, "assembled.png")
    fig.savefig(file_path)
    print(f"Summary plots have been saved to: {file_path}")
    file_path = os.path.join(output_dir, "assembled.pdf")
    fig.savefig(file_path)
    print(f"Summary plots have been saved to: {file_path}")

    plt.close(fig)

    rcdefaults()


def process_all_vtu_steps(dir_2d, Case_Options_2d, *, 
                          graphical_step_min=None,
                          graphical_step_max=None,
                          one_vtu_step=None,
                          include_particles=False,
                          include_topography=False,
                          analyze_shortening=False,
                          analyze_deformation=False):
    """
    Process VTU outputs for one or more graphical output steps.

    This function performs PyVista-based post-processing on selected VTU
    output files from a simulation. For each graphical output step, it
    processes the corresponding PVTU file, extracts the requested
    post-processing quantities, updates the case summary table, and writes
    the updated summary file to disk.

    Parameters
    ----------
    dir_2d : str
        Path to the simulation case directory.

    Case_Options_2d : CASE_OPTIONS_TWOD
        Initialized case object containing simulation metadata and the VTU
        summary table.

    graphical_step_min : int, optional
        Process only graphical output steps greater than this value.

    graphical_step_max : int, optional
        Process only graphical output steps less than this value.

    one_vtu_step : int, optional
        If specified, process only this graphical output step. When provided,
        ``graphical_step_min`` and ``graphical_step_max`` are ignored.

    include_particles : bool, optional
        Whether to load and process particle data associated with each VTU
        output. Default is ``False``.

    include_topography : bool, optional
        Whether to extract and analyze surface topography. Default is
        ``False``.

    analyze_shortening : bool, optional
        Whether to compute crustal shortening and related diagnostics.
        Default is ``False``.

    Notes
    -----
    The function performs the following steps:

    1. Select the graphical output steps to process.
    2. Determine the corresponding PVTU file for each step.
    3. Call ``ProcessVtuFileTwoDStep()`` to perform the requested analyses.
    4. Update the summary table with the returned quantities.
    5. Export the updated summary table to
       ``Case_Options_2d.summary_file``.

    Returns
    -------
    None
    """

    graphical_steps_np = Case_Options_2d.summary_df["Vtu step"].to_numpy()
    graphical_steps = None
    if one_vtu_step is not None:
        graphical_steps = [one_vtu_step]
    else:
        # Start with all True and mask the range of steps
        mask = np.ones(graphical_steps_np.shape, dtype=bool)
    
        if graphical_step_min is not None:
            mask &= (graphical_steps_np > graphical_step_min)
    
        if graphical_step_max is not None:
            mask &= (graphical_steps_np < graphical_step_max)
        
        graphical_steps = [int(step) for step in graphical_steps_np[mask]]
    
    
    # Processing pyvista
    for step in graphical_steps:
    # while True: # debug
        # step = 0 # debug
    
        pvtu_step = Case_Options_2d.get_pvtu_step(step)
        outputs = ProcessVtuFileTwoDStep(dir_2d, pvtu_step, Case_Options_2d, 
                                         include_particles=include_particles, 
                                         include_topography=include_topography,
                                         analyze_shortening=analyze_shortening,
                                         analyze_deformation=analyze_deformation)
        # print("outputs: ", outputs) # debug
    
        for key, value in outputs.items():
            Case_Options_2d.SummaryCaseVtuStepUpdateValue(key, step, value)
        # break # debug
    
    Case_Options_2d.SummaryCaseVtuStepExport(Case_Options_2d.summary_file)

# todo_topo