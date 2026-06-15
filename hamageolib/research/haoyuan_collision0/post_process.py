import os
import numpy as np
import time
import pyvista as pv
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree
from hamageolib.core.post_process import PYVISTA_PROCESS, PYVISTA_PROCESS_WORKFLOW_ERROR
from hamageolib.utils.exception_handler import my_assert
from hamageolib.utils.interp_utilities import KNNInterpolatorND
from hamageolib.utils.handy_shortcuts_haoyuan import func_name
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD

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
        self.topography_points = None
        self.iso_volume_dict = {}

        # placeholder for interpolation functions
        self.particle_ul_func = None

        # placeholder for topography functions
        self.topography_func = None

        # placeholder for suture position     
        self.suture_point = None


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
            x,
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
    
    def extract_topography(self, *, dx=5e3, dr=0.001,
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
            self.topography_points  : original cleaned (x, y, z)
            self.topography_profile : interpolated (x, y) if interp_dx is not None
        '''
        start = time.time()
        print("PYVISTA_PROCESS:\n\t%s" % func_name())
    
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
            query_pt = np.array([x / self.Max2])
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
    
        # --- optional interpolation to regular grid ---
        if interp_dx is not None:
            x_reg = np.arange(x_surf[0], x_surf[-1], interp_dx)
            y_reg = np.interp(x_reg, x_surf, y_surf)
            z_reg = np.zeros_like(x_reg)
    
            self.topography_points = np.vstack([x_reg, y_reg, z_reg]).T
        else: 
            z_surf = np.zeros_like(x_surf)
            self.topography_points = np.vstack([x_surf, y_surf, z_surf]).T
    
        # --- optional output ---
        if output_surface:
            point_cloud = pv.PolyData(self.topography_points)
            filename = "topography_%05d.vtp" % self.pvtu_step
            filepath = os.path.join(self.pyvista_outdir, filename)
            point_cloud.save(filepath)
            print(f"\tSave file {filepath}")
        
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

    def analyze_shortening(self):
        
        start = time.time()

        my_assert((self.topography_func is not None) and (self.particle_initial_X_func is not None), 
                  PYVISTA_PROCESS_WORKFLOW_ERROR,
                  "%s requires both the topography function and the particle results" % func_name())
        
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

        # todo_short
        pass 
    
    def extract_final(self, *,
                      upper_lower_plate=True,
                      threshold=0.8):
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

            # todo_short
            # process suture profile
            suture_max_depth = 100e3
            suture_depth_interval = 5e3

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
                    & (ov_crust_upper_comps > threshold)
                )

                if np.any(mask):
                    self.suture_profile_x[i] = x_all[mask].min()

            print("self.suture_profile_x: ")
            print(self.suture_profile_x)

            self.export_suture_profile()

            # mask = ov_crust_upper_comps > threshold

            # assert np.any(mask), (
            #     f"No subducting-plate upper crust points found above threshold={threshold}."
            # )

            # min_x = x_all[mask].min()
            # self.suture_point = min_x

            # print("self.suture_point: ")
            # print(self.suture_point)

    
        # compute composition indicator
        self.add_composition_indicator(upper_lower_plate=upper_lower_plate)
    
        # write output
        self.write_object_to_file(self.grid, "final", "vtu")
    
        end = time.time()
        print("\tPYVISTA_PROCESS_THD: %s" % func_name())
        print("\ttakes %.1f s" % (end - start))

    
    # todo_short
    def export_suture_profile(self):

        start = time.time()
        
        my_assert(
            (self.suture_profile_x is not None) and (self.suture_profile_depths is not None),
            PYVISTA_PROCESS_WORKFLOW_ERROR,
            "Needs to first process suture position")

        valid_foo = ~np.isnan(self.suture_profile_x)

        if np.sum(valid_foo) < 2:
            raise ValueError(
                "Need at least two valid suture profile points."
            )

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
                           analyze_shortening=False):
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
        ProcessCollision.load_topograph(time_step)

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
        ProcessCollision.extract_final(upper_lower_plate=upper_lower_plate)
    
    if analyze_shortening:

        assert (include_topography and include_particles)
        # todo_short
        ProcessCollision.analyze_shortening()

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