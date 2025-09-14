# use the environment of py-gplate
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import re
import gplately

from matplotlib import gridspec
import cartopy.crs as ccrs
from shutil import rmtree

from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import map_mid_point, remove_substrings
from hamageolib.utils.exception_handler import my_assert
from plate_model_manager import PlateModelManager

# Retrieve the default color cycle
default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]

class GPLATE_PROCESS_WORKFLOW_ERROR(Exception):
    pass
class GPLATE_PROCESS():
    # Summary:
    # GPLATE_PROCESS orchestrates loading a plate-tectonic model, reconstructing subduction zones
    # at a specified geological time, and enriching the resulting subduction features with seafloor
    # age values from a raster. It manages output directories for images/csv and holds state needed
    # for plotting and downstream analysis.
    # Attributes (types and meaning):
    # - case_dir: str — root directory for this analysis run (created if missing).
    # - csv_dir: str — directory under case_dir where CSV outputs are stored (created if missing).
    # - model_name: Optional[str] — identifier of the plate model to use.
    # - reconstruction_time: Optional[int] — geological time (Ma) at which to reconstruct.
    # - anchor_plate_id: Optional[int] — plate ID used as rotation anchor (e.g., 0 for Africa).
    # - model: Optional[gplately.PlateReconstruction] — reconstruction engine for plate motions.
    # - plate_model: Optional[PlateModel] — model bundle returned by PlateModelManager.get_model(...).
    # - subduction_data: Optional[pd.DataFrame] — tabular subduction geometries and kinematics.
    # - gPlotter: Optional[GPLOTTER] — plotting helper bound to plate_model and model.
    # - all_columns: List[str] — column schema from gplately tessellation results.
    # - all_columns_1: List[str] — extended schema including 'age' from the raster.
    # Suggested renames:
    # - GPLATE_PROCESS -> GPlatesProcessor
    # - gPlotter -> plotter
    # - all_columns -> subduction_columns
    # - all_columns_1 -> subduction_columns_with_age


    def __init__(self, case_dir):
        # Summary:
        # Initialize processor state and ensure output directories exist.
        # Parameters:
        # - case_dir: str — base directory for outputs ("img", "csv") and local model data.
        # Returns:
        # - None
        # Side effects:
        # - Creates directories if they do not exist.
        # - Initializes attribute placeholders for model configuration and data.
        # Raises:
        # - None directly (os errors would propagate if permissions fail).
        # Implementation starts below.
        # Paths
        if not os.path.isdir(case_dir):
            os.mkdir(case_dir)
        if not os.path.isdir(os.path.join(case_dir, "img")):
            os.mkdir(os.path.join(case_dir, "img"))
        csv_dir = os.path.join(case_dir, "csv")
        if not os.path.isdir(csv_dir):
            os.mkdir(csv_dir)
        self.case_dir = case_dir
        self.csv_dir = csv_dir

        # Record options
        self.model_name = None
        self.reconstruction_time = None
        self.anchor_plate_id = None

        # Record reconstruction
        self.model = None
        self.plate_model = None

        # Record subduction data
        self.subduction_data = None

        # Plotter
        self.gPlotter = None

        # Define the columns used in the subduction data DataFrame
        # all_columns: original columns from gplately
        # all_columns_1: append sp age from raster
        self.all_columns = ['lon', 'lat', 'conv_rate', 'conv_angle', 'trench_velocity', 
                                'trench_velocity_angle', 'arc_length', 'trench_azimuth_angle', 
                                'subducting_pid', 'trench_pid']
        self.all_columns_1 = self.all_columns + ['age']


    def reconstruct(self, model_name, reconstruction_time, anchor_plate_id):
        # Summary:
        # Load a plate model, build a PlateReconstruction with a chosen anchor plate, and
        # tessellate subduction zones at the requested reconstruction time. Stores a DataFrame
        # of subduction features and prepares a GPLOTTER for visualization.
        # Parameters:
        # - model_name: str — key/name of the plate model available to PlateModelManager.
        # - reconstruction_time: int — time (in Ma) for reconstruction (validated to be int).
        # - anchor_plate_id: int — plate ID to use as the rotational anchor (e.g., 0 for Africa).
        # Returns:
        # - None
        # Side effects:
        # - Creates an image output subdirectory for this time step.
        # - Populates self.subduction_data (pd.DataFrame) with columns in self.all_columns.
        # - Initializes self.model, self.plate_model, self.gPlotter, and records config.
        # Raises:
        # - TypeError via my_assert if reconstruction_time is not int.
        # - File/IO errors if model data cannot be found or read by PlateModelManager.

        self.model_name = model_name
        my_assert(type(reconstruction_time) == int, TypeError, "reconstruction_time must be int")
        self.reconstruction_time = reconstruction_time

        # set up a directory to output for every step
        self.img_dir = os.path.join(os.path.join(self.case_dir, "img", "%05dMa" % reconstruction_time))
        if not os.path.isdir(self.img_dir):
            os.mkdir(self.img_dir)

        # Create an instance of the PlateModelManager to manage plate models
        pm_manager = PlateModelManager()

        # Load the plate model from the specified data directory
        plate_model = pm_manager.get_model(model_name, data_dir=os.path.join(self.case_dir, "plate-model-repo"))

        # Set up the PlateReconstruction model using the loaded plate model data
        # This includes rotation models, topologies, and static polygons, with the specified anchor plate ID
        # anchor_plate_id - anchor plate ID for the reconstruction model, 0 for Africa
        model = gplately.PlateReconstruction(
            plate_model.get_rotation_model(), 
            plate_model.get_topologies(), 
            plate_model.get_static_polygons(),
            anchor_plate_id=anchor_plate_id
        )

        # get the reconstruction of subduction zones
        subduction_data_raw = model.tessellate_subduction_zones(self.reconstruction_time, 
                                                            tessellation_threshold_radians=0.01, 
                                                            anchor_plate_id=anchor_plate_id,
                                                            ignore_warnings=True)



        self.subduction_data = pd.DataFrame(subduction_data_raw, columns=self.all_columns)

        self.gPlotter = GPLOTTER(plate_model, model)
        self.gPlotter.set_time(self.reconstruction_time)
        self.model = model
        self.plate_model = plate_model
        self.anchor_plate_id = anchor_plate_id

    def add_age_raster(self):
        # Summary:
        # Attach seafloor age to each subduction data point by interpolating a global age raster
        # at the current reconstruction time. Fills NaNs in the raster before interpolation to
        # avoid gaps along trench boundaries.
        # Parameters:
        # - None
        # Returns:
        # - None (updates self.subduction_data in place by adding an 'age' column).
        # Preconditions:
        # - self.reconstruct(...) has been called, and self.subduction_data is not None.
        # Raises:
        # - GPLATE_PROCESS_WORKFLOW_ERROR if reconstruct() was not called prior to this method.

        my_assert(self.subduction_data is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"reconstruct\" first.")
        # Initialize the age grid raster, which will be used for age-related computations
        self.age_grid_raster = gplately.Raster(
                                        data=self.plate_model.get_raster("AgeGrids", self.reconstruction_time),
                                        plate_reconstruction=self.model,
                                        extent=[-180, 180, -90, 90]
                                        )
        # fill Nan values, it seems to not cause any issue in interpolating the ages.
        # otherwise, there are many points where the trench point are not covered in the Raster.
        # Thus, it seems these points are just on the boundary where some other value could be filled.
        self.age_grid_raster.fill_NaNs(inplace=True)

        self.subduction_data['age'] = self.age_grid_raster.interpolate(self.subduction_data.lon, self.subduction_data.lat, method="nearest")
        
    
    def resample_subduction(self, arc_length_edge, arc_length_resample_section):
        # Summary:
        # Resample subduction-zone point sets for each unique subducting plate ID so that points are
        # spaced by a uniform arc-length step, trimming a margin at both ends. Stores results in
        # self.subduction_data_resampled and exports a CSV.
        # Parameters:
        # - arc_length_edge: float — arc-length margin (same units as 'arc_length') to exclude at each edge.
        # - arc_length_resample_section: float — resampling step in arc-length (uniform spacing).
        # Returns:
        # - None (updates internal attributes; writes a CSV by calling export_csv()).
        # Preconditions:
        # - self.subduction_data is a populated DataFrame from reconstruct().
        # Raises:
        # - GPLATE_PROCESS_WORKFLOW_ERROR if reconstruct() was not called.
        # - ValueError caught per-subduction zone; zones that fail resampling are skipped.
        
        my_assert(self.subduction_data is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"reconstruct\" first.")

        subduction_data = self.subduction_data

        self.arc_length_edge, self.arc_length_resample_section = arc_length_edge, arc_length_resample_section

        # get all subducting id values
        subducting_pids = subduction_data["subducting_pid"].unique()
        trench_pids = subduction_data["trench_pid"].unique()

        print("Total subduction zones: ", len(subducting_pids))
        print("Total trenches: ", len(trench_pids))
        print("subducting_pids: ", subducting_pids)

        # plot data by trench_pid
        data_list = [pd.DataFrame(columns=self.all_columns_1)]
        for i, subducting_pid in enumerate(subducting_pids):
            mask = subduction_data.subducting_pid == subducting_pid
            one_subduction_data = subduction_data[mask]
            try:
                one_subduction_data_resampled = resample_subduction(one_subduction_data, arc_length_edge, arc_length_resample_section)
            except ValueError:
                continue
            data_list.append(one_subduction_data_resampled)
            
        subduction_data_resampled = pd.concat(data_list)


        re_subducting_pids = subduction_data_resampled["subducting_pid"].unique()
        re_trench_pids = subduction_data_resampled["trench_pid"].unique()
        print("Total subduction zones (after resampling): ", len(re_subducting_pids))
        print("Total trenches (after resampling): ", len(re_trench_pids))
        print("subducting_pids (after resampling): ", re_subducting_pids)
        print("Total resampled points: ", len(subduction_data_resampled))

        self.subduction_data_resampled = subduction_data_resampled

        self.export_csv("subduction_data_resampled", "resampled_edge%.1f_section%.1f.csv" % (arc_length_edge, arc_length_resample_section))
    
    def export_csv(self, _name, file_name):
        # Summary:
        # Write a DataFrame attribute of this instance to CSV under self.csv_dir.
        # Parameters:
        # - _name: str — attribute name of the DataFrame to export (e.g., "subduction_data_resampled").
        # - file_name: str — target file name (e.g., "resampled_edgeX_sectionY.csv").
        # Returns:
        # - None (writes to disk and prints the destination path).
        # Raises:
        # - TypeError via my_assert if the named attribute is not a pandas DataFrame.

        csv_obj = getattr(self, _name)
        my_assert(isinstance(csv_obj, pd.DataFrame), TypeError, "object with name %s is not a pd.DataFrame" % _name)

        file_path = os.path.join(self.csv_dir, file_name)
        csv_obj.to_csv(file_path, index=False)
        print("Saved file %s" % file_path)

    def save_results_ori(self, inspect_all_slabs_in_separate_plots=False, only_one_pid=None, **kwargs):
        # Summary:
        # Render and save global and optional per-slab figures for the original (unresampled) subduction data.
        # Parameters:
        # - inspect_all_slabs_in_separate_plots: bool — if True, also produce per-slab diagnostic panels with
        #   point indices and trench IDs annotated.
        # - only_one_pid: None or value - if value, skip all other subducting pids.
        # Returns:
        # - None (saves .png/.pdf figures into an "ori" subdirectory).
        # Preconditions:
        # - self.subduction_data is available from reconstruct().
        # - self.age_grid_raster should exist if age shading is desired (set in add_age_raster()).
        # Raises:
        # - GPLATE_PROCESS_WORKFLOW_ERROR if reconstruct() was not called.
       
        my_assert(self.subduction_data is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"reconstruct\" first.")

        color_dict = kwargs.get("color_dict", {})
        
        local_img_dir = os.path.join(self.img_dir, "ori")
        if os.path.isdir(local_img_dir):
            rmtree(local_img_dir)
        os.mkdir(local_img_dir)

        # set alias for class attributes
        gPlotter = self.gPlotter
        subduction_data = self.subduction_data
        age_grid_raster = self.age_grid_raster
        reconstruction_time = self.reconstruction_time

        subducting_pids = subduction_data["subducting_pid"].unique()

        # start figure
        # plot the subducting_pid in the globe
        # Separatly handle the dataset and the trench / transform boundaries
        fig0 = plt.figure(figsize=(10, 12), dpi=100, tight_layout=True)
        gs = gridspec.GridSpec(2, 1)

        gPlotter.set_region("default")

        ax0 = fig0.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
        gPlotter.plot_global_basics(ax0, age_grid_raster=age_grid_raster)
        color_dict = gPlotter.plot_subduction_pts(ax0, subduction_data, color_dict=color_dict) 
        
        ax1 = fig0.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
        gPlotter.plot_global_basics(ax1, age_grid_raster=age_grid_raster, plot_boundaries=True)

        # test adding convergence vectors
        # plot_conv=True, stepping=5)

        for i, subducting_pid in enumerate(subducting_pids):
            
            if only_one_pid is not None and int(subducting_pid) != int(only_one_pid):
                # if only one pid is given, skip all others
                continue

            one_subduction_data = subduction_data[subduction_data.subducting_pid==subducting_pid]

            # add marker to summary plot
            ax0.text(one_subduction_data["lon"].iloc[0], one_subduction_data["lat"].iloc[0], str(subducting_pid), transform=ccrs.PlateCarree(),
                fontsize=8,
                ha="left",   # horizontal alignment
                va="bottom"  # vertical alignment
            )

            if inspect_all_slabs_in_separate_plots:
                # plot individual subduction zone
                # 0. mark all the boundaries in the plot
                # 1. mark indices of points
                # 2. mark trench pid values
                # 3. also include makers
                region = crop_region_by_data(one_subduction_data, 15.0)
                gPlotter.set_region(region) # set region to default, no need to filter
            
                fig = plt.figure(figsize=(20, 12), dpi=100, tight_layout=True)
                gs = gridspec.GridSpec(2, 2)

                # todo_gp 
                ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster, plot_boundaries=True)
                ax.set_extent(region, crs=ccrs.PlateCarree())
                
                ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                sub_color_dict = gPlotter.plot_subduction_pts(ax, one_subduction_data, color_dict=color_dict[int(subducting_pid)])
                for j, (lon, lat) in enumerate(zip(one_subduction_data.lon, one_subduction_data.lat)):
                    if j % 10 != 0:
                        continue
                    ax.text(lon, lat, str(j), transform=ccrs.PlateCarree(),
                        fontsize=8,
                        ha="left",   # horizontal alignment
                        va="bottom"  # vertical alignment
                    )
                ax.set_extent(region, crs=ccrs.PlateCarree())
                
                # update color dictionary for specific subduction zone 
                color_dict[int(subducting_pid)] = sub_color_dict

                ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                _ = gPlotter.plot_subduction_pts(ax, one_subduction_data, "trench_pid", color_dict=color_dict[int(subducting_pid)])
                trench_pids = one_subduction_data["trench_pid"].unique()
                ax.set_extent(region, crs=ccrs.PlateCarree())

                ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                _ = gPlotter.plot_subduction_pts(ax, one_subduction_data, "trench_pid", color_dict=color_dict[int(subducting_pid)])
                trench_pids = one_subduction_data["trench_pid"].unique()
                ax.set_extent(region, crs=ccrs.PlateCarree())
                for j, trench_pid in enumerate(trench_pids):
                    sub_subduction_data = one_subduction_data[one_subduction_data.trench_pid==trench_pid]
                    lon, lat = sub_subduction_data.lon.iloc[0], sub_subduction_data.lat.iloc[0]
                    ax.text(lon, lat, str(trench_pid), transform=ccrs.PlateCarree(),
                        fontsize=8,
                        ha="left",   # horizontal alignment
                        va="bottom"  # vertical alignment
                    )
            
                # save figure of individual subduction zone
                ofile_path = os.path.join(local_img_dir, "global_subduction_ori_t%.2fMa_pid%06d" % (float(reconstruction_time), subducting_pid))
                fig.savefig(ofile_path + ".png")
                print("Saved figure %s" % (ofile_path + ".png"))
                fig.savefig(ofile_path + ".pdf")
                print("Saved figure %s" % (ofile_path + ".pdf"))

        ax0.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        ofile_path = os.path.join(local_img_dir, "global_subduction_ori_t%.2fMa" % float(reconstruction_time))
        fig0.savefig(ofile_path + ".png")
        print("Saved figure %s" % (ofile_path + ".png"))
        fig0.savefig(ofile_path + ".pdf")
        print("Saved figure %s" % (ofile_path + ".pdf"))

        return color_dict

    def update_unique_pid_dict(self, resample_dataset, pid_dict={}):

        if resample_dataset:
            s_data = self.subduction_data_resampled
        else:
            s_data = self.subduction_data
            
        subducting_pids = s_data.subducting_pid.unique()
        for i, subducting_pid in enumerate(subducting_pids):
            if subducting_pid in pid_dict:
                if int(self.reconstruction_time) not in pid_dict[subducting_pid]:
                    pid_dict[subducting_pid].append(int(self.reconstruction_time))
            else:
                pid_dict[subducting_pid] = [int(self.reconstruction_time)]

        return pid_dict

    def update_region_dict(self, resample_dataset, region_dict={}):
        if resample_dataset:
            s_data = self.subduction_data_resampled
        else:
            s_data = self.subduction_data
        subducting_pids = s_data.subducting_pid.unique()
        for i, subducting_pid in enumerate(subducting_pids):
            one_subduction_data = s_data[s_data.subducting_pid==subducting_pid]
            new_region =  crop_region_by_data(one_subduction_data, 15.0)
            if int(subducting_pid) in region_dict:
                region_dict[int(subducting_pid)] = merge_region(region_dict[int(subducting_pid)], new_region)
            else:
                region_dict[int(subducting_pid)] = new_region
        return region_dict

    def save_results_resampled(self, inspect_all_slabs_resampled_plot_individual=False, only_one_pid=None, **kwargs):
        # Summary:
        # Render and save global and optional per-slab figures for the resampled subduction data.
        # Parameters:
        # - inspect_all_slabs_resampled_plot_individual: bool — if True, produce per-slab diagnostic panels
        #   with point indices (denser labels than the original plots) and trench IDs.
        # - only_one_pid: None or value - if value, skip all other subducting pids.
        # Returns:
        # - None (saves .png/.pdf figures into a resampled-specific subdirectory).
        # Preconditions:
        # - self.subduction_data_resampled exists from resample_subduction().
        # - self.age_grid_raster is available for background shading (if add_age_raster() was used).
        # Raises:
        # - GPLATE_PROCESS_WORKFLOW_ERROR if resample_subduction() was not called.

        my_assert(self.subduction_data_resampled is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"resample_subduction\" first.")

        color_dict = kwargs.get("color_dict", {})
        region_dict = kwargs.get("region_dict", {})
        
        local_img_dir = os.path.join(self.img_dir, "resampled_edge%.1f_section%.1f" % (self.arc_length_edge, self.arc_length_resample_section))
        if os.path.isdir(local_img_dir):
            rmtree(local_img_dir)
        os.mkdir(local_img_dir)

        subduction_data_resampled = self.subduction_data_resampled
        gPlotter = self.gPlotter
        age_grid_raster = self.age_grid_raster
        reconstruction_time = self.reconstruction_time

        subducting_pids = subduction_data_resampled["subducting_pid"].unique()

        # start figure
        # plot the subducting_pid in the globe
        fig0 = plt.figure(figsize=(10, 12), dpi=100, tight_layout=True)
        gs = gridspec.GridSpec(2, 1)

        gPlotter.set_region("default")

        ax0 = fig0.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
        gPlotter.plot_global_basics(ax0, age_grid_raster=age_grid_raster)
        color_dict = gPlotter.plot_subduction_pts(ax0, subduction_data_resampled, color_dict=color_dict)
        
        ax1 = fig0.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
        gPlotter.plot_global_basics(ax1, age_grid_raster=age_grid_raster, plot_boundaries=True)

        for i, subducting_pid in enumerate(subducting_pids):

            if only_one_pid is not None and int(subducting_pid) != int(only_one_pid):
                # if only one pid is given, skip all others
                continue

            one_subduction_data = subduction_data_resampled[subduction_data_resampled.subducting_pid==subducting_pid]

            # add marker to summary plot
            ax0.text(one_subduction_data["lon"].iloc[0], one_subduction_data["lat"].iloc[0], str(subducting_pid), transform=ccrs.PlateCarree(),
                fontsize=8,
                ha="left",   # horizontal alignment
                va="bottom"  # vertical alignment
            )

            # plot individual subduction zone
            if inspect_all_slabs_resampled_plot_individual:
                # plot individual subduction zone
                # 1. mark indices of points
                # 2. mark trench pid values
                # 3. also include makers
                try:
                    region = region_dict[int(subducting_pid)]
                except KeyError:
                    region = crop_region_by_data(one_subduction_data, 15.0)
                gPlotter.set_region(region) # set region to default, no need to filter

                fig = plt.figure(figsize=(20, 12), dpi=100)
                gs = gridspec.GridSpec(2, 2)
                
                ax = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster, plot_boundaries=True)
                ax.set_extent(region, crs=ccrs.PlateCarree())
            
                ax = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                sub_color_dict = gPlotter.plot_subduction_pts(ax, one_subduction_data, color_dict=color_dict[int(subducting_pid)])
                for j, (lon, lat) in enumerate(zip(one_subduction_data.lon, one_subduction_data.lat)):
                    if j % 2 != 0:
                        continue
                    ax.text(lon, lat, str(j), transform=ccrs.PlateCarree(),
                        fontsize=8,
                        ha="left",   # horizontal alignment
                        va="bottom"  # vertical alignment
                    )
                ax.set_extent(region, crs=ccrs.PlateCarree())
                
                # update color dictionary for specific subduction zone 
                color_dict[int(subducting_pid)] = sub_color_dict

                ax = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                _ = gPlotter.plot_subduction_pts(ax, one_subduction_data, "trench_pid", color_dict=color_dict[int(subducting_pid)])
                trench_pids = one_subduction_data["trench_pid"].unique()
                ax.set_extent(region, crs=ccrs.PlateCarree())

                ax = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree(central_longitude=gPlotter.get_central_longitude()))
                gPlotter.plot_global_basics(ax, age_grid_raster=age_grid_raster)
                _ = gPlotter.plot_subduction_pts(ax, one_subduction_data, "trench_pid", color_dict=color_dict[int(subducting_pid)])
                trench_pids = one_subduction_data["trench_pid"].unique()
                ax.set_extent(region, crs=ccrs.PlateCarree())
                for j, trench_pid in enumerate(trench_pids):
                    sub_subduction_data = one_subduction_data[one_subduction_data.trench_pid==trench_pid]
                    lon, lat = sub_subduction_data.lon.iloc[0], sub_subduction_data.lat.iloc[0]
                    ax.text(lon, lat, str(trench_pid), transform=ccrs.PlateCarree(),
                        fontsize=8,
                        ha="left",   # horizontal alignment
                        va="bottom"  # vertical alignment
                    )
            
                # save figure of individual subduction zone
                ofile_path = os.path.join(local_img_dir, "global_subduction_resampled_t%.2fMa_pid%06d" % (float(reconstruction_time), subducting_pid))
                fig.savefig(ofile_path + ".png")
                print("Saved figure %s" % (ofile_path + ".png"))
                fig.savefig(ofile_path + ".pdf")
                print("Saved figure %s" % (ofile_path + ".pdf"))

        ax0.set_extent((-180, 180, -90, 90), crs=ccrs.PlateCarree())
        ofile_path = os.path.join(local_img_dir, "global_subduction_resampled_t%.2fMa" % float(reconstruction_time))
        fig0.savefig(ofile_path + ".png")
        print("Saved figure %s" % (ofile_path + ".png"))
        fig0.savefig(ofile_path + ".pdf")
        print("Saved figure %s" % (ofile_path + ".pdf"))

        return color_dict

    def plot_age_combined(self, resample_dataset, plot_options=None):
        # Summary:
        # Generate and save age-vs-metrics figures for either the original or resampled subduction dataset.
        # For each selected subducting plate ID, saves an individual figure, and also saves an aggregate figure.
        # Parameters:
        # - resample_dataset: bool — if True, use the resampled table (self.subduction_data_resampled) and
        #   write to a directory labeled with resampling parameters; otherwise, use the original table.
        # - plot_options: Optional[List[Tuple[int, Dict[str, Any]]]] — per-subducting_pid plot settings.
        #   If None, a default configuration is constructed assigning marker shapes and colors.
        # Returns:
        # - None (saves .png and .pdf figures into a time- or parameter-stamped directory).
        # Preconditions:
        # - If resample_dataset is True, resample_subduction() must have been called (checks enforced).
        # - If resample_dataset is False, reconstruct() must have been called (checks enforced).
        # Raises:
        # - GPLATE_PROCESS_WORKFLOW_ERROR if required upstream step was not executed.
        # - File/IO errors may propagate on directory/figure writes.
        
        if resample_dataset:
            local_img_dir = os.path.join(self.img_dir, "age_combined_resampled_edge%.1f_section%.1f" %\
                                        (self.arc_length_edge, self.arc_length_resample_section))
            my_assert(self.subduction_data_resampled is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"resample_subduction\" first.")
            s_data = self.subduction_data_resampled
        else:
            local_img_dir = os.path.join(self.img_dir, "age_combined_t%.2fMa" % float(self.reconstruction_time))
            my_assert(self.subduction_data is not None, GPLATE_PROCESS_WORKFLOW_ERROR, "Need to call function \"reconstruct\" first.")
            s_data = self.subduction_data

        if os.path.isdir(local_img_dir):
            rmtree(local_img_dir)
        os.mkdir(local_img_dir)

        if plot_options is None:
            markers = ["o", '*', "d", "x", "v", "s"]
            n_color = 10
            plot_options = []
            subducting_pids = s_data.subducting_pid.unique()
            for i, subducting_pid in enumerate(subducting_pids):
                plot_options.append((int(subducting_pid), {"marker": markers[i//n_color],  "markerfacecolor": default_colors[i%10], "name": str(int(subducting_pid))}))

        for sub_plot_options in plot_options:
            print("sub_plot_options: ", sub_plot_options)
            _name = sub_plot_options[1]["name"]
            fig, axes = plot_age_combined(s_data, [sub_plot_options], plot_index=True)
            file_path = os.path.join(local_img_dir, "age_combined_%s" % _name)
            fig.savefig(file_path + ".png")
            print("Saved figure %s" % (file_path + ".png"))
            fig.savefig(file_path + ".pdf")
            print("Saved figure %s" % (file_path + ".pdf"))
            
        
        fig, _ = plot_age_combined(s_data, plot_options)
        file_path = os.path.join(local_img_dir, "age_combined")
        fig.savefig(file_path + ".png")
        print("Saved figure %s" % (file_path + ".png"))
        fig.savefig(file_path + ".pdf")
        print("Saved figure %s" % (file_path + ".pdf"))




def mask_by_pids(subduction_data, subducting_pid_p, trench_pid_p=None):
    """
    Generates a mask based on proximity to subducting and trench process IDs.
    
    Parameters:
        subduction_data (object): Object containing subducting_pid and trench_pid attributes.
        subducting_pid_p (float): Target value for the subducting process ID to match.
        trench_pid_p (float, optional): Target value for the trench process ID to match.
        
    Returns:
        mask (bool array): Boolean array where True values indicate proximity to specified process IDs.
    """
    
    # Create initial mask based on subducting process ID within a tolerance of 0.1.
    mask1 = (abs(subduction_data.subducting_pid - subducting_pid_p) < 0.1)
    
    # If trench process ID is specified, create an additional mask based on its proximity.
    if trench_pid_p is not None:
        mask2 = (abs(subduction_data.trench_pid - trench_pid_p) < 0.1)
    
    # Combine masks using logical AND operation.
    mask = mask1 & mask2

    return mask


def haversine(lat1, lon1, lat2, lon2, radius=6371e3):
    """
    Calculates the great-circle distance between two points on the Earth using the Haversine formula.
    
    Parameters:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.
        radius (float, optional): Radius of the sphere; default is Earth's radius in meters (6371e3).
        
    Returns:
        distance (float): The distance between the two points in the specified radius unit.
    """
    
    # Convert latitude and longitude values from degrees to radians for calculation.
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Apply the Haversine formula to calculate the angular distance between points.
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Calculate the physical distance by scaling the angular distance by the specified radius.
    distance = radius * c
    return distance


def read_subduction_reconstruction_data(infile):
    """
    Reads a file of plate reconstruction data and extracts subduction zone information.
    
    Parameters:
        infile (str): The file path to the input data file.
    
    Returns:
        dict: A dictionary containing the following keys:
            - 'n_trench' (int): The number of subduction zones found in the file.
            - 'trench_data' (list): A list of arrays containing coordinates for each subduction zone.
            - 'trench_names' (list): A list of names for each subduction zone.
            - 'trench_pids' (list): A list of plate IDs associated with each subduction zone.
            - 'trench_begin_times' (list): A list of the beginning times for each subduction zone.
            - 'trench_end_times' (list): A list of the end times for each subduction zone.
    
    Implementation:
        - Opens the input file and iterates through each line.
        - Uses flags `sbd_begin` and `sbd_end` to track whether the current section is a subduction zone.
        - Extracts coordinates and header information, such as trench names, plate IDs, and time intervals.
        - Appends the extracted data into respective lists and returns them in a dictionary.
    """
    assert(os.path.isfile(infile))

    trench_data = []
    trench_names = []
    trench_pids = []
    trench_begin_times = []
    trench_end_times = []

    i = 0
    temp_l = []  # Stores line indices of each subduction zone section
    temp_d = []  # Temporarily holds coordinates of the current subduction zone
    n_trench = 0  # Counts the number of subduction zones
    sbd_begin = False  # Flag indicating the start of a subduction zone section
    sbd_end = False  # Flag indicating the end of a subduction zone section
    read = True  # Flag for continuing to read the file

    with open(infile, 'r') as fin:
        line = fin.readline()
        i += 1
        while line:
            read = True  # Default to continue reading each loop
            
            # Check if the end of a subduction zone section is reached
            if sbd_begin and re.match('^>', line):
                sbd_end = True

            # Handle the different scenarios based on the flags
            if sbd_begin and (not sbd_end):
                # Reading subduction zone data
                temp_data = line.split()
                temp_data = [float(x) for x in temp_data]
                temp_d.append(temp_data)
            elif sbd_begin and sbd_end:
                # Reached the end of a section, store the data and reset flags
                trench_data.append(temp_d)
                sbd_begin = False
                sbd_end = False
                read = False
            elif re.match('^>SubductionZone', line):
                # Found the start of a new subduction zone section
                temp_l.append(i)
                sbd_begin = True
                temp_d = []
                # Continue reading the headers of the section
                while line and re.match('^>', line):
                    line = fin.readline()
                    i += 1
                    if re.match('^> name', line):
                        trench_names.append(remove_substrings(line, ["> name ", '\n']))
                    elif re.match('> reconstructionPlateId', line):
                        trench_pids.append(int(remove_substrings(line, ["> reconstructionPlateId ", '\n'])))
                    elif re.match('> validTime TimePeriod <begin> TimeInstant <timePosition>', line):
                        temp0 = remove_substrings(line, ["> validTime TimePeriod <begin> TimeInstant <timePosition>", '</timePosition>.*\n'])
                        trench_begin_times.append(float(temp0))
                        temp1 = remove_substrings(line, ['^.*<end> TimeInstant <timePosition>', '</timePosition>.*\n'])
                        trench_end_times.append(float(temp1) if type(temp1) == float else 0.0)
                read = False
            
            if read:
                line = fin.readline()
                i += 1

    i -= 1  # Adjust for the last unsuccessful read
    n_trench = len(trench_data)

    outputs = {
        "n_trench": n_trench, 
        "trench_data": trench_data, 
        "trench_names": trench_names,
        "trench_pids": trench_pids, 
        "trench_begin_times": trench_begin_times, 
        "trench_end_times": trench_end_times
    }

    return outputs


def LookupNameByPid(trench_pids, trench_names, pid):
    """
    Looks up the name of a trench using its plate ID.

    Parameters:
        trench_pids (list): A list of plate IDs corresponding to subduction zones.
        trench_names (list): A list of names corresponding to the trench IDs.
        pid (int): The plate ID for which the trench name is being looked up.

    Returns:
        str: The name of the trench corresponding to the given plate ID. Returns an empty
             string if the plate ID is not found.
    
    Implementation:
        - Asserts that the `pid` provided is of type `int`.
        - Attempts to find the index of `pid` in the `trench_pids` list.
        - If the `pid` is found, retrieves the name from `trench_names` using the index.
        - If the `pid` is not found (raises a `ValueError`), returns an empty string.
    """
    _name = ""
    assert(type(pid) == int)
    try:
        _index = trench_pids.index(pid)
    except ValueError:
        _name = ""
    else:
        _name = trench_names[_index]
    return _name

class GPLOTTER():
    """
    A class to interact with gplate data and generate basic plots
    Attributes:
        plate_model: plate model used in reconstruction
        model: reconstruction model
        reconstruction_time (float): The geological time at which the reconstruction is plotted.
        region (list): lon_min, lon_max, lat_min, lat_max to plot
    """
    def __init__(self, plate_model, model):

        # set reconstruction model
        self.plate_model = plate_model
        self.model  = model

        # Initialize the plotting object for visualizing topologies
        # The layers used for plotting include coastlines, continental polygons, and COBs (Continental Ocean Boundaries)
        self.gplot = gplately.plot.PlotTopologies(
            self.model, 
            self.plate_model.get_layer('Coastlines'), 
            self.plate_model.get_layer('ContinentalPolygons'), 
            self.plate_model.get_layer('COBs')
        )

        # Set default values
        self.reconstruction_time = 0.0
        self.gplot.time = 0.0
        self.region = (-180, 180, -90, 90)

    def set_time(self, reconstruction_time):
        '''
        Set reconstruction time
        Inputs:
            reconstruction_time (float): The geological time at which the reconstruction is plotted.
        '''
        self.reconstruction_time = reconstruction_time
        self.gplot.time = self.reconstruction_time

    def set_region(self, region):
        '''
        Set inputs to plot
        Inputs:
            region (list): lon_min, lon_max, lat_min, lat_max to plot
                or (string) = "default": reset to default values
        '''
        if isinstance(region, list):
            assert(len(region) == 4)
            self.region = region
        elif isinstance(region, str):
            assert(region == "default")
            self.region = (-180, 180, -90, 90)
        else:
            raise TypeError("region must by list or \"default\"")

    def get_central_longitude(self):
        '''
        Get the central longitude for plotting
        '''
        if self.region[0] == -180 and self.region[1] == 180:
            central_longitude = 180
        else:
            central_longitude = (self.region[0] + self.region[1])/2.0
        return central_longitude

    def plot_global_basics(self, ax, age_grid_raster=None, plot_boundaries=False):
        """
        Plots basic global geological features on a given axis, including coastlines and an age grid.

        Parameters:
            ax (matplotlib.axes._axes.Axes): The axis on which to plot the global features.
            age_grid_raster: A raster object containing age data, typically used for visualizing geological ages.
            plot_bounareis: add plot of tranform and subduction boundaries
        
        Implementation:
            - Configures global gridlines on the plot with specific color, linestyle, and locations.
            - Sets the map extent to global.
            - Uses the `gplot` object to plot coastlines at the given reconstruction time.
            - Plots the age grid data on the map using a specified colormap and transparency level.
            - Adds a color bar to the plot to represent ages, with a labeled color bar axis.
        """
        # Configure global gridlines with specified color and linestyle
        gl = ax.gridlines(color='0.7', linestyle='--', xlocs=np.arange(self.region[0], self.region[1], 15),\
                        ylocs=np.arange(self.region[2], self.region[3], 15))
        gl.left_labels = True

        # Set the map extent to global
        ax.set_global()

        # Set the reconstruction time for the gplot object and plot coastlines in grey
        self.gplot.plot_coastlines(ax, color='grey')

        if plot_boundaries:
            self.gplot.plot_ridges(ax, color='red')
            self.gplot.plot_transforms(ax, color='red')
            self.gplot.plot_trenches(ax, color='k')
            self.gplot.plot_subduction_teeth(ax, color='k')

        # Plot the age grid on the map using a colormap from yellow to blue
        if age_grid_raster is not None:
            im_age = self.gplot.plot_grid(ax, age_grid_raster.data, cmap='YlGnBu', vmin=0, vmax=200, alpha=0.8)

            # Add a color bar for the age grid with a label
            cbar_age = plt.colorbar(im_age)
            cbar_age.ax.get_yaxis().labelpad = 15
            cbar_age.ax.set_ylabel("Age (Ma)", rotation=90)

        return ax


    def plot_subduction_pts(self, ax, subduction_data, plot_by="subducting_pid", **kwargs):
        '''
        Plot points in a subduction dataset
        Parameters:
            ax (matplotlib.axes._axes.Axes): The axis on which to plot the global features.
            subduction_data: subduction dataset
            plot_by (str): separte individual subduction by this option
            kwargs:
                color0 - use as color to plot (otherwise assign individual colors)
        
        Return:
            color_dict (dict): color map for all the trenches.
        '''
        color_dict  = kwargs.get("color_dict", {})
        plot_trench_normal = kwargs.get("plot_trench_normal", False)
        plot_conv = kwargs.get("plot_conv", False)
        stepping = kwargs.get("stepping", 1)

        import cartopy.crs as ccrs
        
        default_colors = [color['color'] for color in plt.rcParams['axes.prop_cycle']]
        
        pids = subduction_data[plot_by].unique()

        # plot data by trench_pid
        for i, pid in enumerate(pids):
            mask = subduction_data[plot_by] == pid
            one_subduction_data = subduction_data[mask]

            try:
                _color = color_dict[int(pid)]["color"]
            except KeyError:
                _color = default_colors[i%4]
                color_dict[int(pid)] = {"color": _color}

            ax.scatter(one_subduction_data.lon, one_subduction_data.lat, marker=".", s=30, c=_color, transform=ccrs.PlateCarree(), label=pid)

            if plot_trench_normal:
                # Convert (angle, length) -> PlateCarree components (in "degree-like" units)
                lon, lat, trench_azimuth_angle =\
                      one_subduction_data.lon.to_numpy(), one_subduction_data.lat.to_numpy(),\
                            one_subduction_data.trench_azimuth_angle.to_numpy()
                dx = 5.0 * np.sin(trench_azimuth_angle)   # +east
                dy = 5.0 * np.cos(trench_azimuth_angle)   # +north

                # Optional: scale factor so arrows look reasonable on a degree grid
                deg_per_unit = 1.0       # tweak this to taste (bigger = longer arrows)
                u = deg_per_unit * dx
                v = deg_per_unit * dy

                idx = slice(0, None, stepping) 
                Q = ax.quiver(
                    lon[idx], lat[idx], u[idx], v[idx],
                    transform=ccrs.PlateCarree(),   # u,v are in the same PlateCarree coordinate system
                    angles='xy', scale_units='xy',  # interpret in data coords, not data-dependent scaling
                    scale=1.0,                      # with scale_units='xy', scale=1 uses u,v as-is
                    width=0.003, color='k', alpha=0.9# regrid to declutter
                )

            if plot_conv:
                # Convert (angle, length) -> PlateCarree components (in "degree-like" units)
                lon, lat, conv_rate, conv_angle, trench_azimuth_angle =\
                      one_subduction_data.lon.to_numpy(), one_subduction_data.lat.to_numpy(),\
                          one_subduction_data.conv_rate.to_numpy(), one_subduction_data.conv_angle.to_numpy(),\
                            one_subduction_data.trench_azimuth_angle.to_numpy()
                conv_angle_map_north = conv_angle + trench_azimuth_angle
                dx = conv_rate * np.sin(conv_angle_map_north)   # +east
                dy = conv_rate * np.cos(conv_angle_map_north)   # +north

                # Optional: scale factor so arrows look reasonable on a degree grid
                deg_per_unit = 1.0       # tweak this to taste (bigger = longer arrows)
                u = deg_per_unit * dx
                v = deg_per_unit * dy

                idx = slice(0, None, stepping) 
                Q = ax.quiver(
                    lon[idx], lat[idx], u[idx], v[idx],
                    transform=ccrs.PlateCarree(),   # u,v are in the same PlateCarree coordinate system
                    angles='xy', scale_units='xy',  # interpret in data coords, not data-dependent scaling
                    scale=1.0,                      # with scale_units='xy', scale=1 uses u,v as-is
                    width=0.003, color='k', alpha=0.9# regrid to declutter
                )

        return color_dict


def resample_subduction(one_subduction_data, arc_length_edge, arc_length_resample_section, **kwargs):
    """
    Resamples data points from a dense subduction zone at specified intervals along its arc length.
    This helps simplify and extract key properties of the subduction zone for plotting and analysis.

    Parameters:
        one_subduction_data (pd.DataFrame): A pandas DataFrame containing data for a single subduction zone.
        arc_length_edge (float): The arc length distance from the edges where no resampling is performed.
        arc_length_resample_section (float): The interval at which the arc length is resampled.
        all_columns (list): A list of column names for the output DataFrame.

    Returns:
        tuple: 
            - pd.DataFrame: A DataFrame of the resampled subduction zone data.
            - str: A log of the resampled points' coordinates for debugging or output purposes.

    Implementation:
        - Initializes variables, including indentation and a log for output content.
        - Computes cumulative arc lengths for all points in the original data.
        - Determines resampling points centered at the midpoint of the arc length and propagates outward.
        - Resamples properties by linear interpolation between points, including special handling of longitude and latitude.
        - Collects and logs each resampled point's coordinates, and returns the resampled DataFrame and the log.
    """
    from hamageolib.utils.nump_utilities import interval_with_fraction

    # Initialize variables, including default indentation for output
    all_columns = one_subduction_data.columns

    # Upcast to float 64
    one_subduction_data = one_subduction_data.astype(np.float64)

    # Determine columns to pick for nearest neighbors or to interpolate
    pick_columns = ["trench_pid"]
    interp_columns = [col for col in all_columns if col not in pick_columns]
    
    # Compute cumulative arc lengths
    arc_lengths = one_subduction_data.arc_length.to_numpy()
    arc_length_sums, arc_length_total = compute_sum_of_arc_lengths(arc_lengths)

    # Positions of the resampled points
    # Assign offsets from mid to the edge
    positions = resample_positions(arc_length_total, arc_length_edge, arc_length_resample_section)

    # Get the nearby indexes, fractions of resampled points
    idx0, idx1, frac = interval_with_fraction(arc_length_sums, positions)
    pick = np.where(frac < 0.5, idx0, idx1)

    # Initiate resampled data, fill with nan value and have rows as the size of positions
    one_subduction_data_resampled = one_subduction_data.iloc[:0].reindex(range(positions.size)).reset_index(drop=True)

    # Pick nearest neighbor for pick_columns
    target_data = one_subduction_data.iloc[pick]
    one_subduction_data_resampled.loc[:, pick_columns] = target_data[pick_columns].reset_index(drop=True).to_numpy()

    # Interpolate for interp_columns
    left, right = one_subduction_data.iloc[idx0][interp_columns].reset_index(drop=True), \
        one_subduction_data.iloc[idx1][interp_columns].reset_index(drop=True)
    
    one_subduction_data_resampled.loc[:, interp_columns] = left.mul(1-frac, axis=0) + right.mul(frac, axis=0)

    # assign the resampled arc length
    one_subduction_data_resampled.arc_length = arc_length_resample_section
    
    # make sure the subducting pid are unchanged
    one_subduction_data_resampled.subducting_pid = one_subduction_data.subducting_pid.iloc[0]

    return one_subduction_data_resampled


def compute_sum_of_arc_lengths(arc_lengths):

    assert(isinstance(arc_lengths, np.ndarray) and arc_lengths.ndim == 1)
    arc_length_sums = np.zeros(arc_lengths.size)
    arc_length_sums[0] = arc_lengths[0] / 2.0
    for i in range(1, arc_lengths.size):
        arc_length_sums[i] = arc_length_sums[i - 1] + (arc_lengths[i-1] + arc_lengths[i])/2.0
    arc_sum = arc_length_sums[-1] + arc_lengths[-1]/2.0

    return arc_length_sums, arc_sum


def resample_positions(arc_length_total, arc_length_edge, arc_length_resample_section):
    """
    Return symmetric resample positions starting at the midpoint and stepping
    outward by `arc_length_resample_section`, stopping before the edge strips.
    Only one midpoint is included.
    """
    if arc_length_resample_section <= 0:
        raise ValueError("arc_length_resample_section must be > 0")

    mid = arc_length_total / 2.0
    d_max = mid - arc_length_edge  # maximum distance allowed from mid

    if d_max < 0.0:
        raise ValueError("arc_length_edge is larger than half the total arc length")

    # Start offsets at one step to avoid duplicating the midpoint
    pos_offsets = np.arange(arc_length_resample_section, d_max + 1e-12, arc_length_resample_section)

    # Build positions: one midpoint + symmetric shells
    left = mid - pos_offsets
    right = mid + pos_offsets
    positions = np.sort(np.concatenate(([mid], left, right)))
    return positions



def FixTrenchAgeLocal(subduction_data, age_grid_raster, i_p, theta):
    """
    Fixes invalid age values in a subduction data object using age interpolation
    from nearby points along a specified direction.

    Parameters:
        subduction_data (pd.DataFrame): The dataset containing subduction zone data.
        age_grid_raster: A raster object containing age data, typically used for visualizing geological ages.
        i_p (int): The index of the subduction data point to be fixed.
        theta (float): The direction (in degrees) to search for new data points for interpolation.

    Returns:
        float: The newly interpolated age value. If interpolation fails, returns NaN.
    
    Implementation:
        - Defines a set of distances `ds` to search for new points around the specified index.
        - Iterates over pairs of distances to generate two nearby points in the specified direction.
        - Uses `map_point_by_distance` to calculate the longitude and latitude of the new points.
        - If both ages are valid, interpolates between them to determine the new age.
        - Updates the `subduction_data` object with the interpolated age and records the fixed location.
        - If interpolation is not successful, sets the age to NaN.
    """
    ds = [12.5e3, 25e3, 50e3, 75e3, 100e3, 150e3, 200e3, 300e3, 400e3]
    new_age = np.nan

    # Iterate over the distances to generate two points for age interpolation
    for j in range(len(ds) - 1):
        # Generate two local points at distances `ds[j]` and `ds[j+1]` in the direction `theta`
        subduction_data_local0 = pd.DataFrame([subduction_data.iloc[i_p]])
        subduction_data_local1 = pd.DataFrame([subduction_data.iloc[i_p]])
        
        subduction_data_local0.loc[:, "lon"], subduction_data_local0.loc[:, "lat"] = map_point_by_distance(
            subduction_data.iloc[i_p].lon, subduction_data.iloc[i_p].lat, theta, ds[j]
        )
        subduction_data_local1.loc[:, "lon"], subduction_data_local1.loc[:, "lat"] = map_point_by_distance(
            subduction_data.iloc[i_p].lon, subduction_data.iloc[i_p].lat, theta, ds[j + 1]
        )
        
        # Interpolate ages at the two new points
        new_age0 = age_grid_raster.interpolate(subduction_data_local0.lon, subduction_data_local0.lat, method="nearest")
        new_age1 = age_grid_raster.interpolate(subduction_data_local1.lon, subduction_data_local1.lat, method="nearest")
        
        # If both ages are valid, perform interpolation and update the subduction data
        if (not np.isnan(new_age0)) and (not np.isnan(new_age1)):
            new_age = (new_age0 * ds[j + 1] - new_age1 * ds[j]) / (ds[j + 1] - ds[j])
            subduction_data.loc[i_p, "age"] = new_age
            # debug
            subduction_data.loc[i_p, "lon_fix"] = subduction_data_local1.lon.iloc[0]  # Records the further point
            subduction_data.loc[i_p, "lat_fix"] = subduction_data_local0.lat.iloc[0]  # Records the closer point
            break
        else:
            subduction_data.loc[i_p, "age"] = np.nan  # Mark as NaN if interpolation fails

    return new_age


def FixTrenchAge(subduction_data, age_grid_raster, **kwargs):
    '''
    Fix the trench ages in subduction_data
    Inputs:
        subduction_data: pandas object, subduction dataset
        age_grid_raster: A raster object containing age data, typically used for visualizing geological ages.
    '''
    # automatically fix the invalid ages 
    for i in range(len(subduction_data)):
        fix_age_polarity = subduction_data.fix_age_polarity[i]
        if not np.isnan(fix_age_polarity):
            # fix with existing polarity
            # 0 and 1: on different side of the trench
            # 2: manually assign values of longitude and latitude
            if (fix_age_polarity == 0): 
                new_age = FixTrenchAgeLocal(subduction_data, age_grid_raster, i, subduction_data.trench_azimuth_angle[i] + 180.0)
            elif (fix_age_polarity == 1): 
                new_age = FixTrenchAgeLocal(subduction_data, age_grid_raster, i, subduction_data.trench_azimuth_angle[i])
            elif (fix_age_polarity == 2):
                subduction_data_local0 = pd.DataFrame([subduction_data.iloc[i]])
                subduction_data_local0.loc[:, "lon"], subduction_data_local0.loc[:, "lat"] = subduction_data.iloc[i].lon_fix, subduction_data.iloc[i].lat_fix
                new_age = age_grid_raster.interpolate(subduction_data_local0.lon, subduction_data_local0.lat, method="nearest")
                subduction_data.loc[i, 'age'] = new_age
                pass
            else:
                raise NotImplementedError
        else:
            # figure out a possible polarity
            new_age = FixTrenchAgeLocal(subduction_data, age_grid_raster, i, subduction_data.trench_azimuth_angle[i] + 180.0)
            if np.isnan(new_age):
                # next, try the other direction
                new_age = FixTrenchAgeLocal(subduction_data, age_grid_raster, i, subduction_data.trench_azimuth_angle[i])
                if not np.isnan(new_age):
                    subduction_data.loc[i, "fix_age_polarity"] = 1
            else:
                subduction_data.loc[i, "fix_age_polarity"] = 0


def MaskBySubductionTrenchIds(subduction_data, subducting_pid, trench_pid, i_p):
    """
    Generates a combined mask for subduction data based on user selection or specific 
    subducting and trench IDs.
    
    Parameters:
        subduction_data (pd.DataFrame): The DataFrame containing subduction data to be filtered.
        subducting_pid (int or None): The subducting plate ID to match. If None, all IDs are included.
        trench_pid (int or None): The trench plate ID to match. If None, all IDs are included.
        i_p (list or None): List of indices selected by the user. If not None, these indices are used.
    
    Returns:
        np.ndarray: A boolean mask combining the specified conditions for filtering the data.
    
    Implementation:
        - If `i_p` is provided, create `mask1` to select only those indices.
        - If `subducting_pid` is provided, create `mask1` to select rows matching the `subducting_pid`.
        - If neither is provided, `mask1` includes all rows.
        - If `trench_pid` is provided, create `mask2` to select rows matching the `trench_pid`.
        - If `trench_pid` is not provided, `mask2` includes all rows.
        - The final mask is the logical AND of `mask1` and `mask2`.
    """
    if i_p is not None:
        mask1 = np.zeros(len(subduction_data), dtype=bool)
        mask1[i_p] = 1
    elif subducting_pid is not None:
        # Generate mask1 based on the provided subducting plate ID
        mask1 = subduction_data.subducting_pid == subducting_pid
    else:
        mask1 = np.ones(len(subduction_data), dtype=bool)

    if trench_pid is not None:
        # Generate mask2 based on the provided trench plate ID
        mask2 = subduction_data.trench_pid == trench_pid
    else:
        mask2 = np.ones(len(subduction_data), dtype=bool)

    return (mask1 & mask2)


def crop_region_by_data(s_data, interval):
    """
    Determine the bounding region of a dataset (longitude–latitude) with optional longitude wrapping.

    Parameters
    ----------
    s_data : DataFrame-like
        A dataset containing at least the attributes `.lon` and `.lat`. These must be 
        accessible as array-like objects (e.g., pandas/xarray with `.to_numpy()`).
    interval : float
        The step interval (e.g., in degrees) used to round the min/max boundaries.

    Returns
    -------
    region : list of float
        A list of [lon_min, lon_max, lat_min, lat_max] defining the cropped bounding box.
        Two bounding regions are tested:
            - region0: using raw longitudes (possibly spanning negative and positive values).
            - region1: longitudes are wrapped into [0, 360).
        The function selects the region with the smaller longitude span.
    """
    if isinstance(s_data, pd.DataFrame):
        my_assert(len(s_data) > 0, ValueError, "s_data cannot be vacant")
        lon_np = s_data.lon.to_numpy()
        lat_np = s_data.lat.to_numpy()
    elif isinstance(s_data, np.ndarray):
        my_assert(s_data.size > 0, ValueError, "s_data cannot be vacant")
        my_assert(s_data.shape[1] == 2, ValueError, "s_data should contain two columns \"lon\" and \"lat\"")
        lon_np = s_data[:, 0]
        lat_np = s_data[:, 1]

    # Extract raw longitude and latitude ranges
    lon_min_raw, lon_max_raw = np.min(lon_np), np.max(lon_np)
    lat_min, lat_max = np.min(lat_np), np.max(lat_np)

    # Extract longitude ranges of > 0 values
    mask = lon_np < 0
    lon_np[mask] += 360
    lon_min_1, lon_max_1 = np.min(lon_np), np.max(lon_np)

    # Determine which range of longitude is smaller
    tolerance = 1e-6
    if lon_max_raw - lon_min_raw < lon_max_1 - lon_min_1 + tolerance:
        lon_min, lon_max = lon_min_raw, lon_max_raw
    else:
        lon_min, lon_max = lon_min_1, lon_max_1

    # Round to nearest interval boundaries
    lon_min = np.floor(lon_min/interval) * interval
    lon_max = np.ceil(lon_max/interval) * interval
    lat_min = np.floor(lat_min/interval) * interval
    lat_max = np.ceil(lat_max/interval) * interval
    region = [lon_min, lon_max, lat_min, lat_max]

    return region

def merge_region(region0, region1):
    """
    Merge two lon–lat rectangular regions into a single bounding region.

    Parameters
    ----------
    region0 : list[float]
        [lon_min0, lon_max0, lat_min0, lat_max0].
    region1 : list[float]
        [lon_min1, lon_max1, lat_min1, lat_max1].

    Returns
    -------
    list[float]
        [lon_min, lon_max, lat_min, lat_max] describing a region that
        encloses points sampled from both input rectangles. The exact
        rounding/extension behavior is delegated to `crop_region_by_data`
        (with interval = 15.0).
    """
    # parse the two ranges
    lon_np0 = np.linspace(region0[0], region0[1], 99)
    lat_np0 = np.linspace(region0[2], region0[3], 101)
    
    lon_np1 = np.linspace(region1[0], region1[1], 99)
    lat_np1 = np.linspace(region1[2], region1[3], 101)

    Lon0, Lat0 = np.meshgrid(lon_np0, lat_np0)
    Lon1, Lat1 = np.meshgrid(lon_np1, lat_np1)
    mesh_np0  = np.column_stack([Lon0.ravel(), Lat0.ravel()])
    mesh_np1  = np.column_stack([Lon1.ravel(), Lat1.ravel()])

    mesh_np = np.concatenate([mesh_np0, mesh_np1], axis=0)

    region = crop_region_by_data(mesh_np, 15.0)

    return region


def mask_data_by_region(s_data, region):
    """
    Build a boolean mask selecting rows of `s_data` inside a lon–lat box.

    Parameters
    ----------
    s_data : pandas.DataFrame
        Must contain columns `lon` and `lat` (in degrees). Longitudes are
        assumed on a continuous circle where -180 and 180 are adjacent.
    region : list[float]
        [lon_a, lon_b, lat_min, lat_max]. If lon_a < lon_b, the mask
        selects lon in (lon_a, lon_b); otherwise it wraps across the
        dateline and selects lon in (lon_a, 360] ∪ (-∞, lon_b), effectively
        handling cases like [170, -170] for a small window straddling 180°.

    Returns
    -------
    pandas.Series (bool)
        True for rows inside the open intervals (lon, lat).
    """

    my_assert(isinstance(region, list) and len(region)== 4, ValueError, "region must be a list of length 4")

    if region[0] < region[1]:
        # in case a smaller value of longitude is given first
        mask_lon = (region[0] < s_data.lon) & (s_data.lon < region[1])
    else:
        # in case a larger value of longitude is given first
        # Note -180  is connected to 180
        mask_lon = (region[0] < s_data.lon) | (s_data.lon < region[1])

    mask_lat = (region[2] < s_data.lat) & (s_data.lat < region[3])

    mask = mask_lon & mask_lat

    return mask


def parse_subducting_trench_option(s_data, options, parse_by="subducting_pid"):
    """
    Extract subset(s) of s_data based on subducting_pid(s).

    Parameters
    ----------
    s_data : pd.DataFrame
        Input dataset with a 'subducting_pid' column.
    options : int, list, or tuple
        - int: return rows matching a single subducting_pid
        - list/tuple: return rows matching any of the given subducting_pid values

    Returns
    -------
    pd.DataFrame
        Subset of s_data with matching subducting_pid values.
    """
    from collections.abc import Sequence
    if options is None:
        # Return as it is if no options given
        return s_data
    elif isinstance(options, int):
        # A single int value - this represents a subduction / trench
        return s_data[s_data[parse_by] == options]
    elif isinstance(options, Sequence) or isinstance(options, np.ndarray) and not isinstance(options, str):
        # A sequence (list or tuple) - multiple subduction / trenches
        return s_data[s_data[parse_by].isin(options)]
    elif isinstance(options, dict):
        # Dictionary - subduction and specific trenches
        all_data = [pd.DataFrame(columns=s_data.columns)]
        for key, value in options.items():
            foo_data = parse_subducting_trench_option(s_data[s_data.subducting_pid == key], value, "trench_pid")
            if len(foo_data) == 0:
                continue
            all_data.append(foo_data)
        return pd.concat(all_data, ignore_index=True)

    else:
        raise TypeError("options must be int, list, or tuple")


def plot_age_combined(s_data, plot_options, **kwargs):
    '''
    Plot combined plots of vs age resuls
    Inputs:
        s_data:
            a pandas object of subduction zone observations
        plot_options:
            options of plots
    '''
    from matplotlib import gridspec
    from matplotlib import rcdefaults
    from matplotlib.ticker import MultipleLocator
    from hamageolib.research.haoyuan_3d_subduction.gplately_utilities import parse_subducting_trench_option
    import hamageolib.utils.plot_helper as plot_helper

    # Additional options

    plot_index = kwargs.get("plot_index", False)
    plot_index_stepping = kwargs.get("plot_index_stepping", 1)

    # Example usage
    # Rule of thumbs:
    # 1. Set the limit to something like 5.0, 10.0 or 50.0, 100.0 
    # 2. Set five major ticks for each axis
    scaling_factor = 1.0  # scale factor of plot
    font_scaling_multiplier = 2.0 # extra scaling multiplier for font
    legend_font_scaling_multiplier = 0.75
    line_width_scaling_multiplier = 2.0 # extra scaling multiplier for lines
    age_lim = (0.0, 200.0)
    Vtr_lim = (-12.0, 12.0)
    age_tick_interval = 50.0   # tick interval along x
    Vtr_tick_interval = 5.0  # tick interval along y
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
    
    fig = plt.figure(figsize=(15*scaling_factor, 5*scaling_factor), tight_layout=True)
    gs = gridspec.GridSpec(1, 4)

    ax0 = fig.add_subplot(gs[0, 0:3])

    patches = []
    for options in plot_options:
        assert(len(options) == 2)
        sp_option, plot_option = options[0], options[1]
        one_subduction_data = parse_subducting_trench_option(s_data, sp_option)

        if "name" in plot_option:
            # a name is assigned to subduction, go a head to plot
            my_assert("marker" in plot_option, ValueError, "marker is missing in plot option (pids = " + str(sp_option) + " )")
            my_assert("markerfacecolor" in plot_option, ValueError, "markerfacecolor is missing in plot option (pids = " + str(sp_option) + " )")

            _patch = ax0.plot(one_subduction_data.age, one_subduction_data.trench_velocity,\
                marker=plot_option["marker"], markerfacecolor=plot_option["markerfacecolor"],\
                markeredgecolor='black', markersize=10, linestyle='', label=plot_option["name"])[0]
            
            if plot_index:
                for i, (x, y) in enumerate(zip(one_subduction_data.age, one_subduction_data.trench_velocity)):
                    if i % plot_index_stepping == 0:
                        ax0.text(x, y + 0.5, str(i), fontsize=8*scaling_factor*font_scaling_multiplier, color="black")
            
            patches.append(_patch)
    ax0.set_xlabel("Age (Ma)")
    ax0.set_ylabel("Trench Velocity (cm/yr)")

    ax0.set_xlim(age_lim)
    ax0.set_ylim(Vtr_lim)

    ax0.xaxis.set_major_locator(MultipleLocator(age_tick_interval))
    ax0.xaxis.set_minor_locator(MultipleLocator(age_tick_interval/(n_minor_ticks+1)))
    ax0.yaxis.set_major_locator(MultipleLocator(Vtr_tick_interval))
    ax0.yaxis.set_minor_locator(MultipleLocator(Vtr_tick_interval/(n_minor_ticks+1)))

    for spine in ax0.spines.values():
        spine.set_linewidth(0.5 * scaling_factor * line_width_scaling_multiplier)

    ax0.grid()

    # add legend
    ax1 = fig.add_subplot(gs[0, 3])    
    ax1.legend(handles=patches, bbox_to_anchor=(0.5, 0.5), loc='center', ncol=2, numpoints=1, frameon=False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values():
        # Hide the rectangular box (spines)
        spine.set_visible(False)

    return fig, [ax0, ax1]
    # Reset rcParams to defaults
    rcdefaults()