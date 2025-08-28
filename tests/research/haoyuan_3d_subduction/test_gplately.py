import os
import gplately
import filecmp  # for compare file contents
from shutil import rmtree
import numpy as np
from plate_model_manager import PlateModelManager

from hamageolib.research.haoyuan_3d_subduction.gplately_utilities import ResampleAllSubduction

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_3d_subduction")

# ---------------------------------------------------------------------
# Check and make test directories
# ---------------------------------------------------------------------
test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_3d_subduction-gplately"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

def test_resample_subduction():

    # set up directory    
    output_dir = os.path.join(test_dir, "test_resample_subduction")
    source_dir = os.path.join(fixture_root, "gplately")

    if os.path.isdir(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)

    # assign a reconstruction time
    reconstruction_time=0 # time of reconstruction, must be integar

    # fact checks
    assert(type(reconstruction_time) == int)

    # Initiation
    # Initialize the anchor plate ID for the reconstruction model
    anchor_plate_id = 0

    # Define the columns used in the subduction data DataFrame
    all_columns = ['lon', 'lat', 'conv_rate', 'conv_angle', 'trench_velocity', 
                            'trench_velocity_angle', 'arc_length', 'trench_azimuth_angle', 
                            'subducting_pid', 'trench_pid']

    # Create an instance of the PlateModelManager to manage plate models
    pm_manager = PlateModelManager()

    # Load the "Muller2019" plate model from the specified data directory
    plate_model = pm_manager.get_model("Muller2019", data_dir="plate-model-repo")

    # Set up the PlateReconstruction model using the loaded plate model data
    # This includes rotation models, topologies, and static polygons, with the specified anchor plate ID
    model = gplately.PlateReconstruction(
        plate_model.get_rotation_model(), 
        plate_model.get_topologies(), 
        plate_model.get_static_polygons(),
        anchor_plate_id=anchor_plate_id
    )

    # Initialize the plotting object for visualizing topologies
    # The layers used for plotting include coastlines, continental polygons, and COBs (Continental Ocean Boundaries)
    gplot = gplately.plot.PlotTopologies(
        model, 
        plate_model.get_layer('Coastlines'), 
        plate_model.get_layer('ContinentalPolygons'), 
        plate_model.get_layer('COBs')
    )

    # Initialize the reconstruction time at 0 (current time)
    reconstruction_time = 0

    # Initialize variables to hold subduction data and trench IDs
    subduction_data = None

    # Initialize the age grid raster, which will be used for age-related computations
    age_grid_raster = None

    # get the reconstruction of subduction zones
    subduction_data = model.tessellate_subduction_zones(reconstruction_time, 
                                                    # tessellation_threshold_radians=0.01, 
                                                        anchor_plate_id=anchor_plate_id,
                                                        ignore_warnings=True)
    # get all the trench ids
    temp = [row[9] for row in subduction_data]
    trench_pids = sorted(set(temp))

    # get the age grid raster
    age_grid_raster = gplately.Raster(
                                    data=plate_model.get_raster("AgeGrids",reconstruction_time),
                                    plate_reconstruction=model,
                                    extent=[-180, 180, -90, 90]
                                    )

    age_grid_raster.fill_NaNs(inplace=True)

    arc_length_edge = 2.0
    arc_length_resample_section = 2.0

    subduction_data_resampled = ResampleAllSubduction(subduction_data, trench_pids, arc_length_edge, arc_length_resample_section, all_columns)
    
    subduction_data_resampled.loc[:, 'age'] = [np.nan for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'lon_fix'] = [np.nan for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'lat_fix'] = [np.nan for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'fix_age_polarity'] = [np.nan for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'marker'] = [np.nan for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'marker_fill'] = ['none' for i in range(len(subduction_data_resampled))]
    subduction_data_resampled.loc[:, 'color'] = [np.nan for i in range(len(subduction_data_resampled))]

    assert("trench_azimuth_angle" in subduction_data_resampled.columns and "arc_length" in subduction_data_resampled.columns)
    output_file = os.path.join(output_dir, "resampled.csv")
    subduction_data_resampled.to_csv(output_file)

    output_file_std = os.path.join(source_dir, "resampled_std.csv") 
    assert(filecmp.cmp(output_file, output_file_std))
