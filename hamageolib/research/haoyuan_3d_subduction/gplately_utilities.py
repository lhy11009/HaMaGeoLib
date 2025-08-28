# use the environment of py-gplate
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

from hamageolib.research.haoyuan_2d_subduction.legacy_utilities import map_mid_point, remove_substrings


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


def ReadFile(infile):
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


def get_one_subduction_by_trench_id(subduction_data, trench_pid, all_columns):
    """
    Extracts data for a specific subduction zone from the global dataset at a given reconstruction time.

    Parameters:
        subduction_data (list): The global dataset of subduction zones at a particular reconstruction time.
                                Each element in this list represents a subduction zone and contains values
                                corresponding to columns in `all_columns`.
        trench_pid (int): The ID of the specific subduction zone to extract.
        all_columns (list): The list of column names for the final DataFrame, containing 10 entries:
                            ['lon', 'lat', 'conv_rate', 'conv_angle', 'trench_velocity', 
                             'trench_velocity_angle', 'arc_length', 'trench_azimuth_angle',
                             'subducting_pid', 'trench_pid'].

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data for the specified subduction zone, 
                      sorted by latitude (column index 1).
    
    Implementation:
        - Initializes an empty list `ret` to store the selected subduction data.
        - Iterates over `subduction_data` and appends rows that match the `trench_pid` at index 9.
        - Sorts the collected rows by latitude (index 1) to order them spatially.
        - Converts the sorted list into a pandas DataFrame using `all_columns` as headers.
        - Returns the resulting DataFrame.
    """
    ret = []
    Found = False
    for row in subduction_data:
        if row[9] == trench_pid:  # Only select rows where the trench ID matches
            ret.append(row)
            Found = True

    # Assert the the given pid is contained in the subduction_data
    assert(Found)
    
    # Sort the selected data by latitude (index 1)
    ret.sort(key=lambda row: row[1])
    
    # Create a DataFrame with the selected and sorted data
    one_subduction_data = pd.DataFrame(ret, columns=all_columns)
    
    return one_subduction_data


def plot_global_basics(ax, gplot, age_grid_raster, reconstruction_time):
    """
    Plots basic global geological features on a given axis, including coastlines and an age grid.

    Parameters:
        ax (matplotlib.axes._axes.Axes): The axis on which to plot the global features.
        gplot (gplately.plot.PlotTopologies): An object used for plotting geological features such as coastlines.
        age_grid_raster: A raster object containing age data, typically used for visualizing geological ages.
        reconstruction_time (float): The geological time at which the reconstruction is plotted.
    
    Implementation:
        - Configures global gridlines on the plot with specific color, linestyle, and locations.
        - Sets the map extent to global.
        - Uses the `gplot` object to plot coastlines at the given reconstruction time.
        - Plots the age grid data on the map using a specified colormap and transparency level.
        - Adds a color bar to the plot to represent ages, with a labeled color bar axis.
    """
    # Configure global gridlines with specified color and linestyle
    gl = ax.gridlines(color='0.7', linestyle='--', xlocs=np.arange(-180, 180, 15), ylocs=np.arange(-90, 90, 15))
    gl.left_labels = True

    # Set the map extent to global
    ax.set_global()

    # Set the reconstruction time for the gplot object and plot coastlines in grey
    gplot.time = reconstruction_time
    gplot.plot_coastlines(ax, color='grey')

    # Plot the age grid on the map using a colormap from yellow to blue
    im_age = gplot.plot_grid(ax, age_grid_raster.data, cmap='YlGnBu', vmin=0, vmax=200, alpha=0.8)

    # Add a color bar for the age grid with a label
    cbar_age = plt.colorbar(im_age)
    cbar_age.ax.get_yaxis().labelpad = 15
    cbar_age.ax.set_ylabel("Age (Ma)", rotation=90)

    return ax


def resample_subduction(one_subduction_data, arc_length_edge, arc_length_resample_section, all_columns, **kwargs):
    """
    Resamples data points from a dense subduction zone at specified intervals along its arc length.
    This helps simplify and extract key properties of the subduction zone for plotting and analysis.

    Parameters:
        one_subduction_data (pd.DataFrame): A pandas DataFrame containing data for a single subduction zone.
        arc_length_edge (float): The arc length distance from the edges where no resampling is performed.
        arc_length_resample_section (float): The interval at which the arc length is resampled.
        all_columns (list): A list of column names for the output DataFrame.
        **kwargs: Additional keyword arguments.
            - indent (int, optional): Indentation for the output log content. Defaults to 0.

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
    # Initialize variables, including default indentation for output
    indent = kwargs.get("indent", 0)  # Default is no indentation
    log_output_contents = ""
    data_len = len(one_subduction_data)
    
    # Compute cumulative arc lengths
    arc_lengths = one_subduction_data['arc_length']
    arc_length_sums = np.zeros(data_len)
    arc_length_sums[0] = arc_lengths[0]
    for i in range(1, data_len):
        arc_length_sums[i] = arc_length_sums[i - 1] + arc_lengths[i]

    # Compute resampling points: start at the center and propagate outward
    temp = []
    if arc_length_sums[-1] > 2 * arc_length_edge:
        temp.append(arc_length_sums[-1] / 2.0)
    i = 1
    arc_length_sum_temp = arc_length_sums[-1] / 2.0 - arc_length_resample_section / 2.0
    arc_length_sum_temp1 = arc_length_sums[-1] / 2.0 + arc_length_resample_section / 2.0
    while arc_length_sum_temp > arc_length_edge:
        temp.append(arc_length_sum_temp)
        temp.append(arc_length_sum_temp1)
        arc_length_sum_temp -= arc_length_resample_section
        arc_length_sum_temp1 += arc_length_resample_section
    arc_length_sums_resampled = sorted(temp)

    # Resample properties of the subduction zone by interpolation
    one_subduction_data_resampled = pd.DataFrame(columns=all_columns)
    i_sbd_re = 0
    is_first = True
    for arc_length_sum_resampled in arc_length_sums_resampled:
        for i in range(len(arc_length_sums) - 1):
            if (arc_length_sums[i] <= arc_length_sum_resampled) and (arc_length_sum_resampled < arc_length_sums[i + 1]):
                # Calculate the interpolation fraction
                fraction = (arc_length_sum_resampled - arc_length_sums[i]) / (arc_length_sums[i + 1] - arc_length_sums[i])
                row_temp = fraction * one_subduction_data.iloc[i] + (1. - fraction) * one_subduction_data.iloc[i + 1]
                
                # Interpolate longitude and latitude using a custom mapping method
                row_temp.loc["lon"], row_temp.loc["lat"] = map_mid_point(
                    one_subduction_data.iloc[i].lon, one_subduction_data.iloc[i].lat,
                    one_subduction_data.iloc[i + 1].lon, one_subduction_data.iloc[i + 1].lat, fraction
                )

                # Log the resampled point's coordinates
                log_output_contents += "%s%d th resampled point: (%.2f, %.2f)\n" % (" " * indent, i_sbd_re, row_temp.lon, row_temp.lat)
                
                # Append the interpolated row to the resampled DataFrame
                if is_first:
                    one_subduction_data_resampled = pd.DataFrame([row_temp])
                    is_first = False
                else:
                    one_subduction_data_resampled = pd.concat([one_subduction_data_resampled, pd.DataFrame([row_temp])], ignore_index=True)
        i_sbd_re += 1

    return one_subduction_data_resampled, log_output_contents


def ResampleAllSubduction(subduction_data, trench_pids, arc_length_edge, arc_length_resample_section, all_columns):
    """
    Resamples all specified subduction zones in the dataset by their trench plate IDs.

    Parameters:
        subduction_data (pd.DataFrame): The global dataset of subduction zones at a reconstruction time.
        trench_pids (list): A list of trench plate IDs for which to perform resampling.
        arc_length_edge (float): The arc length distance from the edges where no resampling is performed.
        arc_length_resample_section (float): The interval at which the arc length is resampled.
        all_columns (list): A list of column names for the final resampled DataFrame.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the resampled data for all specified subduction zones.
    
    Implementation:
        - Initializes an empty DataFrame and a string to collect log output.
        - Iterates through each `trench_pid` in `trench_pids`:
            - Extracts data for the subduction zone with `trench_pid`.
            - Calls `resample_subduction` to resample the data, logging the process.
            - Appends the resampled data to the overall DataFrame.
        - Concatenates all resampled subduction data into a single DataFrame and returns it.
    """
    subduction_data_resampled = None
    log_output_contents = ""

    # Iterate over each trench plate ID and resample the subduction zone data
    for i in range(len(trench_pids)):
        trench_pid = trench_pids[i]
        
        # Extract data for the current subduction zone
        one_subduction_data = get_one_subduction_by_trench_id(subduction_data, trench_pid, all_columns)
        
        # Resample the subduction data and collect log output
        one_subduction_data_resampled, log_output_contents = resample_subduction(
            one_subduction_data, arc_length_edge, arc_length_resample_section, all_columns, indent=4
        )

        # Add information about the start and end points of the subduction zone
        log_output_contents += "%d th arc\n" % i
        log_output_contents += "start (%.2f, %.2f)\n" % (one_subduction_data.iloc[0].lon, one_subduction_data.iloc[0].lat)
        log_output_contents += "end (%.2f, %.2f)\n" % (one_subduction_data.iloc[-1].lon, one_subduction_data.iloc[-1].lat)

        # Initialize or concatenate the resampled data into the main DataFrame
        if i == 0:
            subduction_data_resampled = one_subduction_data_resampled
        else:
            if not one_subduction_data_resampled.empty:
                subduction_data_resampled = pd.concat(
                    [subduction_data_resampled, one_subduction_data_resampled], ignore_index=True
                )

    subduction_data_resampled = pd.DataFrame(subduction_data_resampled, columns=all_columns)

    return subduction_data_resampled


def ResampleSubductionById(subduction_data, trench_pid, arc_length_edge, arc_length_resample_section, all_columns):
    """
    Resamples a specific subduction zone from the global dataset using its trench plate ID.

    Parameters:
        subduction_data (pd.DataFrame): The global dataset of subduction zones at a reconstruction time.
        trench_pid (int): The ID of the trench subduction zone to resample.
        arc_length_edge (float): The arc length distance from the edges where no resampling is performed.
        arc_length_resample_section (float): The interval at which the arc length is resampled.
        all_columns (list): A list of column names for the final resampled DataFrame.

    Returns:
        tuple:
            - pd.DataFrame: A DataFrame of the resampled subduction zone data.
            - str: A log of the resampled points' details for debugging or output purposes.
    
    Implementation:
        - Extracts data for the subduction zone using `get_one_subduction_by_trench_id`.
        - Calls `resample_subduction` to resample the data based on the provided parameters.
        - Returns the resampled DataFrame and the log output string.
    """
    # Extract data for the specified subduction zone using the trench plate ID
    one_subduction_data = get_one_subduction_by_trench_id(subduction_data, trench_pid)

    # Resample the subduction zone data and get the log output
    one_subduction_data_resampled, log_output_contents = resample_subduction(
        one_subduction_data, arc_length_edge, arc_length_resample_section, all_columns, indent=4
    )

    return one_subduction_data_resampled, log_output_contents


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