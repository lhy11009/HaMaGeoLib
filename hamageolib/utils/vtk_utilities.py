import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def create_vtk_grid(points, cells):
    """
    Helper function to create a VTK unstructured grid from given points and cells.
    
    Args:
        points (np.ndarray): An array of point coordinates (N x 3).
        cells (list of list): A list of cells, where each cell contains point indices.

    Returns:
        vtkUnstructuredGrid: A VTK grid with the specified points and cells.
    """
    # Create VTK points
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)
    
    # Create VTK cells
    vtk_cells = vtk.vtkCellArray()
    for cell in cells:
        vtk_cell = vtk.vtkTriangle() if len(cell) == 3 else vtk.vtkLine()
        for i, pid in enumerate(cell):
            vtk_cell.GetPointIds().SetId(i, pid)
        vtk_cells.InsertNextCell(vtk_cell)
    
    # Create grid
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(vtk_points)
    grid.SetCells(vtk.VTK_TRIANGLE, vtk_cells)
    return grid


def calculate_resolution(grid):
    """
    Calculate the mesh resolution at different positions.

    Args:
        grid (vtkUnstructuredGrid): The VTK grid containing the mesh.

    Returns:
        dict: A dictionary with the point indices as keys and the local resolution as values.
    """
    # Extract point coordinates as a numpy array
    points = vtk_to_numpy(grid.GetPoints().GetData())
    
    # Extract cell data from the grid
    cells = grid.GetCells()

    # Initialize resolution array with infinity for each point
    resolution = np.full(points.shape[0], float('inf'))

    # Loop through each cell in the grid
    for cell_id in range(grid.GetNumberOfCells()):
        # Get the current cell and extract its point IDs
        cell = grid.GetCell(cell_id)
        cell_point_ids = [cell.GetPointId(i) for i in range(cell.GetNumberOfPoints())]

        # Compute pairwise distances between points in the cell
        for i, id1 in enumerate(cell_point_ids):
            for j, id2 in enumerate(cell_point_ids):
                if i < j:  # Avoid redundant computations
                    # Retrieve the coordinates of the two points
                    p1 = points[id1]
                    p2 = points[id2]

                    # Calculate the Euclidean distance between the two points
                    distance = np.linalg.norm(p1 - p2)

                    # Update the minimum resolution for the two points
                    resolution[id1] = min(resolution[id1], distance)
                    resolution[id2] = min(resolution[id2], distance)

    return resolution  # Return the computed resolution values


def estimate_memory_usage(u_grid):
    """
    Estimate the memory cost of loading a vtkUnstructuredGrid in ParaView.

    Parameters:
    u_grid (vtkUnstructuredGrid): The input unstructured grid.

    Returns:
    float: Estimated memory usage in megabytes (MB).
    """
    num_points = u_grid.GetNumberOfPoints()
    num_cells = u_grid.GetNumberOfCells()

    # Memory for points: Each point has 3 float (4 bytes each)
    point_memory = num_points * 3 * 4  # 3 floats (x, y, z) per point

    # Memory for cells: Each cell stores connectivity (varies by type)
    cell_array = u_grid.GetCells()
    cell_memory = cell_array.GetData().GetNumberOfValues() * 4  # Assuming 4 bytes per index

    # Memory for cell types
    cell_type_memory = num_cells * 1  # 1 byte per cell type (UInt8)

    # Memory for additional point data arrays
    point_data_memory = 0
    point_data = u_grid.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array = point_data.GetArray(i)
        if array:
            point_data_memory += array.GetDataSize() * 4  # Assuming float (4 bytes)

    # Memory for additional cell data arrays
    cell_data_memory = 0
    cell_data = u_grid.GetCellData()
    for i in range(cell_data.GetNumberOfArrays()):
        array = cell_data.GetArray(i)
        if array:
            cell_data_memory += array.GetDataSize() * 4  # Assuming float (4 bytes)

    # Total memory in bytes
    total_memory_bytes = (point_memory + cell_memory + cell_type_memory +
                          point_data_memory + cell_data_memory)
    
    # Convert to megabytes (MB)
    total_memory_mb = total_memory_bytes / (1024 * 1024)

    return total_memory_mb



def vtk_extract_comp(u_grid, *field_names, **kwargs):
    """
    Extracts the composition shape based on sum of multiple composition markers in a vtkUnstructuredGrid.

    Parameters:
        u_grid (vtk.vtkUnstructuredGrid): The input unstructured grid containing the composition fields.
        *field_names (str): Variable-length list of field names whose sum will define the slab composition.
                            At least one field must be provided.
        **kwargs:
            - threshold_value (float): The threshold value used to filter slab points (default: 0.5).

    Returns:
        vtk.vtkPolyData: A polydata object representing the extracted slab surface.
                         Returns None if any specified field is not found.

    Raises:
        AssertionError: If any of the specified fields are missing from the dataset.

    Notes:
        - The function sums the values of all specified composition fields to define the slab.
        - If any specified field is not found in `u_grid`, an assertion error is raised.
    """

    # Get threshold value from kwargs (default: 0.5)
    threshold_value = kwargs.get("threshold_value", 0.5)

    assert field_names, "At least one field name must be specified."

    # Check if the point data exists
    point_data = u_grid.GetPointData()
    
    # Verify that all requested fields exist in the dataset
    missing_fields = [field for field in field_names if not point_data.GetArray(field)]
    assert not missing_fields, f"Error: The following fields were not found in point data: {missing_fields}"

    # Initialize a new scalar field to store the summed values
    slab_field = vtk.vtkDoubleArray()
    slab_field.SetName("slab_field")
    slab_field.SetNumberOfComponents(1)
    slab_field.SetNumberOfTuples(u_grid.GetNumberOfPoints())

    # Initialize array with zeros
    for i in range(u_grid.GetNumberOfPoints()):
        slab_field.SetValue(i, 0.0)

    # Sum up all specified fields
    for field_name in field_names:
        scalar_field = point_data.GetArray(field_name)
        for i in range(u_grid.GetNumberOfPoints()):
            slab_field.SetValue(i, slab_field.GetValue(i) + scalar_field.GetValue(i))
    
    # Add the computed slab field to the unstructured grid
    point_data.AddArray(slab_field)
    point_data.SetActiveScalars("slab_field")

    # Apply threshold filter using the summed field
    threshold_filter = vtk.vtkThreshold()
    threshold_filter.SetInputData(u_grid)
    threshold_filter.SetInputArrayToProcess(0, 0, 0, 0, "slab_field")  # Use computed slab field
    threshold_filter.ThresholdByUpper(threshold_value)  # Keep values >= threshold
    threshold_filter.Update()

    # Convert to surface representation
    geometry_filter = vtk.vtkGeometryFilter()
    geometry_filter.SetInputData(threshold_filter.GetOutput())
    geometry_filter.Update()

    return geometry_filter.GetOutput()  # PolyData containing the slab surface


def extract_phi_max(points_xyz, r_bounds, theta_bounds, n_r=100, n_theta=100):
    """
    Given 3D points, extract curve of max φ at each (r, θ) location.
    
    Parameters:
        points_xyz: (N, 3) array of 3D points
        r_bounds: (r_min, r_max) in meters
        theta_bounds: (theta_min, theta_max) in radians
        n_r: number of radial bins
        n_theta: number of theta bins (colatitude)
        
    Returns:
        result_points: (n_r * n_theta, 3) array of 3D Cartesian coords of max-φ points
    """
    x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    
    # Spherical conversion
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)            # colatitude in radians
    phi = np.arctan2(y, x)              # azimuthal angle (longitude), in [-π, π]
    phi = np.mod(phi, 2 * np.pi)        # make it in [0, 2π]

    # Only use points within desired r and θ ranges
    mask = (
        (r >= r_bounds[0]) & (r <= r_bounds[1]) &
        (theta >= theta_bounds[0]) & (theta <= theta_bounds[1])
    )
    r, theta, phi, x, y, z = r[mask], theta[mask], phi[mask], x[mask], y[mask], z[mask]
    
    # Define target grid in (r, θ)
    r_grid = np.linspace(*r_bounds, n_r)
    theta_grid = np.linspace(*theta_bounds, n_theta)
    r_mesh, theta_mesh = np.meshgrid(r_grid, theta_grid)
    grid_points = np.vstack([r_mesh.ravel(), theta_mesh.ravel()]).T
    
    # Use griddata to interpolate φ onto the (r, θ) grid (using max logic below)
    result_xyz = []

    for (r0, theta0) in grid_points:
        # Small neighborhood threshold (tunable)
        dr = (r_bounds[1] - r_bounds[0]) / n_r
        dtheta = (theta_bounds[1] - theta_bounds[0]) / n_theta
        
        local_mask = (
            (np.abs(r - r0) < dr) &
            (np.abs(theta - theta0) < dtheta)
        )
        if not np.any(local_mask):
            continue
        
        # Among the local points, find the one with max φ
        idx = np.argmax(phi[local_mask])
        px = x[local_mask][idx]
        py = y[local_mask][idx]
        pz = z[local_mask][idx]
        result_xyz.append([px, py, pz])

    return np.array(result_xyz)