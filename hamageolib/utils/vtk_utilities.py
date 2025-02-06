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



