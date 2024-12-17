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
