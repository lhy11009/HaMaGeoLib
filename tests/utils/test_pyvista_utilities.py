import numpy as np
import pyvista as pv

from hamageolib.utils.pyvista_utilities import *


def test_get_corner_point_ids():
    """
    Create a single quadrilateral cell:

        3 ----- 2
        |       |
        |       |
        0 ----- 1

    Point ids:
        [0, 1, 2, 3]

    Expected:
        [2, 3, 0, 1]
        [TR, TL, BL, BR]
    """

    points = np.array([
        [0.0, 0.0, 0.0],  # 0 BL
        [1.0, 0.0, 0.0],  # 1 BR
        [1.0, 1.0, 0.0],  # 2 TR
        [0.0, 1.0, 0.0],  # 3 TL
    ])

    cells = np.array([
        4, 0, 1, 2, 3
    ])

    celltypes = np.array([pv.CellType.QUAD])

    grid = pv.UnstructuredGrid(cells, celltypes, points)

    cell = grid.get_cell(0)

    corner_ids = get_corner_point_ids(cell)

    assert corner_ids == [2, 3, 0, 1]