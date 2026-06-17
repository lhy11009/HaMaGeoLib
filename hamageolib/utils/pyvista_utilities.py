import numpy as np

def get_corner_point_ids(cell):
    """
    Return point IDs in the order:
        [top_right, top_left, bottom_left, bottom_right]

    Parameters
    ----------
    cell : pyvista.Cell

    Returns
    -------
    list[int]
    """
    point_ids = np.array(cell.point_ids)
    points = cell.points

    # Sort by y coordinate
    sorted_by_y = np.argsort(points[:, 1])

    bottom_local = sorted_by_y[:2]
    top_local = sorted_by_y[-2:]

    # Top points
    top_points = points[top_local]
    top_ids = point_ids[top_local]

    top_left_id = top_ids[np.argmin(top_points[:, 0])]
    top_right_id = top_ids[np.argmax(top_points[:, 0])]

    # Bottom points
    bottom_points = points[bottom_local]
    bottom_ids = point_ids[bottom_local]

    bottom_left_id = bottom_ids[np.argmin(bottom_points[:, 0])]
    bottom_right_id = bottom_ids[np.argmax(bottom_points[:, 0])]

    return [
        top_right_id,
        top_left_id,
        bottom_left_id,
        bottom_right_id,
    ]