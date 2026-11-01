"""Test spherical frame geometry without requiring ParaView's Python runtime."""
import ast
from pathlib import Path

import numpy as np
import pytest

BASE = Path(__file__).parents[2] / 'scripts/paraview_scripts/base.py'


@pytest.fixture
def geometry():
    tree = ast.parse(BASE.read_text())
    names = {'spherical_segment_points', 'spherical_chunk_edges'}
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {'np': np, 'math': __import__('math')}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(BASE), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('longitude,latitude', [((0, 60), (50, 50)),
                                                ((20, 20), (-10, 50)),
                                                ((350, 370), (0, 0))])
def test_arc_geometry_and_chord_error(geometry, longitude, latitude):
    radius = 6371e3
    points = geometry['spherical_segment_points'](radius, longitude, latitude)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), radius)
    lon = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    lat = np.arcsin(points[:, 2] / radius)
    for index, angles in enumerate((longitude, latitude)):
        actual = lon if index == 0 else lat
        expected = np.radians(np.linspace(*angles, len(points)))
        np.testing.assert_allclose(np.exp(1j * actual), np.exp(1j * expected), atol=1e-14)
    # Analytic curve midpoint versus midpoint of each straight chord.
    mlon, mlat = (lon[:-1] + lon[1:]) / 2, (lat[:-1] + lat[1:]) / 2
    exact = radius * np.column_stack((np.cos(mlat)*np.cos(mlon),
                                     np.cos(mlat)*np.sin(mlon), np.sin(mlat)))
    error = np.linalg.norm(exact - (points[:-1]+points[1:])/2, axis=1)
    assert error.max() <= 100 + 1e-8
    assert len(points) > 2


def test_degenerate_and_polar_edges(geometry):
    sample = geometry['spherical_segment_points']
    assert sample(6371e3, (0, 0), (20, 20)).shape == (2, 3)
    points = sample(6371e3, (0, 360), (90, 90))
    assert points.shape == (2, 3)
    np.testing.assert_allclose(points[0], points[-1], atol=1e-8)


def test_frame_edges_and_grid(geometry):
    edges = geometry['spherical_chunk_edges']((5451e3, 6371e3), (0, 25), (-10, 50))
    assert len(edges) == 12
    assert sum(len(edge) == 2 for edge in edges) == 4
    grid = geometry['spherical_chunk_edges']((5451e3, 6371e3), (0, 25), (-10, 50),
                                            grid_longitudes=[10], grid_latitudes=[20])
    assert len(grid) == 14
    for edge in grid[-2:]:
        np.testing.assert_allclose(np.linalg.norm(edge, axis=1), 6371e3)


@pytest.mark.parametrize('kwargs', [dict(radius=-1), dict(chord_error=0),
                                   dict(latitude=(-91, -91)),
                                   dict(longitude=(0, float('nan'))),
                                   dict(longitude=(0, 10), latitude=(0, 10))])
def test_invalid_arc(geometry, kwargs):
    options = dict(radius=6371e3, longitude=(0, 20), latitude=(0, 0))
    options.update(kwargs)
    with pytest.raises(ValueError):
        geometry['spherical_segment_points'](**options)


@pytest.mark.parametrize('kwargs', [dict(radii=(6371e3, 5451e3)),
                                   dict(longitude=(350, 10)),
                                   dict(latitude=(-10, 91)),
                                   dict(grid_longitudes=[30]),
                                   dict(grid_latitudes=[float('nan')])])
def test_invalid_frame(geometry, kwargs):
    options = dict(radii=(5451e3, 6371e3), longitude=(0, 25), latitude=(-10, 50))
    options.update(kwargs)
    with pytest.raises(ValueError):
        geometry['spherical_chunk_edges'](**options)


def test_tighter_tolerance_and_reverse_arc(geometry):
    sample = geometry['spherical_segment_points']
    loose = sample(6371e3, (0, 60), (50, 50))
    tight = sample(6371e3, (0, 60), (50, 50), chord_error=10)
    assert len(tight) > len(loose)
    np.testing.assert_allclose(sample(6371e3, (60, 0), (50, 50)), loose[::-1], atol=1e-8)
