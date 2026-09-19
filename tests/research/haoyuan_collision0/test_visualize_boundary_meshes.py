from types import SimpleNamespace

import numpy as np
import pytest

from hamageolib.research.haoyuan_collision0.scripts.create_boundary_meshes import (
    create_boundary_grid,
)
from hamageolib.research.haoyuan_collision0.scripts.visualize_boundary_meshes import (
    boundary_longitude,
    create_radius_spheres,
    create_velocity_glyphs,
    longitude_slice_normal,
    maximum_plane_distance,
    show_pipeline,
)


RADIUS_BOUNDS = (4_760e3, 6_360e3)
LATITUDE_BOUNDS = (-55.0, -20.0)
LONGITUDE_BOUNDS = (150.0, 210.0)


class FakeSimple:
    def __init__(self):
        self.sphere_calls = []
        self.glyph_calls = []
        self.displays = {}
        self.color_by_calls = []
        self.lookup_table = SimpleNamespace(
            presets=[],
            ranges=[],
            ApplyPreset=lambda preset, rescale: self.lookup_table.presets.append(
                (preset, rescale)
            ),
            RescaleTransferFunction=lambda minimum, maximum: (
                self.lookup_table.ranges.append((minimum, maximum))
            ),
        )

    def Sphere(self, **kwargs):
        sphere = SimpleNamespace(**kwargs)
        self.sphere_calls.append(sphere)
        return sphere

    def Glyph(self, **kwargs):
        glyph = SimpleNamespace(**kwargs)
        self.glyph_calls.append(glyph)
        return glyph

    def GetColorTransferFunction(self, array_name):
        assert array_name == "velocity"
        return self.lookup_table

    def ColorBy(self, display, array_specification):
        self.color_by_calls.append((display, array_specification))

    def GetActiveViewOrCreate(self, view_type):
        return SimpleNamespace(view_type=view_type, ResetCamera=lambda: None)

    def Show(self, source, render_view):
        display = SimpleNamespace()
        self.displays[id(source)] = display
        return display


@pytest.mark.parametrize(
    ("boundary_name", "expected_longitude"),
    [("west", 150.0), ("east", 210.0)],
)
def test_boundary_longitude_selects_exact_chunk_boundaries(
    boundary_name, expected_longitude
):
    assert boundary_longitude(boundary_name, LONGITUDE_BOUNDS) == expected_longitude


def test_boundary_longitude_rejects_non_longitudinal_boundary():
    with pytest.raises(ValueError, match="east or west"):
        boundary_longitude("north", LONGITUDE_BOUNDS)


@pytest.mark.parametrize(
    ("longitude", "expected_normal"),
    [
        (0.0, (0.0, 1.0, 0.0)),
        (90.0, (-1.0, 0.0, 0.0)),
        (150.0, (-0.5, -np.sqrt(3.0) / 2.0, 0.0)),
        (210.0, (0.5, -np.sqrt(3.0) / 2.0, 0.0)),
    ],
)
def test_longitude_slice_normal(longitude, expected_normal):
    assert longitude_slice_normal(longitude) == pytest.approx(expected_normal)


@pytest.mark.parametrize("boundary_name", ["west", "east"])
def test_boundary_mesh_points_lie_on_matching_longitude_plane(boundary_name):
    grid = create_boundary_grid(
        boundary_name,
        RADIUS_BOUNDS,
        LATITUDE_BOUNDS,
        LONGITUDE_BOUNDS,
        radius_spacing=100e3,
        lateral_spacing=2.5,
    )
    longitude = boundary_longitude(boundary_name, LONGITUDE_BOUNDS)

    distance = maximum_plane_distance(
        grid.GetPoints().GetData(), longitude_slice_normal(longitude)
    )

    assert distance < 1e-8


def test_create_radius_spheres_uses_embedded_bounds_and_smooth_resolution():
    simple = FakeSimple()

    spheres = create_radius_spheres(simple, (4_760_000.0, 6_360_000.0))

    assert set(spheres) == {"inner", "outer"}
    assert [sphere.Radius for sphere in simple.sphere_calls] == [
        4_760_000.0,
        6_360_000.0,
    ]
    assert all(sphere.ThetaResolution == 128 for sphere in simple.sphere_calls)
    assert all(sphere.PhiResolution == 128 for sphere in simple.sphere_calls)


def test_create_velocity_glyphs_uses_velocity_for_scale_and_orientation():
    simple = FakeSimple()
    boundaries = {
        name: SimpleNamespace(name=name)
        for name in ("west", "east", "north", "south")
    }

    glyphs = create_velocity_glyphs(simple, boundaries)

    assert set(glyphs) == set(boundaries)
    for boundary_name, glyph in glyphs.items():
        assert glyph.Input is boundaries[boundary_name]
        assert glyph.GlyphType == "Arrow"
        assert glyph.OrientationArray == ["POINTS", "velocity"]
        assert glyph.ScaleArray == ["POINTS", "velocity"]
        assert glyph.ScaleFactor == 1.0e6
        assert glyph.GlyphMode == "All Points"


def test_show_pipeline_colors_boundaries_by_velocity_and_glyphs_white():
    simple = FakeSimple()
    boundaries = {
        name: SimpleNamespace()
        for name in ("west", "east", "north", "south")
    }
    glyphs = {name: SimpleNamespace() for name in boundaries}

    show_pipeline(simple, boundaries, {}, glyphs)

    assert simple.lookup_table.presets == [("Blue Green Orange", True)]
    assert simple.lookup_table.ranges == [(0.0, 1.0)]
    assert [specification for _, specification in simple.color_by_calls] == [
        ("POINTS", "velocity", "Magnitude")
    ] * 4
    for boundary in boundaries.values():
        display = simple.displays[id(boundary)]
        assert display.Representation == "Surface"
    for glyph in glyphs.values():
        display = simple.displays[id(glyph)]
        assert display.ColorArrayName == [None, ""]
        assert display.DiffuseColor == (1.0, 1.0, 1.0)


def test_show_pipeline_renders_radius_spheres_with_requested_opacity():
    simple = FakeSimple()
    spheres = {
        "inner": SimpleNamespace(),
        "outer": SimpleNamespace(),
    }

    show_pipeline(simple, {}, spheres, {})

    for sphere_name, sphere in spheres.items():
        display = simple.displays[id(sphere)]
        assert display.Representation == "Surface"
        assert display.Opacity == {"inner": 1.0, "outer": 0.2}[sphere_name]
        assert display.ColorArrayName == [None, ""]
        assert display.DiffuseColor == (1.0, 1.0, 1.0)
