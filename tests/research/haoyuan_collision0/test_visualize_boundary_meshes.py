from types import SimpleNamespace

from hamageolib.research.haoyuan_collision0.scripts.visualize_boundary_meshes import (
    create_radius_spheres,
    show_pipeline,
)


class FakeSimple:
    def __init__(self):
        self.sphere_calls = []
        self.displays = {}

    def Sphere(self, **kwargs):
        sphere = SimpleNamespace(**kwargs)
        self.sphere_calls.append(sphere)
        return sphere

    def GetActiveViewOrCreate(self, view_type):
        return SimpleNamespace(view_type=view_type, ResetCamera=lambda: None)

    def Show(self, source, render_view):
        display = SimpleNamespace()
        self.displays[id(source)] = display
        return display

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


def test_show_pipeline_renders_radius_spheres_as_transparent_solid_colors():
    simple = FakeSimple()
    spheres = {
        "inner": SimpleNamespace(),
        "outer": SimpleNamespace(),
    }

    show_pipeline(simple, {}, spheres)

    for sphere in spheres.values():
        display = simple.displays[id(sphere)]
        assert display.Representation == "Surface"
        assert display.Opacity == 0.02
        assert display.ColorArrayName == [None, ""]
        assert len(display.DiffuseColor) == 3
