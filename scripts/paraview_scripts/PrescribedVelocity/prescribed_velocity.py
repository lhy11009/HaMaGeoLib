"""Reproduce the prescribed-velocity visual.pvsm using utilities from base.py.

Load base.py first (or pass its runpy namespace when loading this file).
No ParaView state file is required at runtime. Paths/time in the main block
are workflow placeholders; the notebook calls the function directly.
"""
import math
from pathlib import Path


def plot_prescribed_velocity(solution_file, image_file, *, time=0.0,
                             glyph_scale=5e6, domain_samples=5000,
                             slab_samples=500, state_file=None, frame_options=None):
    """Render a green translucent slab with white domain and blue slab arrows.

    Time uses the solution PVD's units (years for the reference case). Select
    an actual saved timestep. Optionally save the reconstructed ParaView state.
    Returns the reader, threshold, glyphs and render view for interactive use.
    To add a chunk frame, supply frame_options with radii, longitude, latitude
    and optional styling/grid arguments accepted by add_spherical_chunk_frame.
    For example: dict(radii=(5451e3, 6371e3), longitude=(0, 25),
                      latitude=(-10, 50), color=(1, 1, 1)).
    The default None leaves the reference visualization unchanged.
    """
    solution_file = Path(solution_file)
    if not solution_file.is_file():
        raise FileNotFoundError(solution_file)
    if not math.isfinite(glyph_scale) or glyph_scale <= 0:
        raise ValueError("glyph_scale must be finite and positive")
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    for count in (domain_samples, slab_samples):
        if not isinstance(count, int) or count <= 0:
            raise ValueError("glyph sample counts must be positive integers")
    image_file = Path(image_file)
    image_file.parent.mkdir(parents=True, exist_ok=True)
    base_output_dir = (image_file.parent.parent if image_file.parent.name == "pv_outputs"
                       else image_file.parent)
    plot = PARAVIEW_PLOT(str(solution_file), project="ThDSubduction",
                         all_variables=["velocity", "plate"],
                         output_dir=str(base_output_dir))
    reader = plot.solutionpvd
    available = list(reader.TimestepValues)
    if available and time not in available:
        raise ValueError(f"Time {time} not in saved timesteps: {available}")
    reader.UpdatePipeline(time)
    for field in ("plate", "velocity"):
        if not source_has_field(reader, field):
            raise ValueError(f"Missing required point field: {field}")
    plot.goto_time(time)
    view = plot.renderView1
    Hide(reader, view)
    slab = add_threshold_by_field(reader, "plate", [0.8, 1.0],
                                  registrationName="Threshold1")
    slab.AllScalars = 1
    slab.UseContinuousCellRange = 0
    slab_display = Show(slab, view, "UnstructuredGridRepresentation")
    slab_display.ColorArrayName = ["POINTS", ""]
    slab_display.Representation = "Surface"
    slab_display.DiffuseColor = [0.0, 1.0, 0.0]
    slab_display.AmbientColor = [0.0, 1.0, 0.0]
    slab_display.Opacity = 0.1
    domain_glyph, _ = add_solid_velocity_glyph(
        reader, view, registration_name="Glyph1", scale_factor=glyph_scale,
        sample_points=domain_samples, color=(1.0, 1.0, 1.0))
    slab_glyph, _ = add_solid_velocity_glyph(
        slab, view, registration_name="Glyph2", scale_factor=glyph_scale,
        sample_points=slab_samples, color=(0.0, 0.0, 1.0))
    hide_all_colorbars(view)
    view.UseColorPaletteForBackground = 0
    view.Background = [0.384313725, 0.364705882, 0.352941176]
    view.ViewSize = [1024, 782]
    view.OrientationAxesVisibility = 1
    view.CenterAxesVisibility = 0
    adjust_camera(view,
                  [18877380.668035828, 1346250.5, 1887078.25],
                  [4773276.625, 1346250.5, 1887078.25],
                  3650410.7404451356, [0.0, 0.0, 1.0])
    # The shared helper defaults to 2-D interaction; this reference is 3-D.
    view.InteractionMode = "3D"
    view.CameraParallelProjection = 0
    view.CameraViewAngle = 30.0
    view.CenterOfRotation = [4773276.625, 1346250.5, 1887078.25]
    frame = (add_spherical_chunk_frame(view, **frame_options)
             if frame_options is not None else None)
    Render(view)
    SaveScreenshot(str(image_file), view, ImageResolution=[1024, 782])
    if state_file is not None:
        SaveState(str(state_file))
    return {"reader": reader, "slab": slab, "domain_glyph": domain_glyph,
            "slab_glyph": slab_glyph, "view": view, "frame": frame}


if __name__ == "__main__":
    plot_prescribed_velocity(
        "VISUALIZATION_DIRECTORY/solution.pvd",
        "IMAGE_DIRECTORY/pv_outputs/prescribed_velocity.png",
        time=float("PLOT_TIME"))
