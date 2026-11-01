# Prescribed velocity and spherical frames

Load the shared utilities and the workflow separately with `runpy` in ParaView's
`pvpython` interpreter:

```python
import os
import runpy
from pathlib import Path

os.environ.setdefault("PV_INTERFACE_PATH", "")
repo = Path("/path/to/HaMaGeoLib")
base = runpy.run_path(str(repo / "scripts/paraview_scripts/base.py"))
workflow = runpy.run_path(
    str(repo / "scripts/paraview_scripts/PrescribedVelocity/prescribed_velocity.py"),
    init_globals=base,
)
result = workflow["plot_prescribed_velocity"](
    "/path/to/case/output/solution.pvd",
    "/path/to/images/velocity.png",
    frame_options=dict(
        radii=(5451e3, 6371e3),
        longitude=(0, 25),
        latitude=(-10, 50),
        chord_error=100.0,
        color=(0.8, 0.8, 0.8),
        line_width=2.0,
        visible=True,
        grid_longitudes=[5, 10, 15, 20],
        grid_latitudes=[0, 10, 20, 30, 40],
    ),
)
```

Omit `frame_options` to reproduce the existing plot without a frame. Specify
bounds from your case: the frame does not infer geometry from the solution mesh.
`result["frame"]` is `(source, display)` when enabled, otherwise `None`.

For another workflow, call `base["add_spherical_chunk_frame"](view, ...)`
directly. The frame contains eight curved edges and four straight radial edges.
Optional grid curves lie on the outer surface. Radii and `chord_error` use metres;
angles use degrees, with latitude measured from the equator. Bounds must increase.
Use unwrapped longitudes, e.g. `(350, 370)` for a 20 degree crossing of zero.

Constant-latitude edges follow small circles, and constant-longitude edges follow
meridians. Sampling adapts to curvature so the maximum separation between a
straight VTK chord and its analytic arc is at most `chord_error` (default 100 m).
Endpoints are included. At 6371 km radius this gives angular steps no larger than
about 0.64 degrees along a meridian, or 0.80 degrees along latitude 50 degrees.

The pure geometry helpers `spherical_segment_points` and `spherical_chunk_edges`
return NumPy Cartesian coordinates. Saved ParaView states embed the sampled
points and do not require HaMaGeoLib to reconstruct the frame.
