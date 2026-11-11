"""Verify animation plots use the constant sea level from the case PRM."""
from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0 import post_process


@pytest.mark.parametrize("plot_name", [
    "plot_topography_trench_centered",
    "plot_topography_orogen",
    "plot_topography_full_domain_fastscape",
])
@pytest.mark.parametrize("sea_level", [None, "0.0", "1000.0", "-500.0"])
@pytest.mark.parametrize("corrected_time", [None, 0.0, 1e5])
@pytest.mark.parametrize("second_stage_tag", [None, "St2.0Ma_Cv4.00e-02_Sp2.00e-02"])
def test_topography_plots_relative_to_sea_level(
    tmp_path, monkeypatch, plot_name, sea_level, corrected_time, second_stage_tag,
):
    summary = pd.DataFrame({
        "Time": [0.0, 1e5], "Vtu step": [0, 1], "trench_center_50": [1000.0, 1000.0],
        "lower_plate_deformation": [0.0, 0.0], "upper_plate_deformation": [0.0, 0.0],
    })
    if corrected_time is not None:
        summary["average_topography"] = [1500.0, 1750.0]
    summary.attrs["Time between graphical output"] = 1e5
    options = SimpleNamespace(
        summary_df=summary,
        options={"RIGHT": 2000.0, "SEA_LEVEL": float(sea_level or 0.0)},
        resample_visualization_df=lambda interval: summary,
        pyvista_dir=str(tmp_path / post_process.collision_stage_paths(second_stage_tag)["pyvista_basename"]),
    )
    x = np.array([0.0, 1000.0, 2000.0])
    topography = np.array([500.0, 1500.0, 2500.0])
    original = topography.copy()
    monkeypatch.setattr(post_process, "read_topography_data", lambda *a, **kw: (x, topography))
    step = 1 if corrected_time == 1e5 else 0
    fastscape = Path(options.pyvista_dir) / f"{step:05d}/fastscape_{step:05d}.txt"
    fastscape.parent.mkdir(parents=True)
    np.savetxt(fastscape, np.column_stack((x, original - 100, original + 100)),
               header="x topography_min topography_max")
    original_file = fastscape.read_bytes()
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    try:
        kwargs = {"correct_average_topography": True} if corrected_time is not None else {}
        getattr(post_process, plot_name)(
            str(tmp_path), options, corrected_time or 0.0, prep_dir=str(tmp_path), **kwargs,
        )
        ax = figures[0].axes[0]
        offset = float(sea_level or 0.0) + (250.0 if corrected_time == 1e5 else 0.0)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), original - offset)
        if plot_name != "plot_topography_trench_centered":
            np.testing.assert_allclose(ax.lines[1].get_ydata(), original - 100 - offset)
            np.testing.assert_allclose(ax.lines[2].get_ydata(), original + 100 - offset)
        assert ax.get_ylabel() == "Elevation above sea level (m)"
        np.testing.assert_array_equal(topography, original)
        assert fastscape.read_bytes() == original_file
    finally:
        plt.close("all")


@pytest.mark.parametrize("sea_level", [None, "0.0", "1000.0", "-500.0"])
def test_collision_interpret_exposes_sea_level(monkeypatch, sea_level):
    from hamageolib.research.haoyuan_collision0 import case_options

    # Isolate collision-specific interpretation from generic file/time setup.
    monkeypatch.setattr(case_options.CASE_OPTIONS_BASE, "Interpret", lambda *a, **kw: None)
    case = case_options.CASE_OPTIONS_TWOD.__new__(case_options.CASE_OPTIONS_TWOD)
    case.options = {"COMPOSITION_FIELDS": []}
    case.idict = {"Material model": {"Visco Plastic": {
        "Minimum viscosity": "background: 1e19",
        "Maximum viscosity": "background: 1e24",
    }}}
    if sea_level is not None:
        case.idict["Mesh deformation"] = {
            "Fastscape": {"Marine parameters": {"Sea level": sea_level}}
        }
    case.wb_dict = {"features": [{"name": "Subducting Plate", "coordinates": [[2, 0], [1, 0]]}]}
    case.Interpret()
    assert case.options["SEA_LEVEL"] == float(sea_level or 0.0)
    assert isinstance(case.options["SEA_LEVEL"], float)
    case.contents = "elevation = topography - SEA_LEVEL"
    case.substitute()
    assert case.contents == f"elevation = topography - {float(sea_level or 0.0)}"


@pytest.mark.parametrize("second_stage_tag", [None, "St2.0Ma_Cv4.00e-02_Sp2.00e-02"])
def test_given_time_notebook_shifts_topography_average_and_basement(tmp_path, monkeypatch, second_stage_tag):
    import json
    import os
    import re
    from pathlib import Path
    from matplotlib import gridspec
    from hamageolib.utils import plot_helper

    notebook = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if c["cell_type"] == "code" and
                  "plot_topography_at_time = True" in "".join(c.get("source", [])))
    # The synthetic cases contain only t=0, independent of the notebook's chosen time.
    source, count = re.subn(r"(?m)^    plot_time = .*", "    plot_time = 0.0", source)
    assert count == 1
    dirs, options = [], []
    x = np.array([1500e3, 2500e3, 4000e3])
    topo = np.array([500., 1500., 2500.])
    basement = topo - 200.
    paths = post_process.collision_stage_paths(second_stage_tag)
    for i, sea_level in enumerate([1000., -500.]):
        case = tmp_path / f"case{i}"
        (case / paths["output_directory"] / "topography").mkdir(parents=True)
        fsdir = case / paths["pyvista_basename"] / "00000"
        fsdir.mkdir(parents=True)
        np.savetxt(case / paths["output_directory"] / "topography/topography.00000", np.column_stack((x, x*0, topo)))
        np.savetxt(fsdir / "fastscape_00000.txt", np.column_stack((x, topo, basement, x*0, x, x*0)))
        summary = pd.DataFrame({"Time": [0.], "Time step number": [0], "Vtu step": [0],
                                "lower_plate_deformation": [0.], "upper_plate_deformation": [0.]})
        dirs.append(str(case))
        options.append(SimpleNamespace(options={"SEA_LEVEL": sea_level}, summary_df=summary,
                                       output_dir=str(case / paths["output_directory"]),
                                       pyvista_dir=str(case / paths["pyvista_basename"]),
                                       img_dir=str(case / paths["image_directory"]),
                                       resample_visualization_df=lambda interval, df=summary: df))
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    ns = dict(do_post_process=True, combine_plots=True, dirs_2d=dirs,
              Case_Options_2d_array=options, output_directory="output", Path=Path,
              os=os, np=np, plt=plt, gridspec=gridspec, plot_helper=plot_helper,
              my_assert=post_process.my_assert)
    try:
        exec(compile(source, str(notebook), "exec"), ns)
        assert len(figures[0].axes) == 2
        ax, ax_basement = figures[0].axes
        for i, sea_level in enumerate([1000., -500.]):
            # Each case plots one curve and three marked points.
            np.testing.assert_allclose(ax.lines[i*4].get_ydata(), topo - sea_level)
            np.testing.assert_allclose(ax_basement.lines[i].get_ydata(), basement - sea_level)
            _, average = post_process.integrate_topography(x, topo - sea_level)
            assert ax.collections[i].get_segments()[0][0, 1] == pytest.approx(average)
        assert ax.get_ylabel() == "Elevation above sea level (m)"
        assert ax_basement.get_ylabel() == "Averaged basement above sea level (m)"
    finally:
        plt.close("all")


@pytest.mark.parametrize("rows, message", [
    ({"Time": [0.0, 1e5]}, "Missing average_topography"),
    ({"Time": [1.0, 1e5], "average_topography": [1500.0, 1750.0]}, "t=0"),
    ({"Time": [0.0, 1e5], "average_topography": [np.nan, 1750.0]}, "t=0"),
    ({"Time": [0.0, 1e5], "average_topography": [1500.0, np.nan]}, "t=100000 yr"),
])
def test_average_topography_correction_requires_summary_values(rows, message):
    with pytest.raises(ValueError, match=message):
        post_process._average_topography_correction(pd.DataFrame(rows), 1)
