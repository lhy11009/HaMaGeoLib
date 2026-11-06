"""Verify animation plots use the constant sea level from the case PRM."""
from types import SimpleNamespace

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
def test_topography_plots_relative_to_sea_level(tmp_path, monkeypatch, plot_name, sea_level):
    summary = pd.DataFrame({
        "Time": [0.0], "Vtu step": [0], "trench_center_50": [1000.0],
        "lower_plate_deformation": [0.0], "upper_plate_deformation": [0.0],
    })
    summary.attrs["Time between graphical output"] = 1e5
    options = SimpleNamespace(
        summary_df=summary,
        options={"RIGHT": 2000.0, "SEA_LEVEL": float(sea_level or 0.0)},
        resample_visualization_df=lambda interval: summary,
    )
    x = np.array([0.0, 1000.0, 2000.0])
    topography = np.array([500.0, 1500.0, 2500.0])
    original = topography.copy()
    monkeypatch.setattr(post_process, "read_topography_data", lambda *a, **kw: (x, topography))
    fastscape = tmp_path / "pyvista_outputs/00000/fastscape_00000.txt"
    fastscape.parent.mkdir(parents=True)
    np.savetxt(fastscape, np.column_stack((x, original - 100, original + 100)),
               header="x topography_min topography_max")
    original_file = fastscape.read_bytes()
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    try:
        getattr(post_process, plot_name)(str(tmp_path), options, 0.0, prep_dir=str(tmp_path))
        ax = figures[0].axes[0]
        offset = float(sea_level or 0.0)
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
