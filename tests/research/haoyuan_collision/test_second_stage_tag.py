"""Keep post-processing inputs and outputs separate for each second-stage tag."""

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0 import post_process


TAG = "St2.0Ma_Cv4.00e-02_Sp2.00e-02"
FIXTURE = Path(__file__).parents[2] / "fixtures/research/haoyuan_collision/test_two_stage_setup/base_case"


@pytest.mark.parametrize("tag", [None, TAG])
def test_stage_case_options_and_summary(tmp_path, tag):
    case = tmp_path / "case"
    shutil.copytree(FIXTURE, case)
    suffix = "" if tag is None else f"_{tag}"
    output = case / f"output{suffix}"
    if tag is not None:
        shutil.copytree(case / "output", output)
        stats = output / "statistics"
        stats.write_text(stats.read_text().replace("output/", f"output_{tag}/"))
        shutil.copyfile(case / "case.prm", case / f"case_two_stage_{tag}.prm")
        tagged_prm = case / f"case_two_stage_{tag}.prm"
        tagged_prm.write_text(tagged_prm.read_text().replace(
            "set Output directory = output\n", f"set Output directory = output_{tag}\n"
        ))
        tagged_summary = pd.read_csv(case / "summary.csv")
        tagged_summary["sp_velocity"] = 123.0
        tagged_summary.to_csv(case / f"summary_{tag}.csv", index=False)
        original_summary = tagged_summary.copy()
        original_summary["Time"] += 1e10
        original_summary.to_csv(case / "summary.csv", index=False)
    options = post_process.prepare_case_option_2d(case, second_stage_tag=tag)
    assert options.output_dir == str(output)
    assert options.idict["Output directory"] == f"output{suffix}"
    assert options.img_dir == str(case / f"img{suffix}")
    assert options.pyvista_dir == str(case / f"pyvista_outputs{suffix}")
    assert options.summary_file == str(case / f"summary{suffix}.csv")
    assert options.paraview_dir == str(case / f"paraview_scripts{suffix}")
    assert Path(options.paraview_dir).is_dir()
    if tag is not None:
        assert (options.summary_df["sp_velocity"] == 123.0).all()
        # Interpret also loads a summary; it must not import the original stage.
        assert options.options["GRAPHICAL_TIMES"] == options.summary_df["Time"].tolist()
    (output / "topography").mkdir()
    np.savetxt(output / "topography/topography.00000", [[0, 0, 12], [1, 0, 14]])
    x, topo = post_process.read_topography_data(str(case), options, 0.)
    np.testing.assert_array_equal(topo, [12, 14])


@pytest.mark.parametrize("animation", [False, True])
def test_tagged_paraview_scripts(tmp_path, animation):
    saved = []
    options = SimpleNamespace(
        paraview_dir=str(tmp_path / f"paraview_scripts_{TAG}"), options={},
        get_pvtu_step=lambda step: step, read_contents=lambda *a: None,
        substitute=lambda: None, save=lambda path: saved.append(Path(path)) or path,
    )
    post_process.GenerateParaviewScript(
        str(tmp_path), options, ["collision0.py"], animation=animation, steps=[2]
    )
    expected = Path(options.paraview_dir)
    if animation:
        expected /= "00002"
    assert saved == [expected / "collision0.py"]


def test_notebook_stage_selection_has_no_legacy_flag():
    import json
    notebook = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    sources = "\n".join("".join(c.get("source", [])) for c in json.loads(notebook.read_text(encoding="utf-8"))["cells"])
    assert "second_stage_tag = None" in sources
    assert TAG in sources
    for old in ("is_process_second_stage", "second_stage_outputs", "output_re", "img_1", "summary_1.csv", "pyvista_outputs_1"):
        assert old not in sources


@pytest.mark.parametrize("plot_name", ["plot_topography_trench_centered", "plot_topography_orogen"])
def test_tagged_suture_plot_reads_selected_stage(tmp_path, monkeypatch, plot_name):
    import pyvista as pv
    from matplotlib.figure import Figure
    from matplotlib import pyplot as plt

    summary = pd.DataFrame({
        "Time": [0.], "Vtu step": [0], "trench_center_50": [1000.],
        "lower_plate_deformation": [0.], "upper_plate_deformation": [0.],
    })
    summary.attrs["Time between graphical output"] = 1e5
    directory = tmp_path / f"pyvista_outputs_{TAG}"
    (directory / "00000").mkdir(parents=True)
    pv.PolyData(np.array([[1500., 2000., 0.], [1600., 1000., 0.]])).save(
        directory / "00000/suture_00000.vtp"
    )
    options = SimpleNamespace(
        summary_df=summary, pyvista_dir=str(directory),
        options={"SEA_LEVEL": 0., "RIGHT": 2000.},
        resample_visualization_df=lambda interval: summary,
    )
    monkeypatch.setattr(post_process, "read_topography_data", lambda *a, **kw: (
        np.array([0., 1000., 2000.]), np.array([500., 1500., 2500.])
    ))
    figures = []
    monkeypatch.setattr(Figure, "savefig", lambda fig, *a, **kw: figures.append(fig))
    try:
        getattr(post_process, plot_name)(str(tmp_path), options, 0.)
        assert any(np.all(np.asarray(line.get_xdata()) == 0.5) for line in figures[0].axes[0].lines)
    finally:
        plt.close("all")
