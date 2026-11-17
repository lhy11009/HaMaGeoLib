"""Store shallow suture positions without rerunning existing VTU processing."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyvista as pv
import pytest

from hamageolib.research.haoyuan_collision0 import post_process
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD


@pytest.mark.parametrize("tag", [None, "test"])
@pytest.mark.parametrize("save_depth", [True, False])
def test_backfill_suture_summary(tmp_path, tag, save_depth):
    options = CASE_OPTIONS_TWOD.__new__(CASE_OPTIONS_TWOD)
    options.case_dir = str(tmp_path)
    options.pp_dir = str(tmp_path / "processed")
    options.pyvista_dir = str(Path(options.pp_dir) / ("pyvista_outputs_test" if tag else "pyvista_outputs"))
    options.second_stage_tag = tag
    options.options = {}
    options.summary_file = str(tmp_path / ("summary_test.csv" if tag else "summary.csv"))
    options.summary_df = pd.DataFrame({
        "Vtu step": [0, 1, 2, 3], "Vtu snapshot": [4, 5, 6, 7],
        "Time step number": [10, 20, 30, 40],
        "suture_shallow_x": [np.nan, np.nan, 999., np.nan],
    })
    options.visualization_df = pd.DataFrame({
        "Vtu step": [4, 5, 6, 7], "Time step number": [10, 20, 30, 40],
        "Visualization file name": [
            "output/solution/solution-00004",
            f"output{'_test' if tag else ''}/solution/solution-00005",
            "output/solution/solution-00006", "output/solution/solution-00007",
        ],
    })
    for snap, directory, x in [(4, Path(options.pp_dir) / "pyvista_outputs", 100.),
                               (5, Path(options.pyvista_dir), 200.)]:
        folder = directory / f"{snap:05d}"
        folder.mkdir(parents=True)
        # Unsorted profile: the shallowest finite point is not the first point.
        grid = pv.PolyData(np.array([[x+10, 900., 0.], [np.nan, 1000., 0.], [x, 995., 0.]]))
        if save_depth:
            grid["depth"] = [100., 0., 5.]
        grid.save(folder / f"suture_{snap:05d}.vtp")
    assert options.backfill_suture_shallow_x(export=True) == 2
    result = pd.read_csv(options.summary_file)
    assert result["suture_shallow_x"].iloc[:3].tolist() == [100., 200., 999.]
    assert np.isnan(result["suture_shallow_x"].iloc[3])
    assert result["Vtu snapshot"].tolist() == [4, 5, 6, 7]
    assert options.backfill_suture_shallow_x() == 0


def test_invalid_suture_profile_leaves_missing_value(tmp_path, monkeypatch):
    options = CASE_OPTIONS_TWOD.__new__(CASE_OPTIONS_TWOD)
    options.pyvista_dir = str(tmp_path)
    options.summary_df = pd.DataFrame({"Vtu snapshot": [7]})
    folder = tmp_path / "00007"
    folder.mkdir()
    profile = pv.PolyData(np.array([[np.nan, 1000., 0.], [np.inf, 995., 0.]]))
    (folder / "suture_00007.vtp").touch()
    monkeypatch.setattr(pv, "read", lambda path: profile)
    assert options.backfill_suture_shallow_x() == 0
    assert np.isnan(options.summary_df.loc[0, "suture_shallow_x"])


@pytest.mark.parametrize("particles", [False, True])
def test_new_processing_records_shallow_suture(tmp_path, monkeypatch, particles):
    processor = SimpleNamespace(
        grid=object(), suture_profile_x=np.array([np.nan, 20., 30.]),
        suture_profile_depths=np.array([0., 5e3, 10e3]),
        read=lambda *a: None, extract_slab=lambda **kw: None,
        analyze_slab=lambda: {"trench_center_50": 5.}, analyze_velocity=lambda *a: {},
        process_particles=lambda: None, analyze_trench_origin_from_particles=lambda: {},
        extract_additionals=lambda **kw: None, write_object_to_file=lambda *a: None,
    )
    class Factory:
        InitialParticlePositionException = RuntimeError
        def __new__(cls, *a, **kw):
            return processor
    monkeypatch.setattr(post_process, "PYVISTA_PROCESS_COLLISION", Factory)
    options = SimpleNamespace(
        case_dir=str(tmp_path), pyvista_dir=str(tmp_path), options={},
        summary_df=pd.DataFrame({"Vtu snapshot": [4], "Time": [0.], "Time step number": [10]}),
        retrieve_visualization_file=lambda step: "output/solution/solution-00004",
    )
    result = post_process.ProcessVtuFileTwoDStep(str(tmp_path), 4, options, include_particles=particles)
    if particles:
        assert result["suture_shallow_x"] == 20.
    else:
        # No particles: leave the initialized NaN or an existing summary value alone.
        assert "suture_shallow_x" not in result
