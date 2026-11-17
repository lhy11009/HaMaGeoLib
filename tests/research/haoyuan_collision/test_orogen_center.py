"""Orogen cameras and profiles share the shallow suture coordinate."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0 import post_process
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD


@pytest.mark.parametrize("animation", [False, True])
@pytest.mark.parametrize("tag", [None, "test"])
def test_orogen_generated_camera(tmp_path, animation, tag):
    case = CASE_OPTIONS_TWOD.__new__(CASE_OPTIONS_TWOD)
    case.case_dir = str(tmp_path)
    case.pp_dir = str(tmp_path)
    case.pyvista_dir = str(tmp_path / ("pyvista_outputs_test" if tag else "pyvista_outputs"))
    case.paraview_dir = str(tmp_path / "scripts")
    case.second_stage_tag = tag
    case.summary_df = pd.DataFrame({
        "Vtu step": [190], "Vtu snapshot": [190], "Time step number": [2739],
        "Time": [19e6], "trench_center_50": [3e6], "suture_shallow_x": [2376543.21],
    })
    case.visualization_df = pd.DataFrame({
        "Vtu step": [190], "Time step number": [2739],
        "Visualization file name": ["output/solution/solution-00190"],
    })
    case.options = {"PLOT_TYPE": "orogen", "GRAPHICAL_STEPS": [190], "PP_INITIAL_REFINEMENT": False}
    if animation:
        case.Interpret = lambda **kw: None
        notebook = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        source = next("".join(c.get("source", [])) for c in cells
                      if "Start generating paraview scripts" in "".join(c.get("source", [])))
        df = case.summary_df.copy()
        df.attrs["Time between graphical output"] = 1e5
        from types import SimpleNamespace
        exec(compile(source, str(notebook), "exec"), dict(
            do_post_process=True, animate_2d_case_basic=True,
            generate_paraview_scripts_for_animate_2d_case_basic=True,
            animation_script_file=str(tmp_path / "run.sh"), animation_df=df,
            Case_Options_2d=case, graphical_steps=[190], animation_types=["orogen"],
            debug_step0_animate_2d_case_basic=False, os=os,
            SCRIPT_DIR=post_process.SCRIPT_DIR, subprocess=SimpleNamespace(run=lambda *a: None),
        ))
        output = Path(case.paraview_dir) / "stepwise/collision_orogen_190.py"
    else:
        post_process.GenerateParaviewScript(str(tmp_path), case, ["collision0.py"], require_base=False)
        output = Path(case.paraview_dir) / "collision0.py"
    text = output.read_text()
    assert "CameraPosition = [2376543.21, TOP-148468.75" in text
    assert "CameraFocalPoint = [2376543.21, TOP-148468.75" in text
    assert "CameraParallelScale = 179491.639356" in text


@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_missing_suture_is_actionable(value):
    summary = pd.DataFrame({"Time": [0.], "trench_center_50": [3e6]})
    if value is not None:
        summary["suture_shallow_x"] = value
    with pytest.raises(ValueError, match="backfill_suture_shallow_x"):
        post_process.orogen_suture_center(summary, 0)
