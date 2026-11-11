"""Exercise the notebook's snapshot selection and AVI assembly without rendering."""
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import pandas as pd
import pytest

from hamageolib.research.haoyuan_2d_subduction import workflow_scripts
from hamageolib.research.haoyuan_collision0 import case_options, post_process


@pytest.mark.parametrize(("start", "end", "expected_times", "tag"), [
    (None, None, [0, 100000, 200000, 300000], ""),
    (100000, 200000, [100000, 200000], "_start0.1Myr_end0.2Myr"),
    (150000, None, [200000, 300000], "_start0.15Myr_endopen"),
    (None, 100000, [0, 100000], "_startopen_end0.1Myr"),
    (200000, 200000, [200000], "_start0.2Myr_end0.2Myr"),
    (250000, 250000, [], None),
    (300000, 100000, [], None),
])
def test_basic_animation_window(tmp_path, monkeypatch, start, end, expected_times, tag):
    path = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(path.read_text())["cells"]
    sources = ["".join(cell.get("source", [])) for cell in cells]
    section = next(i for i, source in enumerate(sources) if source.startswith("### 2-d case, basic\n"))
    setup = sources[section + 1]
    def run_cell(message, namespace):
        source = next(s for s in sources[section:] if f'print("{message}")' in s)
        exec(compile(source, str(path), "exec"), namespace)
    times = [0, 100000, 200000, 300000]
    summary = pd.DataFrame({
        "Time": times, "Vtu step": [10, 20, 30, 40], "trench_center_50": [0] * 4,
    })
    summary.attrs["Time between graphical output"] = 100000
    plotted_times, scripted_steps = [], []
    case = SimpleNamespace(
        Interpret=lambda **kw: None, SummaryCaseVtuStep=lambda path: None,
        summary_df=summary, retrieve_visualization_file=scripted_steps.append,
        read_contents=lambda *a: None, substitute=lambda: None, save=lambda path: None,
        resample_visualization_df=lambda interval: summary,
        options={"INCLUDE_FASTSCAPE": "False"},
    )
    monkeypatch.setattr(case_options, "CASE_OPTIONS_TWOD", lambda *a, **kw: case)
    monkeypatch.setattr(
        post_process, "plot_topography_full_domain_fastscape",
        lambda directory, options, time, **kw: plotted_times.append(time),
    )
    avi_calls, overlays = [], []
    monkeypatch.setattr(workflow_scripts, "create_avi_from_images", lambda *a: avi_calls.append(a))
    namespace = dict(
        do_post_process=True, is_process_second_stage=False, local_dir_2d=str(tmp_path),
        prm_basename_2d="case.prm", wb_basename_2d="case.wb", os=os, shutil=shutil,
        SCRIPT_DIR=str(tmp_path), py_temp_file=str(tmp_path / "animate.sh"),
        subprocess=SimpleNamespace(run=lambda *a: None), plt=SimpleNamespace(close=lambda *a: None),
        str_to_bool=lambda value: value == "True",
        plot_helper=SimpleNamespace(
            overlay_images_on_blank_canvas=lambda **kw: overlays.append(kw),
            add_text_to_image=lambda *a: None,
        ),
    )
    (tmp_path / "img").mkdir()
    for name, bound in [("animation_start_time", start), ("animation_end_time", end)]:
        setup = re.sub(rf"(?m)^{name} = None\b", f"{name} = {bound!r}", setup)
    if tag is None:
        with pytest.raises(ValueError, match="start time must be at or before|No snapshots"):
            exec(compile(setup, str(path), "exec"), namespace)
        return

    exec(compile(setup, str(path), "exec"), namespace)
    expected_steps = [summary.loc[times.index(t), "Vtu step"] for t in expected_times]
    run_cell("Start generating topography plot", namespace)
    assert plotted_times == expected_times
    run_cell("Start generating paraview scripts", namespace)
    assert scripted_steps == expected_steps
    prep = Path(namespace["prep_dir"])
    pv_outputs = tmp_path / "img/pv_outputs"
    pv_outputs.mkdir()
    for time in expected_times:
        (pv_outputs / f"full_domain_viscosity_t{time:.4e}.png").touch()
        (pv_outputs / f"full_domain_viscosity_t{time:.4e}.pdf").touch()
        (prep / f"topography_full_domain_fastscape_t{time:.4e}.png").touch()
    run_cell("Start Finalizing Plots", namespace)
    assert sorted(f.name for f in prep.glob("full_domain_viscosity_*.png")) == sorted(
        f"full_domain_viscosity_t{time:.4e}.png" for time in expected_times
    )
    run_cell("Start making animation", namespace)
    assert len(avi_calls) == 1
    frames, output, _ = avi_calls[0]
    assert [Path(frame).name for frame in frames] == [
        f"full_domain_fastscape_t{time:.4e}.png" for time in expected_times
    ]
    assert Path(output).name == f"full_domain_fastscape{tag}.avi"
    assert all(len(overlay["image_files"]) == 2 for overlay in overlays)
