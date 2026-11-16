"""Exercise the notebook's snapshot selection and AVI assembly without rendering."""
import json
import os
from pathlib import Path
import re
import shutil
from types import SimpleNamespace

import pandas as pd
import numpy as np
import pytest

from hamageolib.research.haoyuan_2d_subduction import workflow_scripts
from hamageolib.research.haoyuan_collision0 import case_options, post_process
from hamageolib.utils.case_options import resample_time_series_df


@pytest.mark.parametrize(("start", "end", "expected_times", "tag"), [
    (None, None, [0, 100000, 200000, 300000], ""),
    (100000, 200000, [100000, 200000], "_start0.1Myr_end0.2Myr"),
    (150000, None, [200000, 300000], "_start0.15Myr_endopen"),
    (None, 100000, [0, 100000], "_startopen_end0.1Myr"),
    (200000, 200000, [200000], "_start0.2Myr_end0.2Myr"),
    (250000, 250000, [], None),
    (300000, 100000, [], None),
])
@pytest.mark.parametrize("second_stage_tag", [None, "St2.0Ma_Cv4.00e-02_Sp2.00e-02"])
@pytest.mark.parametrize("interval", [1e5, 2e5])
def test_basic_animation_window(tmp_path, monkeypatch, start, end, expected_times, tag, second_stage_tag, interval):
    path = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(path.read_text())["cells"]
    sources = ["".join(cell.get("source", [])) for cell in cells]
    section = next(i for i, source in enumerate(sources) if source.startswith("### 2-d case, basic\n"))
    setup = sources[section + 1]
    assert re.search(r"(?m)^time_interval = ", setup)
    setup = re.sub(r"(?m)^time_interval = [^\n]*", f"time_interval = {interval!r}", setup)
    setup = re.sub(r"(?m)^animation_types = [^\n]*", 'animation_types = ["full_domain_fastscape"]', setup)
    expected_times = [t for t in expected_times if t % interval == 0]
    if not expected_times:
        tag = None
    def run_cell(message, namespace):
        source = next(s for s in sources[section:] if f'print("{message}")' in s)
        source = source.replace("generate_topography_plot_for_animate_2d_case_basic = False",
                                "generate_topography_plot_for_animate_2d_case_basic = True")
        exec(compile(source, str(path), "exec"), namespace)
    times = [0, 100000, 200000, 300000]
    summary = pd.DataFrame({
        "Time": times, "Vtu step": [10, 20, 30, 40], "trench_center_50": [0] * 4,
    })
    summary.attrs["Time between graphical output"] = 100000
    def resample(interval):
        sampled = resample_time_series_df(summary, interval)
        sampled.attrs.update(summary.attrs)
        return sampled
    plotted_times, scripted_steps = [], []
    stage_paths = post_process.collision_stage_paths(second_stage_tag)
    case = SimpleNamespace(
        Interpret=lambda **kw: None, SummaryCaseVtuStep=lambda path: None,
        summary_df=summary, retrieve_visualization_file=scripted_steps.append,
        read_contents=lambda *a: None, substitute=lambda: None, save=lambda path: None,
        resample_visualization_df=resample,
        options={"INCLUDE_FASTSCAPE": "False"},
        img_dir=str(tmp_path / stage_paths["image_directory"]),
        paraview_dir=str(tmp_path / stage_paths["paraview_directory"]),
    )
    monkeypatch.setattr(case_options, "CASE_OPTIONS_TWOD", lambda *a, **kw: case)
    prepared = []
    def prepare(*args, **kwargs):
        prepared.append(kwargs)
        return case
    monkeypatch.setattr(post_process, "prepare_case_option_2d", prepare)
    monkeypatch.setattr(
        post_process, "plot_topography_full_domain_fastscape",
        lambda directory, options, time, **kw: plotted_times.append(time),
    )
    avi_calls, overlays = [], []
    monkeypatch.setattr(workflow_scripts, "create_avi_from_images", lambda *a: avi_calls.append(a))
    namespace = dict(
        do_post_process=True, second_stage_tag=second_stage_tag, stage_paths=stage_paths, np=np,
        local_dir_2d=str(tmp_path), output_directory="output", Path=Path,
        prm_basename_2d="case.prm", wb_basename_2d="case.wb", os=os, shutil=shutil,
        SCRIPT_DIR=str(tmp_path), py_temp_file=str(tmp_path / "animate.sh"),
        subprocess=SimpleNamespace(run=lambda *a: None), plt=SimpleNamespace(close=lambda *a: None),
        str_to_bool=lambda value: value == "True",
        plot_helper=SimpleNamespace(
            overlay_images_on_blank_canvas=lambda **kw: overlays.append(kw),
            add_text_to_image=lambda *a: None,
        ),
    )
    Path(case.img_dir).mkdir()
    Path(case.paraview_dir).mkdir()
    for name, bound in [("animation_start_time", start), ("animation_end_time", end)]:
        setup = re.sub(rf"(?m)^{name} = [^\n]*", f"{name} = {bound!r}", setup)
    if tag is None:
        with pytest.raises(ValueError, match="start time must be at or before|No snapshots"):
            exec(compile(setup, str(path), "exec"), namespace)
        return

    exec(compile(setup, str(path), "exec"), namespace)
    assert prepared[0]["second_stage_tag"] == second_stage_tag
    expected_steps = [summary.loc[times.index(t), "Vtu step"] for t in expected_times]
    run_cell("Start generating topography plot", namespace)
    assert plotted_times == expected_times
    run_cell("Start generating paraview scripts", namespace)
    assert scripted_steps == expected_steps
    prep = Path(namespace["prep_dir"])
    pv_outputs = Path(case.img_dir) / "pv_outputs"
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
    assert Path(output).name == f"full_domain_fastscape{tag}_interval{interval / 1e6:g}Myr.avi"
    assert Path(output).parent == Path(case.img_dir) / "animation"
    assert prep.parent == Path(case.img_dir)
    script = Path(namespace["animation_script_file"])
    assert f"{tag}_interval{interval / 1e6:g}Myr" in script.name
    assert second_stage_tag is None or second_stage_tag in script.name
    assert all(str(Path(case.paraview_dir)) in line for line in script.read_text().splitlines() if line.startswith("pvpython"))
    assert all(len(overlay["image_files"]) == 2 for overlay in overlays)


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_basic_animation_rejects_invalid_interval(interval):
    path = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(path.read_text())["cells"]
    setup = next("".join(c.get("source", [])) for c in cells
                 if "".join(c.get("source", [])).startswith("animate_2d_case_basic ="))
    setup = re.sub(r"(?m)^time_interval = [^\n]*", "time_interval = requested_interval", setup)
    with pytest.raises(ValueError, match="time_interval"):
        exec(compile(setup, str(path), "exec"), dict(np=np, do_post_process=True,
                                                  requested_interval=interval))
