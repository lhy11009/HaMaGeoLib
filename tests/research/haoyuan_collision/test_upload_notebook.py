"""Run the notebook upload cell against temporary case files."""
import json
import os
from pathlib import Path
import shutil

import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0.post_process import collision_stage_paths

ANIMATIONS = ["full_domain", "trench_centered", "orogen", "full_domain_fastscape_start1Myr_end2Myr"] + [
    f"{kind}_start1Myr_end2Myr_interval0.2Myr"
    for kind in ["full_domain", "full_domain_fastscape", "trench_centered", "orogen"]
]


@pytest.mark.parametrize("tag", [None, "St20.0Ma_Cv4.00e-02_Sp2.00e-02"])
def test_upload_cell(tmp_path, tag):
    notebook = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    section = next(i for i, cell in enumerate(cells)
                   if "### Wrap up case folder to upload" in "".join(cell.get("source", [])))
    source = "".join(cells[section + 1]["source"])
    shared = tmp_path / "shared"
    shared.mkdir()
    folder = shared / "0007"
    folder.mkdir()
    (folder / "first_stage.txt").write_text("preserved")
    source = source.replace('"/home/lochy/Desktop/shared"', repr(str(shared)))
    case = tmp_path / "case"
    paths = collision_stage_paths(tag)
    files = {
        paths["case_file"]: "parameters", "case.wb": "world builder",
        f'{paths["image_directory"]}/statistic_plots/assembled.png': "statistics plot",
        f'{paths["image_directory"]}/runtime_plots/assembled.png': "runtime plot",
        f'{paths["image_directory"]}/slab_morphology.png': "slab",
        "doc/doc.md": "documentation",
    }
    for animation in ANIMATIONS:
        files[f'{paths["image_directory"]}/animation/{animation}.avi'] = animation
    for name in ["log.txt", "statistics", "original.prm"]:
        files[f'{paths["output_directory"]}/{name}'] = f"selected {name}"
    for name, contents in files.items():
        path = case / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (tmp_path / paths["case_summary_filename"]).write_text("case summary")
    (tmp_path / "readme.md").write_text("readme")
    exec(compile(source, str(notebook), "exec"), dict(
        do_post_process=True, is_case_summary=True, case_id=7,
        second_stage_tag=tag, stage_paths=paths, local_dir_2d=str(case),
        local_Collision_dir=str(tmp_path), os=os, shutil=shutil, Path=Path,
        rmtree=shutil.rmtree,
    ))
    suffix = "" if tag is None else f"_{tag}"
    assert not (shared / f"0007_{tag}").exists()
    expected = {
        "first_stage.txt": "preserved",
        paths["case_file"]: "parameters", f"case{suffix}.wb": "world builder",
        f"statistics{suffix}.png": "statistics plot", f"runtime{suffix}.png": "runtime plot",
        f"slab_morphology{suffix}.png": "slab", f"doc{suffix}.md": "documentation",
    }
    for animation in ANIMATIONS:
        expected[f"{animation}{suffix}.avi"] = animation
    if tag:
        for name in ["log.txt", "statistics", "original.prm"]:
            path = Path(name)
            expected[f"{path.stem}{suffix}{path.suffix}"] = f"selected {name}"
    assert {p.name: p.read_text() for p in folder.iterdir()} == expected
    assert (shared / paths["case_summary_filename"]).read_text() == "case summary"
    assert (shared / paths["shared_readme_filename"]).read_text() == "readme"


@pytest.mark.parametrize("existing_stage_id", [None, 0, 7])
@pytest.mark.parametrize("registered", [True, False])
def test_second_stage_inherits_registered_id(tmp_path, existing_stage_id, registered):
    notebook = Path(__file__).parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells
                  if "case_manager = CaseManager" in "".join(c.get("source", [])))
    case = tmp_path / "case"
    original = tmp_path / "case_summary_2d.csv"
    pd.DataFrame({"case_id": [7], "case_path": [str(case if registered else tmp_path / "other")]}).to_csv(original, index=False)
    original_bytes = original.read_bytes()
    paths = collision_stage_paths("test")
    tagged = tmp_path / paths["case_summary_filename"]
    if existing_stage_id is not None:
        pd.DataFrame({"case_id": [existing_stage_id], "case_path": [str(case)]}).to_csv(tagged, index=False)
    namespace = dict(do_post_process=True, local_Collision_dir=str(tmp_path),
                     local_dir_2d=str(case), case_name_2d="case", second_stage_tag="test",
                     stage_paths=paths, os=os)
    if not registered:
        from hamageolib.utils.case_manager import CaseManager
        tagged_bytes = tagged.read_bytes() if tagged.exists() else None
        with pytest.raises(CaseManager.TabelRowError):
            exec(compile(source, str(notebook), "exec"), namespace)
        assert (tagged.read_bytes() if tagged.exists() else None) == tagged_bytes
        assert original.read_bytes() == original_bytes
        return
    exec(compile(source, str(notebook), "exec"), namespace)
    assert namespace["case_id"] == 7
    assert pd.read_csv(tagged)["case_id"].tolist() == [7]
    assert original.read_bytes() == original_bytes
