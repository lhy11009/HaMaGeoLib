"""Exercise two-stage configuration using a portable snapshot of a real case."""

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0.config import TwoStageRule
from hamageolib.utils.aspect.io import parse_parameters_to_dict, save_parameters_from_dict


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/research/haoyuan_collision/test_two_stage_setup"


@pytest.fixture
def two_stage_case(tmp_path):
    # CASE_OPTIONS creates plotting directories; keep the committed fixture pristine.
    case_dir = tmp_path / "base_case"
    shutil.copytree(FIXTURE / "base_case", case_dir)
    config = json.loads((FIXTURE / "current.json").read_text())
    config["two_stage_base_case"] = str(case_dir)
    with (case_dir / "case.prm").open() as stream:
        prm = parse_parameters_to_dict(stream, format_entry=True)
    wb = json.loads((case_dir / "case.wb").read_text())
    context = {"domain_depth": float(prm["Geometry model"]["Box"]["Y extent"])}
    rule = TwoStageRule()
    rule.add_default(config)
    return rule, config, prm, wb, context, case_dir


@pytest.mark.parametrize("prescribe_y", [True, False])
@pytest.mark.parametrize("checkpointing_present", [True, False])
def test_two_stage_setup_from_case(two_stage_case, tmp_path, prescribe_y, checkpointing_present):
    rule, config, prm, wb, context, case_dir = two_stage_case
    config["second_stage_prescribe_y_motion"] = prescribe_y
    if not checkpointing_present:
        prm.pop("Checkpointing", None)
    original_wb = copy.deepcopy(wb)
    original_prm = copy.deepcopy(prm)

    rule.apply(config, prm, wb, context)

    prescribed = prm["Prescribed solution"]
    assert prescribed["List of model names"] == (
        original_prm["Prescribed solution"]["List of model names"] + ", velocity function"
    )
    expected = json.loads((FIXTURE / "expected_velocity.json").read_text())
    if not prescribe_y:
        indicator = expected["Indicator function"]
        indicator["Function expression"] = indicator["Function expression"].split(";")[0] + "; 0"
    assert prescribed["Velocity function"] == expected
    assert prm["Output directory"] == "output_1"
    expected_checkpointing = original_prm.get("Checkpointing", {}).copy()
    expected_checkpointing["Resume time"] = "16207888.72701"
    assert prm["Checkpointing"] == expected_checkpointing
    assert wb == original_wb
    unchanged = copy.deepcopy(prm)
    unchanged["Prescribed solution"] = original_prm["Prescribed solution"]
    unchanged["Output directory"] = original_prm["Output directory"]
    if checkpointing_present:
        unchanged["Checkpointing"] = original_prm["Checkpointing"]
    else:
        unchanged.pop("Checkpointing")
    assert unchanged == original_prm

    # Independent check of the selected real summary row and first-match behavior.
    summary = pd.read_csv(case_dir / "summary.csv")
    convergence = summary["sp_velocity"] - summary["ov_velocity"]
    matches = summary.loc[(summary["Time"] > 14e6) & ((convergence - 0.04).abs() < 0.001)]
    assert len(matches) > 1
    selected = matches.iloc[0]
    assert selected["Vtu step"] == 162
    assert selected["Time"] == pytest.approx(16207888.72701)
    assert float(prm["Checkpointing"]["Resume time"]) == selected["Time"]
    assert float(prm["Checkpointing"]["Resume time"]) > config["second_stage_earliest_time"]
    assert selected["sp_velocity"] == pytest.approx(0.0383175835013389)
    assert selected["ov_velocity"] == pytest.approx(-0.0022147244308143)
    assert selected["trench_center_50"] == pytest.approx(2964843.75)

    # Verify the generated ASPECT parameter file retains both function components.
    output = tmp_path / "second_stage.prm"
    with output.open("w") as stream:
        save_parameters_from_dict(stream, prm)
    with output.open() as stream:
        reloaded = parse_parameters_to_dict(stream, format_entry=True)
    assert reloaded["Prescribed solution"]["Velocity function"] == expected
    assert reloaded["Output directory"] == "output_1"
    assert reloaded["Checkpointing"] == expected_checkpointing


@pytest.mark.parametrize("overrides", [
    {"second_stage_convergence": 1.0},
    {"second_stage_earliest_time": 1e12},
])
def test_two_stage_setup_no_match(two_stage_case, overrides):
    rule, config, prm, wb, context, _ = two_stage_case
    config.update(overrides)
    before = copy.deepcopy(prm)
    with pytest.raises(ValueError, match="No matching value found"):
        rule.apply(config, prm, wb, context)
    assert prm == before


def test_two_stage_setup_disabled():
    rule = TwoStageRule()
    config = {}
    rule.add_default(config)
    # No files or model dictionaries are needed when the base-case path is empty.
    prm, wb = {}, {}
    rule.apply(config, prm, wb, {})
    assert prm == {}
    assert wb == {}
