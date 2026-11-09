"""Exercise two-stage configuration using a portable snapshot of a real case."""

import copy
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from hamageolib.research.haoyuan_collision0.config import CaseNameFromVariables, TwoStageRule
from hamageolib.utils.aspect.config_engine import RuleEngine
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


@pytest.mark.parametrize("requested", [0.0, 14012345.6789])
@pytest.mark.parametrize("sp_velocity", [0.0, 0.0312345])
def test_two_stage_direct_time(two_stage_case, tmp_path, requested, sp_velocity):
    rule, config, prm, wb, context, case_dir = two_stage_case
    config.update(second_stage_start_time=requested, second_stage_earliest_time=None,
                  second_stage_sp_velocity=sp_velocity, second_stage_convergence=0.0478912)
    # Missing observed velocities must not prevent direct assignment.
    summary = pd.read_csv(case_dir / "summary.csv")
    summary.drop(columns=["sp_velocity", "ov_velocity"]).to_csv(case_dir / "summary.csv", index=False)
    # The rule engine calls add_default again; explicit None must survive it.
    RuleEngine([rule]).add_default(config)
    assert config["second_stage_earliest_time"] is None
    rule.apply(config, prm, wb, context)
    assert float(prm["Checkpointing"]["Resume time"]) == requested
    assert prm["Output directory"] == "output_1"
    velocity = prm["Prescribed solution"]["Velocity function"]
    def constants(section):
        return {key.strip(): float(value) for key, value in
                (entry.split("=") for entry in section["Function constants"].split(","))}
    indicator = constants(velocity["Indicator function"])
    assigned = constants(velocity["Function"])
    assert indicator["t0"] == requested
    assert assigned["vx"] == pytest.approx(sp_velocity)
    assert assigned["vx_ov"] == pytest.approx(sp_velocity - 0.0478912)
    assert assigned["vx"] - assigned["vx_ov"] == pytest.approx(0.0478912)
    nearest = summary.iloc[(summary["Time"] - requested).abs().argmin()]
    assert indicator["x2"] == pytest.approx(nearest["trench_center_50"] - 500e3, rel=0.005)
    output = tmp_path / "direct.prm"
    with output.open("w") as stream:
        save_parameters_from_dict(stream, prm)
    with output.open() as stream:
        assert parse_parameters_to_dict(stream, format_entry=True) == prm


@pytest.mark.parametrize("start, earliest", [(14e6, 5e6), (None, None)])
def test_two_stage_time_options_are_exclusive(two_stage_case, start, earliest):
    rule, config, prm, wb, context, _ = two_stage_case
    config.update(second_stage_start_time=start, second_stage_earliest_time=earliest,
                  second_stage_sp_velocity=0.03)
    rule.add_default(config)
    before = copy.deepcopy(prm)
    with pytest.raises(ValueError, match="exactly one.*second_stage_start_time.*second_stage_earliest_time"):
        rule.apply(config, prm, wb, context)
    assert prm == before


def test_two_stage_direct_time_requires_sp_velocity(two_stage_case):
    rule, config, prm, wb, context, _ = two_stage_case
    config.update(second_stage_start_time=14e6, second_stage_earliest_time=None,
                  second_stage_sp_velocity=None)
    rule.add_default(config)
    with pytest.raises(ValueError, match="second_stage_sp_velocity"):
        rule.apply(config, prm, wb, context)


def test_two_stage_defaults_preserve_search_mode():
    config = {}
    TwoStageRule().add_default(config)
    assert config["second_stage_start_time"] is None
    assert config["second_stage_sp_velocity"] is None
    assert config["second_stage_earliest_time"] == 5e6


@pytest.mark.parametrize("direct, suffix", [(True, "St1.40e+07"), (False, "Et1.40e+07")])
def test_two_stage_case_name_time_modes(direct, suffix):
    variables = {
        "weak_layer_rheology_scheme": "low friction",
        "customize_corner": False, "customize_corner_viscosity": 0.0,
        "include_fastscape": False, "include_initial_topography": False,
        "include_initial_topograph_filepath": "", "include_initial_isostacy": False,
        "two_stage_base_case": "base_case",
        "second_stage_earliest_time": None if direct else 14e6,
        "second_stage_start_time": 14e6 if direct else None,
    }
    assert CaseNameFromVariables(
        variables, prefix="C", use_all=False,
        use_keys=["two_stage_base_case", "second_stage_earliest_time", "second_stage_start_time"],
    ) == "C_Two" + suffix


@pytest.mark.parametrize("variables, expected", [
    ({"second_stage_start_time": 2e6, "second_stage_convergence": 0.04,
      "second_stage_sp_velocity": 0.02}, "St2.0Ma_Cv4.00e-02_Sp2.00e-02"),
    ({"second_stage_start_time": 0.0, "second_stage_convergence": 0.04,
      "second_stage_sp_velocity": 0.0}, "St0.0Ma_Cv4.00e-02_Sp0.00e+00"),
    ({"second_stage_start_time": None, "second_stage_convergence": 0.04,
      "second_stage_sp_velocity": None}, "Cv4.00e-02"),
])
def test_two_stage_name_patterns(variables, expected):
    from hamageolib.research.haoyuan_collision0.config import TwoStageNameFromVariables
    assert TwoStageNameFromVariables(variables) == expected
    assert TwoStageNameFromVariables(variables, prefix="stage") == "stage_" + expected


def test_two_stage_name_selected_keys():
    from hamageolib.research.haoyuan_collision0.config import TwoStageNameFromVariables
    assert TwoStageNameFromVariables(
        {"second_stage_start_time": 2e6, "second_stage_convergence": 0.04},
        use_all=False, use_keys=["second_stage_start_time"],
    ) == "St2.0Ma"


def test_two_stage_notebook_named_outputs(tmp_path):
    import ast

    notebook = Path(__file__).resolve().parents[3] / "notebooks/research/haoyuan_collision/Collision.ipynb"
    cells = json.loads(notebook.read_text(encoding="utf-8"))["cells"]
    source = "".join(next(c for c in cells if c.get("id") == "two-stage-creation")["source"])
    base = tmp_path / "base_case"
    shutil.copytree(FIXTURE / "base_case", base)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id == "is_create_two_stage_case":
                node.value = ast.Constant(True)
            elif node.targets[0].id == "two_stage_base_case":
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(str(base))], keywords=[])
            elif node.targets[0].id == "two_stage_config":
                node.value = ast.parse(repr({
                    "two_stage_base_case": str(base), "second_stage_start_time": 2e6,
                    "second_stage_earliest_time": None, "second_stage_convergence": 0.04,
                    "second_stage_sp_velocity": 0.02,
                }), mode="eval").body
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(notebook), "exec"), {})
    name = "St2.0Ma_Cv4.00e-02_Sp2.00e-02"
    with (base / f"case_two_stage_{name}.prm").open() as stream:
        prm = parse_parameters_to_dict(stream, format_entry=True)
    assert prm["Output directory"] == f"output_{name}"
