"""Optional crustal sediment thermodynamics are independent of rheology."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from hamageolib.research.haoyuan_collision0.config import PhaseTransitionRule, FastScapeRule, CaseNameFromVariables
from hamageolib.utils.aspect.io import parse_composition_entry

OPTION = "sediments_use_upper_crust_PT"
PT_FILE = "hamageolib/research/haoyuan_collision0/files/HeFESTo/phases_fit_10212026.json"
ENTRIES = ["Phase transition depths", "Phase transition widths", "Phase transition temperatures",
           "Phase transition Clapeyron slopes", "Reference densities", "Reference temperatures",
           "Reference thermal expansivities", "Reference isothermal compressibilities",
           "Isochoric specific heats", "Isothermal bulk modulus pressure derivatives"]


def parameters():
    return {
        "Mesh deformation": {"Free surface": {}, "Diffusion": {}},
        "Compositional fields": {"Names of fields": "sediment, crust_upper", "Number of fields": "2"},
        "Material model": {"Compositing": {}, "Multicomponent compressible": {},
                           "Visco Plastic": {"Cohesions": "sediment: 1e6, crust_upper: 2e7"}},
    }


@pytest.mark.parametrize("enabled", [None, False, True])
def test_sediment_thermodynamics_and_erosion_duplication(enabled):
    config = {"include_phase_transition": True, "PT_config_file": PT_FILE,
              "include_fastscape": True, "add_erosion_sediment": True}
    if enabled is not None:
        config[OPTION] = enabled
    prm = parameters()
    old_vp = deepcopy(prm["Material model"]["Visco Plastic"])
    rule = PhaseTransitionRule()
    rule.add_default(config)
    assert config[OPTION] is bool(enabled)
    rule.apply(config, prm, {}, {})
    assert prm["Material model"]["Visco Plastic"] == old_vp
    fastscape = FastScapeRule()
    fastscape.add_default(config)
    fastscape.apply(config, prm, {}, {"total_refinement": 8})
    for entry in ENTRIES:
        values = parse_composition_entry(prm["Material model"]["Multicomponent compressible"][entry])
        expected = values["crust_upper" if enabled else "MORB"]
        assert values["sediment"] == expected
        assert values["sediment1"] == expected
        assert values["gabbro"] == values["MORB"]
    plasticity = parse_composition_entry(prm["Material model"]["Visco Plastic"]["Cohesions"])
    assert plasticity["sediment"] == plasticity["sediment1"] == 1e6


def test_existing_sediment1_and_scalar_thermodynamics(tmp_path):
    data = json.loads((Path(__file__).parents[3] / PT_FILE).read_text())
    for key in ["beta_ref", "T_ref", "alpha_ref", "cp_ref", "Kp"]:
        data[key] = 4.0
    path = tmp_path / "pt.json"
    path.write_text(json.dumps(data))
    config = {OPTION: True, "include_phase_transition": True, "PT_config_file": str(path)}
    prm = parameters()
    prm["Compositional fields"]["Names of fields"] += ", sediment1"
    rule = PhaseTransitionRule()
    rule.add_default(config)
    rule.apply(config, prm, {}, {})
    material = prm["Material model"]["Multicomponent compressible"]
    for entry in ENTRIES[:5]:
        values = parse_composition_entry(material[entry])
        assert values["sediment"] == values["sediment1"] == values["crust_upper"]
    for entry in ENTRIES[5:]:
        assert material[entry] == "4.0"


def test_upper_crust_pt_requires_phase_transitions():
    rule = PhaseTransitionRule()
    config = {OPTION: True}
    rule.add_default(config)
    prm = parameters()
    before = deepcopy(prm)
    with pytest.raises(ValueError, match="requires include_phase_transition"):
        rule.apply(config, prm, {}, {})
    assert prm == before


@pytest.mark.parametrize("enabled", [False, True])
def test_case_name_identifies_sediment_pt(enabled):
    variables = {OPTION: enabled, "weak_layer_rheology_scheme": "low friction",
                 "customize_corner": False, "customize_corner_viscosity": 0.,
                 "include_fastscape": False, "include_initial_topography": False,
                 "include_initial_topograph_filepath": "", "include_initial_isostacy": False}
    assert CaseNameFromVariables(variables, prefix="C", use_all=False, use_keys=[OPTION]) == (
        "C_SedUCPT" if enabled else "C")


def test_explicit_erosion_density_override_is_retained():
    config = {OPTION: True, "include_phase_transition": True, "PT_config_file": PT_FILE,
              "include_fastscape": True, "add_erosion_sediment": True,
              "erosion_sediment_reference_density": 2650.}
    prm = parameters()
    for rule in [PhaseTransitionRule(), FastScapeRule()]:
        rule.add_default(config)
        rule.apply(config, prm, {}, {"total_refinement": 8})
    values = parse_composition_entry(prm["Material model"]["Multicomponent compressible"]["Reference densities"])
    assert values["sediment"] == values["crust_upper"]
    assert values["sediment1"] == [2650., *values["crust_upper"][1:]]
