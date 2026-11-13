"""Optional upper-crust plasticity for the shallow erosion-sediment phase."""

from copy import deepcopy

import pytest

from hamageolib.research.haoyuan_collision0.config import CaseNameFromVariables, FastScapeRule
from hamageolib.utils.aspect.io import parse_composition_entry


OPTION = "erosion_sediment_use_upper_crust_plasticity"


@pytest.mark.parametrize("enabled, selected, expected", [
    (True, True, "C_ESUCP"),
    (False, True, "C"),
    (None, True, "C"),
    (True, False, "C"),
])
def test_upper_crust_plasticity_case_name(enabled, selected, expected):
    variables = {
        "weak_layer_rheology_scheme": "low friction",
        "customize_corner": False, "customize_corner_viscosity": 0.0,
        "include_fastscape": True, "include_initial_topography": False,
        "include_initial_topograph_filepath": "", "include_initial_isostacy": False,
    }
    if enabled is not None:
        variables[OPTION] = enabled
    assert CaseNameFromVariables(
        variables, prefix="C", use_all=False, use_keys=[OPTION] if selected else [],
    ) == expected


def parameters():
    return {
        "Mesh deformation": {"Free surface": {}, "Diffusion": {}},
        "Compositional fields": {"Names of fields": "sediment, crust_upper", "Number of fields": "2"},
        "Material model": {
            "Visco Plastic": {
                "Angles of internal friction": "sediment: 0.6616|30|31, crust_upper: 29|40",
                "Cohesions": "sediment: 1.2e6|6e7|7e7, crust_upper: 2e7|8e7",
                "Prefactors for dislocation creep": "sediment: 8.57e-28|3e-17|1e-50, crust_upper: 7e-28|1e-50",
                "Minimum viscosity": "sediment: 2.5e19|3e19|4e19, crust_upper: 1e19|2e19",
                "Maximum viscosity": "sediment: 2.5e23|3e23|4e23, crust_upper: 1e23|2e23",
            },
            "Multicomponent compressible": {
                "Reference densities": "sediment: 3175|3300|3400, crust_upper: 2700|3300",
            },
        },
    }


@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("density", [-1., 2650.])
def test_erosion_sediment_shallow_plasticity_only(enabled, density):
    prm = parameters()
    original = deepcopy(prm)
    config = {"include_fastscape": True, "add_erosion_sediment": True,
              "erosion_sediment_reference_density": density}
    if enabled is not None:
        config[OPTION] = enabled
    rule = FastScapeRule()
    rule.add_default(config)
    assert config[OPTION] is bool(enabled)
    rule.apply(config, prm, {}, {"total_refinement": 8})

    for name, entry in prm["Material model"]["Visco Plastic"].items():
        values = parse_composition_entry(entry)
        old = parse_composition_entry(original["Material model"]["Visco Plastic"][name])
        assert values["sediment"] == old["sediment"]
        assert values["crust_upper"] == old["crust_upper"]
        expected = old["sediment"].copy()
        if enabled and name in ("Angles of internal friction", "Cohesions"):
            expected[0] = old["crust_upper"][0]
        assert values["sediment1"] == expected
    densities = parse_composition_entry(prm["Material model"]["Multicomponent compressible"]["Reference densities"])
    assert densities["sediment1"] == [density if density > 0 else 3175., 3300., 3400.]


@pytest.mark.parametrize("fastscape, erosion", [(False, False), (False, True), (True, False)])
def test_upper_crust_plasticity_requires_erosion_sediment(fastscape, erosion):
    rule = FastScapeRule()
    config = {OPTION: True, "include_fastscape": fastscape, "add_erosion_sediment": erosion}
    rule.add_default(config)
    prm = {}
    with pytest.raises(ValueError, match="Upper-crust plasticity requires.*erosion sediment"):
        rule.apply(config, prm, {}, {})
    assert prm == {}


def test_upper_crust_plasticity_accepts_uniform_plastic_parameters():
    prm = parameters()
    plasticity = prm["Material model"]["Visco Plastic"]
    plasticity["Angles of internal friction"] = "30"
    plasticity["Cohesions"] = "2e7"
    config = {OPTION: True, "include_fastscape": True, "add_erosion_sediment": True}
    rule = FastScapeRule()
    rule.add_default(config)
    rule.apply(config, prm, {}, {"total_refinement": 8})
    assert parse_composition_entry(plasticity["Angles of internal friction"])["sediment1"] == [30.] * 3
    assert parse_composition_entry(plasticity["Cohesions"])["sediment1"] == [2e7] * 3


def test_upper_crust_plasticity_requires_upper_crust():
    prm = parameters()
    plasticity = prm["Material model"]["Visco Plastic"]
    plasticity["Prefactors for dislocation creep"] = "sediment: 8.57e-28|3e-17|1e-50"
    config = {OPTION: True, "include_fastscape": True, "add_erosion_sediment": True}
    rule = FastScapeRule()
    rule.add_default(config)
    with pytest.raises(ValueError, match="requires the crust_upper composition"):
        rule.apply(config, prm, {}, {"total_refinement": 8})
