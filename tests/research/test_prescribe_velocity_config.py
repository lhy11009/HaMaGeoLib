"""Tests for the rule-based Stegman prescribed-velocity setup."""

import json
from pathlib import Path

from gdmate.aspect.io import parse_parameters_to_dict

from hamageolib.research.prescribe_vecloity.config import make_stegman_case


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PACKAGE_ROOT / "tests/fixtures/research/prescribe_velocity"


def test_defaults_reproduce_stegman_2d_case():
    """Default rules reproduce every parameter and World Builder value."""
    with (FIXTURE_ROOT / "case.prm").open() as prm_file:
        expected_prm = parse_parameters_to_dict(prm_file, format_entry=True)
    with (FIXTURE_ROOT / "case.wb").open() as wb_file:
        expected_wb = json.load(wb_file)

    prm_dict, wb_dict, context = make_stegman_case()

    assert prm_dict == expected_prm
    assert wb_dict == expected_wb
    assert context["plate_length"] == 2200e3
    assert context["trench_position"] == 2800e3


def test_geometry_and_slab_values_are_configurable():
    """Scientific inputs are propagated through PRM and WB structures."""
    prm_dict, wb_dict, context = make_stegman_case({
        "domain_length": 3000e3,
        "domain_depth": 800e3,
        "plate_start": 400e3,
        "trench_position": 2400e3,
        "plate_thickness": 80e3,
        "gravity": 9.81,
    })

    assert prm_dict["Geometry model"]["Box"]["X extent"] == "3e+06"
    assert prm_dict["Geometry model"]["Box"]["Y extent"] == "800000"
    assert prm_dict["Gravity model"]["Vertical"]["Magnitude"] == "9.81"
    assert wb_dict["cross section"] == [[0, 0], [3000e3, 0]]
    assert wb_dict["features"][0]["max depth"] == 80e3
    assert context["plate_length"] == 2000e3
