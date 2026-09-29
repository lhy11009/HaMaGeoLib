import json
from pathlib import Path

from gdmate.aspect.config_engine import RuleEngine
from gdmate.aspect.io import parse_parameters_to_dict

from hamageolib.research.haoyuan_collision0.plates_test_config import (
    PLATES_TEST_RULES,
)


package_root = Path(__file__).resolve().parents[3]
fixture_dir = (
    package_root
    / "tests/fixtures/research/haoyuan_collision/test_plates_test_defaults"
)


def test_plates_test_rules_build_reference_case_from_scratch():
    with (fixture_dir / "case.prm").open() as stream:
        expected_prm = parse_parameters_to_dict(stream, format_entry=True)
    with (fixture_dir / "case.wb").open() as stream:
        expected_wb = json.load(stream)

    config = {}
    prm_dict = {}
    wb_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(config, prm_dict, wb_dict)

    assert prm_dict == expected_prm
    assert wb_dict == expected_wb


def test_plates_test_rules_are_isolated_and_consistently_named():
    assert PLATES_TEST_RULES
    assert all(
        rule.__class__.__name__.startswith("PlatesTest")
        for rule in PLATES_TEST_RULES
    )


def test_plates_test_ascii_topography_option(tmp_path):
    config = {
        "plates_test_ascii_topography": True,
        "temporary_directory": str(tmp_path),
    }
    prm_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(config, prm_dict, {})

    assert prm_dict["Output directory"] == "output"
    assert prm_dict["Mesh deformation"][
        "Mesh deformation boundary indicators"
    ] == "top: ascii data & free surface & diffusion"
    assert prm_dict["Mesh deformation"]["Ascii data model"] == {
        "Data directory": "./",
        "Data file name": "initial_topography.txt",
    }
    assert "Initial topography model" not in prm_dict["Geometry model"]
    assert (tmp_path / "initial_topography.txt").read_text() == (
        "# Initial topography equivalent to the piecewise-linear function "
        "in case.prm.\n"
        "# POINTS: 10\n"
        "# Columns: x Topography[m]\n"
        "0       -3500\n"
        "500000  -3500\n"
        "700000    500\n"
        "1700000   500\n"
        "2000000 -3500\n"
        "3000000 -3500\n"
        "3300000   500\n"
        "4800000   500\n"
        "5000000 -3500\n"
        "5500000 -3500\n"
    )
