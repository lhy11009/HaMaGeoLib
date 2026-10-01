import json
from pathlib import Path

import pytest

from gdmate.aspect.config_engine import RuleEngine
from gdmate.aspect.io import parse_parameters_to_dict

from hamageolib.research.haoyuan_collision0.plates_test_config import (
    PLATES_TEST_RULES,
    PlatesTestCaseNameFromVariables,
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


def test_plates_test_fastscape_with_analytic_topography():
    prm_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(
        {"plates_test_fastscape": True}, prm_dict, {}
    )

    mesh_deformation = prm_dict["Mesh deformation"]
    assert mesh_deformation[
        "Mesh deformation boundary indicators"
    ] == "top: fastscape"
    assert "Free surface" not in mesh_deformation
    assert "Diffusion" not in mesh_deformation
    assert "Ascii data model" not in mesh_deformation
    assert mesh_deformation["Fastscape"][
        "Maximum surface refinement level"
    ] == "3"
    assert mesh_deformation["Fastscape"]["Use marine component"] == "true"
    assert mesh_deformation["Fastscape"][
        "Number of fastscape timesteps per aspect timestep"
    ] == "5"
    assert mesh_deformation["Fastscape"]["Erosional parameters"] == {
        "Drainage area exponent": "0.5",
        "Bedrock diffusivity": "0.01",
        "Slope exponent": "1.0",
        "Bedrock deposition coefficient": "1.0",
        "Multi-direction slope exponent": "-1.0",
        "Use kf distribution function": "true",
        "kf distribution function": {
            "Variable names": "x, y, t",
            "Function constants": "t0=1.00e+06, IR0=3.00e-06",
            "Function expression": "(t>t0)? IR0: 0.0",
        },
    }
    assert "Initial topography model" in prm_dict["Geometry model"]
    assert prm_dict["Output directory"] == "output"


def test_plates_test_fastscape_uses_global_plus_adaptive_refinement():
    config = {
        "plates_test_fastscape": True,
        "plates_test_numerics": {
            "Mesh refinement": {
                "Initial global refinement": "4",
                "Initial adaptive refinement": "4",
            }
        },
    }
    prm_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(config, prm_dict, {})

    assert prm_dict["Mesh deformation"]["Fastscape"][
        "Maximum surface refinement level"
    ] == "8"


def test_plates_test_fastscape_with_ascii_topography(tmp_path):
    config = {
        "plates_test_fastscape": True,
        "plates_test_ascii_topography": True,
        "temporary_directory": str(tmp_path),
    }
    prm_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(config, prm_dict, {})

    mesh_deformation = prm_dict["Mesh deformation"]
    assert mesh_deformation[
        "Mesh deformation boundary indicators"
    ] == "top: ascii data & fastscape"
    assert "Fastscape" in mesh_deformation
    assert mesh_deformation["Ascii data model"] == {
        "Data directory": "./",
        "Data file name": "initial_topography.txt",
    }
    assert "Initial topography model" not in prm_dict["Geometry model"]
    assert (tmp_path / "initial_topography.txt").is_file()
    assert prm_dict["Output directory"] == "output"


def test_plates_test_case_name_uses_all_default_options():
    config = {}
    RuleEngine(PLATES_TEST_RULES).add_default(config)

    assert PlatesTestCaseNameFromVariables(config, prefix="C") == "C_gr3_ar0_S"


def test_plates_test_case_name_uses_refinement_and_active_options():
    config = {
        "plates_test_ascii_topography": True,
        "plates_test_fastscape": True,
        "plates_test_timestepping": "long",
        "plates_test_numerics": {
            "Mesh refinement": {
                "Initial global refinement": "4",
                "Initial adaptive refinement": "4",
            }
        },
    }
    RuleEngine(PLATES_TEST_RULES).add_default(config)

    assert (
        PlatesTestCaseNameFromVariables(config, prefix="C")
        == "C_gr4_ar4_L_FS_ascii"
    )


def test_plates_test_long_timestepping_scheme():
    prm_dict = {}

    RuleEngine(PLATES_TEST_RULES).apply_all(
        {"plates_test_timestepping": "long"}, prm_dict, {}
    )

    assert prm_dict["End time"] == "1e6"
    assert "Termination criteria" not in prm_dict


def test_plates_test_rejects_unknown_timestepping_scheme():
    with pytest.raises(ValueError, match="plates_test_timestepping"):
        RuleEngine(PLATES_TEST_RULES).apply_all(
            {"plates_test_timestepping": "medium"}, {}, {}
        )
