"""Tests for the rule-based Stegman prescribed-velocity setup."""

import json
from pathlib import Path

import pytest

from gdmate.aspect.io import parse_parameters_to_dict

from hamageolib.research.prescribe_vecloity.config import make_stegman_case


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PACKAGE_ROOT / "tests/fixtures/research/prescribe_velocity"


def load_fixture(case_name):
    """Load a prescribed-velocity PRM and World Builder fixture."""
    case_dir = FIXTURE_ROOT / case_name
    with (case_dir / "case.prm").open() as prm_file:
        prm_dict = parse_parameters_to_dict(prm_file, format_entry=True)
    with (case_dir / "case.wb").open() as wb_file:
        wb_dict = json.load(wb_file)
    return prm_dict, wb_dict


def test_defaults_reproduce_stegman_2d_case():
    """Default rules reproduce every parameter and World Builder value."""
    expected_prm, expected_wb = load_fixture("default_case")

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


def test_plate_isosurface_refinement():
    """Refine the plate composition within the available adaptive levels."""
    expected_prm, expected_wb = load_fixture("adaptive_refinement")
    prm_dict, wb_dict, _ = make_stegman_case({
        "adaptive_refinement": 2,
        "refine_plate_with_isosurfaces": True,
    })

    assert prm_dict == expected_prm
    assert wb_dict == expected_wb


def test_custom_plate_isosurface_refinement():
    """Allow custom plate thresholds and refinement-level selectors."""
    prm_dict, _, _ = make_stegman_case({
        "adaptive_refinement": 3,
        "refine_plate_with_isosurfaces": True,
        "plate_isosurface_min_value": 0.25,
        "plate_isosurface_max_value": 0.75,
        "plate_isosurface_min_level": "max-1",
        "plate_isosurface_max_level": "max",
    })

    assert prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] == (
        "max-1, max, plate: 0.25 | 0.75"
    )


def test_plate_isosurface_refinement_requires_adaptive_levels():
    """Reject isosurface refinement when no adaptive level is available."""
    with pytest.raises(ValueError, match="adaptive_refinement must be positive"):
        make_stegman_case({"refine_plate_with_isosurfaces": True})


@pytest.mark.parametrize(
    ("min_value", "max_value"),
    [(-0.1, 0.5), (0.5, 0.5), (0.75, 0.5), (0.5, 1.1)],
)
def test_plate_isosurface_refinement_validates_thresholds(min_value, max_value):
    """Reject invalid composition-value intervals."""
    with pytest.raises(ValueError, match="0 <= min < max <= 1"):
        make_stegman_case({
            "adaptive_refinement": 1,
            "refine_plate_with_isosurfaces": True,
            "plate_isosurface_min_value": min_value,
            "plate_isosurface_max_value": max_value,
        })
