"""Independent RuleEngine rules for the Collision plates test case.

These rules intentionally do not import or compose the production collision rules.
They construct both ASPECT and World Builder dictionaries from empty inputs.
"""

from copy import deepcopy
import json
from pathlib import Path

from gdmate.aspect.config_engine import Rule


_FILES_DIR = Path(__file__).with_name("files")


def _load_json(filename):
    """Load a bundled plates-test settings file."""
    with (_FILES_DIR / filename).open(encoding="utf-8") as stream:
        return json.load(stream)


_PLATES_TEST_PRM_SECTIONS = _load_json("plates_test_prm_sections.json")
_PLATES_TEST_WB_SETTINGS = _load_json("plates_test_world_builder.json")


def _analytic_topography_points():
    """Return the corners of the default piecewise-linear topography."""
    geometry = _PLATES_TEST_PRM_SECTIONS["geometry"]["Geometry model"]
    constants_string = geometry["Initial topography model"]["Function"][
        "Function constants"
    ]
    constants = {
        name.strip(): float(value)
        for assignment in constants_string.split(",")
        for name, value in [assignment.split("=", 1)]
    }
    x_max = float(geometry["Box"]["X extent"])

    x0 = constants["x0"]
    x1 = constants["x1"]
    x2 = constants["x2"]
    x3 = constants["x3"]
    l0 = constants["l0"]
    l1 = constants["l1"]
    l2 = constants["l2"]
    topo_continent = constants["topoC"]
    topo_ocean = constants["topoO"]

    return [
        (0.0, topo_ocean),
        (x0 - l2, topo_ocean),
        (x0, topo_continent),
        (x1 - l0, topo_continent),
        (x1, topo_ocean),
        (x2, topo_ocean),
        (x2 + l1, topo_continent),
        (x3, topo_continent),
        (x3 + l2, topo_ocean),
        (x_max, topo_ocean),
    ]


def _write_ascii_topography(case_directory):
    """Write the default analytic profile as an exact linear ASCII profile."""
    output_path = Path(case_directory) / "initial_topography.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = _analytic_topography_points()

    with output_path.open("w", encoding="utf-8") as stream:
        stream.write(
            "# Initial topography equivalent to the piecewise-linear function "
            "in case.prm.\n"
        )
        stream.write(f"# POINTS: {len(points)}\n")
        stream.write("# Columns: x Topography[m]\n")
        for x, topography in points:
            stream.write(f"{x:<8.0f}{topography:>5.0f}\n")


class PlatesTestRuntimeRule(Rule):
    """Build the runtime portion of the plates test PRM dictionary."""

    requires = ["plates_test_runtime"]
    defaults = {"plates_test_runtime": _PLATES_TEST_PRM_SECTIONS["runtime"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_runtime"]))


class PlatesTestCompositionRule(Rule):
    """Build the composition portion of the plates test PRM dictionary."""

    requires = ["plates_test_composition"]
    defaults = {"plates_test_composition": _PLATES_TEST_PRM_SECTIONS["composition"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_composition"]))


class PlatesTestGeometryRule(Rule):
    """Build the geometry portion of the plates test PRM dictionary."""

    requires = ["plates_test_geometry"]
    defaults = {"plates_test_geometry": _PLATES_TEST_PRM_SECTIONS["geometry"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_geometry"]))


class PlatesTestMaterialRule(Rule):
    """Build the material portion of the plates test PRM dictionary."""

    requires = ["plates_test_material"]
    defaults = {"plates_test_material": _PLATES_TEST_PRM_SECTIONS["material"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_material"]))


class PlatesTestNumericsRule(Rule):
    """Build the numerics portion of the plates test PRM dictionary."""

    requires = ["plates_test_numerics"]
    defaults = {"plates_test_numerics": _PLATES_TEST_PRM_SECTIONS["numerics"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_numerics"]))


class PlatesTestPostprocessRule(Rule):
    """Build the postprocess portion of the plates test PRM dictionary."""

    requires = ["plates_test_postprocess"]
    defaults = {"plates_test_postprocess": _PLATES_TEST_PRM_SECTIONS["postprocess"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_postprocess"]))


class PlatesTestAsciiTopographyRule(Rule):
    """Optionally replace analytic geometry with equivalent ASCII topography."""

    requires = [
        "plates_test_ascii_topography",
        "temporary_directory",
    ]
    defaults = {
        "plates_test_ascii_topography": False,
        "temporary_directory": ".dtemp",
    }

    def apply(self, config, prm_dict, wb_dict, context):
        if not config["plates_test_ascii_topography"]:
            return

        prm_dict["Mesh deformation"][
            "Mesh deformation boundary indicators"
        ] = "top: ascii data & free surface & diffusion"
        prm_dict["Mesh deformation"]["Ascii data model"] = {
            "Data directory": "./",
            "Data file name": "initial_topography.txt",
        }
        del prm_dict["Geometry model"]["Initial topography model"]
        _write_ascii_topography(config["temporary_directory"])


class PlatesTestWorldBuilderGlobalRule(Rule):
    """Initialize World Builder metadata without any features."""

    requires = ["plates_test_world_builder_global"]
    defaults = {
        "plates_test_world_builder_global": _PLATES_TEST_WB_SETTINGS["global"]
    }

    def apply(self, config, prm_dict, wb_dict, context):
        wb_dict.update(deepcopy(config["plates_test_world_builder_global"]))
        wb_dict["features"] = []


class PlatesTestOverridingPlateRule(Rule):
    """Append the overriding-plate World Builder features."""

    requires = ["plates_test_overriding_plate_features"]
    defaults = {
        "plates_test_overriding_plate_features": _PLATES_TEST_WB_SETTINGS[
            "overriding_plate_features"
        ]
    }

    def apply(self, config, prm_dict, wb_dict, context):
        wb_dict["features"].extend(
            deepcopy(config["plates_test_overriding_plate_features"])
        )


class PlatesTestSubductingPlateRule(Rule):
    """Append the subducting-plate World Builder features."""

    requires = ["plates_test_subducting_plate_features"]
    defaults = {
        "plates_test_subducting_plate_features": _PLATES_TEST_WB_SETTINGS[
            "subducting_plate_features"
        ]
    }

    def apply(self, config, prm_dict, wb_dict, context):
        wb_dict["features"].extend(
            deepcopy(config["plates_test_subducting_plate_features"])
        )


class PlatesTestSlabRule(Rule):
    """Append the slab World Builder features."""

    requires = ["plates_test_slab_features"]
    defaults = {"plates_test_slab_features": _PLATES_TEST_WB_SETTINGS["slab_features"]}

    def apply(self, config, prm_dict, wb_dict, context):
        wb_dict["features"].extend(deepcopy(config["plates_test_slab_features"]))


PLATES_TEST_RULES = [
    PlatesTestRuntimeRule(),
    PlatesTestCompositionRule(),
    PlatesTestGeometryRule(),
    PlatesTestMaterialRule(),
    PlatesTestNumericsRule(),
    PlatesTestPostprocessRule(),
    PlatesTestAsciiTopographyRule(),
    PlatesTestWorldBuilderGlobalRule(),
    PlatesTestOverridingPlateRule(),
    PlatesTestSubductingPlateRule(),
    PlatesTestSlabRule(),
]
