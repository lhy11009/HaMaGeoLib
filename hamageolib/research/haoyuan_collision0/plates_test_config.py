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
_PLATES_TEST_FASTSCAPE = _load_json("plates_test_fastscape.json")
_PLATES_TEST_WB_SETTINGS = _load_json("plates_test_world_builder.json")


def PlatesTestCaseNameFromVariables(
    variables, *, prefix="", use_all=True, use_keys=None
):
    """Build a PlateTest case name from its scientific configuration."""
    if use_keys is None:
        use_keys = []

    mesh_refinement = variables["plates_test_numerics"]["Mesh refinement"]
    global_refinement = int(mesh_refinement["Initial global refinement"])
    adaptive_refinement = int(
        mesh_refinement.get("Initial adaptive refinement", 0)
    )

    name_parts = [prefix] if prefix else []
    if use_all or "global_refinement" in use_keys:
        name_parts.append(f"gr{global_refinement}")
    if use_all or "adaptive_refinement" in use_keys:
        name_parts.append(f"ar{adaptive_refinement}")
    if use_all or "plates_test_timestepping" in use_keys:
        timestepping_tags = {"short": "S", "long": "L"}
        timestepping = variables["plates_test_timestepping"]
        if timestepping not in timestepping_tags:
            raise ValueError(
                "plates_test_timestepping must be either 'short' or 'long'"
            )
        name_parts.append(timestepping_tags[timestepping])
    if variables["plates_test_fastscape"] and (
        use_all or "plates_test_fastscape" in use_keys
    ):
        name_parts.append("FS")
    if variables["plates_test_ascii_topography"] and (
        use_all or "plates_test_ascii_topography" in use_keys
    ):
        name_parts.append("ascii")

    return "_".join(name_parts)


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


class PlatesTestTimesteppingRule(Rule):
    """Select the short test run or a million-year production run."""

    requires = ["plates_test_timestepping"]
    defaults = {"plates_test_timestepping": "short"}

    def apply(self, config, prm_dict, wb_dict, context):
        timestepping = config["plates_test_timestepping"]
        if timestepping == "short":
            return
        if timestepping == "long":
            prm_dict["End time"] = "1e6"
            prm_dict.pop("Termination criteria", None)
            return
        raise ValueError(
            "plates_test_timestepping must be either 'short' or 'long'"
        )


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


class PlatesTestTopRefinementRule(Rule):
    """Keep the uppermost 100 km at the model's maximum refinement."""

    requires = ["plates_test_refine_top_100_km"]
    defaults = {"plates_test_refine_top_100_km": True}

    def apply(self, config, prm_dict, wb_dict, context):
        if not config["plates_test_refine_top_100_km"]:
            return

        mesh_refinement = prm_dict["Mesh refinement"]
        global_refinement = int(mesh_refinement["Initial global refinement"])
        adaptive_refinement = int(
            mesh_refinement.get("Initial adaptive refinement", 0)
        )
        maximum_refinement = global_refinement + adaptive_refinement

        strategies = [
            strategy.strip()
            for strategy in mesh_refinement.get("Strategy", "").split(",")
            if strategy.strip()
        ]
        if "minimum refinement function" not in strategies:
            strategies.append("minimum refinement function")
        mesh_refinement["Strategy"] = ", ".join(strategies)

        domain_height = prm_dict["Geometry model"]["Box"]["Y extent"]
        mesh_refinement["Minimum refinement function"] = {
            "Coordinate system": "cartesian",
            "Variable names": "x, y",
            "Function constants": (
                f"Ymax = {domain_height}, Dp = 100e3, "
                f"R = {maximum_refinement}"
            ),
            "Function expression": "if(y > Ymax - Dp, R, 0)",
        }


class PlatesTestPostprocessRule(Rule):
    """Build the postprocess portion of the plates test PRM dictionary."""

    requires = ["plates_test_postprocess"]
    defaults = {"plates_test_postprocess": _PLATES_TEST_PRM_SECTIONS["postprocess"]}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update(deepcopy(config["plates_test_postprocess"]))


class PlatesTestFastscapeRule(Rule):
    """Optionally replace free-surface diffusion with FastScape."""

    requires = ["plates_test_fastscape"]
    defaults = {"plates_test_fastscape": False}

    def apply(self, config, prm_dict, wb_dict, context):
        if not config["plates_test_fastscape"]:
            return

        mesh_deformation = prm_dict["Mesh deformation"]
        mesh_deformation[
            "Mesh deformation boundary indicators"
        ] = "top: fastscape"
        mesh_deformation.pop("Free surface", None)
        mesh_deformation.pop("Diffusion", None)
        mesh_deformation["Fastscape"] = deepcopy(_PLATES_TEST_FASTSCAPE)
        mesh_refinement = prm_dict["Mesh refinement"]
        global_refinement = int(mesh_refinement["Initial global refinement"])
        adaptive_refinement = int(
            mesh_refinement.get("Initial adaptive refinement", 0)
        )
        mesh_deformation["Fastscape"][
            "Maximum surface refinement level"
        ] = str(global_refinement + adaptive_refinement)


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

        boundary_models = (
            "ascii data & fastscape"
            if config.get("plates_test_fastscape", False)
            else "ascii data & free surface & diffusion"
        )
        prm_dict["Mesh deformation"][
            "Mesh deformation boundary indicators"
        ] = f"top: {boundary_models}"
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
    PlatesTestTimesteppingRule(),
    PlatesTestCompositionRule(),
    PlatesTestGeometryRule(),
    PlatesTestMaterialRule(),
    PlatesTestNumericsRule(),
    PlatesTestTopRefinementRule(),
    PlatesTestPostprocessRule(),
    PlatesTestFastscapeRule(),
    PlatesTestAsciiTopographyRule(),
    PlatesTestWorldBuilderGlobalRule(),
    PlatesTestOverridingPlateRule(),
    PlatesTestSubductingPlateRule(),
    PlatesTestSlabRule(),
]
