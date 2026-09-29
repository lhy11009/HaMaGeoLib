"""Rule-based configuration for the 2-D Stegman et al. (2006) model.

The defaults reproduce Model 16 as represented by the ``stegman_2d`` MOW
case. The finite slab width is intentionally omitted in this 2-D reduction.
"""

from copy import deepcopy
from math import asin, cos, degrees

from gdmate.aspect.config_engine import Rule, RuleEngine


def _number(value):
    """Return a compact ASPECT-compatible representation of a number."""
    return f"{value:g}"


def _reference_number(value, default, reference_text):
    """Preserve the notation used by the reference case for default values."""
    return reference_text if value == default else _number(value)


class StegmanGeometryRule(Rule):
    """Configure the Cartesian domain, runtime, boundaries, and temperature."""

    requires = [
        "dimension", "end_time", "output_directory", "world_builder_file",
        "domain_length", "domain_depth", "x_repetitions", "y_repetitions",
        "global_refinement", "adaptive_refinement", "gravity",
        "reference_temperature", "refine_plate_with_isosurfaces",
        "plate_isosurface_min_value", "plate_isosurface_max_value",
        "plate_isosurface_min_level", "plate_isosurface_max_level",
    ]
    defaults = {
        "dimension": 2,
        "end_time": 30e6,
        "output_directory": "output",
        "world_builder_file": "case.wb",
        "domain_length": 4000e3,
        "domain_depth": 1000e3,
        "x_repetitions": 4,
        "y_repetitions": 1,
        "global_refinement": 4,
        "adaptive_refinement": 0,
        "gravity": 10.0,
        "reference_temperature": 1573.0,
        "refine_plate_with_isosurfaces": False,
        "plate_isosurface_min_value": 0.5,
        "plate_isosurface_max_value": 1.0,
        "plate_isosurface_min_level": "max",
        "plate_isosurface_max_level": "max",
    }
    provides = ["domain_length", "domain_depth"]

    def apply(self, config, prm_dict, wb_dict, context):
        temperature = _number(config["reference_temperature"])
        mesh_refinement = {
            "Initial global refinement": str(config["global_refinement"]),
            "Initial adaptive refinement": str(config["adaptive_refinement"]),
            "Time steps between mesh refinement": "1",
        }
        if config["refine_plate_with_isosurfaces"]:
            if config["adaptive_refinement"] <= 0:
                raise ValueError(
                    "adaptive_refinement must be positive when plate isosurface "
                    "refinement is enabled."
                )
            min_value = config["plate_isosurface_min_value"]
            max_value = config["plate_isosurface_max_value"]
            if not 0 <= min_value < max_value <= 1:
                raise ValueError(
                    "Plate isosurface values must satisfy 0 <= min < max <= 1."
                )
            mesh_refinement.update({
                "Strategy": "isosurfaces",
                "Isosurfaces": {
                    "Isosurfaces": (
                        f"{config['plate_isosurface_min_level']}, "
                        f"{config['plate_isosurface_max_level']}, plate: "
                        f"{_number(min_value)} | {_number(max_value)}"
                    ),
                },
            })
        prm_dict.update({
            "Dimension": str(config["dimension"]),
            "Use years instead of seconds": "true",
            "Start time": "0",
            "End time": _reference_number(config["end_time"], 30e6, "30e6"),
            "Output directory": str(config["output_directory"]),
            "World builder file": str(config["world_builder_file"]),
            "Adiabatic surface temperature": temperature,
            "Pressure normalization": "surface",
            "Surface pressure": "0",
            "Geometry model": {
                "Model name": "box",
                "Box": {
                    "X extent": _reference_number(
                        config["domain_length"], 4000e3, "4000e3"
                    ),
                    "Y extent": _reference_number(
                        config["domain_depth"], 1000e3, "1000e3"
                    ),
                    "X repetitions": str(config["x_repetitions"]),
                    "Y repetitions": str(config["y_repetitions"]),
                },
            },
            "Mesh refinement": mesh_refinement,
            "Boundary velocity model": {
                "Tangential velocity boundary indicators": "left, right, bottom, top",
            },
            "Gravity model": {
                "Model name": "vertical",
                "Vertical": {"Magnitude": _number(config["gravity"])},
            },
            "Initial temperature model": {
                "List of model names": "function",
                "Function": {"Function expression": temperature},
            },
            "Boundary temperature model": {
                "Fixed temperature boundary indicators": "left, right, bottom, top",
                "List of model names": "constant",
                "Constant": {
                    "Boundary indicator to temperature mappings":
                        f"left:{temperature}, right:{temperature}, "
                        f"bottom:{temperature}, top:{temperature}",
                },
            },
        })
        context["domain_length"] = config["domain_length"]
        context["domain_depth"] = config["domain_depth"]


class StegmanSlabRule(Rule):
    """Configure the plate field and its initial World Builder geometry."""

    requires = [
        "plate_name", "plate_start", "trench_position", "plate_thickness",
        "slab_segment_lengths", "slab_segment_angles", "slab_max_depth",
        "dip_point", "long_slab", "long_slab_length", "long_slab_end_angle",
    ]
    defaults = {
        "plate_name": "plate",
        "plate_start": 600e3,
        "trench_position": 2800e3,
        "plate_thickness": 100e3,
        "slab_segment_lengths": [100e3, 150e3],
        "slab_segment_angles": [[0.0, 45.0], [45.0, 90.0]],
        "slab_max_depth": 300e3,
        "dip_point": [4000e3, 0.0],
        "long_slab": False,
        "long_slab_length": 300e3,
        "long_slab_end_angle": 60.0,
    }
    provides = ["plate_length", "trench_position"]

    def apply(self, config, prm_dict, wb_dict, context):
        if len(config["slab_segment_lengths"]) != len(config["slab_segment_angles"]):
            raise ValueError("Each slab segment length must have an angle pair.")

        prm_dict["Initial composition model"] = {"List of model names": "world builder"}
        prm_dict["Compositional fields"] = {
            "Number of fields": "1",
            "Names of fields": str(config["plate_name"]),
        }

        thickness = config["plate_thickness"]
        trench = config["trench_position"]
        composition_model = {
            "model": "uniform",
            "compositions": [0],
            "min depth": 0,
            "max depth": thickness,
        }
        segments = [
            {
                "length": length,
                "thickness": [thickness],
                "angle": list(angles),
            }
            for length, angles in zip(
                config["slab_segment_lengths"], config["slab_segment_angles"]
            )
        ]
        if config["long_slab"]:
            if config["long_slab_length"] <= 0:
                raise ValueError("long_slab_length must be positive.")
            if not 0 <= config["long_slab_end_angle"] <= 180:
                raise ValueError("long_slab_end_angle must be between 0 and 180 degrees.")
            segments.append({
                "length": config["long_slab_length"],
                "thickness": [thickness],
                "angle": [
                    config["slab_segment_angles"][-1][-1],
                    config["long_slab_end_angle"],
                ],
            })
        wb_dict.clear()
        wb_dict.update({
            "version": "1.2",
            "coordinate system": {"model": "cartesian"},
            "cross section": [[0, 0], [context["domain_length"], 0]],
            "features": [
                {
                    "model": "oceanic plate",
                    "name": "horizontal plate",
                    "min depth": 0,
                    "max depth": thickness,
                    "coordinates": [
                        [config["plate_start"], -1e3], [trench, -1e3],
                        [trench, 1e3], [config["plate_start"], 1e3],
                    ],
                    "composition models": [composition_model],
                },
                {
                    "model": "subducting plate",
                    "name": "initial slab perturbation",
                    "coordinates": [[trench, -1e3], [trench, 1e3]],
                    "dip point": list(config["dip_point"]),
                    "max depth": config["slab_max_depth"],
                    "segments": segments,
                    "composition models": [{
                        "model": "uniform",
                        "compositions": [0],
                        "min distance slab top": 0,
                        "max distance slab top": thickness,
                    }],
                },
            ],
        })
        context["plate_length"] = trench - config["plate_start"]
        context["trench_position"] = trench


class StegmanPrescribedVelocityRule(Rule):
    """Prescribe World Builder along-surface velocity within the slab."""

    requires = [
        "prescribe_slab_velocity",
        "slab_velocity_magnitude",
        "plate_thickness",
    ]
    defaults = {
        "prescribe_slab_velocity": False,
        "slab_velocity_magnitude": 0.05,
        "plate_thickness": 100e3,
    }
    provides = ["prescribed_slab_velocity"]

    def apply(self, config, prm_dict, wb_dict, context):
        context["prescribed_slab_velocity"] = config["prescribe_slab_velocity"]
        if not config["prescribe_slab_velocity"]:
            return

        velocity_magnitude = config["slab_velocity_magnitude"]
        if velocity_magnitude <= 0:
            raise ValueError("slab_velocity_magnitude must be positive.")

        slab_features = [
            feature for feature in wb_dict.get("features", [])
            if feature.get("model") == "subducting plate"
        ]
        if len(slab_features) != 1:
            raise ValueError(
                "Prescribing slab velocity requires exactly one subducting plate feature."
            )

        wb_dict["indicator properties"] = [
            {"index": 0, "name": "temperature"},
            {"index": 1, "name": "velocity"},
            {"index": 2, "name": "composition"},
        ]
        slab = slab_features[0]
        slab["velocity models"] = [{
            "model": "along surface",
            "velocity magnitude": velocity_magnitude,
        }]
        slab["indicator models"] = [{
            "model": "uniform",
            "min distance slab top": 0,
            "max distance slab top": config["plate_thickness"],
            "indicators": ["velocity"],
        }]

        prm_dict["Prescribed solution"] = {
            "List of model names": "world builder",
        }


class StegmanMaterialRule(Rule):
    """Configure Model 16 densities, viscosities, and plastic yielding."""

    requires = [
        "mantle_density", "plate_density", "heat_capacity",
        "phase_transition_depth", "phase_transition_width",
        "upper_mantle_viscosity", "lower_mantle_viscosity",
        "plate_viscosity", "minimum_viscosity", "maximum_viscosity",
        "reference_strain_rate", "yield_cohesion", "friction_coefficient",
    ]
    defaults = {
        "mantle_density": 3300.0,
        "plate_density": 3380.0,
        "heat_capacity": 1250.0,
        "phase_transition_depth": 660e3,
        "phase_transition_width": 5e3,
        "upper_mantle_viscosity": 1e20,
        "lower_mantle_viscosity": 1e22,
        "plate_viscosity": 2e22,
        "minimum_viscosity": 1e17,
        "maximum_viscosity": 1e25,
        "reference_strain_rate": 1e-15,
        "yield_cohesion": 40e6,
        "friction_coefficient": 0.2,
    }

    def apply(self, config, prm_dict, wb_dict, context):
        phi = asin(config["friction_coefficient"])
        aspect_cohesion = config["yield_cohesion"] / cos(phi)
        diffusion_prefactors = (
            f"background:{_number(0.5 / config['upper_mantle_viscosity'])}|"
            f"{_number(0.5 / config['lower_mantle_viscosity'])}, "
            f"plate:{_number(0.5 / config['plate_viscosity'])}|"
            f"{_number(0.5 / config['plate_viscosity'])}"
        )
        prm_dict["Material model"] = {
            "Model name": "visco plastic",
            "Material averaging": "harmonic average only viscosity",
            "Visco Plastic": {
                "Densities": (
                    f"background:{_number(config['mantle_density'])}|"
                    f"{_number(config['mantle_density'])}, "
                    f"plate:{_number(config['plate_density'])}|"
                    f"{_number(config['plate_density'])}"
                ),
                "Thermal expansivities": "background:0|0, plate:0|0",
                "Heat capacities": (
                    f"background:{_number(config['heat_capacity'])}|"
                    f"{_number(config['heat_capacity'])}, plate:"
                    f"{_number(config['heat_capacity'])}|{_number(config['heat_capacity'])}"
                ),
                "Thermal diffusivities": "0, 0",
                "Phase transition depths": (
                    f"background:{_reference_number(config['phase_transition_depth'], 660e3, '660e3')}, "
                    f"plate:{_reference_number(config['phase_transition_depth'], 660e3, '660e3')}"
                ),
                "Phase transition widths": (
                    f"background:{_reference_number(config['phase_transition_width'], 5e3, '5e3')}, "
                    f"plate:{_reference_number(config['phase_transition_width'], 5e3, '5e3')}"
                ),
                "Phase transition temperatures": "background:1573, plate:1573",
                "Phase transition Clapeyron slopes": "background:0, plate:0",
                "Viscous flow law": "diffusion",
                "Prefactors for diffusion creep": diffusion_prefactors,
                "Grain size exponents for diffusion creep": "background:0|0, plate:0|0",
                "Activation energies for diffusion creep": "background:0|0, plate:0|0",
                "Activation volumes for diffusion creep": "background:0|0, plate:0|0",
                "Minimum viscosity": _reference_number(
                    config["minimum_viscosity"], 1e17, "1e17"
                ),
                "Maximum viscosity": _reference_number(
                    config["maximum_viscosity"], 1e25, "1e25"
                ),
                "Reference strain rate": _number(config["reference_strain_rate"]),
                "Viscosity averaging scheme": "harmonic",
                "Yield mechanism": "drucker",
                "Angles of internal friction": (
                    f"background:0|0, plate:{degrees(phi):.10f}|{degrees(phi):.10f}"
                ),
                "Cohesions": (
                    "background:1e30|1e30, plate:"
                    + (
                        "4.0824829046e7|4.0824829046e7"
                        if config["yield_cohesion"] == 40e6
                        and config["friction_coefficient"] == 0.2
                        else f"{aspect_cohesion:.10e}|{aspect_cohesion:.10e}"
                    )
                ),
                "Maximum yield stress": "background:1e30|1e30, plate:1e30|1e30",
            },
        }


class StegmanSolverRule(Rule):
    """Configure nonlinear and Stokes solver controls."""

    requires = [
        "nonlinear_tolerance", "max_nonlinear_iterations", "first_time_step",
        "maximum_time_step", "maximum_relative_increase_in_time_step",
        "cfl_number", "linear_solver_failure_strategy", "stokes_solver_type",
        "gmres_solver_restart_length", "number_of_cheap_stokes_solver_steps",
        "use_full_a_block_as_preconditioner", "linear_solver_tolerance",
        "maximum_expensive_stokes_solver_steps",
    ]
    defaults = {
        "nonlinear_tolerance": 1e-3,
        "max_nonlinear_iterations": 50,
        "first_time_step": 1e3,
        "maximum_time_step": 1e5,
        "maximum_relative_increase_in_time_step": 1e4,
        "cfl_number": 0.5,
        "linear_solver_failure_strategy": "continue with nonlinear solver",
        "stokes_solver_type": "block GMG",
        "gmres_solver_restart_length": 100,
        "number_of_cheap_stokes_solver_steps": 60,
        "use_full_a_block_as_preconditioner": True,
        "linear_solver_tolerance": 1e-7,
        "maximum_expensive_stokes_solver_steps": 0,
    }

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict.update({
            "Nonlinear solver scheme": "single Advection, iterated Stokes",
            "Nonlinear solver tolerance": _reference_number(
                config["nonlinear_tolerance"], 1e-3, "1e-3"
            ),
            "Max nonlinear iterations": str(config["max_nonlinear_iterations"]),
            "Max nonlinear iterations in pre-refinement": "0",
            "Maximum first time step": _reference_number(
                config["first_time_step"], 1e3, "1e3"
            ),
            "Maximum time step": _reference_number(
                config["maximum_time_step"], 1e5, "1e5"
            ),
            "Maximum relative increase in time step": _reference_number(
                config["maximum_relative_increase_in_time_step"], 1e4, "1e4"
            ),
            "CFL number": _number(config["cfl_number"]),
            "Linear solver failure strategy": config["linear_solver_failure_strategy"],
            "Solver parameters": {
                "Stokes solver parameters": {
                    "Stokes solver type": config["stokes_solver_type"],
                    "GMRES solver restart length": str(
                        config["gmres_solver_restart_length"]
                    ),
                    "Number of cheap Stokes solver steps": str(
                        config["number_of_cheap_stokes_solver_steps"]
                    ),
                    "Use full A block as preconditioner": (
                        "true" if config["use_full_a_block_as_preconditioner"]
                        else "false"
                    ),
                    "Linear solver tolerance": _reference_number(
                        config["linear_solver_tolerance"], 1e-7, "1e-7"
                    ),
                    "Maximum number of expensive Stokes solver steps": str(
                        config["maximum_expensive_stokes_solver_steps"]
                    ),
                },
            },
        })


class StegmanPostprocessRule(Rule):
    """Configure diagnostics and visualization output."""

    requires = ["graphical_output_interval"]
    defaults = {"graphical_output_interval": 5e5}

    def apply(self, config, prm_dict, wb_dict, context):
        prm_dict["Postprocess"] = {
            "List of postprocessors": (
                "velocity statistics, composition statistics, material statistics, "
                "visualization"
            ),
            "Visualization": {
                "Time between graphical output": _reference_number(
                    config["graphical_output_interval"], 5e5, "5e5"
                ),
                "List of output variables": (
                    "material properties, named additional outputs, strain rate, stress"
                ),
                "Interpolate output": "true",
            },
        }


stegman_rules = [
    StegmanGeometryRule(),
    StegmanSlabRule(),
    StegmanPrescribedVelocityRule(),
    StegmanMaterialRule(),
    StegmanSolverRule(),
    StegmanPostprocessRule(),
]


def make_stegman_case(config=None, prm_dict=None, wb_dict=None):
    """Apply the Stegman rule set and return configured PRM and WB dictionaries."""
    config = {} if config is None else deepcopy(config)
    prm_dict = {} if prm_dict is None else deepcopy(prm_dict)
    wb_dict = {} if wb_dict is None else deepcopy(wb_dict)
    context, _, _ = RuleEngine(stegman_rules).apply_all(config, prm_dict, wb_dict)
    return prm_dict, wb_dict, context
