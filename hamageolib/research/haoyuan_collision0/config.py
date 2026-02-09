import numpy as np
from pathlib import Path
from ...utils.exception_handler import my_assert
from gdmate.aspect.config_engine import Rule, RuleConflictError
from gdmate.aspect.table import DepthAverageTable
from gdmate.aspect.io import parse_composition_entry, format_composition_entry, parse_entry_as_list, format_list_as_entry, \
    parse_isosurfaces_entry, format_isosurfaces_entry
from gdmate.aspect.prm_wb_utils import delete_composition_from_prm, duplicate_composition_from_prm, remove_composition_from_prm_recursive # todo_ct

# root of the package
package_root = Path(__file__).resolve().parents[3]


def CaseNameFromVariables(variables:dict, *, prefix="", use_all=True, use_keys=[]):

    # First add the prefix
    case_name = prefix

    # Geometry
    if use_all or "domain_depth" in use_keys:
        # As the first options, append _ if prefix is given
        if len(prefix) > 0:
            case_name += "_"
        case_name += "D%d" % int(variables["domain_depth"]/1e3)

    # viscosity
    if use_all or "viscosity_range" in use_keys:
        if not np.isclose(variables["viscosity_range"][0], 2.5e18, rtol=1e-6):
            case_name += "_minV%.1e" % variables["viscosity_range"][0]
        if not np.isclose(variables["viscosity_range"][1], 2.5e23, rtol=1e-6):
            case_name += "_maxV%.1e" % variables["viscosity_range"][1]
    
    # Weak layer
    if use_all or "weak_layer_compositions" in use_keys:
        if "gabbro" in variables["weak_layer_compositions"]:
            case_name += "_WLCG"
    if use_all or "weak_layer_viscosity" in use_keys:
        case_name += "_WLV%.1e" % int(variables["weak_layer_viscosity"])
    
    # Prescribe condition
    if use_all or "prescribe_subducting_plate_velocity" in use_keys:
        if variables["prescribe_subducting_plate_velocity"]:
            case_name += "_PC%.1e" % variables["convergence_rate"]
    
    # todo_ct
    # Continent
    if use_all or "add_continents" in use_keys:
        if variables["add_continents"] == "overriding":
            case_name += "_CTover"
        if variables["add_continents"] == "both":
            case_name += "_CTboth"

    return case_name


class RemovePeridotiteRule(Rule):
    """
    This rule customizes composition field such that one can choose
    to remove the peridotite compositin. User needs to specify
    remove_peridotite = True, otherwise, this class does nothing

    Required configuration parameters:
        remove_peridotite (bool) - whether to remove fluid compositions

    Provided configuration parameters:
        removed_peridotite_compositional_indexes (list) - list of compositional indexes removed
    """
    requires = ["remove_peridotite", "slab_layer_depths"]

    requires_comments = {"remove_peridotite": "Remove the peridotitle composition options and modify other compositional indices consistently",
                         "slab_layer_depths": "Layer depths of the compositions, start from 0 and has number of layer compositions + 1"}
    
    defaults = {"remove_peridotite": False, "slab_layer_depths": [0.0, 4e3, 7.5e3, 11.5e3]}
    
    provides = ["removed_peridotite_compositional_indexes"]

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to the configuration and parameter dictionaries.

        Parameters
        ----------
        config : dict
            Rule configuration dictionary containing validated values for all
            required configuration parameters.
        prm_dict : dict
            Deal.II/ASPECT-style nested parameter dictionary to be modified
            in-place by the rule.
        wb_dict : dict
            WorldBuilder configuration dictionary. This rule does not assume any
            specific structure but receives it for consistency with the Rule
            interface.
        context : dict
            Shared execution context passed between rules, used for coordination
            and state-sharing at the framework level.

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and
            does not return a value.
        """
        remove_peridotite = config["remove_peridotite"]
        slab_layer_depths = config["slab_layer_depths"]

        # initialize the provided variables to trivial initial values
        context["removed_peridotite_compositional_indexes"] = []

        if remove_peridotite:
            # todo_ct
            # Remove the composition and record the removed indexes
            remove_idx = delete_composition_from_prm(prm_dict, "peridotite")
            context["removed_peridotite_compositional_indices"] = sorted([remove_idx])

            # Recursively manage other entries related to compositions
            # remove_composition_from_prm_recursive(prm_dict, remove_compositions=["peridotite"])
            remove_composition_from_wb_recursive(wb_dict, context["removed_peridotite_compositional_indices"])

            # the peridotite mantle section is also removed
            features = wb_dict["features"]
            for i, feature in enumerate(features):
                if feature["name"] == "peridotite mantle":
                    features.pop(i)
                    break
            
            # Edit the slab to make sure the overiding composition is reset
            # To handle this, we add a composition model that replac e that
            # composition with 0.0
            if context["remove_fluid"]:
                index_diff = -2
            else:
                index_diff = 0
            for i, feature in enumerate(features):
                if feature["name"] == "Slab":
                    rm_ov_composition_model = \
                    {
                        "model": "uniform",
                        "compositions": [
                            5
                        ],
                        "fractions": [0.0],
                        "min distance slab top": slab_layer_depths[-1] + index_diff,
                        "max distance slab top": 150000.0,
                        "operation": "replace" 
                    }
                    
                    feature["composition models"].append(rm_ov_composition_model)
                
                    break
                




class RemoveFluidRule(Rule):
    """
    This rule customizes composition field such that one can choose
    to remove the composition related to fluids. User needs to specify
    remove_fluid = True, otherwise, this class does nothing

    Required configuration parameters:
        remove_fluid (bool) - whether to remove fluid compositions
        remove_fluid_compositions (list) - compositions to remove

    Provided configuration parameters:
        remove_fluid - whether fluid composition is removed
        removed_fluid_compositional_indexes (list) - list of compositional indexes removed

    
    """
    requires = ["remove_fluid", "remove_fluid_compositions"]
    
    defaults = {"remove_fluid": False, "remove_fluid_compositions":[]}

    requires_comments = {"remove_fluid": "Remove the fluid compositions and modify other compositional indices consistently",
                         "remove_fluid_compositions": "Specify which are the fluid compositions"}
    
    provides = ["remove_fluid", "removed_fluid_compositional_indexes"]

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to the configuration and parameter dictionaries.

        Parameters
        ----------
        config : dict
            Rule configuration dictionary containing validated values for all
            required configuration parameters.
        prm_dict : dict
            Deal.II/ASPECT-style nested parameter dictionary to be modified
            in-place by the rule.
        wb_dict : dict
            WorldBuilder configuration dictionary. This rule does not assume any
            specific structure but receives it for consistency with the Rule
            interface.
        context : dict
            Shared execution context passed between rules, used for coordination
            and state-sharing at the framework level.

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and
            does not return a value.
        """
        remove_fluid = config["remove_fluid"]
        remove_fluid_compositions = config["remove_fluid_compositions"]

        # initialize the provided variables to trivial initial values
        context["remove_fluid"] = remove_fluid
        context["removed_fluid_compositional_indexes"] = []

        if remove_fluid:
            # Directly manage entries in Compositional fields
            # For this, first figure out the indexes of the fields to remove
            # Then remove this from related fields
            names_of_fields = parse_entry_as_list(prm_dict["Compositional fields"]["Names of fields"])
            compositional_field_methods = parse_entry_as_list(prm_dict["Compositional fields"]["Compositional field methods"])
            relevant_compositions_in_worldbuilder = parse_entry_as_list(prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"])
            remove_idxs = []
            for composition in remove_fluid_compositions:
                # First, record the indexes to remove
                remove_idx = names_of_fields.index(composition)
                remove_idxs.append(remove_idx)
            for remove_idx in sorted(remove_idxs, reverse=True):
                # Second, remove them in reverse order
                names_of_fields.pop(remove_idx)
                compositional_field_methods.pop(remove_idx)
                relevant_compositions_in_worldbuilder.pop(remove_idx)
            number_of_compositional_fields = len(names_of_fields)

            prm_dict["Compositional fields"]["Number of fields"] = str(number_of_compositional_fields)
            prm_dict["Compositional fields"]["Names of fields"] = format_list_as_entry(names_of_fields)
            prm_dict["Compositional fields"]["Compositional field methods"] = format_list_as_entry(compositional_field_methods)
            prm_dict["Initial composition model"]["World builder"]["List of relevant compositions"] = format_list_as_entry(relevant_compositions_in_worldbuilder)

            # Record the removed indexes
            context["removed_fluid_compositional_indices"] = sorted(remove_idxs)

            # Reset the related entry in mesh refinement
            compositional_field_thresholds = parse_entry_as_list(prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"])
            for remove_idx in sorted(remove_idxs, reverse=True):
                compositional_field_thresholds.pop(remove_idx)
            prm_dict["Mesh refinement"]["Composition threshold"]["Compositional field thresholds"] = format_list_as_entry(compositional_field_thresholds)

            # Reset the related viscosity scheme
            prm_dict["Material model"]["Visco Plastic"]["Viscosity prefactor scheme"] = "none"
            try:
                prm_dict["Material model"]["Visco Plastic"].pop("Water fugacity exponents for diffusion creep")
            except KeyError:
                pass
            try:
                prm_dict["Material model"]["Visco Plastic"].pop("Water fugacity exponents for dislocation creep")
            except KeyError:
                pass
            try:
                prm_dict["Material model"]["Visco Plastic"].pop("Minimum mass fraction bound water content for fugacity")
            except KeyError:
                pass

            # Recursively manage other entries related to compositions
            remove_composition_from_prm_recursive(prm_dict, remove_compositions=remove_fluid_compositions)
            remove_composition_from_wb_recursive(wb_dict, context["removed_fluid_compositional_indices"])

            # the operation section is related to the fluid
            # compositions, and are thus removed
            features = wb_dict["features"]
            for feature in features:
                try:
                    composition_models = feature["composition models"]
                except KeyError:
                    pass
                else:
                    for composition_model in composition_models:
                        if "operation" in composition_model:
                            composition_model.pop("operation")


def remove_composition_from_wb_recursive(wb_dict, removed_indices):
    """
    Remove entries related to composition from the wb file
    wb_dict : dict
        WorldBuilder configuration dictionary. This rule does not assume any
        specific structure but receives it for consistency with the Rule
        interface.
    context : dict
        A dict of provided parameters passed to the rule or defined in previous
        substeps by the rule.
    """
    features = wb_dict["features"]

    # loop for every feature and modify the composition models
    for feature in features:
        try:
            composition_models = feature["composition models"]
        except KeyError:
            pass
        else:
            indices_to_remove = []
            for i, composition_model in enumerate(composition_models):
                compositions = composition_model["compositions"]
                if compositions[0] in removed_indices:
                    # In case this is the compsotion to remove, mark
                    # the composition model to be removed
                    indices_to_remove.append(i)
                else:
                    # In case this is not the composition to remove,
                    # modify the compositional index to account for
                    # the removed compositions
                    subtract_value = len(removed_indices)
                    for j, removed_idx in enumerate(removed_indices):
                        if compositions[0] < removed_idx:
                            subtract_value = j
                            break
                    compositions[0] -= subtract_value
            # Remove the marked composition model
            for i in sorted(indices_to_remove, reverse=True):
                composition_models.pop(i)



    

def expand_phase_in_composition_from_prm_recursive(prm_dict:dict, composition:str, from_index:int, to_index:int, *, force_expand_entries=[], force_expand_length=2):
    """
    expand phase entries of composition in the prm file
    prm_dict : dict
        Deal.II/ASPECT-style nested parameter dictionary to be modified
        in-place by the rule.
    compositions: str
        names of composition to expand the phases
    from_index: int
        index of phase to expand from
    to_index: int
        index of phase to expand to
    """
    for key, value in prm_dict.items():
        if isinstance(value, dict):
            # call function recursively in case value is dict
            expand_phase_in_composition_from_prm_recursive(value, composition, from_index, to_index,
                                                           force_expand_entries=force_expand_entries, force_expand_length=force_expand_length)
        else:
            my_assert(isinstance(value, str), TypeError, "value must be dict or str, get %s" % str(value))
            try:
                # look for entries with composition options
                comp_dict = parse_composition_entry(value)
            except ValueError:
                pass
            else:
                # expand phase configurations of the given composition
                try:
                    phase_configurations = comp_dict[composition]
                except KeyError:
                    pass
                else:
                    # if this object is a list, then it means the phases
                    # are explicit in the entries of the assigned composition
                    # and we should expand the existing phases
                    if isinstance(phase_configurations, list):
                        configuration = phase_configurations[from_index]
                        phase_configurations.insert(from_index, configuration)
                        prm_dict[key] = format_composition_entry(comp_dict)
                    # The force expand entries give it an option to expand
                    # a single value represents all phases to multiple entris
                    elif isinstance(phase_configurations, (str, float)):
                        if key in force_expand_entries:
                            comp_dict[composition] = [phase_configurations for i in range(force_expand_length)]
                            prm_dict[key] = format_composition_entry(comp_dict)


# todo_config
class PostProcessorRule(Rule):
    """
    For this class, I vary the configurations for post-processor.
    Here, I add the depth average postprocessors

    This rule customizes post-processing behavior by optionally injecting a
    predefined postprocessor setup and by harmonizing output intervals across
    multiple postprocessor sections. It is intended to provide a lightweight,
    configuration-driven way to control how frequently diagnostic outputs are
    written and whether additional diagnostics (such as depth averages) are enabled.

    Required configuration parameters:
    - use_my_setup_of_postprocess (bool):
      Controls whether a custom postprocessor configuration is injected.
      Default value: False
    - time_between_output (float):
      Time interval (in model time units) used to synchronize output frequency
      across supported postprocessors.
      Default value: 100000.0
    """

    requires = ["use_my_setup_of_postprocess", "time_between_output"]
    
    defaults = {"use_my_setup_of_postprocess": False, "time_between_output": 100e3}

    requires_comments = {"use_my_setup_of_postprocess": "Add depth_average plot",
                        "time_between_output": "Set time between output for all postprocess modules"}

    provides = []

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to the configuration and parameter dictionaries.

        Parameters
        ----------
        config : dict
            Rule configuration dictionary containing validated values for all
            required configuration parameters.
        prm_dict : dict
            Deal.II/ASPECT-style nested parameter dictionary to be modified
            in-place by the rule.
        wb_dict : dict
            WorldBuilder configuration dictionary. This rule does not assume any
            specific structure but receives it for consistency with the Rule
            interface.
        context : dict
            Shared execution context passed between rules, used for coordination
            and state-sharing at the framework level.

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and
            does not return a value.
        """

        # Get values of configurations
        use_my_setup_of_postprocess = config["use_my_setup_of_postprocess"]
        time_between_output = config["time_between_output"]

        # Things to add in my own setup of the postprocessors
        if use_my_setup_of_postprocess:
            # Inject a custom postprocessor setup when explicitly enabled
            prm_dict["Postprocess"]["List of postprocessors"] = "visualization, composition statistics, velocity statistics, particles, material statistics, depth average"
            prm_dict["Postprocess"]["Depth average"] = {
                "Number of zones": "50",
                "Output format": "txt",
                "Time between graphical output": "0"
            }
            # set up output variables in visualization
            prm_dict["Postprocess"]["Visualization"]["List of output variables"] = "material properties, strain rate, named additional outputs, stress second invariant, depth, heat flux map, nonadiabatic pressure, principal stress"
        else:
            pass

        # Fix the section of post-process
        if "Postprocess" in prm_dict:
            # Synchronize output intervals across supported postprocessor blocks
            if "Visualization" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Visualization"]["Time between graphical output"] = "%de3" % int(time_between_output/1e3)
            if "Particles" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Particles"]["Time between data output"] = "%de3" % int(time_between_output/1e3)
            if "Depth average" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Depth average"]["Time between graphical output"] = "%de3" % int(time_between_output/1e3)


class GeometryRule(Rule):
    """
    For this class, I vary the domain depth and figure out what would be the change
    in the mesh refinement scheme. The mesh refinement should be exactly the same as
    the template if the default values is used. The idea is to scale the "repetitions"
    variable with the domain size.

    Required configuration parameters:
    - domain_length (float):
      The horizontal length of the model domain in meters.
      Default value: 8700e3.

    - domain_depth (float):
      The vertical depth of the model domain in meters.
      Default value: 2900e3.

    - global_refinement (int):
      The base level of global mesh refinement applied at the start of the simulation.
      Default value: 6.

    - adaptive_refinement (int):
      The number of adaptive refinement levels applied on top of the global refinement.
      Default value: 4.

    - repetition_length (float):
      The target physical length scale (in meters) used to determine how many mesh
      repetitions are applied in each direction.
      Default value: 1450e3.

    - use_isosurfaces (bool):
      Flag indicating whether isosurfaces are intended to be used in the workflow.
      This rule modify the usage of composition threshold to isosurfaces in case this is True
      Default value: False.
    
    Provided configuration parameters:
    - use_isosurfaces (bool):
      Flag indicating whether isosurfaces are intended to be used in the workflow.
      This rule modify the usage of composition threshold to isosurfaces in case this is True
      Default value: False.
    """

    requires = ["domain_length", "domain_depth", "global_refinement", "adaptive_refinement", "repetition_length", "use_isosurfaces"]
    
    defaults = {
        "domain_length": 8700e3,
        "domain_depth": 2900e3,
        "repetition_length": 1450e3,
        "global_refinement": 6,
        "adaptive_refinement": 4,
        "use_isosurfaces": False
    }

    requires_comments = {
        "domain_length": "Length of the domain, also modifies the global refinement to maintain the cell size.",
        "global_refinement": "Global refinement before modified for the domain size change to maintain the cell size.",
        "use_isosurfaces": "Whether to replace composition threshold with isosurfaces."
        }

    provides = ["use_isosurfaces", "domain_depth", "domain_length"]

    def apply(self, config, prm_dict, wb_dict, context):

        # Get values of configurations
        domain_depth = config["domain_depth"]
        domain_length = config["domain_length"]
        repetition_length = config["repetition_length"]
        use_isosurfaces = config["use_isosurfaces"]
        default_repetition_length = self.defaults["repetition_length"]

        # Derive the number of repetition using the length of repetition
        n_repetition_x = int(np.round(domain_length/repetition_length))
        n_repetition_y = int(np.round(domain_depth/repetition_length))

        # Modify the mesh refinement scheme using a modifier
        # Each higher refinements with lower the cell size by 2
        # If repetition length decreases, we can use less refinements
        refinement_modifier = int(np.round(np.log(repetition_length/default_repetition_length) / np.log(2)))

        global_refinement = config["global_refinement"] + refinement_modifier
        adaptive_refinement = config["adaptive_refinement"]
        total_refinement = global_refinement + adaptive_refinement
        my_assert(global_refinement > 0, ValueError, "The global refinement has to be positive")
        my_assert(adaptive_refinement > 0, ValueError, "The adaptive refinement has to be positive")

        # Calculate bottom temperature
        bottom_T = self.calculate_bottom_T(domain_depth)

        # Apply to parameter setup
        prm_dict["Geometry model"]["Box"]["Y extent"] = "%de3" % int(domain_depth/1e3)
        prm_dict["Geometry model"]["Box"]["X repetitions"] = str(n_repetition_x)
        prm_dict["Geometry model"]["Box"]["Y repetitions"] = str(n_repetition_y)
        prm_dict["Mesh refinement"]["Initial global refinement"] = str(global_refinement)
        prm_dict["Mesh refinement"]["Initial adaptive refinement"] = str(adaptive_refinement)
        prm_dict["Mesh refinement"]["Minimum refinement function"]["Function constants"] = "litho_thickness=120e3, ymax=%de3" % int(domain_depth/1e3)
        prm_dict["Mesh refinement"]["Minimum refinement function"]["Function expression"] = rf"""if(y>(ymax-litho_thickness),{global_refinement}, \
                              if(y>=(ymax - 662e3) && y<=(ymax - 658e3), {global_refinement-1}, {global_refinement-2}))"""
        prm_dict["Mesh refinement"]["Maximum refinement function"]["Function constants"] = "xmax=5500e3, xmin=4000e3, ymax=%de3" % int(domain_depth/1e3)
        prm_dict["Mesh refinement"]["Maximum refinement function"]["Function expression"] = \
            rf"""if( x<=xmax && x>=xmin && y>=(ymax - 660e3), {total_refinement}, {total_refinement-2})"""
        prm_dict["Boundary temperature model"]["Constant"]["Boundary indicator to temperature mappings"] = f"bottom:{bottom_T}, top:273"

        context["domain_depth"] = domain_depth
        context["domain_length"] = domain_length

        # Change composition threshold to isosurfaces if requried.
        # At this, record this option in the context
        if use_isosurfaces:
            # recover from the context what are the compositions got removed
            removed_fluid_compositional_indices = context.get("removed_fluid_compositional_indices", [])
            mesh_refinement_strategies = parse_entry_as_list(prm_dict["Mesh refinement"]["Strategy"])
            # remove the composition threshold
            mesh_refinement_strategies.remove("composition threshold")
            prm_dict["Mesh refinement"].pop("Composition threshold")
            mesh_refinement_strategies.append("isosurfaces")
            # add isosurfaces
            isosurfaces_configuration = "max, max, sediment: 0.5 | 1.0; max-1, max, gabbro: 0.5 | 1.0; max-1, max, MORB: 0.5 | 1.0; max-2, max, overriding: 0.5 | 1.0"
            if removed_fluid_compositional_indices == []:
                # in case no composition is removed by other rules, add the configurations of porosity and bound_fluid
                isosurfaces_configuration = "max, max, porosity: 0.5 | 1.0; max, max, bound_fluid: 0.5 | 1.0; " + isosurfaces_configuration
            elif len(removed_fluid_compositional_indices) == 2:
                # in case porosity and bound_fluid are already removed
                isosurfaces_configuration = isosurfaces_configuration
            else:
                raise NotImplementedError("removed_fluid_compositional_indices has to be [] or [0, 1] in current implementation")
            isosurfaces_dict = {"Isosurfaces": isosurfaces_configuration}
            prm_dict["Mesh refinement"]["Strategy"] = format_list_as_entry(mesh_refinement_strategies)
            prm_dict["Mesh refinement"]["Isosurfaces"] = isosurfaces_dict

        context["use_isosurfaces"] = use_isosurfaces

    def calculate_bottom_T(self, depth):
        """
        Compute the bottom boundary temperature by interpolating from a depth-averaged profile.

        Parameters
        ----------
        depth : float
            The model domain depth (in meters) at which the temperature should be evaluated.

        Returns
        -------
        float
            The interpolated temperature value corresponding to the given depth.
        """

        da_file = package_root/"hamageolib/research/haoyuan_collision0/files/01112026/depth_average.txt"
        my_assert(da_file.is_file(), FileNotFoundError, "%s doesn't exist" % str(da_file))

        my_table = DepthAverageTable(da_file)

        # extract data at time 1
        profile = my_table.profile(time=1.0, field="temperature")

        # Interpolate temperature to the requested depth
        T = np.interp(depth, profile["depth"].to_numpy(), profile["temperature"].to_numpy())

        return T

# todo_slab
class SlabRule(Rule):
    """
    Configures slab-layer geometry based on composition-specific depth intervals.

    This rule assigns depth ranges to slab-related composition models in the
    WorldBuilder dictionary. It interprets user-provided slab layering information
    (compositions and corresponding depth boundaries) and applies these consistently
    to the relevant composition models in the WorldBuilder features. The rule also
    performs validation to ensure consistency with other rules and with the internal
    model configuration.

    Required configuration parameters:

    - slab_layer_compositions (list[str]):
      A list of composition names that define the ordered layers within the slab.
      Each entry corresponds to one slab layer.
      Default value: ["sediment", "MORB", "gabbro"].

    - slab_layer_depths (list[float]):
      A list of depth boundaries (in meters) defining the slab layering. This list
      must contain exactly one more element than ``slab_layer_compositions``, since
      each layer is bounded by a lower and upper depth.
      Default value: [0.0, 4e3, 7.5e3, 11.5e3].

    Provided configuration parameters:
    - spreading_velocity (float)
      The speed of ridge spreading rate (half spreading rate geologically)
    """
    
    requires = ["slab_layer_compositions", "slab_layer_depths", "plate_start_point", "slab_hinge_point", "slab_age"]
    
    defaults = {"slab_layer_compositions": ["sediment", "MORB", "gabbro"], "slab_layer_depths": [0.0, 4e3, 7.5e3, 11.5e3],
                "plate_start_point": 1000e3, "slab_hinge_point": 5000e3, "slab_age": 100e6}
    
    requires_comments = {"slab_layer_compositions": "Include these compositions for the slab",
                         "slab_layer_depths": "Layer depths of the compositions, start from 0 and has number of layer compositions + 1"}
    
    provides = ["spreading_velocity", "slab_hinge_point"]
    
    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply the rule to modify model configuration dictionaries in-place.

        Parameters
        ----------
        config : dict
            A dictionary of resolved configuration values for the current model case.
            This contains all parameters listed in ``requires`` and any additional
            configuration provided by the broader rule system.

        prm_dict : dict
            A nested dictionary representation of the ASPECT parameter file that this
            rule may modify in-place.

        wb_dict : dict
            A dictionary containing WorldBuilder-related configuration or data that
            rules may read from or modify in-place.

        context : dict
            A dictionary for passing shared contextual information between rules during
            execution (e.g., derived values, cached objects, or workflow state).

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and does
            not return a value.
        """

        # read the slab layer configurations
        slab_layer_compositions = config["slab_layer_compositions"]
        slab_layer_depths = config["slab_layer_depths"]
        plate_start_point = config["plate_start_point"]
        slab_hinge_point = config["slab_hinge_point"]
        slab_age = config["slab_age"]


        # compute the spreading rate based on the slab age, hinge point and starting point of the plate
        spreading_velocity = (slab_hinge_point - plate_start_point) / slab_age

        # assert that the layer depths has number of compositions + 1
        my_assert(len(slab_layer_compositions) + 1 == len(slab_layer_depths), ValueError,
                  "slab_layer_depths should be exactly one element longer than slab_layer_compositions.")

        # check no confliction with other rules
        if context["removed_peridotite_compositional_indexes"]:
            my_assert("peridotie" not in slab_layer_compositions, RuleConflictError, "If peridotite is removed by other rules, it cannot be included here")

        # get indices of the slab compositions
        names_of_fields = parse_entry_as_list(prm_dict["Compositional fields"]["Names of fields"])
        slab_composition_indices = [names_of_fields.index(composition) for composition in slab_layer_compositions]

        # loop features and apply the composition layer configurations for slab
        for feature in wb_dict["features"]:
            try:
                composition_models = feature["composition models"]
            except KeyError:
                pass
            else:
                for composition_model in composition_models:
                    try:
                        index = slab_composition_indices.index(composition_model["compositions"][0])
                    except ValueError:
                        pass
                    else:
                        # Update depth bounds depending on whether the model uses absolute depth
                        # or distance relative to slab top
                        if "min depth" in composition_model and "max depth" in composition_model:
                            composition_model["min depth"] = slab_layer_depths[index]
                            composition_model["max depth"] = slab_layer_depths[index+1]
                        elif "min distance slab top" in composition_model and "max distance slab top" in composition_model:
                            composition_model["min distance slab top"] = slab_layer_depths[index]
                            composition_model["max distance slab top"] = slab_layer_depths[index+1]
                        else:
                            raise NotImplementedError("This composition model is neither decribed by min/max depth nor described by min/max distance slab top")

        # configure the start point of the plage, age of the slab and the spreading rate
        for feature in wb_dict["features"]:
            if feature["name"] == "Subducting Plate":
                feature["coordinates"] = [[slab_hinge_point, -100e3], [slab_hinge_point, 100e3], [plate_start_point, 100e3], [plate_start_point, -100e3]]
                feature["temperature models"][0]["ridge coordinates"] = [[[plate_start_point,-100e3],[plate_start_point,100e3]]]
                feature["temperature models"][0]["spreading velocity"] = spreading_velocity

            if feature["name"] == "Slab":
                feature["coordinates"] = [[slab_hinge_point, -100e3],[slab_hinge_point, 100e3]]
                feature["temperature models"][0]["ridge coordinates"] = [[[plate_start_point,-100e3],[plate_start_point,100e3]]]
                feature["temperature models"][0]["subducting velocity"] = spreading_velocity
                feature["temperature models"][0]["spreading velocity"] = spreading_velocity

        # put the related parameters into context for later rules 
        context["spreading_velocity"] = spreading_velocity
        context["slab_hinge_point"] = slab_hinge_point

# todo_slab
class PrescribConditionRule(Rule):

    requires = ["prescribe_subducting_plate_velocity", "prescribe_subducting_plate_velocity_region_method",
                "prescribe_subducting_plate_velocity_velocity_method", "prescribe_subducting_plate_velocity_hinge_relative_distance",
                "prescribe_subducting_plate_velocity_length", "prescribe_subducting_plate_velocity_depth_range", "convergence_rate"]

    defaults = {"prescribe_subducting_plate_velocity": False, "prescribe_subducting_plate_velocity_region_method":"relative_to_hinge",
                "prescribe_subducting_plate_velocity_velocity_method":"spreading_velocity", "prescribe_subducting_plate_velocity_hinge_relative_distance": 1000e3,
                "prescribe_subducting_plate_velocity_length": 500e3, "prescribe_subducting_plate_velocity_depth_range": [10e3, 30e3], "convergence_rate":0.05}
    
    requires_comments = {"convergence_rate": "The overriding plate velocity is prescribed with the assigned subducting plate velocity to maintain this convergence rate."}

    provides = []
        
    def apply(self, config, prm_dict, wb_dict, context):

        prescribe_subducting_plate_velocity = config["prescribe_subducting_plate_velocity"]
        prescribe_subducting_plate_velocity_region_method = config["prescribe_subducting_plate_velocity_region_method"]
        prescribe_subducting_plate_velocity_velocity_method = config["prescribe_subducting_plate_velocity_velocity_method"]
        prescribe_subducting_plate_velocity_hinge_relative_distance = config["prescribe_subducting_plate_velocity_hinge_relative_distance"]
        prescribe_subducting_plate_velocity_length = config["prescribe_subducting_plate_velocity_length"]
        prescribe_subducting_plate_velocity_depth_range = config["prescribe_subducting_plate_velocity_depth_range"]
        convergence_rate = config["convergence_rate"]

        # get additional parameters from context
        try:
            domain_depth = context["domain_depth"]
        except KeyError:
            raise RuleConflictError("GeometryRule needs to be put before PrescribConditionRule so that there are domain parameters in the context")
        try:
            spreading_velocity = context["spreading_velocity"]
            slab_hinge_point = context["slab_hinge_point"]
        except KeyError:
            raise RuleConflictError("SlabRule needs to be put before PrescribConditionRule so that there are slab parameters in the context")
        
        # prescribe the velocity
        if prescribe_subducting_plate_velocity > 0:
            # figure out x1, x2, depth1 and depth2
            # note that depth1 should be deeper as it is the one has smaller y
            if prescribe_subducting_plate_velocity_region_method == "relative_to_hinge":
                x1, x2 = slab_hinge_point - prescribe_subducting_plate_velocity_hinge_relative_distance - prescribe_subducting_plate_velocity_length,\
                            slab_hinge_point - prescribe_subducting_plate_velocity_hinge_relative_distance
                x3, x4 = slab_hinge_point + prescribe_subducting_plate_velocity_hinge_relative_distance, \
                            slab_hinge_point + prescribe_subducting_plate_velocity_hinge_relative_distance + prescribe_subducting_plate_velocity_length
                depth1, depth2 = prescribe_subducting_plate_velocity_depth_range[1], prescribe_subducting_plate_velocity_depth_range[0]
            else:
                raise NotImplementedError("Now prescribe_subducting_plate_velocity_region_method could only be relative_to_hinge")

            if prescribe_subducting_plate_velocity_velocity_method == "spreading_velocity":
                velocity_constants_str = "vx=%.2f, vy=0.0, vx_ov=%.2f, vy_ov=0.0" % (spreading_velocity, spreading_velocity-convergence_rate)
            else:
                raise NotImplementedError("Now only the method of spreading_velocity method is implemented for the prescribe velocity condition")
            # check this section exists, if not, add it            
            try:
                prescribed_solution = prm_dict["Prescribed solution"]
            except KeyError:
                prm_dict["Prescribed solution"] = default_prescribed_solution_options
                prescribed_solution = prm_dict["Prescribed solution"]
                domain_constants_str = "x1=%de3, x2=%de3, x3=%de3, x4=%de3, y1=%de3, y2=%de3" % \
                    (int(x1/1e3), int(x2/1e3), int(x3/1e3), int(x4/1e3), int((domain_depth - depth1)/1e3), int((domain_depth - depth2)/1e3))
                prescribed_solution["Velocity function"]["Indicator function"]["Function constants"] = domain_constants_str
                prescribed_solution["Velocity function"]["Function"]["Function constants"] = domain_constants_str + ", " + velocity_constants_str


# The default options for prescribed solution of velocity
default_prescribed_solution_velocity_function_options = {
    "Indicator function":{
        "Variable names": "x, y",
        "Function constants": "x1=3000e3, x2=4000e4, x3=6000e3, x4=7000e3, y1=950e3, y2=980e3",
        "Function expression": "((x>x1)&&(x<x2)&&(y>y1)&&(y<y2))||((x>x3)&&(x<x4)&&(y>y1)&&(y<y2)) ? 1:0; ((x>x1)&&(x<x2)&&(y>y1)&&(y<y2))||((x>x3)&&(x<x4)&&(y>y1)&&(y<y2)) ? 1:0 "
    },
    "Function":{
        "Variable names": "x, y",
        "Function constants": "x1=3000e3, x2=4000e4, x3=6000e3, x4=7000e3, y1=950e3, y2=980e3, vx=0.04, vy=0.0, vx_ov=-0.01, vy_ov=0.0",
        "Function expression": "(x>x1)&&(x<x2)? vx:vx_ov; (x>x1)&&(x<x2)? vy:vy_ov"
    }   
}


# The default options for prescribed solution
default_prescribed_solution_options = {
    "List of model names": "velocity function",
    "Velocity function": default_prescribed_solution_velocity_function_options
}




class RheologyRule(Rule):
    """
    Controls whether a custom rheological setup is applied to the model configuration.

    This rule provides a simple switch that allows the user to enable or disable a
    predefined rheology configuration. When activated through the configuration,
    it modifies rheology-related entries in the parameter dictionary while otherwise
    leaving the template unchanged. This supports optional, user-controlled
    customization of rheological behavior without requiring changes to the base
    parameter setup.

    Required configuration parameters:

    - use_my_setup_of_rheology (bool):
      Flag indicating whether the custom rheology setup should be applied.
      Default value: False.
    - viscosity_range (list of 2):
      The range of viscosity in the model.
      Default value: [2.5e18, 2.5e23]
    """
    requires = ["use_my_setup_of_rheology", "viscosity_range", "use_safer_options"]
    
    defaults = {"use_my_setup_of_rheology": False, "viscosity_range": [2.5e18, 2.5e23], "use_safer_options": False}

    requires_comments = {"use_my_setup_of_rheology": "Set reference strain rate to 1e-15.", "viscosity_range": "Range of viscosity in the model",
                         "use_safer_options": "Use adiabatic pressure in rheology and also cutoff unrealistic temperatures"}

    provides = []

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply the rule to modify model configuration dictionaries in-place.

        Parameters
        ----------
        config : dict
            A dictionary of resolved configuration values for the current model case.
            This contains all parameters listed in ``requires`` and any additional
            configuration provided by the broader rule system.

        prm_dict : dict
            A nested dictionary representation of the ASPECT parameter file that this
            rule may modify in-place.

        wb_dict : dict
            A dictionary containing WorldBuilder-related configuration or data that
            rules may read from or modify in-place.

        context : dict
            A dictionary for passing shared contextual information between rules during
            execution (e.g., derived values, cached objects, or workflow state).

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and does
            not return a value.
        """

        # Get values of configurations
        use_my_setup_of_rheology = config["use_my_setup_of_rheology"]

        # In my setup, I use a reference strain rate of 1e-15 to start the model
        if use_my_setup_of_rheology:
            prm_dict["Material model"]["Visco Plastic"]["Reference strain rate"] = "1e-15"

        # Prescribe maximum and minimum viscosity
        # Here, just assign every phase to min and max values.
        # Later, the WeakLayerRule will further modify these by weak layer phases.
        viscosity_range = config["viscosity_range"]

        max_viscosity_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Maximum viscosity"])
        min_viscosity_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Minimum viscosity"])

        for key, value in min_viscosity_dict.items():
            min_viscosity_dict[key] = ["%.1e" % viscosity_range[0] for i in range(len(min_viscosity_dict[key]))]
        for key, value in max_viscosity_dict.items():
            max_viscosity_dict[key] = ["%.1e" % viscosity_range[1] for i in range(len(max_viscosity_dict[key]))]
    
        prm_dict["Material model"]["Visco Plastic"]["Maximum viscosity"] = format_composition_entry(max_viscosity_dict)
        prm_dict["Material model"]["Visco Plastic"]["Minimum viscosity"] = format_composition_entry(min_viscosity_dict)

        # Use adiabatic pressure in rheology and also cutoff unrealistic temperatures
        use_safer_options = config["use_safer_options"]
        if use_safer_options:
            prm_dict["Material model"]["Visco Plastic"]["Use adiabatic pressure in creep viscosity"] = "true"
            prm_dict["Material model"]["Visco Plastic"]["Minimum temperature for viscosity"] = "273.15"


class WeakLayerRule(Rule):
    """
    Configures the presence and properties of a weak layer in the model.

    This rule modifies material and mesh-related parameter entries to represent a
    mechanically weak layer associated with selected compositions. It adjusts phase
    transition parameters, constrains viscosities around a prescribed weak-layer
    value, and (optionally) enforces maximum mesh refinement for the weak layer when
    isosurfaces are used. The rule operates by updating the parameter dictionary in a
    controlled, composition-aware manner while leaving unrelated settings unchanged.

    Required configuration parameters:

    - weak_layer_compositions (list[str]):
      A list of composition names that should be treated as belonging to the weak
      layer.
      Default value: ["MORB", "sediment"].

    - weak_layer_viscosity (float):
      The target viscosity (in Pa·s) assigned to the weak layer. The rule enforces
      this value by slightly bounding the minimum and maximum viscosities around it.
      Default value: 2.5e19.

    - weak_layer_cutoff_depth (float):
      Depth (in meters) of the upper phase transition associated with the weak layer,
      used to modify phase transition depths for the selected compositions.
      Default value: 125e3.

    - weak_layer_transition_width (float):
      Width (in meters) of the phase transition zone for the weak layer compositions.
      Default value: 10e3.

    - force_weak_layer_max_refinement (bool):
      Flag indicating whether the mesh refinement level should be forced to the
      maximum for weak layer compositions when isosurfaces are enabled.
      Default value: False.
    """
    
    requires = ["weak_layer_compositions", "weak_layer_viscosity", "weak_layer_cutoff_depth", "weak_layer_transition_width", "force_weak_layer_max_refinement"]
    
    defaults = {"weak_layer_compositions": ["MORB", "sediment"], "weak_layer_viscosity": 2.5e19, "weak_layer_cutoff_depth": 125e3, 
                "weak_layer_transition_width":10e3, "force_weak_layer_max_refinement": False}
    
    requires_comments = {"weak_layer_compositions": "The weak layer is expand to these compositions. Phase transitions of them are set up consistently.",
                         "force_weak_layer_max_refinement": "Force all weak layer composition with the maximum refinement level with the isosurfaces."}

    provides = []
    
    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply the rule to modify model configuration dictionaries in-place.

        Parameters
        ----------
        config : dict
            A dictionary of resolved configuration values for the current model case.
            This contains all parameters listed in ``requires`` and any additional
            configuration provided by the broader rule system.

        prm_dict : dict
            A nested dictionary representation of the ASPECT parameter file that this
            rule may modify in-place.

        wb_dict : dict
            A dictionary containing WorldBuilder-related configuration or data that
            rules may read from or modify in-place.

        context : dict
            A dictionary for passing shared contextual information between rules during
            execution (e.g., derived values, cached objects, or workflow state).

        Returns
        -------
        None
            This method always modifies the provided dictionaries in-place and does
            not return a value.
        """

        # Get the compositions of weak layer and the assigned viscosity
        weak_layer_compositions = config["weak_layer_compositions"]
        weak_layer_viscosity = config["weak_layer_viscosity"]
        weak_layer_cutoff_depth = config["weak_layer_cutoff_depth"]
        weak_layer_transition_width = config["weak_layer_transition_width"]
        force_weak_layer_max_refinement = config["force_weak_layer_max_refinement"]

        # Expand and make a new phase if compositions are not from "MORB" or "sediment"
        for composition in weak_layer_compositions:
            if composition not in ["MORB", "sediment"]:
                expand_phase_in_composition_from_prm_recursive(prm_dict, composition, 0, 0,
                                                                force_expand_entries=["Phase transition depths", "Phase transition widths",\
                                                                                     "Phase transition temperatures", "Phase transition Clapeyron slopes"],
                                                                force_expand_length=2)

        # Set up the related phase transitions 
        PT_depth_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Phase transition depths"])
        PT_width_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Phase transition widths"])
        for composition in weak_layer_compositions:
            PT_depth_dict[composition][0] = "%.1f" % weak_layer_cutoff_depth
            PT_width_dict[composition][0] = "%.1f" % weak_layer_transition_width
        prm_dict["Material model"]["Visco Plastic"]["Phase transition depths"] = format_composition_entry(PT_depth_dict)
        prm_dict["Material model"]["Visco Plastic"]["Phase transition widths"] = format_composition_entry(PT_width_dict)

        # Make sure the refinement is set to max level for the weak layer compositions
        if force_weak_layer_max_refinement:
            if context["use_isosurfaces"]:
                isosurfaces = parse_isosurfaces_entry(prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"])
                for isosurface in isosurfaces:
                    if isosurface["composition"] in weak_layer_compositions:
                        isosurface["min"] = "max"
                        isosurface["max"] = "max"
                prm_dict["Mesh refinement"]["Isosurfaces"]["Isosurfaces"] = format_isosurfaces_entry(isosurfaces)
            else:
                raise NotImplementedError("The option of force_weak_layer_max_refinement is only implemented with use_isosurfaces")

        # Maximum viscosity is slightly larger than the assigned weak layer viscosity (by 0.4%)
        # Minimum viscosity is slightly smaller than the assigned weak layer viscosity (by 0.4%)
        max_viscosity_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Maximum viscosity"])
        min_viscosity_dict = parse_composition_entry(prm_dict["Material model"]["Visco Plastic"]["Minimum viscosity"])
        for composition in weak_layer_compositions:
            max_viscosity_dict[composition][0] = "%.3e" % (weak_layer_viscosity * 1.0004)
            min_viscosity_dict[composition][0] = "%.3e" % (weak_layer_viscosity * 0.9996)
        prm_dict["Material model"]["Visco Plastic"]["Maximum viscosity"] = format_composition_entry(max_viscosity_dict)
        prm_dict["Material model"]["Visco Plastic"]["Minimum viscosity"] = format_composition_entry(min_viscosity_dict)


class SolverRule(Rule):
    
    requires = ["stokes_solver_type", "skip_expensive_stokes", "max_nonlinear_iterations", "linear_solver_tolerance", "number_of_cheap_Stokes_solver_steps",
                "GMRES_solver_restart_length"]
    
    defaults = {"stokes_solver_type": "block AMG", "skip_expensive_stokes": False, "max_nonlinear_iterations": 70, "linear_solver_tolerance":5e-5,
                "number_of_cheap_Stokes_solver_steps":20000, "GMRES_solver_restart_length": 1000}
    
    requires_comments = {"skip_expensive_stokes": "Within a nonlinear solver, Skip the expensive stokes iteration even the cheap one fails and continue next linear iteration."}

    provides = []

    def apply(self, config, prm_dict, wb_dict, context):
        skip_expensive_stokes = config["skip_expensive_stokes"]
        max_nonlinear_iterations = config["max_nonlinear_iterations"]
        linear_solver_tolerance = config["linear_solver_tolerance"]
        number_of_cheap_Stokes_solver_steps = config["number_of_cheap_Stokes_solver_steps"]
        GMRES_solver_restart_length = config["GMRES_solver_restart_length"]
        stokes_solver_type = config["stokes_solver_type"]

        # configure max nonlinear interations
        prm_dict["Max nonlinear iterations"] = "%d" % max_nonlinear_iterations

        # configure linear solver scheme
        prm_dict["Solver parameters"]["Stokes solver parameters"]["Stokes solver type"] = stokes_solver_type
        prm_dict["Solver parameters"]["Stokes solver parameters"]["Linear solver tolerance"] = "%.0e" % linear_solver_tolerance
        prm_dict["Solver parameters"]["Stokes solver parameters"]["Number of cheap Stokes solver steps"] = "%d" % number_of_cheap_Stokes_solver_steps
        prm_dict["Solver parameters"]["Stokes solver parameters"]["GMRES solver restart length"] = "%d" % GMRES_solver_restart_length

        # configure solver to skip expensive stokes solves
        if skip_expensive_stokes:
            prm_dict["Solver parameters"]["Stokes solver parameters"]["Skip expensive stokes solver"] = "true"

# todo_ct
class ContinentRule(Rule): 

    requires = ["add_continents"]

    defaults = {"add_continents": "none"}

    requires_comments = {"add_continents": "Whether to add continents in the model. This option could be \"none\", \"overriding\", \"subducting\", \"both\""} 
    
    provides = []

    def apply(self, config, prm_dict, wb_dict, context):

        # handle the overriding plate
        if config["add_continents"] == "both" or config["add_continents"] == "overriding":
            pass
            name_upper = "crust_upper"
            name_lower = "crust_lower"
            duplicate_composition_from_prm(prm_dict, "gabbro", name_upper)