from pathlib import Path
from shutil import rmtree
from .config_engine import Rule

class CasePathRule(Rule):
    """
    Normalize and resolve case-related paths.

    This rule ensures that path-related configuration entries are consistently
    interpreted and propagated into the parameter dictionaries used by ASPECT.

    Relative paths are expected to be resolved against case_path upstream.

    Required configuration parameters:
        wb_file (str):
            Path to the WorldBuilder file.
            May be absolute or relative.
            Default value: "case.wb"

        output_directory (str):
            Path to the directory where ASPECT outputs will be written.
            May be absolute or relative.
            Default value: "output"

        temporary_output_directory (str):
            Path to the repository to hold temporary outputs.
            It's absolute if present.
            Default value: None
    """

    requires = [
        "wb_file",
        "output_directory",
        "temporary_output_directory",
    ]

    defaults = {
        "wb_file": "case.wb",
        "output_directory": "output",
        "temporary_output_directory" : ""
    }

    provides = [
    ]

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to update configuration-dependent dictionaries.

        This method follows the standard Rule interface and is responsible for
        modifying the provided dictionaries in-place based on the rule's logic.

        Parameters:
            config (dict): Configuration dictionary containing user-provided and
                           defaulted values.
            prm_dict (dict): Dictionary representing parsed ASPECT .prm parameters
                             that may be modified in-place.
            wb_dict (dict): Dictionary representing parsed WorldBuilder configuration
                            that may be modified in-place.
            context (dict): Shared execution context passed between rules for
                            coordination or data sharing.

        Returns:
            None: This function modifies one or more dictionaries in-place and does
                  not return a value.
        """

        # Get the path of prm and wb files from configuration
        wb_file = Path(config.get("wb_file"))
        output_dir = Path(config.get("output_directory"))
        temporary_output_directory = config.get("temporary_output_directory")

        # Fix entries in the prm file using string representations of the paths
        prm_dict["World builder file"] = str(wb_file)
        prm_dict["Output directory"] = str(output_dir)

        # Create a temporary output directory for outputs
        if len(temporary_output_directory) > 0:
            temporary_output_directory = Path(temporary_output_directory)
            if temporary_output_directory.exists():
                rmtree(temporary_output_directory)
            temporary_output_directory.mkdir(parents=True)
        context["temporary_output_directory"] = str(temporary_output_directory)



class InitialStepRule(Rule):
    """
    Setup the case to only run the zeroth step and output the initial condition.

    This rule ensures fast-run of the zeroth step using trivial solver setup

    Required configuration parameters:
        n_trivial_step (int):
            Extend the run to trivial steps.
            This is helpful for softward like Paraview which has
            problem with the first outputs
            Default value: 0
    """

    requires = [
        "n_trivial_step",
    ]

    defaults = {
        "n_trivial_step": 0,
    }

    provides = [
    ]

    def apply(self, config, prm_dict, wb_dict, context):
        """
        Apply this rule to update configuration-dependent dictionaries.

        This method follows the standard Rule interface and is responsible for
        modifying the provided dictionaries in-place based on the rule's logic.

        Parameters:
            config (dict): Configuration dictionary containing user-provided and
                           defaulted values.
            prm_dict (dict): Dictionary representing parsed ASPECT .prm parameters
                             that may be modified in-place.
            wb_dict (dict): Dictionary representing parsed WorldBuilder configuration
                            that may be modified in-place.
            context (dict): Shared execution context passed between rules for
                            coordination or data sharing.

        Returns:
            None: This function modifies one or more dictionaries in-place and does
                  not return a value.
        """

        # Get number of trivial steps to generate
        n_trivial_step = config["n_trivial_step"]

        # Fix solver type
        prm_dict["Nonlinear solver scheme"] = "no Advection, no Stokes"

        # Fix start and end time
        prm_dict["Start time"] = "0"
        prm_dict["Maximum time step"] = "1"
        prm_dict["End time"] = f"{n_trivial_step}"

        # Fix the section of post-process
        if "Postprocess" in prm_dict:
            if "Visualization" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Visualization"]["Time between graphical output"] = "0"
            if "Particles" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Particles"]["Time between data output"] = "0"
            if "Depth average" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Depth average"]["Time between graphical output"] = "0"
            if "Topography" in prm_dict["Postprocess"]:
                prm_dict["Postprocess"]["Topography"]["Time between text output"] = "0"

        if "Linear solver failure strategy" in prm_dict:
            prm_dict["Linear solver failure strategy"] = "abort"
