"""
Rheology Module - HaMaGeoLib

MIT License

Author: Haoyuan Li
Affiliation: UC Davis, EPS Department
Email: hylli@ucdavis.edu

Overview:
    This module provides rheological models for geodynamic simulations, including 
    viscosity laws, deformation mechanisms, and stress-strain relationships.

Classes:
    - `RheologyModel`: A base class for different rheological models.
    - `PowerLawRheology`: Implements a power-law viscosity model.
    - `NewtonianRheology`: Implements a Newtonian viscosity model.

Functions:
    - `compute_viscosity`: General function to compute viscosity for a given model.
    - `strain_rate_tensor`: Computes strain rate tensor from velocity gradients.
"""

import os
import numpy as np
import pandas as pd
from collections import namedtuple

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import os
import pandas as pd
from collections import namedtuple

class RheologyModel:
    """
    Base class for rheology models.

    Attributes:
        gas_constant (float): Universal gas constant (J/mol/K).
        rheology_data (pd.DataFrame): Dataframe containing rheology parameters.
    """

    # Define mapping of mechanism strings to integer flags
    MECHANISM_MAPPING = {
        "dislocation": 1,
        "diffusion": 2,
        "peierls": 3
    }
    
    UNIT_MAPPING = {
        "p-MPa_d-mum": 1
    }
    
    WET_MAPPING = {
        "wet": 1,
        "constant-coh": 2,
        "dry": 3,
    }
    
    EXPERIMENT_MAPPING = {
        "axial_compression": 1, 
        "simple_shear": 2
    }

    def __init__(self, csv_file: str = os.path.join(package_root, "files/csv/rheology.csv")):
        """
        Initialize the rheology model and load rheology parameters from a CSV file.

        Args:
            csv_file (str, optional): Path to the CSV file containing rheology parameters.
                                      Defaults to "files/csv/rheology.csv".
        """
        self.gas_constant = 8.314  # J/(mol*K)
        
        # Load rheology data from CSV
        try:
            self.rheology_data = pd.read_csv(csv_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Rheology parameter file not found: {csv_file}")
        
        # Define NamedTuple only once
        if not self.rheology_data.empty:
            self.RheologyParams = namedtuple("RheologyParams", list(self.rheology_data.columns)\
                                             + ["mechanism_flag"] + ["unit_flag"] + ["wet_flag"]\
                                                + ["experiment_flag"])
        else:
            raise FileNotFoundError(f"Rheology parameter is empty: {csv_file}")
        
    def set_trivial_rheology_parameters(self):
        '''
        Return a set of rheology parameters with trivial values.
        '''
        # todo_vary
        # Initialize an object with NaN/default values
        trivial_rheology_params = self.RheologyParams(
            name=None,              # String fields set to None
            target=None,
            mechanism=None,
            experiment=None,
            unit=None,
            unit_water=None,
            wet=None,
            pre_factor=1.0,      # Numerical fields set to NaN
            grain_size_exponent=0.0,
            water_fugacity_exponent=0.0,
            stress_exponent=0.0,
            activation_energy=0.0,
            activation_volume=0.0,
            mechanism_flag=0,       # Flags default to 0 (or NaN if preferred)
            unit_flag=0,
            wet_flag=0,
            experiment_flag=0
        )

        return trivial_rheology_params
    
    def select_rheology_parameters(self, model_name: str):
        """
        Select rheology parameters for a given model name.
        """
        if "name" not in self.rheology_data.columns:
            raise ValueError("CSV file must contain a 'name' column to identify models.")

        #
        selected_row = self.rheology_data[self.rheology_data["name"] == model_name]
        
        if selected_row.empty:
            raise ValueError(f"Rheology model '{model_name}' not found in data.")

        # Get the string options for mechanism, unit ...
        mechanism_str = selected_row.iloc[0]["mechanism"].lower()
        unit_str = selected_row.iloc[0]["unit"].lower()
        wet_str = selected_row.iloc[0]["wet"].lower()
        experiment_str = selected_row.iloc[0]["experiment"].lower()

        # Convert to ordered values and add flag options for mechanism, unit ... 
        ordered_values = selected_row.iloc[0].values.tolist() + \
              [self.MECHANISM_MAPPING.get(mechanism_str, 0), self.UNIT_MAPPING.get(unit_str, 0),\
               self.WET_MAPPING.get(wet_str, 0), self.EXPERIMENT_MAPPING.get(experiment_str, 0)]

        return self.RheologyParams(*ordered_values)  # Use pre-defined NamedTuple
    
    def vary_rheology_parameters(self, rheology_params, rheology_params_variations, options):
        '''
        Vary the rheology parameters and refit the prefactor given by experimental options
        Args:
            rheology_params (NamedTuple): Rheology parameters.
            rheology_params_variations (NamedTuple): Rheology parameters variations
            options (class CreepOptions): Options for creep rheology.
        '''
        # todo_vary

        rheology_params_new = rheology_params
        
        correction_factor = self.compute_strain_rate_creep(rheology_params_variations, options)

        rheology_params_new = rheology_params_new._replace(pre_factor=rheology_params.pre_factor / correction_factor,\
                                        grain_size_exponent=rheology_params.grain_size_exponent + rheology_params_variations.grain_size_exponent,
                                        water_fugacity_exponent=rheology_params.water_fugacity_exponent + rheology_params_variations.water_fugacity_exponent,
                                        stress_exponent=rheology_params.stress_exponent + rheology_params_variations.stress_exponent,
                                        activation_energy=rheology_params.activation_energy + rheology_params_variations.activation_energy,
                                        activation_volume=rheology_params.activation_volume + rheology_params_variations.activation_volume)
        
        return rheology_params_new

    def compute_stress_creep(self, rheology_params, options, **kwargs) -> float:
        """
        Compute stress using the modified power-law rheology equation.
    
        Args:
            rheology_params (NamedTuple): Rheology parameters from select_rheology_parameters().
            options (class CreepOptions): Options for creep rheology.
            **kwargs: Additional keyword arguments (e.g., debug=True to print variables).
    
        Returns:
            float: Computed stress (Pa or MPa, depending on the unit of A).
        """
        assert(not np.isnan(options.strain_rate))

        F_factor = compute_F_factor(rheology_params.experiment_flag, rheology_params.stress_exponent)
        
        # Compute the modified pre-exponential factor
        A_modified = F_factor * rheology_params.pre_factor
 
        # Debug mode: Print related variables from `options`
        if kwargs.get("debug", False):
            print("DEBUG MODE: Computing stress")
            print(f"  A_modified: {A_modified:.2e}")
            print(f"  Strain Rate: {options.strain_rate:.2e} s^-1")
            print(f"  Pressure: {options.pressure:.2e} Pa")
            print(f"  Temperature: {options.temperature:.2f} K")
            print(f"  Grain Size: {options.grain_size:.2e} m")
            print(f"  cOH (Water Fugacity/Cohesion): {options.cOH:.2e}")

        if rheology_params.wet_flag == 3:
            # dry
            return compute_stress_vectorized_dry(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.strain_rate,
                options.pressure,
                options.temperature,
                options.grain_size
            )
        else:
            # wet
            return compute_stress_vectorized(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.water_fugacity_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.strain_rate,
                options.pressure,
                options.temperature,
                options.grain_size,
                options.cOH
            )

    def compute_strain_rate_creep(self, rheology_params, options, **kwargs) -> float:
        """
        Compute strain rate using the modified power-law rheology equation.
    
        Args:
            rheology_params (NamedTuple): Rheology parameters from select_rheology_parameters().
            options (class CreepOptions): Options for creep rheology.
            **kwargs: Additional keyword arguments (e.g., debug=True to print variables).
    
        Returns:
            float: Computed strain rate (s^-1).
        """
        F_factor = compute_F_factor(rheology_params.experiment_flag, rheology_params.stress_exponent)
        
        # Compute the modified pre-exponential factor
        A_modified = F_factor * rheology_params.pre_factor
        
        assert(not np.isnan(options.stress))
    
        # Debug mode: Print related variables from `options`
        if kwargs.get("debug", False):
            print("DEBUG MODE: Computing strain rate")
            print("  A_modified: ", A_modified, "\n")
            print("  Stress: ", options.stress, "\n")
            print("  Pressure: ", options.pressure,"\n")
            print("  Temperature: ", options.temperature, "\n")
            print("  Grain Size: ", options.grain_size, "\n")
            print("  cOH (Water Fugacity/Cohesion): ", options.cOH, "\n")
    
        if rheology_params.wet_flag == 3:
            # dry
            return compute_strain_rate_vectorized_dry(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.stress,
                options.pressure,
                options.temperature,
                options.grain_size
            )
        else:
            # wet
            return compute_strain_rate_vectorized(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.water_fugacity_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.stress,
                options.pressure,
                options.temperature,
                options.grain_size,
                options.cOH
            )
    
    def compute_viscosity_creep(self, rheology_params, options, **kwargs) -> float:
        """
        Compute stress using the modified power-law rheology equation.
    
        Args:
            rheology_params (NamedTuple): Rheology parameters from select_rheology_parameters().
            options (class CreepOptions): Options for creep rheology.
            **kwargs: Additional keyword arguments (e.g., debug=True to print variables).
    
        Returns:
            float: Computed viscosity (Pa s).
        """
        if type(options.strain_rate) == np.ndarray:
            assert not np.isnan(options.strain_rate).any(), "strain_rate contains NaN"
        else:
            assert not np.isnan(options.strain_rate), "strain_rate is NaN"

        F_factor = compute_F_factor(rheology_params.experiment_flag, rheology_params.stress_exponent)
        unit_factor = compute_unit_factor(rheology_params.unit_flag, rheology_params.stress_exponent, rheology_params.grain_size_exponent)
        
        # Compute the modified pre-exponential factor
        A_modified = F_factor *  unit_factor * rheology_params.pre_factor
    
        # Debug mode: Print related variables from `options`
        if kwargs.get("debug", False):
            print("DEBUG MODE: Computing stress")
            print(f"  A_modified: {A_modified:.2e}")
            print(f"  Strain Rate: {options.strain_rate:.2e} s^-1")
            print(f"  Pressure: {options.pressure:.2e} Pa")
            print(f"  Temperature: {options.temperature:.2f} K")
            print(f"  Grain Size: {options.grain_size:.2e} m")
            print(f"  cOH (Water Fugacity/Cohesion): {options.cOH:.2e}")
    
        if rheology_params.wet_flag == 3:
            # dry
            return compute_viscosity_vectorized_dry(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.strain_rate,
                options.pressure,
                options.temperature,
                options.grain_size)
        else:
            # wet
            return compute_viscosity_vectorized(
                A_modified,
                rheology_params.grain_size_exponent,
                rheology_params.water_fugacity_exponent,
                rheology_params.stress_exponent,
                rheology_params.activation_energy,
                rheology_params.activation_volume,
                self.gas_constant,  # Universal gas constant
                options.strain_rate,
                options.pressure,
                options.temperature,
                options.grain_size,
                options.cOH
        )


# Define the namedtuple class with fields
CreepOptions = namedtuple("CreepOptions", [
    "strain_rate",
    "pressure",
    "temperature",
    "grain_size",
    "cOH",
    "stress"
])


def compute_stress_vectorized(A, p, r, n, E, V, R, strain_rate, pressure, temperature, grain_size, cOH):
    """
    Compute stress for vectorized inputs using NumPy.
    
    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        r (float): Water fugacity exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        strain_rate (float or np.ndarray): Stress (s^-1).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).
        cOH (float or np.ndarray): Water fugacity or cohesion.
    """
    B = A * (grain_size ** (-p)) * (cOH ** r)
    stress = (strain_rate / B) ** (1.0 / n) * np.exp((E + pressure * V) / (n * R * temperature))
    return stress


def compute_strain_rate_vectorized(A, p, r, n, E, V, R, stress, pressure, temperature, grain_size, cOH):
    """
    Compute strain rate for vectorized inputs using NumPy.

    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        r (float): Water fugacity exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        stress (float or np.ndarray): Stress (Pa or MPa, depending on unit in A).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).
        cOH (float or np.ndarray): Water fugacity or cohesion.

    Returns:
        float or np.ndarray: Computed strain rate (1/s).
    """
    # Compute rheological prefactor B
    B = A * (grain_size ** (-p)) * (cOH ** r)

    # Compute strain rate
    strain_rate = B * (stress ** n) * np.exp(-(E + pressure * V) / (R * temperature))

    return strain_rate

def compute_viscosity_vectorized(A, p, r, n, E, V, R, strain_rate, pressure, temperature, grain_size, cOH):
    """
    Compute viscosity (η) using the rheology equation.

    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        r (float): Water fugacity exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        strain_rate (float or np.ndarray): Strain rate (1/s).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).
        cOH (float or np.ndarray): Water fugacity or cohesion.

    Returns:
        float or np.ndarray: Computed viscosity (Pa·s).
    """
    # Compute rheological prefactor B
    B = A * (grain_size ** (-p)) * (cOH ** r)

    # Compute viscosity η
    eta = (1 / 2.0) * (strain_rate ** (1.0 / n - 1)) * (B ** (-1.0 / n)) \
          * np.exp((E + pressure * V) / (n * R * temperature)) * 1e6  # Convert to MPa·s
    
    return eta

def compute_stress_vectorized_dry(A, p, n, E, V, R, strain_rate, pressure, temperature, grain_size):
    """
    Compute stress for vectorized inputs using NumPy.
    
    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        strain_rate (float or np.ndarray): Stress (s^-1).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).
    """
    B = A * (grain_size ** (-p))
    stress = (strain_rate / B) ** (1.0 / n) * np.exp((E + pressure * V) / (n * R * temperature))
    return stress


def compute_strain_rate_vectorized_dry(A, p, n, E, V, R, stress, pressure, temperature, grain_size):
    """
    Compute strain rate for vectorized inputs using NumPy.

    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        stress (float or np.ndarray): Stress (Pa or MPa, depending on unit in A).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).

    Returns:
        float or np.ndarray: Computed strain rate (1/s).
    """
    # Compute rheological prefactor B
    B = A * (grain_size ** (-p))

    # Compute strain rate
    strain_rate = B * (stress ** n) * np.exp(-(E + pressure * V) / (R * temperature))

    return strain_rate

def compute_viscosity_vectorized_dry(A, p, n, E, V, R, strain_rate, pressure, temperature, grain_size):
    """
    Compute viscosity (η) using the rheology equation.

    Args:
        A (float): Pre-exponential factor.
        p (float): Grain size exponent.
        n (float): Stress exponent.
        E (float): Activation energy (J/mol).
        V (float): Activation volume (m³/mol).
        R (float): Universal gas constant (J/mol/K).
        strain_rate (float or np.ndarray): Strain rate (1/s).
        pressure (float or np.ndarray): Pressure (Pa).
        temperature (float or np.ndarray): Temperature (K).
        grain_size (float or np.ndarray): Grain size (m).

    Returns:
        float or np.ndarray: Computed viscosity (Pa·s).
    """
    # Compute rheological prefactor B
    B = A * (grain_size ** (-p))

    # Compute viscosity η
    eta = (1 / 2.0) * (strain_rate ** (1.0 / n - 1)) * (B ** (-1.0 / n)) \
          * np.exp((E + pressure * V) / (n * R * temperature)) * 1e6  # Convert to MPa·s
    
    return eta


def compute_SS_factors(experiment_flag: int) -> float:
    """
    Compute the strain rate factor and the stress factor based on the experiment_flag and stress exponent.
    strain_rate_factor = dot_gamma / dot_epsilon_ii
    stress_factor = sigma_d / sigma_ii;

    Args:
        experiment_flag (int): Integer flag indicating the experiment type.
        stress_exponent (float): Stress exponent (n) from rheology parameters.

    Returns:
        float, float: strain_rate_factor, stress_factor.
    """
    if experiment_flag == 0:  # Already calibrated
        return 1.0, 1.0
    elif experiment_flag == 1:  # Axial compression
        return 2 / 3.0**0.5, 3.0**0.5
    elif experiment_flag == 2:  # Simple shear
        return 2.0, 2.0
    else:
        raise ValueError(f"Unknown experiment_flag: {experiment_flag}")


def compute_F_factor(experiment_flag: int, stress_exponent: float) -> float:
    """
    Compute the F_factor based on the experiment_flag and stress exponent.

    Args:
        experiment_flag (int): Integer flag indicating the experiment type.
        stress_exponent (float): Stress exponent (n) from rheology parameters.

    Returns:
        float: Computed F_factor.
    """
    if experiment_flag == 0:  # Already calibrated
        return 1.0
    elif experiment_flag == 1:  # Axial compression
        return 3.0**((stress_exponent + 1.0) / 2.0) / 2.0
    elif experiment_flag == 2:  # Simple shear
        return 2.0**(stress_exponent - 1.0)  # Common correction factor for shear experiments
    else:
        raise ValueError(f"Unknown experiment_flag: {experiment_flag}")


def compute_unit_factor(unit_flag: int, stress_exponent: float, grain_size_exponent: float) -> float:
    """
    Compute the F_factor based on the experiment_flag and stress exponent.

    Args:
        experiment_flag (int): Integer flag indicating the experiment type.
        stress_exponent (float): Stress exponent (n) from rheology parameters.
        grain_size_exponent (float): grain_size_exponent (p) from rheology parameters.

    Returns:
        float: Computed F_factor.
    """
    if unit_flag == 0:  # Already calibrated
        return 1.0
    elif unit_flag == 1:  # um^p * mPa^(-n)
        return 10**(-6*(stress_exponent + grain_size_exponent))
    else:
        raise ValueError(f"Unknown experiment_flag: {unit_flag}")