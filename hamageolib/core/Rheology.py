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
        "coh": 1,
        "constant-coh": 2
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

    def compute_stress_creep(self, rheology_params, options) -> float:
        """
        Compute stress using the modified power-law rheology equation.

        Args:
            rheology_params (NamedTuple): Rheology parameters from select_rheology_parameters().
            options (class CreepOptions): options for creep rheology

        Returns:
            float: Computed stress (Pa).
        """
        F_factor = compute_F_factor(rheology_params.experiment_flag, rheology_params.stress_exponent)
        print("A_modified: ", F_factor * rheology_params.pre_factor) # debug
        return compute_stress_vectorized(
            F_factor * rheology_params.pre_factor,
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

class CreepOptions:
    """
    Stores environmental conditions and computed stress for creep calculations.

    Attributes:
        strain_rate (float): Strain rate (1/s), default NaN.
        pressure (float): Pressure (Pa), default NaN.
        temperature (float): Temperature (K), default NaN.
        grain_size (float): Grain size (m), default NaN.
        cOH (float): H/10^6 Si, default NaN.
        stress (float): Computed stress (Pa), default NaN.
    """
    def __init__(self, strain_rate=np.nan, pressure=np.nan, temperature=np.nan, grain_size=np.nan, cOH=np.nan, stress=np.nan):
        self.strain_rate = strain_rate
        self.pressure = pressure
        self.temperature = temperature
        self.grain_size = grain_size
        self.cOH = cOH
        self.stress = stress  # Computed stress will be stored here


def compute_stress_vectorized(A, p, r, n, E, V, R, strain_rate, pressure, temperature, grain_size, cOH):
    """
    Compute stress for vectorized inputs using NumPy.
    """
    B = A * (grain_size ** (-p)) * (cOH ** r)
    stress = (strain_rate / B) ** (1.0 / n) * np.exp((E + pressure * V) / (n * R * temperature))
    return stress


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