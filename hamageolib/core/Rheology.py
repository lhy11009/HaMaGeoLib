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
                                             + ["mechanism_flag"] + ["unit_flag"] + ["wet_flag"])
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

        # Convert to ordered values and add flag options for mechanism, unit ... 
        ordered_values = selected_row.iloc[0].values.tolist() + \
              [self.MECHANISM_MAPPING.get(mechanism_str, 0), self.UNIT_MAPPING.get(unit_str, 0),\
               self.WET_MAPPING.get(wet_str, 0)]

        return self.RheologyParams(*ordered_values)  # Use pre-defined NamedTuple
