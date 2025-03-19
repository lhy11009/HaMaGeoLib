"""
Test Suite for Rheology Module (hamageolib/core/Rheology.py)

MIT License

Author: Haoyuan Li
Affiliation: UC Davis, EPS Department
Email: hylli@ucdavis.edu

Description:
    This test file is for unit testing the `Rheology` module in `hamageolib/core/Rheology.py`.
"""

import os
import pytest
import pandas as pd
from hamageolib.core.Rheology import RheologyModel  # Adjust imports if needed

# Define test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../files/csv/")
TEST_CSV_FILE = os.path.join(TEST_DATA_DIR, "rheology.csv")

def test_HK03():
    '''
    test RheologyModel.__init__();
    test contents of RheologyParams from function select_rheology_parameters
    '''
    RM = RheologyModel(TEST_CSV_FILE)
    rheology_params = RM.select_rheology_parameters("HK03_diffusion")

    # Assert expected values from actual output
    assert rheology_params.name == "HK03_diffusion"
    assert rheology_params.mechanism == "diffusion"
    assert rheology_params.unit == "p-MPa_d-mum"
    assert rheology_params.wet == "constant-coh"
    
    # Assert numerical parameters
    assert rheology_params.pre_factor == 1000000.0
    assert rheology_params.grain_size_exponent == 3.0
    assert rheology_params.water_fugacity_exponent == 1.0
    assert rheology_params.stress_exponent == 1.0
    assert rheology_params.activation_energy == 335000
    assert rheology_params.activation_volume == 4e-06

    # Assert flag values
    assert rheology_params.mechanism_flag == 2  # Expected from MECHANISM_MAPPING
    assert rheology_params.unit_flag == 0       # Expected from UNIT_MAPPING
    assert rheology_params.wet_flag == 2        # Expected from WET_MAPPING