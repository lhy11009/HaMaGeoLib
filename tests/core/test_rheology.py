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
from hamageolib.core.Rheology import RheologyModel, CreepOptions, compute_stress_vectorized, compute_SS_factors  # Adjust imports if needed

# Define test data directory
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../files/csv/")
TEST_CSV_FILE = os.path.join(TEST_DATA_DIR, "rheology.csv")

def test_RheologyModel_init_RheologyParams():
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


def test_RheologyModel_HK03():
    '''
    test RheologyModel and compute HK03 rheology
    '''
    # Compute directly
    stress1 = compute_stress_vectorized(1000000.0, 3.0, 1.0, 1.0, 335e3, 4e-6, 8.314, 7.8e-15, 1e9, 1673.15, 1e4, 1000.0)
    stress1_std = 0.2991116739021758
    tolerance = 1e-6
    assert(abs(stress1-stress1_std)/stress1_std < tolerance)

    # Using the RheologyModel
    # Diffusion Creep
    RM = RheologyModel(TEST_CSV_FILE)
    rheology_params_diff = RM.select_rheology_parameters("HK03_diffusion")

    strain_rate_factor, _ = compute_SS_factors(rheology_params_diff.experiment_flag)

    options = CreepOptions(
        strain_rate=7.8e-15/strain_rate_factor,
        pressure=1e9,
        temperature=1673.0,
        grain_size=1e4,
        cOH=1000.0
    )

    stress = RM.compute_stress_creep(rheology_params_diff, options)
    stress_std = 0.173
    tolerance = 1e-2
    assert(abs(stress-stress_std)/stress_std < tolerance)

    options.stress = stress

    strain_rate = RM.compute_strain_rate_creep(rheology_params_diff, options)
    strain_rate_std = 7.8e-15/strain_rate_factor
    tolerance = 1e-6
    assert(abs(strain_rate-strain_rate_std)/strain_rate_std < tolerance)

    # Dislocation Creep
    rheology_params_disl = RM.select_rheology_parameters("HK03_dislocation")
    
    options = CreepOptions(
        strain_rate=2.5e-12/strain_rate_factor,
        pressure=1e9,
        temperature=1673.0,
        grain_size=1e4,
        cOH=1000.0
    )

    stress = RM.compute_stress_creep(rheology_params_disl, options)
    stress_std = 0.173
    tolerance = 1e-2
    assert(abs(stress-stress_std)/stress_std < tolerance)
    
    options.stress = stress
    
    strain_rate = RM.compute_strain_rate_creep(rheology_params_disl, options)
    strain_rate_std = 2.5e-12/strain_rate_factor
    tolerance = 1e-6
    assert(abs(strain_rate-strain_rate_std)/strain_rate_std < tolerance)