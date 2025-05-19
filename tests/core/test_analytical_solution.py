"""
Test Suite for Analytical Solution Module (hamageolib/core/AnalyticalSolution.py)

MIT License

Author: Haoyuan Li
Affiliation: UC Davis, EPS Department
Email: hylli@ucdavis.edu

Description:
    This test file is for unit testing the `Analytical Solution` module in `hamageolib/core/AnalyticalSolution.py`.
"""

import os
import pytest
import numpy as np
from hamageolib.core.AnalyticalSolution import *
    
year = 365 * 24 * 3600.0 # s in year

# todo_smooth
def test_wk2004_default():
    '''
    Test the default variable values in England & Wilkins 2004
    Check temperature at a given depth (e.g. 100 km)
    '''
    constants = WK2004.Constants(
        age=100.0e6*year,            # s
        U=0.1/year,                # m/s
        theta_d=np.pi/6.0, # radians
        a=100e3,               # meters
        zw=60e3,
        Ta=1280.0 + 273.15,             # Kelvin
        Tsf=273.15,
        kappa=8e-7,
        rho=3300.0,
        cp=1e3, # J Kg^-1 K^-1
        plate_T_model=0
    )
    
    WK_model = WK2004(constants)
    
    # Check value shallower than depth of the overidding plate thickness 
    depth = 40e3
    surface_temperature = WK_model.ss_temperature(depth, use_top_thickness=True, debug=True)
    surface_temperature_std = 300.16806615732156
    assert(np.isclose(surface_temperature, surface_temperature_std, 1e-6))

    # Check value deeper than depth of the overidding plate thickness 
    depth = 90e3
    surface_temperature = WK_model.ss_temperature(depth, use_top_thickness=True, debug=True)
    surface_temperature_std = 700.494909371
    assert(np.isclose(surface_temperature, surface_temperature_std, 1e-6))

    pass


# todo_EW
def test_wk2004_bd_temperature():
    '''
    Test calculation of temperature within slab surface boundary layer
    '''
    constants = WK2004.Constants(
        age=100.0e6*year,            # s
        U=0.1/year,                # m/s
        theta_d=np.pi/6.0, # radians
        a=100e3,               # meters
        zw=60e3,
        Ta=1280.0 + 273.15,             # Kelvin
        Tsf=273.15,
        kappa=8e-7,
        rho=3300.0,
        cp=1e3, # J Kg^-1 K^-1
        plate_T_model=0
    )
    
    WK_model = WK2004(constants)
    
    # Check value deeper than depth of the overidding plate thickness 
    # bd temperature at 10 km on top of the slab surface
    depth = 100e3
    y = 10e3
    bd_temperature = WK_model.bd_temperature_by_zw(depth, y)
    bd_temperature_std = 1090.3207095851174
    assert(np.isclose(bd_temperature, bd_temperature_std, 1e-6))

def test_single_age_above_cap():
    # 200 Ma in seconds
    age_s = 200e6 * 365 * 24 * 3600
    result = plate_thickness_from_age(age_s)
    assert np.isclose(result, 125e3, rtol=1e-5)

def test_array_input_mixed():
    ages_ma = np.array([1, 50, 200])  # in Ma
    ages_s = ages_ma * 365 * 24 * 3600 * 1e6
    expected = 11e3 * np.sqrt(ages_ma)
    expected = np.minimum(expected, 125e3)
    result = plate_thickness_from_age(ages_s)
    assert np.allclose(result, expected, rtol=1e-5)