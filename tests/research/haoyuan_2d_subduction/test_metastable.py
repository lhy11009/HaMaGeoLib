## Test hamageolib/research/haoyuan_2d_subduction/metastable.py

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from hamageolib.research.haoyuan_2d_subduction.metastable import *


'''
tests for growth_rate_hosoya_06_eq2
'''
def test_growth_rate_scalar_inputs():
    P = 1e7  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    result = growth_rate_hosoya_06_eq2(P, T, P_eq, Coh)
    result_std = 5.3613444537140843e-17
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

def test_growth_rate_array_inputs():
    P = np.array([1e7, 1.1e7, 1.2e7])  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    result = growth_rate_hosoya_06_eq2(P, T, P_eq, Coh)
    assert result.shape == P.shape, "Output should have the same shape as input pressures"
    assert np.all(result > 0), "Growth rate should be positive for all P > P_eq"

def test_pressure_not_above_equilibrium():
    P = 8e6  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    with pytest.raises(AssertionError):
        growth_rate_hosoya_06_eq2(P, T, P_eq, Coh)

def test_invalid_pressure_type():
    P = "1e7"  # Invalid type
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    with pytest.raises(TypeError):
        growth_rate_hosoya_06_eq2(P, T, P_eq, Coh)

def test_edge_case_zero_temperature():
    P = 1e7  # Pressure in Pascals
    T = 0  # Absolute zero (not physically meaningful)
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    with pytest.raises(ZeroDivisionError):
        growth_rate_hosoya_06_eq2(P, T, P_eq, Coh)

'''
tests for nucleation_rate_yoshioka_2015
'''
def test_nucleation_rate_scalar_inputs():
    P = 1.5e10  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    result = nucleation_rate_yoshioka_2015(P, T, P_eq)
    assert result > 0, "Nucleation rate should be positive for P > P_eq"
    assert isinstance(result, float), "Nucleation rate should be a float for scalar inputs"

def test_nucleation_rate_array_inputs():
    P = np.array([1.5e10, 1.6e10, 1.7e10])  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    result = nucleation_rate_yoshioka_2015(P, T, P_eq)
    assert result.shape == P.shape, "Output should have the same shape as input pressures"
    assert np.all(result > 0), "Nucleation rate should be positive for all P > P_eq"

def test_pressure_not_above_equilibrium():
    P = 1.3e9  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    with pytest.raises(AssertionError):
        nucleation_rate_yoshioka_2015(P, T, P_eq)

def test_invalid_pressure_type():
    P = "1.5e9"  # Invalid type
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    with pytest.raises(TypeError):
        nucleation_rate_yoshioka_2015(P, T, P_eq)

def test_zero_pressure_difference():
    P = 1.4e9  # Pressure equal to equilibrium pressure
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    with pytest.raises(ZeroDivisionError):
        nucleation_rate_yoshioka_2015(P, T, P_eq)

def test_nucleation_rate_high_temperature():
    P = 1.5e10  # Pressure in Pascals
    T = 3000  # Very high temperature in Kelvin
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    result = nucleation_rate_yoshioka_2015(P, T, P_eq)
    assert result > 0, "Nucleation rate should be positive at high temperatures"

def test_nucleation_rate_low_temperature():
    P = 1.5e9  # Pressure in Pascals
    T = 1  # Extremely low temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    result = nucleation_rate_yoshioka_2015(P, T, P_eq)
    assert result == 0 or result < 1e-10, "Nucleation rate should approach zero at very low temperatures"

def test_edge_case_large_pressure_difference():
    P = 2.5e9  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    result = nucleation_rate_yoshioka_2015(P, T, P_eq)
    assert result > 0, "Nucleation rate should be positive for large pressure differences"

'''
test functions for the MO_KINETICS class
'''
# Helper functions for testing
def mock_growth_rate(P, T, Peq, Coh):
    return (Peq - P) * 0.1

def mock_nucleation_rate(P, T, Peq):
    return (Peq - P) * 0.05

def mock_growth_rate_normalized(P, T, Peq, Coh):
    return 1.0

def mock_nucleation_rate_normalized(P, T, Peq):
    return 1.0

def test_initialization():
    """
    Test the initialization of MO_KINETICS attributes.
    """
    kinetics = MO_KINETICS()
    assert kinetics.Y_func_ori is None
    assert kinetics.I_func_ori is None
    assert kinetics.Y_func is None
    assert kinetics.I_func is None
    assert kinetics.kappa == 1e-6
    assert kinetics.D == 100e3
    assert kinetics.d0 == 1e-2
    assert kinetics.PT_eq == {"T": None, "P": None, "cl": None}

def test_set_kinetics_model():
    """
    Test setting the kinetics model functions.
    """
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(mock_growth_rate, mock_nucleation_rate)
    assert kinetics.Y_func_ori == mock_growth_rate
    assert kinetics.I_func_ori == mock_nucleation_rate

def test_set_PT_eq():
    """
    Test setting the phase transformation equilibrium parameters.
    """
    kinetics = MO_KINETICS()
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    assert kinetics.PT_eq == {"T": 1000.0, "P": 1.0, "cl": 0.01}

def test_compute_eq_P():
    """
    Test equilibrium pressure calculation.
    """
    PT_eq = {"T": 1000.0, "P": 1.0, "cl": 0.01}
    T = 1100.0
    expected_P_eq = (T - PT_eq["T"]) * PT_eq["cl"] + PT_eq["P"]
    assert compute_eq_P(PT_eq, T) == pytest.approx(expected_P_eq)

def test_compute_eq_T():
    """
    Test equilibrium temperature calculation.
    """
    PT_eq = {"T": 1000.0, "P": 1.0, "cl": 0.01}
    P = 1.5
    expected_T_eq = (P - PT_eq["P"]) / PT_eq["cl"] + PT_eq["T"]
    assert compute_eq_T(PT_eq, P) == pytest.approx(expected_T_eq)

def test_set_kinetics_fixed():
    """
    Test fixing the kinetics model based on pressure, temperature, and cohesion.
    """
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(mock_growth_rate, mock_nucleation_rate)
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    kinetics.set_kinetics_fixed(1.2, 1100.0, 0.5)

    Peq = compute_eq_P(kinetics.PT_eq, 1100.0)
    assert kinetics.Y_func(0) == mock_growth_rate(1.2, 1100.0, Peq, 0.5)
    assert kinetics.I_func(0) == mock_nucleation_rate(1.2, 1100.0, Peq)

def test_solve_modified_equation():
    """
    Test solving the modified equation.
    """
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(mock_growth_rate_normalized, mock_nucleation_rate_normalized)
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    kinetics.set_kinetics_fixed(1.2, 1100.0, 0.5)
    
    t_span = np.array([0.0, 10.0])
    X_ini = np.array([0.0, 0.0, 0.0, 0.0])
    is_saturated = False
    
    X_array, is_saturated_array = kinetics.solve_modified_equation(t_span, X_ini, is_saturated, n_span=5)
    
    assert X_array.shape == (4, 5)
    assert len(is_saturated_array) == 5

def test_solve():
    """
    Test solving the kinetics equations over a time span.
    """
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(mock_growth_rate_normalized, mock_nucleation_rate_normalized)
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    kinetics.set_kinetics_fixed(1.2, 1100.0, 0.5)
    
    P = 1.5
    P_eq = 1.0
    t_max = 100.0
    n_t = 10
    n_span = 5
    
    results = kinetics.solve(P, P_eq, t_max, n_t, n_span)
    
    assert not results.empty
    assert "t" in results.columns
    assert "N" in results.columns
    assert "Dn" in results.columns
    assert "S" in results.columns
    assert "Vtilde" in results.columns
    assert "V" in results.columns
    assert "is_saturated" in results.columns