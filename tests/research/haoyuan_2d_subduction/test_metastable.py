## Test hamageolib/research/haoyuan_2d_subduction/metastable.py

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from hamageolib.research.haoyuan_2d_subduction.metastable import *


'''
tests for growth_rate_interface_P2
'''
def test_growth_rate_scalar_inputs():
    P = 1e7  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    T_eq = 1000  # Temperature in Kelvin
    Coh = 100  # Concentration in wt.ppm H2O

    pTKinetics = PTKinetics()
    result = pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    result_std = 7.058780930270711e-17
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

def test_growth_rate_scalar_inputs_1():
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    
    # here we first compute a P_eq and T_eq
    P = 1.5e10  # Pressure in Pascals
    T = 1173.15  # Temperature in Kelvin
    Coh = 100  # Concentration in wt.ppm H2O

 
    P_eq = PT410["P"] + (T - PT410["T"])*PT410["cl"]
    T_eq = PT410["T"] + (P - PT410["P"])/PT410["cl"]

    pTKinetics = PTKinetics()
    result = pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    result_std = 1.0170410597642686e-13
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

def test_growth_rate_scalar_inputs_bd():
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    
    # here we first compute a P_eq and T_eq
    # make the PT near the boundary
    P = 12366300000.0 + 1e9  # Pressure in Pascals
    T = 1173.15  # Temperature in Kelvin
    Coh = 150  # Concentration in wt.ppm H2O

 
    P_eq = PT410["P"] + (T - PT410["T"])*PT410["cl"]
    T_eq = PT410["T"] + (P - PT410["P"])/PT410["cl"]

    pTKinetics = PTKinetics()
    result = pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    result_std = 3.1191400335966567e-13
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

    # test 2: using the MO_KINETICS class and get the same result
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)
    result_1 = Mo_Kinetics.compute_Y(0.0)
    assert abs(result_1 - result_std) / result_std < 1e-6, "Check the value of Growth rate"


def test_growth_rate_array_inputs():
    P = np.array([1e7, 1.1e7, 1.2e7])  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    T_eq = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    Coh = 100  # Concentration in wt.ppm H2O

    pTKinetics = PTKinetics()
    result = pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    assert result.shape == P.shape, "Output should have the same shape as input pressures"
    assert np.all(result > 0), "Growth rate should be positive for all P > P_eq"

def test_pressure_not_above_equilibrium():
    P = 8e6  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    T_eq = 1000  # Temperature in Kelvin
    Coh = 100  # Concentration in wt.ppm H2O

    pTKinetics = PTKinetics()
    with pytest.raises(AssertionError):
        pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)

def test_invalid_pressure_type():
    P = "1e7"  # Invalid type
    T = 1000  # Temperature in Kelvin
    P_eq = 9e6  # Equilibrium pressure in Pascals
    T_eq = 1000  # Temperature in Kelvin
    Coh = 100  # Concentration in wt.ppm H2O

    pTKinetics = PTKinetics()
    with pytest.raises(TypeError):
        pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)

def test_edge_case_zero_temperature():
    P = 1e7  # Pressure in Pascals
    T = 0  # Absolute zero (not physically meaningful)
    P_eq = 9e6  # Equilibrium pressure in Pascals
    T_eq = 0  # Absolute zero (not physically meaningful)
    Coh = 100  # Concentration in wt.ppm H2O

    pTKinetics = PTKinetics()
    with pytest.raises(ZeroDivisionError):
        pTKinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)

'''
tests for 
'''
def test_nucleation_rate_scalar_inputs():
    P = 1.5e10  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    T_eq = 1000
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    assert abs(result - 903686.9915438003)/903686.9915438003 < 1e-6, "Nucleation rate should be positive for P > P_eq"
    assert isinstance(result, float), "Nucleation rate should be a float for scalar inputs"

def test_nucleation_rate_array_inputs():
    P = np.array([1.5e10, 1.6e10, 1.7e10])  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    T_eq = 1000
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    assert result.shape == P.shape, "Output should have the same shape as input pressures"
    assert np.all(result > 0), "Nucleation rate should be positive for all P > P_eq"

def test_nulceaion_rate_scalar_inputs_bd():
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    
    # here we first compute a P_eq and T_eq
    # make the PT near the boundary
    P = 12366300000.0 + 0.7e9  # Pressure in Pascals
    T = 1173.15  # Temperature in Kelvin
    Coh = 150  # Concentration in wt.ppm H2O

    P_eq = PT410["P"] + (T - PT410["T"])*PT410["cl"]
    T_eq = PT410["T"] + (P - PT410["P"])/PT410["cl"]

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    result_std = 9.256578901496377e-06
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

    # test 2: using the MO_KINETICS class and get the same result
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)
    result_1 = Mo_Kinetics.compute_Iv(0.0)
    assert abs(result_1 - result_std) / result_std < 1e-6, "Check the value of Growth rate"

def test_nulceaion_rate_scalar_inputs_critical_size():
    # check the value of critical size in nucleation
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    
    # here we first compute a P_eq and T_eq
    # make the PT near the boundary
    P = 12366300000.0 + 0.7e9  # Pressure in Pascals
    T = 1173.15  # Temperature in Kelvin
    Coh = 150  # Concentration in wt.ppm H2O

    P_eq = PT410["P"] + (T - PT410["T"])*PT410["cl"]
    T_eq = PT410["T"] + (P - PT410["P"])/PT410["cl"]

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    critical_radius = pTKinetics.critical_radius(P, T, P_eq, T_eq)

    result_std = 9.256578901496377e-06
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Nucleation rate"
    
    critical_radius_std = 2.197106690777577e-09
    assert abs(critical_radius - critical_radius_std) / critical_radius_std < 1e-6, "Check the value of Critical radius"

    # test 2: using the MO_KINETICS class and get the same result
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh, True)

    result_1 = Mo_Kinetics.compute_Iv(0.0)
    assert abs(result_1 - result_std) / result_std < 1e-6, "Check the value of Growth rate"
    
    critical_radius_1 = Mo_Kinetics.compute_rc(0.0)
    assert abs(critical_radius_1 - critical_radius_std) / critical_radius_std < 1e-6, "Check the value of Critical radius"



def test_nulceaion_rate_scalar_inputs_bd_big():
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    
    # here we first compute a P_eq and T_eq
    # make the value of nucleation big
    P = 14.75e9 # Pressure in Pascals
    T = 600 + 273.15  # Temperature in Kelvin
    Coh = 150  # Concentration in wt.ppm H2O

    P_eq = PT410["P"] + (T - PT410["T"])*PT410["cl"]
    T_eq = PT410["T"] + (P - PT410["P"])/PT410["cl"]

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    result_std = 6.2154552608671056e+19
    assert abs(result - result_std) / result_std < 1e-6, "Check the value of Growth rate"

    # test 2: using the MO_KINETICS class and get the same result
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)
    result_1 = Mo_Kinetics.compute_Iv(0.0)
    assert abs(result_1 - result_std) / result_std < 1e-6, "Check the value of Growth rate"


def test_pressure_not_above_equilibrium():
    P = 1.3e9  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    T_eq = 1000
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    with pytest.raises(AssertionError):
        pTKinetics.nucleation_rate(P, T, P_eq, T_eq)

def test_invalid_pressure_type():
    P = "1.5e9"  # Invalid type
    T = 1000  # Temperature in Kelvin
    T_eq = 1000
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    with pytest.raises(TypeError):
        pTKinetics.nucleation_rate(P, T, P_eq, T_eq)

def test_zero_pressure_difference():
    P = 1.4e9  # Pressure equal to equilibrium pressure
    T = 1000  # Temperature in Kelvin
    T_eq = 1000
    P_eq = 1.4e9  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    with pytest.raises(ZeroDivisionError):
        pTKinetics.nucleation_rate(P, T, P_eq, T_eq)

def test_nucleation_rate_high_temperature():
    P = 1.5e10  # Pressure in Pascals
    T = 3000  # Very high temperature in Kelvin
    T_eq = 3000
    P_eq = 1.4e10  # Equilibrium pressure in Pascals

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    assert result > 0, "Nucleation rate should be positive at high temperatures"

def test_nucleation_rate_low_temperature():
    P = 1.5e9  # Pressure in Pascals
    T = 1  # Extremely low temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals
    T_eq = 1

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    assert result == 0 or result < 1e-10, "Nucleation rate should approach zero at very low temperatures"

def test_edge_case_large_pressure_difference():
    P = 2.5e9  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e9  # Equilibrium pressure in Pascals
    T_eq = 1000

    pTKinetics = PTKinetics()
    result = pTKinetics.nucleation_rate(P, T, P_eq, T_eq)
    assert result > 0, "Nucleation rate should be positive for large pressure differences"

def test_avrami_number():
    '''
    Test of the Avrami number
    '''
    P = 1.5e10  # Pressure in Pascals
    T = 1000  # Temperature in Kelvin
    P_eq = 1.4e10  # Equilibrium pressure in Pascals
    T_eq = 1000

    d0 = 0.01 # m
    Coh = 1000.0
    
    Kinetics = PTKinetics()
    I_max = max(1e-50, Kinetics.nucleation_rate(P, T, P_eq, T_eq)) # per unit volume
    Y_max = max(1e-50, Kinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh))
    Av = calculate_avrami_number(I_max, Y_max)

    Av_expected = 1.2917411325502283e+32

    assert(abs(Av-Av_expected)/Av_expected < 1e-6)

def test_solve_modified_equations_eq18():
    '''
    test solve_modified_equations_eq18
    '''
    # set the condition
    P = 12.5e9 # Pa
    T = 623.75 # K
    Coh = 1000.0

    # equilibrium condition
    PT_dict = {"P": 14e9, "T": 1760.0, "cl": 4e6}
    P_eq = (T - PT_dict["T"]) * PT_dict["cl"] + PT_dict["P"]
    T_eq = (P - PT_dict["P"]) / PT_dict["cl"] + PT_dict["T"]

    # scaling parameters
    D = 100e3 # m
    d0 = 1e-2 # m
    kappa = 1e-6
    t_scale = D**2.0/kappa

    # compute parameters
    Kinetics = PTKinetics()
    I0 = Kinetics.nucleation_rate(P, T, P_eq, T_eq) # per unit volume
    Y0 = Kinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    Av = calculate_avrami_number(I0, Y0)

    Y_prime_func = lambda s: 1.0 
    I_prime_func = lambda s: 1.0
    rc_prime_func = lambda s: 0.0

    # set parameters for solution
    X_scale_array = np.array([I0**(3.0/4.0)*Y0**(-3.0/4.0), I0**(1.0/2.0)*Y0**(-1.0/2.0), I0**(1.0/4.0)*Y0**(-1.0/4.0), 1.0])

    s_span = np.array([0.0000e+00, 3.1536e+12]) / t_scale
    X = np.array([0., 0., 0., 0.])
    kwargs = {"n_span": 10}

    # compute the solution
    solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, rc_prime_func, s_span, X, **kwargs)
    X_new = solution_nd.y # * X_scale_array[:, np.newaxis]
    X_scaled_new = X_new * X_scale_array[:, np.newaxis]

    # compare with the expected value 
    X1 = X_new[:, -1]
    expected_X1 = np.array([0.10182481, 0.01036829, 0.00110558, 0.00011258]) 
    assert np.allclose(X1, expected_X1, rtol=1e-6), f"solution mismatch: {X1} != {expected_X1}"
    
    X_scaled1 = X_scaled_new[:, -1]
    expected_X_scaled1 = np.array([1.66641775e+24, 6.68334760e+14, 2.80693809e+05, 1.12575271e-04]) 
    assert np.allclose(X_scaled1, expected_X_scaled1, rtol=1e-6), f"solution mismatch: {X_scaled1} != {expected_X_scaled1}"

def test_solve_modified_equations_eq18_1():
    '''
    test solve_modified_equations_eq18
    Compared to the previous test, this is at a higher T
    '''
    # set the condition
    P = 14.75e9 # Pa
    T = 600 + 273.15 # K
    Coh = 1000.0

    # equilibrium condition
    PT_dict = {"P": 14e9, "T": 1760.0, "cl": 4e6}
    P_eq = (T - PT_dict["T"]) * PT_dict["cl"] + PT_dict["P"]
    T_eq = (P - PT_dict["P"]) / PT_dict["cl"] + PT_dict["T"]

    # scaling parameters
    D = 100e3 # m
    d0 = 1e-2 # m
    kappa = 1e-6
    t_scale = D**2.0/kappa

    # compute parameters
    Kinetics = PTKinetics()
    I0 = Kinetics.nucleation_rate(P, T, P_eq, T_eq) # per unit volume
    Y0 = Kinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    Av = calculate_avrami_number(I0, Y0)

    Y_prime_func = lambda s: 1.0 
    I_prime_func = lambda s: 1.0
    rc_prime_func = lambda s: 0.0

    # set parameters for solution
    X_scale_array = np.array([I0**(3.0/4.0)*Y0**(-3.0/4.0), I0**(1.0/2.0)*Y0**(-1.0/2.0), I0**(1.0/4.0)*Y0**(-1.0/4.0), 1.0])

    s_span = np.array([0.0000e+00, 3.1536e+12]) / t_scale
    X = np.array([0., 0., 0., 0.])
    kwargs = {"n_span": 10}

    # compute the solution
    solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, rc_prime_func, s_span, X, **kwargs)
    X_new = solution_nd.y # * X_scale_array[:, np.newaxis]

    # compare with the expected value 
    X1 = X_new[:, -1]
    expected_X1 = np.array([5.88659712e+06, 3.46520257e+13, 2.13609990e+20, 1.25743595e+27]) 
    assert np.allclose(X1, expected_X1, rtol=1e-6), f"solution mismatch: {X1} != {expected_X1}"

def test_solve_modified_equations_eq18_1S():
    '''
    test solve_modified_equations_eq18
    Compared to the previous test, this is at a higher T
    '''
    # set the condition
    P = 14.75e9 # Pa
    T = 600 + 273.15 # K
    Coh = 1000.0

    # equilibrium condition
    PT_dict = {"P": 14e9, "T": 1760.0, "cl": 4e6}
    P_eq = (T - PT_dict["T"]) * PT_dict["cl"] + PT_dict["P"]
    T_eq = (P - PT_dict["P"]) / PT_dict["cl"] + PT_dict["T"]

    # scaling parameters
    D = 100e3 # m
    d0 = 1e-2 # m
    kappa = 1e-6
    t_scale = D**2.0/kappa

    # compute parameters
    _constants = PTKinetics.Constants(
            R=8.31446,
            k=1.38e-23,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.6,
            fs=1e-3,
            K0=1e30,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=1
        )

    Kinetics = PTKinetics(_constants)
    I0 = Kinetics.nucleation_rate(P, T, P_eq, T_eq) # per unit volume
    Y0 = Kinetics.growth_rate_interface_P2(P, T, P_eq, T_eq, Coh)
    Av = calculate_avrami_number(I0, Y0)

    Y_prime_func = lambda s: 1.0 
    I_prime_func = lambda s: 1.0
    rc_prime_func = lambda s: 0.0

    # set parameters for solution
    X_scale_array = np.array([I0**(3.0/4.0)*Y0**(-3.0/4.0), I0**(1.0/2.0)*Y0**(-1.0/2.0), I0**(1.0/4.0)*Y0**(-1.0/4.0), 1.0])

    s_span = np.array([0.0000e+00, 3.1536e+12]) / t_scale
    X = np.array([0., 0., 0., 0.])
    kwargs = {"n_span": 10}

    # compute the solution
    solution_nd = solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, rc_prime_func, s_span, X, **kwargs)
    X_new = solution_nd.y # * X_scale_array[:, np.newaxis]

    # compare with the expected value 
    X1 = X_new[:, -1]
    expected_X1 = np.array([4.25883778e+04,1.81376992e+09,8.08913178e+13,3.44503000e+18]) 
    assert np.allclose(X1, expected_X1, rtol=1e-6), f"solution mismatch: {X1} != {expected_X1}"


'''
test functions for the MO_KINETICS class
'''
# Helper functions for testing
class MOCK_KINETICS():
    def __init__(self):
        self.growth_rate = mock_growth_rate
        self.nucleation_rate = mock_nucleation_rate
        self.critical_radius = mock_critical_size
        self.nucleation_type = 0

class MOCK_KINETICS_NORMALIZED():
    def __init__(self):
        self.growth_rate = mock_growth_rate_normalized
        self.nucleation_rate = mock_nucleation_rate_normalized
        self.critical_radius = mock_critical_size_normalized
        self.nucleation_type = 0
    


def mock_growth_rate(P, T, Peq, Teq, Coh):
    return (Peq - P) * 0.1

def mock_nucleation_rate(P, T, Peq, Teq):
    return (Peq - P) * 0.05

def mock_critical_size(P, T, Peq, Teq):
    return 0.0

def mock_growth_rate_normalized(P, T, Peq, Teq, Coh):
    return 1.0

def mock_nucleation_rate_normalized(P, T, Peq, Teq):
    return 1.0

def mock_critical_size_normalized(P, T, Peq, Teq):
    return 0.0

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
    pTKinetics = MOCK_KINETICS()
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(pTKinetics)
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    kinetics.set_kinetics_fixed(1.2, 1100.0, 0.5)

    Peq = compute_eq_P(kinetics.PT_eq, 1100.0)
    assert kinetics.Y_func(0) == mock_growth_rate(1.2, 1100.0, Peq, 0.0, 0.5)
    assert kinetics.I_func(0) == mock_nucleation_rate(1.2, 1100.0, Peq, 0.0)


def test_solve_modified_equation():
    """
    Test solving the modified equation.
    """
    pTKinetics = MOCK_KINETICS_NORMALIZED()
    kinetics = MO_KINETICS()
    kinetics.set_kinetics_model(pTKinetics)
    kinetics.set_PT_eq(1.0, 1000.0, 0.01)
    kinetics.set_kinetics_fixed(1.2e11, 1100.0, 0.5)
    
    t_span = np.array([0.0, 10.0])
    X_ini = np.array([0.0, 0.0, 0.0, 0.0])
    is_saturated = False
    
    X_array, is_saturated_array = kinetics.solve_modified_equation(t_span, X_ini, is_saturated, n_span=5)
    
    assert X_array.shape == (4, 6)
    assert len(is_saturated_array) == 6

def test_solve_values():
    """
    Test solving the kinetics equations over a time span and check the value
    """
    year = 3.15576e7  # seconds in a year
    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6}
    year = 365.0 * 24.0 * 3600.0  # Seconds in one year
    Coh = 1000.0 # wt.ppm H2O

    # Test parameters
    P = 15.75e9  # Pa
    T = 600 + 273.15  # K
    t_max = 10e6 * year  # s
    n_t = 100
    n_span = 10

    # Initialize the MO_KINETICS class
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

    # Solve the kinetics
    results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug=False)
    
    # "t" "N" "Dn" "S" "Vtilde" "V" "is_saturated"
    assert(results.shape == (1001, 7))

    # Expected result for results[10, :]
    expected_row = np.array([
        3.15360000e+12,  # Time
        4.14864390e+24,  # N
        5.27218305e+13,  # Dn
        7.01622359e+02,  # S
        2.61841251e+00,  # Dimensionless volume
        9.27081472e-01,  # Volume fraction
        1.00000000e+00   # Saturation status
    ])

    # Check results[10, :]
    actual_row = results[10, :]
    assert np.allclose(actual_row, expected_row, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"

def test_solve_values_un():
    """
    Test solving the kinetics equations over a time span and check the value
    Solve under a unsaturated condition, near the equilibrium condition
    """
    year = 3.15576e7  # seconds in a year
    PT410 = {"P": 14e9, "T": 1760.0, "cl": 4e6} # equilibrium phase transition for 410 km
    year = 365.0 * 24.0 * 3600.0  # Seconds in one year

    # Test parameters
    P = 10.255e9 + 1.25e9 # Pa
    T = 823.75 # K
    Coh = 150.0
    t_max = 10e6 * year  # s
    n_t = 100
    n_span = 10

    # Initialize the MO_KINETICS class
    _constants = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=1e-2,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.6,
            fs=1e-3,
            K0=1e30,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=1
        )

    Mo_Kinetics = MO_KINETICS(_constants)
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

    # Solve the kinetics
    results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug=False)
    
    # "t" "N" "Dn" "S" "Vtilde" "V" "is_saturated"
    assert(results.shape == (1001, 7))

    # Expected result for results[10, :]
    expected_row = np.array([
        3.15360000e+12,  # Time
        8.38648452e+13,  # N
        2.21027530e+08,  # Dn
        6.10016258e+02,  # S
        1.60771044e-03,  # Dimensionless volume
        1.60641876e-03,  # Volume fraction
        0.00000000e+00   # Saturation status
    ])

    # Check results[10, :]
    actual_row = results[10, :]
    assert np.allclose(actual_row, expected_row, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"

def test_solve_values_un_rc():
    """
    Test solving the kinetics equations over a time span and check the value
    Solve under a unsaturated condition, near the equilibrium condition.
    Also assume new nuclei forms with rc and compare the result to the previous test_solve_values_un test.
    """
    year = 3.15576e7  # seconds in a year
    PT410 = {"P": 14e9, "T": 1760.0, "cl": 4e6} # equilibrium phase transition for 410 km
    year = 365.0 * 24.0 * 3600.0  # Seconds in one year

    # Test parameters
    P = 10.255e9 + 1.25e9 # Pa
    T = 823.75 # K
    Coh = 150.0
    t_max = 10e6 * year  # s
    n_t = 100
    n_span = 10

    # Initialize the MO_KINETICS class
    _constants = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=1e-2,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.6,
            fs=1e-3,
            K0=1e30,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=1
        )

    Mo_Kinetics = MO_KINETICS(_constants)
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh, True)

    # Solve the kinetics
    results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug=False)

    # "t" "N" "Dn" "S" "Vtilde" "V" "is_saturated"
    assert(results.shape == (1001, 7))

    # Expected result for results[10, :]
    expected_row = np.array([
        3.15360000e+12,  # Time
        8.38648452e+13,  # N
        2.21233901e+08,  # Dn
        6.10871006e+02,  # S
        1.61071475e-03,  # Dimensionless volume
        1.60941824e-03,  # Volume fraction
        0.00000000e+00   # Saturation status
    ])

    # Check results[10, :]
    actual_row = results[10, :]
    assert np.allclose(actual_row, expected_row, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"


def test_solve_values_un_derivative():
    """
    Test solving the kinetics equations over a time span and check the value
    Solve under a unsaturated condition, near the equilibrium condition
    """
    year = 3.15576e7  # seconds in a year
    PT410 = {"P": 14e9, "T": 1760.0, "cl": 4e6} # equilibrium phase transition for 410 km
    year = 365.0 * 24.0 * 3600.0  # Seconds in one year

    # Test parameters
    P = 10.255e9 + 1.25e9 # Pa
    T = 823.75 # K
    Coh = 150.0
    t_max = 10e6 * year  # s
    n_t = 100
    n_span = 10

    # Initialize the MO_KINETICS class
    _constants = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=1e-2,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.6,
            fs=1e-3,
            K0=1e30,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=1
        )

    Mo_Kinetics = MO_KINETICS(_constants, include_derivative=True)
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

    # Solve the kinetics
    results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug=False)
    
    # "t" "N" "Dn" "S" "Vtilde" "V" "is_saturated"
    assert(results.shape == (1001, 8))

    # Expected result for results[10, :]
    expected_row = np.array([
        3.15360000e+12,  # Time
        8.38648452e+13,  # N
        2.21027530e+08,  # Dn
        6.10016258e+02,  # S
        1.60771044e-03,  # Dimensionless volume
        1.60641876e-03,  # Volume fraction
        0.00000000e+00,   # Saturation status
        1.75087542e-15  # Derivative
    ])

    # Check results[10, :]
    actual_row = results[10, :]
    assert np.allclose(actual_row, expected_row, atol=0.0, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"

    # Expected result for results[11, :]
    expected_row = np.array([
        3.46896000e+12,  # Time
        9.22513298e+13,  # N
        2.67443312e+08,  # Dn
        8.11931640e+02,  # S
        2.35384885e-03,  # Dimensionless volume
        2.35108072e-03,  # Volume fraction
        0.00000000e+00,   # Saturation status
        2.36130758e-15  # Derivative
    ])

    actual_row = results[11, :]
    assert np.allclose(actual_row, expected_row, atol=0.0, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"


def test_solve_values_low_T():
    """
    Test solving the kinetics equations over a time span and check the value
    """
    year = 3.15576e7  # seconds in a year
    PT410 = {"P": 14e9, "T": 1760.0, "cl": 4e6} # equilibrium phase transition for 410 km
    year = 365.0 * 24.0 * 3600.0  # Seconds in one year
    Coh = 1000.0 # wt.ppm H2O

    # Test parameters
    P = 9e9  # Pa
    T = 273.15  # K
    t_max = 10e6 * year  # s
    n_t = 100
    n_span = 10

    # Initialize the MO_KINETICS class
    Mo_Kinetics = MO_KINETICS()
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)
    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

    # Solve the kinetics
    results = Mo_Kinetics.solve(P, T, 0.0, t_max, n_t, n_span, debug=False)
    
    # "t" "N" "Dn" "S" "Vtilde" "V" "is_saturated"
    assert(results.shape == (1001, 7))

    # Expected result for results[10, :]
    expected_row = np.array([
        3.15360000e+12,  # Time
        0.0,  # N
        0.00000000e+00,  # Dn
        0.00000000e+00,  # S
        0.00000000e+00,  # Dimensionless volume
        0.00000000e+00,  # Volume fraction
        0.00000000e+00   # Saturation status
    ])

    # Check results[10, :]
    actual_row = results[10, :]
    assert np.allclose(actual_row, expected_row, rtol=1e-6), f"Row 10 mismatch: {actual_row} != {expected_row}"


def test_solve_profile():
    '''
    Test solution along a slab internal profile
    '''
    nucleation_type = 1

    PT410 = {"P": 13.5e9, "T": 1740.0, "cl": 2e6} # equlibrium conditions
    
    Coh = 150.0 # wt% for methods with mo kinetics
    d_ol = 1e-2 # m background grain size for methods with mo kinetics


    # intiate the class
    _constants1 = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=d_ol,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.46,
            fs=6e-4,
            K0=1e30,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=1
        )
    Mo_Kinetics = MO_KINETICS(_constants1)

    Mo_Kinetics.set_PT_eq(PT410['P'], PT410['T'], PT410['cl'])
    Mo_Kinetics.link_and_set_kinetics_model(PTKinetics)

    # Load the numpy dataset
    ifile = "tests/fixtures/research/haoyuan_metastable_subduction/foo_contour_data.txt"
    assert(os.path.isfile(ifile))

    data = np.loadtxt(ifile, skiprows=1)

    foo_contour_Ps = data[:, 0]
    foo_contour_Ts = data[:, 1]
    foo_contour_ts = data[:, 2]

    # set metastable contents along the profile
    # first scheme: n_span = 10
    n_t = 1; n_span = 10 # kinetic parameters
    foo_contents_wl_mo = np.zeros(foo_contour_Ps.size)
    for i in range(foo_contour_Ps.size-1):
        # parse variables:
        # P, T
        # t0, t1 - start and end of the time step
        P = foo_contour_Ps[i]
        T = foo_contour_Ts[i]
        t0 = foo_contour_ts[i]
        t1 = foo_contour_ts[i+1]
        Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

        # solve the ODEs
        if i == 0:
            _initial = None
        else:
            _initial = results[-1, :]
        
        results = Mo_Kinetics.solve(P, T, t0, t1, n_t, n_span, initial=_initial)
            
        foo_contents_wl_mo[i+1] = results[-1, 5]

    result_std = 0.6225033807716868
    assert(abs((foo_contents_wl_mo[257]-result_std)/result_std) < 1e-6)
    
    # second scheme: n_span = 20
    n_t = 1; n_span = 20 # kinetic parameters
    foo_contents_wl_mo1 = np.zeros(foo_contour_Ps.size)
    for i in range(foo_contour_Ps.size-1):
        # parse variables:
        # P, T
        # t0, t1 - start and end of the time step
        P = foo_contour_Ps[i]
        T = foo_contour_Ts[i]
        t0 = foo_contour_ts[i]
        t1 = foo_contour_ts[i+1]
        Mo_Kinetics.set_kinetics_fixed(P, T, Coh)

        # solve the ODEs
        if i == 0:
            _initial = None
        else:
            _initial = results[-1, :]
        
        results = Mo_Kinetics.solve(P, T, t0, t1, n_t, n_span, initial=_initial)
            
        foo_contents_wl_mo1[i+1] = results[-1, 5]

    result_std = 0.622758932311021
    assert(abs((foo_contents_wl_mo1[257]-result_std)/result_std) < 1e-6)