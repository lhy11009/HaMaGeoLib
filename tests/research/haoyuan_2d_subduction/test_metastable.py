## Test hamageolib/research/haoyuan_2d_subduction/metastable.py

import pytest
import numpy as np
from scipy.integrate import solve_ivp
from hamageolib.research.haoyuan_2d_subduction.metastable import *

# Dummy functions for Y(t) and I(t)
def Y_dummy(t):
    return 1.0

def I_dummy(t):
    return 10.0

# Helper function for a mock Avrami number calculation
def calculate_avrami_number_yoshioka_2015(I_max, Y_max):
    return (I_max * Y_max)**(1/4)

# Mock function for saturation sigma calculation
def calculate_sigma_s(I_prime_array, Y_prime_array, d0, kappa, D):
    return 0.5  # Dummy value for saturation sigma

# Mock extended volume calculation (already in the main code)
def solve_extended_volume_post_saturation(Y, s, **kwargs):
    kappa = kwargs.get("kappa", 1e-6)
    D = kwargs.get("D", 100e3)
    d0 = kwargs.get("d0", 1e-2)
    return 6.7 * D**2.0 / d0 / kappa * Y * s

# Test class
class TestMOKinetics:
    def setup_method(self):
        """
        Setup the MO_KINETICS instance before each test.
        """
        self.kinetics = MO_KINETICS(Y_dummy, I_dummy)

    def test_initialization(self):
        """
        Test initialization of MO_KINETICS attributes.
        """
        assert self.kinetics.Y_func == Y_dummy
        assert self.kinetics.I_func == I_dummy
        assert self.kinetics.kappa == 1e-6
        assert self.kinetics.D == 100e3
        assert self.kinetics.d0 == 1e-2
        assert self.kinetics.t_scale is None
        assert self.kinetics.Av is None
        assert self.kinetics.last_solution is None
        assert self.kinetics.last_is_saturated is False
        assert self.kinetics.Y_prime_func is None
        assert self.kinetics.I_prime_func is None

    def test_solve_modified_equation_no_saturation(self):
        """
        Test solve_modified_equation for a case where site saturation is not reached.
        """
        t_span = np.array([0, 1])
        X_ini = np.array([0, 0, 0, 0])
        is_saturated = False

        X_array, is_saturated_array = self.kinetics.solve_modified_equation(t_span, X_ini, is_saturated, n_span=10)
        assert X_array.shape == (4, 10)
        assert is_saturated_array.shape == (10,)
        assert not np.any(is_saturated_array)

    def test_solve_modified_equation_with_saturation(self):
        """
        Test solve_modified_equation for a case where site saturation is reached.
        """
        t_span = np.array([0, 1])
        X_ini = np.array([0, 0, 0, 0])
        is_saturated = True

        X_array, is_saturated_array = self.kinetics.solve_modified_equation(t_span, X_ini, is_saturated, n_span=10)
        assert X_array.shape == (4, 10)
        assert is_saturated_array.shape == (10,)
        assert np.all(is_saturated_array)

    def test_solve_modified_equation_invalid_inputs(self):
        """
        Test solve_modified_equation with invalid inputs.
        """
        t_span = np.array([0, 1])
        X_ini = np.array([0, 0])  # Incorrect shape for X_ini
        is_saturated = False

        with pytest.raises(ValueError):
            self.kinetics.solve_modified_equation(t_span, X_ini, is_saturated, n_span=10)
