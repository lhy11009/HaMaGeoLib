import sys, os
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, package_root)

from hamageolib.utils.exception_handler import my_assert

# from ...utils.exception_handler import my_assert

def calculate_sigma_s(I_PT, Y_PT, d_0, **kwargs):
    """
    Calculate the dimensionless time (sigma_s) for the phase transformation process.

    Parameters:
    - I_PT (float): Nucleation rate as a function of pressure and temperature (s^-1 m^-3).
    - Y_PT (float): Growth rate as a function of pressure and temperature (m/s).
    - d_0 (float): Grain size of olivine (m).

    Returns:
    - sigma_s (float): Dimensionless time for site saturation.
    """
    kappa = kwargs.get("kappa", 1e-6) # Thermal diffusivity (m^2/s).
    D = kwargs.get("D", 100e3) # slab thickness
    # Compute the dimensionless time
    if I_PT == 0 or Y_PT == 0 or d_0 == 0:
        sigma_s = float('inf')
    else:
        sigma_s = (kappa / D**2) * ((I_PT * Y_PT**2 * d_0) / 6.7)**(-1/3)
    return sigma_s


def calculate_avrami_number_yoshioka_2015(I_max, Y_max, **kwargs):
    """
    Calculate the Avrami number (Av) using the corrected Equation (19).
    
    Parameters:
    - I_max (float): Maximum nucleation rate in s^-1 m^-2.
    - Y_max (float): Maximum growth rate in m/s.
    
    Returns:
    - Av (float): Avrami number (dimensionless).
    """
    kappa = kwargs.get("kappa", 1e-6) # Thermal diffusivity (m^2/s).
    D = kwargs.get("D", 100e3) # slab thickness
    # Compute the Avrami number
    Av = (D**2 / kappa)**4 * I_max * Y_max**3
    return Av


def solve_extended_volume_post_saturation(Y, s, **kwargs):
    '''
    Solve for the extended volume after site saturation is reached.
    
    Parameters:
    - Y (float): Current saturation level.
    - s (float): Dimensionless time.
    - kwargs (dict): Optional parameters including:
        - kappa (float): Thermal diffusivity (default=1e-6, m^2/s).
        - D (float): Slab thickness (default=100e3, m).
        - d0 (float): Parental grain size (default=1e-2, m).
    
    Returns:
    - float: Extended volume after site saturation.
    '''
    # Calculate the extended volume based on the provided parameters
    kappa = kwargs.get("kappa", 1e-6) # Thermal diffusivity (m^2/s).
    D = kwargs.get("D", 100e3) # Slab thickness
    d0 = kwargs.get("d0", 1e-2) # Parental grain size (m)
    X3 = 6.7 * D**2.0 / d0 / kappa * Y * s
    return X3


def ode_system(s, X, Av, Y_prime, I_prime):
    """
    Define the ODE system based on the modified Equation (18).
    
    Parameters:
    - s (float): Dimensionless time.
    - X (array): State vector [X3, X2, X1, X0].
    - Av (float): Avrami number.
    - Y_prime (callable): Function returning Y'(s).
    - I_prime (callable): Function returning I'(s).
    
    Returns:
    - list: Derivatives of the state variables [dX0, dX1, dX2, dX3].
    """
    # Extract state variables
    X0, X1, X2, X3 = X
    Av_factor = Av**(1/4)
    
    # Calculate the derivatives based on the Avrami equation
    dX3 = Av_factor * (4 * np.pi * Y_prime(s) * X2)
    dX2 = Av_factor * (2 * Y_prime(s) * X1)
    dX1 = Av_factor * (Y_prime(s) * X0)
    dX0 = Av_factor * I_prime(s)

    return [dX0, dX1, dX2, dX3]


def solve_modified_equations_eq18(Av, Y_prime_func, I_prime_func, s_span, X, **kwargs):
    """
    Solve the modified Equation (18) using numerical integration.
    
    Parameters:
    - Av (float): Avrami number.
    - Y_prime_func (callable): Function for Y'(s) (growth rate).
    - I_prime_func (callable): Function for I'(s) (nucleation rate).
    - s_span (tuple): Time span (s_start, s_end) for integration.
    - X (array): Initial conditions [X3, X2, X1, X0].
    - kwargs (dict): Additional options, including:
        - n_span (int): Number of time steps (default=100).
    
    Returns:
    - OdeResult: Integrated solution from solve_ivp.
    """
    print("call function solve_modified_equations_eq18") # debug

    # Define the system of ODEs to solve
    def odes(s, X):
        return ode_system(s, X, Av, Y_prime_func, I_prime_func)

    # Extract optional parameters
    debug = kwargs.get("debug", False)
    n_span = kwargs.get("n_span", 100)  # Number of time steps

    # debug outputs
    if debug:
        print("X0 = ", X)
        print("s_span = ", s_span)
        print("Av = %.4e" % Av)
        print("Y_prime_func0 = %.4e, Y_prime_func1 = %.4e" % (Y_prime_func(s_span[0]), Y_prime_func(s_span[1])))
        print("I_prime_func0 = %.4e, I_prime_func1 = %.4e" % (I_prime_func(s_span[0]), I_prime_func(s_span[1])))

    # solve the odes
    solution = solve_ivp(odes, s_span, X, method='RK45', t_eval=np.linspace(s_span[0], s_span[1], n_span))
    return solution

def growth_rate_hosoya_06_eq2_P1(P, T, Coh):
    """
    Calculate the growth rate following Equation 2 in Hosoya 2006.

    Parameters:
    - P (float): Pressure in Pascals.
    - T (float): Temperature in Kelvin.
    - Coh (float): Concentration of water in weight parts per million (wt.ppm H2O).

    Returns:
    - float: The growth rate calculated using the given parameters.
    """
    R = 8.31446  # J / mol*K, universal gas constant

    # Constants based on Hosoya 2006
    A = np.exp(-18.0)  # m s-1 wt.ppmH2O^(-3.2)
    n = 3.2
    dHa = 274.0e3  # J / mol, activation enthalpy
    Vstar = 3.3e-6  # m^3 / mol, activation volume

    growth_rate_part = A * Coh**n * np.exp(-(dHa + P * Vstar) / (R * T))

    return growth_rate_part

def growth_rate_hosoya_06_eq2(P, T, P_eq, Coh):
    """
    Calculate the growth rate following Equation 2 in Hosoya 2006, considering 
    pressure and temperature variations.

    Parameters:
    - P (float or np.ndarray): Pressure in Pascals.
    - T (float or np.ndarray): Temperature in Kelvin.
    - P_eq (float or np.ndarray): Equilibrium pressure in Pascals.
    - Coh (float): Concentration of water in weight parts per million (wt.ppm H2O).

    Returns:
    - float or np.ndarray: The growth rate for each pressure point.
    """
    R = 8.31446  # J / mol*K, universal gas constant
    dV_ol_wd = 2.4e-6  # m^3 / mol, difference in volume between phases
    
    if type(P) in [float, np.float64]:
        assert(P > P_eq)
    elif type(P) == np.ndarray:
        assert(np.min(P - P_eq) > 0.0)
    else:
        raise TypeError("P must be float or ndarray")

    # Determine growth rate based on pressure type (float or array)
    dGr = dV_ol_wd * (P - P_eq)
    growth_rate = growth_rate_hosoya_06_eq2_P1(P, T, Coh) * T * (1 - np.exp(-dGr / (R * T)))

    return growth_rate

def nucleation_rate_yoshioka_2015(P, T, P_eq):
    """
    Compute the nucleation rate using Equation (10) from Yoshioka et al. (2015).
    
    Parameters:
    - T (float): Temperature in Kelvin.
    - P (float): Pressure in Pa
    - delta_G_v (float): Free energy change per volume in J/m^3.

    Constants
    - gamma (float): Surface energy in J/m^2 (default: 0.0506).
    - K0 (float): Pre-exponential factor in s^-1 m^-2 K^-1 (default: 3.54e38).
    - Q_a (float): Activation energy for growth in J/mol (default: 355e3).
    - k (float): Boltzmann constant in J/K (default: 1.38e-23).
    - R (float): Universal gas constant in J/(mol*K) (default: 8.314).
    - dV_ol_wd (float): difference in mole volume between phase.
    - V_initial (float): for olivine, estimation at 410 km, mole volume
    
    Returns:
    - I (float): Nucleation rate in s^-1 m^-2.
    """
    gamma=0.0506; K0=3.54e38; dH_a=344e3; V_star=4e-6; k=1.38e-23; R=8.314
    dV_ol_wd = 2.4e-6; V_initial = 35.17e-6

    if type(P) in [float, np.float64]:
        assert(P >= P_eq)
    elif type(P) == np.ndarray:
        assert(np.min(P - P_eq) >= 0.0)
    else:
        raise TypeError("P must be float or ndarray")

    delta_G_v = dV_ol_wd / V_initial * (P - P_eq)

    # print("P_eq = %.2f GPa, dGr_vs = %.4e" % (P_eq/1e9, delta_G_v)) # debug

    # Compute the homogeneous nucleation activation energy
    delta_G_hom = (16 * np.pi * gamma**3) / (3 * (delta_G_v)**2)
    
    # Compute the nucleation rate
    Q_a = dH_a + P * V_star 
    I = K0 * T * np.exp(-delta_G_hom / (k * T)) * np.exp(-Q_a / (R * T))
    
    return I

def compute_eq_P(PT_dict, T):
    """
    Computes the equilibrium pressure based on the given temperature and a PT_dict containing parameters.
    
    Parameters:
    - PT_dict (dict): A dictionary containing the following keys:
        - "T" (float): Reference temperature.
        - "cl" (float): Calibration constant or slope related to pressure-temperature relationship.
        - "P" (float): Reference pressure.
    - T (float): The temperature at which to compute the equilibrium pressure.
    
    Returns:
    - float: The equilibrium pressure at the given temperature.
    """
    # Calculate equilibrium pressure using the linear relationship in PT_dict
    P_eq = (T - PT_dict["T"]) * PT_dict["cl"] + PT_dict["P"]

    return P_eq

def compute_eq_T(PT_dict, P):
    """
    Computes the equilibrium temperature based on the given pressure and a PT_dict containing parameters.
    
    Parameters:
    - PT_dict (dict): A dictionary containing the following keys:
        - "T" (float): Reference temperature.
        - "cl" (float): Calibration constant or slope related to pressure-temperature relationship.
        - "P" (float): Reference pressure.
    - P (float): The pressure at which to compute the equilibrium temperature.
    
    Returns:
    - float: The equilibrium temperature at the given pressure.
    """
    # Calculate equilibrium temperature using the inverse linear relationship in PT_dict
    T_eq = (P - PT_dict["P"]) / PT_dict["cl"] + PT_dict["T"]
    
    return T_eq



class MO_KINETICS:
    """
    Class to handle the kinetics of phase transformations, including nucleation and growth rates.

    Attributes:
    - Y_func_ori (callable): Function for the growth rate Y(P, T, Peq, Coh).
    - I_func_ori (callable): Function for the nucleation rate I(P, T, Peq).
    - Y_func (callable): Function for the growth rate Y(t).
    - I_func (callable): Function for the nucleation rate I(t).
    - kappa (float): Thermal diffusivity (default=1e-6, m^2/s).
    - D (float): Slab thickness (default=100e3, m).
    - d0 (float): Parental grain size (default=1e-2, m).
    - t_scale (float): Scaling factor for time.
    - Av (float): Avrami number.
    - last_solution (OdeResult): Solution from the last numerical integration.
    - last_is_saturated (bool): Indicator for site saturation in the last step.
    - Y_prime_func (callable): Function for normalized growth rate Y'(s).
    - I_prime_func (callable): Function for normalized nucleation rate I'(s).
    - PT_eq (dict): phase transition equilibrium parameters, T, P, cl-Claypeyron slope
    """

    def __init__(self):
        """
        Initialize the MO_KINETICS class with growth and nucleation rate functions.
        
        Parameters:
        - Y_func (callable): Function for the growth rate Y(t).
        - I_func (callable): Function for the nucleation rate I(t).
        """
        self.Y_func_ori = None
        self.I_func_ori = None
        self.Y_func = None
        self.I_func = None
        self.kappa = 1e-6
        self.D = 100e3
        self.d0 = 1e-2
        self.t_scale = self.D**2.0 / self.kappa
        self.Av = None
        self.last_solution = None
        self.last_is_saturated = False
        self.Y_prime_func = None
        self.I_prime_func = None
        self.PT_eq = {"T": None, "P": None, "cl": None}
        self.result_columns = ["t", "N", "Dn", "S", "Vtilde", "V", "is_saturated"]
        self.n_col = len(self.result_columns)

    def set_kinetics_model(self, Y_func, I_func):
        """
        Set the kinetics model by assigning nucleation and growth rate functions.
        
        Parameters:
        - Y_func (callable): Function for the growth rate Y(t).
        - I_func (callable): Function for the nucleation rate I(t).
        """
        self.Y_func_ori = Y_func
        self.I_func_ori = I_func

    def set_kinetics_fixed(self, P, T, Coh):
        """
        Fix the kinetics model based on specific pressure, temperature, and cohesion values.

        Parameters:
        - P (float): Pressure value.
        - T (float): Temperature value.
        - Coh (float): Cohesion parameter.
        """
        assert(self.PT_eq is not None)

        # compute equilibrium condition
        Peq = compute_eq_P(self.PT_eq, T)

        # fix value to Y_func and I_func
        self.Y_func = lambda t: self.Y_func_ori(P, T, Peq, Coh)
        self.I_func = lambda t: self.I_func_ori(P, T, Peq)
    
    def set_PT_eq(self, P0, T0, cl):
        """
        Set equilibrium conditions for phase transformation.
        
        Parameters:
        - P0 (float): Reference pressure.
        - T0 (float): Reference temperature.
        - cl (float): Clapeyron slope.
        """
        self.PT_eq = {"T": T0, "P": P0, "cl": cl} 

    class MO_INITIATION_Error(Exception):
        """
        Custom exception for errors in the initiation of MO_KINETICS.
        """
        pass

    def solve_modified_equation(self, t_span, X_ini, is_saturated, **kwargs):
        '''
        Solve the kinetic equations with piecewise time normalization.
        
        Parameters:
        - t_span (array): Time span for integration [t_start, t_end].
        - X_ini (array): Initial conditions [X3, X2, X1, X0].
        - is_saturated (bool): Whether site saturation is reached.
        - kwargs (dict): Additional options, including:
            - n_span (int): Number of time steps (default=10).
        
        Returns:
        - tuple: Dimensional solution array (X_array) and saturation status array.
        Note this function is designed to re-normalize and solve the problem piecewise in time.
        The reason is that the nucleation and growth rate change their value by orders of magnitude.
        This makes the tasks of nondimensionalization harder. Conceptually, this helps, but we will investigate
        more to see whether this approach is indeed needed.
        '''
        debug = kwargs.get("debug", False)  # print debug messages
        n_span = kwargs.get("n_span", 10)

        # assert previous steps have been taken 
        my_assert(self.PT_eq is not None, self.MO_INITIATION_Error, "Initiation error: call set_PT_eq function first.")
        assert(self.Y_func is not None and self.I_func is not None, self.MO_INITIATION_Error, "Initiation error: call one of the set_kinetics function first.")


        # compute scaling variables
        # The Av value is prevented from being too small. Note this will only affect
        # the nondimensional variables. 
        I_max = max(1e-50, 6.0*self.I_func(t_span[0]) / self.d0) # per unit volume
        Y_max = max(1e-50, self.Y_func(t_span[0]))
        self.Av = calculate_avrami_number_yoshioka_2015(I_max, Y_max)

        # print debug message of scaling values
        if debug:
            print("solve_modified_equation: I_max = %.4e, Y_max = %.4e, Av = %.4e" % (I_max, Y_max, self.Av))
        
        self.Y_prime_func = lambda s: self.Y_func(s*self.t_scale) / Y_max
        self.I_prime_func = lambda s: self.I_func(s*self.t_scale) *6.0 / self.d0 / I_max

        # update the t scaling with local values of nucleation and growth rates
        self.X_scale_array = np.array([I_max**(3.0/4.0)*Y_max**(-3.0/4.0), I_max**(1.0/2.0)*Y_max**(-1.0/2.0), I_max**(1.0/4.0)*Y_max**(-1.0/4.0), 1.0])
        
        # nondimensionalize the time variable and calculate nondimensional constants
        s_span = t_span / self.t_scale
        s_values = np.linspace(s_span[0], s_span[1], n_span)
        I_array = self.I_func(s_values * self.t_scale)
        Y_array = self.Y_func(s_values * self.t_scale)

        # print("s_span: ", s_span) # debug

        # nondimensionalize the initial solution
        X_ini_nl = X_ini / self.X_scale_array

        # compute saturation condition for s 
        s_saturation = calculate_sigma_s(6.0*I_array/self.d0, Y_array, self.d0, kappa=self.kappa, D=self.D)

        if debug:
            print("solve_modified_equation: t_span = ", t_span)
            print("solve_modified_equation: t_saturation = ", s_saturation*self.t_scale)
            print("solve_modified_equation: is_saturated = ", is_saturated)

        if not is_saturated:
            # in case site situation is not reached for the initial condition,
            # compute a potential saturation index
            indices = np.where(s_values > s_saturation)[0]
            if len(indices) > 0 and indices[0] > 1:
                # saturation is reached after at least 2 points
                i0 = indices[0]
                s_span_us = np.array([s_values[0], s_values[i0]])

                # print("i0 = %d" % i0) # debug

                # solve for a pre-saturation subset 
                kwargs["n_span"] = i0
                solution_nd = solve_modified_equations_eq18(self.Av, self.Y_prime_func, self.I_prime_func, s_span_us, X_ini_nl, **kwargs)
                
                # parse the solution at the last time step
                X_array_nd = solution_nd.y

                # parse to X_array
                X_array = np.zeros((4, n_span)) 
                X_array[:, 0: i0] = X_array_nd * self.X_scale_array[:, np.newaxis] # scale by the rows
                
                # compute the other subset with saturation conditions
                X_array[:, i0:] = X_array[:, i0 - 1][:, np.newaxis]  # replicate the i0 - 1 column
                X_array[3, i0:] = solve_extended_volume_post_saturation(Y_max, s_values[i0:], kappa=self.kappa, D=self.D, d0=self.d0)

                if debug:
                    print("saturation is reached after at least 2 points")
                    print("X_array_nd[:, -1] = ", X_array_nd[:, -1])
                    print("X_array[:, -1] = ", X_array[:, -1])
                
                # record saturation
                is_saturated_array = np.full(n_span, False)
                is_saturated_array[i0: ] = True
                
                # record the whole solution
                self.last_solution = solution_nd
                self.last_is_saturated = True

            elif len(indices) > 0 and indices[0] <= 1:
                # saturation is reached at either the 0th or 1st point
                X_array = np.tile(X_ini, (n_span, 1)).T
                X_array[3,:] = solve_extended_volume_post_saturation(Y_max, s_values, kappa=self.kappa, D=self.D, d0=self.d0)
            
                is_saturated_array = np.full(n_span, True)
                is_saturated_array[0:indices[0]] = False
            
                # record the whole solution
                self.last_solution = None
                self.last_is_saturated = True

                if debug: 
                    print("saturation is reached at either the 0th or 1st point")
                    print("X_array[:, -1] = ", X_array[:, -1])

            else:
                # saturation is not reached
                # print("solving modified equations eq18") # debug
                solution_nd = solve_modified_equations_eq18(self.Av, self.Y_prime_func, self.I_prime_func, s_span, X_ini_nl, **kwargs)

                # record the whole solution
                self.last_solution = solution_nd
                self.last_is_saturated = False
                
                # parse the solution at the last time step
                X_array_nd = solution_nd.y
                
                # rescale the X_array
                X_array = X_array_nd * self.X_scale_array[:, np.newaxis] # scale by the rows
                
                # record saturation
                is_saturated_array = np.full(n_span, False)
            
                # record the whole solution
                self.last_solution = solution_nd
                self.last_is_saturated = False

                if debug:
                    print("saturation is not reached.") 
                    print("X_array_nd[:, -1] = ", X_array_nd[:, -1])
                    print("X_array[:, -1] = ", X_array[:, -1])
        else:
            # in case site situation is already reached for the initial condition,
            # use the formulate post saturation
            X_array = np.tile(X_ini, (n_span, 1)).T
            X_array[3,:] = solve_extended_volume_post_saturation(Y_max, s_values, kappa=self.kappa, D=self.D, d0=self.d0)
            is_saturated_array = np.full(n_span, True)
            
            # record the whole solution
            self.last_solution = None
            self.last_is_saturated = True

            if debug: 
                print("saturation is already reached previously.") 
                print("X_array[:, -1] = ", X_array[:, -1])
        
        return X_array, is_saturated_array
    
    def solve(self, P, T, t_max, n_t, n_span, **kwargs):

        debug = kwargs.get("debug", False)
        
        X = np.array([0.0, 0.0, 0.0, 0.0])
        is_saturated = False
        results = np.zeros([n_t * n_span, self.n_col])

        # compute equilibrium condition
        Peq = compute_eq_P(self.PT_eq, T)

        # Loop over time steps
        for i_t in range(n_t):
            if debug:
                print("i_t: %d" % i_t) # debug

            # Assert that X remains a 1-d numpy array
            my_assert(type(X) == np.ndarray and X.ndim == 1, TypeError, f"Check at loop {i_t}: X must be a 1-d numpy array.")
                
            # Define the time span for the current step
            t_piece_min = t_max / n_t * i_t
            t_piece_max = t_max / n_t * (i_t + 1)
            t_span = np.array([t_piece_min, t_piece_max])
            if P > Peq:
                # Solve the kinetics if equilibrium condition is met
                # Note here the X_array is dimensional
                kwargs["n_span"] = n_span
                X_array, is_saturated_array = self.solve_modified_equation(t_span, X, is_saturated, **kwargs)
                X = X_array[:, -1]
                is_saturated = is_saturated_array[-1]
            else:
                # Assign trivial values if equilirium condition is not met
                X_array = np.zeros([4, n_span])
                is_saturated_array = np.zeros(n_span)
                X = np.array([0.0, 0.0, 0.0, 0.0])
                is_saturated = False
            
            # Record the results in the DataFrame
            V_array = 1 - np.exp(-X_array[3, :])
            for j_s in range(n_span):
                results[i_t*n_span + j_s, :] = [t_piece_min + (t_piece_max - t_piece_min)/n_span*j_s,\
                                                X_array[0, j_s], X_array[1, j_s], X_array[2, j_s], X_array[3, j_s],\
                                                    V_array[j_s], is_saturated_array[j_s]]
                

        return results
