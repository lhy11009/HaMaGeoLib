import sys, os
from scipy.integrate import solve_ivp
import numpy as np
import pandas as pd
from collections import namedtuple

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
    # check if a 0.0 value is parsed in as inputs,
    # if so, return an infinite number.
    # Maintain consistency between float number and numpy arrays.
    numerator = I_PT * Y_PT**2 * d_0 / 6.7
    if type(I_PT) in [float, np.float64]:
        safe_numerator = np.nan if numerator == 0 else numerator # replace zero with NaN
    elif type(I_PT) == np.ndarray:
        safe_numerator = np.where(numerator == 0, np.nan, numerator)  # replace zero with NaN
    else:
        raise NotImplementedError("I_PT and Y_PT needs to be either float or np.ndarrray. Get %s" % str(type(I_PT)))
    sigma_s = (kappa / D**2) * (safe_numerator)**(-1/3)
    return sigma_s


def calculate_avrami_number(I_max, Y_max, **kwargs):
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

def solve_extended_volume_post_saturation_by_increment(Y, s, s_ini, V_extended_ini, **kwargs):
    '''
    Solve for the extended volume after site saturation is reached.
    Derive the solution from the time increment
    
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
    X3 = V_extended_ini + solve_extended_volume_post_saturation(Y, s, **kwargs)\
          - solve_extended_volume_post_saturation(Y, s_ini, **kwargs)
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
    dX3 = Av_factor * (4 * Y_prime(s) * X2)
    dX2 = Av_factor * (np.pi * Y_prime(s) * X1)
    dX1 = Av_factor * (2 * Y_prime(s) * X0)
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
    debug = kwargs.get("debug", False)
    if debug:
        print("call function solve_modified_equations_eq18")

    # Define the system of ODEs to solve
    def odes(s, X):
        return ode_system(s, X, Av, Y_prime_func, I_prime_func)

    # Extract optional parameters
    n_span = kwargs.get("n_span", 100)  # Number of time steps

    # debug outputs
    if debug:
        print("X0 = ", X)
        print("s_span = ", s_span)
        print("Av = %.4e" % Av)
        print("Y_prime_func0 = %.4e, Y_prime_func1 = %.4e" % (Y_prime_func(s_span[0]), Y_prime_func(s_span[1])))
        print("I_prime_func0 = %.4e, I_prime_func1 = %.4e" % (I_prime_func(s_span[0]), I_prime_func(s_span[1])))

    # solve the odes
    solution = solve_ivp(odes, s_span, X, method='RK45', t_eval=np.linspace(s_span[0], s_span[1], n_span+1))

    return solution


class PTKinetics:
    """
    Encapsulates kinetic models for grain growth and nucleation:
    - Hosoya (2006) growth rate models
    - Yoshioka et al. (2015) nucleation rate

    Attributes:
        R (float): Universal gas constant (J/mol/K)
        k (float): Boltzmann constant (J/K)
        A (float): Pre-exponential factor for growth (m/s/(wt.ppmH2O)^n)
        n (float): Water content exponent in growth law
        dHa (float): Activation enthalpy for growth (J/mol)
        V_star_growth (float): Activation volume for growth (m^3/mol)
        gamma (float): Interfacial energy (J/m^2)
        fs (float): Shape factor for nucleation
        K0 (float): Pre-exponential factor for nucleation (s^-1 m^-2 K^-1)
        Vm (float): Molar volume at transition (m^3/mol)
        dS (float): Entropy change across transition (J/mol/K)
        dV (float): Volume change across transition (m^3/mol)
        I_func (func): selected nucleation mechenism
        I_type (int): type of the nucleation mechanism; 0 - volumetic; 1 - surface; 2 - line; 3 - corner
        Y_func (func): selected grain growth mechenism
    """
    Constants = namedtuple("Constants", [
        "R", "k",
        "A", "n", "dHa", "V_star_growth",
        "gamma", "fs", "K0", "Vm", "dS", "dV",
        "nucleation_type"
    ])

    def __init__(self, constants=None):
        """
        Initialize the kinetic model with optional user-specified constants.

        Args:
            constants (namedtuple, optional): A Constants namedtuple with all parameters.
                                               Defaults to preset values from Hosoya (2006) and Yoshioka (2015).
        """
        if constants is None:
            constants = self._default_constants()

        # Assign constants to class attributes
        self.R = constants.R
        self.k = constants.k
        self.A = constants.A
        self.n = constants.n
        self.dHa = constants.dHa
        self.V_star_growth = constants.V_star_growth
        self.gamma = constants.gamma
        self.fs = constants.fs
        self.K0 = constants.K0
        self.Vm = constants.Vm
        self.dS = constants.dS
        self.dV = constants.dV
        self.growth_rate = self.growth_rate_interface_P2
        self.nucleation_type = constants.nucleation_type

    def _default_constants(self):
        return self.Constants(
            R=8.31446,
            k=1.38e-23,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.6,
            fs=1e-3,
            K0=3.65e38,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=0
        )

    def growth_rate_interface_P1(self, P, T, Coh):
        """
        Calculate the first part of the growth rate (Eq. 2 from Hosoya 2006).
        """
        return self.A * Coh**self.n * np.exp(-(self.dHa + P * self.V_star_growth) / (self.R * T))

    def growth_rate_interface_P2(self, P, T, P_eq, T_eq, Coh):
        """
        Full Hosoya (2006) Eq. 2: includes free energy driving force.
        """
        if isinstance(P, (float, np.floating)):
            assert P > P_eq
        elif isinstance(P, np.ndarray):
            assert np.min(P - P_eq) > 0.0
        else:
            raise TypeError("P must be float or ndarray")

        delta_G_d = self.dV * (P - P_eq) - self.dS * (T - T_eq)
        base_growth = self.growth_rate_interface_P1(P, T, Coh)
        return base_growth * T * (1 - np.exp(-delta_G_d / (self.R * T)))

    def nucleation_rate(self, P, T, P_eq, T_eq):
        """
        Nucleation rate from Yoshioka et al. (2015), Eq. 10.
        """
        if isinstance(P, (float, np.floating)):
            assert P >= P_eq
        elif isinstance(P, np.ndarray):
            assert np.min(P - P_eq) >= 0.0
        else:
            raise TypeError("P must be float or ndarray")

        delta_G_d = self.dV * (P - P_eq) - self.dS * (T - T_eq)
        delta_G_hom = (16 * self.fs * np.pi * self.Vm**2 * self.gamma**3) / (3 * delta_G_d**2)
        Q_a = self.dHa + P * self.V_star_growth

        return self.K0 * T * np.exp(-delta_G_hom / (self.k * T)) * np.exp(-Q_a / (self.R * T))

    def critical_radius(self, P, T, P_eq, T_eq):
        """
        critical_radius of nuclei
        """
        if isinstance(P, (float, np.floating)):
            assert P >= P_eq
        elif isinstance(P, np.ndarray):
            assert np.min(P - P_eq) >= 0.0
        else:
            raise TypeError("P must be float or ndarray")

        delta_G_d = self.dV * (P - P_eq) - self.dS * (T - T_eq)
        rc = 2 * self.fs**(1.0/3) * self.gamma * self.Vm / delta_G_d
        return rc


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
    - P (float): The pressure at which to compute the equilibrium pressure.
    
    Returns:
    - float: The equilibrium pressure at the given temperature.
    """
    # Calculate equilibrium pressure using the linear relationship in PT_dict
    T_eq = (P - PT_dict["P"]) / PT_dict["cl"] + PT_dict["T"]

    return T_eq

class MO_KINETICS:
    """
    Class to handle the kinetics of phase transformations, including nucleation and growth rates.

    Attributes:
    - Kinetics (class): class for kinetic functions (nucleation and grain growth)
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
    - is_P_higher_than_Peq (bool): is the P value we have higher than the equilirbium value
    - last_is_saturated (bool): Indicator for site saturation in the last step.
    - Y_prime_func (callable): Function for normalized growth rate Y'(s).
    - I_prime_func (callable): Function for normalized nucleation rate I'(s).
    - PT_eq (dict): phase transition equilibrium parameters, T, P, cl-Claypeyron slope
    - post_process (list): additional post-process steps
    - X_saturated: solution when citet situation is reached.
    """
    Constants = namedtuple("Constants", [
        "R", "k", "kappa", "D", "d0",
        "A", "n", "dHa", "V_star_growth",
        "gamma", "fs", "K0", "Vm", "dS", "dV",
        "nucleation_type"
    ])

    def __init__(self, constants=None, **kwargs):
        """
        Initialize the MO_KINETICS class with growth and nucleation rate functions.
        
        Parameters:
        - Y_func (callable): Function for the growth rate Y(t).
        - I_func (callable): Function for the nucleation rate I(t).
        """
        self.post_process = kwargs.get("post_process", [])
        assert(type(self.post_process) == list)
        if "ts" in self.post_process:
            self.post_process_ts = True
        else:
            self.post_process_ts = False
        if "tg" in self.post_process:
            self.post_process_tg = True
        else:
            self.post_process_tg = False

        if constants is None:
            self.constants = self._default_constants()
        else:
            self.constants = constants

        # model functions
        self.Y_func_ori = None
        self.I_func_ori = None
        self.Y_func = None
        self.I_func = None
        self.Y_prime_func = None
        self.I_prime_func = None

        # model parameters
        self.PT_eq = {"T": None, "P": None, "cl": None}
        self.kappa = self.constants.kappa
        self.D = self.constants.D
        self.d0 = self.constants.d0
        self.t_scale = self.D**2.0 / self.kappa
        self.S0 = 6.7 / self.d0

        # model solutions
        self.Av = None
        self.is_P_higher_than_Peq = False
        self.result_columns = ["t", "N", "Dn", "S", "Vtilde", "V", "is_saturated"]
        if self.post_process_ts:
            self.result_columns.append("t_saturated")
        if self.post_process_tg:
            self.result_columns.append("t_growth")
        self.n_col = len(self.result_columns)
        self.last_is_saturated = False
        self.last_solution = None
        self.X_saturated = None


    def _default_constants(self):
        return self.Constants(
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
            K0=3.65e38,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=0
        )
    
    def set_initial_grain_size(self, d0):
        """
        Set the initial grain size
        """
        self.d0 = d0

    def set_kinetics_model(self, mKinetics):
        """
        Set the kinetics model by assigning nucleation and growth rate functions.
        
        Parameters:
        - Y_func (callable): Function for the growth rate Y(t).
        - I_func (callable): Function for the nucleation rate I(t).
        """
        self.Kinetics = mKinetics
        self.Y_func_ori = mKinetics.growth_rate
        self.I_func_ori = mKinetics.nucleation_rate
        self.I_type_ori = mKinetics.nucleation_type
    
    def link_and_set_kinetics_model(self, MKinetics):
        """
        Link and set the kinetics model by assigning nucleation and growth rate functions.
        """
        constants = MKinetics.Constants(
            R=self.constants.R,
            k=self.constants.k,
            A=self.constants.A,
            n=self.constants.n,
            dHa=self.constants.dHa,
            V_star_growth=self.constants.V_star_growth,
            gamma=self.constants.gamma,
            fs=self.constants.fs,
            K0=self.constants.K0,
            Vm=self.constants.Vm,
            dS=self.constants.dS,
            dV=self.constants.dV,
            nucleation_type=self.constants.nucleation_type
        )
        
        mKinetics = MKinetics(constants)

        self.set_kinetics_model(mKinetics)

        assert(self.I_type_ori == self.constants.nucleation_type)

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
        Teq = compute_eq_T(self.PT_eq, P)

        if P > Peq:
            self.is_P_higher_than_Peq = True
        else:
            self.is_P_higher_than_Peq = False

        # fix value to Y_func and I_func
        self.Y_func = lambda t: self.Y_func_ori(P, T, Peq, Teq, Coh)
        
        if self.I_type_ori == 0:
            # volumetric nucleation
            f0 = 1
        elif self.I_type_ori == 1:
            # surface nucleation
            f0 = self.S0
        else:
            return NotImplementedError()
        self.I_func = lambda t: f0*self.I_func_ori(P, T, Peq, Teq)
    
    def set_PT_eq(self, P0, T0, cl):
        """
        Set equilibrium conditions for phase transformation.
        
        Parameters:
        - P0 (float): Reference pressure.
        - T0 (float): Reference temperature.
        - cl (float): Clapeyron slope.
        """
        self.PT_eq = {"T": T0, "P": P0, "cl": cl} 

    def compute_Av(self, t, **kwargs):
        """
        Compute the Avrami number at time t
        """
        # See if the current P, T condition is higher than the equilibrium
        # If now, do not compute a Av value
        D = kwargs.get("D", self.D)
        if self.is_P_higher_than_Peq:
            I_max = max(1e-50, self.I_func(t)) # per unit volume
            Y_max = max(1e-50, self.Y_func(t))
            Av = calculate_avrami_number(I_max, Y_max, D=D)
        else:
            Av = float('inf')
        return Av
    
    def compute_Iv(self, t):
        """
        Compute the Volumetric nucleation rate at time t
        Return nan value if equilibrium boundary is not reached
        """
        if self.is_P_higher_than_Peq:
            Iv = self.I_func(t)
        else:
            Iv = 0.0
        return Iv
    
    def compute_Y(self, t):
        """
        Compute the growth rate at time t
        """
        if self.is_P_higher_than_Peq:
            Y = self.Y_func(t)
        else:
            Y = 0.0
        return Y
    
    def compute_ts(self, t):
        """
        Compute the site situation time ts
        """
        if self.is_P_higher_than_Peq:
            ts = self.D**2.0/self.kappa * calculate_sigma_s(self.I_func(t), self.Y_func(t), self.d0, kappa=self.kappa, D=self.D)
        else:
            ts = float('inf')
        return ts
    
    def compute_tg(self, t):
        """
        Compute the grain growth time tg assuming site situation
        """
        if self.is_P_higher_than_Peq:
            tg =  0.693 / self.S0 / self.Y_func(t)
        else:
            tg = float('inf')
        return tg

    def compute_rc(self, P, T):
        """
        Compute the critical radius
        """
        Peq = compute_eq_P(self.PT_eq, T)
        Teq = compute_eq_T(self.PT_eq, P)
        if self.is_P_higher_than_Peq:
            rc = self.Kinetics.critical_radius(P, T, Peq, Teq)
        else:
            rc = 0.0
        return rc



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
            - n_span (int): Number of time intervals (default=10).
        
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
        I_max = max(1e-50, self.I_func(t_span[0])) # per unit volume
        Y_max = max(1e-50, self.Y_func(t_span[0]))
        self.Av = self.compute_Av(t_span[0])

        # print debug message of scaling values
        if debug:
            print("solve_modified_equation: I_max = %.4e, Y_max = %.4e, Av = %.4e" % (I_max, Y_max, self.Av))
        
        self.Y_prime_func = lambda s: self.Y_func(s*self.t_scale) / Y_max
        self.I_prime_func = lambda s: self.I_func(s*self.t_scale) / I_max

        # update the t scaling with local values of nucleation and growth rates
        self.X_scale_array = np.array([I_max**(3.0/4.0)*Y_max**(-3.0/4.0), I_max**(1.0/2.0)*Y_max**(-1.0/2.0), I_max**(1.0/4.0)*Y_max**(-1.0/4.0), 1.0])
        
        # nondimensionalize the time variable and calculate nondimensional constants
        s_span = t_span / self.t_scale
        s_values = np.linspace(s_span[0], s_span[1], n_span+1)
        I_array = np.zeros(n_span+1)
        Y_array = np.zeros(n_span+1)
        for i in range(n_span+1):
            I_array = self.I_func(s_values[i] * self.t_scale)
            Y_array = self.Y_func(s_values[i] * self.t_scale)

        # nondimensionalize the initial solution
        X_ini_nl = X_ini / self.X_scale_array

        # compute saturation condition for s 
        # todo_s_saturation
        s_saturation = calculate_sigma_s(I_array, Y_array, self.d0, kappa=self.kappa, D=self.D)
            
        if debug:
            print("First element of I_array:", I_array)
            print("First element of Y_array:", Y_array)
            print("solve_modified_equation: t_span = ", t_span)
            print("solve_modified_equation: s_saturation = ", s_saturation)
            print("solve_modified_equation: t_saturation = ", s_saturation*self.t_scale)
            print("solve_modified_equation: is_saturated = ", is_saturated)
                
        if not is_saturated:
            # in case site situation is not reached for the initial condition,
            # compute a potential saturation index
            indices = np.where(s_values - s_values[0] > s_saturation)[0]
            if len(indices) > 0 and indices[0] > 1:
                # saturation is reached after at least 2 points
                i0 = indices[0]
                s_span_us = np.array([s_values[0], s_values[i0]])

                # solve for a pre-saturation subset 
                kwargs["n_span"] = i0

                solution_nd = solve_modified_equations_eq18(self.Av, self.Y_prime_func, self.I_prime_func, s_span_us, X_ini_nl, **kwargs)
                
                # parse the solution at the last time step
                X_array_nd = solution_nd.y

                # parse to X_array
                X_array = np.zeros((4, n_span+1)) 
                X_array[:, 0: i0+1] = X_array_nd * self.X_scale_array[:, np.newaxis] # scale by the rows

                # record the saturation state
                self.X_saturated = X_array[:, i0]
                
                # compute the other subset with saturation conditions
                X_array[:, i0:] = X_array[:, i0][:, np.newaxis]  # replicate the i0 column

                X_array[3, i0:] = solve_extended_volume_post_saturation_by_increment(Y_max, s_values[i0:], s_values[i0], X_array[3, i0], kappa=self.kappa, D=self.D, d0=self.d0)

                if debug:
                    print("saturation is reached after at least 2 points")
                    print("X_array_nd[:, -1] = ", X_array_nd[:, -1])
                    print("X_array[:, -1] = ", X_array[:, -1])
                
                # record saturation
                is_saturated_array = np.full(n_span+1, False)
                is_saturated_array[i0: ] = True
                
                # record the whole solution
                self.last_solution = solution_nd
                self.last_is_saturated = True

            elif len(indices) > 0 and indices[0] <= 1:
                # saturation is reached at either the 0th or 1st point

                X_array = np.tile(X_ini, (n_span+1, 1)).T

                # solve equation between s_span[0] and s_saturation
                s_span_new = (s_span[0], s_span[0] + s_saturation)
                solution_nd = solve_modified_equations_eq18(self.Av, self.Y_prime_func, self.I_prime_func, s_span_new, X_ini_nl, **kwargs)
                X_array_foo = solution_nd.y*self.X_scale_array[:, np.newaxis]
                for i_raw in range(1, X_array.shape[1]):
                    X_array[:, i_raw] = X_array_foo[:, -1] 
                self.X_saturated  = X_array_foo[:, -1]

                # solve the extended volume after step 1
                X_array[3,1:] = solve_extended_volume_post_saturation_by_increment(Y_max, s_values[1:], s_values[1], X_array[3, 1], kappa=self.kappa, D=self.D, d0=self.d0)
            
                is_saturated_array = np.full(n_span+1, True)
                is_saturated_array[0] = False
            
                # record the whole solution
                self.last_solution = None
                self.last_is_saturated = True

                if debug: 
                    print("saturation is reached at either the 0th or 1st point")
                    print("X_array[:, -1] = ", X_array[:, -1])

            else:
                # saturation is not reached
                solution_nd = solve_modified_equations_eq18(self.Av, self.Y_prime_func, self.I_prime_func, s_span, X_ini_nl, **kwargs)

                # record the whole solution
                self.last_solution = solution_nd
                self.last_is_saturated = False
                
                # parse the solution at the last time step
                X_array_nd = solution_nd.y
                
                # rescale the X_array
                X_array = X_array_nd * self.X_scale_array[:, np.newaxis] # scale by the rows
                
                # record saturation
                is_saturated_array = np.full(n_span+1, False)
            
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
            X_array = np.tile(X_ini, (n_span+1, 1)).T
            # X_array[3,:] = solve_extended_volume_post_saturation(Y_max, s_values, kappa=self.kappa, D=self.D, d0=self.d0)
            X_array[3,:] = solve_extended_volume_post_saturation_by_increment(Y_max, s_values, s_values[0], X_ini[3], kappa=self.kappa, D=self.D, d0=self.d0)
            is_saturated_array = np.full(n_span+1, True)
            
            # record the whole solution
            self.last_solution = None
            self.last_is_saturated = True

            if debug: 
                print("saturation is already reached previously.") 
                print("X_array[:, -1] = ", X_array[:, -1])
        
        return X_array, is_saturated_array

    def solve(self, P, T, t_min, t_max, n_t, n_span, **kwargs):

        debug = kwargs.get("debug", False)
        initial = kwargs.get("initial", None)

        # read the initial state
        if initial is None: 
            X = np.array([0.0, 0.0, 0.0, 0.0])
            is_saturated = False
        else:
            X = initial[1:5]
            is_saturated = initial[6]

        results = np.zeros([n_t * n_span + 1, self.n_col])

        # compute equilibrium condition
        Peq = compute_eq_P(self.PT_eq, T)

        # Loop over time steps
        for i_t in range(n_t):
            if debug:
                print("i_t: %d" % i_t)

            # Assert that X remains a 1-d numpy array
            my_assert(type(X) == np.ndarray and X.ndim == 1, TypeError, f"Check at loop {i_t}: X must be a 1-d numpy array.")
                
            # Define the time span for the current step
            t_piece_min = t_min + (t_max-t_min) / n_t * i_t
            t_piece_max = t_min + (t_max-t_min) / n_t * (i_t + 1)
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
                X_array = np.zeros([4, n_span+1])
                is_saturated_array = np.zeros(n_span+1)
                X = np.array([0.0, 0.0, 0.0, 0.0])
                is_saturated = False
            
            # Record the results in the DataFrame
            V_array = 1 - np.exp(-X_array[3, :])
            for j_s in range(n_span+1):
                t_j = t_piece_min + (t_piece_max - t_piece_min)/n_span*j_s
                result_timestep = [t_j, X_array[0, j_s], X_array[1, j_s], X_array[2, j_s], X_array[3, j_s],\
                                   V_array[j_s], is_saturated_array[j_s]]
                # post_process steps
                if self.post_process_ts:
                    ts = self.compute_ts(t_j) # compute saturation and growth time as post-processing
                    result_timestep.append(ts)
                if self.post_process_tg:
                    tg = self.compute_tg(t_j) # compute saturation and growth time as post-processing
                    result_timestep.append(tg)
                results[i_t*n_span + j_s, :] =  result_timestep
                
        return results
    
def get_kinetic_constants(nucleation_type):
    """
    Get the kinetic model constants for different type of nucleation
    Inputs:
        nucleation_type (int): type of nucleation
            0 - volumetric
            1 - surface
    Returns:
        constants, constants1
            constants - parameters for initiating PTKinetics class
            constants1 - parameters for initiating MO_KINETICS class
    """
    if nucleation_type == 0:
        _constants = PTKinetics.Constants(
            R=8.31446,
            k=1.38e-23,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.46,
            fs=6e-4,
            K0=3.65e38,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=0
        )
        _constants1 = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=1e-2,
            A=np.exp(-18.0),
            n=3.2,
            dHa=274e3,
            V_star_growth=3.3e-6,
            gamma=0.46,
            fs=6e-4,
            K0=3.65e38,
            Vm=4.05e-5,
            dS=7.7,
            dV=3.16e-6,
            nucleation_type=0
        )
        pTKinetics = PTKinetics(_constants)
    elif nucleation_type == 1:
        _constants = PTKinetics.Constants(
            R=8.31446,
            k=1.38e-23,
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
        _constants1 = MO_KINETICS.Constants(
            R=8.31446,
            k=1.38e-23,
            kappa=1e-6,
            D=100e3,
            d0=1e-2,
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
    else:
        raise NotImplementedError()
    return _constants, _constants1