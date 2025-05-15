"""
Analytical Solution Module - HaMaGeoLib

MIT License

Author: Haoyuan Li
Affiliation: UC Davis, EPS Department
Email: hylli@ucdavis.edu

Overview:
    This module provides analytical solutions for geodynamic simulations, including 
    viscosity laws, deformation mechanisms, and stress-strain relationships.
"""

import numpy as np
from collections import namedtuple
from scipy.special import erf

########################################
# Classes and Functions for analytical solution of slab corner flow
########################################
class VISCOUS_CORNER_FLOW():

    def __init__(self):
        '''
        Initiation
        '''
        self.theta_d = 0.0  # dip angle
        self.U = 0.0  # subducting velocity
        self.Aoc = 0.0  # coefficents for solution in the oceanic corner
        self.Boc = 0.0
        self.Coc = 0.0
        self.Doc = 0.0
        self.Aac = 0.0  # coefficents for solution in the arc corner
        self.Bac = 0.0
        self.Cac = 0.0
        self.Dac = 0.0
        self.oceanic_corner_solved = False
        self.arc_corner_solved = False
    
    def Reset(self):
        '''
        reset all the values
        '''
        self.__init__()

    def PrescribeDipVelocity(self, theta_d, U):
        '''
        Prescribe dip angle and velocity
        '''
        self.theta_d = theta_d
        self.U = U
    
    def PrescribeDomainViscosity(self, mu):
        '''
        Prescribe viscosity to the whole domain
        '''
        self.mu = mu

    def SolveOceanicCorner(self):
        '''
        solve coefficents in the oceanic corner
        '''
        theta_d = self.theta_d
        denominator = (np.pi - theta_d)**2.0 - np.sin(theta_d)**2.0
        dividend_C = np.sin(theta_d)**2.0 - (np.pi - theta_d) * np.sin(theta_d)
        dividend_D = (np.pi - theta_d) * (np.cos(theta_d) - 1.0) + np.sin(theta_d)\
        - np.cos(theta_d) * np.sin(theta_d)
        self.Coc = dividend_C / denominator * self.U
        self.Doc = dividend_D / denominator * self.U
        self.Aoc = - self.Coc * np.pi
        self.Boc =  - (self.U + self.Doc * np.pi + self.Coc)
        self.oceanic_corner_solved = True
    
    def SolveArcCorner(self):
        '''
        solve coeffients in the arc corner
        '''
        theta_d = self.theta_d
        denominator = theta_d**2.0 - np.sin(theta_d)**2.0
        dividend_C = theta_d * np.sin(theta_d)
        dividend_D = np.sin(theta_d) - theta_d * np.cos(theta_d)
        self.Cac = dividend_C / denominator * self.U
        self.Dac = dividend_D / denominator * self.U
        self.Aac = 0.0 
        self.Bac = -self.Cac
        self.arc_corner_solved = True
    
    def ProblemSolved(self):
        '''
        Return whether the problem is solved
        '''
        solved = self.oceanic_corner_solved and self.arc_corner_solved\
        and self.U > 0.0 and self.mu > 0.0
        return solved
    
    def GetFlowVelocity(self, x, y):
        '''
        return flow velocity in a viscous corner flow model
        Inputs:
            x, y
            theta_d - the dip angle
        Returns:
            u - velocity along x
            v - velocity along y
            pos - 0 (in the arc corner) or 1 (in the oceanic corner)
        '''
        assert(self.ProblemSolved())  # assert solution is derived
        theta = np.arctan2(y, x)
        pos = GetDomain(x, y, self.theta_d)
        if pos == 0:
            # use the coefficients for the arc corner
            u, v = GeneralSolution(x, y, self.Aac, self.Bac, self.Cac, self.Dac)
        elif pos == 1:
            # use the coefficients for the oceanic corner
            u, v = GeneralSolution(x, y, self.Aoc, self.Boc, self.Coc, self.Doc)
        else:
            raise ValueError("A wrong value for \'pos\'")
        return u, v, pos
    
    def GetPressure(self, x, y):
        '''
        return flow velocity in a viscous corner flow model
        Inputs:
            x, y
            theta_d - the dip angle
        Returns:
            u - velocity along x
            v - velocity along y
            pos - 0 (in the arc corner) or 1 (in the oceanic corner)
        '''
        assert(self.ProblemSolved())  # assert solution is derived
        theta = np.arctan2(y, x)
        pos = GetDomain(x, y, self.theta_d)
        if pos == 0:
            P = PressureSolution(x, y, self.mu, self.Cac, self.Dac)
        elif pos == 1:
            P = PressureSolution(x, y, self.mu, self.Coc, self.Doc)
        else:
            raise ValueError("A wrong value for \'pos\'")
        return P, pos
    
    def ExportCD(self):
        '''
        export C and D in the coefficients
        '''
        assert(self.ProblemSolved())  # assert solution is derived
        return self.Cac, self.Dac, self.Coc, self.Doc
    
def GetDomain(x, y, theta_d):
    '''
    Get which domain a point belongs to
    Inputs:
        x
        y
        theta_d - dip angle
    return:
        pos - 0 (in the arc corner) or 1 (in the oceanic corner)
    '''
    theta = np.arctan2(y, x)
    pos = 0
    if theta > theta_d:
        pos = 1
    return pos


def GeneralSolution(x, y, A, B, C, D):
    '''
    Return the value of velocities from the general solution of the viscous corner flow model
    Inputs:
        x, y
        A, B, C, D - coefficents in the general solution
    Return:
        u - velocity along x
        v - velocity along y
    '''
    u = -B - D * np.arctan2(y, x) + (C*x + D*y)*(-x / (x**2.0 + y**2.0))
    v = A + C * np.arctan2(y, x) + (C*x + D*y)*(-y / (x**2.0 + y**2.0))
    return u, v


def PressureSolution(x, y, mu, C, D):
    '''
    Return the value of pressure from the general solution of the viscous corner flow model
    Inputs:
        x, y
        mu - viscosity
        C, D - coefficents in the general solution
    Return:
        P - pressure
    '''
    P = -2 * mu * (C * x + D * y) / (x**2.0 + y**2.0)
    return P


########################################
# classes and functions for analytical solution from England&Wilkins, 2004
########################################
class WK2004:
    """
    Class representing the WK2004 slab thermal model.

    Attributes:
        age (float): Age of the subducting plate (Ma).
        U (float): Convergence rate (m/s).
        theta_d (float): Slab dip angle (radians).
        a (float): Slab thickness (m).
        zw (float): Overriding plate thickness (m)
        Ta (float): Adiabatic mantle temperature (K).
        Tsf (float): Sea floor temperature (K).
        kappa (float): Thermal diffusivity
        rho (float): Density
        cp (float): specific heat
        plate_T_model (int): Plate temperature model type.
            0 - Linear variation
    """

    Constants = namedtuple("Constants", ["age", "U", "theta_d", "a", "zw", "Ta", "Tsf", "kappa", "rho", "cp", "plate_T_model"])

    year = 365.0 * 24.0 * 3600.0

    def __init__(self, constants=None):
        """
        Initialize the WK2004 model with default or user-defined parameters.
        """
        if constants is None:
            constants = self.Constants(
                age=50.0e6*self.year,            # s
                U=0.05/self.year,                # m/s
                theta_d=np.pi/4, # radians
                a=100e3,               # meters
                zw=40e3,        # meters
                Ta=1573.15,             # Kelvin
                Tsf=273.15,  # kelvin
                kappa = 8e-7,    # m^2 ^-1
                rho = 3300.0,    # kg m^-3 
                cp = 1e3,   # J kg^-1 K^-1
                plate_T_model=0
            )

        self.age = constants.age
        self.U = constants.U
        self.theta_d = constants.theta_d
        self.a = constants.a
        self.zw = constants.zw
        self.Ta = constants.Ta
        self.Tsf = constants.Tsf
        self.kappa = constants.kappa
        self.rho = constants.rho
        self.cp = constants.cp
        self.K = self.rho * self.cp * self.kappa  # compute thermal conductivity accordingly
        self.plate_T_model = constants.plate_T_model
    
    def dimensionless_distance(self, depth):
        '''
        Derive the dimensionless distance
        Inputs:
            depth (float)
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")
       
        return self.U * self.get_r(depth) * self.theta_d**2.0 / self.kappa
    
    def get_r(self, depth):
        '''
        compute the value of r, this is measured from the base of the overriding lithosphere
        to a point on slab surface
        '''
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")
        
        return (depth - self.zw) / np.sin(self.theta_d)
    
    def epsl_factor(self):
        return 1 - 2.0/5.0 / np.cos(2.0*self.theta_d/5.0)
    
    def length_scale(self):
        '''
        Derive a length scale (eq 16)
        '''
        return 400.0 * self.kappa / (self.U * self.theta_d**2.0)
    
    def advective_thickness(self, depth):
        '''
        Derive the advective thickness on the slab top
        Inputs:
            depth (float)
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        # compute value 
        power_base = 16.0 / (9.0*np.pi**0.5*self.epsl_factor()* self.dimensionless_distance(depth))
        alpha = self.get_r(depth) * self.theta_d * (power_base)**(1.0/3.0)
        return alpha
    
    def advective_velocity(self, depth):
        '''
        Derive the advective velocity in the slab surface boundary
        Note this value is U in the paper, but we used that for convergence rate
        Inputs:
            depth (float)
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        # compute value 
        nominator = 9.0 * np.pi**2.0 * self.U * self.epsl_factor() * self.advective_thickness(depth)**2.0
        denominator = 16.0 * self.get_r(depth)**2.0 * self.theta_d
        u = nominator / denominator
        return u
    
    def advective_timescale(self, depth):
        '''
        Derive the advective timescale on the slab top
        Inputs:
            depth (float)
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        # compute value 
        return self.advective_thickness(depth) / self.advective_velocity(depth)
    
    def diffusive_thickness(self, depth):
        '''
        Derive the diffusive thickness on the slab top
        Inputs:
            depth (float)
        '''
        ts = depth / np.sin(self.theta_d) / self.U  # time of subduction
        return (self.kappa * ts)**0.5
    
    def diffusive_timescale(self, depth):
        '''
        Derive the diffusive timescale on the slab top
        Inputs:
            depth (float)
        '''
        return self.advective_thickness(depth)**2.0 / (np.pi**2.0 * self.kappa)

    # todo_top 
    def top_thickness(self, depth):
        '''
        Derive thermal thickness on top of interface, from either
        the advective or the diffusive thickness
        Inputs:
            depth (float)
        '''
        if isinstance(depth, (float, np.floating)):
            if depth <= self.zw:
                top_thickness = self.diffusive_thickness(depth)
            else:
                top_thickness = self.advective_thickness(depth)
        elif isinstance(depth, np.ndarray):
            mask = (depth <= self.zw)
            top_thickness = np.zeros(depth.shape)
            top_thickness[mask] = self.diffusive_thickness(depth[mask])
            top_thickness[~mask] = self.advective_thickness(depth[~mask])
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        return top_thickness


    def peclet_number(self, depth):
        '''
        Derive the Peclet number on the slab top
        Inputs:
            depth (float)
        '''
        return self.advective_velocity(depth) * self.advective_thickness(depth) / (np.pi**2.0 * self.kappa)
    
    def bd_temperature(self, depth, x):
        '''
        Derive the temperature in the advective boundary on top of the slab
        Inputs:
            depth (float)
            x - distance perpendicular to the slab surface
        '''
        Ts = self.ss_temperature(depth)# slab surface temperature
        T1 = self.mw_temperature(depth) # mantle wedge temperature

        Tbd = Ts + (T1 - Ts)*erf(x/self.advective_thickness(depth))

        return Tbd

    def mw_temperature(self, depth):
        '''
        Derive the temperature in the mantle wedge (eq 15)
        Inputs:
            depth (float)
            x - distance perpendicular to the slab surface
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        # get the length scale R
        R = self.length_scale()

        # Switch between float numbers and numpy arrays
        if type(depth) in [float, np.float64]:
            r = self.get_r(depth)
            if r < R:
                exponential = 0.5 * (1.0 - (R/r)**(1.0/3.0))
                T1 = (self.Ta - self.Tsf) * np.exp(exponential) + self.Tsf  # modified with a seafloor temperature, using Kelvin
            else:
                T1 = self.Ta
            return T1
        elif type(depth) == np.ndarray:
            r = self.get_r(depth)
            T1 = np.full(r.shape, self.Ta)
            mask = (r < R)
            exponential = 0.5 * (1.0 - (R/r[mask])**(1.0/3.0))
            T1[mask] = (self.Ta - self.Tsf) * np.exp(exponential) + self.Tsf
            return T1
        else:
            raise TypeError()
    
    def ss_temperature(self, depth, **kwargs):
        '''
        Derive the slab surface temperature (eq 17)
        Inputs:
            depth
            x - distance perpendicular to the slab surface
            kwargs:
                debug(bool): print debug info
                use_top_thickness (bool): whether to use the top thickness or stick with the original advective thickness
        '''
        debug = kwargs.get("debug", False)
        use_top_thickness = kwargs.get("use_top_thickness", False)
        if use_top_thickness:
            # with this option, the requirement of depth being deeper than
            # overriding plate thickness is lifted.
            alpha = self.top_thickness(depth)

            # mantle wedge temperature maximum
            # switch computation of T1 in a few cases
            if isinstance(depth, (float, np.floating)):
                if depth <= self.zw:
                    T1 = self.Tsf
                else:
                    T1 = self.mw_temperature(depth)
            elif isinstance(depth, np.ndarray):
                mask = (depth <= self.zw)
                T1 = np.full(depth.shape, self.Tsf)
                # T1[mask] = 0.0
                T1[~mask] = self.mw_temperature(depth[~mask])
                    
        else:
            # assert depth deeper than overriding plate thickness
            if isinstance(depth, (float, np.floating)):
                assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
            elif isinstance(depth, np.ndarray):
                assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
            else:
                raise TypeError("depth must be a float or numpy.ndarray")
            alpha = self.advective_thickness(depth)
        
            # mantle wedge temperature maximum
            T1 = self.mw_temperature(depth)

        # compute factors in the formula
        # modified with a seafloor temperature, using Kelvin
        erf_multiplier = np.pi**0.5 * alpha * (self.Ta - self.Tsf) / (2.0 * self.a)
        erf_factor = self.a / (2 * self.diffusive_thickness(depth))
        denominator = (1 + np.pi**0.5 * alpha / (2.0 * self.diffusive_thickness(depth)))

        # results: modified with a seafloor temperature, using Kelvin
        Ts = ((T1 - self.Tsf) + erf_multiplier * erf(erf_factor)) / denominator + self.Tsf

        if debug:
            print("alpha = ",  alpha)
            print(f"T1 = {T1}, self.Tsf = {self.Tsf}, erf_multiplier = {erf_multiplier}, "
                f"erf_factor = {erf_factor}, denominator = {denominator}, "
                f"erf(erf_factor) = {erf(erf_factor)}, Ts = {Ts}")

        return Ts
    
    def ss_temperature_approx(self, depth):
        '''
        Derive the slab surface temperature, by the approximate formula (eq 18),
        provided age is older than 50 Ma. This result in alpha << a
        Inputs:
            depth (float)
            x - distance perpendicular to the slab surface
        '''
        # assert depth deeper than overriding plate thickness
        if isinstance(depth, (float, np.floating)):
            assert depth > self.zw, f"Depth {depth} must be greater than self.zw ({self.zw})"
        elif isinstance(depth, np.ndarray):
            assert np.all(depth > self.zw), f"All depths must be greater than self.zw ({self.zw})"
        else:
            raise TypeError("depth must be a float or numpy.ndarray")

        # assert old plage ages 
        assert(self.age > 50.0 * 1e6 * self.year)
        
        T1 = self.mw_temperature(depth) # mantle wedge temperature computed at slab interface
        denominator = (1 + np.pi**0.5 * self.advective_thickness(depth) / (2.0 * self.diffusive_thickness(depth)))

        # modified with a seafloor temperature, using Kelvin
        Ts = (T1 - self.Tsf) / denominator + self.Tsf
        return Ts
    
    def ss_hf_top(self, depth):
        '''
        Derive the slab surface heat flux on the top side (eq A25)
        Inputs:
            depth
            x - distance perpendicular to the slab surface
        '''
        T1 = self.mw_temperature(depth) # mantle wedge temperature computed at slab interface
        Ts = self.ss_temperature(depth)
        Q_top = self.K / np.pi**2.0 * (T1 - Ts) / self.advective_thickness(depth)

        return Q_top
    
    def ss_hf_bot(self, depth):
        '''
        Derive the slab surface heat flux on the bottom side (eq A34)
        Note this is an approximate formula
        Inputs:
            depth
            x - distance perpendicular to the slab surface
        '''
        Ts = self.ss_temperature(depth)

        erf_factor = self.a / (2 * self.diffusive_thickness(depth))

        # compute bottom heat flux of the slab surface 
        # modified with a seafloor temperature, using Kelvin 
        Q_bot = self.K * ((Ts - self.Tsf) / self.diffusive_thickness(depth) - (self.Ta-self.Tsf) / self.a * erf(erf_factor))

        return Q_bot 