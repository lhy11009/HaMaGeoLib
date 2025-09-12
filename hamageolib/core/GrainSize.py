from typing import NamedTuple, Optional
import numpy as np

# todo_ggrowth
class GrainGrowthParams(NamedTuple):
    grain_growth_rate_constant: float
    m: float
    grain_growth_activation_energy: float
    grain_growth_activation_volume: float

class GrainGrowthModel:
    """
    Grain-growth kinetics model (uses SI units).
    Accepts either explicit args or a GrainGrowthParams named tuple.

    Units:
        P [Pa], T [K], E [J/mol], V [m^3/mol]
    """
    def __init__(
        self,
        grain_growth_rate_constant: Optional[float] = None,
        m: Optional[float] = None,
        grain_growth_activation_energy: Optional[float] = None,
        grain_growth_activation_volume: Optional[float] = None,
        *,
        params: Optional[GrainGrowthParams] = None
    ):
        # Enforce exactly one init path
        explicit = all(v is not None for v in (
            grain_growth_rate_constant, m, grain_growth_activation_energy, grain_growth_activation_volume
        ))
        if params is not None and explicit:
            raise ValueError("Provide either 'params' OR the four explicit arguments, not both.")
        if params is None and not explicit:
            raise ValueError("Provide either 'params' or all four explicit arguments.")

        if params is not None:
            self.k = float(params.grain_growth_rate_constant)
            self.m = float(params.m)
            self.E = float(params.grain_growth_activation_energy)
            self.V = float(params.grain_growth_activation_volume)
        else:
            self.k = float(grain_growth_rate_constant)  # type: ignore[arg-type]
            self.m = float(m)                            # type: ignore[arg-type]
            self.E = float(grain_growth_activation_energy)  # type: ignore[arg-type]
            self.V = float(grain_growth_activation_volume)  # type: ignore[arg-type]

        # Universal gas constant [J/(mol·K)] — fixed, not parsed
        self.R = 8.314462618

    def calculate_growth_rate(self, grain_size, P, T):
        """
        k / (m * g^(m-1)) * exp( - (E + P*V) / (R*T) )
        """
        g = np.asarray(grain_size, dtype=float)
        P = np.asarray(P, dtype=float)
        T = np.asarray(T, dtype=float)

        if np.any(g <= 0):
            raise ValueError("grain_size must be > 0.")
        if np.any(T <= 0):
            raise ValueError("Temperature T must be > 0 K.")

        denom = self.m * np.power(g, self.m - 1.0)
        if np.any(denom == 0.0):
            raise ZeroDivisionError("m * grain_size^(m-1) is zero; check m and grain_size.")

        with np.errstate(over="ignore", under="ignore"):
            arrhenius = np.exp(-(self.E + P * self.V) / (self.R * T))

        return (self.k / denom) * arrhenius
    

    def grain_size_at_time(self, g0, t, P, T):
        """
        Analytical solution for constant P,T:
            g(t) = ( g0^m + k * exp(-(E + P*V)/(R*T)) * t )^(1/m)

        Args:
            g0 : float or array-like [m]    initial grain size (>0)
            t  : float or array-like [s]    time (>=0)
            P  : float or array-like [Pa]
            T  : float or array-like [K]
        Returns:
            ndarray or float with broadcasted shape
        """
        g0 = np.asarray(g0, dtype=float)
        t  = np.asarray(t,  dtype=float)
        P  = np.asarray(P,  dtype=float)
        T  = np.asarray(T,  dtype=float)

        if np.any(g0 <= 0): raise ValueError("g0 must be > 0.")
        if np.any(T  <= 0): raise ValueError("T must be > 0 K.")
        if np.any(t  <  0): raise ValueError("t must be >= 0.")

        A = np.exp(-(self.E + P * self.V) / (self.R * T))
        inside = np.power(g0, self.m) + self.k * A * t
        if np.any(inside <= 0):
            raise ValueError("Nonpositive value inside the m-th root; check parameters.")
        return np.power(inside, 1.0 / self.m)

    def time_to_reach_size(self, g0, g1, P, T):
        """
        Inverse relation (constant P,T):
            t = ( g1^m - g0^m ) / ( k * exp(-(E+P*V)/(R*T)) )
        """
        g0 = np.asarray(g0, dtype=float)
        g1 = np.asarray(g1, dtype=float)
        P  = np.asarray(P,  dtype=float)
        T  = np.asarray(T,  dtype=float)

        if np.any(g0 <= 0) or np.any(g1 <= 0): raise ValueError("g0, g1 must be > 0.")
        if np.any(T  <= 0): raise ValueError("T must be > 0 K.")

        A = np.exp(-(self.E + P * self.V) / (self.R * T))
        t = (np.power(g1, self.m) - np.power(g0, self.m)) / (self.k * A)
        return t
