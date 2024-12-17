"""
MIT License

Copyright (c) 2025 Haoyuan Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Module Name: melt
Author: Haoyuan Li
Purpose: This module provides a class to record and manage up-to-date experimental results 
         on the melting of rocks for geodynamic modeling and analysis.
"""
from scipy.interpolate import interp1d
import numpy as np


class ROCK_MELTING:
    '''
    A class to record and manage experimental results on the melting of rocks.
    All inputs and outputs:
        - Pressure (P): in Pascals (Pa)
        - Temperature (T): in Kelvin (K)
    '''

    def __init__(self):
        """
        Initialize the ROCK_MELTING instance.

        Attributes:
            solidus_data (dict): A dictionary to store solidus functions.
        """
        self.solidus_data = {
            "dry_peridotite": self.dry_peridotite_solidus,
            "water_saturated_peridotite_low_pressure": self.water_saturated_peridotite_solidus_low,
            "water_saturated_peridotite_high_pressure": self.water_saturated_peridotite_solidus_high,
            "eclogite": self.eclogite_solidus,
            "peridotite_aH2O_0.1": self.peridotite_aH2O_0_1,
            "peridotite_aH2O_0.3": self.peridotite_aH2O_0_3,
            "peridotite_aH2O_0.5": self.peridotite_aH2O_0_5,
            "peridotite_aH2O_0.7": self.peridotite_aH2O_0_7,
            "basalt_aH2O_0_3": self.basalt_aH2O_0_3,
            "basalt_aH2O_0_8": self.basalt_aH2O_0_8,
            "basalt_aH2O_1_3": self.basalt_aH2O_1_3,
            "basalt_aH2O_1_8": self.basalt_aH2O_1_8,
        }
        # Initialize interpolators for basalt solidus curves
        self._init_basalt_aH2O_0_3()
        self._init_basalt_aH2O_0_8()
        self._init_basalt_aH2O_1_3()
        self._init_basalt_aH2O_1_8()

    def _assert_pressure(self, P, min_val, max_val, name):
        """
        Assert the validity of the pressure input (P).

        Args:
            P (float or np.ndarray): The input pressure(s) in Pascals (Pa).
            min_val (float): The minimum valid pressure in Pascals (Pa).
            max_val (float): The maximum valid pressure in Pascals (Pa).
            name (str): The name of the solidus function for error reporting.
        """
        if isinstance(P, float):
            assert min_val <= P <= max_val, f"Pressure out of range for {name} Solidus ({min_val}-{max_val} Pa)."
        elif isinstance(P, np.ndarray):
            assert np.all((min_val <= P) & (P <= max_val)), (
                f"Pressure array contains values out of range for {name} Solidus ({min_val}-{max_val} Pa)."
            )
        else:
            raise TypeError(f"Invalid type for pressure P in {name} Solidus. Must be float or numpy.ndarray.")
    
    def water_saturated_peridotite_solidus_low(self, P):
        """
        Water-Saturated Peridotite Solidus (Low Pressure).

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = -0.0842*P^4 - 1.7404*P^3 + 36.777*P^2 - 191.69*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 6e9, "Water-Saturated Peridotite (Low Pressure)")
        T_C = (
            -0.0842 * P_GPa**4
            - 1.7404 * P_GPa**3
            + 36.777 * P_GPa**2
            - 191.69 * P_GPa
            + 1120.7
        )
        return T_C + 273.15  # Convert to K

    def water_saturated_peridotite_solidus_high(self, P):
        """
        Water-Saturated Peridotite Solidus (High Pressure).

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = 16.777*P^2 - 149.5*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 6e9, 12e9, "Water-Saturated Peridotite (High Pressure)")
        T_C = 16.777 * P_GPa**2 - 149.5 * P_GPa + 1120.7
        return T_C + 273.15  # Convert to K
    
    def eclogite_solidus(self, P):
        """
        Eclogite (Garnet Pyroxenite) Solidus.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = -3.4524*P^2 + 120.95*P + 1096.8
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 3e9, 7.5e9, "Eclogite")
        T_C = -3.4524 * P_GPa**2 + 120.95 * P_GPa + 1096.8
        return T_C + 273.15  # Convert to K
    
    def peridotite_aH2O_0_1(self, P):
        """
        Peridotite Solidus with aH2O = 0.1.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = -2.848*P^2 + 104.312*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 10e9, "Peridotite aH2O=0.1")
        T_C = -2.848 * P_GPa**2 + 104.312 * P_GPa + 1120.7
        return T_C + 273.15  # Convert to K

    def peridotite_aH2O_0_3(self, P):
        """
        Peridotite Solidus with aH2O = 0.3.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = 1.659*P^2 + 47.152*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 10e9, "Peridotite aH2O=0.3")
        T_C = 1.659 * P_GPa**2 + 47.152 * P_GPa + 1120.7
        return T_C + 273.15  # Convert to K

    def peridotite_aH2O_0_5(self, P):
        """
        Peridotite Solidus with aH2O = 0.5.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = 6.168*P^2 - 10.01*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 10e9, "Peridotite aH2O=0.5")
        T_C = 6.168 * P_GPa**2 - 10.01 * P_GPa + 1120.7
        return T_C + 273.15  # Convert to K

    def peridotite_aH2O_0_7(self, P):
        """
        Peridotite Solidus with aH2O = 0.7.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = 10.677*P^2 - 67.173*P + 1120.7
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 10e9, "Peridotite aH2O=0.7")
        T_C = 10.677 * P_GPa**2 - 67.173 * P_GPa + 1120.7
        return T_C + 273.15  # Convert to K


    def dry_peridotite_solidus(self, P):
        """
        Dry Peridotite Solidus from Hirschmann, Gcubed, 2010.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original equation (valid for P in GPa and T in °C):
        T(C) = -5.104*P^2 + 132.899*P + 1120.61
        """
        P_GPa = P / 1e9  # Convert Pa to GPa for computation
        self._assert_pressure(P, 0, 10e9, "Dry Peridotite")
        T_C = -5.104 * P_GPa**2 + 132.899 * P_GPa + 1120.61
        return T_C + 273.15  # Convert to K

    def basalt_aH2O_0_3(self, P):
        """
        Basalt Solidus with aH2O = 0.3, based on experimental data.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original data:
        - Pressure (P) in GPa
        - Temperature (T) in °C
        """
        self._assert_pressure(P, 0.078e9, 3.665e9, "Basalt aH2O=0.3")
        return self._basalt_aH2O_0_3_interpolator(P / 1e9) + 273.15

    def _init_basalt_aH2O_0_3(self):
        """
        Initialize the interpolator for the basalt solidus with 0.3 wt% H2O.
        """
        # Original data in °C and GPa
        temperatures_C = np.array([
            947.560975609756, 897.560975609756, 840.688760559198, 869.6340519969681,
            893.7289499800734, 906.932147629504, 836.7549953504365, 762.9441505364581,
            706.0710015706929, 730.1631621638486, 746.9933549634992, 751.7805890060523,
            754.0829537655208, 753.9519248767705, 756.2238722156362, 741.4634146341463,
            725.6097560975609, 756.0975609756097, 781.7073170731708
        ])
        pressures_GPa = np.array([
            0.07858946265243958, 0.08884336890243958, 0.13967835960271628, 0.35535559393290512,
            0.6000085958318024, 0.8560784252436132, 0.8954160773312285, 0.9463112159976241,
            0.9917206510951868, 1.2369439071566731, 1.498065764023211, 1.6257253384912957,
            1.915860735009671, 2.2408123791102515, 2.606382978723404, 2.767578125,
            2.9314619855182933, 3.31321455792683, 3.665527343750001
        ])

        # Create the interpolation function
        self._basalt_aH2O_0_3_interpolator = interp1d(
            pressures_GPa,  # Convert back to GPa for the interpolation
            temperatures_C,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

    def basalt_aH2O_0_8(self, P):
        """
        Basalt Solidus with aH2O = 0.8, based on experimental data.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original data:
        - Pressure (P) in GPa
        - Temperature (T) in °C
        """
        self._assert_pressure(P, 0.078e9, 3.665e9, "Basalt aH2O=0.8")
        return self._basalt_aH2O_0_8_interpolator(P / 1e9) + 273.15

    def _init_basalt_aH2O_0_8(self):
        """
        Initialize the interpolator for the basalt solidus with 0.8 wt% H2O.
        """
        # Original data in °C and GPa
        temperatures_C = np.array([
            947.560975609756, 897.560975609756, 840.688760559198, 869.6340519969681,
            893.7289499800734, 906.932147629504, 836.7549953504365, 762.9441505364581,
            706.0710015706929, 730.1631621638486, 746.9933549634992, 751.7805890060523,
            754.0829537655208, 753.9519248767705, 756.2238722156362, 698.1406376739253,
            725.6097560975609, 756.0975609756097, 781.7073170731708
        ])
        pressures_GPa = np.array([
            0.07858946265243958, 0.08884336890243958, 0.13967835960271628, 0.35535559393290512,
            0.6000085958318024, 0.8560784252436132, 0.8954160773312285, 0.9463112159976241,
            0.9917206510951868, 1.2369439071566731, 1.498065764023211, 1.6257253384912957,
            1.915860735009671, 2.2408123791102515, 2.606382978723404, 2.652804642166344,
            2.9314619855182933, 3.31321455792683, 3.665527343750001
        ])

        # Create the interpolation function
        self._basalt_aH2O_0_8_interpolator = interp1d(
            pressures_GPa,  # Convert back to GPa for the interpolation
            temperatures_C,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

    def basalt_aH2O_1_3(self, P):
        """
        Basalt Solidus with aH2O = 1.3, based on experimental data.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original data:
        - Pressure (P) in GPa
        - Temperature (T) in °C
        """
        self._assert_pressure(P, 0.078e9, 3.665e9, "Basalt aH2O=1.3")
        return self._basalt_aH2O_1_3_interpolator(P / 1e9) + 273.15

    def _init_basalt_aH2O_1_3(self):
        """
        Initialize the interpolator for the basalt solidus with 1.3 wt% H2O.
        """
        # Original data in °C and GPa
        temperatures_C = np.array([
            947.560975609756, 897.560975609756, 831.7073170731708, 762.1951219512196,
            710.9756097560976, 698.9712672365384, 658.9770387471143, 701.2361951706496,
            730.1631621638486, 746.9933549634992, 751.7805890060523, 754.0829537655208,
            753.9519248767705, 756.2238722156362, 698.1406376739253, 725.6097560975609,
            756.0975609756097, 781.7073170731708
        ])
        pressures_GPa = np.array([
            0.07858946265243958, 0.08884336890243958, 0.1455078125000007, 0.3485494474085371,
            0.5345488757621958, 0.5928433268858804, 0.7785299806576404, 0.9758220502901352,
            1.2369439071566731, 1.498065764023211, 1.6257253384912957, 1.915860735009671,
            2.2408123791102515, 2.606382978723404, 2.652804642166344, 2.9314619855182933,
            3.31321455792683, 3.665527343750001
        ])

        # Create the interpolation function
        self._basalt_aH2O_1_3_interpolator = interp1d(
            pressures_GPa,  # Convert back to GPa for the interpolation
            temperatures_C,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

    def basalt_aH2O_1_8(self, P):
        """
        Basalt Solidus with aH2O = 1.8, based on experimental data.

        All inputs and outputs are in SI units:
        - Input pressure (P): in Pascals (Pa)
        - Output temperature (T): in Kelvin (K)

        Original data:
        - Pressure (P) in GPa
        - Temperature (T) in °C
        """
        self._assert_pressure(P, 0.078e9, 3.665e9, "Basalt aH2O=1.8")
        return self._basalt_aH2O_1_8_interpolator(P / 1e9) + 273.15

    def _init_basalt_aH2O_1_8(self):
        """
        Initialize the interpolator for the basalt solidus with 1.8 wt% H2O.
        """
        # Original data in °C and GPa
        temperatures_C = np.array([
            947.560975609756, 897.560975609756, 831.7073170731708, 762.1951219512196,
            710.9756097560976, 663.4146341463414, 657.3170731707316, 653.6585365853658,
            650, 646.3414634146342, 647.560975609756, 650, 659.7560975609756,
            675.609756097561, 697.560975609756, 725.6097560975609, 756.0975609756097,
            781.7073170731708
        ])
        pressures_GPa = np.array([
            0.07858946265243958, 0.08884336890243958, 0.1455078125000007, 0.3485494474085371,
            0.5345488757621958, 0.73823361280487845, 0.784929973323171, 0.8434165396341466,
            0.9370593559451223, 1.2182021722560982, 1.5873785251524396, 1.8569812309451226,
            2.079923304115854, 2.344059641768293, 2.6435308689024396, 2.9314619855182933,
            3.31321455792683, 3.665527343750001
        ])

        # Create the interpolation function
        self._basalt_aH2O_1_8_interpolator = interp1d(
            pressures_GPa,  # Convert back to GPa for the interpolation
            temperatures_C,
            kind='linear',
            bounds_error=False,
            fill_value="extrapolate"
        )

