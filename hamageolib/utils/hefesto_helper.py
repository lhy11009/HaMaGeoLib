import numpy as np
import os
import json
from scipy import interpolate
from shutil import copy2, rmtree, copytree
from ..research.haoyuan_2d_subduction.legacy_utilities import UNITCONVERT, func_name, re_neat_word,\
    my_assert, JSON_OPT, var_subs, SLURM_OPERATOR
from ..research.haoyuan_2d_subduction.legacy_tools import LINEARPLOT

class LOOKUP_TABLE():
    '''
    Class for handling HeFESTo / Perple_X lookup tables, including reading, processing,
    interpolation, and exporting thermodynamic data.
    Attributes:
        header (dict): metadata describing columns and units of the dataset.
        data (np.ndarray): main numerical dataset.
        maph_data (np.ndarray): most abundant phase data (string array).
        version (str): version of the lookup table format.
        UnitConvert (callable): unit conversion utility.
        first_dimension_name (str): name of first independent variable.
        second_dimension_name (str): name of second independent variable.
        min1, delta1, number1 (float, float, int): metadata for first dimension.
        min2, delta2, number2 (float, float, int): metadata for second dimension.
        indexes (list[int]): selected indices for output.
        number_out1, number_out2 (int): output grid size.
        delta_out1, delta_out2 (float): output intervals.
        fort_56_header (dict): predefined header for fort.56 files.
        perplex_header (dict): mapping of Perple_X headers.
        ounit (dict): output units.
        PS_data (np.ndarray): pressure-entropy interpolated data.
    '''
    def __init__(self):
        '''
        Initialize lookup table structure and default metadata.
        '''
        self.header = {}
        self.data = []
        # todo_maph
        self.maph_data = []
        self.version = "1.0.0"
        self.UnitConvert = UNITCONVERT()
        self.first_dimension_name = None
        self.min1 = 0.0 
        self.delta1 = 0.0 
        self.number1 = 0
        self.second_dimension_name = None
        self.min2 = 0.0
        self.delta2 = 0.0
        self.number2 = 0
        self.indexes = []  # indexes of output data
        self.number_out1 = 0 # number of output
        self.number_out2 = 0
        self.delta_out1 = 0.0  # intervals used to outptu
        self.delta_out2 = 0.0 
        self.fort_56_header = { "Pressure": {"col": 0, "unit": "GPa"}, "Depth": {"col": 1, "unit": "km"},\
                               "Temperature": {"col": 2, "unit": "K"}, "Density": {"col": 3, "unit": "g/cm^3"},\
                                "Bulk sound Velocity": {"col": 4, "unit": "km/s"}, "VS": {"col": 5, "unit": "km/s"},\
                                "VP": {"col": 6, "unit": 'km/s'}, "VS modified by attenuation": {"col": 7, "unit": 'km/s'},\
                                "VP modified by attenuation": {"col": 8, "unit": 'km/s'}, "Enthalpy":{"col": 9, "unit": 'kJ/g'},\
                                "Entropy": {"col": 10, "unit": "J/g/K"}, "Thermal_expansivity": {"col": 11, "unit": "10-5 K-1"},\
                                "Isobaric_heat_capacity": {"col": 12, "unit": "J/g/K"}, "isothermal bulk modulus": {"col": 13, "unit": "GPa"},\
                                "Shear attenuation": {"col": 14, "unit": "1"}, "Longitudinal attenuation":{"col": 15, "unit": "1"},\
                                "Quenched density": {"col": 16, "unit": "g/cm^3"}, "most abundant phase": {"col": 17, "unit": None}}
        self.oheader = { 'Temperature': 'T(K)',  'Pressure': 'P(bar)' ,  'Density': 'rho,kg/m3',\
        'Thermal_expansivity': 'alpha,1/K', 'Isobaric_heat_capacity': 'cp,J/K/kg',\
        'VP': 'vp,km/s', 'VS': 'vs,km/s', 'Enthalpy': 'h,J/kg', "Entropy": 's,J/K/kg',\
        'Omph_GHP': 'Omph', "Gt_HGP": "Gt", "most abundant phase": "maphs"}
        self.perplex_header = { 'T': "Temperature", "P": 'Pressure', "rho": 'Density',\
                               "alpha": 'Thermal_expansivity', 'cp': "Isobaric_heat_capacity",\
                                'vp': "VP", 'vs': "VS", 'h': "Enthalpy", "s": "Entropy", "H2O": "H2O",\
                                "cAmph(G)": "cAmph_G", "Ep(HP11)": "Ep_HP11", "law": "law",\
                                    "Omph(GHP)": "Omph_GHP", "Gt(HGP)": "Gt_HGP",\
                                    "maphs": "maphs",  # main phase indicator
                                    }
        self.ounit = {'Temperature': 'K', 'Pressure': 'bar', 'Thermal_expansivity': '1/K',\
        'Isobaric_heat_capacity': 'J/K/kg', 'Density': 'kg/m3', 'VP':'km/s', 'VS':'km/s', 'Enthalpy': 'J/kg', "Entropy": "J/K/kg"}
        self.Pinterp_rows = None
        self.PS_data = None
        self.PS_number_out1 = 0
        self.PS_number_out2 = 0
        self.PS_delta_out1 = 0.0
        self.PS_delta_out2 = 0.0 

    def ReadHeFestoTable(self, path):
        '''
        Read HeFESTo table file and populate header and data.
        Parameters:
            path (str): file path.
        '''
        Plotter = LINEARPLOT('hefesto', {})
        print("Reading Header: %s" % path)
        Plotter.ReadHeader(path)
        print("Reading Data: %s" % path)
        Plotter.ReadData(path)
        self.header = Plotter.header
        self.data = Plotter.data

    def ReadPerplex(self, path, **kwargs):
        '''
        Read Perple_X table and map headers to internal format.
        Parameters:
            path (str): file path.
            kwargs:
                header_rows (int): number of header rows.
        '''
        header_rows = kwargs.get('header_rows', 0)
        line = ""
        with open(path, 'r') as fin:
            for i in range(header_rows):
                line = fin.readline()
                assert(line != "")
        header, unit = ParsePerplexHeader(line)

        for i in range(len(header)):
            header_to = self.perplex_header[header[i]]
            self.header[header_to] = {}
            self.header[header_to]['col'] = i
            self.header[header_to]['unit'] = unit[i]
        self.data = np.loadtxt(path, skiprows=header_rows)

    def ReadRawFort56(self, _path):
        '''
        Read raw fort.56 file into internal structure.
        Parameters:
            _path (str): path to fort.56 file.
        '''
        print("%s: reading file" % func_name())
        assert(os.path.isfile(_path))
        self.header = self.fort_56_header
        self.data = np.genfromtxt(_path)
        # todo_maph
        self.maph_data = np.genfromtxt(_path, dtype="U50", encoding=None, usecols=[17])
        print("%s: data dimension: " % func_name(), self.data.shape)

    def AllFields(self):
        '''
        return all fields options
        '''
        fields = []
        for key, _ in self.header.items():
            fields.append(key)
        return fields
    
    def export_fort56_vss(self):
        '''
        export the vss data from the fort.56 file
        '''
        col_depth = self.header["Depth"]["col"]
        col_vs = self.header["VS"]["col"]
        depths = self.data[:, col_depth]
        Vss = self.data[:, col_vs]
        return depths, Vss

    def export_density_profile(self):
        '''
        Export density profile as depth-density arrays.
        Returns:
            tuple (np.ndarray, np.ndarray): depths and densities.
        '''
        col_depth = self.header["Depth"]["col"]
        col_density = self.header["Density"]["col"]
        depths = self.data[:, col_depth]
        densities = self.data[:, col_density]
        return depths, densities

    def export_temperature_profile(self):
        '''
        Export temperature profile as depth-temperature arrays.
        Returns:
            tuple (np.ndarray, np.ndarray): depths and temperatures.
        '''
        col_depth = self.header["Depth"]["col"]
        col_T = self.header["Temperature"]["col"]
        depths = self.data[:, col_depth]
        temperatures = self.data[:, col_T]
        return depths, temperatures

    def export_thermal_expansivity_profile(self):
        '''
        Export thermal expansivity profile.
        Returns:
            tuple (np.ndarray, np.ndarray): depths and alpha values.
        '''
        col_depth = self.header["Depth"]["col"]
        col_alpha = self.header["Thermal_expansivity"]["col"]
        depths = self.data[:, col_depth]
        alphas = self.data[:, col_alpha]
        return depths, alphas
    

    def export_pressure_profile(self):
        '''
        export density profile
        '''
        col_depth = self.header["Depth"]["col"]
        col_pressure = self.header["Pressure"]["col"]
        depths = self.data[:, col_depth]
        pressures = self.data[:, col_pressure]
        return depths, pressures
    
    def export_field_profile(self, field):
        '''
        export field profile
        Inputs:
            field (str): the field to output
        '''
        col_depth = self.header["Depth"]["col"]
        col_field = self.header[field]["col"]
        depths = self.data[:, col_depth]
        _data = self.data[:, col_field]
        return depths, _data

    def export_field(self, field):
        '''
        export field
        Inputs:
            field (str): the field to output
        '''
        col_field = self.header[field]["col"]
        _data = self.data[:, col_field]
        return _data
    
    def export_field_mesh(self, field):
        '''
        export field in a 2-d mesh
        '''
        col_field = self.header[field]["col"]

        # call update and process with information of the mesh
        self.Update()
        col_first = self.header[self.first_dimension_name]['col']
        col_second = self.header[self.second_dimension_name]['col']

        # initiate 2-d arrays 
        D1 = np.zeros((self.number1, self.number2))
        D2 = np.zeros((self.number1, self.number2))
        F = np.zeros((self.number1, self.number2))

        # process data
        for i in range(self.number1):
            for j in range(self.number2):
                D1[i, j] = self.data[i, col_first]
                D2[i, j] = self.data[j*self.number1, col_second]
                F[i, j] = self.data[j*self.number1 + i, col_field]
        
        return D1, D2, F
    
    def fix_field_nan_value(self, field, value):
        '''
        fix nan value of field
        Inputs:
            field (str): name of field
            value (float): value to fix with
        '''
        col_field = self.header[field]["col"]
        for i in range(self.data.shape[0]):
            # if self.data[i, col_field] == float('nan'):
            if np.isnan(self.data[i, col_field]):
                self.data[i, col_field] = value


    def Update(self, **kwargs):
        '''
        Checkt the Hefesto lookup table, this only check that the first dimension
        is aligned and no data is missings.
    
        Inputs:
            kwargs: options
                version: version of this file
        Outputs:
            Output to sceen whether the contents are a rectangular table
        '''
        # read dimension info
        print("Read information of the 1st dimension")

        # get first dimension name
        find_first = False
        find_second = False
        for key, value in self.header.items():
            if value['col'] == 0:
                self.first_dimension_name = key
                find_first = True
            elif value['col'] == 1:
                self.second_dimension_name = key
                find_second = True
        assert(find_first and find_second)

        # update and check the first dimension
        col_first = self.header[self.first_dimension_name]['col']
        min1, delta1, number1 = ReadFirstDimension(self.data[:, col_first])
        self.min1 = min1
        self.delta1 = delta1
        self.number1 = number1
        self.max1 = min1 + delta1 * (number1 - 1.0)
        print("Dimention 1 has %d entries" % number1)
        print("Checking data")
        is_correct = CheckDataDimension(self.data[:, col_first], min1, delta1, number1)
        if is_correct:
            print('Everything is all right of this file')
        else:
            raise Exception('Something is wrong')
        
        # update the second dimension information
        col_second = self.header[self.second_dimension_name]['col']
        min2, delta2, number2 = ReadSecondDimension(self.data[:, col_second])
        self.min2 = min2
        self.delta2 = delta2
        self.number2 = number2
        self.max2 = min2 + delta2 * (number2 - 1.0)

    def Interpolate(self, V1, V2):
        """
        Bilinearly interpolate self.data at coordinates (V1, V2).
    
        Parameters
        ----------
        V1 : float or np.ndarray
            Coordinate(s) in the first dimension.
        V2 : float or np.ndarray
            Coordinate(s) in the second dimension.
    
        Returns
        -------
        np.ndarray
            Interpolated value(s). If V1 and V2 are scalars, returns a 1D array
            of shape (n_fields,). If V1 and V2 are arrays of shape (...),
            returns an array of shape (..., n_fields).
        """
        V1 = np.asarray(V1)
        V2 = np.asarray(V2)
    
        # assert values are in range
        assert np.all(V1 >= self.min1) and np.all(V1 <= self.max1)
        assert np.all(V2 >= self.min2) and np.all(V2 <= self.max2)
    
        # figure out interpolation coordinates
        foo1 = (V1 - self.min1) / self.delta1
        foo2 = (V2 - self.min2) / self.delta2
    
        idx1 = np.floor(foo1).astype(int)
        idx2 = np.floor(foo2).astype(int)
    
        # avoid overflow at the upper boundary
        idx1 = np.clip(idx1, 0, self.number1 - 2)
        idx2 = np.clip(idx2, 0, self.number2 - 2)
    
        frac1 = foo1 - idx1
        frac2 = foo2 - idx2
    
        # if value is exactly at upper boundary, force fraction to 1 on last cell
        frac1 = np.where(V1 == self.max1, 1.0, frac1)
        frac2 = np.where(V2 == self.max2, 1.0, frac2)
    
        # figure out interpolation rows
        row_a = self.number1 * idx2 + idx1
        frac_a = (1.0 - frac1) * (1.0 - frac2)
    
        row_b = self.number1 * idx2 + idx1 + 1
        frac_b = frac1 * (1.0 - frac2)
    
        row_c = self.number1 * (idx2 + 1) + idx1 + 1
        frac_c = frac1 * frac2
    
        row_d = self.number1 * (idx2 + 1) + idx1
        frac_d = (1.0 - frac1) * frac2
    
        # interpolate
        results = (
            self.data[row_a, :] * frac_a[..., np.newaxis]
            + self.data[row_b, :] * frac_b[..., np.newaxis]
            + self.data[row_c, :] * frac_c[..., np.newaxis]
            + self.data[row_d, :] * frac_d[..., np.newaxis]
        )
    
        return results

    def CreateNew(self, new_data, _name, oheader):
        # todo_new
        self.header[_name] = {'col': self.data.shape[1]}
        self.oheader[_name] = oheader
        self.ounit[_name] = oheader
        self.data = np.concatenate((self.data, new_data), axis=1)
    
    def Process(self, field_names, o_path, **kwargs):
        '''
        Process the Hefesto lookup table for aspect
    
        Inputs:
            o_path: a output path
            kwargs: options
                interval1 & 2: interval in the first & second dimension
                digit: digit of output numbers
                file_type: type of the output file, perple_x or structured
        Outputs:
            Output of this function is the Perplex file form that could be recognized by aspect
        Returns:
            -
        '''
        first_dimension_name = kwargs.get('first_dimension', 'Pressure')
        second_dimension_name = kwargs.get('second_dimension', 'Temperature')
        fix_coordinate_minor = kwargs.get("fix_coordinate_minor", False)
        interval1 = kwargs.get('interval1', 1)
        interval2 = kwargs.get('interval2', 1)

        # read dimension info
        col_first = self.header[first_dimension_name]['col']
        col_second = self.header[second_dimension_name]['col']
        self.min1, self.delta1, self.number1 = ReadFirstDimension(self.data[:, col_first])
        self.min2, self.delta2, self.number2 = ReadSecondDimension(self.data[:, col_second])

        # fix minor error in the coordinate data
        if fix_coordinate_minor:
            FixFirstDimensionMinor(self.data[:, col_first], self.number1)
            FixSecondDimensionMinor(self.data[:, col_second], self.number1)

        # output
        if interval1 == 1 and interval2 == 1:
            self.indexes = np.array(range(self.number1*self.number2))
        else:
            print("%s begin indexing" % func_name())  # debug
            self.indexes = np.array(self.IndexesByInterval(interval1, interval2))  # work out indexes
            print("%s finish indexing" % func_name())  # debug
        self.number_out1 = int(np.ceil(self.number1 / interval1)) # number of output
        self.number_out2 = int(np.ceil(self.number2 / interval2))
        # output intervals
        self.delta_out1 = self.delta1 * interval1 # output intervals
        self.delta_out2 = self.delta2 * interval2 # output intervals
        self.OutputPerplexTable(field_names, o_path, **kwargs)

    def OutputPerplexTable(self, field_names, o_path, **kwargs):
        '''
        Process the Hefesto lookup table for aspect
    
        Inputs:
            o_path: a output path
            field_names: field_name to output, the first two are the first and second dimension
            kwargs: options
                version: version of this file
                digit: digit of output numbers
                file_type: type of the output file, perple_x or structured
                exchange_dimension: exchange the 1st and 2nd dimensions
        Outputs:
            Output of this function is the Perplex file form that could be recognized by aspect
        Returns:
            -
        '''
        digit = kwargs.get('digit', 8)
        file_type = kwargs.get("file_type", 'perple_x')
        exchange_dimension = kwargs.get("exchange_dimension", False)
        assert(file_type in ['perple_x', 'structured'])

        UnitConvert = UNITCONVERT()
        print("%s: Outputing Data" % func_name())
        # columns
        print("Outputing fields: %s" % field_names)
        print('first dimension: ', self.number_out1, ", second dimension: ", self.number_out2, ", size:", self.number_out1 * self.number_out2)
        my_assert(len(field_names) >= 2, ValueError, 'Entry of field_names must have more than 2 components')
        columns = []
        
        # handle in most abundant phase in a different way
        # todo_maph
        output_maph = False
        maph_lookup = kwargs.get("maph_lookup", None)
        if "most abundant phase" in field_names:
            field_names.remove("most abundant phase")
            output_maph = True
            my_assert(ValueError, maph_lookup is not None,\
                                "By exporting the most abundant phase, need to include a \'maph_lookup\' option")
            conditions = []
            values = []
            for i in range(len(maph_lookup)):
                conditions.append((self.maph_data==maph_lookup[i]))
                values.append(i+1)
            maph_output = np.select(conditions, values, default=0)

        missing_last = self.data.shape[1]
        missing_fix_values = []
        for field_name in field_names:
            # attach size(field_names) if failed
            try:
                columns.append(self.header[field_name]['col'])
            except KeyError:
                # first check that T or P is not missing
                # then append an imaginary column
                if field_name == 'Temperature':
                    raise KeyError('Abort: Temperature field is missing')
                elif field_name == 'Pressure':
                    raise KeyError('Abort: Pressure field is missing')
                else:
                    # assign an append value
                    print('field %s is missing, going to append manually' % field_name)
                    columns.append(missing_last)
                    missing_last += 1
                    # ask for value
                    missing_fix_value = float(input('Input value:'))
                    missing_fix_values.append(missing_fix_value)

        unit_factors = []
        for field_name in field_names:
            # attach 1 if failed
            try:
                unit_factors.append(self.UnitConvert(self.header[field_name]['unit'], self.ounit[field_name]))
            except KeyError:
                unit_factors.append(1.0)
        # check the output values
        # note that self.indexes[self.number_out1] gives the index of the second member in the second dimension
        tolerance = 1e-5
        temp1 = self.data[self.indexes[1], columns[0]] - self.data[self.indexes[0], columns[0]]
        temp2 = self.data[self.indexes[self.number_out1-1], columns[1]] - self.data[self.indexes[0], columns[1]]  
        my_assert( (abs(temp1 - self.delta_out1) / self.delta_out1) < tolerance,
        ValueError, "Output interval(self.delta_out1) doesn't match the interval in data")
        my_assert( (abs(temp2 - self.delta_out2) / self.delta_out2) < tolerance,
        ValueError, "Output interval(self.delta_out2) doesn't match the interval in data")

        # mend self.data if needed
        if missing_last > self.data.shape[1]:
            print("Concatenating missing data")
            new_data = np.ones((self.data.shape[0], missing_last - self.data.shape[1])) *  missing_fix_values
            self.data = np.concatenate((self.data, new_data), axis=1)

        # output
        with open(o_path, 'w') as fout: 
            if file_type == "perple_x":
                if exchange_dimension:
                    raise NotImplementedError()
                # write header for perple_x file
                fout.write(self.version + '\n')  # version
                fout.write(os.path.basename(o_path) + '\n') # filenamea
                fout.write('2\n')  # dimension
                fout.write('%s\n' % self.oheader[field_names[0]])
                fout.write('\t%.8f\n' % (float(self.min1) * unit_factors[0])) # min value
                fout.write('\t%.8f\n' % (float(self.delta_out1) * unit_factors[0]))  # difference, use the output value
                fout.write('\t%s\n' % self.number_out1)  # number of output
                fout.write('%s\n' % self.oheader[field_names[1]])
                fout.write('\t%.8f\n' % (float(self.min2) * unit_factors[1]))
                fout.write('\t%.8f\n' % (float(self.delta_out2) * unit_factors[1]))
                fout.write('\t%s\n' % self.number_out2)
                fout.write('\t%s\n' % len(columns))
            elif file_type == "structured":
                # write header for structured file
                fout.write("# This is a data output from HeFESTo\n")
                if exchange_dimension:
                    fout.write("# Independent variables are %s and %s\n" % (self.oheader[field_names[1]], self.oheader[field_names[0]]))
                    fout.write("# POINTS: %d %d\n" % (self.number_out2, self.number_out1)) 
                else:
                    fout.write("# Independent variables are %s and %s\n" % (self.oheader[field_names[0]], self.oheader[field_names[1]]))
                    fout.write("# POINTS: %d %d\n" % (self.number_out1, self.number_out2)) 
            temp = ''
            if exchange_dimension:
                temp += '%-20s%-20s' % (self.oheader[field_names[1]], self.oheader[field_names[0]])
            else:
                temp += '%-20s%-20s' % (self.oheader[field_names[0]], self.oheader[field_names[1]])
            for i in range(2, len(field_names)):
                field_name = field_names[i]
                temp += '%-20s' % self.oheader[field_name]
            if output_maph:
                temp += '%-20s' % self.oheader["most abundant phase"]
            temp += '\n'
            fout.write(temp)
            # data is indexes, so that only part of the table is output
            indexes_output = None; columns_output = None; unit_factors_output = None
            if exchange_dimension:
                indexes_output = ExchangeDimensions(self.indexes, self.number_out1, self.number_out2)
                columns_output = columns.copy()
                columns_output[0] = columns[1]
                columns_output[1] = columns[0]
                unit_factors_output = unit_factors.copy()
                tmp = unit_factors[0]
                unit_factors_output[0] = unit_factors[1]
                unit_factors_output[1] = tmp
            else:
                indexes_output = self.indexes
                columns_output = columns
                unit_factors_output = unit_factors
            # todo_maph
            odata = self.data[np.ix_(indexes_output, columns_output)] * unit_factors_output
            _format = None
            if output_maph:
                odata = np.concatenate((odata, maph_output[np.ix_(indexes_output)].reshape(indexes_output.size, 1)), axis=1)
                # _fmt = [_format]*len(field_names) + ['%-' + str(digit+11) + 's']
                _format = '%-19s'
            else:
                _format = '%-' + str(digit + 11) + '.' + str(digit) + 'e'
            # print("_fmt=", _fmt)
            np.savetxt(fout,  odata, fmt=_format)
        print("New file generated: %s" % o_path) 
    
    def OutputPressureEntropyTable(self, field_names, o_path):
        '''
        Converts the dataset and generate a lookup table
        from (P, S) -> T
            o_path: a output path
            field_names: field_name to output, the first two are the first and second dimension
        '''
        # initiation
        UnitConvert = UNITCONVERT()
        my_assert(len(field_names) == self.PS_data.shape[1], ValueError,\
                            'Entry of field_names must equal the size of the pressure-entropy dataset')

        # figure out the factors of unit converting 
        unit_factors = []
        for field_name in field_names:
            # attach 1 if failed
            try:
                unit_factors.append(self.UnitConvert(self.header[field_name]['unit'], self.ounit[field_name]))
            except KeyError:
                unit_factors.append(1.0)

        # write output
        with open(o_path, 'w') as fout:
            fout.write("# This is a data output from.\n")
            fout.write("# Independent variables are entropy and pressure.\n")
            fout.write("# POINTS: %s %s\n" % (self.PS_number_out1, self.PS_number_out2))
            temp = ''
            for field_name in field_names:
                temp += '%-20s' % self.oheader[field_name]
            temp += '\n'
            fout.write(temp)
            np.savetxt(fout, self.PS_data * unit_factors, fmt='%-19.8e')
        print("New file generated: %s" % o_path) 


    def IndexesByInterval(self, interval1, interval2):
        '''
        Work out indexes by giving interval(default is 1, i.e. consecutive)
        '''
        my_assert(type(interval1) == int and type(interval2) == int, TypeError, "interval1(%s) or interval2(%s) is not int" % (interval1, interval2))
        # indexes in 
        indexes_1 = range(0, self.number1, interval1) 
        indexes_2 = range(0, self.number2, interval2)
        # work out the overall indexes
        indexes = []
        i2 = 0 # used for printing the percentage of completeness
        last_ratio = 0.0
        for index_2 in indexes_2:
            ratio = i2 / len(indexes_2)
            if ratio > last_ratio + 0.01:
                last_ratio = ratio
                print("percent = %.2f" % (ratio*100), end='\r')
            for index_1 in indexes_1: 
                indexes.append(index_1 + self.number1 * index_2)
            i2 += 1
        return indexes

    def InterpolatePressureEntropyByIndex(self, index_p, entropies, field_names, PS_rows, **kwargs):
        '''
        Interpolate the data with a pressure index
        Inputs:
            index_p (int): pressure index
            entropies (list): entropy inputs
            field_names (list): names of field to interpolate
            PS_rows (list): range of rows to enter into the PS_data 2-d ndarray
        kwargs:
            debug: run debug mode
        '''
        # initiate
        assert(self.PS_data is not None) # PS_data is initiated
        col_entropy = self.header["Entropy"]['col']
        col_pressure = self.header["Pressure"]['col']
        pressure = 0.0
        debug = kwargs.get("debug", False)

        # row index for this pressure
        if self.first_dimension_name == "Pressure":
            if self.Pinterp_rows is None:
                self.Pinterp_rows = np.zeros(self.number2).astype(int)
            for i in range(self.number2):
                row = self.number1 * i + index_p 
                self.Pinterp_rows[i] = row
            pressure = self.min1 + self.delta1 * index_p
        elif self.second_dimension_name == "Pressure":
            if self.Pinterp_rows is None:
                self.Pinterp_rows = np.zeros(self.number1).astype(int)
            for i in range(self.number1):
                row = index_p*self.number1 + i
                self.Pinterp_rows[i] = row
            pressure = self.min2 + self.delta2 * index_p
        else:
            return ValueError("The column of Pressure is not found.")
        
        # pressure and entropy
        for j in range(len(entropies)):
            self.PS_data[PS_rows[0] + j, 0] = entropies[j]
            self.PS_data[PS_rows[0] + j, 1] = pressure

        # extrapolate data 
        entropy_data = self.data[np.ix_(self.Pinterp_rows), col_entropy]

        if debug:
            print("pressure: ")
            print(pressure)
            print("self.Pinterp_rows")
            print(self.Pinterp_rows)
            print("entropy_data: ")
            print(entropy_data) # debug


        for i in range(len(field_names)) :
            field_name = field_names[i]
            col_data = self.header[field_name]['col']
            field_data = self.data[np.ix_(self.Pinterp_rows), col_data]
            if debug:
                print("field_data: ")
                print(field_data)
            temp = np.interp(entropies, entropy_data[0], field_data[0])
            for j in range(len(entropies)):
                self.PS_data[PS_rows[0]+j, i+2] = temp[j]
    
    def InterpolatePressureEntropy(self, entropies, field_names):
        '''
        Interpolate the data to pressure entropy field
        Inputs:
            entropies (list): entropy inputs
            field_names (list): names of field to interpolate
        '''
        # initiate
        n_field = len(field_names)
        n_entropy = len(entropies)
        if self.first_dimension_name == "Pressure":
            n_p = self.number1
        elif self.second_dimension_name == "Pressure":
            n_p = self.number2
        else:
            return ValueError("The column of Pressure is not found.")
        self.PS_data = np.zeros([n_entropy*n_p, n_field + 2]) 

        # call the function to interpolate data for one pressure value
        for index_p in range(n_p):
            PS_rows = [n_entropy * index_p, n_entropy * (index_p + 1)]
            self.InterpolatePressureEntropyByIndex(index_p, entropies, field_names, PS_rows)
        self.PS_number_out1 = n_entropy
        self.PS_number_out2 = n_p


    def PlotHefesto(self):
        '''
        Plot the Hefesto lookup table
    
        Inputs:
            -
        Returns:
            -col_alpagg
        '''
        pass
    

def convert_mol_fraction(comps):
    '''
    Convert oxide mol% composition to atomic mol fraction.
    Parameters:
        comps (list[float]): mol% composition in order [SiO2, MgO, FeO, CaO, Al2O3, Na2O], must sum to ~100.
    Returns:
        list[float]: atomic mol fraction for each component, accounting for stoichiometry of oxides.
    '''
    assert(len(comps) == 6) # SiO2, MgO, FeO, CaO, Al2O3, Na2O
    mol_total = comps[0] + comps[1] + comps[2] + comps[3] + comps[4] + comps[5]
    assert((100.0 - mol_total)/100.0 < 1e-3)
    mol_atom_total = comps[0] + comps[1] + comps[2] + comps[3] + 2 * comps[4] + 2 * comps[5]
    comps_atom = [0 for i in range(6)]
    comps_atom[0] = comps[0] / mol_atom_total
    comps_atom[1] = comps[1] / mol_atom_total
    comps_atom[2] = comps[2] / mol_atom_total
    comps_atom[3] = comps[3] / mol_atom_total
    comps_atom[4] = 2*comps[4] / mol_atom_total
    comps_atom[5] = 2*comps[5] / mol_atom_total
    return comps_atom


def PlotHeFestoProfile(_path0, **kwargs):
    '''
    Plot the profile from HeFesto. Note that the profile needs to be an adiabat
    Inputs:
        _path0 (str): path of the Hefesto Outputs
        kwargs:
            axT (matplotlib axis): axis to plot the temperatures
            ax_density (matplotlib axis): axis to plot the density
    
    '''
    axT = kwargs.get('axT', None)
    axP = kwargs.get('axP', None)
    ax_density = kwargs.get('ax_density', None)
    _color = kwargs.get("color", None)

    # get data 
    LookupTable=LOOKUP_TABLE()
    LookupTable.ReadRawFort56(_path0)
    depths_0, Ts_0 = LookupTable.export_temperature_profile()
    _, densities_0 = LookupTable.export_density_profile()
    _, pressures_0 = LookupTable.export_pressure_profile()
    
    # plot temperature
    if axT is not None:
        if _color is None:
            plot_color = "tab:red"
        else:
            plot_color = _color
        axT.plot(Ts_0, depths_0, "--", label="HeFesto T0", color=plot_color)
        axT.set_ylabel("Depth [km]")
        axT.set_xlabel("Temperature [K]")
        axT.legend()
 
    # plot pressure
    if axP is not None:
        if _color is None:
            plot_color = "tab:green"
        else:
            plot_color = _color
        axP.plot(pressures_0, depths_0, "--", label="HeFesto P0", color=plot_color)
        axP.set_ylabel("Depth [km]")
        axP.set_xlabel("Pressure [GPa]")
        axP.legend()

    # plot density 
    if ax_density is not None:
        if _color is None:
            plot_color = "tab:blue"
        else:
            plot_color = _color
        ax_density.plot(densities_0*1000.0, depths_0, "--", label="HeFesto Density", color=plot_color)
        ax_density.set_ylabel("Depth [km]")
        ax_density.set_xlabel("Density [kg/m^3]")
        ax_density.legend()


    
    return depths_0, densities_0, Ts_0


def ComputeBuoyancy(_path0, _path1, **kwargs):
    '''
    Compute buoyancy and density ratio profiles from two HeFESTo output files.
    Parameters:
        _path0 (str): path to the first HeFESTo output file.
        _path1 (str): path to the second HeFESTo output file.
        kwargs:
            axT (matplotlib.axes.Axes): axis to plot temperatures (optional).
            ax_density_ratio (matplotlib.axes.Axes): axis to plot density differences (optional).
            ax_buoy (matplotlib.axes.Axes): axis to plot buoyancy (optional).
    Returns:
        tuple:
            depths (np.ndarray): depth array within overlapping range.
            buoyancies (np.ndarray): buoyancy profile [N/m^3].
            density_ratios (np.ndarray): density ratio profile.
    '''
    n_depth = 1000
    g = 9.8
    axT = kwargs.get('axT', None)
    ax_density_ratio = kwargs.get('ax_density_ratio', None)
    ax_buoy = kwargs.get('ax_buoy', None)
    _color = kwargs.get("color", None)
    # read data
    LookupTable=LOOKUP_TABLE()
    LookupTable.ReadRawFort56(_path0)
    depths_0, densities_0 = LookupTable.export_density_profile()
    _, alphas_0 = LookupTable.export_thermal_expansivity_profile()
    _, Ts_0 = LookupTable.export_temperature_profile()
    min_depth0 = depths_0[0]
    max_depth0 = depths_0[-1]
    LookupTable.ReadRawFort56(_path1)
    depths_1, densities_1 = LookupTable.export_density_profile()
    _, alphas_1 = LookupTable.export_thermal_expansivity_profile()
    _, Ts_1 = LookupTable.export_temperature_profile()
    min_depth1 = depths_1[0]
    max_depth1 = depths_1[-1]
    # choose the mutual range in the data
    min_depth = np.max(np.array([min_depth0, min_depth1]))
    max_depth = np.min(np.array([max_depth0, max_depth1]))
    # interpolate data
    DensityFunc0 = interpolate.interp1d(depths_0, densities_0, assume_sorted=True, fill_value="extrapolate")
    AlphaFunc0 = interpolate.interp1d(depths_0, alphas_0, assume_sorted=True, fill_value="extrapolate")
    TFunc0 = interpolate.interp1d(depths_0, Ts_0, assume_sorted=True, fill_value="extrapolate")
    DensityFunc1 = interpolate.interp1d(depths_1, densities_1, assume_sorted=True, fill_value="extrapolate")
    AlphaFunc1 = interpolate.interp1d(depths_1, alphas_1, assume_sorted=True, fill_value="extrapolate")
    TFunc1 = interpolate.interp1d(depths_1, Ts_1, assume_sorted=True, fill_value="extrapolate")
    # compute buoyancy and buoyancy number
    # the buoyancy number is computed with buoyancy / density1
    # density1 is chosen instead of density0 to simulate the buoyancy ratio of density0
    depths = np.linspace(min_depth, max_depth, n_depth)
    diff_densities = np.zeros(n_depth)
    buoyancies = np.zeros(n_depth)
    density_ratios = np.zeros(n_depth)
    for i in range(n_depth):
        # get values at depth
        depth = depths[i]
        alpha = AlphaFunc1(depth)
        alpha = np.min(np.array([alpha, 5.0]))
        # print("alpha: ", alpha) # debug
        density0 = DensityFunc0(depth) * 1000.0
        density1 = DensityFunc1(depth) * 1000.0
        T0 = TFunc0(depth)
        T1 = TFunc1(depth)
        diff_density = density1 - density0
        # print("diff_density: ", diff_density) # debug
        diff_densities[i] = diff_density
        buoyancy =  - diff_density * g
        buoyancies[i] = buoyancy
        density_ratios[i] = diff_density / density1
    # plot temperature
    if axT is not None:
        if _color is None:
            plot_color = "tab:red"
        else:
            plot_color = _color
        axT.plot(Ts_0, depths_0, "--", label="HeFesto T0", color=plot_color)
        if _color is None:
            plot_color = "tab:blue"
        else:
            plot_color = _color
        axT.plot(Ts_1, depths_1, "--", label="HeFesto T1", color=plot_color)
        axT.set_ylabel("Depth [km]")
        axT.set_xlabel("Temperature [K]")
        axT.legend()
    # plot density ratio
    if ax_density_ratio is not None:
        if _color is None:
            plot_color = "tab:red"
        else:
            plot_color = _color
        ax_density_ratio.plot(density_ratios, depths, "--", label="HeFesto Density Ratio", color=plot_color)
        ax_density_ratio.set_ylabel("Depth [km]")
        ax_density_ratio.set_xlabel("Density Ratio")
        ax_density_ratio.legend()
    # plot buoyancy
    if ax_buoy is not None:
        if _color is None:
            plot_color = "tab:blue"
        else:
            plot_color = _color
        ax_buoy.plot(buoyancies, depths, "--", label="HeFesto buoyancy", color=plot_color)
        ax_buoy.set_ylabel("Depth [km]")
        ax_buoy.set_xlabel("Buoyancy [N/m^3]")
        ax_buoy.legend()
    # return variables
    return depths, buoyancies, density_ratios


def ParsePerplexHeader(line):
    '''
    Parse Perple_X header line into field names and units.
    Parameters:
        line (str): header string containing field names and units.
    Returns:
        tuple:
            header (list[str]): list of field names.
            unit (list[str]): list of corresponding units.
    '''
    words = line.split(' ')
    header = []
    unit = []
    for word in words:
        word = re_neat_word(word)
        if len(word) > 0:
            temp = word.split(',')
            if len(temp) == 2:
                unit.append(temp[1])
                header.append(temp[0])
            else:
                temp = word.split('(')
                if len(temp) == 2:
                    unit.append(temp[1].replace(')', ''))
                    header.append(temp[0])
                else:
                    # This is marking the dominant phase (e.g. maphs)
                    assert(len(temp) == 1)
                    unit.append("")
                    header.append(word)
    assert(len(header) == len(unit))
    return header, unit


def ReadFirstDimension(nddata):
    '''
    Process the data in the fisrt dimension(min, delta, number)
    Inputs:
        nddata: a ndarray of the first dimension data
    Returns:
        min: min value
        delta: data interval
        number: number in a column
    '''
    # min value
    min = nddata[0]
    # delta
    delta = nddata[1] - nddata[0]
    # number
    number = nddata.size
    for i in range(0, nddata.size-1):
        if nddata[i] > nddata[i+1]:
            number = i + 1
            break
    return min, delta, number


def FixFirstDimensionMinor(nddata, number1, **kwargs):
    '''
    Fix minor differences in first-dimension coordinate values.
    Parameters:
        nddata (np.ndarray): array of first-dimension data.
        number1 (int): number of entries in the first dimension.
        kwargs:
            limits (list[float]): lower and upper bounds for relative difference.
    Returns:
        None
    '''
    limits = kwargs.get('limits', [1e-16, 1e-4])
    for i in range(0, nddata.size):
        if i >= number1:
            if nddata[i-number1] < 1e-32:
                # don't divide a 0 value
                diff = abs(nddata[i] - nddata[i-number1])
            else:
                diff = abs((nddata[i] - nddata[i-number1])/nddata[i-number1])
            if diff > limits[0] and diff < limits[1]:
                nddata[i] = nddata[i-number1]
            elif diff > limits[1]:
                raise ValueError("Two coordinates (%d, %d) in the first dimension have large differences" % (i-number1, i))


def FixSecondDimensionMinor(nddata, number1, **kwargs):
    '''
    Fix minor differences in second-dimension coordinate values.
    Parameters:
        nddata (np.ndarray): array of second-dimension data.
        number1 (int): number of entries in the second dimension stride.
        kwargs:
            limits (list[float]): lower and upper bounds for relative difference.
    Returns:
        None
    '''
    limits = kwargs.get('limits', [1e-16, 1e-4])
    for i in range(0, nddata.size):
        if i % number1 != 0:
            if nddata[i-1] < 1e-32:
                # don't divide a 0 value
                diff = abs((nddata[i] - nddata[i-1]))
            else:
                diff = abs((nddata[i] - nddata[i-1])/nddata[i-1])
            if diff > limits[0] and diff < limits[1]:
                nddata[i] = nddata[i-1]
            elif diff > limits[1]:
                raise ValueError("Two coordinates (%d, %d) in the second dimension have large differences" % (i-1, i))


def ReadSecondDimension(nddata):
    '''
    Compute minimum value, spacing, and size of second dimension.
    Parameters:
        nddata (np.ndarray): array of second-dimension data.
    Returns:
        tuple:
            min (float): minimum value.
            delta (float): spacing between values.
            number (int): number of entries per column.
    '''
    # min value
    min = nddata[0]
    # delta
    tolerance = 1e-6
    delta = 0.0
    for i in range(0, nddata.size-1):
        if abs(nddata[i] - nddata[i+1]) / abs(nddata[i]) > tolerance:
            delta = nddata[i+1] - nddata[i]
            sub_size = i + 1
            break
    # number
    my_assert(nddata.size % sub_size == 0, ValueError, 'the table is not regular(rectangle)')
    number = nddata.size // sub_size
    return min, delta, number


def CheckDataDimension(nddata, min1, delta1, number1):
    '''
    Check whether the first dimension data matches expected uniform spacing.
    Parameters:
        nddata (np.ndarray): data array for the first dimension.
        min1 (float): minimum value of the first dimension.
        delta1 (float): spacing between consecutive values.
        number1 (int): expected number of entries in the first dimension.
    Returns:
        bool: True if data matches expected structure, False otherwise.
    '''
    tolerance = 1e-6
    i = 0
    i1 = 0
    is_correct = True
    while True:
        value1 = min1 + delta1 * i1
        if i >= nddata.shape[0]:
            if i1 < number1 - 1:
                print('entry %d(index of row in the data part) is incorrect(missing at the end), the correct value is %.7f' % (i, value1))
            break
        elif i1 >= number1:
            # index in the first dimension exceed maximum
            # move to the next value in the second dimension
            i1 = 0
        elif abs(nddata[i] - value1) > tolerance:
            # value in the first dimension doesn't match
            # shoot message and move over to the next value of the second dimension in our data
            print('entry %d(index of row in the data part) is incorrect(%.7f), the correct value is %.7f' % (i, nddata[i], value1))
            is_correct = False
            i1 = 0
            while nddata[i] < nddata[i+1]:
                i += 1
            i += 1
        else:
            # move to the next value
            i1 += 1
            i += 1
        return is_correct
    

class HEFESTO_OPT(JSON_OPT):
    '''
    Define a class to work with CASE
    List of keys:
    '''
    def __init__(self):
        '''
        Initiation, first perform parental class's initiation,
        then perform daughter class's initiation.
        '''
        JSON_OPT.__init__(self)
        self.add_key("HeFESTo Repository", str, ["HeFESTo repository"], ".", nick='hefesto_dir')
        self.add_key("Output directory", str, ["output directory"], ".", nick='o_dir')
        self.add_key("Case name", str, ["case name"], "foo", nick='case_name')
        self.add_key("Number of processor", int, ["nproc"], 1, nick='nproc')
        self.add_key("Path of a control file", str, ["control path"], ".", nick='control_path')
        self.add_key("dimension T: T1", float, ["T", "T1"], 0.0, nick='T1')
        self.add_key("dimension T: T2", float, ["T", "T2"], 0.0, nick='T2')
        self.add_key("dimension T: nT", int, ["T", "nT"], 0, nick='nT')
        self.add_key("dimension P: P1", float, ["P", "P1"], 0.0, nick='P1')
        self.add_key("dimension P: P2", float, ["P", "P2"], 0.0, nick='P2')
        self.add_key("dimension P: nP", int, ["P", "nP"], 0, nick='nP')
        self.add_key("dimension T: variable", str, ["T", "variable"], 'temperature', nick='T_variable')
        self.add_key("path of the slurm file", str, ["slurm path"], 'foo', nick='slurm_file_base')
        self.add_key("path of the parameter directory", str, ["prm directory"], 'foo', nick='prm_dir')
        self.add_key("split by", str, ["split by"], 'P', nick='split_by')


    def check(self):
        '''
        check inputs are valid
        '''
        hefesto_dir = var_subs(self.values[0])
        o_dir = var_subs(self.values[1])
        my_assert(os.path.isdir(o_dir), FileNotFoundError, "No such directory: %s" % o_dir)
        control_path = var_subs(self.values[4])
        my_assert(os.path.isfile(control_path), FileNotFoundError, "No such file: %s" % control_path)
        # order of T1, T2; P1, P2
        T1 = self.values[5]
        T2 = self.values[6]
        assert(T1 <= T2)
        P1 = self.values[8]
        P2 = self.values[9]
        assert(P1 <= P2)
        T_variable = self.values[11]
        assert(T_variable in ["temperature", "entropy"])
        slurm_file_base = var_subs(self.values[12])
        assert(os.path.isfile(slurm_file_base))


    def to_distribute_parallel_control(self):
        '''
        Interface to the DistributeParallelControl function
        '''
        hefesto_dir = var_subs(self.values[0])
        o_dir = var_subs(self.values[1])
        case_name = var_subs(self.values[2])
        nproc = self.values[3]
        control_path = var_subs(self.values[4])
        T1 = self.values[5]
        T2 = self.values[6]
        nT = self.values[7]
        P1 = self.values[8]
        P2 = self.values[9]
        nP = self.values[10]
        T_variable = self.values[11]
        slurm_file_base = var_subs(self.values[12])
        prm_dir = var_subs(self.values[13])
        split_by = var_subs(self.values[14])
        return hefesto_dir, o_dir, case_name, nproc, control_path,\
            T1, T2, nT, P1, P2, nP, T_variable, slurm_file_base, prm_dir,\
            split_by

    def case_dir(self):
        '''
        return the path to the case directory
        '''
        o_dir = var_subs(self.values[1])
        case_name = var_subs(self.values[2])
        case_dir = os.path.join(o_dir, case_name)
        return case_dir
    

class CONTROL_FILE():

    def __init__(self):
        '''
        initiation
        Attributes:
            lines: line inputs
            n_raw (int): number of line inputs
            P1, P2, nP: entries for the P / density dimention
            T1, T2, nT: entries for the T / entropy dimention
            useT (int): 0 - use S; 1 - use T
            prm_dir: directory to the parameter files
        '''
        self.lines = []
        self.n_raw = 0
        self.P1 = 0.0
        self.P2 = 0.0
        self.nP = 0
        self.useT = 1
        self.prm_dir = None
        self.composition = None
        pass

    def ReadFile(self, _path):
        '''
        Inputs:
            _path (str): _path of a control file
        '''
        assert(os.path.isfile(_path))
        with open(_path, 'r') as fin:
            contents = fin.read()
        self.lines = contents.split('\n')
        self.n_raw = len(self.lines)

        # read the first line, dimentions in P, T
        line1 = self.lines[0]
        temp = line1.split(',')
        foo = re_neat_word(temp[0])
        self.P1 = float(foo)
        foo = re_neat_word(temp[1])
        self.P2 = float(foo)
        foo = re_neat_word(temp[2])
        self.nP = int(foo)
        foo = re_neat_word(temp[3])
        self.T1 = float(foo)
        foo = re_neat_word(temp[4])
        self.T2 = float(foo)
        foo = re_neat_word(temp[5])
        self.nT = int(foo)

    # todo_table
    def configure_atom_composition(self, composition):
        '''
        Inputs:
            composition: list of atom contents
        '''
        assert isinstance(composition, (list, tuple, np.ndarray)), \
            "composition must be array-like (list, tuple, or numpy.ndarray)"
        self.composition = composition


    def configureP(self, P1, P2, nP):
        '''
        Inputs:
            P1, P2, nP: entries for the P / density dimention
        '''
        self.P1 = P1
        self.P2 = P2
        self.nP = nP
    
    def configureT(self, T1, T2, nT, **kwargs):
        '''
        Inputs:
            T1, T2, nT: entries for the T / entropy dimention
            kwargs:
                use_T: use temperature

        '''
        useT = kwargs.get("useT", 1)
        self.useT = useT
        self.T1 = T1
        self.T2 = T2
        self.nT = nT
    
    def configurePrm(self, prm_dir):
        '''
        Inputs:
            directory of the parameter files
        '''
        self.prm_dir = prm_dir

    def WriteFile(self, _path):
        '''
        Inputs:
            _path (str): _path of a control file to write
        '''
        o_lines = self.lines.copy()
        # first line: P, T dimension
        o_lines[0] = ''
        line1 = self.lines[0]
        temp = line1.split(',')
        foo = "%s,%s,%s" % (self.P1, self.P2, self.nP)
        o_lines[0] += foo
        foo = ",%s,%s,%s" % (self.T1, self.T2, self.nT)
        o_lines[0] += foo
        if self.useT == 1:
            temp1 = 0
        else:
            # use entropy
            temp1 = -2
        foo = ",%s,%s,%s" % (temp1, temp[7], temp[8])
        o_lines[0] += foo

        if self.composition is not None:
            o_lines[3] = "Si%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[0], 5*" ", 10*self.composition[0], 4*" ")
            o_lines[4] = "Mg%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[1], 5*" ", 10*self.composition[1], 4*" ")
            o_lines[5] = "Fe%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[2], 5*" ", 10*self.composition[2], 4*" ")
            o_lines[6] = "Ca%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[3], 5*" ", 10*self.composition[3], 4*" ")
            o_lines[7] = "Al%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[4], 5*" ", 10*self.composition[4], 4*" ")
            o_lines[8] = "Na%s%.5f%s%.5f%s0" % (11*" ", 10*self.composition[5], 5*" ", 10*self.composition[5], 4*" ")

        # path of the parameter files
        if self.prm_dir is not None:
            o_lines[10] = self.prm_dir

        # write file
        with open(_path, 'w') as fout:
            for i in range(self.n_raw):
                line = o_lines[i]
                if i > 0:
                    fout.write('\n')
                fout.write(line)
        assert(os.path.isfile(_path))
        print("Write file %s" % _path)


def DistributeParallelControl(hefesto_dir, o_dir, case_name, nproc, control_path,\
                              T1, T2, nT, P1, P2, nP, T_variable, slurm_file_base, prm_dir, split_by,*,
                              modules=['intel-oneapi-mkl/2022.2.1'],
                              sources=[],
                              composition=None):
    '''
    Generate controls file for running HeFesto in parallel
    Inputs;
        json_opt (dict or json file): inputs
    '''
    assert(split_by in ["P", "T"])

    if T_variable == "temperature":
        useT = 1
    else:
        useT = 0
    
    # make directory
    case_dir = os.path.join(o_dir, case_name)
    if os.path.isdir(case_dir):
        foo = input("Case directory %s exist, remove? [y/n]" % case_dir)
        if foo == 'y':
            rmtree(case_dir)
        else:
            "Terminating"
            exit(0)
    
    # read file
    ControlFile = CONTROL_FILE()
    ControlFile.ReadFile(control_path)
    ControlFile.configurePrm(prm_dir)
    # todo_table
    if composition is not None:
        ControlFile.configure_atom_composition(composition)

    # P Ranges
    p_ranges = []
    T_ranges = []
    if split_by == "P":
        p_interval = (P2 - P1) / nP
        for i in range(nproc):
            if nP >= nproc and nproc > 1:
                foo = nP // nproc
                foo1 = nP % nproc
                if i == nproc - 1:
                    p_range = [P1 + p_interval * foo*i, P2, int(foo + foo1 + 1)]
                else:
                    p_range = [P1 + p_interval * foo * i, P1 + p_interval * foo * (i+1), int(foo + 1)]
            else:
                p_range = [P1, P2, nP]
            p_ranges.append(p_range)
    elif split_by == "T":
        t_interval = (T2 - T1) / nT
        for i in range(nproc):
            if nT >= nproc and nproc > 1:
                foo = nT  // nproc
                foo1 = nT % nproc
                if i == nproc - 1:
                    t_range = [T1 + t_interval * foo*i, T2, int(foo + foo1 + 1)]
                else:
                    t_range = [T1 + t_interval * foo * i, T1 + t_interval * foo * (i+1), int(foo + 1)]
            else:
                t_range = [T1, T2, nT]
            T_ranges.append(t_range)

    # generate cases 
    os.mkdir(case_dir)
    # make subdirectories
    exe_path = os.path.join(hefesto_dir, "main")
    sh_path = os.path.join(case_dir, "configure.sh")
    sh_contents = "#!/bin/bash\n"
    for iproc in range(nproc):
        sub_dir_name = "sub_%04d" % iproc
        sub_dir = os.path.join(case_dir, sub_dir_name)
        os.mkdir(sub_dir)
        # append new line to sh file 
        o_exe_path = os.path.join(sub_dir_name, "main")
        new_line = "cp %s %s" % (exe_path, o_exe_path)
        sh_contents += (new_line + '\n')
        # configure P, T
        print("split_by: ", split_by) # debug
        if split_by == "P":
            p_range = p_ranges[iproc]
            ControlFile.configureP(p_range[0], p_range[1], p_range[2])
        else:
            ControlFile.configureP(P1, P2, nP)
        if split_by == "T":
            T_range = T_ranges[iproc]
            ControlFile.configureT(T_range[0], T_range[1], T_range[2], useT=useT)
        else:
            ControlFile.configureT(T1, T2, nT, useT=useT)
        # write file
        temp = os.path.join(sub_dir, "control")
        ControlFile.WriteFile(temp)

    # generate bash file 
    with open(sh_path, 'w') as fin:
        fin.write(sh_contents)
    
    # generate the slurm file
    slurm_file_path = os.path.join(case_dir, "job.sh")
    SlurmOperator = SLURM_OPERATOR(slurm_file_base)
    SlurmOperator.SetAffinity(1, nproc, 1)
    SlurmOperator.ResetCommand()
    SlurmOperator.SetName(case_name)
    SlurmOperator.SetModule(modules, [])
    SlurmOperator.SetSource(sources)

    SlurmOperator.SetTimeByHour(300)
    # generate the command to run
    extra_contents = ""
    temp = "subdirs=("
    for iproc in range(nproc):
        if iproc > 0:
            temp += " "
        sub_dir_name = "sub_%04d" % iproc
        temp += "\"%s\"" % sub_dir_name
    temp += ")\n"
    extra_contents += temp

    temp = """
for subdir in ${subdirs[@]}; do
        cd ${subdir}
        srun --exclusive --ntasks 1 ./main control &
        cd ..
done
wait
"""
    extra_contents += temp
    SlurmOperator.SetExtra(extra_contents)
    SlurmOperator(slurm_file_path)
    print("%s: make new case %s" % (func_name(), case_dir))


def AssembleParallelFiles(case_dir):
    '''
    Inputs:
        case_dir: path to the case directory
    '''
    # screen outputs
    print("start AssembleParallelFiles")

    # load options
    json_file = os.path.join(case_dir, "case.json")
    assert(os.path.isfile(json_file))
    with open(json_file, 'r') as fin:
        case_opt = json.load(fin)
    nproc = case_opt['nproc']
    nP = case_opt['P']['nP']
    # screen outputs
    print("nproc: ", nproc)

    # read file contents 
    # read first file
    sub_dir_name = "sub_0000"
    sub_dir = os.path.join(case_dir, sub_dir_name)
    output_dir = os.path.join(case_dir, "output")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    fort_56_o_path = os.path.join(output_dir, "fort.56")
    # read subsequent files
    fort_56_path = os.path.join(sub_dir, "fort.56")
    with open(fort_56_path, 'r') as fin:
        lines = fin.readlines()
    for i in range(1, nproc):
        sub_dir_name = "sub_%04d" % i
        sub_dir = os.path.join(case_dir, sub_dir_name)
        fort_56_path = os.path.join(sub_dir, "fort.56")
        with open(fort_56_path, 'r') as fin:
            temp = fin.readlines()
        for i in range(nP+1,len(temp)):
            lines.append(temp[i])

    # write file 
    with open(fort_56_o_path, 'w') as fout:
        for _line in lines:
            fout.write(_line)
    
    print("File generated:", fort_56_o_path) # debug


def ExchangeDimensions(indexes, number_out1, number_out2):
    '''
    exchange the indexing for the 1st and the 2nd dimensions
    '''
    assert(indexes.ndim==1)
    ixx = np.zeros(indexes.shape, dtype=int)
    i = 0
    for i1 in range(number_out1):
        for j2 in range(number_out2):
            ixx[i] = j2 * number_out1 + i1
            i += 1
    ex_indexes = indexes[np.ix_(ixx)] 

    return ex_indexes