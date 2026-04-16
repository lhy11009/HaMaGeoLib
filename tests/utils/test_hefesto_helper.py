

import os
# import pytest
import filecmp  # for compare file contents
import numpy as np
from numpy.testing import assert_allclose
from shutil import rmtree  # for remove directories
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import LINEARPLOT
from hamageolib.utils.hefesto_helper import *

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
source_dir = os.path.join(package_root, "tests/fixtures/utils/hefesto_helper")
test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "hefesto_helper"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)


def test_parse_perplex_header():
    '''
    test the ParsePerplexHeader function
    '''
    line = "T(K)           P(bar)         s,J/K/kg       rho,kg/m3      alpha,1/K      cp,J/K/m3      vp,km/s        vs,km/s"
    header, unit = ParsePerplexHeader(line)
    assert(header==['T', 'P', 's', 'rho', 'alpha', 'cp', 'vp', 'vs'])
    assert(unit==['K', 'bar', 'J/K/kg', 'kg/m3', '1/K', 'J/K/m3', 'km/s', 'km/s'])


def test_read_perplex():
    '''
    test function ReadPerplex 
    assert:
        1. contents of perplex table generated
    '''
    # input file
    filein = os.path.join(source_dir, 'perplex_lookup_table.txt')
    assert(os.path.isfile(filein))
    LookupTable = LOOKUP_TABLE()
    LookupTable.ReadPerplex(filein, header_rows=4)
    LookupTable.Update()

    # output file
    fileout = os.path.join(test_dir, 'perpelx_lookup_table_0.txt')
    fileout_std = os.path.join(source_dir, 'perpelx_lookup_table_0_std.txt')
    if os.path.isfile(fileout):
        # remove previous results
        os.remove(fileout)
    field_names = ['Temperature', 'Pressure', 'Density']
    LookupTable.Process(field_names, fileout, first_dimension="Temperature", second_dimension="Pressure")
    assert(os.path.isfile(fileout))
    assert(filecmp.cmp(fileout, fileout_std))

    # output pressure entropy lookup tableq
    fileout = os.path.join(test_dir, 'perpelx_ps_lookup_table.txt')
    fileout_std = os.path.join(source_dir, 'perpelx_ps_lookup_table_std.txt')
    entropies = np.linspace(1000.0, 3000.0, 21)
    field_names = ['Temperature']
    output_field_names = ['Entropy', 'Pressure', 'Temperature']
    LookupTable.InterpolatePressureEntropy(entropies, field_names)
    LookupTable.OutputPressureEntropyTable(output_field_names, fileout)
    assert(os.path.isfile(fileout))
    assert(filecmp.cmp(fileout, fileout_std))


def test_read_n_output():
    '''
    test read in and output hefesto lookup table
    Asserts:
        output file is generated
    '''
    # input file
    filein = os.path.join(source_dir, 'lookup_table.txt')
    assert(os.path.isfile(filein))
    # output path
    fileout = os.path.join(test_dir, 'lookup_table_0.txt')
    if os.path.isfile(fileout):
        os.remove(fileout)
    # call processfunction
    LookupTable = LOOKUP_TABLE()
    LookupTable.ReadHeFestoTable(filein)
    field_names = ['Pressure', 'Temperature', 'Density', 'Thermal_expansivity', 'Isobaric_heat_capacity']
    LookupTable.Process(field_names, fileout)
    # assert something 
    assert(os.path.isfile(fileout))


def test_read_dimensions():
    '''
    test reading dimension information
    Asserts:
        read in the correct information
    '''
    # read data
    filein = os.path.join(source_dir, 'lookup_table.txt')
    Plotter = LINEARPLOT('hefesto', {})
    Plotter.ReadHeader(filein)
    Plotter.ReadData(filein)
    col_P = Plotter.header['Pressure']['col']
    min, delta, number = ReadFirstDimension(Plotter.data[:, col_P])
    # check results
    tolerance = 1e-6
    assert(abs(min - 0.0) < tolerance)
    assert(abs(delta - 0.01) / 0.01 < tolerance)
    assert(abs(number - 5) / 4 < tolerance)
    # second dimension
    col_T = Plotter.header['Temperature']['col']
    min, delta, number = ReadSecondDimension(Plotter.data[:, col_T])
    assert(abs(min - 800.0) / 800.0 < tolerance)
    assert(abs(delta - 1.0) / 1.0 < tolerance)
    assert(abs(number - 2) / 2 < tolerance)


def test_read_perplex_and_interpolate():
    '''
    test function ReadPerplex and Interpolate
    assert:
        1. name of the first dimension
        2. interpolation results
    '''
    # input file
    filein = os.path.join(source_dir, 'perplex_lookup_table.txt')
    assert(os.path.isfile(filein))
    LookupTable = LOOKUP_TABLE()
    LookupTable.ReadPerplex(filein, header_rows=4)
    LookupTable.Update()

    assert(LookupTable.first_dimension_name == "Temperature")
    T0, P0 = 2500.0, 4000.0 # K, bar
    results = LookupTable.Interpolate(T0, P0) 

    expected = np.array([
        2.49995409e+03,
        4.00000000e+03,
        3.04532466e+03,
        2.40057528e+03,
        5.04265728e-05,
        3.30272161e+06,
        8.82618008e+00,
        4.69101241e+00,
       -1.01589981e+07
    ])

    assert_allclose(results, expected, rtol=1e-6, atol=1e-8)


def test_process_hefesto_fort56():
    '''
    Test processing hefesto table
    Asserts:
    '''
    input_file = os.path.join(source_dir, "fort.56.PT")
    assert(os.path.isfile(input_file))  # assert there is an existing Hefesto table
    output_file = os.path.join(test_dir, "hefesto_table_from_fort56")
    if (os.path.isfile(output_file)):  # remove old files
        os.remove(output_file)
    output_file_std = os.path.join(source_dir, "hefesto_table_from_fort56_std")
    assert(os.path.isfile(output_file_std))  # assert there is an existing standard file
    
    # call processfunction
    LookupTable = LOOKUP_TABLE()
    LookupTable.ReadRawFort56(input_file)
    # fields to read in
    field_names = ['Pressure', 'Temperature', 'Density', 'Thermal_expansivity', 'Isobaric_heat_capacity', 'VP', 'VS', 'Enthalpy']
    LookupTable.Process(field_names, output_file, interval1=1, interval2=1)
    
    # assert something 
    assert(os.path.isfile(output_file))
    
    # filecmp
    assert(filecmp.cmp(output_file, output_file_std))


def test_distribute_parallel_control():
    '''
    assert function DistributeParallelControl
    '''
    case_dir = os.path.join(test_dir, "test_hefesto_parallel")
    # remove older results
    if os.path.isdir(case_dir):
        rmtree(case_dir)

    json_file = os.path.join(source_dir, "test_hefesto.json")
    HeFESTo_Opt = HEFESTO_OPT()
    # read in json options
    if type(json_file) == str:
        if not os.access(json_file, os.R_OK):
            raise FileNotFoundError("%s doesn't exist" % json_file)
        HeFESTo_Opt.read_json(json_file)
    elif type(json_file) == dict:
        HeFESTo_Opt.import_options(json_file)
    else:
        raise TypeError("Type of json_opt must by str or dict")
    
    DistributeParallelControl(*HeFESTo_Opt.to_distribute_parallel_control())

    # check directories
    assert(os.path.isdir(case_dir))
    sub0_dir = os.path.join(case_dir, "sub_0000")
    assert(os.path.isdir(sub0_dir))
    control0_path = os.path.join(sub0_dir, "control")
    assert(os.path.isfile(control0_path))
    sub1_dir = os.path.join(case_dir, "sub_0001")
    assert(os.path.isdir(sub1_dir))
    sub2_dir = os.path.join(case_dir, "sub_0002")
    assert(os.path.isdir(sub2_dir))

    # check the control file
    control2_std_path = os.path.join(source_dir, "control_std")
    control2_path = os.path.join(sub2_dir, "control")
    assert(os.path.isfile(control2_path))
    assert(filecmp.cmp(control2_path, control2_std_path))


def test_distribute_parallel_control_with_composition():
    '''
    assert function DistributeParallelControl
    '''
    case_dir = os.path.join(test_dir, "test_hefesto_parallel_with_composition")
    # remove older results
    if os.path.isdir(case_dir):
        rmtree(case_dir)

    # convert from mol fraction
    comps = [38.71, 49.85, 6.17, 2.94, 2.22, 0.11] # pyrolite
    comps_atom = convert_mol_fraction(comps)

    json_file = os.path.join(source_dir, "test_hefesto_with_composition.json")
    HeFESTo_Opt = HEFESTO_OPT()
    # read in json options
    if type(json_file) == str:
        if not os.access(json_file, os.R_OK):
            raise FileNotFoundError("%s doesn't exist" % json_file)
        HeFESTo_Opt.read_json(json_file)
    elif type(json_file) == dict:
        HeFESTo_Opt.import_options(json_file)
    else:
        raise TypeError("Type of json_opt must by str or dict")
    
    DistributeParallelControl(*HeFESTo_Opt.to_distribute_parallel_control(),
                              composition=comps_atom)

    # check directories
    assert(os.path.isdir(case_dir))
    sub0_dir = os.path.join(case_dir, "sub_0000")
    assert(os.path.isdir(sub0_dir))
    control0_path = os.path.join(sub0_dir, "control")
    assert(os.path.isfile(control0_path))
    sub1_dir = os.path.join(case_dir, "sub_0001")
    assert(os.path.isdir(sub1_dir))
    sub2_dir = os.path.join(case_dir, "sub_0002")
    assert(os.path.isdir(sub2_dir))

    # check the control file
    control2_std_path = os.path.join(source_dir, "control_with_composition_std")
    control2_path = os.path.join(sub2_dir, "control")
    assert(os.path.isfile(control2_path))
    assert(filecmp.cmp(control2_path, control2_std_path))


def test_ExchangeDimensions():
    '''
    test function ExchangeDimensions
    '''
    input_file = os.path.join(source_dir, "table_index_test.txt")
    indexes = np.loadtxt(input_file)
    number_out1 = 26
    number_out2 = 4
    ex_indexes = ExchangeDimensions(indexes, number_out1, number_out2)
    # save result to file
    output_file = os.path.join(test_dir, "exchange_dimensions_output.txt")
    if os.path.isfile(output_file):
        os.remove(output_file)
    with open(output_file, 'w') as fout:
        np.savetxt(fout, ex_indexes)
    assert(os.path.isfile(output_file))
    # compare
    output_std_file = os.path.join(source_dir, "exchange_dimensions_output_std.txt")
    assert(os.path.isfile(output_std_file))
    assert(filecmp.cmp(output_file, output_std_file))