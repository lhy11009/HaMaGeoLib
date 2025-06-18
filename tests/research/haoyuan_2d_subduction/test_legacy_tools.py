"""
Tests for legacy_tools.py in the HaMaGeoLib research module.
"""

import pytest
import filecmp  # for compare file contents
import numpy as np
from unittest import mock
from shutil import rmtree  # for remove directories

from hamageolib.research.haoyuan_2d_subduction.legacy_tools import *


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_2d_subduction")


# ---------------------------------------------------------------------
# Check and make test directories
# ---------------------------------------------------------------------
test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_2d_subduction-test_legacy_tools"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)


def test_preliminary():
    assert(os.path.isdir(fixture_root))

# ---------------------------------------------------------------------
# Tests for rheology
# ---------------------------------------------------------------------
def test_AspectViscoPlastic():
    """
    check the implementation of aspcet rheology

    Tolerence set to be 1%
    """
    tolerance = 0.01
    check_result = [9.1049e+20]
    # these are two examples from aspect
    # I cannot track where this is from, as it's from a previous
    #version of codes, and I didn't document them properly
    diffusion_creep = \
        {
            "A": 8.571e-16,
            "d": 8.20e-3,
            "m": 3.0,
            "n": 1.0,
            "E": 335.0e3,
            "V": 4.0e-6
        }
    dislocation_creep = \
       {
           "A": 6.859e-15,
           "d": 8.20e-3,
           "m": 1.0,
           "n": 3.5,
           "E": 480.0e3,
           "V": 11.0e-6
       }
    check0 = CreepRheologyInAspectViscoPlastic(diffusion_creep, 1e-15, 10e9, 1300 + 273.15)
    assert(abs(check0 - check_result[0]) / check_result[0] < tolerance)


def test_CoulumbYielding():
    '''
    test the function CoulumbYielding
    '''
    # test 1: dry profile
    P = 500e6
    cohesion=2e6
    friction=0.6
    tau_std = 302e6
    tau = CoulumbYielding(P, cohesion, friction)
    assert(abs(tau - tau_std)/tau_std < 1e-6)
    # test 2: Otherwise same as 1, hydrate profile with _lambda = 0.1
    P = 500e6
    cohesion=2e6
    friction=0.6
    _lambda=0.1
    tau_std = 271.8e6
    tau = CoulumbYielding(P, cohesion, friction, _lambda)
    assert(abs(tau - tau_std)/tau_std < 1e-6)


def test_MK10_peierls():
    '''
    test the MK10 rheology
    assert:
        1. strain rates match with the values from the figure 5 in the original figure
        2. stress value could match vice versa
        3. the viscosity
    '''
    creep = GetPeierlsRheology("MK10")
    # assert 1.1, values at a strain rate = 3e-5
    strain_rate_std = 2.7778604629458894e-05
    stress = 3.82e3 # MPa
    T = 671.0 # K
    P = 0 # not dependent on P (V = 0)
    strain_rate = PeierlsCreepStrainRate(creep, stress, P, T)
    assert(abs(strain_rate_std - strain_rate)/strain_rate_std < 1e-6)
    # assert 1.2
    strain_rate_std = 2.6215973389278528e-05
    stress = 3.25e3 # MPa
    T = 907.0 # K
    P = 0 # not dependent on P (V = 0)
    strain_rate = PeierlsCreepStrainRate(creep, stress, P, T)
    assert(abs(strain_rate_std - strain_rate)/strain_rate_std < 1e-6)
    # assert 1.3: use a variation with activation volume
    strain_rate_std = 2.9979043137974457e-05
    T = 873.0
    P = 4.5e9
    stress = 3.35e3 # MPa
    strain_rate = PeierlsCreepStrainRate(creep, stress, P, T) 
    assert(abs(strain_rate_std - strain_rate)/strain_rate_std < 1e-6)
    strain_rate_std = 2.9979043137974457e-05
    T = 873.0
    P = 4.5e9
    dV = 30e-6
    stress = 3.35e3 # MPa
    strain_rate = PeierlsCreepStrainRate(creep, stress, P, T, dV=dV) 
    assert(abs(strain_rate_std - strain_rate)/strain_rate_std < 1e-6)
    strain_rate_std = 1.0823657173643094e-05
    T = 873.0
    P = 5.5e9
    dV = 30e-6
    stress = 3.35e3 # MPa
    strain_rate = PeierlsCreepStrainRate(creep, stress, P, T, dV=dV) 
    assert(abs(strain_rate_std - strain_rate)/strain_rate_std < 1e-6)
    # assert 2.1
    strain_rate = 2.7778604629458894e-05
    T = 671.0 # K
    P = 0 # not dependent on P (V = 0)
    stress_std = 3.82e3 # MPa
    stress = PeierlsCreepStress(creep, strain_rate, P, T)
    assert(abs(np.log(stress_std/stress)) < 0.05)  # a bigger tolerance, check the log value
    # assert 2.2
    strain_rate = 2.6215973389278528e-05
    T = 907.0 # K
    P = 0 # not dependent on P (V = 0)
    stress_std = 3.25e3 # MPa
    stress = PeierlsCreepStress(creep, strain_rate, P, T)
    assert(abs(np.log(stress_std/stress)) < 0.05)  # a bigger tolerance, check the log value
    # assert 3.1: viscosity
    strain_rate = 2.7778604629458894e-05
    T = 671.0 # K
    P = 0 # not dependent on P (V = 0)
    eta_std = 6.8757953e13
    eta = PeierlsCreepRheology(creep, strain_rate, P, T)
    assert(abs(np.log(eta_std/eta)) < 0.05)  # a bigger tolerance, check the log value
    # assert 3.2: viscosity, a realistic scenario
    # 1e-13
    strain_rate = 1e-13
    T = 800.0 + 273.15 # K
    P = 0 # not dependent on P (V = 0)
    eta_std = 2.125142090914006e+21
    eta = PeierlsCreepRheology(creep, strain_rate, P, T)
    assert(abs(np.log(eta_std/eta)) < 0.05)  # a bigger tolerance, check the log value
    # 1e-15
    strain_rate = 1e-15
    T = 800.0 + 273.15 # K
    P = 0 # not dependent on P (V = 0)
    eta_std = 9.893654111645399e+22
    eta = PeierlsCreepRheology(creep, strain_rate, P, T)
    assert(abs(np.log(eta_std/eta)) < 0.05)  # a bigger tolerance, check the log value

# ---------------------------------------------------------------------
# Tests for thermal models
# ---------------------------------------------------------------------
def test_plate_model():
    '''
    Test the implementation of the plate model
    Asserts:
        a. the value of PM_A is the same with my hand calculation
        a. the value of PM_B is the same with my hand calculation
    '''
    tolerance = 1e-6
    year = 365 * 24 * 3600.0  # s in year
    Pmodel = PLATE_MODEL(150e3, 1e-6, 273.0, 1673.0, 0.05/year) # initiate the plate model
    # assert the PM_A factor is consistent with my hand calculation
    assert(abs(Pmodel.PM_A(1, 1e6*year) - 0.6278754063131151)/0.6278754063131151 < 1e-6)
    assert(abs(Pmodel.PM_A(3, 1e6*year) - 0.1874020031998469)/0.1874020031998469 < 1e-6)
    # assert the PM_B factor is consistent with my hand calculation
    assert(abs(Pmodel.PM_B(0, 1e6*year) - 59957.684736338706)/59957.684736338706 < 1e-6)
    assert(abs(Pmodel.PM_B(1, 1e6*year) - 5965.19103094)/5965.19103094 < 1e-6)

    year = 365.25 * 24 * 3600.0  # s in year
    Pmodel1 = PLATE_MODEL(150e3, 0.804e-6, 273.0, 1673.0, 0.05/year) # initiate the plate model
    # test 3: check the temperature
    # 0 km, 50 km, 95 km (plate thickness)
    # the following parameters are from T&S book
    Myr = 1e6 * year
    Pmodel2 = PLATE_MODEL(95e3,1e-6, 273.0, 1573.0) # initiate the plate model
    Ts = Pmodel2.T(np.array([0.0, 50e3, 95e3]), 60.4 * Myr) 
    T0km_std = 273.0
    assert(abs(Ts[0] - T0km_std) / T0km_std < 1e-6)
    T50km_std = 1059.7755706447608
    assert(abs(Ts[1] - T50km_std) / T50km_std < 1e-6)
    T95km_std = 1573.0
    assert(abs(Ts[2] - T95km_std) / T95km_std < 1e-6)


# todo_cv
# ---------------------------------------------------------------------
# Tests for generating visualization plots
# ---------------------------------------------------------------------
@pytest.mark.big_test  # Optional marker for big tests
def test_get_snaps_steps():
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", "test_get_snaps_steps")
    
    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")
    
    # call function for graphical outputs
    snaps, times, steps = GetSnapsSteps(case_path)
    # assertions
    assert(snaps == [6, 7, 8, 9])
    assert(times == [0.0, 100000.0, 200000.0, 300000.0])
    assert(steps == [0, 104, 231, 373])
    
    # call function for particle outputs
    snaps, times, steps = GetSnapsSteps(case_path, 'particle')
    # assertions
    assert(snaps == [0, 1])
    assert(times == [0.0, 2e5])
    assert(steps == [0, 231])


@pytest.mark.big_test  # Optional marker for big tests
def test_visit_options(): 
    # check visit_options (interpret script from standard ones)
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'test_visit')

    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")
   
    source_dir = os.path.join(fixture_root, "test_visit")
    Visit_Options = VISIT_OPTIONS(case_path)
    # call function
    Visit_Options.Interpret()
    ofile = os.path.join(test_dir, 'temperature.py')
    visit_script = os.path.join(source_dir, 'temperature.py')
    visit_script_base = os.path.join(source_dir, 'base.py')
    Visit_Options.read_contents(visit_script_base, visit_script)
    # make a new directory
    img_dir = os.path.join(test_dir, 'img')
    if os.path.isdir(img_dir):
        rmtree(img_dir)
    os.mkdir(img_dir)
    Visit_Options.options["IMG_OUTPUT_DIR"] = img_dir
    Visit_Options.substitute()
    ofile_path = Visit_Options.save(ofile)
    # assert file generated
    assert(os.path.isfile(ofile_path))
    # assert file is identical with standard
    ofile_std = os.path.join(source_dir, 'temperature_std.py')
    assert(os.path.isfile(ofile_std))
    assert(filecmp.cmp(ofile_path, ofile_std))

@pytest.mark.big_test  # Optional marker for big tests
def test_visit_options_default(): 
    # check visit_options (interpret script from standard ones)
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'test_visit_default')

    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")
    
    # check visit_options (interpret script from standard ones)
    source_dir = os.path.join(fixture_root, "test_visit")
    Visit_Options = VISIT_OPTIONS(case_path)
    # call function
    Visit_Options.Interpret()
    ofile = os.path.join(test_dir, 'default.py')
    visit_script = os.path.join(source_dir, 'default.py')
    visit_script_base = os.path.join(source_dir, 'base.py')
    Visit_Options.read_contents(visit_script_base, visit_script)
    # make a new directory
    img_dir = os.path.join(test_dir, 'img')
    if os.path.isdir(img_dir):
        rmtree(img_dir)
    os.mkdir(img_dir)
    Visit_Options.options["IMG_OUTPUT_DIR"] = img_dir
    Visit_Options.substitute()
    ofile_path = Visit_Options.save(ofile)
    # assert file generated
    assert(os.path.isfile(ofile_path))
    # assert file is identical with standard
    ofile_std = os.path.join(source_dir, 'default_std.py')
    assert(os.path.isfile(ofile_std))
    assert(filecmp.cmp(ofile_path, ofile_std))


# ---------------------------------------------------------------------
# Tests for generating linear plots
# ---------------------------------------------------------------------
@pytest.fixture
def mock_json(monkeypatch):
    dummy_options = {
        "test_plot": {
            "canvas": [1, 1],
            "types": ["main"],
            "main": {
                "xname": "Time", "yname": "Number_of_mesh_cells",
                "color": "r", "label": "Test", "line": "-"
            }
        }
    }
    monkeypatch.setattr("builtins.open", mock.mock_open(read_data='{"test_plot": {"canvas": [1, 1], "types": ["main"], "main": {"xname": "Time", "yname": "Number_of_mesh_cells", "color": "r", "label": "Test", "line": "-"}}}'))
    monkeypatch.setattr("json.load", lambda f: dummy_options)

@pytest.fixture
def dummy_header():
    return {
        "Time": {"col": 0, "unit": "s"},
        "Number_of_mesh_cells": {"col": 1, "unit": "count"},
        "total_col": 2
    }

@pytest.fixture
def dummy_data():
    return np.array([[0.0, 100.0], [1.0, 200.0]])

def test_init(mock_json):
    plot = LINEARPLOT("test_plot", options={"dim": 2})
    assert plot.name == "test_plot"
    assert plot.dim == 2
    assert isinstance(plot.options, dict)

def test_read_data_float(monkeypatch, dummy_data):
    plot = LINEARPLOT("test_plot", options={"dim": 2})
    monkeypatch.setattr("os.access", lambda path, mode: True)
    monkeypatch.setattr("numpy.genfromtxt", lambda *args, **kwargs: dummy_data)
    status = plot.ReadData("dummy.txt", dtype=float)
    assert status == 0
    assert plot.data.shape == (2, 2)

def test_has_field(dummy_header):
    plot = LINEARPLOT("test_plot", options={"dim": 2})
    plot.header = dummy_header
    assert plot.Has("Time") is True
    assert plot.Has("Nonexistent") is False

def test_has_data(dummy_data):
    plot = LINEARPLOT("test_plot", options={"dim": 2})
    plot.data = dummy_data
    assert plot.HasData() is True
    plot.data = np.array([])
    assert plot.HasData() is False

def test_manage_data(dummy_data):
    plot = LINEARPLOT("test_plot", options={"dim": 2})
    plot.data = dummy_data
    data_list = plot.ManageData()
    assert isinstance(data_list, list)
    assert len(data_list) == dummy_data.shape[1]
    assert np.allclose(data_list[0], dummy_data[:, 0])


# ---------------------------------------------------------------------
# Tests for vtk utilities
# ---------------------------------------------------------------------
@pytest.mark.big_test  # Optional marker for big tests
def test_shallow_trench():
    '''
    test the implementation of shallow trench position
    this test only deal with the generation of the data file without generating any plots
    '''
    # test 2: a different snapshot. Initially, the number of points in the cmb envelop is 
    # different from the number of points in the slab envelops, thus, this test assert that
    # this is fixed in the output stage
     
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'EBA_CDPT_test_perplex_mixing_log')
    
    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")

    o_dir = os.path.join(test_dir, "TwoDSubduction_vtk_pp")
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    vtu_snapshot = 104 # 0 Ma
    outputs_std = "100         702         1.0003e+07    6.2646e-01    9.5393e+05    9.6860e-01    3.9703e-02    6.3861e-02    6.0904e-01    \n"
    _, outputs = SlabMorphology_dual_mdd(case_path, vtu_snapshot, find_shallow_trench=True, output_path=test_dir)
    assert(outputs==outputs_std)


@pytest.mark.big_test  # Optional marker for big tests
def test_slab_mdd():
    '''
    test extract the slab mdd and the horizontal profile
    this test only deal with the generation of the data file without generating any plots
    '''
    # test 2: a different snapshot. Initially, the number of points in the cmb envelop is 
    # different from the number of points in the slab envelops, thus, this test assert that
    # this is fixed in the output stage
     
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'EBA_CDPT_test_perplex_mixing_log')
    
    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")

    o_dir = os.path.join(test_dir, "TwoDSubduction_vtk_pp")
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    o_file = os.path.join(test_dir, "test_slab_mdd", "mdd1_profile_00100.txt")
    o_file_std = os.path.join(case_path, "mdd1_profile_00100_std.txt")
    if os.path.isfile(o_file):
        os.remove(o_file)
    vtu_snapshot = 104 # 0 Ma

    # remove old results
    output_path = os.path.join(test_dir, "test_slab_mdd")
    if os.path.isdir(output_path):
        rmtree(output_path)
    os.mkdir(output_path)

    # run analysis
    SlabMorphology_dual_mdd(case_path, vtu_snapshot, project_velocity=True, findmdd=True, find_shallow_trench=True, output_path=output_path)
    assert(os.path.isfile(o_file))  # assert the outputs of temperature profiles
    assert(filecmp.cmp(o_file, o_file_std))  # compare file contents


@pytest.mark.big_test  # Optional marker for big tests
def test_slab_temperature_offsets():
    '''
    test the implementations of SlabTemperature
    this test only deal with the generation of the data file without generating any plots
    This test apply an offset to the slab surface profile to look at profile in the mantle wedge
    '''
     
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'EBA_CDPT_test_perplex_mixing_log')
    
    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")

    o_dir = os.path.join(test_dir, "TwoDSubduction_vtk_pp")
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    o_file = os.path.join(test_dir, "slab_temperature_00104.txt")
    o_file_std = os.path.join(case_path, "slab_temperature_00104_offset_std.txt")
    if os.path.isfile(o_file):
        os.remove(o_file)
    vtu_snapshot = 104 # 0 Ma
    _, _, _ = SlabTemperature(case_path, vtu_snapshot, o_file, output_slab=True, offsets=[-5e3, -10e3])
    assert(os.path.isfile(o_file))  # assert the outputs of temperature profiles
    assert(filecmp.cmp(o_file, o_file_std))  # compare file contents

# todo_thickness
@pytest.mark.big_test  # Optional marker for big tests
def test_slab_temperature_crust_thickness():
    '''
    test the implementations of SlabTemperature
    this test only deal with the generation of the data file without generating any plots
    This test apply an offset to the slab surface profile to look at profile in the mantle wedge
    ''' 
    case_path = os.path.join(package_root, "big_tests", "TwoDSubduction", 'EBA_CDPT_test_perplex_mixing_log')
    
    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")

    o_dir = os.path.join(test_dir, "TwoDSubduction_vtk_pp")
    if not os.path.isdir(o_dir):
        os.mkdir(o_dir)
    o_file = os.path.join(test_dir, "slab_temperature_00104.txt")
    o_file_std = os.path.join(case_path, "slab_temperature_00104_crust_thickness_std.txt")
    if os.path.isfile(o_file):
        os.remove(o_file)
    vtu_snapshot = 104 # 0 Ma
    _, _, _ = SlabTemperature(case_path, vtu_snapshot, o_file, output_slab=True, fix_shallow=True, compute_crust_thickness=True)
    assert(os.path.isfile(o_file))  # assert the outputs of temperature profiles
    assert(filecmp.cmp(o_file, o_file_std))  # compare file contents

########################################
# Tests for case options
########################################
def test_create_case():
    '''
    Use the interface defined in Cases.py. Take a inputs file, do a little multilation and create a new case
    Asserts:
        cases in created(files, contents, etc)
        assert prm file is generated
        assert prm file for fast running 0th step is generated
    '''
    def ConfigureFoo(Inputs, _config):
        """
        an example of configuation
        """
        return Inputs
    

    def ConfigureFoo1(Inputs, _config):
        """
        another example of configuation, second option for renaming
        """
        return Inputs, "_foo"
    
    source_dir = os.path.join(fixture_root, "test_create_case")
    prm_path = os.path.join(source_dir, 'case.prm')
    extra_path = os.path.join(source_dir, 'particle.dat')
    test_case = CASE('test_create_case', prm_path, False)
    case_output_dir = os.path.join(test_dir, 'test_create_case')
    if os.path.isdir(case_output_dir):
        rmtree(case_output_dir)
    test_case.configure(ConfigureFoo, 1)  # do nothing, test interface
    test_case.add_extra_file(extra_path)  # add an extra file
    # todo_intial
    test_case.create(test_dir, fast_first_step=1, test_initial_steps=(3, 1e4))
    # assert prm file is generated
    prm_output_path = os.path.join(case_output_dir, 'case.prm')
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    assert(os.path.isfile(prm_output_path))  # assert generated
    assert(filecmp.cmp(prm_output_path, prm_std_path))  # assert contents
    # assert prm file for fast running 0th step is generated
    prm_output_path = os.path.join(case_output_dir, 'case_f.prm')
    prm_std_path = os.path.join(source_dir, 'case_f_std.prm')
    assert(os.path.isfile(prm_output_path))  # assert generated
    assert(filecmp.cmp(prm_output_path, prm_std_path))  # assert contents
    # assert prm file for testing hte initial steps are generated
    prm_output_path = os.path.join(case_output_dir, 'case_ini.prm')
    prm_std_path = os.path.join(source_dir, 'case_ini_std.prm')
    assert(os.path.isfile(prm_output_path))  # assert generated
    assert(filecmp.cmp(prm_output_path, prm_std_path))  # assert contents
    # assert extra file is generated
    extra_output_path = os.path.join(case_output_dir, 'particle.dat')
    assert(os.path.isfile(extra_output_path))  # assert generated

    # test 2: renaming
    test_case.configure(ConfigureFoo1, 1, rename=True)  # do nothing, test interface
    test_case.create(test_dir)
    # renaming: add string '_foo'
    case_output_dir = os.path.join(test_dir, 'test_create_case_foo')
    # assert prm file is generated
    prm_output_path = os.path.join(case_output_dir, 'case.prm')
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    assert(os.path.isfile(prm_output_path))  # assert generated
    assert(filecmp.cmp(prm_output_path, prm_std_path))  # assert contents
    # assert extra file is generated
    extra_output_path = os.path.join(case_output_dir, 'particle.dat')
    assert(os.path.isfile(extra_output_path))  # assert generated

def test_visc_diff():
    '''
    test the particle method
    '''
    source_dir = os.path.join(fixture_root, "test_visc_diff")
    
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_visc_diff')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_0_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_0_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))


def test_sz_same_composition():
    '''
    values in the CDPT clapeyron slope
    '''
    source_dir = os.path.join(fixture_root, "test_sz_same_composition")

    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_sz_same_composition')
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    
    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_0_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_0_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))


def test_chunk0():
    '''
    test for setting the 3d case in the chunk geometry
    '''
    source_dir = os.path.join(fixture_root, "test_chunk0")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_chunk0')
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    create_case_with_json(json_path, CASE_THD, CASE_OPT_THD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))
