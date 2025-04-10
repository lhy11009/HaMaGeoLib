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
def test_slab_temperature():
    '''
    test the implementation of SlabTemperature
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
    o_file = os.path.join(test_dir, "slab_temperature_00104.txt")
    o_file_std = os.path.join(case_path, "slab_temperature_00104_std.txt")
    if os.path.isfile(o_file):
        os.remove(o_file)
    vtu_snapshot = 104 # 0 Ma
    _, _, _ = SlabTemperature(case_path, vtu_snapshot, o_file, output_slab=True)
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