# test_case_options.py

from hamageolib.utils.handy_shortcuts_haoyuan import check_float
from hamageolib.utils.case_options import *
import pytest
import pandas as pd
import numpy as np
import filecmp


def test_case_options(tmp_path):
    """
    Tests the functionality of the CASE_OPTIONS class by verifying that the 
    options dictionary is populated correctly for a specific test case.

    Parameters:
        tmp_path (Path): Temporary path provided by the testing framework for file handling during the test.
    """
    # Set up test case directory path
    case_dir = os.path.join(os.path.dirname(__file__), "../integration/fixtures/aspect_case_examples/case_2d_subduction")
    
    # Initialize CASE_OPTIONS instance and interpret options
    Case_Options = CASE_OPTIONS(case_dir)

    assert(Case_Options.aspect_version == '2.6.0-pre')
    assert(Case_Options.dealii_version == '9.5.0')
    assert(Case_Options.world_builder_version == 'nan')
    assert(Case_Options.n_mpi == 32)


def test_filter_and_match_single_series():
    # Test with a single additional Series
    series1 = pd.Series([1, 2, np.nan, 4, 5])
    series2 = pd.Series(['a', 'b', 'c', 'd', 'e'])

    indexes, non_nan_first, matched_series_list = filter_and_match(series1, series2)

    # Assertions
    expected_indexes = pd.Index([0, 1, 3, 4])
    expected_non_nan_first = pd.Series([1.0, 2.0, 4.0, 5.0], index=[0, 1, 3, 4])
    expected_matched_series = pd.Series(['a', 'b', 'd', 'e'], index=[0, 1, 3, 4])

    assert indexes.equals(expected_indexes)
    assert non_nan_first.equals(expected_non_nan_first)
    assert matched_series_list[0].equals(expected_matched_series)

def test_filter_and_match_multiple_series():
    # Test with multiple additional Series
    series1 = pd.Series([1, np.nan, 3, 4, np.nan])
    series2 = pd.Series(['x', 'y', 'z', 'w', 'v'])
    series3 = pd.Series([10, 20, 30, 40, 50])

    indexes, non_nan_first, matched_series_list = filter_and_match(series1, series2, series3)

    # Assertions
    expected_indexes = pd.Index([0, 2, 3])
    expected_non_nan_first = pd.Series([1.0, 3.0, 4.0], index=[0, 2, 3])
    expected_matched_series_2 = pd.Series(['x', 'z', 'w'], index=[0, 2, 3])
    expected_matched_series_3 = pd.Series([10, 30, 40], index=[0, 2, 3])

    assert indexes.equals(expected_indexes)
    assert non_nan_first.equals(expected_non_nan_first)
    assert matched_series_list[0].equals(expected_matched_series_2)
    assert matched_series_list[1].equals(expected_matched_series_3)

def test_filter_and_match_no_nan():
    # Test when series1 has no NaN values
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series(['p', 'q', 'r', 's', 't'])

    indexes, non_nan_first, matched_series_list = filter_and_match(series1, series2)

    # Assertions
    expected_indexes = pd.Index([0, 1, 2, 3, 4])
    expected_non_nan_first = series1
    expected_matched_series = series2

    assert indexes.equals(expected_indexes)
    assert non_nan_first.equals(expected_non_nan_first)
    assert matched_series_list[0].equals(expected_matched_series)

def test_filter_and_match_all_nan():
    # Test when series1 is entirely NaN
    series1 = pd.Series([np.nan, np.nan, np.nan])
    series2 = pd.Series(['a', 'b', 'c'])

    indexes, non_nan_first, matched_series_list = filter_and_match(series1, series2)
    # Assertions
    expected_indexes = pd.Index([])
    expected_non_nan_first = pd.Series([], dtype=float)
    expected_matched_series = pd.Series([], dtype=object)

    assert indexes.equals(expected_indexes)
    assert non_nan_first.equals(expected_non_nan_first)
    assert matched_series_list[0].equals(expected_matched_series)

def test_filter_and_match_empty_series():
    # Test when series1 is empty
    series1 = pd.Series([], dtype=float)
    series2 = pd.Series([], dtype=object)

    indexes, non_nan_first, matched_series_list = filter_and_match(series1, series2)

    # Assertions
    expected_indexes = pd.Index([])
    expected_non_nan_first = pd.Series([], dtype=float)
    expected_matched_series = pd.Series([], dtype=object)

    assert indexes.equals(expected_indexes)
    assert non_nan_first.equals(expected_non_nan_first)
    assert matched_series_list[0].equals(expected_matched_series)

@pytest.mark.big_test  # Optional marker for big tests
def test_case_summary_big_test():
    
    case_path = os.path.join(os.path.dirname(__file__), "../../big_tests/TwoDSubduction/eba_cdpt_coh500_SA80.0_cd7.5_log_ss300.0")

    o_path = os.path.join(os.path.dirname(__file__), "../../dtemp/case_summary_big_test.csv")
    if os.path.isfile(o_path):
        # remove old files
        os.remove(o_path)

    o_path_std = os.path.join(os.path.dirname(__file__), "../../big_tests/TwoDSubduction/case_summary_std.csv")

    # Check if the folder exists and contains test files
    if not os.path.exists(case_path) or not os.listdir(case_path):
        pytest.skip("Skipping test: big test contents not found in 'big_tests/'.")

    Case_Summary = CASE_SUMMARY()
    Case_Summary.update_single_case(case_path)

    Case_Summary.df.to_csv(o_path)

    filecmp.cmp(o_path, o_path_std)

@pytest.mark.big_test  # Optional marker for big tests
def test_find_case_files():
    
    case_path = os.path.join(os.path.dirname(__file__), "../../big_tests/TwoDSubduction/eba_cdpt_coh500_SA80.0_cd7.5_log_ss300.0")
    
    prm_file, wb_file = find_case_files(case_path)

    assert(prm_file == os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "big_tests/TwoDSubduction/eba_cdpt_coh500_SA80.0_cd7.5_log_ss300.0/case.prm"))
    assert(wb_file == os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), "big_tests/TwoDSubduction/eba_cdpt_coh500_SA80.0_cd7.5_log_ss300.0/case.wb"))


########################################
# tests for case configurations
########################################
from hamageolib.utils.case_options import ModelConfigManager
@pytest.fixture
def base_config():
    return {
        "Mesh refinement": {
            "Global refinement": 4,
            "Initial adaptive refinement": 2
        },
        "Geometry model": {
            "Slab angle": 30.0,
            "Slab depth": 100e3
        }
    }

@pytest.fixture
def patch_resolution():
    return {
        "Mesh refinement": {
            "Global refinement": 6
        }
    }

@pytest.fixture
def patch_slab():
    return {
        "Geometry model": {
            "Slab angle": 10.0
        }
    }

def test_apply_flat_patch(base_config):
    mgr = ModelConfigManager(base_config)
    flat_patch = {
        "New key": "test",
        "Mesh refinement": "override"
    }
    mgr.apply_patch(flat_patch)
    result = mgr.get_config()
    assert result["New key"] == "test"
    assert result["Mesh refinement"] == "override"

def test_apply_nested_patch_resolution(base_config, patch_resolution):
    mgr = ModelConfigManager(base_config)
    mgr.apply_nested_patch(patch_resolution)
    updated = mgr.get_config()
    assert updated["Mesh refinement"]["Global refinement"] == 6
    assert updated["Mesh refinement"]["Initial adaptive refinement"] == 2  # preserved

def test_apply_multiple_nested_patches(base_config, patch_resolution, patch_slab):
    mgr = ModelConfigManager(base_config)
    mgr.apply_nested_patch(patch_resolution)
    mgr.apply_nested_patch(patch_slab)
    updated = mgr.get_config()
    assert updated["Mesh refinement"]["Global refinement"] == 6
    assert updated["Geometry model"]["Slab angle"] == 10.0
    assert updated["Geometry model"]["Slab depth"] == 100e3  # unchanged

def test_patch_does_not_affect_original(base_config, patch_resolution):
    mgr = ModelConfigManager(base_config)
    mgr.apply_nested_patch(patch_resolution)
    assert base_config["Mesh refinement"]["Global refinement"] == 4  # original preserved