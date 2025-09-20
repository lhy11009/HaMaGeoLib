import os
import pytest
import filecmp  # for compare file contents
from shutil import rmtree
from pathlib import Path
import math
    
from hamageolib.research.mow_subduction.case_options import CASE_OPTIONS_TWOD, CASE_OPTIONS
from hamageolib.research.haoyuan_3d_subduction.post_process import ProcessVtuFileTwoDStep, ProcessVtuFileThDStep


BASE_TEST_DIR = Path(".test") / "haoyuan_metastable_post_process"
if BASE_TEST_DIR.exists():
    rmtree(BASE_TEST_DIR)
BASE_TEST_DIR.mkdir(parents=True, exist_ok=True)

# test_dir = os.path.join(".test", "haoyuan_metastable_post_process")
# if os.path.isdir(test_dir):
#     rmtree(test_dir)
# os.mkdir(test_dir)

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_metastable_trivial():
    '''
    test generating the metastable grid with a trivial step (no mow)
    '''

    # test processing the 2d case
    local_dir_2d=os.path.join("big_tests", "MOW", "eba2d_width80_h1000_bw4000_sw1000_yd300_M_fix")
    pvtu_step = 4

    # remove old results 
    output_dir=os.path.join(local_dir_2d, "pyvisita_outputs")
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    Case_Options_2d = CASE_OPTIONS_TWOD(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)

    # check value of metastable area
    assert(abs(output_dict["metastable_area"]) < 1e-6) 
    assert(abs(output_dict["metastable_area_cold"]) < 1e-6) 


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_metastable():
    '''
    test generating the metastable grid, check the metastable_area_cold in a
    step where the cold area is approximate the total metastable_area
    '''

    # test processing the 2d case
    local_dir_2d=os.path.join("big_tests", "MOW", "eba2d_width80_h1000_bw4000_sw1000_yd300_M_fix")
    pvtu_step = 24

    # remove old results 
    output_dir=os.path.join(local_dir_2d, "pyvisita_outputs")
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    Case_Options_2d = CASE_OPTIONS_TWOD(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)

    # check value of metastable area
    metastable_area_std = 6968080078.125 # m^3
    assert(abs((output_dict["metastable_area"] - metastable_area_std)/metastable_area_std) < 1e-6) 
    
    # check value of cold metastable area
    metastable_area_cold_std = 6123460937.5 # m^3
    assert(abs((output_dict["metastable_area_cold"] - metastable_area_cold_std)/metastable_area_cold_std) < 1e-6) 


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_metastable_no_cold_area():
    '''
    test generating the metastable grid, check the metastable_area_cold again
    for a step where code area is non-existing
    '''

    # test processing the 2d case
    local_dir_2d=os.path.join("big_tests", "MOW", "eba2d_width80_h1000_bw4000_sw1000_yd300_M_fix")
    pvtu_step = 80

    # remove old results 
    output_dir=os.path.join(local_dir_2d, "pyvisita_outputs")
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    Case_Options_2d = CASE_OPTIONS_TWOD(local_dir_2d)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(local_dir_2d, "summary.csv"))

    output_dict = ProcessVtuFileTwoDStep(local_dir_2d, pvtu_step, Case_Options_2d)

    # check value of metastable area
    metastable_area_std = 2111542968.75 # m^3
    assert(abs((output_dict["metastable_area"] - metastable_area_std)/metastable_area_std) < 1e-6) 
    
    # check value of cold metastable area
    metastable_area_cold_std = 497720703.125 # m^3
    assert(abs((output_dict["metastable_area_cold"] - metastable_area_cold_std)/metastable_area_cold_std) < 1e-6) 


@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_thd_metastable():


    local_dir = os.path.join("big_tests", "MOW", "C_mow_h2890.0_M_gr3_ar4")
    test_dir = BASE_TEST_DIR / "test_pyvista_process_thd_metastable"

    test_dir.mkdir(parents=True, exist_ok=True)
    
    pvtu_step = 25

    Case_Options = CASE_OPTIONS(local_dir)
    Case_Options.Interpret()
    Case_Options.SummaryCaseVtuStep(os.path.join(local_dir, "summary.csv"))

    PprocessThD, outputs = ProcessVtuFileThDStep(local_dir, pvtu_step, Case_Options, odir=test_dir, do_clip=True,\
                                                 threshold_lower=0.5, threshold_upper=0.5)
    
    assert(math.isclose(PprocessThD.metastable_volume, 7.222932654311768e15, rel_tol=1e-6))
    assert(math.isclose(PprocessThD.metastable_volume_cold, 5.513010062993408e15, rel_tol=1e-6))
    assert(math.isclose(PprocessThD.metastable_area_center, 9.0663777258125e9, rel_tol=1e-6))
    assert(math.isclose(PprocessThD.metastable_area_cold_center, 7.119436112875e9, rel_tol=1e-6))