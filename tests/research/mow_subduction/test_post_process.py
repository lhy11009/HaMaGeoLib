import os
import pytest
import filecmp  # for compare file contents
from shutil import rmtree

@pytest.mark.big_test  # Optional marker for big tests
def test_pyvista_process_twod_metastable():

    from hamageolib.research.mow_subduction.case_options import CASE_OPTIONS_TWOD
    from hamageolib.research.haoyuan_3d_subduction.post_process import ProcessVtuFileTwoDStep

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