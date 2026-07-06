import pytest
import os
import filecmp
from shutil import rmtree
from pathlib import Path
import numpy as np

from hamageolib.research.haoyuan_collision0.post_process import *
from hamageolib.research.haoyuan_collision0.case_options import CASE_OPTIONS_TWOD

# Resolve the root of the pakage and set up
# test directory
package_root = Path(__file__).resolve().parents[3]
fixture_root = package_root/"tests/fixtures/research/haoyuan_collision"
big_fixture_root = package_root/"big_tests/Collision0"
test_root = package_root/".test"
test_root.mkdir(exist_ok=True)
test_dir = test_root/"research-haoyuan-post-process"
test_dir.mkdir(exist_ok=True)

@pytest.mark.big_test
def test_extract_slab_2d():

    source_case_path = big_fixture_root/"D1000_minV2.5e+19_WLV2.5e+19_PC5.0e-02_CTboth_Cn"
    test_case_path = test_dir/"test_extract_slab_2d"

    # make a clean test folder
    if test_case_path.is_dir():
        rmtree(test_case_path)
    test_case_path.mkdir(exist_ok=False)

    Case_Options_2d = CASE_OPTIONS_TWOD(source_case_path)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(test_case_path, "summary.csv"))
    
    # run the PyVista workflow
    outputs = ProcessVtuFileTwoDStep(source_case_path, 0, Case_Options_2d,
                                         pyvista_outdir=test_case_path/"pyvista_outputs")

    # compare file outputs 
    slab_surface_filepath = test_case_path/"pyvista_outputs"/"slab_surface_00000.vtp"
    slab_surface_std_filepath = source_case_path/"pyvista_outputs"/"slab_surface_00000.vtp"
    assert(slab_surface_filepath.is_file())
    assert(filecmp.cmp(slab_surface_filepath, slab_surface_std_filepath))

    # assert slab characteristics
    assert(np.isclose(outputs["slab_depth"], 165000.0))
    assert(np.isclose(outputs["dip_100"], 0.5931085755109071))
    assert(np.isclose(outputs["dip_300"], 1.050753540782733))
    assert(np.isclose(outputs["trench_center"], 5057674.5))
    assert(np.isclose(outputs["trench_center_50"], 5151631.5))


@pytest.mark.big_test
def test_extract_topography_from_pyvista_2d():
    
    source_case_path = big_fixture_root/"D2000_minV2.5e+19_Coh1.0e+02_WLS_WLF2.0e-02_WLM2.5e+19_CTboth_SL2.00e+06_Cn_PTr"
    test_case_path = test_dir/"test_extract_topography_from_pyvista_2d"
    
    # make a clean test folder
    if test_case_path.is_dir():
        rmtree(test_case_path)
    test_case_path.mkdir(exist_ok=False)
    
    Case_Options_2d = CASE_OPTIONS_TWOD(source_case_path)
    Case_Options_2d.Interpret()
    Case_Options_2d.SummaryCaseVtuStep(os.path.join(test_case_path, "summary.csv"))
    
    # run the PyVista workflow
    outputs = ProcessVtuFileTwoDStep(source_case_path, 350, Case_Options_2d,
                                    pyvista_outdir=test_case_path/"pyvista_outputs",
                                    include_topography=True)
    
    topography_file = test_case_path/"pyvista_outputs/topography_00350.txt"
    topography_std_file = source_case_path/"topography_00350.txt"
    
    assert(filecmp.cmp(topography_file, topography_std_file))

# todo_pp
def test_main_postprocess_workflow():
    '''
    Test the main workflow for post-processing the collision0 project.
    Now only works for 2-d cases.
    Note the idea of this test is to check the workflow and see to it
    the files are generated, but not to check the contents of the files.
    '''
    
    source_case_path = big_fixture_root/"D2000_minV2.5e+19_Coh1.0e+02_WLS_WLF2.0e-02_WLM2.5e+19_CTboth_SL2.00e+06_Cn_PTr"
    pp_directory = os.path.join(test_dir, "test_main_postprocess_workflow")

    # make a new post-process directory
    if os.path.isdir(pp_directory):
        rmtree(pp_directory)
    os.mkdir(pp_directory)

    # Prepare case options
    # The second option of the function is is_process_second_stage, this is not needed
    # if not processing a dual-stage model 
    Case_Options_2d = prepare_case_option_2d(source_case_path, False,
                                            pp_directory=pp_directory)

    # Plot run time information
    # Note these "combined" functions are designed to address multiple cases
    # in a series manner, therefore when there is only one case, it has to
    # be put in a trivial array
    plot_run_time_combined([source_case_path], [Case_Options_2d])

    run_time_image_path = os.path.join(pp_directory, "img", "runtime_plots", "assembled.png")

    assert(os.path.isfile(run_time_image_path))

    # Post-processing the results involving vtu files
    # For this test, we just run the processing of one step
    process_all_vtu_steps(source_case_path, Case_Options_2d,
                          graphical_step_min=None,
                          graphical_step_max=None,
                          one_vtu_step=350,
                          include_particles=False,
                          include_topography=False,
                          analyze_shortening=False)
    
    final_vtu_file_path = os.path.join(pp_directory, "pyvista_outputs", "00350", "final_00350.vtu")
    assert(os.path.isfile(final_vtu_file_path))