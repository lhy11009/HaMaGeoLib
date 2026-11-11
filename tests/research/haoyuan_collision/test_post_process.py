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


def test_integrate_topography_sorts_coordinates():
    x = np.array([2.0, 0.0, 1.0])
    topography = np.array([4.0, 0.0, 2.0])

    integrated, average = integrate_topography(x, topography)

    assert np.isclose(integrated, 4.0)
    assert np.isclose(average, 2.0)


def test_integrate_topography_averages_duplicate_coordinates():
    x = np.array([2.0, 1.0, 0.0, 1.0])
    topography = np.array([4.0, 1.0, 0.0, 3.0])

    integrated, average = integrate_topography(x, topography)

    assert np.isclose(integrated, 4.0)
    assert np.isclose(average, 2.0)


@pytest.mark.parametrize(
    "x, topography, message",
    [
        ([0.0], [1.0], "at least two topography points"),
        ([0.0, 1.0], [1.0], "same length"),
        ([1.0, 1.0], [0.0, 2.0], "two distinct coordinates"),
        ([0.0, np.nan], [0.0, 1.0], "finite values"),
    ],
)
def test_integrate_topography_rejects_invalid_inputs(x, topography, message):
    with pytest.raises(ValueError, match=message):
        integrate_topography(x, topography)


@pytest.mark.parametrize("include_topography", [True, False])
def test_case_summary_average_topography(tmp_path, monkeypatch, include_topography):
    import pandas as pd
    from hamageolib.research.haoyuan_collision0 import post_process

    # Nonuniform spacing makes a spatial average differ from a point mean.
    topo_dir = tmp_path / "output" / "topography"
    topo_dir.mkdir(parents=True)
    (tmp_path / "output" / "solution").mkdir()
    for step, heights in [(10, [400., 0., 100.]), (20, [800., 0., 200.])]:
        np.savetxt(topo_dir / f"topography.{step:05d}",
                   np.column_stack(([4000., 0., 1000.], [0., 0., 0.], heights)))

    processor = PYVISTA_PROCESS_COLLISION.__new__(PYVISTA_PROCESS_COLLISION)
    processor.data_dir = str(tmp_path / "output" / "solution")
    processor.Max0 = 10000.
    processor.grid = object()
    for name in ("read", "extract_slab", "get_active_deforming_region_near_suture",
                 "write_object_to_file"):
        monkeypatch.setattr(processor, name, lambda *a, **kw: None)
    monkeypatch.setattr(processor, "analyze_slab", lambda: {"trench_center_50": 2000.})
    monkeypatch.setattr(processor, "analyze_velocity", lambda *a: {})
    monkeypatch.setattr(post_process, "PYVISTA_PROCESS_COLLISION", lambda *a, **kw: processor)

    options = CASE_OPTIONS_TWOD.__new__(CASE_OPTIONS_TWOD)
    options.visualization_df = pd.DataFrame({
        "Vtu step": [0, 1], "Vtu snapshot": [0, 1],
        "Time": [0., 1e5], "Time step number": [10, 20], "File Exists": [True, True],
    })
    options.summary_file = str(tmp_path / "summary.csv")
    options.case_dir = str(tmp_path)
    options.pyvista_dir = str(tmp_path)
    options.options = {"INITIAL_ADAPTIVE_REFINEMENT": 0}
    monkeypatch.setattr(options, "get_pvtu_step", lambda step: step)
    monkeypatch.setattr(options, "retrieve_visualization_file",
                        lambda step: "output/solution/solution.pvtu")
    options.SummaryCaseVtuStep()

    process_all_vtu_steps(str(tmp_path), options, include_topography=include_topography)

    summary = pd.read_csv(options.summary_file)
    assert "average_topography" in summary.columns
    if include_topography:
        np.testing.assert_allclose(summary["average_topography"], [200., 400.])
    else:
        assert summary["average_topography"].isna().all()


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
    the files are generated.
    '''
    
    source_case_path = big_fixture_root/"D2000_minV2.5e+19_Coh1.0e+02_WLS_WLF2.0e-02_WLM2.5e+19_CTboth_SL2.00e+06_Cn_PTr"
    pp_directory = os.path.join(test_dir, "test_main_postprocess_workflow")

    # make a new post-process directory
    if os.path.isdir(pp_directory):
        rmtree(pp_directory)
    os.mkdir(pp_directory)

    # Prepare case options
    Case_Options_2d = prepare_case_option_2d(source_case_path,
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


def test_main_postprocess_workflow_particles():
    '''
    Test the main workflow for post-processing the collision0 project.
    Now only works for 2-d cases.
    Note this test builds upon the previous one and include particle output
    '''
    source_case_path = big_fixture_root/"D2000_minV2.5e+19_Coh1.0e+02_WLS_WLF2.0e-02_WLM2.5e+19_CTboth_SL2.00e+06_Cn_PTr"
    pp_directory = os.path.join(test_dir, "test_main_postprocess_workflow_particles")

    # make a new post-process directory
    if os.path.isdir(pp_directory):
        rmtree(pp_directory)
    os.mkdir(pp_directory)

    # Prepare case options
    Case_Options_2d = prepare_case_option_2d(source_case_path,
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
                          one_vtu_step=351,
                          include_particles=True,
                          include_topography=True,
                          analyze_shortening=True)
    
    final_vtu_file_path = os.path.join(pp_directory, "pyvista_outputs", "00351", "final_00351.vtu")
    assert(os.path.isfile(final_vtu_file_path))


def test_main_postprocess_fastscape():
    '''
    Test the main workflow for post-processing the collision0 project.
    Now only works for 2-d cases.
    Specifically, this test the workflow involving fastsacpe dataset as vtk outputs
    '''
    
    source_case_path = big_fixture_root/"C_ar4_WLF5.0e-02_SA100.0_CR_CRT2.00e+05_FS_Ebl1.00e-02"
    pp_directory = os.path.join(test_dir, "test_main_postprocess_workflow_fastscape")

    # make a new post-process directory
    if os.path.isdir(pp_directory):
        rmtree(pp_directory)
    os.mkdir(pp_directory)

    # Prepare case options
    Case_Options_2d = prepare_case_option_2d(source_case_path,
                                            pp_directory=pp_directory)


    # Post-processing the results involving vtu files
    # For this test, we just run the processing of one step
    process_all_vtu_steps(source_case_path, Case_Options_2d,
                          graphical_step_min=None,
                          graphical_step_max=None,
                          one_vtu_step=30,
                          include_particles=False,
                          include_fastscape=True,
                          include_topography=False,
                          analyze_shortening=False)
    
    final_vtu_file_path = os.path.join(pp_directory, "pyvista_outputs", "00030", "final_00030.vtu")
    assert(os.path.isfile(final_vtu_file_path))

    fastscape_average_file_path = os.path.join(pp_directory, "pyvista_outputs", "00030", "fastscape_00030.txt")
    assert(os.path.isfile(fastscape_average_file_path))
    fastscape_average_std_file_path = os.path.join(source_case_path, "fastscape_00030.txt")
    assert(filecmp.cmp(fastscape_average_file_path, fastscape_average_std_file_path))
