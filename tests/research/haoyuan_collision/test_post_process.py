import pytest
import os
import filecmp
from shutil import rmtree
from pathlib import Path
import numpy as np

from hamageolib.research.haoyuan_collision0.post_process import ProcessVtuFileTwoDStep
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


