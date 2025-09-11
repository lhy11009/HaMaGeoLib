import pytest
import filecmp  # for compare file contents
import os
import numpy as np
from shutil import rmtree  # for remove directories

from hamageolib.research.haoyuan_2d_subduction.workflow_scripts import run_2d_subduction_visualization
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import PlotCaseRunTwoD, CASE_TWOD, CASE_OPT_TWOD, CASE_THD, CASE_OPT_THD,\
    create_case_with_json
package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

# ---------------------------------------------------------------------
# Check and make test directories
# ---------------------------------------------------------------------

test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_metastable_subduction"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_metastable_subduction")
SCRIPT_DIR = os.path.join(package_root, "scripts")

# ---------------------------------------------------------------------
# Test generate cases
# ---------------------------------------------------------------------
def test_haoyuan_metastable_subduction():
    '''
    test for including metastable_subduction
    cartesian, 2d geometry
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300_M")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300_M')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    # print("output_dir: ", output_dir) # debug

    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))

def test_haoyuan_metastable_subduction_deactivated():
    '''
    test for including metastable_subduction
    cartesian, 3-d consistent geometry
    deactivated the metastable and it returns to a normal 2-d case
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    # print("output_dir: ", output_dir) # debug

    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))

def test_viscosity_profile_middle_mantle():
    '''
    test for including a middle mantle layer
    that have maximum rheology around 1100 km
    cartesian, 2-d consistent geometry
    deactivated the metastable and it returns to a normal 2-d case
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300_middle_mantle")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300_middle_mantle')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    # print("output_dir: ", output_dir) # debug

    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))

def test_viscosity_profile_middle_mantle_metastable():
    '''
    test for including a middle mantle layer
    that have maximum rheology around 1100 km
    cartesian, 2-d consistent geometry
    activate the metastable kinetics
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300_middle_mantle_metastable")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300_middle_mantle_metastable')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    # print("output_dir: ", output_dir) # debug

    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))


def test_haoyuan_metastable_subduction_3d():
    '''
    test for including metastable_subduction
    cartesian, 3d geometry
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300_3d_M")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300_3d_M')
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


def test_haoyuan_metastable_subduction_3d_deactivated():
    '''
    test for including metastable_subduction
    cartesian, 3d geometry, deactivate the MOW kinetics
    '''
    source_dir = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300_3d_deactivated")
    json_path = os.path.join(source_dir, 'case0.json')

    # output directory
    output_dir = os.path.join(test_dir,'eba2d_width80_h1000_bw4000_sw1000_yd300_3d_deactivated')
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


# # ---------------------------------------------------------------------
# # Test generating paraview scripts
# # ---------------------------------------------------------------------
# @pytest.mark.big_test  # Optional marker for big tests
# def test_generate_paraview_script_box_2d():

#     source_dir = "/mnt/lochy/ASPECT_DATA/MOW/mow_tests/eba2d_width80_h1000_bw4000_sw1000_yd300"

#     CaseOptions = CASE_OPTIONS(source_dir)


#     # prepare the graphical_steps
#     graphical_steps = [1]; slices=None # 1. specify steps

#     # types of plot to include
#     plot_types = ["upper_mantle"]; rotation_plus = 0.47 # for plotting the upper mantle

#     additional_fields = [] # in case of one crustal layer

#     config = {
#         "RESULT_DIR": test_dir,                   # directory to write output .txt
#         "py_temp_file": os.path.join(test_dir, "py_temp"),          # where to write pvpython script
#         "PlotCaseRun_base": None,                               # your PlotCase module
#         "PlotCaseRun_project": PlotCaseRunTwoD,                       # your TwoDPlotCase module

#         # ---
#         # Visualization and plotting options
#         # True: save a complete result
#         # False: prepare for figures in a paper
#         # ---
#         "plot_axis": False,
#         "graphical_steps": graphical_steps,
#         "slices": None,
#         "max_velocity": -1.0,
#         "plot_types": plot_types,
#         "rotation_plus": rotation_plus,
#         "additional_fields": [],
#         "CaseOptions": CaseOptions
#         
#     }

#     Visit_Options = run_2d_subduction_visualization(source_dir, config)
    
#     paraview_script = os.path.join(source_dir, "paraview_scripts", "slab.py")
#     paraview_script_std = os.path.join(fixture_root, "eba2d_width80_h1000_bw4000_sw1000_yd300")
#     filecmp.cmp(paraview_script, paraview_script_std)