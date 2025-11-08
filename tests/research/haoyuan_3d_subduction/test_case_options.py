import os
import pytest
import filecmp  # for compare file contents
import json
import numpy as np
from unittest import mock
from shutil import rmtree  # for remove directories

from hamageolib.research.haoyuan_3d_subduction.case_options import CASE_OPTIONS
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import CASE_THD, CASE_OPT_THD, create_case_with_json, CreateGroup


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_2d_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_2d_subduction")
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_3d_subduction")
SCRIPT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../..", "scripts")

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


def test_3d_mantle_jump():
    '''
    test for setting the 3d case in the box geometry
    test for a mantle viscosity discontinuity around 1100 km
    '''
    source_dir = os.path.join(fixture_root, "test_3d_mantle_jump")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_3d_mantle_jump')
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


def test_3d_visualization_basics():
    '''
    test basic utilities for 3d visualization, e.g. generate paraview script
    '''
    source_dir = os.path.join(fixture_root, "test_eba3d_width80_h1000_bw4000_sw1000_yd300")
    options = {"time_range": None, "run_visual": False, "time_interval":None, "visualization": "paraview",
               "step": 10, "plot_axis": False, "max_velocity": -1.0, "rotation_plus": 5.0}
    ofile_std = os.path.join(source_dir, "slab_std.py")

    # output directory
    output_dir = os.path.join(test_dir,'test_3d_visualization_basics')
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    os.mkdir(output_dir)

    Case_Options = CASE_OPTIONS(source_dir)
    # call function
    Case_Options.Interpret(**options)
    ofile_list = ['slab.py']
    for ofile_base in ofile_list:
        ofile = os.path.join(output_dir, ofile_base)
        paraview_script = os.path.join(SCRIPT_DIR, 'paraview_scripts',"ThDSubduction", ofile_base)
        paraview_base_script = os.path.join(SCRIPT_DIR, 'paraview_scripts', 'base.py')  # base.py : base file
        Case_Options.read_contents(paraview_base_script, paraview_script)  # this part combines two scripts
        Case_Options.substitute()  # substitute keys in these combined file with values determined by Interpret() function
        ofile_path = Case_Options.save(ofile, relative=False)  # save the altered script
        print("\t File generated: %s" % ofile_path)

    # generate summary file
    ofile_summary_csv = os.path.join(output_dir, "summary.csv")
    Case_Options.SummaryCaseVtuStepExport(ofile_summary_csv)
    assert(os.path.isfile(ofile_summary_csv))


def test_eba3d_width80_h1000_bw4000_sw1000_yd300():
    '''
    test for setting the 3d case in the box geometry
    '''
    source_dir = os.path.join(fixture_root, "test_eba3d_width80_h1000_bw4000_sw1000_yd300")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_eba3d_width80_h1000_bw4000_sw1000_yd300')
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

def test_sp_migrate_left_bd():
    '''
    test for setting the 3d case in the box geometry, making the trench closer to left side boundary
    Modified from test test_eba3d_width80_h1000_bw4000_sw1000_yd300
    '''
    source_dir = os.path.join(fixture_root, "test_sp_migrate_left_bd")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_sp_migrate_left_bd')
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

def test_chunk_sp_migrate_left_bd():
    '''
    test for setting the 3d case in the chunk geometry, making the trench closer to left side boundary
    modified from test_chunk0
    '''
    source_dir = os.path.join(fixture_root, "test_chunk_sp_migrate_left_bd")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_chunk_sp_migrate_left_bd')
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


def test_group0():
    '''
    test for generating a group of cases.
    '''
    source_dir = os.path.join(fixture_root, "test_group0")
    group_json_path = os.path.join(source_dir, "group.json")
    assert(os.path.isfile(group_json_path)) 
    
    output_dir = os.path.join(test_dir,'group0')
    if os.path.isdir(output_dir):
        rmtree(output_dir)

    # create the group
    CreateGroup(group_json_path, CASE_THD, CASE_OPT_THD)

    # check group directory
    assert(os.path.isdir(output_dir))
    assert(os.path.isfile(os.path.join(output_dir, "group.json")))

    # check case directory 1
    case1_dir = os.path.join(output_dir, "eba3d_1_SA50.0_OA20.0_width51_bw4000_sw1000_yd1000000.0_AR4")
    assert(os.path.isdir(case1_dir))
    case1_prm = os.path.join(case1_dir, "case.prm")
    case1_wb = os.path.join(case1_dir, "case.wb")
    case1_prm_std = os.path.join(source_dir, "case1_std.prm")
    case1_wb_std = os.path.join(source_dir, "case1_std.wb")
    assert(os.path.isfile(case1_prm))
    assert(os.path.isfile(case1_wb))
    assert(filecmp.cmp(case1_prm, case1_prm_std))
    
    # check case directory 2
    case2_dir = os.path.join(output_dir, "eba3d_1_SA50.0_OA20.0_width80_bw4000_sw1000_yd500.0_AR4")
    assert(os.path.isdir(case2_dir))
    case2_prm = os.path.join(case2_dir, "case.prm")
    case2_prm_std = os.path.join(source_dir, "case2_std.prm")
    assert(os.path.isfile(case2_prm))
    case2_wb = os.path.join(case2_dir, "case.wb")
    case2_wb_std = os.path.join(source_dir, "case2_std.wb")
    assert(filecmp.cmp(case2_prm, case2_prm_std))
    assert(filecmp.cmp(case2_wb, case2_wb_std))

