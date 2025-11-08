import os
import pytest
import filecmp  # for compare file contents
import numpy as np
from unittest import mock
from shutil import rmtree  # for remove directories

from hamageolib.research.haoyuan_2d_subduction.legacy_tools import create_case_with_json, CASE_TWOD, CASE_OPT_TWOD


package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_2d_subduction")

test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)
test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_2d_subduction-test_legacy_tools"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

# todo_3d
def test_2d_mantle_jump():
    '''
    test for setting the 3d case in the box geometry
    '''
    source_dir = os.path.join(fixture_root, "test_2d_mantle_jump")
    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_2d_mantle_jump')
    if os.path.isdir(output_dir):
        rmtree(output_dir)
    create_case_with_json(json_path, CASE_TWOD, CASE_OPT_TWOD)  # create case
    assert(os.path.isdir(output_dir))  # check case generation
    prm_std_path = os.path.join(source_dir, 'case_std.prm')
    prm_path = os.path.join(output_dir, 'case.prm')
    assert(filecmp.cmp(prm_path, prm_std_path))
    wb_std_path = os.path.join(source_dir, 'case_std.wb')
    wb_path = os.path.join(output_dir, 'case.wb')
    assert(filecmp.cmp(wb_path, wb_std_path))

def test_visc_diff():
    '''
    test change viscosity
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


def test_chunk_sp_migrate_left_bd_2d():
    '''
    test migrate the side boundary, modified from the case test_sz_same_composition
    '''
    source_dir = os.path.join(fixture_root, "test_chunk_sp_migrate_left_bd_2d")

    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_chunk_sp_migrate_left_bd_2d')
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


def test_box_sp_migrate_left_bd_2d():
    '''
    test migrate the side boundary, modified from the case test_sz_same_composition
    '''
    source_dir = os.path.join(fixture_root, "test_box_sp_migrate_left_bd_2d")

    json_path = os.path.join(source_dir, 'case0.json')
    output_dir = os.path.join(test_dir,'test_box_sp_migrate_left_bd_2d')
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