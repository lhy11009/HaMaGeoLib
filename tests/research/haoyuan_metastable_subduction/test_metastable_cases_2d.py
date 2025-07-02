from shutil import rmtree
import filecmp  # for compare file contents
import os
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import CASE_TWOD, CASE_OPT_TWOD, create_case_with_json

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_metastable_subduction")
SCRIPT_DIR = os.path.join(package_root, "scripts")

test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_metastable_subduction"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)

def test_haoyuan_metastable_subduction():
    '''
    test for including metastable_subduction
    cartesian, 3-d consistent geometry
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
