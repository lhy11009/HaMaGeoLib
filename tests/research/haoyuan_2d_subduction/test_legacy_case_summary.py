import os
import filecmp  # for compare file contents
import pytest
from shutil import rmtree
from hamageolib.research.haoyuan_2d_subduction.legacy_tools import CASE_SUMMARY

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(test_root, "test_legacy_case_summary")
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)


@pytest.mark.big_test  # Optional marker for big tests
def test_case_summary_latex():
    '''
    test writing a latex table
    '''
    # initiate
    Case_Summary = CASE_SUMMARY()

    # import a group 
    group_dir = os.path.join(package_root, "big_tests", "TwoDSubduction", "test_documentation_group_in_dir", "EBA_2d_consistent_1")
    assert(os.path.isdir(group_dir))
    Case_Summary.import_directory(group_dir)

    # write outputs
    o_file = os.path.join(test_dir, "case_summary.tex")
    o_file_std =  os.path.join(group_dir, "ofile_std.tex")
    Case_Summary.write_file(o_file)
    assert(os.path.isfile(o_file))
    assert(os.path.isfile(o_file_std))
    assert(filecmp.cmp(o_file, o_file_std))


@pytest.mark.big_test  # Optional marker for big tests
def test_case_summary():
    '''
    Test the CASE_SUMMARY class
    '''
    # initiate
    Case_Summary = CASE_SUMMARY()

    # import a group 
    group_dir = os.path.join(package_root, "big_tests", "TwoDSubduction", "test_documentation_group_in_dir", "EBA_2d_consistent_1")
    assert(os.path.isdir(group_dir))
    Case_Summary.import_directory(group_dir)

    # write outputs
    o_file = os.path.join(test_dir, "case_summary.txt")
    o_file_std =  os.path.join(group_dir, "ofile_std")
    Case_Summary.write_file(o_file)
    assert(os.path.isfile(o_file))
    assert(os.path.isfile(o_file_std))
    assert(filecmp.cmp(o_file, o_file_std))

    # read file
    # file is generated in the last test
    Case_Summary1 = CASE_SUMMARY()
    Case_Summary1.import_txt(o_file)
    assert(Case_Summary1.cases == ["eba3d_SA80.0_OA40.0_width61_GR4_AR4", "eba3d_SA80.0_OA40.0_width61_GR3_AR3_sc_1e23"])
    assert(Case_Summary1.wallclocks == ['0.06138888888888889', '23.63888888888889'])