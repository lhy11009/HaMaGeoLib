# test_case_options_haoyuan.py

from hamageolib.utils.handy_shortcuts_haoyuan import check_float
from hamageolib.utils.case_options_haoyuan import *

def test_case_options(tmp_path):
    """
    Tests the functionality of the CASE_OPTIONS class by verifying that the 
    options dictionary is populated correctly for a specific test case.

    Parameters:
        tmp_path (Path): Temporary path provided by the testing framework for file handling during the test.
    """
    # Set up test case directory path
    case_dir = os.path.join(os.path.dirname(__file__), "../integration/fixtures/aspect_case_examples/case_2d_subduction")
    
    # Initialize CASE_OPTIONS instance and interpret options
    Case_Options = CASE_OPTIONS(case_dir)
    Case_Options.Interpret()

    # Display the options dictionary for debugging purposes
    print("Case_Options.options:", Case_Options.options)

    # Assert that data output directory matches the expected path
    assert(Case_Options.options["DATA_OUTPUT_DIR"] == os.path.abspath(os.path.join(case_dir, "output")))
    
    # Assert that geometry model is set to 'chunk'
    assert(Case_Options.options['GEOMETRY'] == 'chunk')
    
    # Assert that outer radius matches the expected value with floating-point precision check
    assert(check_float(Case_Options.options['OUTER_RADIUS'], 6371000.0))
    
    # Assert that inner radius matches the expected value with floating-point precision check
    assert(check_float(Case_Options.options['INNER_RADIUS'], 3481000.0))
