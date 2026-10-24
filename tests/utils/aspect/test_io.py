"""
Tests for aspect.io module
"""
import os
import filecmp
import pytest
from shutil import copy
from pathlib import Path
from io import StringIO

from hamageolib.utils.aspect.io import *

def test_parse_parameters_to_dict(tmp_path):
    """
    Tests the parse_parameters_to_dict function to ensure it correctly parses a
    deal.ii parameter file into a dictionary format.

    Args:
        tmp_path (Path): A temporary path provided by pytest to store any test files
                         if needed for the function.
    """
    # Define the path to the fixture file containing sample parameters
    test_dir = Path(__file__).parent
    prm_path = test_dir.parent / "data" / "aspect" / "test_parse_parameters_to_dict.prm"

    # Open the fixture file, parse the parameters, and store them in a dictionary
    with open(prm_path, 'r') as fin:
        params_dict = parse_parameters_to_dict(fin)

    # Assert that parsed parameters match expected values to verify accurate parsing
    assert(params_dict['Dimension'] == '2')
    assert(params_dict['Solver parameters'] == {'Stokes solver parameters': {'Linear solver tolerance': '1e-12'}})
    assert(params_dict['Geometry model']['Spherical shell'] ==  {'Inner radius': '1', 'Outer radius': '2', 'Opening angle': '360'})

def test_repeated_subsection_merging():
    content = """
subsection A
  set x = 1
end

subsection A
  set y = 2
end
"""
    f = StringIO(content)
    result = parse_parameters_to_dict(f)

    assert "A" in result
    assert result["A"]["x"] == "1"
    assert result["A"]["y"] == "2"


def test_save_parameters_from_dict(tmp_path):
    """
    Test that save_parameters_from_dict correctly saves a dictionary of parameters
    to a file in the expected format.

    Args:
        tmp_path (Path): A temporary directory path provided by pytest for storing test output files.

    Asserts:
        Verifies that the saved file matches the expected output file using file comparison.
    """
    # Dictionary containing parameters to be saved to the file
    parameters_dict = {
        'Dimension': '2',
        'Solver parameters': {
            'Stokes solver parameters': {
                'Linear solver tolerance': '1e-12'
            }
        },
        'Geometry model': {
            'Spherical shell': {
                'Inner radius': '1',
                'Outer radius': '2',
                'Opening angle': '360'
            }
        }
    }

    # Define the path for the output file within the temporary directory
    output_file_path = os.path.join(tmp_path, "test_output.prm")

    # Save the parameters to the output file
    with open(output_file_path, 'w') as output_file:
        save_parameters_from_dict(output_file, parameters_dict)

    # Define the path to the expected output file for comparison
    test_dir = Path(__file__).parent
    expected_prm_path = test_dir.parent / "data" / "aspect" / "test_save_parameters_from_dict_expected.prm"

    # Assert that the generated output file matches the expected file
    assert(filecmp.cmp(output_file_path, expected_prm_path))

def test_save_parameters_invalid_value_type_raises():
    bad_dict = {
        "valid": "1",
        "invalid": 123  # neither str nor dict
    }

    fout = StringIO()

    with pytest.raises(ValueError):
        save_parameters_from_dict(fout, bad_dict)
