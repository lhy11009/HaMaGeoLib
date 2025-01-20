import os
import pytest
from hamageolib.utils.file_reader import parse_header_from_lines, read_aspect_header_file

# Fixture to create a temporary test file
@pytest.fixture
def test_file_path(tmp_path):
    """Fixture to create a temporary test file."""
    test_file = tmp_path / "test_log.txt"
    test_file_content = """\
# 1: Time step number
# 2: Time (years)
# 3: Time step size (years)
# 4: Number of mesh cells
# 5: Number of Stokes degrees of freedom
0 0.000000000000e+00 0.000000000000e+00 5120 47043
1 1.000000000000e+00 0.500000000000e+00 8480 78438
"""
    # Write content to the temporary file
    test_file.write_text(test_file_content)
    return test_file

def test_parse_header_from_lines(test_file_path):
    """Test the parse_header_from_lines function."""
    # Read the lines from the test file
    with open(test_file_path, 'r') as file:
        lines = file.readlines()

    column_names, units_map, data_start_index = parse_header_from_lines(lines)

    # Expected column names
    expected_column_names = [
        "Time step number",
        "Time",
        "Time step size",
        "Number of mesh cells",
        "Number of Stokes degrees of freedom"
    ]
    # Expected units map
    expected_units_map = {
        "Time step number": None,
        "Time": "years",
        "Time step size": "years",
        "Number of mesh cells": None,
        "Number of Stokes degrees of freedom": None,
    }
    # Expected start index for data
    expected_data_start_index = 5  # Header ends after line 5 (index 4)

    # Assertions
    assert column_names == expected_column_names
    assert units_map == expected_units_map
    assert data_start_index == expected_data_start_index

def test_read_aspect_header_file(test_file_path):
    """Test the read_aspect_header_file function."""
    df = read_aspect_header_file(test_file_path)

    # Check that the DataFrame contains the correct columns
    expected_columns = [
        "Time step number",
        "Time",
        "Time step size",
        "Number of mesh cells",
        "Number of Stokes degrees of freedom"
    ]
    assert list(df.columns) == expected_columns

    # Check the units map in the DataFrame's metadata
    expected_units_map = {
        "Time step number": None,
        "Time": "years",
        "Time step size": "years",
        "Number of mesh cells": None,
        "Number of Stokes degrees of freedom": None,
    }
    assert df.attrs["units"] == expected_units_map

    # Check that the DataFrame has the correct number of rows
    assert len(df) == 2

    # Check specific values
    # Row 0
    assert df.iloc[0]["Time step number"] == 0
    assert pytest.approx(df.iloc[0]["Time"]) == 0.0
    assert pytest.approx(df.iloc[0]["Time step size"]) == 0.0
    assert df.iloc[0]["Number of mesh cells"] == 5120
    assert df.iloc[0]["Number of Stokes degrees of freedom"] == 47043

    # Row 1 (unique values to check)
    assert df.iloc[1]["Time step number"] == 1
    assert pytest.approx(df.iloc[1]["Time"]) == 1.0
    assert pytest.approx(df.iloc[1]["Time step size"]) == 0.5
    assert df.iloc[1]["Number of mesh cells"] == 8480
    assert df.iloc[1]["Number of Stokes degrees of freedom"] == 78438