import subprocess
import os
import filecmp

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def test_parse_block_newton_v2_6_0():
    awk_script_path = os.path.join(base_dir, "scripts", "awk", "parse_block_newton_v2_6_0.awk")
    input_file = os.path.join(base_dir, "tests/fixtures/scripts/awk/mock_aspect_log_v2_6_0.txt")
    expected_output = os.path.join(base_dir, "tests/fixtures/scripts/awk/expected_newton_output_v2_6_0.txt")
    actual_dir = os.path.join(base_dir, "tmp/awk_tests")
    actual_output = os.path.join(base_dir, "tmp/awk_tests/test_newton_output_v2_6_0.txt")

    # assert script exists
    assert(os.path.isfile(awk_script_path))

    # handle output path
    if not os.path.isdir(actual_dir):
        # Clean up the test output
        os.mkdir(actual_dir)
    if os.path.isfile(actual_output):
        # make the output directory
        os.remove(actual_output)
    
    # Run the awk script
    subprocess.run(
        ["awk", "-f", awk_script_path, input_file],
        stdout=open(actual_output, "w"),
        check=True
    )
    
    # Read the actual and expected outputs
    assert(filecmp.cmp(actual_output, expected_output))


def test_parse_block_newton_v3_1_0():
    # todo_3_1
    awk_script_path = os.path.join(base_dir, "scripts", "awk", "parse_block_newton_v3_1_0.awk")
    input_file = os.path.join(base_dir, "tests/fixtures/scripts/awk/mock_aspect_log_v3_1_0.txt")
    expected_output = os.path.join(base_dir, "tests/fixtures/scripts/awk/expected_newton_output_v3_1_0.txt")
    actual_dir = os.path.join(base_dir, "tmp/awk_tests")
    actual_output = os.path.join(base_dir, "tmp/awk_tests/test_newton_output_v3_1_0.txt")

    # assert script exists
    assert(os.path.isfile(awk_script_path))

    # handle output path
    if not os.path.isdir(actual_dir):
        # Clean up the test output
        os.mkdir(actual_dir)
    if os.path.isfile(actual_output):
        # make the output directory
        os.remove(actual_output)
    
    # Run the awk script
    subprocess.run(
        ["awk", "-f", awk_script_path, input_file],
        stdout=open(actual_output, "w"),
        check=True
    )
    
    # Read the actual and expected outputs
    assert(filecmp.cmp(actual_output, expected_output))


def test_parse_snapshots():
    # in this case, there is no snapshots saved
    awk_script_path = os.path.join(base_dir, "scripts", "awk", "parse_snapshots.awk")
    input_file = os.path.join(base_dir, "tests/fixtures/scripts/awk/mock_aspect_log.txt")
    expected_output = os.path.join(base_dir, "tests/fixtures/scripts/awk/expected_snapshots_output.txt")
    actual_dir = os.path.join(base_dir, "tmp/awk_tests")
    actual_output = os.path.join(base_dir, "tmp/awk_tests/test_snapshots.txt")

    # assert script exists
    assert(os.path.isfile(awk_script_path))

    # handle output path
    if not os.path.isdir(actual_dir):
        # Clean up the test output
        os.mkdir(actual_dir)
    if os.path.isfile(actual_output):
        # make the output directory
        os.remove(actual_output)
    
    # Run the awk script
    subprocess.run(
        ["awk", "-f", awk_script_path, input_file],
        stdout=open(actual_output, "w"),
        check=True
    )
    
    # Read the actual and expected outputs
    assert(filecmp.cmp(actual_output, expected_output))

def test_parse_snapshots():
    # in this case, there are two snapshots saved
    awk_script_path = os.path.join(base_dir, "scripts", "awk", "parse_snapshots.awk")
    input_file = os.path.join(base_dir, "tests/fixtures/scripts/awk/mock_aspect_log_1.txt")
    expected_output = os.path.join(base_dir, "tests/fixtures/scripts/awk/expected_snapshots_output_1.txt")
    actual_dir = os.path.join(base_dir, "tmp/awk_tests")
    actual_output = os.path.join(base_dir, "tmp/awk_tests/test_snapshots_1.txt")

    # assert script exists
    assert(os.path.isfile(awk_script_path))

    # handle output path
    if not os.path.isdir(actual_dir):
        # Clean up the test output
        os.mkdir(actual_dir)
    if os.path.isfile(actual_output):
        # make the output directory
        os.remove(actual_output)
    
    # Run the awk script
    subprocess.run(
        ["awk", "-f", awk_script_path, input_file],
        stdout=open(actual_output, "w"),
        check=True
    )
    
    # Read the actual and expected outputs
    assert(filecmp.cmp(actual_output, expected_output))


def test_parse_time_info():
    # in this case, there are two snapshots saved
    awk_script_path = os.path.join(base_dir, "scripts", "awk", "parse_block_output.awk")
    input_file = os.path.join(base_dir, "tests/fixtures/scripts/awk/mock_aspect_log_1.txt")
    expected_output = os.path.join(base_dir, "tests/fixtures/scripts/awk/expected_time_info.txt")
    actual_dir = os.path.join(base_dir, "tmp/awk_tests")
    actual_output = os.path.join(base_dir, "tmp/awk_tests/test_time_info.txt")

    # assert script exists
    assert(os.path.isfile(awk_script_path))

    # handle output path
    if not os.path.isdir(actual_dir):
        # Clean up the test output
        os.mkdir(actual_dir)
    if os.path.isfile(actual_output):
        # make the output directory
        os.remove(actual_output)
    
    # Run the awk script
    subprocess.run(
        ["awk", "-f", awk_script_path, input_file],
        stdout=open(actual_output, "w"),
        check=True
    )
    
    # Read the actual and expected outputs
    assert(filecmp.cmp(actual_output, expected_output))