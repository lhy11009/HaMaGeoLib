import os
import gplately
import filecmp  # for compare file contents
import json
from shutil import rmtree
import numpy as np
import pandas as pd
import pytest
from plate_model_manager import PlateModelManager

from hamageolib.research.haoyuan_3d_subduction.gplately_utilities import \
      read_subduction_reconstruction_data, parse_subducting_trench_option, crop_region_by_data,\
      resample_subduction, compute_sum_of_arc_lengths, resample_positions

package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
fixture_root = os.path.join(package_root, "tests", "fixtures", "research", "haoyuan_3d_subduction")

# ---------------------------------------------------------------------
# Check and make test directories
# ---------------------------------------------------------------------
test_root = os.path.join(os.path.join(package_root, ".test"))
if not os.path.isdir(test_root):
    os.mkdir(test_root)

test_dir = os.path.join(os.path.join(test_root, "research-haoyuan_3d_subduction-gplately"))
if os.path.isdir(test_dir):
    rmtree(test_dir)
os.mkdir(test_dir)


# -----------------------------
# tests for crop_region_by_data
# -----------------------------
SUCCESS_CASES = [
    {
        "name": "basic positive longitudes",
        "s_data": pd.DataFrame({"lon": [10, 20, 30], "lat": [0, 5, 10]}),
        "interval": 10,
        "expected": [10, 30, 0, 10],
    },
    {
        "name": "negative longitudes wrap",
        "s_data": pd.DataFrame({"lon": [-170, -160, -150], "lat": [5, 10, 15]}),
        "interval": 10,
        "expected": [-170, -150, 0, 20],  # wrap to 190–210
    },
    {
        "name": "mixed lon range, smaller span kept",
        "s_data": pd.DataFrame({"lon": [-5, 5], "lat": [0, 1]}),
        "interval": 5,
        "expected": [-5, 5, 0, 5],  # region0 span (10) < region1 span (360)
    },
    {
        "name": "lat rounding up/down",
        "s_data": pd.DataFrame({"lon": [40, 50], "lat": [12.3, 17.7]}),
        "interval": 5,
        "expected": [40, 50, 10, 20],
    },
    {
        "name": "cross-180 longitudes wrap",
        "s_data": pd.DataFrame({"lon": [-170, -160, 170, 160], "lat": [5, 10, 15, 20]}),
        "interval": 10,
        "expected": [160, 200, 0, 20],  # wrap to 190–210
    },
]

@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_crop_region_by_data_success(case):
    result = crop_region_by_data(case["s_data"], case["interval"])
    assert result == case["expected"]

ERROR_CASES = [
    {
        "name": "empty dataframe",
        "s_data": pd.DataFrame({"lon": [], "lat": []}),
        "interval": 10,
        "error": ValueError,
    },
]


@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_crop_region_by_data_errors(case):
    with pytest.raises(case["error"]):
        crop_region_by_data(case["s_data"], case["interval"])

# ----------------------
# Tests for parse_subducting_trench_option
# ----------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "subducting_pid": [1, 2, 2, 3, 4, 4, 5],
            "trench_pid": [11, 12, 121, 31, 41, 141, 51],
            "lon": [10, 20, 21, 30, 40, 41, 50],
            "lat": [-5, 0, 1, 5, 10, 11, 15],
            "value": [0.1, 0.2, 0.21, 0.3, 0.4, 0.41, 0.5],
        }
    )


SUCCESS_CASES = [
    {"name": "single_int_match",   "options": 2,                 "expected_pids": {2}},
    {"name": "single_int_no_match","options": 999,               "expected_pids": set()},
    {"name": "list_multi",         "options": [2, 4],            "expected_pids": {2, 4}},
    {"name": "tuple_multi",        "options": (1, 3, 5),         "expected_pids": {1, 3, 5}},
    {"name": "numpy_array",        "options": np.array([4, 5]),  "expected_pids": {4, 5}},
    {"name": "empty_sequence",     "options": [],                "expected_pids": set()},
    {"name": "dict_single",        "options": {2: 121},          "expected_pids": {2}},
]

@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_parse_subducting_trench_option_success(sample_df, case):
    out = parse_subducting_trench_option(sample_df, case["options"])
    # expected pids
    assert set(out["subducting_pid"].unique()) == case["expected_pids"]
    # column schema preserved
    assert list(out.columns) == list(sample_df.columns)
    # emptiness check when expected set is empty
    assert out.empty == (len(case["expected_pids"]) == 0)


ERROR_CASES = [
    # wrong type for options
    {"name": "wrong_type_str", "df_op": "none", "options": "5", "exc": TypeError},
    # missing required column
    {"name": "missing_column", "df_op": "drop_pid", "options": 2, "exc": KeyError},
]

@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_parse_subducting_trench_option_errors(sample_df, case):
    df = sample_df.drop(columns=["subducting_pid"]) if case["df_op"] == "drop_pid" else sample_df
    with pytest.raises(case["exc"]):
        parse_subducting_trench_option(df, case["options"])


# ----------------------
# Tests for read_subduction_reconstruction_data
# ----------------------
def test_read_subduction_reconstruction_data():
    
    source_dir = os.path.join(fixture_root, "test_read_subduction_reconstruction_data")
    infile = os.path.join(source_dir, "reconstructed_0.00Ma.xy")
    odir = os.path.join(test_dir, "test_read_subduction_reconstruction_data")
    if os.path.isdir(odir):
        rmtree(odir)
    os.mkdir(odir)   

    outputs = read_subduction_reconstruction_data(infile)

    assert(outputs["n_trench"]==1)
    assert(outputs["trench_data"][0][0] == [-73.570086, -33.935762])
    assert(outputs["trench_data"][0][1] == [-73.33607, -33.199409])
    assert(outputs["trench_names"] == ["Andean Trench Segment", "North Andes Trench Segment"])
    assert(outputs["trench_pids"] ==  [201124, 201122])


# -------------------
# Tests for compute_sum_of_arc_lengths
# -------------------
SUCCESS_CASES = [
    dict(
        name="single_element",
        arr=np.array([4.0]),
        # sums: [4/2] = [2.0]
        exp_sums=np.array([2.0]),
        # total: 2.0 + 4/2 = 4.0
        exp_total=4.0,
    ),
    dict(
        name="two_elements",
        arr=np.array([2.0, 6.0]),
        # sums[0]=1.0; sums[1]=1.0 + (2+6)/2 = 5.0
        exp_sums=np.array([1.0, 5.0]),
        # total: 5.0 + 6/2 = 8.0
        exp_total=8.0,
    ),
    dict(
        name="three_elements",
        arr=np.array([2.0, 4.0, 8.0]),
        # s0=1.0; s1=1+(2+4)/2=4.0; s2=4+(4+8)/2=10.0
        exp_sums=np.array([1.0, 4.0, 10.0]),
        # total: 10.0 + 8/2 = 14.0
        exp_total=14.0,
    ),
    dict(
        name="integers_array",
        arr=np.array([1, 2, 3, 4]),
        # s0=0.5; s1=0.5+(1+2)/2=2.0; s2=2.0+(2+3)/2=4.5; s3=4.5+(3+4)/2=8.0
        exp_sums=np.array([0.5, 2.0, 4.5, 8.0]),
        # total: 8.0 + 4/2 = 10.0
        exp_total=10.0,
    ),
]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_compute_sum_of_arc_lengths_success(case):
    sums, total = compute_sum_of_arc_lengths(case["arr"])
    assert np.allclose(sums, case["exp_sums"], rtol=1e-9, atol=1e-12)
    assert np.isclose(total, case["exp_total"], rtol=1e-9, atol=1e-12)


ERROR_CASES = [
    dict(
        name="not_numpy_array",
        arr=[1, 2, 3],  # plain list
        exc=AssertionError,
    ),
    dict(
        name="wrong_ndim",
        arr=np.array([[1, 2], [3, 4]]),  # 2D
        exc=AssertionError,
    ),
]

@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_compute_sum_of_arc_lengths_errors(case):
    with pytest.raises(case["exc"]):
        compute_sum_of_arc_lengths(case["arr"])

# -------------------
# Tests for resample_positions
# -------------------

SUCCESS_CASES = [
    dict(
        name="exact_multiple_includes_edges_single_mid",
        total=100.0,
        edge=10.0,          # mid=50, d_max=40
        step=10.0,          # offsets: 10,20,30,40
        expected=np.array([10., 20., 30., 40., 50., 60., 70., 80., 90.]),
    ),
    dict(
        name="non_multiple_excludes_edges_single_mid",
        total=100.0,
        edge=10.0,          # mid=50, d_max=40
        step=15.0,          # offsets: 15,30
        expected=np.array([20., 35., 50., 65., 80.]),
    ),
    dict(
        name="tiny_range_less_than_step_only_mid",
        total=10.0,
        edge=4.0,           # mid=5, d_max=1 < step
        step=3.0,
        expected=np.array([5.]),
    ),
    dict(
        name="dmax_equals_step_includes_one_shell_single_mid",
        total=40.0,
        edge=10.0,          # mid=20, d_max=10 == step
        step=10.0,          # offsets: 10
        expected=np.array([10., 20., 30.]),
    ),
]

@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_resample_positions_success(case):
    pos = resample_positions(case["total"], case["edge"], case["step"])
    assert np.allclose(pos, case["expected"], rtol=1e-12, atol=1e-12)

    # Property checks
    mid = case["total"] / 2.0
    assert np.min(pos) >= case["edge"] - 1e-12
    assert np.max(pos) <= case["total"] - case["edge"] + 1e-12
    # Single midpoint only
    assert np.count_nonzero(np.isclose(pos, mid, rtol=0, atol=1e-12)) == 1
    # Symmetry around mid
    assert np.allclose(pos, 2 * mid - pos[::-1], rtol=1e-12, atol=1e-12)


ERROR_CASES = [
    dict(
        name="non_positive_step_raises",
        total=100.0,
        edge=10.0,
        step=0.0,
        exc=ValueError,
    ),
    dict(
        name="edge_greater_than_mid_raises",
        total=10.0,
        edge=6.0,   # mid=5 -> d_max < 0
        step=1.0,
        exc=ValueError,
    ),
]

@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_resample_positions_errors(case):
    with pytest.raises(case["exc"]):
        resample_positions(case["total"], case["edge"], case["step"])
