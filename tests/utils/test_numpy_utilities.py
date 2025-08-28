import numpy as np
import pytest
from hamageolib.utils.nump_utilities import interval_indices, interval_with_fraction

# -------------------
# Tests for interval_indices, interval_with_fraction
# -------------------

SUCCESS_CASES = [
    dict(
        name="scalar_interior",
        x=np.array([1.0, 2.0, 3.5, 7.0]),
        q=2.2,
        clamp=False,
        exp_i=np.array([1]),
        exp_j=np.array([2]),
        exp_t=np.array([(2.2 - 2.0) / (3.5 - 2.0)]),
    ),
    dict(
        name="scalar_exact_interior_node",
        x=np.array([1.0, 2.0, 3.5, 7.0]),
        q=2.0,  # exact node -> right interval [2.0, 3.5)
        clamp=False,
        exp_i=np.array([1]),           # was 0
        exp_j=np.array([2]),           # was 1
        exp_t=np.array([0.0]),         # start of right interval (was 1.0)
    ),
    dict(
        name="endpoints_with_clamp_true",
        x=np.array([0.0, 1.0, 3.0, 6.0]),
        q=np.array([0.0, 6.0]),        # exact endpoints
        clamp=True,
        exp_i=np.array([0, 2]),
        exp_j=np.array([1, 3]),
        exp_t=np.array([0.0, 1.0]),
    ),
    dict(
        name="vector_mixed_queries",
        x=np.array([0.0, 1.0, 3.0, 6.0]),
        q=np.array([0.2, 1.0, 2.9, 4.5]),
        clamp=False,
        exp_i=np.array([0, 1, 1, 2]),  # 1.0 exact -> i=1 (right interval)
        exp_j=np.array([1, 2, 2, 3]),
        exp_t=np.array([
            (0.2 - 0.0) / (1.0 - 0.0),   # 0.2
            0.0,                          # exact at start of [1,3)
            (2.9 - 1.0) / (3.0 - 1.0),   # 0.95
            (4.5 - 3.0) / (6.0 - 3.0),   # 0.5
        ]),
    ),
    dict(
        name="vector_unsorted_with_clamp",
        x=np.array([0.0, 1.0, 3.0, 6.0]),
        q=np.array([5.9, 0.1, 3.0, 2.0]),
        clamp=True,
        exp_i=np.array([2, 0, 2, 1]),  # 3.0 exact -> i=2 (right interval)
        exp_j=np.array([3, 1, 3, 2]),
        exp_t=np.array([
            (5.9 - 3.0) / (6.0 - 3.0),   # ≈0.9666667
            (0.1 - 0.0) / (1.0 - 0.0),   # 0.1
            0.0,                          # exact at start of [3,6)
            (2.0 - 1.0) / (3.0 - 1.0),   # 0.5
        ]),
    ),
]


@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_interval_helpers_success(case):
    # Normalize q to array for uniform assertions
    q = np.atleast_1d(case["q"])

    # interval_indices
    i, j = interval_indices(case["x"], case["q"], clamp=case["clamp"])
    i = np.atleast_1d(i)
    j = np.atleast_1d(j)
    assert np.array_equal(i, case["exp_i"])
    assert np.array_equal(j, case["exp_j"])

    # interval_with_fraction
    i2, j2, t = interval_with_fraction(case["x"], case["q"], clamp=case["clamp"])
    i2 = np.atleast_1d(i2); j2 = np.atleast_1d(j2); t = np.atleast_1d(t)
    assert np.array_equal(i2, case["exp_i"])
    assert np.array_equal(j2, case["exp_j"])
    assert np.allclose(t, case["exp_t"], rtol=1e-12, atol=1e-12)

    # Reconstruction property: q ≈ (1-t)*x[i] + t*x[j]
    xi = case["x"][i]
    xj = case["x"][j]
    q_recon = (1.0 - t) * xi + t * xj
    assert np.allclose(q_recon, q, rtol=1e-12, atol=1e-12)

ERROR_CASES = [
    dict(
        name="scalar_left_oob_raises",
        x=np.array([1.0, 2.0, 3.0]),
        q=0.99,
        clamp=False,
        fn="indices",
        exc=ValueError,
    ),
    dict(
        name="scalar_right_oob_raises",
        x=np.array([1.0, 2.0, 3.0]),
        q=3.01,
        clamp=False,
        fn="fraction",
        exc=ValueError,
    ),
    dict(
        name="vector_oob_raises",
        x=np.array([0.0, 1.0, 3.0]),
        q=np.array([-1.0, 0.5, 5.0]),  # has both sides OOB
        clamp=False,
        fn="indices",
        exc=ValueError,
    ),
]


@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_interval_helpers_errors(case):
    if case["fn"] == "indices":
        with pytest.raises(case["exc"]):
            interval_indices(case["x"], case["q"], clamp=case["clamp"])
    else:
        with pytest.raises(case["exc"]):
            interval_with_fraction(case["x"], case["q"], clamp=case["clamp"])