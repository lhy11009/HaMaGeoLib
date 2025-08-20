import numpy as np
import pytest

# Adjust this import to your project layout
from hamageolib.utils.interp_utilities import KNNInterpolatorND

#### 
# tests for KNNInterpolatorND
####
def as_scalar(x):
    # Helper: convert 0-d numpy arrays (from scalar queries) to Python floats
    return np.asarray(x).item()


def test_KNNInterpolatorND_2d_exact_match_returns_sample_value():
    # Querying exactly at a data point returns that value.
    V0 = np.array([0.0, 1.0, 0.0])
    V1 = np.array([0.0, 0.0, 1.0])
    V2 = np.array([1.0, 2.0, 3.0])

    knn = KNNInterpolatorND([V0, V1], V2, k=3, weights="distance", p=2)
    v = as_scalar(knn(0.0, 0.0))
    assert v == pytest.approx(1.0)


def test_KNNInterpolatorND_2d_uniform_vs_distance_when_equidistant():
    # For equidistant neighbors, uniform and distance weighting give the same mean.
    V0 = np.array([0.0, 2.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([0.0, 10.0])
    xq, yq = 1.0, 0.0

    knn_uniform = KNNInterpolatorND([V0, V1], V2, k=2, weights="uniform")
    knn_idw = KNNInterpolatorND([V0, V1], V2, k=2, weights="distance", p=2)
    vu = as_scalar(knn_uniform(xq, yq))
    vd = as_scalar(knn_idw(xq, yq))
    assert vu == pytest.approx(5.0)
    assert vd == pytest.approx(5.0)


def test_KNNInterpolatorND_2d_distance_weighting_biases_toward_closer_point():
    # IDW with p=1 biases toward the closer neighbor.
    V0 = np.array([1.0, 2.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([10.0, 0.0])
    xq, yq = 0.0, 0.0  # distances: 1 and 2

    knn_idw = KNNInterpolatorND([V0, V1], V2, k=2, weights="distance", p=1)
    v = as_scalar(knn_idw(xq, yq))
    # Expected: (10/1 + 0/2) / (1 + 1/2) = 10 / 1.5
    assert v == pytest.approx(10.0 / 1.5)


def test_KNNInterpolatorND_2d_expected_idw_p2_two_points():
    # Check IDW p=2 formula matches manual computation.
    V0 = np.array([0.0, 3.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([4.0, 10.0])
    xq, yq = 1.0, 0.0  # d0=1, d1=2

    knn = KNNInterpolatorND([V0, V1], V2, k=2, weights="distance", p=2)
    v = as_scalar(knn(xq, yq))
    w0, w1 = 1.0 / (1.0**2), 1.0 / (2.0**2)
    expected = (w0 * 4.0 + w1 * 10.0) / (w0 + w1)
    assert v == pytest.approx(expected)


def test_KNNInterpolatorND_2d_nan_values_are_ignored_and_all_nan_raises():
    # NaN samples are dropped; all-NaN raises ValueError.
    V0 = np.array([0.0, 1.0, 2.0])
    V1 = np.array([0.0, 0.0, 0.0])
    V2 = np.array([1.0, np.nan, 9.0])

    knn = KNNInterpolatorND([V0, V1], V2, k=2, weights="uniform")
    v = as_scalar(knn(0.0, 0.0))
    assert v == pytest.approx(1.0)  # exact hit unaffected by NaN elsewhere

    with pytest.raises(ValueError):
        KNNInterpolatorND([V0, V1], np.array([np.nan, np.nan, np.nan]), k=2)


def test_KNNInterpolatorND_2d_k_greater_than_dataset_size_is_handled():
    # When k > n, the interpolator uses all available points without error.
    V0 = np.array([0.0, 1.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([0.0, 10.0])
    xq, yq = 0.25, 0.0

    knn = KNNInterpolatorND([V0, V1], V2, k=5, weights="distance", p=2)
    v = as_scalar(knn(xq, yq))
    d0, d1 = 0.25, 0.75
    w0, w1 = 1.0 / (d0**2), 1.0 / (d1**2)
    expected = (w0 * 0.0 + w1 * 10.0) / (w0 + w1)
    assert v == pytest.approx(expected)


def test_KNNInterpolatorND_2d_custom_weight_function():
    # Custom weight function should be honored.
    V0 = np.array([0.0, 1.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([0.0, 10.0])
    sigma = 0.5
    weight_fn = lambda d: np.exp(-((d / sigma) ** 2))

    knn = KNNInterpolatorND([V0, V1], V2, k=2, weights=weight_fn)
    v = as_scalar(knn(0.2, 0.0))

    d0, d1 = 0.2, 0.8
    w0, w1 = weight_fn(np.array([[d0, d1]])).ravel()
    expected = (w0 * 0.0 + w1 * 10.0) / (w0 + w1)
    assert v == pytest.approx(expected)


def test_KNNInterpolatorND_2d_output_shape_matches_input_shape_for_grid_and_vector():
    # Output shape follows X/Y query shape (grid and vector).
    V0g, V1g = np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3), indexing="xy")
    V2 = (V0g + V1g).ravel()  # simple plane

    knn = KNNInterpolatorND([V0g.ravel(), V1g.ravel()], V2, k=3, weights="uniform")

    # Grid input
    Xg, Yg = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 4), indexing="xy")
    Zg = knn(Xg, Yg)
    assert Zg.shape == Xg.shape

    # Vector input
    xv = np.linspace(0, 1, 7)
    yv = np.linspace(0, 1, 7)
    Zvec = knn(xv, yv)
    assert Zvec.shape == xv.shape


def test_KNNInterpolatorND_2d_k_equals_one_equivalent_to_nearest():
    # k=1 returns the value of the nearest sample.
    V0 = np.array([0.0, 2.0, 5.0])
    V1 = np.array([0.0, 0.0, 0.0])
    V2 = np.array([1.0, 3.0, 9.0])
    xq = 1.2  # nearest is at x=2.0 (dist 0.8) vs x=0.0 (dist 1.2)

    knn = KNNInterpolatorND([V0, V1], V2, k=1, weights="uniform")
    v = as_scalar(knn(xq, 0.0))
    assert v == pytest.approx(3.0)


def test_KNNInterpolatorND_2d_higher_p_biases_more_toward_closest_neighbor():
    # Increasing p moves result toward the closest neighbor's value.
    V0 = np.array([0.0, 2.0])
    V1 = np.array([0.0, 0.0])
    V2 = np.array([0.0, 10.0])
    xq, yq = 0.5, 0.0  # closer to the 0.0-valued point

    knn_p1 = KNNInterpolatorND([V0, V1], V2, k=2, weights="distance", p=1)
    knn_p4 = KNNInterpolatorND([V0, V1], V2, k=2, weights="distance", p=4)
    v1 = as_scalar(knn_p1(xq, yq))
    v4 = as_scalar(knn_p4(xq, yq))
    # v4 should be smaller (closer to 0) than v1
    assert v4 <= v1 + 1e-12


def test_KNNInterpolatorND_3d_exact_match_and_shape():
    # 8 corners of the unit cube; value = x + y + z
    X = np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1]
    ], dtype=float)
    Z = X.sum(axis=1)

    knn = KNNInterpolatorND(X, Z, k=4, weights="uniform")

    # Exact match at [1,1,1] -> value 3.0
    v = as_scalar(knn(1.0, 1.0, 1.0))
    assert v == pytest.approx(3.0)

    # Shape check on a 3D grid query
    Xg, Yg, Zg = np.meshgrid(
        np.linspace(0, 1, 3),
        np.linspace(0, 1, 4),
        np.linspace(0, 1, 2),
        indexing="xy"
    )
    Vq = knn(Xg, Yg, Zg)
    assert Vq.shape == Xg.shape


def test_KNNInterpolatorND_3d_scale_affects_nearest_selection_k1():
    # Two 3D points equidistant to the query in unscaled Euclidean metric.
    # A at (0,1,0) with value 0; B at (1,0,0) with value 10.
    # Query at (0.6, 0.6, 0.0) is equidistant from A and B.
    A = np.array([0.0, 1.0, 0.0])
    B = np.array([1.0, 0.0, 0.0])
    X = np.vstack([A, B])
    Z = np.array([0.0, 10.0])

    # With anisotropic scaling (divide-by-scale in implementation):
    # scale=[0.5, 2.0, 1.0] -> x differences emphasized, y de-emphasized.
    # Under this scaling, B becomes closer than A -> expect value 10 with k=1.
    knn_scaled = KNNInterpolatorND(X, Z, k=1, weights="uniform", scale=[0.5, 2.0, 1.0])
    v = as_scalar(knn_scaled(0.6, 0.6, 0.0))
    assert v == pytest.approx(10.0)


def test_KNNInterpolatorND_3d_query_matrix_last_dim_equivalence():
    # Random but reproducible small 3D dataset
    rng = np.random.default_rng(0)
    X = rng.random((10, 3))
    Z = rng.normal(size=10)

    knn = KNNInterpolatorND(X, Z, k=3, weights="distance", p=2)

    # Query as a matrix with last dim = d
    Q = rng.random((5, 4, 3))
    V_mat = knn(Q)

    # Same queries passed as separate arrays
    Xq, Yq, Zq = Q[..., 0], Q[..., 1], Q[..., 2]
    V_sep = knn(Xq, Yq, Zq)

    assert V_mat.shape == (5, 4)
    assert V_sep.shape == (5, 4)
    assert np.allclose(V_mat, V_sep, equal_nan=True)


def test_KNNInterpolatorND_4d_accepts_coords_and_exact_match():
    # Simple 4D exact-match check
    X = np.array([
        [0,0,0,0],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,1,1,1],
    ], dtype=float)
    Z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 10.0])

    knn = KNNInterpolatorND(X, Z, k=3, weights="uniform")

    # Exact hit at [1,1,1,1] -> 10.0
    v = as_scalar(knn(np.array([1.0, 1.0, 1.0, 1.0])))
    assert v == pytest.approx(10.0)

    # Also check that passing a (m,d) query works and returns (m,)
    Q = np.array([[1,0,0,0], [0,1,0,0], [0,0,0,1]], dtype=float)
    V = knn(Q)
    assert V.shape == (3,)
    assert np.allclose(V, [1.0, 2.0, 4.0])