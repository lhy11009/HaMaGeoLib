import numpy as np
import pytest

# >>> EDIT THIS IMPORT PATH <<<
from hamageolib.core.GrainSize import GrainGrowthModel, GrainGrowthParams

# Test for class GrainGrowthModel
# -------- Helpers --------
K = 1e-6
M = 2.0
E = 2.5e5       # J/mol
V = 1.0e-6      # m^3/mol

def make_model_explicit():
    return GrainGrowthModel(K, M, E, V)

def make_model_params():
    params = GrainGrowthParams(
        grain_growth_rate_constant=K,
        m=M,
        grain_growth_activation_energy=E,
        grain_growth_activation_volume=V,
    )
    return GrainGrowthModel(params=params)

def expected_rate(model, g, P, T):
    g = np.asarray(g, dtype=float)
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    denom = model.m * np.power(g, model.m - 1.0)
    return (model.k / denom) * np.exp(-(model.E + P * model.V) / (model.R * T))


# -------- Init success equivalence --------
def test_init_equivalence_params_vs_explicit():
    m1 = make_model_explicit()
    m2 = make_model_params()
    g = np.array([5e-4, 1e-3, 2e-3])
    P = 1e9
    T = 1600.0
    r1 = m1.calculate_growth_rate(g, P, T)
    r2 = m2.calculate_growth_rate(g, P, T)
    assert np.allclose(r1, r2)


# -------- Value / broadcasting tests --------
SUCCESS_CASES = [
    dict(name="scalar_inputs", g=1e-3, P=1e9, T=1600.0),
    dict(name="vector_g",      g=np.array([5e-4, 1e-3, 2e-3]), P=1e9, T=1600.0),
    dict(name="broadcast_P",   g=1e-3, P=np.array([0.0, 1e9]), T=1600.0),
    dict(name="broadcast_T",   g=1e-3, P=1e9, T=np.array([1200.0, 1600.0])),
    dict(name="all_vectors_broadcast",
         g=np.array([1e-3, 2e-3])[:, None],
         P=np.array([5e8, 1e9])[None, :],
         T=1600.0),
]

@pytest.mark.parametrize("case", SUCCESS_CASES, ids=[c["name"] for c in SUCCESS_CASES])
def test_calculate_growth_rate_values_and_shape(case):
    model = make_model_params()
    g, P, T = case["g"], case["P"], case["T"]
    got = model.calculate_growth_rate(g, P, T)
    exp = expected_rate(model, g, P, T)
    assert np.allclose(got, exp)
    assert np.asarray(got).dtype == float
    # Shape must match numpy broadcasting rules
    assert np.shape(got) == np.shape(exp)


# -------- Error cases for calculation --------
ERROR_CASES = [
    dict(name="nonpositive_grain", kwargs=dict(grain_size=0.0,  P=1e9, T=1600.0), exc=ValueError),
    dict(name="nonpositive_T",     kwargs=dict(grain_size=1e-3, P=1e9, T=0.0),    exc=ValueError),
]

@pytest.mark.parametrize("case", ERROR_CASES, ids=[c["name"] for c in ERROR_CASES])
def test_calculate_growth_rate_errors(case):
    model = make_model_params()
    with pytest.raises(case["exc"]):
        model.calculate_growth_rate(**case["kwargs"])

def test_zero_division_when_m_is_zero():
    # m = 0 makes denom = m * g^(m-1) = 0 * g^(-1) -> zero
    bad = GrainGrowthModel(K, 0.0, E, V)
    with pytest.raises(ZeroDivisionError):
        bad.calculate_growth_rate(1e-3, 1e9, 1600.0)


# -------- Init error paths --------
def test_init_error_both_styles_given():
    params = GrainGrowthParams(K, M, E, V)
    with pytest.raises(ValueError):
        GrainGrowthModel(K, M, E, V, params=params)

def test_init_error_partial_explicit_args():
    # Missing some explicit args and no params → error
    with pytest.raises(ValueError):
        GrainGrowthModel(K, M)  # incomplete