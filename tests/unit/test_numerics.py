import math

from sbrt.utils.numerics import ewma_update, lgamma_cached, logsumexp, welford_update


def test_welford_matches_hand_computation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    n, mean, m2 = 0, 0.0, 0.0
    for x in xs:
        n, mean, m2 = welford_update(n, mean, m2, x)
    assert n == len(xs)
    assert math.isclose(mean, sum(xs) / len(xs))
    variance = m2 / (n - 1)
    hand_var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    assert math.isclose(variance, hand_var, rel_tol=1e-9)


def test_logsumexp_matches_naive():
    values = [1.0, 2.0, 3.0, -1.0]
    naive = math.log(sum(math.exp(v) for v in values))
    assert math.isclose(logsumexp(values), naive, rel_tol=1e-12)


def test_logsumexp_ignores_neg_inf():
    assert math.isclose(logsumexp([1.0, -math.inf]), 1.0)


def test_lgamma_cached_matches_math_lgamma():
    for x in (0.5, 1.0, 2.5, 10.0, 25.5):
        assert math.isclose(lgamma_cached(x), math.lgamma(x))


def test_ewma_update_basic():
    v = ewma_update(0.0, 1.0, 0.5)
    assert math.isclose(v, 0.5)
    v2 = ewma_update(v, 1.0, 0.5)
    assert math.isclose(v2, 0.75)
