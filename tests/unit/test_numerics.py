import math

from sbrt.utils.numerics import (
    ewma_update,
    lgamma_cached,
    logsumexp,
    vol_adjust_step,
    welford_update,
)


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


def test_vol_adjust_step_is_bit_identical_to_the_inline_form_it_replaced():
    """`vol_adjust_step` foi extraída de `state/scorer.py:update_features` para ser compartilhada com
    o replay do histórico (F1.0). A extração só é legítima se for aritmética idêntica — este teste
    fixa a forma inline original como referência, bit a bit.

    Não usar `isclose`: a promessa da extração é igualdade exata, e a suíte de determinismo verifica
    re-execução, não equivalência com a versão anterior do código."""
    lam = 0.06  # cfg.state.vol_adjust.lambda_v
    v_ref = v_got = 1.0
    for e in (0.3, -1.7, 0.0, 12.5, -0.004, 8.0, -8.0, 1e-9):
        # forma inline original (state/scorer.py antes da extração)
        v_ref = ewma_update(v_ref, e * e, lam)
        evol_ref = e / math.sqrt(max(v_ref, 1e-12))

        v_got, evol_got = vol_adjust_step(v_got, e, lam)

        assert v_got == v_ref
        assert evol_got == evol_ref


def test_vol_adjust_step_floors_variance():
    """O piso de 1e-12 existe para uma sequência de zeros não gerar divisão por zero."""
    v, e_vol = vol_adjust_step(0.0, 0.0, 0.06)
    assert v == 0.0
    assert e_vol == 0.0
