import math

import numpy as np

from sbrt.state.multirep import MultiRepBlock, _RollingBridge


def _direct(x, w):
    """Cálculo O(w) explícito da mesma estatística, para provar a recursão O(1) de `_RollingBridge`.
    Esta é a única forma honesta de defender a expansão algébrica da docstring daquele bloco."""
    win = np.asarray(x[-w:], dtype=np.float64)
    n = len(win)
    P = np.cumsum(win)
    i = np.arange(1, n + 1)
    bridge = P - (i / n) * P[-1]
    var = win.var()
    return float((bridge ** 2).sum() / (n * n * var))


def test_recursion_matches_direct_computation():
    rng = np.random.RandomState(0)
    x = rng.randn(500)
    br = _RollingBridge(50, 20)
    for i, v in enumerate(x, start=1):
        br.update(float(v))
        if i >= 20:
            assert math.isclose(br.value(), _direct(x[:i], 50), rel_tol=1e-8), i


def test_recursion_survives_a_large_drifting_cumulative_sum():
    """Q é um resíduo pequeno de somas grandes de C². Sob quebra de média C cresce linearmente —
    o pior caso numérico. Aqui a soma acumulada chega a ~1000 e a recursão ainda bate com o cálculo
    direto dentro de 1e-6 relativo (float64 sobra)."""
    x = np.full(2000, 0.5) + np.random.RandomState(1).randn(2000) * 0.1
    br = _RollingBridge(100, 20)
    for i, v in enumerate(x, start=1):
        br.update(float(v))
    assert math.isclose(br.value(), _direct(x, 100), rel_tol=1e-6)


def test_invariant_to_shift_and_scale():
    """A ponte zera nas pontas => somar constante não muda nada (por isso `e²` e `e²−1` dão o mesmo
    valor, e nenhuma centragem é necessária); dividir por s²_n => multiplicar por constante também
    não. Auto-normalização por CONSTRUÇÃO, não constante estimada por série — a diferença que a
    autópsia do F1 tornou central."""
    x = np.random.RandomState(2).randn(300)
    def run(y):
        br = _RollingBridge(100, 20)
        for v in y:
            br.update(float(v))
        return br.value()
    base = run(x)
    assert math.isclose(run(x + 13.7), base, rel_tol=1e-8)
    assert math.isclose(run(-4.2 * x), base, rel_tol=1e-8)


def test_h0_level_is_the_cramer_von_mises_mean():
    """Sob H0 a estatística converge para ∫₀¹B°(r)²dr, cuja média é 1/6 — o MESMO valor para
    qualquer série, seja qual for sua variância ou cauda.

    Precisa de muitas réplicas: a distribuição é bem assimétrica (mediana ~0,12 contra média 0,167,
    dp ~0,15), então 40 réplicas dão erro-padrão 0,024 e o teste fica instável. Com 400, o EP é
    0,0073 e a tolerância de 0,025 é ~3,4 EP."""
    rng = np.random.RandomState(3)
    vals = []
    for _ in range(400):
        br = _RollingBridge(50, 20)
        for v in rng.randn(120):
            br.update(float(v))
        vals.append(br.value())
    assert abs(float(np.mean(vals)) - 1 / 6) < 0.025


def test_mean_shift_inside_the_window_raises_the_statistic():
    rng = np.random.RandomState(4)
    quiet = rng.randn(200)
    broken = np.concatenate([rng.randn(150), rng.randn(50) + 1.5])
    def run(y):
        br = _RollingBridge(100, 20)
        for v in y:
            br.update(float(v))
        return br.value()
    assert run(broken) > 3.0 * run(quiet)


def test_block_emits_nan_before_min_n_and_finite_after(cfg):
    """Precisa de um `h0` real: a terceira representação é o PIT contra `sorted_e_hist`. Sem ele o
    bloco não tem contra o que rankear (e a coluna vira constante 0,5, logo NaN por variância nula)
    — o que é o comportamento honesto, não um bug."""
    from sbrt.state.h0 import fit_h0

    rng = np.random.RandomState(5)
    h0 = fit_h0(rng.randn(2000), cfg)
    x = rng.randn(150)
    blk = MultiRepBlock()
    blk.reset(h0, cfg)
    for t, v in enumerate(x[:10], start=1):
        blk.update(float(v), float(v), float(v), t)
    assert all(math.isnan(v) for v in blk.features().values())
    for t, v in enumerate(x[10:], start=11):
        blk.update(float(v), float(v), float(v), t)
    f = blk.features()
    assert all(math.isfinite(v) for v in f.values())
    assert len(f) == 6
