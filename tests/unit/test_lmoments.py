"""P2 (docs/INVESTIGACAO_FALHAS_V3.md): forma de cauda dinâmica via L-momentos."""
from __future__ import annotations

import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.lmoments import LMomentBlock, _l_ratios


def _run(block, series):
    out = []
    for t, x in enumerate(series, start=1):
        block.update(0.0, float(x), 0.0, t)
        out.append(block.features())
    return out


def _new(cfg):
    b = LMomentBlock()
    b.reset(None, cfg)
    return b


def test_l_kurtosis_of_normal_matches_theory(cfg):
    """L-kurtosis teórica de uma normal = 0,1226. Valida o estimador PWM."""
    rng = np.random.RandomState(0)
    tau3, tau4 = _l_ratios(sorted(rng.randn(5000)))
    assert abs(tau4 - 0.1226) < 0.02
    assert abs(tau3) < 0.03  # normal é simétrica -> L-skew ≈ 0


def test_l_skewness_sign(cfg):
    rng = np.random.RandomState(1)
    # exponencial é assimétrica à direita -> L-skew > 0
    tau3, _ = _l_ratios(sorted(rng.exponential(1.0, 3000)))
    assert tau3 > 0.2


def test_heavy_tail_raises_l_kurtosis(cfg):
    rng = np.random.RandomState(2)
    _, tau4_normal = _l_ratios(sorted(rng.randn(3000)))
    _, tau4_heavy = _l_ratios(sorted(rng.standard_t(3, 3000)))
    assert tau4_heavy > tau4_normal + 0.05


def test_features_finite_after_warmup_nan_before(cfg):
    rng = np.random.RandomState(3)
    rows = _run(_new(cfg), rng.randn(200))
    assert all(math.isfinite(v) for v in rows[-1].values())
    b = _new(cfg); b.update(0.0, 0.1, 0.0, 1)
    assert all(math.isnan(v) for v in b.features().values())


def test_online_l_kurtosis_rises_under_tail_break(cfg):
    """A premissa: com variância inalterada, a mudança de FORMA de cauda deve acender o L-kurtosis."""
    rng = np.random.RandomState(4)
    pre = rng.randn(200)
    post = rng.standard_t(3, 200) / math.sqrt(3.0)  # cauda pesada, variância ~1 (mesma escala)
    rows = _run(_new(cfg), np.concatenate([pre, post]))
    assert rows[399]["lmom_lkurt_w100"] > rows[199]["lmom_lkurt_w100"]


def test_scale_invariance(cfg):
    """τ₃/τ₄ são razões -> invariantes a escala. Uma quebra pura de variância NÃO deve movê-las
    (garante ortogonalidade ao eixo de variância)."""
    rng = np.random.RandomState(5)
    base = rng.randn(300)
    r1 = _run(_new(cfg), base)[-1]
    r2 = _run(_new(cfg), base * 5.0)[-1]  # mesma forma, escala 5x
    assert abs(r1["lmom_lkurt_w100"] - r2["lmom_lkurt_w100"]) < 1e-9


def test_history_equivalence(cfg):
    """LMomentBlock foi PODADO do pipeline (scorer/calibration) por ROI negativo, mas o bloco
    continua funcional standalone -- o cálculo sobre o histórico bate com uma re-execução do bloco.
    (Não checamos mais h0.null_stats: a calibração dele foi removida junto com o desligamento.)"""
    from sbrt.state.lmoments import history_null_series
    rng = np.random.RandomState(6)
    e_hist = rng.randn(1500)
    series = history_null_series(e_hist, cfg)
    rows = _run(_new(cfg), e_hist)
    manual = [r["lmom_lkurt_w100"] for r in rows if math.isfinite(r["lmom_lkurt_w100"])]
    from_series = [v for v in series["lmom_lkurt_w100"] if math.isfinite(v)]
    assert np.allclose(manual, from_series, atol=1e-12)
