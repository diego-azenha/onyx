"""P4 (docs/INVESTIGACAO_FALHAS_V3.md): bipower/saltos + leverage."""
from __future__ import annotations

import math

import numpy as np

from sbrt.robustness.generators import generate
from sbrt.state.h0 import fit_h0
from sbrt.state.jumps import JumpBlock


def _run(block, series):
    out = []
    for t, x in enumerate(series, start=1):
        block.update(float(x), float(x), float(x), t)
        out.append(block.features())
    return out


def _new(cfg):
    b = JumpBlock()
    b.reset(None, cfg)
    return b


def test_features_finite_after_warmup_nan_before(cfg):
    rng = np.random.RandomState(0)
    assert all(math.isfinite(v) for v in _run(_new(cfg), rng.randn(200))[-1].values())
    b = _new(cfg); b.update(0.1, 0.1, 0.1, 1)
    assert all(math.isnan(v) for v in b.features().values())


def test_isolated_jumps_raise_jump_ratio(cfg):
    """PREMISSA P4: saltos isolados (descontínuos) elevam a razão (RV−BV)/RV; ruído contínuo não."""
    rng = np.random.RandomState(1)
    clean = rng.randn(200)
    withjumps = rng.randn(200).copy()
    for pos in (40, 90, 140, 180):
        withjumps[pos] += rng.choice([-1, 1]) * 8.0  # saltos isolados
    r_clean = _run(_new(cfg), clean)[-1]["jump_ratio_w100"]
    r_jumps = _run(_new(cfg), withjumps)[-1]["jump_ratio_w100"]
    assert r_jumps > r_clean + 0.1


def test_garch_cluster_has_low_jump_ratio_vs_isolated(cfg):
    """DISCRIMINADOR T6/T9: um cluster GARCH (volatilidade contínua) tem razão de salto BAIXA
    comparada a outliers isolados de mesma energia — é isso que separa T6 de T9."""
    garch_hist, garch_online, _ = generate("t6", seed=0, cfg=cfg)
    rng = np.random.RandomState(2)
    isolated = rng.randn(len(garch_online)).copy()
    for pos in range(50, len(isolated), 60):
        isolated[pos] += 9.0

    r_garch = _run(_new(cfg), garch_online)[-1]["jump_ratio_w100"]
    r_iso = _run(_new(cfg), isolated)[-1]["jump_ratio_w100"]
    assert r_iso > r_garch


def test_bipower_is_jump_robust_variance(cfg):
    """ln(RV)−ln(BV) sobe com saltos (RV infla, BV não) mas fica ~0 sob ruído limpo."""
    rng = np.random.RandomState(3)
    clean = _run(_new(cfg), rng.randn(200))[-1]["jump_rvbv_ln_w100"]
    jx = rng.randn(200).copy(); jx[100] += 12.0
    withjump = _run(_new(cfg), jx)[-1]["jump_rvbv_ln_w100"]
    assert withjump > clean
    assert abs(clean) < 0.15


def test_semivariance_asymmetry_sign(cfg):
    rng = np.random.RandomState(4)
    # cauda inflada só do lado positivo
    x = rng.randn(300).copy()
    x[x > 0] *= 2.5
    asym = _run(_new(cfg), x)[-1]["jump_semivar_asym_w100"]
    assert asym > 0.2


def test_calibration_and_history_equivalence(cfg):
    from sbrt.state.jumps import history_null_series
    rng = np.random.RandomState(5)
    h0 = fit_h0(rng.randn(3000), cfg)
    assert "jump_ratio_w100" in h0.null_stats and "jump_leverage_w100" in h0.null_stats

    e_hist = rng.randn(1500)
    series = history_null_series(e_hist, cfg)
    rows = _run(_new(cfg), e_hist)
    manual = [r["jump_ratio_w100"] for r in rows if math.isfinite(r["jump_ratio_w100"])]
    from_series = [v for v in series["jump_ratio_w100"] if math.isfinite(v)]
    assert np.allclose(manual, from_series, atol=1e-12)
