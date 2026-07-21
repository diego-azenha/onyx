"""BOCPD (Adams-MacKay): posterior de run-length de variância (docs/RESULTADOS_P1_P4.md)."""
from __future__ import annotations

import math

import numpy as np

from sbrt.state.bocpd import BOCPDBlock
from sbrt.state.h0 import fit_h0


def _run(block, series):
    out = []
    for t, x in enumerate(series, start=1):
        block.update(float(x), float(x), float(x), t)
        out.append(block.features())
    return out


def _new(cfg):
    b = BOCPDBlock()
    b.reset(None, cfg)
    return b


def test_posterior_normalized_and_finite(cfg):
    rng = np.random.RandomState(0)
    blk = _new(cfg)
    for t, x in enumerate(rng.randn(300), start=1):
        blk.update(float(x), float(x), float(x), t)
        assert abs(blk.prob.sum() - 1.0) < 1e-9
    feats = blk.features()
    assert all(math.isfinite(v) for v in feats.values())


def test_features_nan_before_warmup(cfg):
    b = _new(cfg)
    b.update(0.3, 0.3, 0.3, 1)
    assert all(math.isnan(v) for v in b.features().values())


def test_regime_var_tracks_post_break_variance(cfg):
    """A feature central: bocpd_regime_var_ln deve rastrear a variância do REGIME ATUAL, localizada
    após o changepoint — não a variância diluída de uma janela fixa que atravessa tau."""
    rng = np.random.RandomState(1)
    online = np.concatenate([rng.randn(300), rng.randn(200) * 3.0])  # var 1 -> 9
    rows = _run(_new(cfg), online)
    pre = rows[299]["bocpd_regime_var_ln"]
    post = rows[-1]["bocpd_regime_var_ln"]
    assert post > pre + 1.0
    # a variância do regime pós-quebra deve se aproximar de ln(9)≈2.2, não da média diluída
    assert post > 1.3


def test_cp_prob_spikes_at_changepoint(cfg):
    """P(run-length recente) deve dar um pico logo após uma quebra abrupta de variância."""
    rng = np.random.RandomState(2)
    online = np.concatenate([rng.randn(300), rng.randn(200) * 4.0])
    rows = _run(_new(cfg), online)
    # janela logo após tau=300
    cp_after = max(rows[t]["bocpd_cp_prob"] for t in range(300, 320))
    cp_before = max(rows[t]["bocpd_cp_prob"] for t in range(250, 300))
    assert cp_after > cp_before


def test_map_runlen_grows_under_stationarity(cfg):
    """Sem quebra, o run-length MAP deve crescer com o tempo (regime longo, sem changepoint)."""
    rng = np.random.RandomState(3)
    rows = _run(_new(cfg), rng.randn(400))
    assert rows[-1]["bocpd_map_runlen"] > rows[100]["bocpd_map_runlen"]


def test_calibration_and_history_equivalence(cfg):
    from sbrt.state.bocpd import history_null_series
    rng = np.random.RandomState(4)
    h0 = fit_h0(rng.randn(3000), cfg)
    assert "bocpd_regime_var_ln" in h0.null_stats
    assert "bocpd_cp_prob" in h0.null_stats
    assert "bocpd_map_runlen" not in h0.null_stats  # localizador de idade, não calibrado

    e_hist = rng.randn(1200)
    series = history_null_series(e_hist, cfg)
    rows = _run(_new(cfg), e_hist)
    manual = [r["bocpd_regime_var_ln"] for r in rows if math.isfinite(r["bocpd_regime_var_ln"])]
    from_series = [v for v in series["bocpd_regime_var_ln"] if math.isfinite(v)]
    assert np.allclose(manual, from_series, atol=1e-12)
