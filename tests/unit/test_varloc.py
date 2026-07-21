"""P3 (docs/INVESTIGACAO_FALHAS_V3.md): variância localizada no changepoint."""
from __future__ import annotations

import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.varloc import VarLocBlock


def _run(block, series):
    out = []
    for t, x in enumerate(series, start=1):
        block.update(float(x), float(x), float(x), t)
        out.append(block.features())
    return out


def _new(cfg):
    b = VarLocBlock()
    b.reset(None, cfg)
    return b


def test_near_zero_under_h0(cfg):
    rng = np.random.RandomState(0)
    r = _run(_new(cfg), rng.randn(400))[-1]
    assert abs(r["varloc_max_z"]) < 4.0  # ruído amostral, não elevação sistemática
    assert math.isfinite(r["varloc_recent_vs_lagged"])


def test_max_z_rises_after_variance_break(cfg):
    rng = np.random.RandomState(1)
    rows = _run(_new(cfg), np.concatenate([rng.randn(300), rng.randn(150) * 2.0]))
    assert rows[449]["varloc_max_z"] > rows[299]["varloc_max_z"] + 3.0


def test_recent_vs_lagged_positive_after_recent_increase(cfg):
    rng = np.random.RandomState(2)
    # quebra recente: variância sobe nos últimos ~25 passos
    rows = _run(_new(cfg), np.concatenate([rng.randn(400), rng.randn(25) * 3.0]))
    assert rows[-1]["varloc_recent_vs_lagged"] > 0.5


def test_argmax_scale_localizes_recent_vs_old(cfg):
    """A propriedade que motiva P3: uma quebra RECENTE é melhor vista numa escala CURTA; uma ANTIGA,
    numa escala longa. argmax_lnscale deve refletir isso."""
    rng = np.random.RandomState(3)
    base = rng.randn(400)
    recent = np.concatenate([base, rng.randn(20) * 3.0])   # quebra de idade ~20
    old = np.concatenate([base[:200], rng.randn(220) * 1.6])  # quebra de idade ~220
    r_recent = _run(_new(cfg), recent)[-1]["varloc_argmax_lnscale"]
    r_old = _run(_new(cfg), old)[-1]["varloc_argmax_lnscale"]
    assert r_recent < r_old  # escala menor para a quebra recente


def test_features_nan_before_warmup(cfg):
    b = _new(cfg)
    b.update(0.5, 0.5, 0.5, 1)
    assert all(math.isnan(v) for v in b.features().values())


def test_calibration_and_history_equivalence(cfg):
    from sbrt.state.varloc import history_null_series
    rng = np.random.RandomState(4)
    h0 = fit_h0(rng.randn(3000), cfg)
    assert "varloc_max_z" in h0.null_stats
    assert "varloc_argmax_lnscale" not in h0.null_stats  # índice de escala, não calibrado

    e_hist = rng.randn(1500)
    series = history_null_series(e_hist, cfg)
    rows = _run(_new(cfg), e_hist)
    manual = [r["varloc_max_z"] for r in rows if math.isfinite(r["varloc_max_z"])]
    from_series = [v for v in series["varloc_max_z"] if math.isfinite(v)]
    assert np.allclose(manual, from_series, atol=1e-12)
