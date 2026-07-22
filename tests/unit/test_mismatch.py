"""F2 (docs/BACKLOG_TSAUC.md): o filtro congelado do histórico ainda vale online?

O teste que importa é `test_white_e_detects_ar_structure_that_lag1_alone_misses`: valida a premissa
do bloco — que a brancura multi-lag sobre `e` puro pega desalinhamento do AR(10) congelado que as
features de lag-1 já existentes (`accum_*_rho1_fz`, `dep_*rho1`) deixam passar."""
from __future__ import annotations

import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.mismatch import MismatchBlock, history_null_series


def _run(cfg, series, h0=None):
    blk = MismatchBlock()
    blk.reset(h0, cfg)
    out = []
    for t, x in enumerate(series, start=1):
        blk.update(float(x), float(x), float(x), t)
        out.append(blk.features())
    return out


def test_emits_nan_during_warmup_and_never_invents_numbers(cfg):
    rng = np.random.RandomState(0)
    feats = _run(cfg, rng.randn(200))
    wmin = cfg.features.warmup_min_n
    for key, val in feats[wmin - 2].items():
        assert math.isnan(val), key
    for key, val in feats[-1].items():
        assert math.isfinite(val), key


def test_white_e_is_near_zero_on_white_noise(cfg):
    """Sob H0 o filtro branqueia: a massa Sum rho_k^2 fica perto de 0 (cada rho_k ~ 1/sqrt(w))."""
    rng = np.random.RandomState(1)
    feats = _run(cfg, rng.randn(4000))[-1]
    assert feats["mismatch_white_e_w050"] < 0.5
    assert feats["mismatch_arch_e2_w050"] < 0.5


def test_white_e_detects_ar_structure_that_lag1_alone_misses(cfg):
    """PREMISSA CENTRAL DE F2. Uma série cuja autocorrelação vive em lags > 1 — aqui um AR sazonal
    puro no lag 3 — tem rho_1 ~ 0 e portanto é invisível para as features de lag-1 que o banco já
    tinha. A massa multi-lag sobre `e` acende."""
    rng = np.random.RandomState(2)
    n = 4000
    x = np.zeros(n)
    eps = rng.randn(n)
    for i in range(3, n):
        x[i] = 0.7 * x[i - 3] + eps[i]      # dependência só no lag 3

    white = _run(cfg, x)[-1]
    iid = _run(cfg, rng.randn(n))[-1]

    from sbrt.state.dependence import _RollingAutocorr
    ac = _RollingAutocorr(50, 1)
    for v in x:
        ac.update(float(v))
    assert abs(ac.rho(1)) < 0.25, "premissa não se sustenta: o lag 1 já veria esta série"

    assert white["mismatch_white_e_w050"] > iid["mismatch_white_e_w050"] + 0.3


def test_arch_e2_detects_multilag_volatility_clustering(cfg):
    """McLeod-Li sobre e²: clustering de volatilidade que não vive no lag 1."""
    rng = np.random.RandomState(3)
    n = 4000
    sigma = np.ones(n)
    for i in range(3, n):
        sigma[i] = math.sqrt(0.2 + 0.75 * sigma[i - 3] ** 2 * (rng.randn() ** 2))
    x = sigma * rng.randn(n)

    garch = _run(cfg, x)[-1]
    iid = _run(cfg, rng.randn(n))[-1]
    assert garch["mismatch_arch_e2_w050"] > iid["mismatch_arch_e2_w050"]


def test_score_cusum_accumulates_under_dependence_and_stays_low_under_h0(cfg):
    rng = np.random.RandomState(4)
    h0 = fit_h0(rng.randn(3000), cfg)

    n = 1500
    ar = np.zeros(n)
    eps = rng.randn(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + eps[i]

    dep = _run(cfg, ar, h0)[-1]
    iid = _run(cfg, rng.randn(n), h0)[-1]

    assert dep["mismatch_score_cusum_pos"] > iid["mismatch_score_cusum_pos"]
    assert iid["mismatch_score_cusum_pos"] < 20.0


def test_score_cusum_normalizer_makes_delta_scale_free(cfg):
    """O escore é dividido por sqrt(n_lags): a variância sob H0 fica ~1 independentemente de L, que é
    o que permite reusar o mesmo `cusum_delta` sem reajuste quando `max_lag` muda."""
    rng = np.random.RandomState(5)
    h0 = fit_h0(rng.randn(3000), cfg)
    e = rng.randn(6000)

    blk = MismatchBlock()
    blk.reset(h0, cfg)
    scores = []
    prev: list = []
    for t, x in enumerate(e, start=1):
        n = min(len(prev), blk.L)
        if n > 0:
            s = sum(float(x) * prev[-k] for k in range(1, n + 1))
            scores.append(s / (math.sqrt(n) * h0.sigma_u))
        prev.append(float(x))
        blk.update(float(x), float(x), float(x), t)

    arr = np.array(scores[blk.L:])
    assert abs(float(arr.mean())) < 0.1
    assert 0.6 < float(arr.std(ddof=1)) < 1.6


def test_history_null_series_excludes_the_recursive_features(cfg):
    """As de janela saem pela passada contínua; os CUSUMs de escore têm transiente e por isso o nulo
    deles vem de réplicas com reinício (calibration.recursive_features), não daqui."""
    rng = np.random.RandomState(6)
    series = history_null_series(rng.randn(2000), cfg)
    assert any(n.startswith("mismatch_white_e_") for n in series)
    assert any(n.startswith("mismatch_arch_e2_") for n in series)
    assert not any(n.startswith("mismatch_score_cusum") for n in series)
    for name, vals in series.items():
        assert len(vals) == 2000, name
