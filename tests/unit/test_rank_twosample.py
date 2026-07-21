import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.rank_twosample import RankTwoSampleBlock


def _new_block(cfg, seed=6):
    rng = np.random.RandomState(seed)
    hist = rng.randn(2000)
    h0 = fit_h0(hist, cfg)
    blk = RankTwoSampleBlock()
    blk.reset(h0, cfg)
    return blk, rng, h0


def test_features_finite_after_warmup(cfg):
    blk, rng, _ = _new_block(cfg)
    for t, x in enumerate(rng.randn(300), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    feats = blk.features()
    assert len(feats) == 3 * len(cfg.rank_twosample.windows)
    assert all(math.isfinite(v) for v in feats.values())


def test_features_nan_before_warmup(cfg):
    blk, rng, _ = _new_block(cfg)
    x = float(rng.randn())
    e = float(np.clip(x, *cfg.h0.clip_e))
    blk.update(e, x, e, 1)
    feats = blk.features()
    assert all(math.isnan(v) for v in feats.values())


def test_wilcoxon_z_falls_under_positive_shift(cfg):
    # p_right e cauda superior (convencao herdada de ConformalBlock): encolhe para perto de 0 sob
    # shift positivo, entao o z fica mais NEGATIVO -- ver docstring de rank_twosample.py.
    blk, rng, _ = _new_block(cfg)
    online = np.concatenate([rng.randn(150), rng.randn(100) + 1.5])
    snapshots = {}
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (150, 250):
            snapshots[t] = blk.features()["ranktwo_wilcoxon_z_w100"]
    assert snapshots[250] < snapshots[150]


def test_dispersion_z_falls_under_variance_shift(cfg):
    # mesma convencao de sinal (cauda superior) aplicada a p_abs.
    blk, rng, _ = _new_block(cfg)
    online = np.concatenate([rng.randn(150), rng.randn(100) * 3.0])
    snapshots = {}
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (150, 250):
            snapshots[t] = blk.features()["ranktwo_dispersion_z_w100"]
    assert snapshots[250] < snapshots[150]


def test_shape_chi2_low_under_null_high_under_shape_break(cfg):
    blk, rng, _ = _new_block(cfg)
    # sob H0 (mesma distribuicao do historico), o chi2 de janela deve ficar tipicamente baixo
    for t, x in enumerate(rng.randn(150), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    chi2_null = blk.features()["ranktwo_shape_chi2_w100"]

    blk2, rng2, _ = _new_block(cfg)
    for t, x in enumerate(rng2.randn(150), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk2.update(e, x, e, t)
    # forca uma janela extrema: tudo no bin mais alto (viola a distribuicao do historico)
    for t in range(151, 251):
        e = 6.0
        blk2.update(e, e, e, t)
    chi2_break = blk2.features()["ranktwo_shape_chi2_w100"]

    assert chi2_break > chi2_null


def test_bin_counts_sum_to_window_size(cfg):
    blk, rng, _ = _new_block(cfg)
    for t, x in enumerate(rng.randn(300), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    for w in cfg.rank_twosample.windows:
        n_eff = min(300, w)
        assert abs(sum(blk.bin_counts[w]) - n_eff) < 1e-9
