import math

import numpy as np

from sbrt.state.cusum import CusumBlock
from sbrt.state.h0 import fit_h0


def _make_block(cfg, seed=4):
    rng = np.random.RandomState(seed)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = CusumBlock()
    blk.reset(params, cfg)
    return blk, rng


def test_cusum_all_finite_and_named(cfg):
    blk, rng = _make_block(cfg)
    for t, x in enumerate(rng.randn(500), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    feats = blk.features()
    assert len(feats) == 21
    assert all(math.isfinite(v) for v in feats.values())


def test_cusum_age_resets_to_zero_on_reset_event(cfg):
    blk, rng = _make_block(cfg, seed=42)
    # inovações negativas fortes o suficiente para nunca deixar mean_pos_d100 sair de zero
    for t, x in enumerate(np.full(20, -5.0), start=1):
        blk.update(x, x, x, t)
    feats = blk.features()
    assert feats["cusum_mean_pos_d100"] == 0.0
    assert feats["cusum_age_mean_pos_d050"] == 0.0


def test_cusum_mean_pos_rises_after_shift(cfg):
    blk, rng = _make_block(cfg, seed=4)
    online = np.concatenate([rng.randn(200), rng.randn(300) + 1.5])
    snapshots = {}
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (200, 500):
            snapshots[t] = blk.features()["cusum_mean_pos_d050"]
    assert snapshots[500] > snapshots[200]
