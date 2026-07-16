import math

import numpy as np

from sbrt.state.accumulators import AccumulatorBlock
from sbrt.state.h0 import fit_h0


def test_accumulators_finite_after_warmup_and_nan_before(cfg):
    rng = np.random.RandomState(3)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = AccumulatorBlock()
    blk.reset(params, cfg)

    feats_t1 = None
    feats_final = None
    for t, x in enumerate(rng.randn(1000), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        feats = blk.features()
        if t == 1:
            feats_t1 = feats
        feats_final = feats

    assert len(feats_final) == 31
    assert any(math.isnan(v) for v in feats_t1.values())
    assert all(math.isfinite(v) for v in feats_final.values())


def test_cusum_mean_z_window_responds_to_shift(cfg):
    rng = np.random.RandomState(3)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = AccumulatorBlock()
    blk.reset(params, cfg)

    online = np.concatenate([rng.randn(300), rng.randn(50) + 3.0])
    z_before = z_after = None
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t == 300:
            z_before = blk.features()["accum_window_mean_z_w010"]
        if t == 320:
            z_after = blk.features()["accum_window_mean_z_w010"]
    assert z_after > z_before
