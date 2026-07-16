import math

import numpy as np

from sbrt.state.conformal import ConformalBlock
from sbrt.state.h0 import fit_h0


def test_conformal_finite_and_bounded_role(cfg):
    rng = np.random.RandomState(6)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = ConformalBlock()
    blk.reset(params, cfg)

    for t, x in enumerate(rng.randn(500), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        feats = blk.features()
        assert all(math.isfinite(v) for v in feats.values())
    assert feats["conformal_logm_abs_reset"] >= 0.0  # reset floor


def test_conformal_right_martingale_rises_under_positive_shift(cfg):
    rng = np.random.RandomState(6)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = ConformalBlock()
    blk.reset(params, cfg)

    online = np.concatenate([rng.randn(500), rng.randn(300) + 1.2])
    snapshots = {}
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (500, 800):
            snapshots[t] = blk.features()["conformal_logm_right"]
    assert snapshots[800] > snapshots[500]
