import math

import numpy as np

from sbrt.state.bayes_filter import BayesFilterBlock
from sbrt.state.h0 import fit_h0


def test_lo_negative_most_of_the_time_without_break(cfg):
    rng = np.random.RandomState(5)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = BayesFilterBlock()
    blk.reset(params, cfg)

    lo_values = []
    for t, x in enumerate(rng.randn(1000), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        f = blk.features()
        assert all(math.isfinite(v) for v in f.values())
        lo_values.append(f["bayes_lo_h0100"])

    frac_negative = np.mean(np.array(lo_values) < 0)
    assert frac_negative > 0.5


def test_tau_hat_accurate_on_abrupt_break(cfg):
    rng = np.random.RandomState(5)
    hist = rng.randn(2000)
    params = fit_h0(hist, cfg)
    blk = BayesFilterBlock()
    blk.reset(params, cfg)

    online = np.concatenate([rng.randn(300), rng.randn(300) + 2.0])
    true_tau = 301  # 1-based step da primeira observação pós-quebra
    age_at_end = None
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t == 350:
            age_at_end = blk.features()["bayes_age_map_h0100"]

    tau_hat = 350 - age_at_end
    assert abs(tau_hat - true_tau) <= 5
