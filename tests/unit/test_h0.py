import numpy as np
import pytest
from scipy import stats

from sbrt.state.h0 import fit_h0, seed_lag_buffer, whiten_step


def test_fit_h0_recovers_known_ar_coefficient(cfg):
    rng = np.random.RandomState(0)
    n = 3000
    x = np.zeros(n)
    eps = rng.randn(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t]

    params = fit_h0(x[:2000], cfg)
    assert abs(params.phi[0] - 0.6) < 0.05


def test_fit_h0_rejects_ar_on_white_noise(cfg):
    rng = np.random.RandomState(1)
    params = fit_h0(rng.randn(2000), cfg)
    assert params.ar_r2 < cfg.h0.ar_r2_min_reduction
    assert np.all(params.phi == 0.0)


def test_whiten_step_produces_approximately_gaussian_innovations(cfg):
    rng = np.random.RandomState(0)
    n = 3000
    x = np.zeros(n)
    eps = rng.randn(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t]

    hist, online = x[:2000], x[2000:]
    params = fit_h0(hist, cfg)
    lags = seed_lag_buffer(params)
    es = [whiten_step(float(v), lags, params, cfg)[0] for v in online]

    ks = stats.kstest(es, "norm")
    assert ks.pvalue > 0.05


def test_fit_h0_raises_below_min_hist_len(cfg):
    with pytest.raises(ValueError):
        fit_h0(np.random.randn(cfg.h0.min_hist_len - 1), cfg)


def test_lag_buffer_crosses_boundary_without_discontinuity(cfg):
    rng = np.random.RandomState(2)
    hist = rng.randn(2000)
    online = rng.randn(100)
    params = fit_h0(hist, cfg)
    lags = seed_lag_buffer(params)
    es = [whiten_step(float(v), lags, params, cfg)[0] for v in online]
    # sem quebra: média dos 5 primeiros passos não deve destoar da média dos passos 6-50
    assert abs(np.mean(es[:5]) - np.mean(es[5:50])) < 1.5
