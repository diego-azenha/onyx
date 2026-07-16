from sbrt.model.fallback import fallback_score


def test_fallback_score_bounded_and_monotonic_in_lo(cfg):
    base = {"bayes_lo_h0025": -5.0, "conformal_logm_abs_reset": 0.0}
    low = fallback_score(base, cfg)
    base_high = dict(base, **{"bayes_lo_h0025": 5.0})
    high = fallback_score(base_high, cfg)
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0
    assert high > low


def test_fallback_score_handles_missing_features():
    class _Cfg:
        class fallback:
            w_lo = 0.9
            w_cusum = 0.4
            w_conformal = 0.3
            bias = 0.0

    score = fallback_score({}, _Cfg())
    assert 0.0 <= score <= 1.0
