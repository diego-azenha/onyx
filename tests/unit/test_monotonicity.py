import dataclasses

import numpy as np

from sbrt.postprocess.monotonicity import apply
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def test_free_mode_is_identity(cfg):
    assert apply(0.3, 0.9, "free", cfg) == 0.3
    assert apply(0.7, None, "free", cfg) == 0.7


def test_hold_mode_never_decreases():
    class _Cfg:
        pass

    assert apply(0.3, 0.9, "hold", _Cfg()) == 0.9
    assert apply(0.95, 0.9, "hold", _Cfg()) == 0.95


def test_soft_mode_decays_at_bounded_rate(cfg):
    v = apply(0.1, 0.9, "soft", cfg)
    assert v == max(0.1, 0.9 - cfg.postprocess.soft_decay)


def test_ema_mode_blends(cfg):
    v = apply(0.0, 1.0, "ema", cfg)
    alpha = cfg.postprocess.ema_alpha
    assert abs(v - (alpha * 0.0 + (1 - alpha) * 1.0)) < 1e-12


def test_ce1_contraexample_hold_traps_free_does_not(cfg):
    """plano §12.5 CE1: série sem quebra, outlier de 6sigma em t=15. V-hold trava o pico pelos
    passos restantes (série negativa o tempo todo); V-livre decai. docs/PLANO_REPOSITORIO.md
    Frente I DoD: 'V-hold trava no contraexemplo CE1 e V-livre não'."""
    rng = np.random.RandomState(7)
    hist = rng.randn(2000)
    online = rng.randn(600)
    online[14] = 6.0  # outlier de 6 sigma em t=15 (1-based)

    h0 = fit_h0(hist, cfg)

    cfg_free = dataclasses.replace(cfg, postprocess=dataclasses.replace(cfg.postprocess, mode="free"))
    cfg_hold = dataclasses.replace(cfg, postprocess=dataclasses.replace(cfg.postprocess, mode="hold"))

    scorer_free = StreamScorer(h0, default_blocks(), None, cfg_free)
    scorer_hold = StreamScorer(h0, default_blocks(), None, cfg_hold)

    scores_free = [scorer_free.update(float(x)) for x in online]
    scores_hold = [scorer_hold.update(float(x)) for x in online]

    peak_free = max(scores_free[10:60])
    assert scores_free[-1] < peak_free - 0.05  # score livre decai após o outlier
    assert scores_hold[-1] >= max(scores_hold[10:60]) - 1e-12  # hold nunca decresce do pico
