import numpy as np

from sbrt.adversarial.leaky_canary import LeakyStreamScorer
from sbrt.evaluation.harness import check_prefix_equivalence
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def _honest_factory(cfg):
    def factory(hist, online):
        h0 = fit_h0(hist, cfg)
        return StreamScorer(h0, default_blocks(), None, cfg)

    return factory


def _leaky_factory(cfg):
    def factory(hist, online):
        h0 = fit_h0(hist, cfg)
        return LeakyStreamScorer(h0, default_blocks(), None, cfg, online)

    return factory


def test_honest_scorer_passes_prefix_equivalence(cfg):
    rng = np.random.RandomState(20)
    hist = rng.randn(1500)
    online = rng.randn(200)
    ok = check_prefix_equivalence(hist, online, _honest_factory(cfg), [1, 10, 50, 100, 199, 200])
    assert ok is True


def test_leaky_canary_is_caught_by_prefix_equivalence(cfg):
    """plano §12.1: o teste de prefixo DEVE reprovar o LeakyStreamScorer — prova de que o
    detector de vazamento funciona."""
    rng = np.random.RandomState(20)
    hist = rng.randn(1500)
    online = rng.randn(200)
    ok = check_prefix_equivalence(hist, online, _leaky_factory(cfg), [1, 10, 50, 100, 199, 200])
    assert ok is False
