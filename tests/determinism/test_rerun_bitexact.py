import numpy as np

from sbrt.adversarial.determinism import rerun_bitexact
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def test_rerun_30_percent_bit_exact(cfg):
    rng = np.random.RandomState(9)
    sample = [(rng.randn(1500), rng.randn(int(rng.randint(50, 300)))) for _ in range(20)]

    def factory(hist):
        h0 = fit_h0(hist, cfg)
        return StreamScorer(h0, default_blocks(), None, cfg)

    ok = rerun_bitexact(sample, factory, fraction=0.3, seed=1, progress=False)
    assert ok is True
