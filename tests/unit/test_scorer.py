import math

import numpy as np
import pytest

from sbrt.robustness.generators import SCENARIO_IDS, generate
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


@pytest.mark.parametrize("scenario_id", SCENARIO_IDS)
def test_scorer_runs_all_scenarios_without_exception_or_nan(cfg, scenario_id):
    """plano §6 / docs/PLANO_REPOSITORIO.md §7.2 (Frente F): roda sobre as 13+2 séries de
    robustness/generators.py garantindo passos sem exceção/NaN fora do warm-up. NÃO checa
    qualidade de detecção (isso é tests/robustness/)."""
    hist, online, _ = generate(scenario_id, seed=0, cfg=cfg)
    h0 = fit_h0(hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)

    scores = [scorer.update(float(x)) for x in online]
    assert all(math.isfinite(s) and 0.0 <= s <= 1.0 for s in scores)


def test_scorer_feature_order_stable_and_no_leakage_of_T(cfg):
    rng = np.random.RandomState(0)
    hist = rng.randn(2000)
    h0 = fit_h0(hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)
    feats = scorer.update_features(float(rng.randn()))
    assert "T" not in feats and "t_total" not in feats
    assert len(feats) == 80


def test_scorer_new_instance_per_series_gives_same_result_regardless_of_order(cfg):
    rng = np.random.RandomState(0)
    series = [(rng.randn(1200), rng.randn(100)) for _ in range(3)]

    def run_all(order):
        out = []
        for i in order:
            hist, online = series[i]
            h0 = fit_h0(hist, cfg)
            scorer = StreamScorer(h0, default_blocks(), None, cfg)
            out.append([scorer.update(float(x)) for x in online])
        return out

    result_a = run_all([0, 1, 2])
    result_b = run_all([2, 0, 1])
    # reordenar result_b de volta para comparar contra result_a (posições, não ordem de execução)
    reordered = [None, None, None]
    for pos, i in enumerate([2, 0, 1]):
        reordered[i] = result_b[pos]
    assert reordered == result_a
