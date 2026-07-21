"""R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): RankModelEnsemble e CombinedModelEnsemble."""
from __future__ import annotations

import dataclasses

import numpy as np

from sbrt.config import load_config
from sbrt.model.predict import CombinedModelEnsemble
from sbrt.model.train import train, train_rank
from sbrt.model.weights import compute_row_weights

from tests.unit.test_train import _tiny_cfg, _tiny_rows


def test_rank_ensemble_save_load_roundtrip(tmp_path):
    cfg = _tiny_cfg(load_config())
    rows = _tiny_rows(n_series=30, seed=2)
    ensemble, _ = train_rank(rows, cfg, progress=False)

    path = tmp_path / "rank_model"
    ensemble.save(path)

    from sbrt.model.predict import RankModelEnsemble

    loaded = RankModelEnsemble.load(path)
    feats = {f"f{i}": 0.2 for i in range(5)}
    assert abs(loaded.predict_one(feats) - ensemble.predict_one(feats)) < 1e-9


def test_combined_ensemble_averages_both_arms():
    cfg = _tiny_cfg(load_config())
    rows = _tiny_rows(n_series=30, seed=4)
    weights = compute_row_weights(rows, cfg)

    binary_ensemble, _ = train(rows, weights, cfg, progress=False)
    rank_ensemble, _ = train_rank(rows, cfg, progress=False)
    combined = CombinedModelEnsemble(binary=binary_ensemble, rank=rank_ensemble)

    feats = {f"f{i}": 0.3 for i in range(5)}
    p_bin = binary_ensemble.predict_one(feats)
    p_rank = rank_ensemble.predict_one(feats)
    p_combined = combined.predict_one(feats)

    assert abs(p_combined - 0.5 * (p_bin + p_rank)) < 1e-12
    assert 0.0 <= p_combined <= 1.0
