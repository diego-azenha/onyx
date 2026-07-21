"""R2 (docs/PARECER_AUDITORIA_ONYX.md §6-R2): valida a fiação do feval custom num dataset sintético
minúsculo -- LightGBM real, poucas rodadas, rápido o suficiente para rodar em CI. Default de
`early_stopping_metric` é "logloss" (achado empírico: "ts_auc_by_t" sozinho regrediu a TS-AUC OOF
real num retreino completo, ver config.py:LightGBMConfig.early_stopping_metric); os testes cobrem
os dois valores para travar a ordenação do feval."""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from sbrt.config import load_config
from sbrt.model.train import train, train_rank
from sbrt.model.weights import compute_row_weights


def _tiny_rows(n_series: int = 60, n_features: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for sid in range(n_series):
        T = rng.integers(20, 40)
        tau = rng.integers(0, T)
        t = np.arange(1, T + 1)
        y = (t > tau).astype(np.int8)
        data = {f"f{i}": rng.normal(0, 1, T).astype(np.float32) + 0.5 * y for i in range(n_features)}
        data["id"] = np.full(T, sid, dtype=np.int32)
        data["t"] = t.astype(np.int32)
        data["y"] = y
        data["thin_weight"] = np.ones(T, dtype=np.float32)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _tiny_cfg(cfg):
    lgb_cfg = dataclasses.replace(
        cfg.lightgbm,
        n_estimators_cap=20,
        early_stopping_rounds=5,
        min_data_in_leaf=5,
        num_leaves=7,
        n_folds=2,
        feval_max_valid_rows=None,
        train_num_threads=1,
        predict_num_threads=1,
    )
    return dataclasses.replace(cfg, lightgbm=lgb_cfg)


def test_train_feval_drives_stopping_and_records_both_metrics():
    cfg = _tiny_cfg(load_config())
    rows = _tiny_rows()
    weights = compute_row_weights(rows, cfg)

    ensemble, oof_pred = train(rows, weights, cfg, progress=False)

    assert len(ensemble.boosters) == cfg.lightgbm.n_folds
    assert np.isfinite(oof_pred).all()
    assert ((oof_pred >= 0.0) & (oof_pred <= 1.0)).all()

    for fold_eval in ensemble.fold_evals:
        assert "valid_0" in fold_eval
        assert "ts_auc_by_t" in fold_eval["valid_0"]
        assert "binary_logloss_diag" in fold_eval["valid_0"]
        assert all(np.isfinite(v) for v in fold_eval["valid_0"]["ts_auc_by_t"])


def test_train_rank_feval_drives_stopping_and_predicts_bounded_oof():
    cfg = _tiny_cfg(load_config())
    rows = _tiny_rows()

    ensemble, oof_pred = train_rank(rows, cfg, progress=False)

    assert len(ensemble.boosters) == cfg.lightgbm.n_folds
    assert np.isfinite(oof_pred).all()
    assert ((oof_pred >= 0.0) & (oof_pred <= 1.0)).all()

    for fold_eval in ensemble.fold_evals:
        assert "ts_auc_by_t" in fold_eval["valid_0"]
        assert "binary_logloss_diag" in fold_eval["valid_0"]
        assert all(np.isfinite(v) for v in fold_eval["valid_0"]["ts_auc_by_t"])


def test_train_rank_predict_one_matches_oof_style_output():
    cfg = _tiny_cfg(load_config())
    rows = _tiny_rows(n_series=20, seed=1)
    ensemble, _ = train_rank(rows, cfg, progress=False)

    feats = {f"f{i}": 0.1 * i for i in range(5)}
    p = ensemble.predict_one(feats)
    assert 0.0 <= p <= 1.0


def test_early_stopping_metric_default_is_logloss():
    cfg = _tiny_cfg(load_config())
    assert cfg.lightgbm.early_stopping_metric == "logloss"


def test_early_stopping_metric_controls_feval_order():
    rows = _tiny_rows()

    cfg_logloss = _tiny_cfg(load_config())
    weights = compute_row_weights(rows, cfg_logloss)
    ensemble_logloss, _ = train(rows, weights, cfg_logloss, progress=False)
    first_key_logloss = next(iter(ensemble_logloss.fold_evals[0]["valid_0"]))
    assert first_key_logloss == "binary_logloss_diag"

    cfg_auc = dataclasses.replace(
        cfg_logloss, lightgbm=dataclasses.replace(cfg_logloss.lightgbm, early_stopping_metric="ts_auc_by_t")
    )
    ensemble_auc, _ = train(rows, weights, cfg_auc, progress=False)
    first_key_auc = next(iter(ensemble_auc.fold_evals[0]["valid_0"]))
    assert first_key_auc == "ts_auc_by_t"
