"""GroupKFold(5, groups=id) + 1 LightGBM por fold (plano §8.3). Predição final = média das
probabilidades dos 5 modelos (model/predict.py).

plano_acao_v1_para_v2.md A2: `y_t = 1{tau<=t}` tem uma taxa-base fortemente crescente com t (~7.6%
a ~39.7%), neutra para TS-AUC por invariância C1 mas dominante para logloss/AUC de linha. Por isso:
(1) a curva de taxa-base vira `init_score`, deixando o LightGBM aprender só o resíduo transversal;
(2) o early stopping usa `binary_logloss` (agora medindo só o resíduo, já que init_score desloca a
métrica), não mais `auc` (que saturava cedo dominada pela taxa-base — 89-110 árvores medidas contra
400-800 esperadas, plano §8.3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb

from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.model.base_rate import fit_base_rate_curve, predict_base_rate_logit
from sbrt.model.predict import ModelEnsemble

_NON_FEATURE_COLS = {"id", "t", "y", "thin_weight"}


def train(rows: pd.DataFrame, weights: np.ndarray, cfg, progress: bool = True) -> tuple:
    """tqdm sobre os 5 folds; dentro de cada fold, LightGBM usa seu próprio log verbose (não
    duplicar barra de progresso, plano §8 regra tqdm). Retorna (ModelEnsemble, oof_pred) — oof_pred
    é a probabilidade calibrada out-of-fold por linha, alinhada a `rows` (diagnóstico A4, não faz
    parte do artefato salvo)."""
    feature_cols = sorted(c for c in rows.columns if c not in _NON_FEATURE_COLS)
    X = rows[feature_cols].to_numpy(dtype=np.float32)  # plano §8.1: float32 no dataset de treino
    y = rows["y"].to_numpy(dtype=np.int32)
    t_values = rows["t"].to_numpy(dtype=np.float64)

    base_rate_curve = fit_base_rate_curve(t_values, y.astype(np.float64))
    init_score_full = predict_base_rate_logit(t_values, base_rate_curve)

    lgb_cfg = cfg.lightgbm
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        max_depth=lgb_cfg.max_depth,
        min_data_in_leaf=lgb_cfg.min_data_in_leaf,
        feature_fraction=lgb_cfg.feature_fraction,
        bagging_fraction=lgb_cfg.bagging_fraction,
        bagging_freq=lgb_cfg.bagging_freq,
        lambda_l2=lgb_cfg.lambda_l2,
        max_bin=lgb_cfg.max_bin,
        deterministic=lgb_cfg.deterministic,
        force_row_wise=lgb_cfg.force_row_wise,
        num_threads=lgb_cfg.train_num_threads,
        seed=cfg.seed,
        verbose=-1,
    )

    boosters = []
    fold_evals = []
    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    folds = list(grouped_stratified_kfold(rows, lgb_cfg.n_folds, cfg.seed))
    fold_iter = tqdm(folds, desc="treinando folds") if progress else folds

    for train_idx, valid_idx in fold_iter:
        dtrain = lgb.Dataset(
            X[train_idx], label=y[train_idx], weight=weights[train_idx], init_score=init_score_full[train_idx]
        )
        dvalid = lgb.Dataset(
            X[valid_idx],
            label=y[valid_idx],
            weight=weights[valid_idx],
            init_score=init_score_full[valid_idx],
            reference=dtrain,
        )
        evals_result: dict = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=lgb_cfg.n_estimators_cap,
            valid_sets=[dvalid],
            callbacks=[
                lgb.early_stopping(lgb_cfg.early_stopping_rounds, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        boosters.append(booster)
        fold_evals.append(evals_result)

        # raw_score=True nunca inclui init_score (é um construto do Dataset, não do modelo salvo) —
        # por isso somamos o mesmo offset usado no treino antes do sigmoid (plano A2, model/predict.py).
        raw_valid = booster.predict(X[valid_idx], raw_score=True)
        full_logit_valid = raw_valid + init_score_full[valid_idx]
        oof_pred[valid_idx] = 1.0 / (1.0 + np.exp(-full_logit_valid))

    ensemble = ModelEnsemble(
        boosters=boosters,
        feature_order=tuple(feature_cols),
        predict_num_threads=lgb_cfg.predict_num_threads,
        fold_evals=fold_evals,
        base_rate_curve=base_rate_curve,
    )
    return ensemble, oof_pred
