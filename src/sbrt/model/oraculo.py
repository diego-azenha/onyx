"""Oráculo destilado (knowledge/frentes/oraculo-destilado/README.md): alvos q_t = P(y_t | série inteira).

O oráculo vê, para cada linha (série i, passo t) das linhas de treino, as colunas `cfg.lightgbm.oraculo_cols`
em t e em snapshots FUTUROS da mesma série (t+50, t+200, último passo), mais log t, log T e log(T−t+1).
Não conhece τ. É ajustado com validação cruzada por série (`grouped_stratified_kfold`), e cada linha
recebe a previsão de um oráculo que não viu a série dela. O aluno (`model/train.py`,
`soft_label_modo="destilacao"`) treina com esse q no lugar do rótulo 0/1: E[q|x] = E[y|x] e a
variância é menor (Rao-Blackwell), o que equivale a mais dados para um aprendiz limitado por amostra.

Só roda no TREINO. A inferência não muda: o aluno é um booster comum, fundido como sempre.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.evaluation.splits import grouped_stratified_kfold

DELTAS = (50, 200)

PARAMS_ORACULO = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
                      feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, extra_trees=True,
                      verbose=-1, deterministic=True, force_row_wise=True)


def features_oraculo(rows: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """Matriz (n_linhas, 4·len(cols)+3), alinhada a `rows`. Exige `rows` ordenado por (id, t)."""
    ids = rows["id"].to_numpy(); t = rows["t"].to_numpy()
    if len(ids) > 1 and not ((np.diff(ids) > 0) | ((np.diff(ids) == 0) & (np.diff(t) > 0))).all():
        raise ValueError("features_oraculo: rows precisa estar ordenado por (id, t)")
    F = rows[cols].to_numpy(dtype=np.float32)
    cortes = np.flatnonzero(np.diff(ids)) + 1
    ini = np.r_[0, cortes]; fim = np.r_[cortes, len(ids)]
    idx = {d: np.empty(len(ids), dtype=np.int64) for d in DELTAS}
    ultimo = np.empty(len(ids), dtype=np.int64)
    T_len = np.empty(len(ids), dtype=np.float32)
    for a, b in zip(ini, fim):
        tt = t[a:b]
        for d in DELTAS:
            idx[d][a:b] = a + np.minimum(np.searchsorted(tt, tt + d, side="left"), b - a - 1)
        ultimo[a:b] = b - 1
        T_len[a:b] = tt[-1]
    extra = np.column_stack([np.log(t.astype(np.float32)), np.log(T_len), np.log(T_len - t + 1)])
    return np.hstack([F] + [F[idx[d]] for d in DELTAS] + [F[ultimo], extra]).astype(np.float32)


def alvos_oraculo(rows: pd.DataFrame, cfg, seed: int, n_threads: int = 6) -> np.ndarray:
    """q por linha (alinhado a `rows`, na ordem original), com cross-fit de um nível por série."""
    lgb_cfg = cfg.lightgbm
    ordem = np.lexsort((rows["t"].to_numpy(), rows["id"].to_numpy()))
    r = rows.iloc[ordem].reset_index(drop=True)
    X = features_oraculo(r, list(lgb_cfg.oraculo_cols))
    y = r["y"].to_numpy()
    q = np.zeros(len(r), dtype=np.float64)
    sub = np.arange(len(r)) % max(int(lgb_cfg.oraculo_sub), 1) == 0
    params = {**PARAMS_ORACULO, "seed": int(seed), "num_threads": n_threads}
    for tr, va in grouped_stratified_kfold(r[["id", "t", "y"]], lgb_cfg.n_folds, int(seed)):
        tr = tr[sub[tr]]
        m = lgb.train(params, lgb.Dataset(X[tr], y[tr]), int(lgb_cfg.oraculo_rounds))
        q[va] = m.predict(X[va])
    out = np.empty(len(rows), dtype=np.float64)
    out[ordem] = q
    return out
