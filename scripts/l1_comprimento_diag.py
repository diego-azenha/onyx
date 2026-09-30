#!/usr/bin/env python
"""L1 (diagnóstico, NUNCA para produção): quanto vale conhecer o comprimento T do online?
O vazamento fechado em 8/jun expunha T. Com τ uniforme no online, T dá o prior P(τ ≤ t) = t/T.
Mede a TS-AUC de (a) só o prior t/T, (b) E1 + prior (logística cross-fit por paridade de id)."""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log1p(-np.clip(p, 1e-6, 1 - 1e-6))
b = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"]).reset_index(drop=True)
T = b.groupby("id")["t"].transform("max").to_numpy() + 1
t, y = b["t"].to_numpy(), b["y"].to_numpy()
w = board_grid_multipliers("data/y_train.parquet", t)
pri = lg((t + 1) / (T + 1)); e1 = lg(b["oof_pred"].to_numpy())
rel = {"e1": weighted_ts_auc(t, y, e1, w), "so_prior_t_sobre_T": weighted_ts_auc(t, y, pri, w)}
X = np.column_stack([e1, pri, np.log(t + 1.0), np.log(T)])
par = (b["id"] % 2 == 0).to_numpy(); comb = np.empty(len(t))
for aj, ap in ((par, ~par), (~par, par)):
    comb[ap] = LogisticRegression(max_iter=500).fit(X[aj], y[aj]).decision_function(X[ap])
rel["e1_mais_prior"] = weighted_ts_auc(t, y, comb, w)
rel["e1_mais_prior_soma"] = weighted_ts_auc(t, y, e1 + pri, w)
print(json.dumps(rel, indent=2))
