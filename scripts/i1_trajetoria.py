#!/usr/bin/env python
"""I1 (knowledge/frentes/estrategias-01-10/README.md): 2º estágio sobre a TRAJETÓRIA do score do E1.

O E1 pontua cada linha (série, t) sozinha, e o v-EMA é uma EWMA só. Aqui a trajetória inteira do logit
do E1 (bag4, sem v-EMA) vira features causais: EWMAs em vários λ, máximo acumulado, inclinação,
tempo acima de limiares, desvio do próprio passado. Um GBM pequeno, cross-fit por paridade de id
(ajusta numa paridade, avalia na outra), estima P(τ ≤ t | trajetória). O OOF está na grade thin
(Δt = 1/2/4), então as EWMAs usam o decaimento por passo real: a = 1 − (1 − λ)^Δt.
"""
from __future__ import annotations

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

LAMBDAS = (0.01, 0.03, 0.1, 0.3)
LIMIARES = (-1.0, 0.0, 1.0)          # no logit do E1 menos a taxa-base do passo


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def trajetoria(d: pd.DataFrame) -> pd.DataFrame:
    """Features causais por (id, t); d ordenado por id, t, com a coluna `z` (logit centrado no passo)."""
    ids = d["id"].to_numpy(); t = d["t"].to_numpy().astype(np.float64); z = d["z"].to_numpy()
    n = len(d)
    novo = np.r_[True, ids[1:] != ids[:-1]]
    dt = np.r_[1.0, np.diff(t)]; dt[novo] = 1.0
    out = {}
    for lam in LAMBDAS:
        e = np.empty(n); a = 1 - (1 - lam) ** dt
        for i in range(n):
            e[i] = z[i] if novo[i] else e[i - 1] + a[i] * (z[i] - e[i - 1])
        out[f"ewma_{lam}"] = e
    mx = np.empty(n); mn = np.empty(n); soma = np.empty(n); cont = np.empty(n)
    acima = {c: np.empty(n) for c in LIMIARES}
    for i in range(n):
        if novo[i]:
            mx[i] = mn[i] = z[i]; soma[i] = z[i] * dt[i]; cont[i] = dt[i]
            for c in LIMIARES:
                acima[c][i] = dt[i] * (z[i] > c)
        else:
            mx[i] = max(mx[i - 1], z[i]); mn[i] = min(mn[i - 1], z[i])
            soma[i] = soma[i - 1] + z[i] * dt[i]; cont[i] = cont[i - 1] + dt[i]
            for c in LIMIARES:
                acima[c][i] = acima[c][i - 1] + dt[i] * (z[i] > c)
    out["max_acum"] = mx; out["min_acum"] = mn; out["media_acum"] = soma / cont
    for c in LIMIARES:
        out[f"frac_acima_{c}"] = acima[c] / cont
    F = pd.DataFrame(out)
    F["z"] = z
    F["rapida_menos_lenta"] = F["ewma_0.3"] - F["ewma_0.01"]
    F["z_menos_media"] = z - F["media_acum"]
    F["z_menos_max"] = z - mx
    F["log_t"] = np.log(t)
    return F


def main() -> None:
    base = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/oof_e1_bag4.parquet"
    d = pd.read_parquet(base).sort_values(["id", "t"]).reset_index(drop=True)
    d["l"] = lg(d["oof_pred"].to_numpy())
    d["z"] = d["l"] - d.groupby("t")["l"].transform("median")
    F = trajetoria(d)
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    wm = board_grid_multipliers("data/y_train.parquet", t)
    g = d.groupby("t")["y"]
    npos = d["t"].map(g.sum()).to_numpy(); nneg = d["t"].map(g.size() - g.sum()).to_numpy()
    w = np.where(y == 1, nneg, npos) * d["t"].map(wm).fillna(1.0).to_numpy()
    w = w / w.mean()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=500,
                  feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, seed=0, num_threads=6, deterministic=True, force_row_wise=True)
    par = (d["id"] % 2 == 0).to_numpy()
    pred = np.empty(len(d))
    for aj, ap in ((par, ~par), (~par, par)):
        m = lgb.train(params, lgb.Dataset(F[aj], y[aj], weight=w[aj]), 300)
        pred[ap] = m.predict(F[ap], raw_score=True)
    vema = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"])
    rel = {"e1_bag4_cru": weighted_ts_auc(t, y, d["l"].to_numpy(), wm),
           "e1_vema (incumbente)": weighted_ts_auc(t, y, vema["oof_pred"].to_numpy(), wm),
           "i1_trajetoria": weighted_ts_auc(t, y, pred, wm)}
    imp = pd.Series(m.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    rel["importancia_top"] = {k: round(float(v / imp.sum()), 3) for k, v in imp.head(8).items()}
    out = d[["id", "t", "y"]].assign(oof_pred=1 / (1 + np.exp(-pred)))
    out.to_parquet("artifacts/models/oof_i1_trajetoria.parquet", index=False)
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
