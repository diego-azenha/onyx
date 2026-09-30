#!/usr/bin/env python
"""S6: combina o E1 com os especialistas por tipo (knowledge/frentes/teto-offline/S6-especialistas-por-tipo.md).

Bag das sementes disponíveis de cada especialista. Combinações cross-fit por paridade de id:
(a) logística sobre [logit E1, logit sobe, logit desce, logit fixa, log t];
(b) E1 + w · max(z_especialistas), w de uma grade.
TS-AUC geral (grade do board) e por tipo de quebra contra todos os negativos.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

SEMENTES = [int(s) for s in (sys.argv[1:] or ["777", "101"])]


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    base = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"]).reset_index(drop=True)
    d = base[["id", "t", "y"]].copy(); d["e1"] = lg(base["oof_pred"].to_numpy())
    for T in ("sobe", "desce", "fixa"):
        acc = None
        for s in SEMENTES:
            x = pd.read_parquet(f"artifacts/models/oof_s6_{T}_s{s}.parquet").sort_values(["id", "t"]).reset_index(drop=True)
            acc = x["oof_pred"].to_numpy() if acc is None else acc + x["oof_pred"].to_numpy()
        d[T] = lg(acc / len(SEMENTES))
    d["log_t"] = np.log(d["t"].astype(float))
    w = board_grid_multipliers("data/y_train.parquet", d["t"].to_numpy())
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    par = (d["id"] % 2 == 0).to_numpy()
    rel = {"e1": weighted_ts_auc(t, y, d["e1"].to_numpy(), w)}
    for T in ("sobe", "desce", "fixa"):
        rel[f"so_{T}"] = weighted_ts_auc(t, y, d[T].to_numpy(), w)
    cols = ["e1", "sobe", "desce", "fixa", "log_t"]
    comb = np.empty(len(d))
    for aj, ap in ((par, ~par), (~par, par)):
        m = LogisticRegression(C=1.0, max_iter=500).fit(d.loc[aj, cols].to_numpy(), y[aj])
        comb[ap] = m.decision_function(d.loc[ap, cols].to_numpy())
    rel["logistica"] = weighted_ts_auc(t, y, comb, w)
    z = {c: (d[c] - d[c].mean()) / d[c].std() for c in ("e1", "sobe", "desce", "fixa")}
    mx = np.maximum.reduce([z["sobe"].to_numpy(), z["desce"].to_numpy(), z["fixa"].to_numpy()])
    comb2 = np.empty(len(d)); esc = {}
    for nome, aj, ap in (("par", par, ~par), ("impar", ~par, par)):
        cur = {a: weighted_ts_auc(t[aj], y[aj], z["e1"].to_numpy()[aj] + a * mx[aj], w) for a in (0, .1, .2, .3, .5, .7, 1.0)}
        a = max(cur, key=cur.get); esc[nome] = a; comb2[ap] = z["e1"].to_numpy()[ap] + a * mx[ap]
    rel["e1_mais_max"] = weighted_ts_auc(t, y, comb2, w); rel["peso_max"] = esc
    tipo = pd.read_parquet("artifacts/roteiro/tipo_quebra.parquet")["tipo"]
    d["tipo"] = d["id"].map(tipo)
    por_tipo = {}
    for tp in ("sobe", "desce", "fixa"):
        m_ = (d["tipo"] == "neg") | (d["tipo"] == tp)
        por_tipo[tp] = {c: weighted_ts_auc(t[m_], y[m_], s[m_], w) for c, s in
                        (("e1", d["e1"].to_numpy()), ("logistica", comb), ("especialista", d[tp].to_numpy()))}
    rel["por_tipo"] = por_tipo
    pd.DataFrame({"id": d["id"], "t": t, "y": y, "oof_pred": comb}).to_parquet("artifacts/roteiro/s6_logistica.parquet", index=False)
    print(json.dumps(rel, indent=2, default=float))


if __name__ == "__main__":
    main()
