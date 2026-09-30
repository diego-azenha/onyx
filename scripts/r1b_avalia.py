#!/usr/bin/env python
"""R1b (knowledge/frentes/teto-offline/R1-refit-completo.md): refit ensacado contra folds ensacados.
Lê artifacts/reports/r1_refit/pred_s*.parquet (logits no holdout externo) e compara
(a) média dos folds, (b) média dos refits, (c) 50/50, com IC por bootstrap de séries do holdout."""
import glob
import json

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

fs = sorted(glob.glob("artifacts/reports/r1_refit/pred_s*.parquet"))
P = [pd.read_parquet(f).sort_values(["id", "t"]).reset_index(drop=True) for f in fs]
b = P[0][["id", "t", "y"]].copy()
b["folds"] = np.mean([p["folds"].to_numpy() for p in P], axis=0)
b["full"] = np.mean([p["full"].to_numpy() for p in P], axis=0)
b["mix"] = (b["folds"] + b["full"]) / 2
wm = board_grid_multipliers("data/y_train.parquet", b["t"].to_numpy())


def auc(d, c, w):
    return weighted_ts_auc(d["t"].to_numpy(), d["y"].to_numpy(), d[c].to_numpy(), w)


res = {"sementes": [f[-12:] for f in fs], **{c: auc(b, c, wm) for c in ("folds", "full", "mix")}}
ids = b["id"].unique(); rng = np.random.default_rng(0); grp = b.groupby("id").indices
dl = {"full": [], "mix": []}
for _ in range(200):
    s = rng.choice(ids, len(ids))
    idx = np.concatenate([grp[i] for i in s]); d = b.iloc[idx].reset_index(drop=True)
    w = wm; a0 = auc(d, "folds", w)
    for c in dl:
        dl[c].append(auc(d, c, w) - a0)
for c, v in dl.items():
    res[f"delta_{c}"] = res[c] - res["folds"]; res[f"ic95_{c}"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
print(json.dumps(res, indent=2))
