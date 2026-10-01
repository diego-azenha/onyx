#!/usr/bin/env python
"""I2 (knowledge/frentes/estrategias-01-10/README.md): modelo de RISCO (hazard) com outro rótulo.

Rótulo h_t = 1{τ ≤ t < τ + K}: "a quebra aconteceu nos últimos K passos". As linhas t ≥ τ + K saem do
treino, então o modelo aprende a ASSINATURA DO INÍCIO da quebra, e não o "já quebrou faz tempo" que
domina o rótulo do E1. Mesmas features e mesmos folds do Onyx (OOF honesto). Depois a hazard vira
features causais por série (máximo acumulado, EWMA, soma de evidência) e entra na combinação com o E1
com pesos ajustados DIRETO na TS-AUC, cross-fit por paridade de id (lição do I1: perda agregada não
segue a ordenação por passo).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, extra_trees=True, verbose=-1, seed=0,
              num_threads=6, deterministic=True, force_row_wise=True)


def lg(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--rounds", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    art = Path("artifacts/i2"); art.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet("data/processed/train_rows.parquet").sort_values(["id", "t"]).reset_index(drop=True)
    cols = [c for c in rows.columns if c not in ("id", "t", "y", "thin_weight")]
    t = rows["t"].to_numpy(); y = rows["y"].to_numpy(); ids = rows["id"].to_numpy()
    tau = rows.loc[y == 1].groupby("id")["t"].min()
    tau_row = pd.Series(ids).map(tau).to_numpy(dtype=np.float64)          # NaN nos negativos
    h = ((tau_row <= t) & (t < tau_row + args.K)).astype(np.int8)
    usa = ~(t >= tau_row + args.K)                                       # tira as linhas tardias
    g = pd.DataFrame({"t": t, "h": h})[usa].groupby("t")["h"]
    npos, ntot = g.sum(), g.size()
    w = np.where(h == 1, pd.Series(t).map(ntot - npos).to_numpy(), pd.Series(t).map(npos).to_numpy()).astype(np.float64)
    w = np.where(usa, w, 0.0); w = w / w[usa].mean()
    X = rows[cols].to_numpy(dtype=np.float32)
    meta = rows[["id", "t", "y"]]
    del rows
    oof = np.full(len(t), np.nan)
    for k, (tr, va) in enumerate(grouped_stratified_kfold(meta, cfg.lightgbm.n_folds, cfg.seed)):
        tr = tr[usa[tr]]; vv = va[usa[va]]
        dtr = lgb.Dataset(X[tr], h[tr], weight=w[tr], free_raw_data=True)
        dva = lgb.Dataset(X[vv], h[vv], weight=w[vv], reference=dtr)
        m = lgb.train(PARAMS, dtr, args.rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va] = m.predict(X[va], raw_score=True, num_iteration=m.best_iteration)
        print(f"fold {k}: {m.best_iteration} árvores", flush=True)
    del X
    d = pd.DataFrame({"id": ids, "t": t, "y": y, "hz": oof})
    d.to_parquet(art / f"oof_hazard_K{args.K}.parquet", index=False)
    # features causais da hazard por série (grade thin; decaimento pelo passo real)
    novo = np.r_[True, ids[1:] != ids[:-1]]
    dt = np.r_[1.0, np.diff(t).astype(float)]; dt[novo] = 1.0
    z = oof - pd.Series(oof).groupby(t).transform("median").to_numpy()
    mx = np.empty(len(t)); e1 = np.empty(len(t)); e2 = np.empty(len(t)); sp = np.empty(len(t))
    a1 = 1 - (1 - 0.05) ** dt; a2 = 1 - (1 - 0.01) ** dt
    for i in range(len(t)):
        if novo[i]:
            mx[i] = e1[i] = e2[i] = z[i]; sp[i] = max(z[i], 0) * dt[i]
        else:
            mx[i] = max(mx[i - 1], z[i]); e1[i] = e1[i - 1] + a1[i] * (z[i] - e1[i - 1])
            e2[i] = e2[i - 1] + a2[i] * (z[i] - e2[i - 1]); sp[i] = sp[i - 1] + max(z[i], 0) * dt[i]
    feats = {"hz": z, "hz_max": mx, "hz_ewma05": e1, "hz_ewma01": e2, "hz_soma_pos": np.log1p(sp)}
    e1df = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"]).reset_index(drop=True)
    assert (e1df["t"].to_numpy() == t).all()
    base = lg(e1df["oof_pred"].to_numpy())
    wm = board_grid_multipliers("data/y_train.parquet", t)
    par = ids % 2 == 0

    def zt(x):
        s = pd.Series(x); gg = s.groupby(t)
        return ((s - gg.transform("mean")) / (gg.transform("std") + 1e-9)).to_numpy()

    zb = zt(base)
    rel = {"K": args.K, "e1": weighted_ts_auc(t, y, base, wm)}
    for nome, x in feats.items():
        zx = zt(x)
        rel[f"so_{nome}"] = weighted_ts_auc(t, y, x, wm)
        comb = np.empty(len(t)); esc = []
        for aj, apar in ((par, ~par), (~par, par)):
            cur = {a: weighted_ts_auc(t[aj], y[aj], zb[aj] + a * zx[aj], wm) for a in (0, .1, .2, .3, .5, .8)}
            a = max(cur, key=cur.get); esc.append(a); comb[apar] = zb[apar] + a * zx[apar]
        rel[f"e1_mais_{nome}"] = {"pesos": esc, "tsauc": weighted_ts_auc(t, y, comb, wm)}
        print(nome, rel[f"so_{nome}"], rel[f"e1_mais_{nome}"], flush=True)
    (art / f"i2_K{args.K}.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
