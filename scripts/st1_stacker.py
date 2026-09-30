#!/usr/bin/env python
"""ST1 (knowledge/frentes/teto-offline/ST1-segundo-estagio.md): segundo estágio pequeno sobre poucas
colunas fortes. Entradas: logit do OOF do Onyx (já out-of-fold) + famílias de evidência direcional do
detector de surpresa + t. GBM raso, com a MESMA partição de folds do Onyx (`grouped_stratified_kfold`,
cfg.seed) e pesos pareados por passo (w_pos(t) ∝ n_neg(t), w_neg(t) ∝ n_pos(t), como o R1). Saída: OOF
do segundo estágio -> v-EMA do B3 -> R0 contra o B0 com v-EMA.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold


def _logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default="artifacts/models/oof_b0_bag4.parquet", help="OOF do Onyx SEM v-EMA")
    ap.add_argument("--colunas", nargs="+",
                    default=["fam_escala_desce", "fam_escala_sobe", "fam_cauda", "fam_forma4", "fam_correlacao",
                             "fam_arch", "fam_media", "sr_cal"])
    ap.add_argument("--out", default="artifacts/models/oof_st1_bag4.parquet")
    ap.add_argument("--num-leaves", type=int, default=15)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed-folds", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    oof = pd.read_parquet(args.oof)
    sc = pd.read_parquet("artifacts/surpresa/scores.parquet", columns=["id", "t"] + args.colunas)
    d = oof.merge(sc, on=["id", "t"], how="left", validate="one_to_one").reset_index(drop=True)
    X = pd.DataFrame({"onyx": _logit(d["oof_pred"].to_numpy()), "t": d["t"].to_numpy(dtype=float)})
    for c in args.colunas:
        X[c] = d[c].to_numpy(dtype=float)
    y = d["y"].to_numpy()
    g = d.groupby("t")["y"]
    n_pos = d["t"].map(g.sum()); n_neg = d["t"].map(g.size() - g.sum())
    w = np.where(y == 1, n_neg, n_pos).astype(float)
    w = w / w.mean()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=args.num_leaves, min_data_in_leaf=2000,
                  feature_fraction=1.0, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1,
                  seed=42, num_threads=args.threads, deterministic=True, force_row_wise=True)
    pred = np.full(len(d), np.nan)
    seed = args.seed_folds if args.seed_folds is not None else cfg.seed
    for tr, va in grouped_stratified_kfold(d[["id", "t", "y"]], cfg.lightgbm.n_folds, seed):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr], weight=w[tr]), args.rounds)
        pred[va] = m.predict(X.iloc[va])
    assert not np.isnan(pred).any()
    out = d[["id", "t", "y"]].copy()
    out["oof_pred"] = pred
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print("gravado", args.out)


if __name__ == "__main__":
    main()
