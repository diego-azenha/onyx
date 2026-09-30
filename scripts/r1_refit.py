#!/usr/bin/env python
"""R1-refit (knowledge/frentes/teto-offline/R1-refit-completo.md): na produção, o modelo é a média dos 5
modelos de fold, cada um treinado com 80% das séries. Um refit com 100% das séries vale mais?

Holdout EXTERNO: séries com id % 5 == 0 ficam fora de tudo. Nas outras 80%: (a) `train()` normal, cuja
média dos 5 boosters de fold é o que a produção usa hoje; (b) refit com todas essas linhas, com os
mesmos parâmetros (lidos do booster de fold) e round(média das best_iteration × mult) rodadas. As duas
predições são avaliadas no holdout (TS-AUC, grade do board). Receita do E1 (linear_tree + extra_trees + X1).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc
from sbrt.model.base_rate import predict_base_rate_logit
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--mult", type=float, default=1.0)
    ap.add_argument("--out", default="artifacts/reports/r1_refit")
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, boost_seed=args.seed, linear_tree=True, extra_trees=True))
    rows = pd.read_parquet("data/processed/train_rows.parquet")
    hold = (rows["id"] % 5 == 0).to_numpy()
    tr_rows, ho_rows = rows[~hold].reset_index(drop=True), rows[hold].reset_index(drop=True)
    del rows
    w = compute_row_weights(tr_rows, cfg, detectability_mode="soft",
                            detectability_path="artifacts/reports/detectability.csv",
                            detect_floor=cfg.weights.detect_floor)
    ens, _ = train_ensemble(tr_rows, w, cfg, progress=True)
    cols = list(ens.feature_order)
    X_ho = ho_rows[cols].to_numpy(dtype=np.float32)
    init_ho = predict_base_rate_logit(ho_rows["t"].to_numpy(dtype=np.float64), ens.base_rate_curve)
    raw_folds = np.mean([b.predict(X_ho, raw_score=True) for b in ens.boosters], axis=0)
    iters = [b.best_iteration if b.best_iteration > 0 else b.current_iteration() for b in ens.boosters]
    n_full = max(1, int(round(np.mean(iters) * args.mult)))
    params = dict(ens.boosters[0].params)
    for k in ("early_stopping_round", "early_stopping_rounds", "metric", "num_iterations", "num_boost_round"):
        params.pop(k, None)
    X_tr = tr_rows[cols].to_numpy(dtype=np.float32)
    init_tr = predict_base_rate_logit(tr_rows["t"].to_numpy(dtype=np.float64), ens.base_rate_curve)
    full = lgb.train(params, lgb.Dataset(X_tr, label=tr_rows["y"].to_numpy(), weight=w, init_score=init_tr),
                     num_boost_round=n_full)
    raw_full = full.predict(X_ho, raw_score=True)
    wm = board_grid_multipliers("data/y_train.parquet", ho_rows["t"].to_numpy())
    t, y = ho_rows["t"].to_numpy(), ho_rows["y"].to_numpy()
    res = {"seed": args.seed, "iters_folds": iters, "n_full": n_full,
           "tsauc_media_folds": weighted_ts_auc(t, y, raw_folds + init_ho, wm),
           "tsauc_refit_100": weighted_ts_auc(t, y, raw_full + init_ho, wm),
           "tsauc_folds_mais_refit": weighted_ts_auc(t, y, (raw_folds + raw_full) / 2 + init_ho, wm)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ho_rows["id"], "t": t, "y": y, "folds": raw_folds + init_ho, "full": raw_full + init_ho}) \
        .to_parquet(f"{args.out}/pred_s{args.seed}.parquet")
    Path(f"{args.out}/r1_s{args.seed}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res))


if __name__ == "__main__":
    main()
