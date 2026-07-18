#!/usr/bin/env python
"""Decompoe a TS-AUC (mesma formula de local_ts_auc.py: AUC ponderada por passo online, peso
n_pos*n_neg) sobre as predicoes out-of-fold do treino, por bucket de t -- diagnostico, nao
substituto do score oficial (README.md, docs/PLANO_TECNICO.md secao 9.0; docs/DIAGNOSTICO_TS_AUC.md)."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def weighted_ts_auc(df: pd.DataFrame, label_col: str, score_col: str) -> float:
    wsum, tot = 0.0, 0.0
    for _, g in df.groupby("t"):
        y = g[label_col].to_numpy()
        s = g[score_col].to_numpy()
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        auc = roc_auc_score(y, s)
        w = n_pos * n_neg
        wsum += w * auc
        tot += w
    return wsum / tot if tot > 0 else 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="artifacts/models/oof_v1.parquet")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col", default="oof_pred")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof)
    t_bins = [0, 50, 150, 400, np.inf]
    t_labels = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]
    oof["t_bucket"] = pd.cut(oof["t"], bins=t_bins, labels=t_labels, right=True)

    overall = weighted_ts_auc(oof, args.label_col, args.score_col)
    print(f"TS-AUC OOF geral: {overall:.4f}")
    print("\npor bucket de t:")
    for lbl in t_labels:
        sub = oof[oof["t_bucket"] == lbl]
        auc = weighted_ts_auc(sub, args.label_col, args.score_col)
        print(f"  {lbl:12s} {auc:.4f}  (n={len(sub)})")


if __name__ == "__main__":
    main()
