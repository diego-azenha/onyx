#!/usr/bin/env python
"""Reproduz a célula 'Computing TS-AUC locally' do quickstarter_notebook.ipynb:
AUC ponderada por passo online, usando o rótulo local reduzido (data/y_test.reduced.parquet)
contra prediction/prediction.parquet gerado por `crunch test`."""
from __future__ import annotations

import argparse

import pandas as pd
from sklearn.metrics import roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", default="prediction/prediction.parquet")
    parser.add_argument("--y-test", default="data/y_test.reduced.parquet")
    args = parser.parse_args()

    prediction = pd.read_parquet(args.prediction)
    y_test = pd.read_parquet(args.y_test)

    merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)
    merged["time_online"] = merged.groupby("id").cumcount()

    weighted_auc_sum = 0.0
    total_weight = 0.0
    for _, group in merged.groupby("time_online"):
        labels = group["target"].values
        scores = group["prediction"].values

        n_pos = int(labels.sum())
        n_neg = int((1 - labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue

        auc_t = float(roc_auc_score(labels, scores))
        weight = float(n_pos * n_neg)

        weighted_auc_sum += weight * auc_t
        total_weight += weight

    ts_auc = weighted_auc_sum / total_weight if total_weight > 0 else 0.5
    print(f"Local TS-AUC: {ts_auc:.4f}")


if __name__ == "__main__":
    main()
