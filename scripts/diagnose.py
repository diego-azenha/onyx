#!/usr/bin/env python
"""CLI fina: gera relatório de diagnóstico local (plano §9.1) — curvas de treino, importância de
features, distribuição de score. NÃO estima TS-AUC (plano §9.0)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.diagnostics import feature_importance_report, score_distribution_report, training_curves
from sbrt.model.predict import ModelEnsemble


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--model", default="artifacts/models/v1")
    parser.add_argument("--out-dir", default="artifacts/reports")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensemble = ModelEnsemble.load(args.model)
    rows = pd.read_parquet(args.rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_curves(ensemble.fold_evals, out_dir / "training_curves.png")
    feature_importance_report(ensemble, out_dir / "feature_importance.csv")

    X = rows[list(ensemble.feature_order)].to_numpy(dtype=np.float32)
    preds = np.mean(
        [b.predict(X, num_threads=cfg.lightgbm.predict_num_threads) for b in ensemble.boosters], axis=0
    )
    score_distribution_report(rows, preds, out_dir / "score_distribution.png")

    print(f"relatórios gravados em {out_dir}/ (curvas de treino, importância, distribuição — não é TS-AUC, §9.0)")


if __name__ == "__main__":
    main()
