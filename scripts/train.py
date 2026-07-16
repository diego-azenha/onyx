#!/usr/bin/env python
"""CLI fina: treina o ensemble LightGBM (plano §8.3) a partir de data/processed/train_rows.parquet.
Salva também as predições out-of-fold (id, t, y, tau_index-derivado, oof_pred) para diagnósticos
(A4: resposta ao degrau alinhada em tau, plano_acao_v1_para_v2.md)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--out", default="artifacts/models/v1")
    parser.add_argument("--oof-out", default=None, help="default: <out>/../oof_<basename(out)>.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = pd.read_parquet(args.rows)
    weights = compute_row_weights(rows, cfg)
    ensemble, oof_pred = train_ensemble(rows, weights, cfg, progress=True)
    ensemble.save(args.out)
    print(f"ensemble salvo em {args.out} ({len(ensemble.boosters)} folds, {len(ensemble.feature_order)} features)")

    out_dir = Path(args.out)
    oof_out = Path(args.oof_out) if args.oof_out else out_dir.parent / f"oof_{out_dir.name}.parquet"
    oof_df = rows[["id", "t", "y"]].copy()
    oof_df["oof_pred"] = np.asarray(oof_pred, dtype=np.float64)
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_out)
    print(f"predições out-of-fold salvas em {oof_out} ({len(oof_df)} linhas)")


if __name__ == "__main__":
    main()
