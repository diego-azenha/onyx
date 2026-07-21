#!/usr/bin/env python
"""R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): CLI fina, espelha scripts/train.py mas treina o
braço de RANKING por grupo t (model/train.py:train_rank) -- membro PARALELO do ensemble binário,
salvo num diretório separado. Compare o OOF resultante contra o binário com scripts/compare_oof.py
(R0) e combine os dois com scripts/combine_oof.py antes de decidir adotar qualquer braço."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.train import train_rank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--out", default="artifacts/models/v1_rank")
    parser.add_argument("--oof-out", default=None, help="default: <out>/../oof_<basename(out)>.parquet")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = pd.read_parquet(args.rows)
    ensemble, oof_pred = train_rank(rows, cfg, progress=True)
    ensemble.save(args.out)
    print(f"ensemble (rank) salvo em {args.out} ({len(ensemble.boosters)} folds, {len(ensemble.feature_order)} features)")

    out_dir = Path(args.out)
    oof_out = Path(args.oof_out) if args.oof_out else out_dir.parent / f"oof_{out_dir.name}.parquet"
    oof_df = rows[["id", "t", "y"]].copy()
    oof_df["oof_pred"] = np.asarray(oof_pred, dtype=np.float64)
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_out)
    print(f"predições out-of-fold (rank) salvas em {oof_out} ({len(oof_df)} linhas)")


if __name__ == "__main__":
    main()
