#!/usr/bin/env python
"""R6-ii (docs/PARECER_AUDITORIA_ONYX.md §6-R6): resposta ao degrau OOF alinhada em tau (A4,
plano_acao_v1_para_v2.md) como artefato PADRÃO por versão de modelo -- média/mediana do score OOF
em função de (t - tau) para séries com quebra, agregada sobre todas as séries. `tau` (o primeiro t
com y=1) é derivado do próprio parquet de OOF -- não precisa de y_train_index.parquet.

Diagnóstico visual/tabular de "o score sobe de fato depois da quebra e decai depois de estabilizar
o alarme" -- NUNCA um substituto de TS-AUC (plano §9.0)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_step_response(oof: pd.DataFrame, label_col: str = "y", score_col: str = "oof_pred") -> pd.DataFrame:
    has_break = oof.groupby("id")[label_col].transform("max") > 0
    broken = oof.loc[has_break].copy()
    if broken.empty:
        return pd.DataFrame(columns=["offset", "mean_score", "median_score", "n"])

    tau_first_t = broken.loc[broken[label_col] == 1].groupby("id")["t"].min()
    broken["tau_first_t"] = broken["id"].map(tau_first_t)
    broken["offset"] = broken["t"] - broken["tau_first_t"]

    curve = broken.groupby("offset")[score_col].agg(mean_score="mean", median_score="median", n="count")
    return curve.reset_index().sort_values("offset")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof", default="artifacts/models/oof_v1.parquet")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col", default="oof_pred")
    parser.add_argument("--out", default="artifacts/reports/oof_step_response.csv")
    parser.add_argument("--offset-range", type=int, default=300, help="imprime a curva só até +-N no console")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof)
    curve = compute_step_response(oof, args.label_col, args.score_col)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out_path, index=False)

    print(f"resposta ao degrau (n={oof['id'].nunique()} séries, "
          f"{(oof.groupby('id')[args.label_col].max() > 0).sum()} com quebra)\n")
    window = curve[(curve["offset"] >= -20) & (curve["offset"] <= args.offset_range)]
    print(window.to_string(index=False))
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
