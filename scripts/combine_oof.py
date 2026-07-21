#!/usr/bin/env python
"""R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): combina os OOF dos dois braços (binário-R1 e rank)
por RANK-AVERAGE -- média do percentil intra-t de cada modelo, não a média crua dos scores. Os dois
braços vivem em escalas diferentes (probabilidade calibrada vs. score de ranking cru); percentil
intra-t é a forma de combiná-los que já respeita a única coisa que a TS-AUC mede: ordem relativa
dentro do mesmo passo (parecer §3.1). Isto é diagnóstico OFFLINE (usa a seção transversal completa
de cada t, indisponível em tempo real) -- o combinador IMPLANTÁVEL é a média simples de
`predict_one`, `model/predict.py:CombinedModelEnsemble`.

Uso típico (parecer §6-R3, "três braços"): rode isto para produzir o terceiro braço e julgue os três
(binário-R1, rank, rank-average) com scripts/compare_oof.py entre pares."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _within_t_percentile(df: pd.DataFrame, score_col: str) -> pd.Series:
    return df.groupby("t")[score_col].rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary-oof", required=True)
    parser.add_argument("--rank-oof", required=True)
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col", default="oof_pred")
    parser.add_argument("--out", default="artifacts/models/oof_combined.parquet")
    args = parser.parse_args()

    binary_df = pd.read_parquet(args.binary_oof).rename(columns={args.score_col: "score_binary"})
    rank_df = pd.read_parquet(args.rank_oof).rename(columns={args.score_col: "score_rank"})

    merged = binary_df[["id", "t", args.label_col, "score_binary"]].merge(
        rank_df[["id", "t", "score_rank"]], on=["id", "t"], how="inner"
    )

    merged["pctl_binary"] = _within_t_percentile(merged, "score_binary")
    merged["pctl_rank"] = _within_t_percentile(merged, "score_rank")
    merged["oof_pred"] = 0.5 * (merged["pctl_binary"] + merged["pctl_rank"])

    out_df = merged[["id", "t", args.label_col, "oof_pred"]].rename(columns={args.label_col: "y"})
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"combinado (rank-average) salvo em {out_path} ({len(out_df)} linhas)")


if __name__ == "__main__":
    main()
