#!/usr/bin/env python
"""Decompoe a TS-AUC (mesma formula de local_ts_auc.py: AUC ponderada por passo online, peso
n_pos*n_neg) sobre as predicoes out-of-fold do treino, por bucket de t -- diagnostico, nao
substituto do score oficial (README.md, docs/PLANO_TECNICO.md secao 9.0; docs/DIAGNOSTICO_TS_AUC.md).

A2 (CAMPANHA_POLIMENTO.md): por default reponderapara a grade do BOARD. O OOF vive na grade com
thinning (399 passos) e o board avalia todos os ~999; cada `t` retido entra com o proprio `w_t` em
vez da massa do bloco que representa, o que subamostra 401+ por ~2x. Foi ESTE script que produziu a
tabela de pesos por bucket do NOTAS_AGENTES.md §5, e ela estava errada: 401+ vale 31,9%, nao 16,5%.
`--grid thin` reproduz o comportamento antigo.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sbrt.evaluation.ts_auc import board_grid_multipliers


def weighted_ts_auc(df: pd.DataFrame, label_col: str, score_col: str,
                    w_mult: dict[int, float] | None = None) -> tuple[float, float]:
    """Devolve (TS-AUC, massa de peso) -- a massa e o que torna a tabela por bucket legivel como
    'quanto este bucket pesa no total', que era exatamente o numero errado antes do A2."""
    wsum, tot = 0.0, 0.0
    for t, g in df.groupby("t"):
        y = g[label_col].to_numpy()
        s = g[score_col].to_numpy()
        n_pos, n_neg = int(y.sum()), int((1 - y).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        w = n_pos * n_neg * (w_mult.get(int(t), 1.0) if w_mult else 1.0)
        wsum += w * roc_auc_score(y, s)
        tot += w
    return (wsum / tot if tot > 0 else 0.5), tot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", default="artifacts/models/oof_v1.parquet")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col", default="oof_pred")
    parser.add_argument("--grid", default="full", choices=["full", "thin"],
                        help="A2: 'full' pondera pela grade do BOARD (default); 'thin' reproduz a "
                             "ponderacao antiga, so para reauditar numeros historicos.")
    parser.add_argument("--y-train", default="data/y_train.parquet")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof)
    t_bins = [0, 50, 150, 400, np.inf]
    t_labels = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]
    oof["t_bucket"] = pd.cut(oof["t"], bins=t_bins, labels=t_labels, right=True)

    w_mult = board_grid_multipliers(args.y_train, oof["t"].to_numpy()) if args.grid == "full" else None

    overall, massa = weighted_ts_auc(oof, args.label_col, args.score_col, w_mult)
    print(f"TS-AUC OOF geral (grade={args.grid}): {overall:.4f}")
    print("\npor bucket de t:")
    print(f"  {'bucket':12s} {'TS-AUC':>7s} {'peso':>7s} {'n linhas':>10s}")
    for lbl in t_labels:
        sub = oof[oof["t_bucket"] == lbl]
        auc, w = weighted_ts_auc(sub, args.label_col, args.score_col, w_mult)
        print(f"  {lbl:12s} {auc:7.4f} {100 * w / massa:6.1f}% {len(sub):10d}")


if __name__ == "__main__":
    main()
