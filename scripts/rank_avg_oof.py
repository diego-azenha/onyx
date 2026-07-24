#!/usr/bin/env python
"""Rank-average de N OOFs em espaço de PERCENTIL intra-t — a agregação que casa com a métrica (TS-AUC
= rank de Mann-Whitney dentro do passo, parecer §3.1). Generaliza `scripts/combine_oof.py` (que faz o
mesmo para 2 membros) para N entradas, com pesos opcionais.

T0.2 (BRAINSTORM_RUPTURA_V2.md §1.1): sobre K sementes HOMOGÊNEAS o efeito esperado é ~0 (as marginais
de score por passo são quase idênticas ⇒ o percentil intra-t é quase a mesma transformação monótona
para todas, e média-após-transformação-comum ≈ transformação-da-média; o desvio é só Jensen sobre
discordâncias). Serve para registrar e fechar o eixo homogêneo; o valor real está em membros
HETEROGÊNEOS (A2). Diagnóstico OFFLINE (usa a seção transversal completa de cada t)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", nargs="+", required=True, help="parquets OOF (mesmas linhas id/t/y)")
    p.add_argument("--weights", nargs="*", type=float, default=None, help="pesos por membro (default: iguais)")
    p.add_argument("--score-col", default="oof_pred")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = pd.read_parquet(args.inputs[0])[["id", "t", "y", args.score_col]].copy()
    w = args.weights or [1.0] * len(args.inputs)
    if len(w) != len(args.inputs):
        raise SystemExit("nº de pesos != nº de entradas")
    w = np.asarray(w, dtype=np.float64) / float(np.sum(w))

    acc = np.zeros(len(base), dtype=np.float64)
    for wi, path in zip(w, args.inputs):
        df = pd.read_parquet(path)
        if len(df) != len(base) or not df["id"].equals(base["id"]) or not df["t"].equals(base["t"]):
            raise SystemExit(f"{path}: linhas não casam com {args.inputs[0]} (mesmos folds/dataset exigidos)")
        pctl = df.groupby("t")[args.score_col].rank(method="average", pct=True).to_numpy(dtype=np.float64)
        acc += wi * pctl

    out = base[["id", "t", "y"]].copy()
    out["oof_pred"] = acc
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"rank-average de {len(args.inputs)} membros (pesos {np.round(w,3).tolist()}) salvo em {args.out}")


if __name__ == "__main__":
    main()
