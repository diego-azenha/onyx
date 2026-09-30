#!/usr/bin/env python
"""G2 (knowledge/frentes/roteiro-30-09/G2-auto-referencia.md): cada série como controle de si mesma.
s'(t) = logit(t) - alpha * base(t), com base(t) = (a) mínimo corrente do logit até t, (b) média do logit
nos primeiros k passos online (k=10/25, disponível só depois de k), (c) média corrente. Causal. alpha
ajustado em ids pares e aplicado nos ímpares, e vice-versa. Sobre o OOF do E1 (grade thin; os
agregados correntes usam só os passos retidos, aproximação do caminho de produção)."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

ALFAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]


def main() -> None:
    arq = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/oof_e1_vema_bag4.parquet"
    d = pd.read_parquet(arq).sort_values(["id", "t"]).reset_index(drop=True)
    p = np.clip(d["oof_pred"].to_numpy(), 1e-9, 1 - 1e-9)
    d["lg"] = np.log(p) - np.log1p(-p)
    g = d.groupby("id")["lg"]
    bases = {"min_corrente": g.cummin().to_numpy(), "media_corrente": (g.cumsum() / (g.cumcount() + 1)).to_numpy()}
    for k in (10, 25):
        prim = d[d["t"] <= k].groupby("id")["lg"].mean()
        b = d["id"].map(prim).to_numpy()
        bases[f"media_primeiros_{k}"] = np.where(d["t"].to_numpy() > k, b, d["lg"].to_numpy())  # antes de k: sem ajuste
    w = board_grid_multipliers("data/y_train.parquet", d["t"].to_numpy())
    t, y, lg = d["t"].to_numpy(), d["y"].to_numpy(), d["lg"].to_numpy()
    par = (d["id"] % 2 == 0).to_numpy()
    rel = {"base": weighted_ts_auc(t, y, lg, w)}
    for nome, b in bases.items():
        novo = lg.copy(); escolha = {}
        for lado, aj, ap in (("par", par, ~par), ("impar", ~par, par)):
            curva = {a: weighted_ts_auc(t[aj], y[aj], lg[aj] - a * b[aj], w) for a in ALFAS}
            a = max(curva, key=curva.get); escolha[lado] = a
            novo[ap] = lg[ap] - a * b[ap]
        rel[nome] = {"alfa": escolha, "tsauc": weighted_ts_auc(t, y, novo, w)}
        rel[nome]["delta"] = rel[nome]["tsauc"] - rel["base"]
        print(nome, json.dumps(rel[nome]), flush=True)


if __name__ == "__main__":
    main()
