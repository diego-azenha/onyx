#!/usr/bin/env python
"""Passo 4 do roteiro de 30/09 (knowledge/frentes/roteiro-30-09/P4-especialistas.md): especialistas somados
ao score do E1 com DOIS pesos, cross-fit por paridade de id. logit(E1) + w1*desce/dp + w2*sobe/dp, grade
de pesos ajustada numa metade e aplicada na outra. Grava o OOF combinado para o compare_oof."""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

GRADE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    oof = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet")
    sc = pd.read_parquet("artifacts/surpresa/scores.parquet", columns=["id", "t", "fam_escala_desce", "fam_escala_sobe"])
    d = oof.merge(sc, on=["id", "t"], how="left", validate="one_to_one")
    p = np.clip(d["oof_pred"].to_numpy(), 1e-9, 1 - 1e-9)
    base = np.log(p) - np.log1p(-p)
    zd = d["fam_escala_desce"].to_numpy() / d["fam_escala_desce"].std()
    zs = d["fam_escala_sobe"].to_numpy() / d["fam_escala_sobe"].std()
    w = board_grid_multipliers("data/y_train.parquet", d["t"].to_numpy())
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    par = d["id"].to_numpy() % 2 == 0
    comb = np.empty_like(base)
    rel = {}
    for nome, aj, ap in (("par->impar", par, ~par), ("impar->par", ~par, par)):
        curva = {(a, b): weighted_ts_auc(t[aj], y[aj], base[aj] + a * zd[aj] + b * zs[aj], w)
                 for a, b in itertools.product(GRADE, GRADE)}
        (a, b) = max(curva, key=curva.get)
        rel[nome] = {"w_desce": a, "w_sobe": b, "ganho_no_ajuste": curva[(a, b)] - curva[(0.0, 0.0)]}
        comb[ap] = base[ap] + a * zd[ap] + b * zs[ap]
    out = d[["id", "t", "y"]].copy()
    out["oof_pred"] = comb
    out.to_parquet("artifacts/roteiro/p4_e1_especialistas.parquet", index=False)
    rel["tsauc_e1"] = weighted_ts_auc(t, y, base, w)
    rel["tsauc_combinado"] = weighted_ts_auc(t, y, comb, w)
    print(json.dumps(rel, indent=2, default=float))


if __name__ == "__main__":
    main()
