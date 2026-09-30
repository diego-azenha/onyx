#!/usr/bin/env python
"""S7 (knowledge/frentes/teto-offline/S7-limiar.md): especialistas por tipo combinados com LIMIAR.
s = z(E1) + a · relu(z(esp) − c), com z padronizado dentro de cada passo t. O especialista só conta quando
a evidência dele é extrema, sem mexer nas quebras `fixa` (que são "anti-queda") nem nos negativos comuns.
(a, c) por especialista escolhidos por cross-fit de paridade de id."""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

SEM = [int(s) for s in (sys.argv[1:] or ["777", "101"])]
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
GA, GC = (0.0, 0.5, 1.0, 2.0), (1.0, 1.5, 2.0, 2.5, 3.0)


def main() -> None:
    b = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"]).reset_index(drop=True)
    t, y = b["t"].to_numpy(), b["y"].to_numpy()
    df = pd.DataFrame({"t": t, "e1": lg(b["oof_pred"].to_numpy())})
    for T in ("desce", "sobe"):
        df[T] = lg(sum(pd.read_parquet(f"artifacts/models/oof_s6_{T}_s{s}.parquet").sort_values(["id", "t"])["oof_pred"].to_numpy() for s in SEM) / len(SEM))
    g = df.groupby("t")
    for c in ("e1", "desce", "sobe"):
        df[c + "z"] = (df[c] - g[c].transform("mean")) / g[c].transform("std")
    w = board_grid_multipliers("data/y_train.parquet", t)
    tipo = b["id"].map(pd.read_parquet("artifacts/roteiro/tipo_quebra.parquet")["tipo"]).to_numpy()
    par = (b["id"] % 2 == 0).to_numpy()
    e1z, dz, sz = df["e1z"].to_numpy(), df["descez"].to_numpy(), df["sobez"].to_numpy()
    f = lambda m, ad, cd, as_, cs: e1z[m] + ad * np.maximum(dz[m] - cd, 0) + as_ * np.maximum(sz[m] - cs, 0)
    comb = np.empty(len(t)); esc = {}
    for nome, aj, ap in (("par", par, ~par), ("impar", ~par, par)):
        melhor, arg = -1, None
        for ad, cd in itertools.product(GA, GC):                     # 1º o especialista de quedas
            a = weighted_ts_auc(t[aj], y[aj], f(aj, ad, cd, 0, 9), w)
            if a > melhor:
                melhor, arg = a, (ad, cd)
        ad, cd = arg; melhor2, arg2 = melhor, (0.0, 9.0)
        for as_, cs in itertools.product(GA[1:], GC):                # depois o de altas, condicionado
            a = weighted_ts_auc(t[aj], y[aj], f(aj, ad, cd, as_, cs), w)
            if a > melhor2:
                melhor2, arg2 = a, (as_, cs)
        esc[nome] = {"a_desce": ad, "c_desce": cd, "a_sobe": arg2[0], "c_sobe": arg2[1], "ganho_no_ajuste": melhor2 - weighted_ts_auc(t[aj], y[aj], e1z[aj], w)}
        comb[ap] = f(ap, ad, cd, *arg2)
    rel = {"e1": weighted_ts_auc(t, y, e1z, w), "limiar_crossfit": weighted_ts_auc(t, y, comb, w), "escolhas": esc}
    for tp in ("fixa", "sobe", "desce"):
        m = (tipo == "neg") | (tipo == tp)
        rel[f"por_tipo_{tp}"] = {"e1": weighted_ts_auc(t[m], y[m], e1z[m], w), "limiar": weighted_ts_auc(t[m], y[m], comb[m], w)}
    pd.DataFrame({"id": b["id"], "t": t, "y": y, "oof_pred": comb}).to_parquet("artifacts/roteiro/s7_limiar.parquet", index=False)
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
