#!/usr/bin/env python
"""G1 (knowledge/frentes/roteiro-30-09/G1-calibracao-por-grupo.md): a TS-AUC ordena séries de TIPOS
diferentes no mesmo passo. Se a escala do score do Onyx difere entre tipos (grupos pelo perfil do
histórico: ACF lag 1 × curtose), a ordenação conjunta perde. Recalibração cruzada por grupo × faixa de
t: logística por célula sobre o logit do OOF, ajustada nos ids pares e aplicada nos ímpares, e
vice-versa. Só usa o histórico (causal) para definir o grupo."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc


def main() -> None:
    arq = sys.argv[1] if len(sys.argv) > 1 else "artifacts/models/oof_e1_vema_bag4.parquet"
    d = pd.read_parquet(arq)
    perfil = pd.read_parquet("artifacts/surpresa/perfil_series.parquet")
    acf = pd.cut(perfil["acf1"], [-1.1, -0.3, -0.05, 0.05, 0.3, 0.7, 1.1], labels=False)
    kurt = pd.cut(perfil["kurt"], [-10, 0.3, 2, 10, 1e9], labels=False)
    grupo = (acf.astype(int) * 10 + kurt.astype(int))
    d["g"] = d["id"].map(grupo)
    d["tb"] = pd.cut(d["t"], [0, 50, 150, 400, 10**6], labels=False)
    p = np.clip(d["oof_pred"].to_numpy(), 1e-9, 1 - 1e-9)
    d["lg"] = np.log(p) - np.log1p(-p)
    w = board_grid_multipliers("data/y_train.parquet", d["t"].to_numpy())
    par = (d["id"] % 2 == 0).to_numpy()
    novo = d["lg"].to_numpy().copy()
    for aj, ap in ((par, ~par), (~par, par)):
        A = d[aj]
        for (g, tb), cel in A.groupby(["g", "tb"]):
            if cel["y"].nunique() < 2 or len(cel) < 500:
                continue
            m = LogisticRegression(C=1.0, max_iter=300).fit(cel[["lg"]].to_numpy(), cel["y"].to_numpy())
            sel = ap & (d["g"] == g).to_numpy() & (d["tb"] == tb).to_numpy()
            novo[sel] = m.decision_function(d.loc[sel, ["lg"]].to_numpy())
    t, y = d["t"].to_numpy(), d["y"].to_numpy()
    rel = {"arquivo": arq, "base": weighted_ts_auc(t, y, d["lg"].to_numpy(), w),
           "calibrado_por_grupo": weighted_ts_auc(t, y, novo, w), "n_grupos": int(d["g"].nunique())}
    rel["delta"] = rel["calibrado_por_grupo"] - rel["base"]
    out = d[["id", "t", "y"]].copy(); out["oof_pred"] = novo
    out.to_parquet("artifacts/roteiro/g1_calibrado.parquet", index=False)
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
