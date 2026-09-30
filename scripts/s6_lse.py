#!/usr/bin/env python
"""S6: combinação BAYESIANA dos especialistas por tipo. Cada especialista estima as chances "quebra do tipo
k vs negativo", então P(quebra|x) ∝ Σ_k exp(logit_k + c_k): log-sum-exp, não uma logística linear nos logits.
c_k (prior por tipo) e o peso do E1 são ajustados por cross-fit de paridade de id."""
from __future__ import annotations

import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

SEM = [int(s) for s in (sys.argv[1:] or ["777", "101"])]
lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))


def main() -> None:
    base = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet").sort_values(["id", "t"]).reset_index(drop=True)
    t, y = base["t"].to_numpy(), base["y"].to_numpy()
    L = {}
    for T in ("sobe", "desce", "fixa"):
        acc = sum(pd.read_parquet(f"artifacts/models/oof_s6_{T}_s{s}.parquet").sort_values(["id", "t"])["oof_pred"].to_numpy() for s in SEM)
        L[T] = lg(acc / len(SEM))
    e1 = lg(base["oof_pred"].to_numpy())
    w = board_grid_multipliers("data/y_train.parquet", t)
    par = (base["id"] % 2 == 0).to_numpy()
    grade_c = (-1.0, -0.5, 0.0, 0.5)
    rel = {"e1": weighted_ts_auc(t, y, e1, w),
           "lse_puro": weighted_ts_auc(t, y, logsumexp(np.vstack([L["sobe"], L["desce"], L["fixa"]]), axis=0), w)}
    comb = np.empty(len(t)); comb2 = np.empty(len(t)); esc = {}
    for nome, aj, ap in (("par", par, ~par), ("impar", ~par, par)):
        melhor, arg = -1, None
        for cs, cd in itertools.product(grade_c, grade_c):          # c_fixa = 0 (referência)
            s = logsumexp(np.vstack([L["sobe"][aj] + cs, L["desce"][aj] + cd, L["fixa"][aj]]), axis=0)
            a = weighted_ts_auc(t[aj], y[aj], s, w)
            if a > melhor:
                melhor, arg = a, (cs, cd)
        cs, cd = arg
        lse = lambda m: logsumexp(np.vstack([L["sobe"][m] + cs, L["desce"][m] + cd, L["fixa"][m]]), axis=0)
        comb[ap] = lse(ap)
        # mistura com o E1 (padronizados), peso de uma grade
        z = lambda v, m: (v[m] - v[aj].mean()) / v[aj].std()
        lse_aj = lse(aj)
        cur = {b: weighted_ts_auc(t[aj], y[aj], (1 - b) * (e1[aj] - e1[aj].mean()) / e1[aj].std() + b * (lse_aj - lse_aj.mean()) / lse_aj.std(), w)
               for b in (0, .2, .4, .5, .6, .8, 1.0)}
        b = max(cur, key=cur.get)
        comb2[ap] = (1 - b) * (e1[ap] - e1[aj].mean()) / e1[aj].std() + b * (comb[ap] - lse_aj.mean()) / lse_aj.std()
        esc[nome] = {"c_sobe": cs, "c_desce": cd, "peso_lse_vs_e1": b}
    rel["lse_crossfit"] = weighted_ts_auc(t, y, comb, w)
    rel["e1_mais_lse_crossfit"] = weighted_ts_auc(t, y, comb2, w)
    rel["escolhas"] = esc
    pd.DataFrame({"id": base["id"], "t": t, "y": y, "oof_pred": comb2}).to_parquet("artifacts/roteiro/s6_e1_lse.parquet", index=False)
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
