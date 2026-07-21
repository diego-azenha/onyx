"""TS-AUC ponderada por passo (docs/PLANO_TECNICO.md §1: AUC_t agregada com peso n_pos(t)*n_neg(t))
— implementação vetorizada via rank médio intra-grupo (estatística de Mann-Whitney), equivalente a
`roc_auc_score` por grupo em laço Python (scripts/oof_ts_auc_by_bucket.py, scripts/local_ts_auc.py)
mas ordens de magnitude mais rápida — necessária onde isto roda centenas/milhares de vezes: o `feval`
de treino por rodada de boosting (model/train.py, R2) e o bootstrap pareado (scripts/compare_oof.py,
R0). Nunca usada como estimador de leaderboard (plano §9.0) — é critério interno de fold/diagnóstico
relativo."""
from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_ts_auc(t: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    if len(t) == 0:
        return float("nan")
    df = pd.DataFrame({"t": t, "y": y, "s": score})
    df["rank"] = df.groupby("t")["s"].rank(method="average")
    g = df.groupby("t")
    n = g["y"].size()
    n_pos = g["y"].sum()
    n_neg = n - n_pos
    r_pos = df.loc[df["y"] == 1].groupby("t")["rank"].sum().reindex(n.index, fill_value=0.0)

    valid = (n_pos > 0) & (n_neg > 0)
    if not valid.any():
        return float("nan")
    auc_t = (r_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / (n_pos[valid] * n_neg[valid])
    w = (n_pos[valid] * n_neg[valid]).astype(np.float64)
    tot = w.sum()
    return float((auc_t * w).sum() / tot) if tot > 0 else float("nan")
