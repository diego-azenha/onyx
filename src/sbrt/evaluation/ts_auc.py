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


def weighted_ts_auc(
    t: np.ndarray, y: np.ndarray, score: np.ndarray, w_mult: dict[int, float] | None = None
) -> float:
    """`w_mult` (A2, CAMPANHA_POLIMENTO.md) — multiplicador por passo aplicado a `w_t`.

    O OOF vive na grade com thinning (`configs/default.yaml:thinning`), o board avalia TODOS os
    passos. `AUC_t` em cada passo retido e exato (o thinning descarta passos, nunca series), mas o
    AGREGADO nao e: cada `t` retido entra com o proprio `w_t` em vez da massa do bloco que ele
    representa, o que subamostra 401+ por ~2x contra 151-400. MEDIDO: a TS-AUC agregada sobe
    +0,0139 ao trocar de grade, e os pesos por bucket mudam de (8,1 / 26,6 / 48,7 / 16,5)% para
    (3,9 / 17,5 / 46,7 / 31,9)%. `w_mult[t]` = massa do board atribuida ao passo `t` dividida pelo
    `w_t` dele; passar isso reproduz a ponderacao do board. `None` = comportamento historico.

    E um multiplicador, e nao um peso absoluto, de proposito: no bootstrap pareado as contagens
    `n_pos`/`n_neg` mudam a cada reamostragem de series, e o multiplicador e propriedade da GRADE
    (fixa), nao da amostra -- entao os dois se compoem corretamente.
    """
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
    if w_mult is not None:
        # `.map` vetoriza a busca; um laço Python aqui roda 5x por réplica de bootstrap (agregado +
        # 4 buckets) e aparece no perfil de `compare_oof.py` com 300 réplicas.
        w = w * w.index.to_series().map(w_mult).fillna(1.0).to_numpy(dtype=np.float64)
    tot = w.sum()
    return float((auc_t * w).sum() / tot) if tot > 0 else float("nan")


def board_grid_multipliers(y_train_path: str, oof_steps: np.ndarray) -> dict[int, float]:
    """Multiplicadores `w_mult` da grade do board para uma grade de OOF (A2).

    `y_train.parquet` e a grade cheia (todos os passos online de todas as series), entao os `w_t`
    verdadeiros saem dele sem depender de nenhuma feature. Cada passo do board e atribuido ao passo
    do OOF mais proximo; `w_mult[t]` = (massa do board do bloco de `t`) / (w_t do board em `t`).
    Sai ~1 em t<=100, ~2 em 101-400 e ~4 em 401+, com as bordas e o decaimento da populacao viva
    tratados exatamente em vez de por regra de tres.
    """
    y = pd.read_parquet(y_train_path).reset_index().sort_values(["id", "time"])
    y["t"] = y.groupby("id").cumcount() + 1
    g = y.groupby("t")["target"]
    n, n_pos = g.size(), g.sum()
    w_board = pd.Series((n_pos * (n - n_pos)).astype(np.float64).values, index=n.index)
    w_board = w_board[w_board > 0]

    steps = np.sort(np.unique(np.asarray(oof_steps, dtype=np.int64)))
    nearest = np.abs(w_board.index.to_numpy()[:, None] - steps[None, :]).argmin(axis=1)
    block = np.zeros(len(steps), dtype=np.float64)
    np.add.at(block, nearest, w_board.to_numpy())
    own = w_board.reindex(steps).to_numpy()
    return {int(s): float(b / o) for s, b, o in zip(steps, block, own) if o > 0}
