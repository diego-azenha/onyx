"""Divisão agrupada e estratificada por série (plano §9.4). GroupKFold por `id` é obrigatório:
linhas da mesma série são fortemente autocorrelacionadas — um split não agrupado infla o CV de forma
catastrófica (armadilha §13.6)."""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_series_meta(rows: pd.DataFrame) -> pd.DataFrame:
    """Um registro por id: rótulo da série (teve quebra?) e terço de tau (bucket 0/1/2 do primeiro
    t com y=1 relativo ao T observado da série, ou -1 se não houver quebra)."""
    g = rows.groupby("id")
    has_break = g["y"].max().astype(int)

    def _tau_bucket(sub: pd.DataFrame) -> int:
        pos = sub.loc[sub["y"] == 1, "t"]
        if pos.empty:
            return -1
        tau = pos.min()
        t_max = sub["t"].max()
        frac = tau / max(t_max, 1)
        return min(int(frac * 3), 2)

    tau_bucket = g.apply(_tau_bucket, include_groups=False)
    meta = pd.DataFrame({"has_break": has_break, "tau_bucket": tau_bucket})
    return meta


def grouped_stratified_kfold(meta: pd.DataFrame, k: int, seed: int) -> Iterator[tuple]:
    """`meta` = linhas por passo (uma linha por (id,t)), como produzido por model/dataset.py.
    Agrupado por id; estratificado por (rótulo da série, terço de tau). Retorna, por fold, posições
    0-based (não ids) em `meta` para treino/validação — prontas para indexar a matriz X."""
    series = build_series_meta(meta)
    strata = series["has_break"].astype(str) + "_" + series["tau_bucket"].astype(str)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    series_ids = series.index.to_numpy()

    id_to_positions = meta.groupby("id").indices

    for train_pos, valid_pos in skf.split(series_ids, strata):
        train_ids = series_ids[train_pos]
        valid_ids = series_ids[valid_pos]
        train_rows = np.concatenate([id_to_positions[i] for i in train_ids])
        valid_rows = np.concatenate([id_to_positions[i] for i in valid_ids])
        yield train_rows, valid_rows
