"""Pesos de linha alinhados à métrica (plano §8.2). w(t) = n_pos(t)*n_neg(t) / n_alive(t), medido
EMPIRICAMENTE no próprio conjunto de treino (tau e T conhecidos) — não a fórmula idealizada —,
multiplicado pelo fator de thinning e normalizado para média 1."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_row_weights(rows: pd.DataFrame, cfg) -> np.ndarray:
    counts = rows.groupby("t")["y"].agg(n_pos="sum", n_alive="count")
    counts["n_neg"] = counts["n_alive"] - counts["n_pos"]
    counts["w_t"] = counts["n_pos"] * counts["n_neg"] / counts["n_alive"].clip(lower=1)

    w_t_map = counts["w_t"].to_dict()
    base_w = rows["t"].map(w_t_map).to_numpy(dtype=np.float64)
    w = base_w * rows["thin_weight"].to_numpy(dtype=np.float64)

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w
