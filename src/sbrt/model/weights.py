"""Pesos de linha pareado-consistentes com a TS-AUC (parecer de auditoria §3.10/§4.4, roadmap R1).

A TS-AUC se reescreve como a fração de pares (positivo, negativo) do MESMO passo t corretamente
ordenados, agregada sobre todos os t (parecer §3.1). Nesse pool de pares, cada linha positiva de t
participa de n_neg(t) pares e cada linha negativa de n_pos(t) pares — logo o surrogate pontual
pareado-consistente dá peso ∝ n_neg(t) aos positivos e ∝ n_pos(t) aos negativos, e não o mesmo peso
às duas classes como antes (w(t) = n_pos(t)*n_neg(t)/n_alive(t) para TODA linha de t, sem distinguir
classe). O esquema antigo acerta a massa agregada por passo (∝ n_pos*n_neg) mas erra a partição
intra-passo (1:1 em vez de n_neg:n_pos) — em t<=50 isso dilui o gradiente dos positivos por ~12x,
exatamente no bucket onde a AUC medida é mais fraca (parecer §3.10).

Contagens suavizadas por pseudo-contagem (t com poucos positivos não vira peso quase-infinito) e a
razão w_pos(t)/w_neg(t) capada em `max_ratio` (t muito pequeno tem n_pos raro, n_neg~5000 — sem cap
isso troca viés por variância de gradiente, parecer §4.4). Multiplicado pelo fator de thinning e
normalizado para média 1, como antes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_row_weights(
    rows: pd.DataFrame, cfg, pseudo_count: float = 5.0, max_ratio: float = 50.0
) -> np.ndarray:
    counts = rows.groupby("t")["y"].agg(n_pos="sum", n_alive="count")
    counts["n_neg"] = counts["n_alive"] - counts["n_pos"]
    n_pos_s = counts["n_pos"] + pseudo_count
    n_neg_s = counts["n_neg"] + pseudo_count

    # w_pos(t) ~ n_neg_s (número de pares que cada positivo participa), w_neg(t) ~ n_pos_s,
    # capando a razão entre as duas em max_ratio (equivalente a clipar n_neg_s/n_pos_s em
    # [1/max_ratio, max_ratio] e manter a proporcionalidade exata dentro do cap).
    counts["w_t_pos"] = np.minimum(n_neg_s, max_ratio * n_pos_s)
    counts["w_t_neg"] = np.minimum(n_pos_s, max_ratio * n_neg_s)

    w_t_pos_map = counts["w_t_pos"].to_dict()
    w_t_neg_map = counts["w_t_neg"].to_dict()
    t_pos = rows["t"].map(w_t_pos_map).to_numpy(dtype=np.float64)
    t_neg = rows["t"].map(w_t_neg_map).to_numpy(dtype=np.float64)
    is_pos = rows["y"].to_numpy(dtype=bool)
    base_w = np.where(is_pos, t_pos, t_neg)

    w = base_w * rows["thin_weight"].to_numpy(dtype=np.float64)

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w
