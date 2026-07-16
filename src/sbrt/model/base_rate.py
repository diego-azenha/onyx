"""Curva de taxa-base empírica p_hat(t) (plano_acao_v1_para_v2.md §4, ação A2).

y_t = 1{tau<=t} tem uma taxa-base que cresce fortemente com t (~7.6% em t<=50 até ~39.7% em t>400,
plano_acao_v1_para_v2.md §1.1) -- uma amplitude de ~2 em log-odds, previsível a partir de t sozinho.
Por invariância C1 (plano_structural_break_realtime.md §1.2), um componente de score que depende só
de t desloca todas as séries vivas igualmente no mesmo passo e é NEUTRO para a TS-AUC -- mas domina
a logloss binária/AUC de linha usadas para treinar e parar o LightGBM. Esta curva vira `init_score`
(plano §8.3 revisado) para que o modelo aprenda só o resíduo: a discriminação transversal que a
métrica de fato mede.

Ajustada UMA VEZ sobre o dataset de treino completo (não por fold) -- simplificação documentada: a
curva agrega milhares de séries sem usar identidade de série nenhuma, então o vazamento marginal de
incluir ~20% de linhas de validação no ajuste é desprezível frente ao que ela corrige.
"""
from __future__ import annotations

import numpy as np


def fit_base_rate_curve(t: np.ndarray, y: np.ndarray, bin_width: int = 20, pseudo_count: float = 10.0) -> dict:
    """p_hat(t) por bins de largura `bin_width`, com suavização aditiva (pseudo_count em direção a
    0.5) para bins esparsos em t alto não colapsarem para 0/1. Retorna centros e taxas para
    interpolação linear em `predict_base_rate_logit`."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    max_t = int(np.ceil(t.max())) if len(t) else 1
    edges = np.arange(0, max_t + bin_width, bin_width, dtype=np.float64)
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)

    centers, rates = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        pos = float(y[mask].sum())
        rate = (pos + pseudo_count * 0.5) / (n + pseudo_count)
        centers.append(float((edges[b] + edges[b + 1]) / 2.0))
        rates.append(rate)

    return {"centers": centers, "rates": rates}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def predict_base_rate_logit(t: np.ndarray, curve: dict) -> np.ndarray:
    """logit(p_hat(t)) via interpolação linear entre os centros ajustados; constante fora do
    intervalo (mantém comportamento definido nas bordas)."""
    t = np.asarray(t, dtype=np.float64)
    centers = np.asarray(curve["centers"], dtype=np.float64)
    rates = np.asarray(curve["rates"], dtype=np.float64)
    rate_at_t = np.interp(t, centers, rates)
    return _logit(rate_at_t)
