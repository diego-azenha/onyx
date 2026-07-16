"""Primitivas numéricas reaproveitadas por múltiplos state blocks (plano §4, docs/CONTRACTS.md).

NUNCA reimplementar Welford ou logsumexp localmente em outro arquivo — é a defesa contra a classe de
bug "duas implementações que divergem depois de 500 passos por causa de um `if` diferente"
(docs/PLANO_REPOSITORIO.md §1).
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence


def welford_update(n: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
    """Recursão de Welford (1962) — atualização estável de média e soma de quadrados."""
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2 + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def logsumexp(values: Sequence[float]) -> float:
    """log(sum(exp(values))), estável numericamente. Ignora -inf (candidatos mortos)."""
    finite_max = -math.inf
    for v in values:
        if v > finite_max:
            finite_max = v
    if finite_max == -math.inf:
        return -math.inf
    total = 0.0
    for v in values:
        if v != -math.inf:
            total += math.exp(v - finite_max)
    return finite_max + math.log(total)


@lru_cache(maxsize=4096)
def lgamma_cached(x: float) -> float:
    """math.lgamma com cache — plano §4.3: nu_n = nu0 + n_j cresce em passos inteiros, então os
    mesmos argumentos reaparecem massivamente entre candidatos e passos."""
    return math.lgamma(x)


def ewma_update(prev: float, x: float, lam: float) -> float:
    return (1.0 - lam) * prev + lam * x
