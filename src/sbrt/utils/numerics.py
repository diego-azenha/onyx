"""Primitivas numéricas reaproveitadas por múltiplos state blocks (plano §4, docs/CONTRACTS.md).

NUNCA reimplementar Welford ou logsumexp localmente em outro arquivo — é a defesa contra a classe de
bug "duas implementações que divergem depois de 500 passos por causa de um `if` diferente"
(docs/PLANO_REPOSITORIO.md §1).
"""
from __future__ import annotations

import math
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


_LGAMMA_CACHE: dict = {}


def lgamma_cached(x: float) -> float:
    """math.lgamma com cache — plano §4.3: nu_n = nu0 + n_j cresce em passos inteiros, então os
    mesmos argumentos reaparecem massivamente entre candidatos e passos.

    Memo explícito em vez de `@lru_cache`: no notebook de submissão o pipeline é achatado em
    `__main__`, e `functools._lru_cache_wrapper` se serializa POR REFERÊNCIA (nome qualificado). O
    worker do loky (spawn) não tem esse nome no seu `__main__` e o `train()` paralelo morre com
    `Can't get attribute 'lgamma_cached'`. Uma função normal + dict são serializados por VALOR pelo
    cloudpickle e atravessam o fork/spawn intactos. Valores são idênticos aos do lru_cache."""
    hit = _LGAMMA_CACHE.get(x)
    if hit is not None:
        return hit
    if len(_LGAMMA_CACHE) >= 4096:  # mesmo teto do lru_cache anterior
        _LGAMMA_CACHE.clear()
    value = _LGAMMA_CACHE[x] = math.lgamma(x)
    return value


def ewma_update(prev: float, x: float, lam: float) -> float:
    return (1.0 - lam) * prev + lam * x


def vol_adjust_step(v: float, e: float, lam: float) -> tuple[float, float]:
    """Um passo do ajuste de volatilidade (plano §3.4). Devolve `(v_novo, e_vol)`.

    A EWMA é atualizada ANTES de dividir — `e_vol_t` usa o `v_t` que já contém `e_t²`. A ordem
    importa para a equivalência bit-a-bit e não deve ser "simplificada".

    Existe como primitiva compartilhada porque a mesma recursão roda em DOIS lugares que precisam
    concordar exatamente: o laço online (`state/scorer.py`) e o replay sobre o histórico que mede o
    nulo por série (`state/calibration.py:history_evol`, F1). Duas implementações da mesma recursão
    é precisamente a classe de bug que este módulo existe para evitar — e aqui o custo de um
    desalinhamento é alto e silencioso: envenenaria o nulo de toda feature `e_vol`-based calibrada,
    sem erro visível."""
    v_new = ewma_update(v, e * e, lam)
    return v_new, e / math.sqrt(max(v_new, 1e-12))
