"""Harness causal — replay ponto a ponto (plano §9.2, §12.1). Papel único: verificação de
CORRETUDE (causalidade, determinismo) — nunca agrega scores numa estimativa de desempenho (§9.0).
Necessário também para construir o dataset de treino (motor único, model/dataset.py)."""
from __future__ import annotations

from typing import Callable, List

import numpy as np
from tqdm import tqdm


def replay(hist: np.ndarray, online: np.ndarray, scorer, progress: bool = False) -> List[float]:
    """Alimenta o online um ponto por vez. NUNCA vetoriza. `scorer` já foi construído a partir de
    `hist` (via fit_h0 + StreamScorer) — `hist` é aceito aqui só por simetria com
    `check_prefix_equivalence` (mesma assinatura). progress=True só em uso manual/exploratório."""
    iterator = tqdm(online, desc="replay causal") if progress else online
    return [scorer.update(float(x)) for x in iterator]


def check_prefix_equivalence(
    hist: np.ndarray,
    online: np.ndarray,
    scorer_factory: Callable[[np.ndarray, np.ndarray], object],
    cut_points: List[int],
) -> bool:
    """Para cada k em cut_points: score(replay completo)[:k] == score(replay truncado em k), bit a
    bit. `scorer_factory(hist, online_segment)` deve devolver um StreamScorer NOVO (estado zerado) a
    cada chamada, ligado exatamente ao segmento que será replay-ado (nunca ao `online` completo) —
    é isso que permite ao canário de vazamento (`adversarial/leaky_canary.py`) ser efetivamente
    reprovado: um scorer honesto ignora o 2º argumento; um scorer que espia o futuro só pode fazê-lo
    dentro do que o 2º argumento contém, então a truncagem realmente corta o que ele "vê" (armadilha
    §13.1). NÃO calcula nenhuma métrica de desempenho — só corretude de código (§9.0)."""
    full_scorer = scorer_factory(hist, online)
    full_scores = replay(hist, online, full_scorer)

    for k in cut_points:
        if k <= 0 or k > len(online):
            continue
        trunc_online = online[:k]
        trunc_scorer = scorer_factory(hist, trunc_online)
        trunc_scores = replay(hist, trunc_online, trunc_scorer)
        if trunc_scores != full_scores[:k]:
            return False
    return True
