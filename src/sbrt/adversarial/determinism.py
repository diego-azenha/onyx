"""Re-execução bit-a-bit (plano §12.4, checklist §15.2) — mais estrita que a tolerância 1e-8 da
plataforma (F9): compara re-execuções float a float, sem tolerância nenhuma. Fontes de risco
cobertas por construção: `num_threads=1` no predict, ausência de RNG no caminho de inferência,
Welford/log-space (erro relativo ~n*eps_machine, irrelevante), nenhuma iteração sobre set/dict no
caminho de inferência (só arrays/listas)."""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np
from tqdm import tqdm

from sbrt.evaluation.harness import replay


def rerun_bitexact(
    series_sample: List[Tuple[np.ndarray, np.ndarray]],
    scorer_factory: Callable[[np.ndarray], object],
    fraction: float,
    seed: int,
    progress: bool = True,
) -> bool:
    n = len(series_sample)
    if n == 0:
        return True
    k = max(1, int(round(n * fraction)))
    rng = np.random.RandomState(seed)
    chosen = rng.choice(n, size=k, replace=False)

    iterator = tqdm(chosen, desc="reexecução bit-exata (determinismo)") if progress else chosen
    for i in iterator:
        hist, online = series_sample[i]
        scores1 = replay(hist, online, scorer_factory(hist))
        scores2 = replay(hist, online, scorer_factory(hist))
        if scores1 != scores2:
            return False
    return True
