"""Features de convolução aleatória estilo MiniRocket (Dempster et al., 2021), em duas amostras.

Os 84 kernels de comprimento 9 do MiniRocket (pesos -1, com 3 posições valendo 2) em várias
dilatações. Para cada kernel × dilatação, os vieses são os quantis da saída convolvida no HISTÓRICO.
Assim a fração de valores positivos (PPV) no histórico vale 1-q por construção, e a feature é o desvio
dessa fração no trecho candidato: uma medida de mudança de distribuição do sinal filtrado, cobrindo
forma local, escala e estrutura temporal sem eixos escolhidos à mão. Determinístico.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

_IDX = np.array(list(combinations(range(9), 3)))            # 84 kernels
KERNELS = -np.ones((84, 9))
for _i, _c in enumerate(_IDX):
    KERNELS[_i, _c] = 2.0
DILATACOES = (1, 2, 4, 8)
QUANTIS = (0.1, 0.5, 0.9)


def _conv(x: np.ndarray, d: int) -> np.ndarray:
    """Saída (n_kernels, n_validos) da convolução dilatada, sem padding (causal no trecho)."""
    n = len(x) - 8 * d
    if n <= 0:
        return np.empty((84, 0))
    janelas = np.stack([x[j * d: j * d + n] for j in range(9)])          # (9, n)
    return KERNELS @ janelas


def vieses_hist(hist: np.ndarray) -> dict[int, np.ndarray]:
    return {d: np.quantile(_conv(hist, d), QUANTIS, axis=1).T for d in DILATACOES}   # d -> (84, 3)


def features_rocket(vieses: dict[int, np.ndarray], trecho: np.ndarray, antes: np.ndarray) -> np.ndarray:
    """Vetor (84 * len(DILATACOES) * len(QUANTIS),) de PPV_trecho - PPV_hist. `antes` fornece o
    contexto das dilatações para que o trecho tenha saídas desde o primeiro ponto."""
    out = []
    for d in DILATACOES:
        x = np.concatenate([np.asarray(antes)[-8 * d:], trecho])
        c = _conv(x, d)
        b = vieses[d]
        ppv = (c[:, :, None] > b[:, None, :]).mean(axis=1)                  # (84, 3)
        out.append(ppv - (1.0 - np.array(QUANTIS))[None, :])
    return np.concatenate([o.ravel() for o in out])
