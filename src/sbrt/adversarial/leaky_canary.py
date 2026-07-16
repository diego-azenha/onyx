"""Canário de vazamento (plano §12.1): `LeakyStreamScorer` espia deliberadamente x_{t+1} ao montar o
score. `evaluation.harness.check_prefix_equivalence` DEVE reprovar esta classe (e aprovar o
`StreamScorer` real) — é a prova de que o detector de vazamento funciona."""
from __future__ import annotations

import numpy as np

from sbrt.state.scorer import StreamScorer


class LeakyStreamScorer(StreamScorer):
    def __init__(self, h0, blocks, ensemble, cfg, online: np.ndarray):
        super().__init__(h0, blocks, ensemble, cfg)
        self._online = np.asarray(online, dtype=np.float64)

    def update(self, x: float) -> float:
        score = super().update(x)
        t_now = self.t  # já incrementado por update_features -> update_features -> self.t
        if t_now < len(self._online):
            leak = self._online[t_now]  # x_{t+1}: NÃO deveria estar visível ainda (armadilha §13.1)
            score = float(np.clip(0.5 * score + 0.5 * (1.0 if leak > 0 else 0.0), 0.0, 1.0))
        return score
