"""ConformalBlock — martingales conformais sobre p-values causais das inovações contra a
distribuição do histórico (plano §4.2 #23; Vovk et al. 2005; Volkhonskiy et al. 2017).

Evidência livre de distribuição, O(log n_h)/passo via busca binária nos arrays ordenados do
histórico (`H0Params.sorted_e_hist` / `sorted_abs_e_hist`). Opera sempre sobre `e` (escala congelada)
porque os arrays ordenados de referência foram construídos a partir do resíduo/sigma_e do histórico —
comparar `e_vol` contra eles seria inconsistente de escala quando o ajuste de volatilidade está ativo.

Três variantes de p-value (todas via mid-rank, para lidar com empates):
- abs: cauda superior de |e_t| contra |e| do histórico — sensível a variância/cauda.
- right: cauda superior de e_t (com sinal) — sensível a shift positivo/skew à direita.
- sign: cauda inferior de e_t (com sinal) — sensível a shift negativo/skew à esquerda.

Cada uma vira um log-martingale (mistura sobre epsilons, "apostas" de Vovk); "6->4 usadas" (plano
tabela §5 #23): abs tem variante acumulada E com reset (SR-like), right/sign só acumuladas.
"""
from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

from sbrt.utils.numerics import logsumexp

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params


def _mid_rank(sorted_arr, x: float) -> float:
    lo = bisect.bisect_left(sorted_arr, x)
    hi = bisect.bisect_right(sorted_arr, x)
    return (lo + hi) / 2.0


def _upper_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (n - rank + 0.5) / (n + 1.0)


def _lower_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (rank + 0.5) / (n + 1.0)


class ConformalBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.sorted_e_hist = h0.sorted_e_hist
        self.sorted_abs_e_hist = h0.sorted_abs_e_hist
        self.n_h = h0.n_h
        self.epsilons = list(cfg.conformal.epsilons)
        self._log_k = math.log(len(self.epsilons))

        self.L_abs = {eps: 0.0 for eps in self.epsilons}
        self.L_abs_reset = {eps: 0.0 for eps in self.epsilons}
        self.L_right = {eps: 0.0 for eps in self.epsilons}
        self.L_sign = {eps: 0.0 for eps in self.epsilons}

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        p_abs = _upper_tail_p(self.sorted_abs_e_hist, abs(e), self.n_h)
        p_right = _upper_tail_p(self.sorted_e_hist, e, self.n_h)
        p_sign = _lower_tail_p(self.sorted_e_hist, e, self.n_h)

        log_p_abs = math.log(p_abs)
        log_p_right = math.log(p_right)
        log_p_sign = math.log(p_sign)

        for eps in self.epsilons:
            log_eps = math.log(eps)
            inc_abs = log_eps + (eps - 1.0) * log_p_abs
            self.L_abs[eps] += inc_abs
            self.L_abs_reset[eps] = max(0.0, self.L_abs_reset[eps] + inc_abs)
            self.L_right[eps] += log_eps + (eps - 1.0) * log_p_right
            self.L_sign[eps] += log_eps + (eps - 1.0) * log_p_sign

    def features(self) -> dict[str, float]:
        return {
            "conformal_logm_abs": logsumexp(list(self.L_abs.values())) - self._log_k,
            "conformal_logm_abs_reset": logsumexp(list(self.L_abs_reset.values())) - self._log_k,
            "conformal_logm_right": logsumexp(list(self.L_right.values())) - self._log_k,
            "conformal_logm_sign": logsumexp(list(self.L_sign.values())) - self._log_k,
        }
