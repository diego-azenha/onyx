"""RankTwoSampleBlock — duas amostras rank-based janela-vs-histórico (parecer de auditoria
docs/PARECER_AUDITORIA_ONYX.md §3.6/§6-R4: "o análogo causal do que venceu 2025").

O banco atual é forte em detectores sequenciais paramétricos (CUSUM, filtro bayesiano) e fraco no
paradigma que dominou a edição batch: comparação distribucional direta janela-vs-histórico. Os
p-values conformais de `ConformalBlock` (rank de e_t contra o histórico ordenado, livres de
distribuição e comparáveis entre séries por construção) são reaproveitados aqui como a "moeda" de
um agregador de POTÊNCIA (médias de janela) em vez do agregador de martingale de Vovk (otimizado
para controle de erro tipo Ville sob H0, não para ranking) — parecer §3.6.

Três famílias, cada uma O(log n_h) por passo via busca binária nos arrays ordenados do histórico
(`H0Params.sorted_e_hist` / `sorted_abs_e_hist`), sempre sobre `e` (escala congelada, mesma
convenção de `ConformalBlock` — os arrays de referência foram construídos a partir do resíduo do
histórico, comparar `e_vol` seria inconsistente de escala):

- localização: z de Wilcoxon de janela = média_w(p_right − ½)·√(12·n_eff) — estatística de
  Mann-Whitney da janela contra o histórico, robusta a caudas pesadas (parecer §3.6). CONVENÇÃO DE
  SINAL herdada de `p_right`/`p_abs` (cauda superior, mesma de ConformalBlock): p encolhe para perto
  de 0 quando o valor observado é extremo à direita, então um shift POSITIVO faz este z ficar mais
  NEGATIVO (não positivo) — sinal válido e monótono para uma árvore, só invertido do que a
  nomenclatura "Wilcoxon" sugeriria ingenuamente.
- dispersão/cauda: o mesmo, sobre p_abs em vez de p_right (tipo Ansari-Bradley) — sensível a
  quebras de variância/forma sem depender de momentos; mesma convenção de sinal (mais negativo sob
  variância maior, já que p_abs também é cauda superior).
- forma: chi²-de-janela sobre 4 bins de quantis do histórico (quartis) — frações observadas vs.
  nominais (25% cada), generaliza o quantile-crossing existente (accumulators.py #18).
"""
from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

import numpy as np

from sbrt.utils.ring_buffer import RingBuffer

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


class _WindowSum:
    """RingBuffer(w) + soma incremental — mesmo padrão de accumulators.py."""

    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, value: float) -> None:
        evicted = self.ring.push(value)
        self.total += value - (evicted if evicted is not None else 0.0)


class RankTwoSampleBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.warmup_min_n = cfg.features.warmup_min_n
        self.sorted_e_hist = h0.sorted_e_hist
        self.sorted_abs_e_hist = h0.sorted_abs_e_hist
        self.n_h = h0.n_h
        self.windows = list(cfg.rank_twosample.windows)
        self.t = 0

        self.loc_sum = {w: _WindowSum(w) for w in self.windows}
        self.disp_sum = {w: _WindowSum(w) for w in self.windows}

        # quartis do histórico (a partir de e_hist ordenado -- não depende de h0.q, que não inclui
        # a mediana); 4 bins: (-inf,q25], (q25,q50], (q50,q75], (q75,inf), nominal 25% cada.
        self._q25, self._q50, self._q75 = (float(v) for v in np.quantile(self.sorted_e_hist, [0.25, 0.5, 0.75]))
        self.bin_counts = {w: [0.0, 0.0, 0.0, 0.0] for w in self.windows}
        self.bin_ring = {w: RingBuffer(w) for w in self.windows}

    def _bin_of(self, e: float) -> int:
        if e <= self._q25:
            return 0
        if e <= self._q50:
            return 1
        if e <= self._q75:
            return 2
        return 3

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        p_right = _upper_tail_p(self.sorted_e_hist, e, self.n_h)
        p_abs = _upper_tail_p(self.sorted_abs_e_hist, abs(e), self.n_h)

        for w in self.windows:
            self.loc_sum[w].push(p_right - 0.5)
            self.disp_sum[w].push(p_abs - 0.5)

        b = self._bin_of(e)
        for w in self.windows:
            ring = self.bin_ring[w]
            counts = self.bin_counts[w]
            counts[b] += 1.0
            evicted = ring.push(float(b))
            if evicted is not None:
                counts[int(evicted)] -= 1.0

    def features(self) -> dict[str, float]:
        t = self.t
        wmin = self.warmup_min_n
        out: dict[str, float] = {}

        for w in self.windows:
            n_eff = min(t, w)
            key_loc = f"ranktwo_wilcoxon_z_w{w:03d}"
            key_disp = f"ranktwo_dispersion_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key_loc] = (self.loc_sum[w].total / n_eff) * math.sqrt(12.0 * n_eff)
                out[key_disp] = (self.disp_sum[w].total / n_eff) * math.sqrt(12.0 * n_eff)
            else:
                out[key_loc] = math.nan
                out[key_disp] = math.nan

            key_shape = f"ranktwo_shape_chi2_w{w:03d}"
            if t >= wmin and n_eff > 0:
                expected = n_eff / 4.0
                counts = self.bin_counts[w]
                out[key_shape] = sum((c - expected) ** 2 / expected for c in counts) if expected > 0 else math.nan
            else:
                out[key_shape] = math.nan

        return out
