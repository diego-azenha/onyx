"""LMomentBlock — forma de cauda dinâmica via L-momentos (docs/INVESTIGACAO_FALHAS_V3.md P2).

## Por que existe

Segundo maior ponto cego do modelo (INVESTIGACAO §1.2, §4.2): quebras *puras* de cauda/forma (Δkurt
alto, Δlogvar baixo) têm detectabilidade 0,553 — sinal real e **independente** da variância (β=+0,05,
corr 0,0 com os outros eixos). O banco mede cauda por *contagem de excedência* e pelo conforme
`conformal_logm_abs`; a *forma* da cauda com variância inalterada é fracamente captada.

Momentos clássicos (curtose = m4/m2²) são péssimos em amostra pequena e dominados por um único
outlier — exatamente o regime que importa (t pequeno, cauda pesada, e o falso-positivo T9). **Os
L-momentos** (combinações lineares de estatísticas de ordem) caracterizam forma com robustez e baixa
variância amostral. L-skewness (τ₃) e L-kurtosis (τ₄) são **razões** (L3/L2, L4/L2), portanto
**invariantes a escala** — medem forma *ortogonalmente* à variância, que é precisamente o eixo que o
modelo já domina e que não queremos duplicar.

Estimador via momentos ponderados por probabilidade (PWM), sobre a janela ordenada mantida
incrementalmente (bisect insere/remove em O(w); PWM em O(w)):
  b_r = (1/n) Σ_i [C(i,r)/C(n-1,r)] x_(i)   (x_(i) ordenado ascendente, i=0..n-1)
  L1=b0; L2=2b1−b0; L3=6b2−6b1+b0; L4=20b3−30b2+12b1−b0
  τ₃=L3/L2 (assimetria);  τ₄=L4/L2 (curtose)

Opera sobre `e_raw` (resíduo padronizado NÃO clipado): as razões τ são invariantes a escala, então a
padronização por sigma_e é irrelevante, e o não-clipar preserva a cauda que é justamente o sinal (a
robustez dos L-momentos torna o outlier bruto seguro, ao contrário da curtose clássica). A versão
calibrada (F1) subtrai a forma do próprio histórico da série — uma série de cauda naturalmente pesada
tem τ₄ histórico alto, então só um *excesso* pós-quebra acende.
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


def _l_ratios(sorted_vals: list) -> tuple[float, float]:
    """(τ₃, τ₄) da janela ordenada via PWM. NaN se n<4 ou L2≈0 (janela degenerada)."""
    n = len(sorted_vals)
    if n < 4:
        return _NAN, _NAN
    b0 = b1 = b2 = b3 = 0.0
    for i, x in enumerate(sorted_vals):
        b0 += x
        b1 += i * x
        b2 += i * (i - 1) * x
        b3 += i * (i - 1) * (i - 2) * x
    nm1 = n - 1
    nm2 = nm1 * (n - 2)
    nm3 = nm2 * (n - 3)
    b0 /= n
    b1 /= n * nm1
    b2 /= n * nm2
    b3 /= n * nm3
    l2 = 2 * b1 - b0
    if abs(l2) < 1e-9:
        return _NAN, _NAN
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l3 / l2, l4 / l2


class _SortedWindow:
    """Janela deslizante ordenada, O(w) por passo (bisect + shift de lista)."""

    __slots__ = ("W", "sorted", "order")

    def __init__(self, window: int):
        self.W = window
        self.sorted: list = []
        self.order: deque = deque()

    def update(self, v: float) -> None:
        if len(self.order) >= self.W:
            old = self.order.popleft()
            pos = bisect.bisect_left(self.sorted, old)
            del self.sorted[pos]
        bisect.insort(self.sorted, v)
        self.order.append(v)


class LMomentBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(cfg.lmoments.windows)
        self.win = {w: _SortedWindow(w) for w in self.windows}
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        for w in self.windows:
            self.win[w].update(e_raw)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            if warm:
                tau3, tau4 = _l_ratios(self.win[w].sorted)
            else:
                tau3 = tau4 = _NAN
            out[f"lmom_lskew_w{w:03d}"] = tau3
            out[f"lmom_lkurt_w{w:03d}"] = tau4
        return out


def history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio LMomentBlock sobre o histórico (H0), para a calibração de nulo por série (F1).
    Rodar o bloco real garante equivalência online/nulo por construção. `e_hist` é o resíduo
    padronizado não-clipado, equivalente ao `e_raw` do online."""
    blk = LMomentBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(0.0, float(ev), 0.0, i)
        for name, val in blk.features().items():
            acc.setdefault(name, []).append(val)
    return acc
