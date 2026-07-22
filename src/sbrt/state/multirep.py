"""MultiRepBlock — a MESMA estatística de tipo-integral sobre três representações do fluxo.

## Dois eixos novos de uma vez

**(a) Estatística de tipo-integral, não de tipo-supremo.** Todo detector sequencial do banco é
`sup`: CUSUM guarda o máximo do passeio refletido, `bayes` guarda o máximo a posteriori,
`varloc` guarda max/min de z de variância, `conformal` guarda o martingale (que é dominado pelo pico).
A teoria clássica de quebra estrutural distingue duas famílias com perfis de potência **diferentes**:

- tipo-supremo (Kolmogorov-Smirnov / CUSUM): forte contra **uma** quebra brusca e localizada;
- tipo-integral (Cramér-von Mises / KPSS, `∫ B°(r)² dr`): forte contra alternativas **difusas** —
  deriva gradual, quebras múltiplas pequenas, quebra perto da borda da janela.

Perto da borda o contraste é grande: uma quebra a 5 passos do fim da janela dá um pico ainda pequeno
(o CUSUM ainda não somou evidência), mas já desloca a ponte inteira do resto da janela. O banco não
tem *nenhuma* estatística integral. Este é o buraco.

A estatística, sobre uma janela de n pontos x com somas parciais P_i:

    Q = Σ_{i=1..n} (P_i − (i/n)·P_n)² ,   KPSS = Q / (n²·s²_n)

`P_i − (i/n)P_n` é a ponte: ela zera nas duas pontas e é **invariante a somar constante a x** (por
isso `e²` e `e²−1` dão exatamente o mesmo valor, e nenhuma centragem é necessária). Dividir por
`s²_n` (variância da janela) torna-a **invariante a escala**: sob H0 o valor converge para
`∫₀¹B°(r)²dr` — média 1/6 — para *qualquer* série, independentemente de variância, cauda ou n. Isso
é auto-normalização por construção, não constante estimada por série: a lição do fracasso do F1 é
que constante estimada por série injeta ruído transversal puro num alvo que só pontua ordenação
transversal.

**(b) Três representações do mesmo fluxo.** A mesma ponte, aplicada a:

| representação | o que a ponte enxerga |
|---|---|
| `e`            | deriva/quebra de **média** |
| `e²`           | deriva/quebra de **variância** (análogo integral de `cusum_var`) |
| `u = F̂_h0(e)` | deriva/quebra de **distribuição**, livre de distribuição (PIT contra o histórico) |

A terceira é a que não tem análogo: `rank_twosample` compara janela-vs-histórico com estatísticas
de nível, e `conformal` acumula p-values num martingale (tipo-supremo). Nenhuma das duas olha a
*trajetória* do PIT dentro da janela. Sendo baseada em rank, ela é imune a cauda pesada — o regime
em que a versão sobre `e²` é menos confiável.

**A representação diferenciada foi deliberadamente descartada.** Com `d_k = e_k − e_{k-1}` a soma
parcial telescopa (`P_i = e_{a+i} − e_a`), a ponte vira função dos valores crus e Q colapsa em
`n·var(janela)` — exatamente `accum_window_var_ln_*`. Diferenciar não é uma representação nova aqui;
é a mesma coluna com outro nome. (Ela aparece na proposta original de multi-representação; é o tipo
de duplicata que o rastreio pegaria, mas 30 minutos de build depois.)

## Custo: O(1) por passo

Q parece exigir varrer a janela, mas expande em somas mantidas incrementalmente sobre a soma
acumulada C (ver `_RollingBridge`). O único cuidado é numérico: Q é um resíduo pequeno de somas
grandes de C². Com float64 e o `e` padronizado (|C| ~ √t, ou ~t·Δ sob quebra de média), a perda é de
poucos dígitos significativos numa margem de 16 — verificado contra o cálculo direto em
`tests/unit/test_multirep.py`.

Consome `e` (escala congelada), como as demais famílias de variância/cauda — trava CE2 do plano §3.4.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.state.conformal import _upper_tail_p
from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


class _RollingBridge:
    """KPSS/Cramér-von Mises de janela deslizante, O(1) por passo.

    Identidades usadas (janela = índices a+1..t, n pontos, C_k = soma acumulada até k, C_a = valor
    imediatamente antes da janela):

        Σ P_i²   = B − 2·C_a·A + n·C_a²
        Σ i·P_i  = D + n·A − C_a·n(n+1)/2
        Σ i²     = n(n+1)(2n+1)/6
        Q        = Σ P_i² − (2·P_n/n)·Σ i·P_i + (P_n²/n²)·Σ i²

    com A = Σ_janela C_k, B = Σ_janela C_k², D = Σ_janela (k−t)·C_k. Os três são atualizáveis em
    O(1); D usa `D_t = D_{t-1} − A_{t-1} + n·C_evicto`, que é o que mantém a magnitude de D presa a
    O(n·|C|) em vez de crescer com t.
    """

    __slots__ = ("W", "C", "C_a", "A", "B", "D", "ring", "sx", "sxx", "ringx", "min_n")

    def __init__(self, window: int, min_n: int):
        self.W = window
        self.min_n = min_n
        self.C = 0.0
        self.C_a = 0.0        # C imediatamente antes da janela; 0 enquanto a janela não encheu
        self.A = 0.0
        self.B = 0.0
        self.D = 0.0
        self.ring = RingBuffer(window)    # valores de C dentro da janela
        self.ringx = RingBuffer(window)   # os x, para a variância da janela
        self.sx = 0.0
        self.sxx = 0.0

    def update(self, x: float) -> None:
        n_old = len(self.ring)
        self.D -= self.A                  # todos os (k−t) caem 1 quando t avança
        self.C += x
        evicted = self.ring.push(self.C)
        if evicted is not None:
            self.A -= evicted
            self.B -= evicted * evicted
            self.D += n_old * evicted     # remove o termo (k_ev − t)·C_ev = −n·C_ev
            self.C_a = evicted
        self.A += self.C
        self.B += self.C * self.C
        # o termo novo tem (k − t) = 0: não entra em D

        ex = self.ringx.push(x)
        if ex is None:
            self.sx += x
            self.sxx += x * x
        else:
            self.sx += x - ex
            self.sxx += x * x - ex * ex

    def value(self) -> float:
        n = len(self.ring)
        if n < self.min_n:
            return _NAN
        var = self.sxx / n - (self.sx / n) ** 2
        if var <= 1e-12:
            return _NAN
        Ca = self.C_a
        sum_p2 = self.B - 2.0 * Ca * self.A + n * Ca * Ca
        sum_ip = self.D + n * self.A - Ca * n * (n + 1) / 2.0
        sum_i2 = n * (n + 1) * (2 * n + 1) / 6.0
        Pn = self.C - Ca
        q = sum_p2 - (2.0 * Pn / n) * sum_ip + (Pn * Pn / (n * n)) * sum_i2
        return max(q, 0.0) / (n * n * var)


class MultiRepBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        mc = cfg.multirep
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(mc.windows)
        self.br_e = {w: _RollingBridge(w, mc.min_n) for w in self.windows}
        self.br_e2 = {w: _RollingBridge(w, mc.min_n) for w in self.windows}
        self.br_rank = {w: _RollingBridge(w, mc.min_n) for w in self.windows}
        self.sorted_e_hist = h0.sorted_e_hist if h0 is not None else None
        self.n_h = h0.n_h if h0 is not None else 0
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        if self.sorted_e_hist is not None:
            u = _upper_tail_p(self.sorted_e_hist, e, self.n_h)
        else:
            u = 0.5
        se = e * e
        for w in self.windows:
            self.br_e[w].update(e)
            self.br_e2[w].update(se)
            self.br_rank[w].update(u)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            out[f"mrep_kpss_e_w{w:03d}"] = self.br_e[w].value() if warm else _NAN
            out[f"mrep_kpss_e2_w{w:03d}"] = self.br_e2[w].value() if warm else _NAN
            out[f"mrep_kpss_rank_w{w:03d}"] = self.br_rank[w].value() if warm else _NAN
        return out
