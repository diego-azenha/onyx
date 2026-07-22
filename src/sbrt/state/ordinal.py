"""OrdinalBlock — padrões ordinais e entropia de permutação (Bandt & Pompe 2002).

## Por que este eixo não está coberto

Toda estatística do banco lê **valores**: variâncias, autocorrelações, quantis, energias, p-values
conformais. Mesmo as "rank-based" (`rank_twosample`, `conformal`) comparam a janela contra a
distribuição do histórico — ou seja, ainda leem a *marginal*, só que de forma robusta.

O que não existe é uma estatística que leia **apenas a ordem relativa dentro de uma janela curta**.
Essa é a construção de Bandt-Pompe: para cada bloco de m pontos consecutivos, guarda-se só o padrão
de ordenação (qual é o menor, o segundo, ...) — um de m! símbolos. A distribuição desses símbolos
numa janela é o objeto medido.

A propriedade que torna isto ortogonal ao banco: o padrão ordinal é **invariante a qualquer
transformação monótona ponto-a-ponto**. Mudar a variância, a cauda, a escala, aplicar um log — nada
disso muda um único símbolo. Logo:

- a família de variância, que domina o xs-SHAP (`meta_h0` 34,9%, acumuladores 14,7%), não pode
  explicar estas colunas nem por acaso;
- elas só se movem quando a **estrutura temporal** muda — precisamente o ponto cego medido, quebras
  puras de dependência com detectabilidade 0,492, *abaixo do acaso* (docs/INVESTIGACAO_FALHAS_V3.md §1).

Três leituras da distribuição de padrões:

1. **Entropia de permutação** (normalizada por log m!): sob H0 (i.i.d.) todos os m! padrões são
   equiprováveis e a entropia é máxima. Qualquer dependência serial a reduz — sem escolher lag,
   sem supor linearidade. É o complemento não-paramétrico de `dep_mass`/`mismatch_white`, que só
   veem correlação (momento de segunda ordem).
2. **Padrões proibidos** (m=4): a fração dos 24 símbolos com contagem zero na janela. Séries
   determinísticas/caóticas e séries com estrutura forte têm padrões que simplesmente não ocorrem;
   ruído os visita todos. É sensível a um tipo de estrutura que a entropia dilui.
3. **Irreversibilidade temporal**: ½·Σ_π |p(π) − p(π^R)|, com π^R o padrão do bloco lido de trás
   para frente. Sob qualquer processo reversível no tempo — o que inclui **todo** processo linear
   gaussiano, e portanto todo o modelo H0 deste projeto — esta soma é 0 em esperança. Ela só é
   positiva sob não-linearidade com direção do tempo (assimetria de subida/descida, efeito
   alavancagem, saltos com recuperação lenta). **Nenhuma feature do banco tem essa propriedade**:
   variância, |e|, e², ρ_k e energia de Haar são todas simétricas no tempo por construção.

## Escolhas de estimação

`m=3` (6 símbolos) para janelas curtas e `m=4` (24 símbolos) para a longa: o número de símbolos tem
de ser bem menor que a janela, senão a entropia estimada vira ruído de contagem.

O bloco emite valor com a janela ainda **parcial** (a partir de `min_counts` símbolos), em vez de
esperar a janela encher. A entropia de contagem é enviesada para baixo com n pequeno — mas o viés
depende só de n, e **n é igual para todas as séries no mesmo passo t**. Sob a invariância C1 da
TS-AUC (só a ordenação dentro do passo pontua) um viés comum a todas as séries é exatamente neutro.
Esperar a janela encher custaria NaN em todo o bucket 51-150 em troca de nada.

Consome `e` (escala congelada). Como o padrão ordinal é invariante a transformação monótona fixa, a
escolha `e` vs `e_vol` só importa porque o ajuste de volatilidade é *variável no tempo* — ele
reordena pontos vizinhos e, com isso, apagaria parte da estrutura que este bloco procura.
"""
from __future__ import annotations

import math
from collections import deque
from itertools import permutations
from typing import TYPE_CHECKING

from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan
_FACT = (1, 1, 2, 6, 24, 120)


def pattern_index(vals, m: int) -> int:
    """Código de Lehmer do padrão de ordenação de `vals` (do mais antigo ao mais recente).

    Empates são desempatados pelo índice (comparação estrita `<`), o que é determinístico e
    irrelevante na prática: `e` é contínuo."""
    idx = 0
    for i in range(m):
        c = 0
        vi = vals[i]
        for j in range(i + 1, m):
            if vals[j] < vi:
                c += 1
        idx += c * _FACT[m - 1 - i]
    return idx


def _reversal_map(m: int) -> list[int]:
    """π -> π^R (o padrão do mesmo bloco lido de trás para frente), pré-computado uma vez."""
    out = [0] * _FACT[m]
    for perm in permutations(range(m)):
        out[pattern_index(perm, m)] = pattern_index(perm[::-1], m)
    return out


_REV = {m: _reversal_map(m) for m in (3, 4)}


class _RollingPatternHist:
    """Histograma deslizante de símbolos ordinais. O(1) por passo (a entropia é O(m!) na leitura)."""

    __slots__ = ("M", "counts", "ring", "min_counts", "m")

    def __init__(self, window: int, m: int, min_counts: int):
        self.m = m
        self.M = _FACT[m]
        self.counts = [0] * self.M
        self.ring = RingBuffer(window)
        self.min_counts = min_counts

    def update(self, idx: int) -> None:
        evicted = self.ring.push(float(idx))
        self.counts[idx] += 1
        if evicted is not None:
            self.counts[int(evicted)] -= 1

    def _n(self) -> int:
        n = len(self.ring)
        return n if n >= self.min_counts else 0

    def entropy(self) -> float:
        n = self._n()
        if n == 0:
            return _NAN
        h = 0.0
        for c in self.counts:
            if c > 0:
                p = c / n
                h -= p * math.log(p)
        return h / math.log(self.M)

    def forbidden(self) -> float:
        if self._n() == 0:
            return _NAN
        z = 0
        for c in self.counts:
            if c == 0:
                z += 1
        return z / self.M

    def irreversibility(self) -> float:
        n = self._n()
        if n == 0:
            return _NAN
        rev = _REV[self.m]
        s = 0.0
        for i, c in enumerate(self.counts):
            s += abs(c - self.counts[rev[i]])
        return 0.5 * s / n


class OrdinalBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        oc = cfg.ordinal
        self.warmup = cfg.features.warmup_min_n
        self.m3_windows = list(oc.m3_windows)
        self.m4_windows = list(oc.m4_windows)
        self.h3 = {w: _RollingPatternHist(w, 3, oc.min_counts_m3) for w in self.m3_windows}
        self.h4 = {w: _RollingPatternHist(w, 4, oc.min_counts_m4) for w in self.m4_windows}
        self.irrev_window = max(self.m3_windows)
        self.buf: deque = deque(maxlen=4)
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        self.buf.append(e)
        n = len(self.buf)
        if n >= 3:
            i3 = pattern_index((self.buf[-3], self.buf[-2], self.buf[-1]), 3)
            for w in self.m3_windows:
                self.h3[w].update(i3)
        if n >= 4:
            i4 = pattern_index((self.buf[-4], self.buf[-3], self.buf[-2], self.buf[-1]), 4)
            for w in self.m4_windows:
                self.h4[w].update(i4)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.m3_windows:
            out[f"ord_pe_m3_w{w:03d}"] = self.h3[w].entropy() if warm else _NAN
        for w in self.m4_windows:
            out[f"ord_pe_m4_w{w:03d}"] = self.h4[w].entropy() if warm else _NAN
            out[f"ord_forbidden_m4_w{w:03d}"] = self.h4[w].forbidden() if warm else _NAN
        iw = self.irrev_window
        out[f"ord_irrev_m3_w{iw:03d}"] = self.h3[iw].irreversibility() if warm else _NAN
        return out
