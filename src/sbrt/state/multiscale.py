"""MultiScaleBlock — decomposição causal de energia por escala (Haar diádico)
(docs/PROPOSTA_FEATURES_V2.md F4).

Motivação. O parecer de auditoria classificou o confundimento CE2×T6 como "indecidível no detector"
(§4.3): um patamar novo de variância e um cluster GARCH longo são indistinguíveis *numa janela*. Mas
eles não são indistinguíveis *entre escalas* — e é isso que nenhuma feature atual mede:

- um **patamar persistente** de variância eleva a energia em TODAS as escalas aproximadamente igual;
- um **burst GARCH** (oscilação rápida de volatilidade) concentra energia nas escalas FINAS;
- um **drift lento** concentra nas escalas GROSSAS.

Logo, o *formato* da curva energia-vs-escala discrimina o que o nível sozinho não discrimina. As
janelas existentes (`accum_window_var_ln_w010..w250`) são **suavizações do mesmo nível** em
comprimentos diferentes, não uma decomposição de escala: todas medem E[e²] numa janela, apenas com
mais ou menos suavização. A transformada de Haar separa a energia em bandas de frequência
*disjuntas* — informação genuinamente diferente.

Implementação. Cascata diádica causal e O(1) amortizado: em cada escala j mantém-se no máximo um
valor pendente; quando o par (a, b) fica completo emite-se o detalhe d = (a−b)/√2 (banda daquela
escala) e a aproximação s = (a+b)/√2, que sobe para a escala j+1. A energia por escala é uma EWMA de
d². Um coeficiente da escala j nasce a cada 2^(j+1) amostras — por isso as escalas grossas ficam em
NaN por muitos passos (tratado nativamente pelo LightGBM; é honesto: não há informação multi-escala
antes de haver amostras).

Sob H0 (ruído branco de variância unitária) a transformada de Haar preserva variância, então
E[d²] = 1 em toda escala e `haar_energy_ln_s*` ≈ 0 — as features já nascem aproximadamente
comparáveis entre séries, e a calibração de nulo por série (F1) corrige o resíduo de dependência.

Consome `e` (escala congelada do histórico), não `e_vol`: é família de variância — a trava
anti-absorção CE2 do plano §3.4 vale aqui igual às demais.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter  # topo, não dentro da função: o conversor do Crunch avisa em
                                  # import aninhado (pode não virar requirement do submission)

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_SQRT2 = math.sqrt(2.0)


class MultiScaleBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        ms = cfg.multiscale
        self.J = ms.n_scales
        self.lam = ms.ewma_lambda
        self.min_coeffs = ms.warmup_min_coeffs

        self.pending: list[float | None] = [None] * self.J
        self.energy = [1.0] * self.J  # prior H0: Haar preserva variância -> E[d²]=1
        self.count = [0] * self.J

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        x = e
        for j in range(self.J):
            p = self.pending[j]
            if p is None:
                self.pending[j] = x
                return  # nada sobe para as escalas seguintes neste passo
            self.pending[j] = None
            d = (p - x) / _SQRT2
            s = (p + x) / _SQRT2
            self.energy[j] += self.lam * (d * d - self.energy[j])
            self.count[j] += 1
            x = s

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        ln_e: list[float] = []
        for j in range(self.J):
            if self.count[j] >= self.min_coeffs:
                v = math.log(max(self.energy[j], 1e-12))
            else:
                v = math.nan
            out[f"haar_energy_ln_s{j}"] = v
            ln_e.append(v)

        fine, mid, coarse = ln_e[0], ln_e[min(2, self.J - 1)], ln_e[self.J - 1]
        out["haar_contrast_fine_coarse"] = (
            fine - coarse if not (math.isnan(fine) or math.isnan(coarse)) else math.nan
        )
        out["haar_contrast_fine_mid"] = fine - mid if not (math.isnan(fine) or math.isnan(mid)) else math.nan
        return out


def history_series(e_hist: np.ndarray, cfg: "Config") -> dict:
    """As mesmas estatísticas sobre o histórico, vetorizadas, para a calibração de nulo por série
    (F1). O pareamento é idêntico ao do laço online (pares consecutivos disjuntos, aproximação
    subindo de escala) — equivalência verificada em
    `tests/unit/test_multiscale.py::test_history_series_matches_online_block`."""
    ms = cfg.multiscale
    e = np.asarray(e_hist, dtype=np.float64)
    out: dict = {}
    ln_by_scale: list[np.ndarray] = []

    s = e
    for j in range(ms.n_scales):
        n_pairs = len(s) // 2
        if n_pairs < 1:
            break
        a = s[: 2 * n_pairs : 2]
        b = s[1 : 2 * n_pairs : 2]
        d = (a - b) / _SQRT2
        s = (a + b) / _SQRT2

        zi = [(1.0 - ms.ewma_lambda) * 1.0]  # energia inicial 1.0, igual ao online
        energy, _ = lfilter([ms.ewma_lambda], [1.0, -(1.0 - ms.ewma_lambda)], d * d, zi=zi)
        valid = energy[ms.warmup_min_coeffs - 1:] if ms.warmup_min_coeffs > 0 else energy
        ln_v = np.log(np.maximum(valid, 1e-12))
        out[f"haar_energy_ln_s{j}"] = ln_v
        ln_by_scale.append(ln_v)

    if len(ln_by_scale) >= 2:
        fine = ln_by_scale[0]
        coarse = ln_by_scale[-1]
        mid = ln_by_scale[min(2, len(ln_by_scale) - 1)]
        n_fc = min(len(fine), len(coarse))
        n_fm = min(len(fine), len(mid))
        # Escalas grossas têm menos coeficientes; para a distribuição nula basta parear os
        # primeiros n comuns (a média/desvio não dependem do alinhamento temporal exato).
        out["haar_contrast_fine_coarse"] = fine[:n_fc] - coarse[:n_fc]
        out["haar_contrast_fine_mid"] = fine[:n_fm] - mid[:n_fm]
    return out
