"""BOCPDBlock — detecção bayesiana online de changepoint com posterior de run-length
(docs/RESULTADOS_P1_P4.md §próximo passo; Adams & MacKay 2007).

## Por que existe

A família mais valiosa que a sessão adicionou foi `varloc` (P3, 4,81% de SHAP): variância
*localizada* no changepoint via um max heurístico sobre escalas fixas. Isto é a versão PRINCIPIADA da
mesma ideia. Em vez de escolher a melhor de um punhado de janelas, o BOCPD mantém o **posterior
completo sobre o run-length** r_t = número de passos desde o último changepoint, por passo, via a
recursão de mensagem de Adams-MacKay — O(R_max) por passo, sem armazenar a série.

Modelo por regime: eₜ ~ N(0, σ²), σ² ~ Inverse-Gamma(α₀, β₀) (conjugado; preditiva Student-t, robusta
a cauda pesada — apropriado dado o censo). Hazard constante H=1/λ (prior geométrico no run-length).
Recursão: para cada eₜ, cresce cada hipótese de run-length pela verossimilhança preditiva e pelo
(1−H), e acumula massa em r=0 pela hazard. As estatísticas suficientes (contagem, Σe²) por
run-length crescem por mensagem.

Features:
- `bocpd_regime_var_ln` = ln(Σ_r p(r)·E[σ²|r]) — a variância do regime ATUAL, ponderada pelo
  posterior de run-length. É o estimador de variância limpo e localizado em tau que o oracle usa
  (INVESTIGACAO §2, AUC 0,856) e que as janelas fixas do banco diluem por não saber onde é tau.
- `bocpd_cp_prob` = Σ_{r<k} p(r) — probabilidade de um changepoint MUITO recente (alarme localizado).
- `bocpd_rl_entropy` = −Σ p(r) ln p(r) — incerteza da localização (baixa = changepoint nítido).
- `bocpd_map_runlen` = ln(1+argmax_r p(r)) — idade MAP do changepoint (localizador).

Consome `e` (escala congelada) — família de variância, trava CE2 (plano §3.4). A versão calibrada
(F1) usa o nulo da própria série (uma série naturalmente ruidosa acumula mais changepoints espúrios
no histórico), tornando `regime_var_ln`/`cp_prob`/`rl_entropy` comparáveis entre séries.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


class BOCPDBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        bc = cfg.bocpd
        self.warmup = cfg.features.warmup_min_n
        self.R = bc.r_max
        self.H = 1.0 / bc.hazard_lambda
        self.a0 = bc.alpha0
        self.b0 = bc.beta0
        self.recent_k = bc.recent_k
        self.t = 0

        r = np.arange(self.R + 1, dtype=np.float64)
        self.alpha = self.a0 + r / 2.0                 # α_r determinístico em r -> precomputa lgammas
        self._lg_ah = gammaln(self.alpha + 0.5)
        self._lg_a = gammaln(self.alpha)
        self._alpha_m1 = np.maximum(self.alpha - 1.0, 0.5)  # guarda para E[σ²|r]=β_r/(α_r−1)

        self.prob = np.zeros(self.R + 1, dtype=np.float64)
        self.prob[0] = 1.0
        self.sum_e2 = np.zeros(self.R + 1, dtype=np.float64)

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        se = e * e
        beta = self.b0 + self.sum_e2 / 2.0
        # log-preditiva Student-t de cada run-length (lgammas precomputados)
        logpred = (
            self._lg_ah - self._lg_a
            - 0.5 * np.log(2.0 * math.pi * beta)
            - (self.alpha + 0.5) * np.log1p(se / (2.0 * beta))
        )
        pred = np.exp(logpred - logpred.max())
        w = self.prob * pred
        cp = self.H * w.sum()          # massa de changepoint -> r=0
        growth = (1.0 - self.H) * w    # crescimento -> r+1

        new = np.empty(self.R + 1, dtype=np.float64)
        new[0] = cp
        new[1:] = growth[:-1]
        new[self.R] += growth[self.R]  # dobra o overflow no bin "≥R_max" (truncagem)
        s = new.sum()
        self.prob = new / s if s > 0 else new

        ns = np.empty(self.R + 1, dtype=np.float64)
        ns[0] = 0.0
        ns[1:] = self.sum_e2[:-1] + se
        ns[self.R] = self.sum_e2[self.R] + se  # o bin truncado continua acumulando
        self.sum_e2 = ns

    def features(self) -> dict[str, float]:
        if self.t < self.warmup:
            return {
                "bocpd_regime_var_ln": _NAN, "bocpd_cp_prob": _NAN,
                "bocpd_rl_entropy": _NAN, "bocpd_map_runlen": _NAN,
            }
        beta = self.b0 + self.sum_e2 / 2.0
        e_var = beta / self._alpha_m1                       # E[σ²|r]
        regime_var = float(np.dot(self.prob, e_var))
        p = self.prob
        pp = p[p > 1e-12]  # evita log(0) (numpy avalia ambos os ramos de np.where)
        entropy = float(-np.sum(pp * np.log(pp)))
        cp_prob = float(p[: self.recent_k].sum())
        map_rl = int(np.argmax(p))
        return {
            "bocpd_regime_var_ln": math.log(max(regime_var, 1e-12)),
            "bocpd_cp_prob": cp_prob,
            "bocpd_rl_entropy": entropy,
            "bocpd_map_runlen": math.log1p(map_rl),
        }


def history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio BOCPDBlock sobre o histórico (H0), para a calibração F1. `map_runlen` não é
    calibrado (é um localizador de idade, não uma magnitude)."""
    blk = BOCPDBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name == "bocpd_map_runlen":
                continue
            acc.setdefault(name, []).append(val)
    return acc
