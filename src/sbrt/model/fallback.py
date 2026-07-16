"""Fallback puro-estatístico (plano §8.5) — caminho de emergência determinístico, sem ML. Também
serve de baseline (ii) do gate G-0 e de score por padrão até a camada supervisionada (Frente H) ser
treinada e congelada.

score = sigma(w_lo * LO_{1/400} + w_cusum * sqrt(2 * max(banco_CUSUM)) + w_conformal * logM_abs_reset - bias)

O banco de CUSUM acumula log-likelihood-ratios truncadas em 0 (recursão max); sob H0, 2*LLR se
comporta como qui-quadrado, então sqrt(2*LLR) é uma transformação monótona para uma escala ~z,
usada aqui só para combinar grandezas heterogêneas num único logit (não entra no LightGBM, que
recebe as features cruas — plano §5).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sbrt.config import Config

_CUSUM_BANK_KEYS = (
    "cusum_mean_pos_d025", "cusum_mean_pos_d050", "cusum_mean_pos_d100",
    "cusum_mean_neg_d025", "cusum_mean_neg_d050", "cusum_mean_neg_d100",
    "cusum_var_up_r150", "cusum_var_up_r250", "cusum_var_down_r050",
    "cusum_exceed_q95", "cusum_exceed_q99",
    "cusum_sign_pos", "cusum_sign_neg",
    "cusum_dep_pos", "cusum_dep_neg",
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fallback_score(feats: dict, cfg: "Config") -> float:
    lo = feats.get("bayes_lo_h0025", 0.0)

    max_cusum = 0.0
    for k in _CUSUM_BANK_KEYS:
        v = feats.get(k)
        if v is not None and not math.isnan(v) and v > max_cusum:
            max_cusum = v
    z_cusum = math.sqrt(2.0 * max_cusum)

    logm = feats.get("conformal_logm_abs_reset", 0.0)

    fb = cfg.fallback
    logit = fb.w_lo * lo + fb.w_cusum * z_cusum + fb.w_conformal * logm - fb.bias
    return _sigmoid(logit)
