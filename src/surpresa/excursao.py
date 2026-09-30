"""Calibração das famílias de evidência pelo nulo de EXCURSÕES da própria série
(knowledge/frentes/teto-offline/X1-excursoes.md).

O ST1 mostrou que evidência forte de queda de variância é comum entre negativos: trechos calmos de
séries heteroscedásticas. O η (têmpera) corrige a variância dos incrementos, mas não a frequência de
excursões do CUSUM. Aqui, o CUSUM de cada família roda no 1/3 final do histórico (H0 fora da amostra,
modelo ajustado nos 2/3 iniciais) e o valor online é expresso como quantil dessa distribuição:
exc_f(t) = -log(1 - F_hold(W_f(t)) + 1/(n+1)). Uma série que naturalmente produz excursões grandes
recebe evidência calibrada menor. Causal: só usa o histórico e o online até t.
"""
from __future__ import annotations

import numpy as np

from surpresa.evidencia import Alternativas, acumular
from surpresa.modelo import ajustar, fluxos
from surpresa.stream import calibracao_por_direcao

BURN = 20


def _cusum_familias(L: np.ndarray, eta: np.ndarray, alt: Alternativas) -> dict[str, np.ndarray]:
    _, w = acumular(eta * L)
    return {f: w[:, idx].max(axis=1) for f, idx in alt.familias.items()}


def pontuar_excursao(hist: np.ndarray, online: np.ndarray, cfg: dict, alt: Alternativas) -> dict[str, np.ndarray]:
    hist = np.asarray(hist, dtype=np.float64)
    vies, eta = calibracao_por_direcao(hist, cfg, alt)
    # nulo de excursões: CUSUM temperado no 1/3 final com o modelo dos 2/3
    n_fit = int(len(hist) * float(cfg["calibracao"]["frac_ajuste"]))
    m_h = ajustar(hist[:n_fit], cfg)
    fh = fluxos(m_h, hist[n_fit:])
    Lh = alt.incrementos(fh["g_a"], fh["u_a"], fh["g_b"], np.concatenate([[m_h.gb_ult], fh["g_b"][:-1]]))
    nulo = {f: np.sort(v[BURN:]) for f, v in _cusum_familias(Lh, eta, alt).items()}
    # online com o modelo do histórico inteiro
    m = ajustar(hist, cfg)
    fo = fluxos(m, online)
    Lo = alt.incrementos(fo["g_a"], fo["u_a"], fo["g_b"], np.concatenate([[m.gb_ult], fo["g_b"][:-1]]))
    out = {}
    for f, v in _cusum_familias(Lo, eta, alt).items():
        ref = nulo[f]
        F = np.searchsorted(ref, v, side="right") / (len(ref) + 1.0)
        out[f"exc_{f}"] = -np.log(1.0 - F + 1.0 / (len(ref) + 1.0))
        out[f"cus_{f}"] = v
    return out
