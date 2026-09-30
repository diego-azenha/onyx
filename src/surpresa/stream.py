"""Pontuação de uma série: em lote (avaliação) e passo a passo (o caminho que vira StateBlock no S4).

Os dois caminhos usam as mesmas funções (`modelo.pit_normal`, `Alternativas.incrementos`); o teste
`tests/surpresa/test_stream.py` exige que concordem, e que o lote seja causal (prefixo).
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from surpresa.evidencia import Alternativas, acumular, agregar
from surpresa.modelo import ModeloSerie, ajustar, fluxos, pit_normal

SAIDAS_BASE = ("sr", "sr_eta", "sr_cal", "cusum", "ingenua_z")


def calibracao_por_direcao(hist: np.ndarray, cfg: dict, alt: Alternativas) -> tuple[np.ndarray, np.ndarray]:
    """(vies, eta) de cada direção, medidos no 1/3 final do histórico (H0 fora da amostra) com o
    modelo ajustado nos 2/3 iniciais.

    - eta_k = Var_LP_teórica / Var_LP_medida, cortado em [eta_min, 1]: tempera a razão de
      verossimilhança onde a série tem mais variância de incremento do que o nulo N(0,1) prevê
      (clustering de volatilidade no fluxo A, por exemplo). S0 rodada 1 mostrou que sem isso séries
      GARCH acumulam evidência falsa (knowledge/frentes/surpresa-acumulada/S0-sanidade-sintetica.md).
    - vies_k = média medida - média teórica (-KL), encolhido por James-Stein positivo contra o próprio
      erro-padrão: a correção crua da rodada 1 injetava ruído que o logsumexp transformava em inflação.
    """
    cal = cfg["calibracao"]
    zeros, uns = np.zeros(alt.K), np.ones(alt.K)
    n_fit = int(len(hist) * float(cal["frac_ajuste"]))
    fora = hist[n_fit:]
    if len(fora) < int(cal["min_holdout"]) or n_fit < 5 * int(cfg["ar_p"]):
        return zeros, uns
    m = ajustar(hist[:n_fit], cfg)
    f = fluxos(m, fora)
    g_b_ant = np.concatenate([[m.gb_ult], f["g_b"][:-1]])
    L = alt.incrementos(f["g_a"], f["u_a"], f["g_b"], g_b_ant)
    b = int(cal["lote"])
    n_lotes = len(L) // b
    medias_lote = L[: n_lotes * b].reshape(n_lotes, b, alt.K).mean(axis=1)
    var_lp = b * medias_lote.var(axis=0, ddof=1)
    var_lp_nula = alt.var_longo_prazo_nula(int(cfg["quadratura_nos"]))
    eta = np.clip(var_lp_nula / np.maximum(var_lp, 1e-12), float(cal["eta_min"]), 1.0)
    vies = L.mean(axis=0) - alt.media_nula(int(cfg["quadratura_nos"]))
    ep2 = var_lp / len(L)
    vies = vies * np.maximum(0.0, 1.0 - float(cal["js_k"]) * ep2 / np.maximum(vies * vies, 1e-300))
    return vies, eta


def _incrementos_cal(L: np.ndarray, vies: np.ndarray, eta: np.ndarray) -> np.ndarray:
    """[cru | temperado | temperado e sem viés], lado a lado: 3K colunas numa recursão só."""
    return np.concatenate([L, eta * L, eta * (L - vies)], axis=-1)


def _saidas(log_r3: np.ndarray, w3: np.ndarray, soma_ing, t, alt: Alternativas) -> dict[str, np.ndarray]:
    K = alt.K
    log_r, log_r_eta, log_r_cal = log_r3[..., :K], log_r3[..., K:2 * K], log_r3[..., 2 * K:]
    w = w3[..., :K]
    out = {
        "sr": agregar(log_r),
        "sr_eta": agregar(log_r_eta),
        "sr_cal": agregar(log_r_cal),
        "cusum": w.max(axis=-1),
        "ingenua_z": soma_ing / np.sqrt(t / 2.0),
    }
    for fam, idx in alt.familias.items():
        out[f"fam_{fam}"] = log_r_eta[..., idx].max(axis=-1)
    return out


def pontuar_serie(hist: np.ndarray, online: np.ndarray, cfg: dict,
                  alt: Alternativas | None = None) -> dict[str, np.ndarray]:
    """Todas as saídas para cada passo online (arrays de comprimento len(online)), em lote."""
    alt = alt or Alternativas(cfg)
    hist = np.asarray(hist, dtype=np.float64)
    m = ajustar(hist, cfg)
    vies, eta = calibracao_por_direcao(hist, cfg, alt)
    f = fluxos(m, online)
    g_b_ant = np.concatenate([[m.gb_ult], f["g_b"][:-1]])
    L = alt.incrementos(f["g_a"], f["u_a"], f["g_b"], g_b_ant)
    log_r3, w3 = acumular(_incrementos_cal(L, vies, eta))
    t = np.arange(1, len(L) + 1, dtype=np.float64)
    soma_ing = np.cumsum(0.5 * (f["g_a"] ** 2 - 1.0))
    return _saidas(log_r3, w3, soma_ing, t, alt)


class SurpresaStream:
    """Mesmo cálculo de `pontuar_serie`, um ponto por vez. O(p + K) por passo."""

    def __init__(self, hist: np.ndarray, cfg: dict, alt: Alternativas | None = None):
        self.alt = alt or Alternativas(cfg)
        hist = np.asarray(hist, dtype=np.float64)
        self.m: ModeloSerie = ajustar(hist, cfg)
        self.vies, self.eta = calibracao_por_direcao(hist, cfg, self.alt)
        self.lags = deque(self.m.cauda_x.tolist(), maxlen=len(self.m.phi))
        self.r2_ant, self.v_ant, self.gb_ant = self.m.r2_ult, self.m.v_ult, self.m.gb_ult
        self.log_r = np.full(3 * self.alt.K, -np.inf)
        self.w = np.zeros(3 * self.alt.K)
        self.soma_ing = 0.0
        self.t = 0

    def update(self, x: float) -> dict[str, float]:
        m = self.m
        lags = np.array(self.lags)[::-1]                       # x_{t-1}, ..., x_{t-p}
        r = float(x) - m.c - float(lags @ m.phi)
        v = m.omega + m.alpha * self.r2_ant + m.beta * self.v_ant
        g_a, u_a = pit_normal(np.array([r / m.sigma]), m.ref_a)
        g_b, _ = pit_normal(np.array([r / math.sqrt(v)]), m.ref_b)
        L = self.alt.incrementos(g_a, u_a, g_b, np.array([self.gb_ant]))[0]
        L3 = _incrementos_cal(L, self.vies, self.eta)
        self.log_r = np.logaddexp(0.0, self.log_r) + L3
        self.w = np.maximum(0.0, self.w + L3)
        self.soma_ing += 0.5 * (float(g_a[0]) ** 2 - 1.0)
        self.t += 1
        self.lags.append(float(x))
        self.r2_ant, self.v_ant, self.gb_ant = r * r, v, float(g_b[0])
        out = _saidas(self.log_r, self.w, np.float64(self.soma_ing), np.float64(self.t), self.alt)
        return {k: float(v) for k, v in out.items()}
