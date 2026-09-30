"""Modelo por série, ajustado uma vez no histórico (knowledge/frentes/surpresa-acumulada/README.md §1).

Leva cada ponto a dois fluxos que sob H0 são N(0,1) iid em QUALQUER série:
- fluxo A (escala congelada): resíduo AR / sigma do histórico -> PIT -> Phi^-1. Vigia variância e
  cauda; não pode ter volatilidade adaptativa, senão absorve a quebra de variância (trava CE2 do
  Onyx, MODELO.md §3.4).
- fluxo B (vol-ajustado): resíduo AR / sqrt(v_t) de um GARCH(1,1) com variance targeting -> PIT ->
  Phi^-1. Vigia média, dependência e clustering sem a confusão da volatilidade.
Tudo causal: o resíduo em t usa x_{t-p..t-1}; a variância v_t usa r_{t-1} e v_{t-1}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import lfilter
from scipy.special import ndtri

_LOG_MIN = -700.0


@dataclass(frozen=True)
class RefPIT:
    """CDF empírica interpolada, com cauda exponencial além dos quantis q e 1-q."""
    ordenado: np.ndarray
    probs: np.ndarray
    x_lo: float
    x_hi: float
    p_lo: float
    p_hi: float
    beta_lo: float
    beta_hi: float


@dataclass(frozen=True)
class ModeloSerie:
    phi: np.ndarray          # coeficientes AR dos lags 1..p
    c: float
    sigma: float             # escala congelada do resíduo (fluxo A)
    omega: float             # GARCH (fluxo B)
    alpha: float
    beta: float
    ref_a: RefPIT
    ref_b: RefPIT
    cauda_x: np.ndarray      # últimos p valores do histórico (contexto dos lags)
    r2_ult: float            # último resíduo^2 do histórico (estado GARCH)
    v_ult: float             # última variância condicional do histórico
    gb_ult: float            # último g do fluxo B no histórico (lag da alternativa de correlação)
    ar_r2: float             # diagnóstico: R^2 do AR no histórico


def _ajustar_ar(x: np.ndarray, p: int) -> tuple[np.ndarray, float, np.ndarray, float]:
    lags = sliding_window_view(x[:-1], p)[:, ::-1]          # linha t: x_{t-1..t-p} para o alvo x[p+t]
    y = x[p:]
    X = np.column_stack([np.ones(len(y)), lags])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    var_y = float(np.var(y))
    ar_r2 = 1.0 - float(np.var(r)) / var_y if var_y > 0 else 0.0
    return coef[1:].copy(), float(coef[0]), r, ar_r2


def residuos_ar(x_contexto: np.ndarray, x_novo: np.ndarray, phi: np.ndarray, c: float) -> np.ndarray:
    """Resíduo AR de cada ponto de `x_novo`, usando os últimos p pontos de `x_contexto` como lags."""
    p = len(phi)
    x_all = np.concatenate([x_contexto[-p:], x_novo])
    lags = sliding_window_view(x_all[:-1], p)[:, ::-1]
    return x_novo - c - lags @ phi


def variancia_garch(r2: np.ndarray, omega: float, alpha: float, beta: float,
                    r2_ant: float, v_ant: float) -> np.ndarray:
    """v_t = omega + alpha r2_{t-1} + beta v_{t-1}; (r2_ant, v_ant) = estado imediatamente anterior."""
    s = omega + alpha * np.concatenate([[r2_ant], r2[:-1]])
    return lfilter([1.0], [1.0, -beta], s, zi=[beta * v_ant])[0]


def _escolher_garch(r: np.ndarray, sigma2: float, cfg: dict) -> tuple[float, float, float]:
    melhor, arg = -np.inf, (sigma2, 0.0, 0.0)
    r2 = r * r
    for a in cfg["garch"]["alphas"]:
        for pers in cfg["garch"]["persistencias"] if a > 0 else [0.0]:
            b = pers - a
            if b < 0:
                continue
            omega = sigma2 * (1.0 - pers) if a > 0 else sigma2
            v = variancia_garch(r2, omega, a, b, sigma2, sigma2)
            ql = -0.5 * float(np.sum(np.log(v) + r2 / v))
            if ql > melhor:
                melhor, arg = ql, (omega, a, b)
    return arg


def _ref_pit(z: np.ndarray, q: float) -> RefPIT:
    s = np.sort(z)
    n = len(s)
    probs = (np.arange(n) + 0.5) / n
    k_lo = max(int(np.floor(q * n)), 1)
    k_hi = min(n - 1 - k_lo, n - 2)
    x_lo, x_hi = float(s[k_lo]), float(s[k_hi])
    beta_lo = float(np.mean(x_lo - s[:k_lo])) or 1e-3
    beta_hi = float(np.mean(s[k_hi + 1:] - x_hi)) or 1e-3
    return RefPIT(s, probs, x_lo, x_hi, float(probs[k_lo]), float(probs[k_hi]),
                  max(beta_lo, 1e-3), max(beta_hi, 1e-3))


def pit_normal(z: np.ndarray, ref: RefPIT) -> tuple[np.ndarray, np.ndarray]:
    """(g, u): g = Phi^-1(F(z)), u = F(z). Caudas em log para não perder resolução nos extremos."""
    z = np.asarray(z, dtype=np.float64)
    u = np.interp(z, ref.ordenado, ref.probs)
    g = ndtri(np.clip(u, 1e-300, 1.0 - 1e-16))
    lo = z < ref.x_lo
    if lo.any():
        logf = np.maximum(np.log(ref.p_lo) + (z[lo] - ref.x_lo) / ref.beta_lo, _LOG_MIN)
        u[lo] = np.exp(logf)
        g[lo] = ndtri(u[lo])
    hi = z > ref.x_hi
    if hi.any():
        logs = np.maximum(np.log1p(-ref.p_hi) - (z[hi] - ref.x_hi) / ref.beta_hi, _LOG_MIN)
        sobrev = np.exp(logs)
        u[hi] = 1.0 - sobrev
        g[hi] = -ndtri(sobrev)
    return g, u


def ajustar(hist: np.ndarray, cfg: dict) -> ModeloSerie:
    hist = np.asarray(hist, dtype=np.float64)
    p = int(cfg["ar_p"])
    phi, c, r, ar_r2 = _ajustar_ar(hist, p)
    sigma2 = max(float(np.mean(r * r)), 1e-12)
    omega, a, b = _escolher_garch(r, sigma2, cfg)
    v = variancia_garch(r * r, omega, a, b, sigma2, sigma2)
    q = float(cfg["pit"]["cauda_q"])
    z_b = r / np.sqrt(v)
    ref_b = _ref_pit(z_b, q)
    return ModeloSerie(
        phi=phi, c=c, sigma=float(np.sqrt(sigma2)), omega=omega, alpha=a, beta=b,
        ref_a=_ref_pit(r / np.sqrt(sigma2), q), ref_b=ref_b,
        cauda_x=hist[-p:].copy(), r2_ult=float(r[-1] ** 2), v_ult=float(v[-1]),
        gb_ult=float(pit_normal(z_b[-1:], ref_b)[0][0]), ar_r2=ar_r2,
    )


def fluxos(modelo: ModeloSerie, x_novo: np.ndarray) -> dict[str, np.ndarray]:
    """Fluxos A e B (g e u) dos pontos novos, em lote. Causal ponto a ponto."""
    r = residuos_ar(modelo.cauda_x, np.asarray(x_novo, dtype=np.float64), modelo.phi, modelo.c)
    v = variancia_garch(r * r, modelo.omega, modelo.alpha, modelo.beta, modelo.r2_ult, modelo.v_ult)
    g_a, u_a = pit_normal(r / modelo.sigma, modelo.ref_a)
    g_b, _ = pit_normal(r / np.sqrt(v), modelo.ref_b)
    return {"g_a": g_a, "u_a": u_a, "g_b": g_b}
