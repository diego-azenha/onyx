"""Fase-histórico: caracterização do regime H0 e whitening causal (plano §3).

Executado uma vez por série, sobre o histórico completo (livre de quebra por definição). Tudo aqui é
determinístico e O(n_h*p + n_h log n_h). `H0Params` é imutável — não existe `.refit()`: torna
estruturalmente impossível reestimar o H0 no meio do online (bloqueio B2 do plano técnico, §2.2-B2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from sbrt.state.calibration import compute_null_stats
from sbrt.state.fingerprint import compute_fingerprint
from sbrt.state.mmd import history_reference as mmd_history_reference
from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config


@dataclass(frozen=True)
class H0Params:
    phi: np.ndarray
    c: float
    mu0: float
    sigma0: float
    sigma_e: float
    sigma_e_rob: float
    nu_hat: float
    q: dict
    sorted_e_hist: np.ndarray
    sorted_abs_e_hist: np.ndarray
    sigma_u: float
    rho1_e: float
    rho1_abs_e: float
    seasonal_lag: int | None
    seasonal_coef: float
    ar_r2: float
    n_h: int
    last_hist_e: float
    lag_seed: np.ndarray
    # --- proposta V2 (docs/PROPOSTA_FEATURES_V2.md), tudo calculado uma vez aqui e nunca no online ---
    fingerprint: dict          # F2: descritores estendidos do regime H0 (state/fingerprint.py)
    rff_href: np.ndarray       # F3: média de z(e) sobre o histórico, referência congelada do MMD
    rff_href_joint: np.ndarray # F3: idem para o par (e_t, e_{t-1})
    null_stats: dict           # F1: {feature: (mu, sd, min_t)} do nulo da própria série

    @property
    def lag_capacity(self) -> int:
        base = len(self.phi)
        return max(base, self.seasonal_lag or 0)


def _design_matrix(hist: np.ndarray, lags: list) -> tuple[np.ndarray, np.ndarray]:
    """Monta [1, x_{t-l1}, x_{t-l2}, ...] -> x_t para t = max(lags)+1 .. n_h (1-based)."""
    max_lag = max(lags)
    n_h = len(hist)
    rows = n_h - max_lag
    X = np.empty((rows, len(lags) + 1), dtype=np.float64)
    X[:, 0] = 1.0
    for i, lag in enumerate(lags):
        X[:, i + 1] = hist[max_lag - lag: max_lag - lag + rows]
    y = hist[max_lag: max_lag + rows]
    return X, y


def _acf(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    if lag <= 0 or lag >= n:
        return 0.0
    num = float(np.dot(x[: n - lag], x[lag:]))
    den = float(np.dot(x, x))
    return num / den if den > 0 else 0.0


def fit_h0(hist: np.ndarray, cfg: "Config") -> H0Params:
    """plano §3.1. Puro e determinístico. ValueError se n_h < mínimo configurado."""
    hist = np.asarray(hist, dtype=np.float64)
    n_h = len(hist)
    if n_h < cfg.h0.min_hist_len:
        raise ValueError(f"histórico com {n_h} pontos, mínimo exigido {cfg.h0.min_hist_len}")

    mu0 = float(hist.mean())
    sigma0 = float(hist.std(ddof=1)) if n_h > 1 else 1.0
    sigma0 = max(sigma0, 1e-8)

    p = cfg.h0.ar_order
    base_lags = list(range(1, p + 1))

    # AR(p) via mínimos quadrados (equivalente a Yule-Walker/Levinson-Durbin para este propósito,
    # e permite adicionar o lag sazonal ao mesmo design matrix sem recursão separada, §3.1 item 3).
    var_x = float(hist.var(ddof=1)) if n_h > 1 else 1.0
    var_x = max(var_x, 1e-12)

    if n_h > p + 5:
        X, y = _design_matrix(hist, base_lags)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        var_resid = float(resid.var(ddof=1)) if len(resid) > 1 else var_x
        ar_r2 = 1.0 - var_resid / var_x
    else:
        coef = np.zeros(p + 1)
        resid = hist - mu0
        ar_r2 = 0.0

    accept_ar = ar_r2 >= cfg.h0.ar_r2_min_reduction
    if accept_ar:
        c = float(coef[0])
        phi = coef[1:].copy()
    else:
        c = mu0
        phi = np.zeros(p)
        resid = hist[p:] - mu0

    seasonal_lag: int | None = None
    seasonal_coef = 0.0
    lo, hi = cfg.h0.seasonal_lag_range
    if len(resid) > hi + 10:
        best_lag, best_abs_rho = None, cfg.h0.seasonal_acf_threshold
        for lag in range(lo, hi + 1):
            rho = _acf(resid, lag)
            if abs(rho) > best_abs_rho:
                best_abs_rho = abs(rho)
                best_lag = lag
        if best_lag is not None:
            seasonal_lags = base_lags + [best_lag]
            X2, y2 = _design_matrix(hist, seasonal_lags)
            coef2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
            resid2 = y2 - X2 @ coef2
            var_resid2 = float(resid2.var(ddof=1)) if len(resid2) > 1 else var_x
            ar_r2 = 1.0 - var_resid2 / var_x
            c = float(coef2[0])
            phi = coef2[1: 1 + p].copy()
            seasonal_coef = float(coef2[-1])
            seasonal_lag = best_lag
            resid = resid2

    sigma_e = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
    sigma_e = max(sigma_e, 1e-8)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_e_rob = max(1.4826 * mad, 1e-8)

    e_hist = resid / sigma_e

    m2 = float(np.mean(e_hist ** 2))
    m4 = float(np.mean(e_hist ** 4))
    kappa_ex = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0
    nu_lo, nu_hi = cfg.h0.nu_clip
    nu_hat = float(np.clip(4.0 + 6.0 / kappa_ex, nu_lo, nu_hi)) if kappa_ex > 0 else nu_hi

    q = {f"{level:.2f}": float(np.quantile(e_hist, level)) for level in cfg.h0.quantile_levels}

    rho1_e = _acf(e_hist, 1)
    rho1_abs_e = _acf(np.abs(e_hist), 1)

    if len(e_hist) > 2:
        u = e_hist[1:] * e_hist[:-1]
        sigma_u = float(u.std(ddof=1)) if len(u) > 1 else 1.0
    else:
        sigma_u = 1.0
    sigma_u = max(sigma_u, 1e-8)

    sorted_e_hist = np.sort(e_hist)
    sorted_abs_e_hist = np.sort(np.abs(e_hist))
    last_hist_e = float(e_hist[-1])

    lag_capacity = max(p, seasonal_lag or 0)
    lag_seed = hist[-lag_capacity:].copy()

    # --- proposta V2: tudo abaixo é função APENAS do histórico (H0 por definição), calculado uma
    # vez por série. Nada disso adiciona custo ao laço online. ---
    fingerprint = compute_fingerprint(e_hist, hist, q, cfg)
    rff_href, rff_href_joint = mmd_history_reference(e_hist, cfg)
    null_stats = compute_null_stats(
        e_hist, sorted_e_hist, sorted_abs_e_hist, rff_href, rff_href_joint, cfg
    )

    return H0Params(
        phi=phi,
        c=c,
        mu0=mu0,
        sigma0=sigma0,
        sigma_e=sigma_e,
        sigma_e_rob=sigma_e_rob,
        nu_hat=nu_hat,
        q=q,
        sorted_e_hist=sorted_e_hist,
        sorted_abs_e_hist=sorted_abs_e_hist,
        sigma_u=sigma_u,
        rho1_e=rho1_e,
        rho1_abs_e=rho1_abs_e,
        seasonal_lag=seasonal_lag,
        seasonal_coef=seasonal_coef,
        ar_r2=ar_r2,
        n_h=n_h,
        last_hist_e=last_hist_e,
        lag_seed=lag_seed,
        fingerprint=fingerprint,
        rff_href=rff_href,
        rff_href_joint=rff_href_joint,
        null_stats=null_stats,
    )


def seed_lag_buffer(params: H0Params) -> RingBuffer:
    """Semeia o ring de lags com a cauda do histórico (`params.lag_seed`, guardada por `fit_h0`) —
    garante continuidade exata na fronteira histórico->online (plano §3.1 item 8, armadilha §13.3):
    e_1 do online usa os mesmos lags que teriam sido usados se o histórico continuasse."""
    buf = RingBuffer(params.lag_capacity)
    for x in params.lag_seed:
        buf.push(float(x))
    return buf


def whiten_step(x: float, lags: RingBuffer, params: H0Params, cfg: "Config") -> tuple[float, float]:
    """plano §3.2. Retorna (e_clipado, e_raw); empurra x em `lags`. `params` é imutável — nunca
    reestimado no online (bloqueio B2)."""
    x_hat = params.c
    for j, phi_j in enumerate(params.phi):
        x_hat += phi_j * lags.peek(j)
    if params.seasonal_lag is not None:
        x_hat += params.seasonal_coef * lags.peek(params.seasonal_lag - 1)

    e_raw = (x - x_hat) / params.sigma_e
    lo, hi = cfg.h0.clip_e
    e_clip = float(np.clip(e_raw, lo, hi))

    lags.push(float(x))
    return e_clip, float(e_raw)
