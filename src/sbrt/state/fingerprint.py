"""Impressão digital estendida do regime H0 (docs/PROPOSTA_FEATURES_V2.md F2).

Motivação empírica. A decomposição de SHAP do modelo mostrou que as features `meta_h0_*` —
constantes dentro de uma série — são a **maior família do modelo (34,3% do |SHAP|)**, apesar de o
CE6 provar que elas não carregam efeito principal (classificador só-histórico: AUC 0,5067). Ou seja,
um terço da capacidade explicativa do modelo é gasto **calibrando**: decidindo o quanto um dado
desvio é surpreendente *para uma série com esta cara*. O modelo está faminto por contexto de
calibração — e dar-lhe um retrato melhor da série custa **zero latência por passo** (tudo é
calculado uma vez em `fit_h0`).

Todos os descritores são funções apenas do histórico (H0 por definição) e portanto não podem vazar
rótulo; a checagem CE6 (`scripts/ce6_history_classifier.py`) deve continuar ≈0,5 depois desta adição
— se subir, é sinal de que o gerador correlaciona propriedades do histórico com a existência de
quebra, e isso mudaria a leitura de várias decisões do projeto.
"""
from __future__ import annotations

import math

import numpy as np


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def _acf(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 0:
        return 0.0
    return float(np.dot(xc[:-lag], xc[lag:]) / den)


def _hurst_aggvar(e: np.ndarray, scales: list) -> float:
    """Hurst pelo método da variância agregada: Var(média de blocos de tamanho m) ∝ m^(2H−2).
    H≈0,5 para ruído branco; H>0,5 indica memória longa (persistência)."""
    xs, ys = [], []
    n = len(e)
    for m in scales:
        if m < 1 or n // m < 8:
            continue
        k = n // m
        blocks = e[: k * m].reshape(k, m).mean(axis=1)
        v = float(blocks.var(ddof=1)) if k > 1 else 0.0
        if v > 0:
            xs.append(math.log(m))
            ys.append(math.log(v))
    if len(xs) < 3:
        return 0.5
    slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
    return _safe(1.0 + slope / 2.0, 0.5)


def _hill_xi(e: np.ndarray, frac: float) -> float:
    """Estimador de Hill do índice de cauda sobre |e|. Retorna xi = 1/alpha (xi maior = cauda mais
    pesada); 0 para cauda fina. Complementa `nu_hat` (que vem da curtose e é sensível a outliers de
    forma diferente)."""
    a = np.sort(np.abs(np.asarray(e, dtype=np.float64)))[::-1]
    n = len(a)
    k = max(10, int(frac * n))
    if n < 20 or k >= n:
        return 0.0
    thresh = a[k]
    if thresh <= 1e-12:
        return 0.0
    top = a[:k]
    top = top[top > 0]
    if len(top) < 2:
        return 0.0
    return _safe(float(np.mean(np.log(top / thresh))), 0.0)


def _spectral_slope(e: np.ndarray) -> float:
    """Inclinação da log-periodograma vs. log-frequência. ≈0 para ruído branco; negativa indica
    dominância de baixa frequência (drift/memória longa); positiva, alta frequência."""
    n = len(e)
    if n < 64:
        return 0.0
    x = e - e.mean()
    psd = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (freqs > 0) & (psd > 0)
    if mask.sum() < 16:
        return 0.0
    lf, lp = np.log(freqs[mask]), np.log(psd[mask])
    return _safe(float(np.polyfit(lf, lp, 1)[0]), 0.0)


def _ljung_box(x: np.ndarray, max_lag: int) -> float:
    """Q de Ljung-Box normalizada por graus de liberdade (Q/L) — massa de dependência linear.
    Aplicada a |e| mede clustering de volatilidade; ≈1 sob independência."""
    n = len(x)
    if n < max_lag + 10:
        return 1.0
    q = 0.0
    for lag in range(1, max_lag + 1):
        r = _acf(x, lag)
        q += r * r / max(n - lag, 1)
    return _safe(n * (n + 2) * q / max_lag, 1.0)


def compute_fingerprint(e_hist: np.ndarray, hist: np.ndarray, q: dict, cfg) -> dict:
    """Descritores escalares do regime H0. Chamado uma vez por série em `fit_h0`; custo O(n_h log n_h)
    dominado pela FFT/ordenação, desprezível frente ao que `fit_h0` já faz."""
    fp_cfg = cfg.h0_fingerprint
    e = np.asarray(e_hist, dtype=np.float64)
    abs_e = np.abs(e)
    n = len(e)

    lags = list(range(1, fp_cfg.acf_max_lag + 1))
    acf_abs = [abs(_acf(abs_e, l)) for l in lags]
    acf_mass = float(np.mean(acf_abs)) if acf_abs else 0.0

    # decaimento: inclinação de log|acf| vs log(lag) (mais negativa = dependência morre mais rápido)
    xs, ys = [], []
    for l, a in zip(lags, acf_abs):
        if a > 1e-6:
            xs.append(math.log(l))
            ys.append(math.log(a))
    acf_decay = _safe(float(np.polyfit(np.array(xs), np.array(ys), 1)[0]), 0.0) if len(xs) >= 3 else 0.0

    w = fp_cfg.volvol_window
    if n >= 4 * w:
        k = n // w
        block_var = e[: k * w].reshape(k, w).var(axis=1, ddof=1)
        mean_bv = float(block_var.mean())
        volvol = _safe(float(block_var.std(ddof=1)) / mean_bv, 0.0) if mean_bv > 1e-12 else 0.0
    else:
        volvol = 0.0

    q01, q25, q75, q99 = q["0.01"], q["0.25"], q["0.75"], q["0.99"]
    tail_span = q99 - q01
    iqr_tail_ratio = _safe((q75 - q25) / tail_span, 0.0) if tail_span > 1e-12 else 0.0

    return {
        "hurst": _hurst_aggvar(e, list(fp_cfg.hurst_scales)),
        "hill_xi": _hill_xi(e, fp_cfg.hill_frac),
        "acf_e2_l1": _safe(_acf(e * e, 1), 0.0),
        "acf_abs_mass": acf_mass,
        "acf_decay": acf_decay,
        "spectral_slope": _spectral_slope(e),
        "ljungbox_abs": _ljung_box(abs_e, fp_cfg.acf_max_lag),
        "volvol": volvol,
        "iqr_tail_ratio": iqr_tail_ratio,
    }
