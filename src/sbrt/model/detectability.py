"""X1 (BRAINSTORM_RUPTURA_TSAUC.md §3): detectabilidade por série, computada INLINE no treino.

`model/weights.py` usa a detectabilidade `d_i` para tirar peso dos positivos que não carregam sinal no
passo em que pesam (magnitude baixa; linhas logo após τ). O experimento offline consumia
`artifacts/reports/detectability.csv` (gerado por `scripts/detectability_report.py` a partir do censo
A1 `scripts/break_type_census.py`), mas a nuvem treina do zero e não tem esse CSV -- então o cálculo é
reproduzido aqui, model-free, a partir dos próprios registros de treino (mesma matemática dos dois
scripts): `analyze_series` (censo por série via fit_h0 + whiten_step) e `estimate_detectability`
(divergência multi-eixo × √m_bucket).

`d_i` só é finito para séries de quebra PRECOCE (0<τ<t_max): m_bucket = min(t_max−τ, n_post). As
demais séries ficam FORA do mapa e recebem multiplicador 1 em `compute_row_weights` -- idêntico ao
comportamento do experimento (detectability.csv só continha as ~712 séries com τ<50)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as spstats

from sbrt.state.h0 import fit_h0, seed_lag_buffer, whiten_step

# Eixos do censo A1 usados como divergência (o canal de média fica de fora: medido como fraco). Idêntico
# a scripts/detectability_report.py:AXES.
AXES = ("delta_logvar_e", "delta_rho1", "delta_kurt", "delta_exceed")
MIN_SEGMENT = 10  # pontos mínimos de cada lado de τ (idêntico a scripts/break_type_census.py)


def _acf1(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    xc = x - x.mean()
    den = np.dot(xc, xc)
    return float(np.dot(xc[:-1], xc[1:]) / den) if den > 0 else np.nan


def analyze_series(x_hist: np.ndarray, x_online: np.ndarray, tau_index: int, cfg) -> dict | None:
    """Censo model-free de uma série em τ (fit_h0 + whiten_step causal). Idêntico a
    scripts/break_type_census.py:analyze_series."""
    if tau_index < MIN_SEGMENT or (len(x_online) - tau_index) < MIN_SEGMENT:
        return None

    h0 = fit_h0(x_hist, cfg)
    lags = seed_lag_buffer(h0)
    e_vals = np.empty(len(x_online))
    for i, x in enumerate(x_online):
        e, _ = whiten_step(float(x), lags, h0, cfg)
        e_vals[i] = e

    pre_e, post_e = e_vals[:tau_index], e_vals[tau_index:]
    var_pre_e = max(pre_e.var(ddof=1), 1e-12)
    var_post_e = max(post_e.var(ddof=1), 1e-12)
    delta_logvar_e = float(np.log(var_post_e) - np.log(var_pre_e))
    delta_rho1 = _acf1(post_e) - _acf1(pre_e)
    kurt_pre = spstats.kurtosis(pre_e, fisher=True, bias=False) if len(pre_e) > 3 else np.nan
    kurt_post = spstats.kurtosis(post_e, fisher=True, bias=False) if len(post_e) > 3 else np.nan
    delta_kurt = float(kurt_post - kurt_pre)
    delta_exceed = float(np.mean(np.abs(post_e) > 2) - np.mean(np.abs(pre_e) > 2))
    # C2(ii) (CAMPANHA_POLIMENTO.md frente C): deslocamento de NÍVEL do resíduo. Sempre computado --
    # é uma coluna a mais no censo, inerte enquanto não entrar em `AXES` (ver `estimate_detectability`).
    # Em unidades do desvio pré-τ, para não deixar a escala bruta de `e` dominar a norma L2 dos eixos.
    delta_mean_e = float((post_e.mean() - pre_e.mean()) / np.sqrt(var_pre_e))
    return {
        "delta_logvar_e": delta_logvar_e,
        "delta_rho1": float(delta_rho1),
        "delta_kurt": delta_kurt,
        "delta_exceed": delta_exceed,
        "delta_mean_e": delta_mean_e,
        "n_post": len(post_e),
    }


def estimate_detectability(census: pd.DataFrame, t_max: int, axes=None) -> pd.DataFrame:
    """delta_efetivo × √m_bucket, delta_efetivo = norma L2 dos eixos padronizados pelo desvio ENTRE
    séries. Idêntico a scripts/detectability_report.py:estimate_detectability.

    `axes=None` usa `AXES` (os quatro eixos do censo A1) -- o default reproduz bit-a-bit o X1 adotado.
    C2(ii) passa `AXES + ("delta_mean_e",)` para medir o braço que a campanha listou e nunca rodou."""
    out = census.copy()
    z = np.zeros(len(out), dtype=np.float64)
    for axis in (AXES if axes is None else tuple(axes)):
        v = out[axis].to_numpy(dtype=np.float64)
        sd = np.nanstd(v)
        if sd > 0:
            z = z + np.nan_to_num((v / sd) ** 2)
    out["divergence"] = np.sqrt(z)
    tau = out["tau_index"].to_numpy(dtype=np.float64)
    m_bucket = np.clip(t_max - tau, 0.0, None)
    m_bucket = np.minimum(m_bucket, out["n_post"].to_numpy(dtype=np.float64))
    out["m_bucket"] = m_bucket
    out["detectability"] = out["divergence"] * np.sqrt(m_bucket)
    return out


def compute_detectability_map(records, cfg, t_max: int = 50, n_jobs: int = 1) -> pd.DataFrame:
    """Censo + detectabilidade dos registros de treino (só os que têm quebra). Devolve
    DataFrame[id, detectability, tau_index] restrito a m_bucket>0 (quebra precoce) -- o resto fica fora
    e recebe peso 1. `records`: iterável de model/dataset.py:SeriesRecord."""
    recs = [r for r in records if r.tau_index is not None]

    def _one(r):
        res = analyze_series(r.x_hist, r.x_online, int(r.tau_index), cfg)
        if res is None:
            return None
        res["id"] = int(r.dataset_id)
        res["tau_index"] = int(r.tau_index)
        return res

    if n_jobs == 1:
        rows = [_one(r) for r in recs]
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(_one)(r) for r in recs)
    rows = [x for x in rows if x is not None]
    if not rows:
        return pd.DataFrame(columns=["id", "detectability", "tau_index"])

    axes = AXES + ("delta_mean_e",) if getattr(cfg.weights, "detect_include_delta_mean", False) else None
    det = estimate_detectability(pd.DataFrame(rows), t_max, axes=axes)
    det = det[det["m_bucket"] > 0]  # só quebra precoce (τ<t_max), como o experimento
    return det[["id", "detectability", "tau_index"]].reset_index(drop=True)
