"""Features de duas amostras entre o histórico (H0) e um trecho candidato.

Tudo determinístico, só numpy. `contexto_hist(hist)` é calculado uma vez por série; `features(ctx,
trecho, antes)` compara um trecho com o histórico. `antes` são os valores imediatamente anteriores
ao trecho (lags do AR). Custo ~1-3 ms por chamada para trechos de algumas centenas de pontos.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

P_AR = 5
QS = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
GRADE_W = np.linspace(0.02, 0.98, 25)
LAGS = (1, 2, 3, 5, 10)
N_BANDAS = 6


def _acf(x: np.ndarray, lags=LAGS) -> np.ndarray:
    x = x - x.mean()
    d = float(np.dot(x, x))
    out = np.full(len(lags), 0.0)
    if d <= 0:
        return out
    for i, k in enumerate(lags):
        if k < len(x):
            out[i] = float(np.dot(x[:-k], x[k:])) / d
    return out


def _momentos(x: np.ndarray) -> tuple[float, float]:
    xc = x - x.mean()
    s2 = float(np.mean(xc * xc))
    if s2 <= 0:
        return 0.0, 0.0
    return float(np.mean(xc**3)) / s2**1.5, float(np.mean(xc**4)) / s2**2 - 3.0


def _bandas(x: np.ndarray, nper: int) -> np.ndarray:
    """Fração de potência em N_BANDAS bandas de frequência (média de Welch sem sobreposição)."""
    n = len(x) // nper
    if n < 1:
        return np.full(N_BANDAS, np.nan)
    seg = x[: n * nper].reshape(n, nper)
    seg = seg - seg.mean(axis=1, keepdims=True)
    p = (np.abs(np.fft.rfft(seg * np.hanning(nper), axis=1)) ** 2).mean(axis=0)[1:]
    idx = np.array_split(np.arange(len(p)), N_BANDAS)
    b = np.array([p[i].sum() for i in idx])
    return b / max(b.sum(), 1e-12)


@dataclass
class ContextoHist:
    x: np.ndarray
    phi: np.ndarray
    c: float
    sd: float
    e_sd: float
    reps: dict          # representação -> (ordenado, média, dp, acf)
    qs: dict            # representação -> quantis QS
    wq: dict            # representação -> quantis GRADE_W
    mom: dict
    bandas: dict        # nper -> fração de potência por banda
    resumo: dict


def _reps(x: np.ndarray, e: np.ndarray, e_sd: float) -> dict:
    ez = e / e_sd
    return {"x": x, "ax": np.abs(x), "dx": np.diff(x), "e": ez, "ae": np.abs(ez), "e2": ez * ez}


def contexto_hist(hist: np.ndarray) -> ContextoHist:
    x = np.asarray(hist, dtype=np.float64)
    lags = sliding_window_view(x[:-1], P_AR)[:, ::-1]
    X = np.column_stack([np.ones(len(lags)), lags])
    coef, *_ = np.linalg.lstsq(X, x[P_AR:], rcond=None)
    e = x[P_AR:] - X @ coef
    e_sd = max(float(e.std()), 1e-9)
    reps = _reps(x, e, e_sd)
    ctx_reps, qs, wq, mom = {}, {}, {}, {}
    for k, r in reps.items():
        ctx_reps[k] = (np.sort(r), float(r.mean()), max(float(r.std()), 1e-9), _acf(r))
        qs[k] = np.quantile(r, QS)
        wq[k] = np.quantile(r, GRADE_W)
        mom[k] = _momentos(r)
    # não-estacionariedade do próprio histórico: dispersão da log-variância em blocos de 100
    nb = len(x) // 100
    blocos = x[: nb * 100].reshape(nb, 100)
    lv = np.log(blocos.var(axis=1) + 1e-12)
    resumo = {"h_n": float(len(x)), "h_acf1": float(_acf(x)[0]), "h_kurt": mom["x"][1],
              "h_ar_r2": 1.0 - e.var() / max(x[P_AR:].var(), 1e-12), "h_lv_dp": float(lv.std()),
              "h_e_kurt": mom["e"][1], "h_ae_acf1": float(ctx_reps["ae"][3][0])}
    return ContextoHist(x, coef[1:], float(coef[0]), max(float(x.std()), 1e-9), e_sd, ctx_reps, qs, wq, mom,
                        {n: _bandas(e / e_sd, n) for n in (16, 32)}, resumo)


def features(ctx: ContextoHist, trecho: np.ndarray, antes: np.ndarray) -> dict[str, float]:
    s = np.asarray(trecho, dtype=np.float64)
    ante = np.asarray(antes, dtype=np.float64)[-P_AR:]
    todo = np.concatenate([ante, s])
    lags = sliding_window_view(todo[:-1], P_AR)[:, ::-1]
    e = s - ctx.c - lags @ ctx.phi
    reps = _reps(s, e, ctx.e_sd)
    f: dict[str, float] = dict(ctx.resumo)
    f["n"] = float(len(s))
    for k, r in reps.items():
        ordenado, mu, sd, acf_h = ctx.reps[k]
        f[f"{k}_dmedia"] = (float(r.mean()) - mu) / sd
        f[f"{k}_lvar"] = float(np.log(max(r.var(), 1e-12) / sd**2))
        dq = (np.quantile(r, QS) - ctx.qs[k]) / sd
        for q, v in zip(QS, dq):
            f[f"{k}_dq{int(q * 100):02d}"] = float(v)
        f[f"{k}_w1"] = float(np.mean(np.abs(np.quantile(r, GRADE_W) - ctx.wq[k]))) / sd
        # KS contra a CDF do histórico
        cdf_h = np.searchsorted(ordenado, np.sort(r), side="right") / len(ordenado)
        cdf_s = np.arange(1, len(r) + 1) / len(r)
        f[f"{k}_ks"] = float(np.max(np.abs(cdf_s - cdf_h)))
        sk, ku = _momentos(r)
        f[f"{k}_dskew"] = sk - ctx.mom[k][0]
        f[f"{k}_dkurt"] = ku - ctx.mom[k][1]
        if k in ("x", "e", "ae", "e2"):
            for lag, a, b in zip(LAGS, _acf(r), acf_h):
                f[f"{k}_dacf{lag}"] = a - b
    # excedências de cauda do resíduo contra os quantis do histórico
    qe = ctx.qs["e"]
    f["e_exc_lo01"] = float(np.mean(reps["e"] < qe[0])) / 0.01
    f["e_exc_hi99"] = float(np.mean(reps["e"] > qe[-1])) / 0.01
    f["e_exc_lo05"] = float(np.mean(reps["e"] < qe[1])) / 0.05
    f["e_exc_hi95"] = float(np.mean(reps["e"] > qe[-2])) / 0.05
    # verossimilhança média sob o AR gaussiano do histórico e taxa de cruzamentos de zero
    f["e_llr_gauss"] = float(np.mean(reps["e2"]) - 1.0)
    f["zc_ratio"] = float(np.mean(np.diff(np.sign(s)) != 0)) / max(float(np.mean(np.diff(np.sign(ctx.x)) != 0)), 1e-6)
    nper = 32 if len(s) >= 64 else 16
    b = _bandas(reps["e"], nper)
    bh = ctx.bandas[nper]
    for i in range(N_BANDAS):
        f[f"esp_b{i}"] = float(np.log((b[i] + 1e-6) / (bh[i] + 1e-6))) if np.isfinite(b[i]) else np.nan
    return f
