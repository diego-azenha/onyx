"""Alternativas direcionais e acumulação de evidência (knowledge/frentes/surpresa-acumulada/README.md §2).

Cada alternativa k define um incremento l_k(t) = log p_k(g_t | passado) - log phi(g_t), a
log-razão de verossimilhança exata contra o nulo universal g ~ N(0,1) iid. A evidência acumula por
Shiryaev-Roberts (soma sobre todos os instantes de início possíveis) e por CUSUM (máximo sobre eles).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import gammaln, logsumexp


class Alternativas:
    """Lista ordenada (determinística) de alternativas; `familias` agrupa colunas por família."""

    def __init__(self, cfg: dict):
        alt = cfg["alternativas"]
        self.cols: list[tuple[str, str, float]] = []   # (familia, tipo, parametro)
        for s in alt["escala_sobe"]:
            self.cols.append(("escala_sobe", "escala", float(s)))
        for s in alt["escala_desce"]:
            self.cols.append(("escala_desce", "escala", float(s)))
        for nu in alt["cauda_nu"]:
            self.cols.append(("cauda", "t", float(nu)))
        th = float(alt["forma_theta"])
        for k in (3, 4):
            for sinal in (1.0, -1.0):
                self.cols.append((f"forma{k}", f"legendre{k}", sinal * th))
        for d in alt["media_delta"]:
            for sinal in (1.0, -1.0):
                self.cols.append(("media", "media", sinal * float(d)))
        for rho in alt["correlacao_rho"]:
            for sinal in (1.0, -1.0):
                self.cols.append(("correlacao", "correlacao", sinal * float(rho)))
        for a in alt["arch_a"]:
            self.cols.append(("arch", "arch", float(a)))
        self.h_min = float(alt["arch_h_min"])
        self.K = len(self.cols)
        nomes = []
        for fam, _, _ in self.cols:
            if fam not in nomes:
                nomes.append(fam)
        self.familias = {f: np.array([i for i, c in enumerate(self.cols) if c[0] == f]) for f in nomes}

    def incrementos(self, g_a: np.ndarray, u_a: np.ndarray, g_b: np.ndarray, g_b_ant: np.ndarray) -> np.ndarray:
        """Matriz (T, K). `g_b_ant[t]` = g_b no passo anterior (o último do contexto em t=0)."""
        g_a = np.asarray(g_a, dtype=np.float64)
        T = len(g_a)
        L = np.empty((T, self.K))
        ga2 = g_a * g_a
        x = 2.0 * np.asarray(u_a) - 1.0
        leg = {3: math.sqrt(7.0) * (5.0 * x**3 - 3.0 * x) / 2.0,
               4: 3.0 * (35.0 * x**4 - 30.0 * x**2 + 3.0) / 8.0}
        gb2_ant = g_b_ant * g_b_ant
        for j, (_, tipo, par) in enumerate(self.cols):
            if tipo == "escala":
                L[:, j] = -math.log(par) + 0.5 * ga2 * (1.0 - 1.0 / (par * par))
            elif tipo == "t":
                nu = par
                k = math.sqrt(nu / (nu - 2.0))          # t de variância unitária: g = t / k
                y = g_a * k
                logt = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * math.log(nu * math.pi)
                        - (nu + 1) / 2 * np.log1p(y * y / nu)) + math.log(k)
                L[:, j] = logt + 0.5 * ga2 + 0.5 * math.log(2 * math.pi)
            elif tipo.startswith("legendre"):
                L[:, j] = np.log1p(par * leg[int(tipo[-1])])
            elif tipo == "media":
                L[:, j] = par * g_b - 0.5 * par * par
            elif tipo == "correlacao":
                rho = par
                q = 1.0 - rho * rho
                d = g_b - rho * g_b_ant
                L[:, j] = -0.5 * math.log(q) - d * d / (2 * q) + 0.5 * g_b * g_b
            elif tipo == "arch":
                h = np.maximum(self.h_min, 1.0 + par * (gb2_ant - 1.0))
                L[:, j] = -0.5 * np.log(h) - g_b * g_b / (2 * h) + 0.5 * g_b * g_b
        return L

    def media_nula(self, n_nos: int) -> np.ndarray:
        return _media_nula(tuple(self.cols), self.h_min, n_nos)

    def var_longo_prazo_nula(self, n_nos: int) -> np.ndarray:
        return _var_lp_nula(tuple(self.cols), self.h_min, n_nos)


@lru_cache(maxsize=8)
def _media_nula(cols: tuple, h_min: float, n_nos: int) -> np.ndarray:
    """E_H0[l_k] (= -KL(N(0,1) || alternativa k)) por Gauss-Hermite 2D em (g_t, g_{t-1})."""
    from scipy.special import ndtr
    nos, pesos = np.polynomial.hermite_e.hermegauss(n_nos)
    pesos = pesos / pesos.sum()
    G, Gant = np.meshgrid(nos, nos, indexing="ij")
    W = np.outer(pesos, pesos).ravel()
    g, gant = G.ravel(), Gant.ravel()
    alt = Alternativas.__new__(Alternativas)
    alt.cols, alt.h_min, alt.K = list(cols), h_min, len(cols)
    L = alt.incrementos(g, ndtr(g), g, gant)
    return W @ L


@lru_cache(maxsize=8)
def _var_lp_nula(cols: tuple, h_min: float, n_nos: int) -> np.ndarray:
    """Variância de longo prazo de l_k sob H0: Var + 2 Cov(l_t, l_{t+1}). Os incrementos dependem de
    (g_{t-1}, g_t), então só o lag 1 é não nulo; Gauss-Hermite 3D em (g_{t-1}, g_t, g_{t+1})."""
    from scipy.special import ndtr
    nos, pesos = np.polynomial.hermite_e.hermegauss(n_nos)
    pesos = pesos / pesos.sum()
    A, B, C = np.meshgrid(nos, nos, nos, indexing="ij")
    W = np.einsum("i,j,k->ijk", pesos, pesos, pesos).ravel()
    a, b, c = A.ravel(), B.ravel(), C.ravel()
    alt = Alternativas.__new__(Alternativas)
    alt.cols, alt.h_min, alt.K = list(cols), h_min, len(cols)
    L1 = alt.incrementos(b, ndtr(b), b, a)          # l_t
    L2 = alt.incrementos(c, ndtr(c), c, b)          # l_{t+1}
    mu = W @ L1
    var = W @ (L1 * L1) - mu * mu
    cov = W @ (L1 * L2) - mu * mu
    return var + 2.0 * cov


def acumular(L: np.ndarray, log_r0: np.ndarray | None = None, w0: np.ndarray | None = None):
    """Shiryaev-Roberts em log e CUSUM, passo a passo. Devolve (logR (T,K), W (T,K))."""
    T, K = L.shape
    log_r = np.full(K, -np.inf) if log_r0 is None else log_r0.copy()
    w = np.zeros(K) if w0 is None else w0.copy()
    out_r = np.empty((T, K))
    out_w = np.empty((T, K))
    for t in range(T):
        log_r = np.logaddexp(0.0, log_r) + L[t]
        w = np.maximum(0.0, w + L[t])
        out_r[t] = log_r
        out_w[t] = w
    return out_r, out_w


def agregar(log_r: np.ndarray) -> np.ndarray:
    """Score único: mistura uniforme das alternativas, logsumexp_k log R_k - log K."""
    return logsumexp(log_r, axis=-1) - math.log(log_r.shape[-1])
