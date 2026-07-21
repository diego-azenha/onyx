"""DependenceBlock — dependência serial não-linear e multi-lag (docs/INVESTIGACAO_FALHAS_V3.md P1).

## Por que existe

O cruzamento censo×OOF (INVESTIGACAO §1) mostrou que quebras *puras* de dependência (Δρ₁ alto, Δlogvar
baixo) têm detectabilidade **0,492 — abaixo do acaso** — apesar de o limite de Neyman-Pearson (§2.1)
mostrar que uma mudança de ρ₁ de magnitude moderada é altamente detectável (0,81–0,99 com janela
média/longa). É o maior ponto cego do modelo, e um eixo de sinal *independente* da variância (β=+0,04,
corr 0,14 com variância).

O banco só media dependência **linear lag-1**: `accum_*_rho1_fz` (Fisher-z de ρ₁ de e_vol), `cusum_dep`
(produto defasado), `mmd_joint` (conjunta lag-1). O SHAP transversal (INVESTIGACAO §4.1) mostrou os
lineares clássicos **mortos** (0,1–0,6%); só o MMD-joint vive (~10%), e mesmo ele não crava as quebras
de dependência. Este bloco cobre o que faltava:

- **Clustering de volatilidade** (ρ₁ de |e| e de e²): uma quebra pode mudar a *persistência* da
  volatilidade sem mudar seu nível médio. Nada online via isso (só o `meta_h0_acf_e2_l1` estático, da
  F2). Bônus: separa GARCH de quebra-de-nível de variância — um cluster GARCH tem ρ₁(e²) alto
  *persistente* (no histórico e no online), então a versão calibrada (contra o nulo da própria série)
  fica baixa; uma quebra de nível dá um ρ₁(e²) transitório *em excesso* sobre o nulo — ataca T6.
- **Massa multi-lag** (Σ_{k=1}^{L} ρ_k²): dependência em lags > 1 que o lag-1 sozinho perde.

Roteamento (plano §3.4): |e| e e² usam `e` (escala congelada — família de variância/cauda, trava CE2);
a massa linear usa `e_vol` (vol-ajustado — dependência de média/forma), consistente com o resto do banco.
Custo medido: ~1 µs/passo (produtos defasados incrementais, O(L)).
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


def _d(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RollingAutocorr:
    """Autocorrelação de janela, O(L) por passo, para lags 1..max_lag. Estimador enviesado padrão:
    ρ_k = (E_jan[v_t·v_{t-k}] − v̄²) / (E_jan[v²] − v̄²), tudo sobre a mesma janela deslizante."""

    __slots__ = ("W", "L", "val", "sv", "svv", "prod", "sp", "recent")

    def __init__(self, window: int, max_lag: int):
        self.W = window
        self.L = max_lag
        self.val = RingBuffer(window)
        self.sv = 0.0
        self.svv = 0.0
        self.prod = [RingBuffer(window) for _ in range(max_lag)]
        self.sp = [0.0] * max_lag
        self.recent: deque = deque(maxlen=max_lag)  # v_{t-1}, ..., v_{t-L}

    def update(self, v: float) -> None:
        for k in range(1, self.L + 1):
            if len(self.recent) >= k:
                p = v * self.recent[-k]  # recent[-1]=v_{t-1}, recent[-k]=v_{t-k}
                ev = self.prod[k - 1].push(p)
                self.sp[k - 1] += p - _d(ev)
        ev = self.val.push(v)
        self.sv += v - _d(ev)
        self.svv += v * v - _d(ev) ** 2
        self.recent.append(v)

    def _mean_var(self):
        n = len(self.val)
        if n < 2:
            return None
        mean = self.sv / n
        var = self.svv / n - mean * mean
        return mean, var, n

    def rho(self, k: int) -> float:
        mv = self._mean_var()
        if mv is None:
            return _NAN
        mean, var, _ = mv
        nk = len(self.prod[k - 1])
        if nk < 1 or var <= 1e-12:
            return 0.0
        autocov = self.sp[k - 1] / nk - mean * mean
        return autocov / var

    def mass(self) -> float:
        mv = self._mean_var()
        if mv is None:
            return _NAN
        s = 0.0
        for k in range(1, self.L + 1):
            r = self.rho(k)
            s += r * r
        return s


class DependenceBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        dc = cfg.dependence
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(dc.windows)
        self.mass_window = dc.mass_window
        self.abs_ac = {w: _RollingAutocorr(w, 1) for w in self.windows}
        self.sq_ac = {w: _RollingAutocorr(w, 1) for w in self.windows}
        self.mass_abs = _RollingAutocorr(dc.mass_window, dc.mass_max_lag)
        self.mass_evol = _RollingAutocorr(dc.mass_window, dc.mass_max_lag)
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        ae = abs(e)
        se = e * e
        for w in self.windows:
            self.abs_ac[w].update(ae)
            self.sq_ac[w].update(se)
        self.mass_abs.update(ae)
        self.mass_evol.update(e_vol)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            out[f"dep_absrho1_w{w:03d}"] = self.abs_ac[w].rho(1) if warm else _NAN
            out[f"dep_sqrho1_w{w:03d}"] = self.sq_ac[w].rho(1) if warm else _NAN
        mw = self.mass_window
        out[f"dep_mass_abs_w{mw:03d}"] = self.mass_abs.mass() if warm else _NAN
        out[f"dep_mass_evol_w{mw:03d}"] = self.mass_evol.mass() if warm else _NAN
        return out


def history_null_series(e_hist, cfg) -> dict:
    """Roda o PRÓPRIO DependenceBlock sobre o histórico (H0 por definição) e devolve a série de cada
    feature e-based, para a calibração de nulo por série (F1, state/calibration.py). Rodar o bloco
    real — em vez de uma reimplementação vetorizada — garante por construção que o nulo é medido
    exatamente com a mesma estatística do online (elimina o risco de desalinhamento que exigiu testes
    dedicados no MMD/Haar). `e_vol` é aproximado por `e` no histórico (o ajuste de volatilidade
    converge a ~1 sobre o histórico estacionário); por isso `dep_mass_evol_*` NÃO é calibrado."""
    blk = DependenceBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name.startswith("dep_mass_evol"):
                continue
            acc.setdefault(name, []).append(val)
    return acc
