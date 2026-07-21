"""VarLocBlock — variância localizada no changepoint (docs/INVESTIGACAO_FALHAS_V3.md P3).

## Por que existe

A maior folga de EXTRAÇÃO do modelo (INVESTIGACAO §2–3): um detector ótimo de variância que conhece
tau atinge AUC≈0,856; o V3 fica em 0,604, e a folga é máxima em t alto. A hipótese diagnosticada (§3):
**toda janela fixa (accum_window_var_ln_w010..w250) dilui o sinal porque mistura pontos pré e
pós-quebra** — uma janela de 250 com tau no meio estima uma variância *atenuada*. O oracle usa só os
pontos pós-tau; o modelo não sabe onde é tau.

Este bloco não estima tau explicitamente — ele **seleciona a escala** que melhor revela a elevação de
variância, o que atinge o mesmo efeito de forma auto-contida:

- `varloc_max_z` = max sobre escalas d∈{10..250} do z padronizado de ln(E[e²]) na janela-d recente.
  Uma quebra de idade ~a é melhor vista com d≈a; ao maximizar sobre d, a feature *localiza a escala*
  automaticamente (uma quebra recente acende as escalas curtas; uma antiga, as longas), em vez de
  diluir num comprimento fixo. z_d = (ln(mean e²_d) + 1/n) / sqrt(2/n), n=min(t,d) — a padronização
  teórica i.i.d.-gaussiana; a inflação por curtose da série é corrigida pela calibração F1.
- `varloc_min_z` = min sobre d (elevação negativa: melhor escala para uma *queda* de variância).
- `varloc_argmax_lnscale` = ln(d*) da escala que maximiza z — um localizador barato (quão recente é
  a elevação mais forte).
- `varloc_recent_vs_lagged` = ln(E[e²] dos últimos R) − ln(E[e²] da janela [R, R+L) atrás): contraste
  direto "regime recente vs. regime anterior", sensível justamente quando a variância mudou de patamar.

Consome `e` (escala congelada) — família de variância, trava CE2 (plano §3.4). A versão calibrada
(F1) usa o nulo de max_z/min_z da própria série, o que remove a inflação por curtose (D-10) que
tornaria a padronização teórica incomparável entre séries.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


def _d0(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RollingSum:
    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, v: float) -> None:
        ev = self.ring.push(v)
        self.total += v - _d0(ev)

    def mean(self) -> float:
        n = len(self.ring)
        return self.total / n if n > 0 else _NAN

    def __len__(self) -> int:
        return len(self.ring)


class VarLocBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        vc = cfg.varloc
        self.warmup = cfg.features.warmup_min_n
        self.scales = list(vc.scales)
        self.recent = vc.recent
        self.lagged = vc.lagged
        self.t = 0
        needed = sorted(set(self.scales) | {self.recent, self.recent + self.lagged})
        self.sums = {w: _RollingSum(w) for w in needed}

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        se = e * e
        for s in self.sums.values():
            s.push(se)

    def _z(self, d: int) -> float:
        n = min(self.t, d)
        if n < 2:
            return _NAN
        mean_e2 = self.sums[d].mean()
        ln_v = math.log(max(mean_e2, 1e-12))
        return (ln_v + 1.0 / n) / math.sqrt(2.0 / n)

    def features(self) -> dict[str, float]:
        if self.t < self.warmup:
            return {
                "varloc_max_z": _NAN, "varloc_min_z": _NAN,
                "varloc_argmax_lnscale": _NAN, "varloc_recent_vs_lagged": _NAN,
            }
        zs = [(self._z(d), d) for d in self.scales]
        zs = [(z, d) for z, d in zs if not math.isnan(z)]
        if zs:
            zmax, dmax = max(zs, key=lambda p: p[0])
            zmin, _ = min(zs, key=lambda p: p[0])
            argmax_ln = math.log(dmax)
        else:
            zmax = zmin = argmax_ln = _NAN

        rl = _NAN
        s_full = self.sums[self.recent + self.lagged]
        s_rec = self.sums[self.recent]
        if len(s_full) >= self.recent + self.lagged:
            mean_rec = s_rec.mean()
            mean_lag = (s_full.total - s_rec.total) / self.lagged
            rl = math.log(max(mean_rec, 1e-12)) - math.log(max(mean_lag, 1e-12))

        return {
            "varloc_max_z": zmax,
            "varloc_min_z": zmin,
            "varloc_argmax_lnscale": argmax_ln,
            "varloc_recent_vs_lagged": rl,
        }


def history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio VarLocBlock sobre o histórico (H0), para a calibração F1. `argmax_lnscale` NÃO
    é calibrado (é um índice de escala, não uma magnitude)."""
    blk = VarLocBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name == "varloc_argmax_lnscale":
                continue
            acc.setdefault(name, []).append(val)
    return acc
