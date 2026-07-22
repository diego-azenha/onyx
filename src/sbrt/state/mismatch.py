"""MismatchBlock — o filtro congelado do histórico ainda vale online? (F2, docs/BACKLOG_TSAUC.md)

## O que já existia, e por que este bloco é menor do que o plano supunha

O plano consolidado pede "congelar o AR(10) do histórico, aplicar no online e monitorar o erro de
predição um-passo". **Isso já é a arquitetura inteira**: `H0Params` é `frozen=True` e não tem
`.refit()` (bloqueio B2), e `h0.py:whiten_step` aplica o filtro do histórico em todo ponto online.
`e` *é* o erro de predição um-passo do filtro congelado. Logo:

- crescimento da variância residual  -> já emitido por `accum_window_var_ln_w*` (que é literalmente
  log da razão variância-online/variância-histórico, pois `e` já vem dividido por `sigma_e`);
- autocorrelação residual lag-1      -> já emitida por `accum_*_rho1_fz` e `dep_absrho1/sqrho1`.

O que **não** existia é a parte multi-lag, e é só ela que este bloco acrescenta.

## As três direções, e o cuidado de não duplicar

1. **Portmanteau sobre `e`** (níveis, escala congelada): Σ_{k=1..L} ρ_k². É o teste direto de "o
   AR(10) do histórico ainda branqueia esta série". O banco tinha massa multi-lag sobre `|e|`
   (`dep_mass_abs`) e sobre `e_vol` (`dep_mass_evol`), mas **nunca sobre `e` puro** — e é o `e` puro
   que testa o filtro, porque `|e|` mede volatilidade e `e_vol` já teve a escala reajustada.
2. **McLeod-Li sobre `e²`** (efeito ARCH multi-lag): `dep_sqrho1_w*` cobre só o lag 1. Uma quebra
   pura em clustering de volatilidade pode viver em lags maiores.
3. **CUSUM de escore multi-lag**: resposta rápida em janela curta. `cusum_dep_*` faz isto para o
   lag 1 sobre `e_vol`; aqui o escore é agregado sobre lags 1..L e roda sobre `e`.

Sobre o ponto de Berkes, Gombay, Horváth e Kokoszka (2004): a calibração sequencial deve vir do
**escore**, não de `e²` cru, porque resíduos ao quadrado não satisfazem o FCLT de Wiener. Por isso o
detector recursivo (item 3) opera sobre o produto defasado — que é o escore da autocovariância — e o
`e²` aparece só nas estatísticas de janela (item 2), onde a calibração é empírica por série (F1) e
não depende de teoria assintótica.

Fluxo (plano §3.4): tudo usa `e` — escala congelada. Este bloco existe justamente para medir
desalinhamento CONTRA o histórico, então reajustar a escala com `e_vol` apagaria parte do sinal que
ele procura (a trava CE2 vale na direção oposta: variância/cauda nunca em `e_vol`).
"""
from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from sbrt.state.dependence import _RollingAutocorr

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


class MismatchBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        mc = cfg.mismatch
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(mc.windows)
        self.arch_windows = list(mc.arch_windows)
        self.L = mc.max_lag
        self.arch_L = mc.arch_max_lag

        self.white = {w: _RollingAutocorr(w, self.L) for w in self.windows}
        self.arch = {w: _RollingAutocorr(w, self.arch_L) for w in self.arch_windows}

        # escore da autocovariância: produto defasado agregado sobre 1..L. sigma_u é o dp de
        # e_t*e_{t-1} medido no histórico — o mesmo normalizador que CusumBlock usa, e a melhor
        # estimativa disponível da escala do produto para esta série (uma série de cauda pesada tem
        # sigma_u alto e não deve disparar por isso).
        self.sigma_u = h0.sigma_u if h0 is not None else 1.0
        self.delta = mc.cusum_delta
        self.recent: deque = deque(maxlen=self.L)
        self.score_pos = 0.0
        self.score_neg = 0.0
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        for w in self.windows:
            self.white[w].update(e)
        se = e * e
        for w in self.arch_windows:
            self.arch[w].update(se)

        n = len(self.recent)
        if n > 0:
            # escore agregado; dividir por sqrt(n) mantém variância ~1 sob H0 (os produtos de lags
            # distintos são não-correlacionados), então o mesmo `delta` vale em qualquer L
            s = 0.0
            for k in range(1, n + 1):
                s += e * self.recent[-k]
            u = s / (math.sqrt(n) * max(self.sigma_u, 1e-8))
            d = self.delta
            self.score_pos = max(0.0, self.score_pos + d * u - d * d / 2.0)
            self.score_neg = max(0.0, self.score_neg - d * u - d * d / 2.0)
        self.recent.append(e)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            out[f"mismatch_white_e_w{w:03d}"] = self.white[w].mass() if warm else _NAN
        for w in self.arch_windows:
            out[f"mismatch_arch_e2_w{w:03d}"] = self.arch[w].mass() if warm else _NAN
        out["mismatch_score_cusum_pos"] = self.score_pos if warm else _NAN
        out["mismatch_score_cusum_neg"] = self.score_neg if warm else _NAN
        return out


def history_null_series(e_hist, cfg) -> dict:
    """Passada contínua do PRÓPRIO MismatchBlock sobre o histórico, para as estatísticas de JANELA
    (F1). Rodar o bloco real — em vez de uma reimplementação vetorizada — garante por construção que
    o nulo é medido com a mesma estatística do online.

    Só as de janela saem por aqui. Os dois CUSUMs de escore são recursivos (partem de 0 e têm um
    transiente), então o nulo deles vem de réplicas com reinício — ver
    `calibration.py:_add_from_replicates` e a whitelist `calibration.recursive_features`."""
    blk = MismatchBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        e = float(ev)
        blk.update(e, e, e, i)
        for name, val in blk.features().items():
            if name.startswith("mismatch_score_cusum"):
                continue
            acc.setdefault(name, []).append(val)
    return acc


def history_null_replicates(
    e_hist, e_vol_hist, h0: "H0Params", cfg: "Config", restart_every: int,
    max_reps: int = 0, wanted: frozenset = frozenset(),
) -> dict:
    """Réplicas com reinício, para os CUSUMs de escore. Mesmo contrato de
    `cusum.history_null_series` — ver aquela docstring."""
    import numpy as np

    e = np.asarray(e_hist, dtype=np.float64)
    K = int(restart_every)
    n_reps = len(e) // K
    if max_reps > 0:
        n_reps = min(n_reps, int(max_reps))
    if n_reps < 4 or K < 2:
        return {}

    acc: dict = {}
    for r in range(n_reps):
        blk = MismatchBlock()
        blk.reset(h0, cfg)
        base = r * K
        for j in range(K):
            ev = float(e[base + j])
            blk.update(ev, ev, ev, j + 1)
            feats = blk.features()
            for name in (wanted or feats):
                val = feats.get(name)
                if val is None:
                    continue
                mat = acc.get(name)
                if mat is None:
                    mat = acc[name] = np.full((n_reps, K), np.nan, dtype=np.float64)
                mat[r, j] = val
    return acc
