"""ConformalBlock — martingales conformais sobre p-values causais das inovações contra a
distribuição do histórico (plano §4.2 #23; Vovk et al. 2005; Volkhonskiy et al. 2017).

Evidência livre de distribuição, O(log n_h)/passo via busca binária nos arrays ordenados do
histórico (`H0Params.sorted_e_hist` / `sorted_abs_e_hist`). Opera sobre a escala congelada (nunca
`e_vol`): os arrays ordenados de referência foram construídos a partir do resíduo/sigma_e do
histórico, e comparar `e_vol` contra eles seria inconsistente de escala quando o ajuste de
volatilidade está ativo.

## A6 (CAMPANHA_POLIMENTO.md) — `e` clipado contra um histórico NÃO clipado

A referência é `e_hist = resid / sigma_e` (`state/h0.py`), construída **sem clip**. O valor online,
porém, chega aqui já clipado em `cfg.h0.clip_e` = [-8, 8] (`whiten_step`). A comparação é portanto
ASSIMÉTRICA na cauda: toda observação com `|e_raw| > 8` é ranqueada como se valesse exatamente 8,0
contra um histórico que preservou os seus próprios extremos, então o p-value dela satura na massa de
cauda do histórico acima de 8 em vez de cair para o piso `1/(n_h+1)`.

Isso importa porque (a) `conformal_logm_abs` é a feature nº 1 do modelo (xs-SHAP 0,061; conv_share
0,147) e (b) MEDIDO em `train_rows_bocpd`: o clip morde em 7,50% das séries, e de forma
DIFERENCIAL por classe -- 6,99% das linhas positivas contra 2,64% das negativas (2,6x). O clip está
comprimindo justamente o material que separa as classes.

Um p-value de rank é robusto por construção -- o clip não lhe compra estabilidade nenhuma, só custa
resolução. `cfg.conformal.use_raw` restaura a simetria (estatística e nulo na mesma escala).

Três variantes de p-value (todas via mid-rank, para lidar com empates):
- abs: cauda superior de |e_t| contra |e| do histórico — sensível a variância/cauda.
- right: cauda superior de e_t (com sinal) — sensível a shift positivo/skew à direita.
- sign: cauda inferior de e_t (com sinal) — sensível a shift negativo/skew à esquerda.

Cada uma vira um log-martingale (mistura sobre epsilons, "apostas" de Vovk); "6->4 usadas" (plano
tabela §5 #23): abs tem variante acumulada E com reset (SR-like), right/sign só acumuladas.
"""
from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING

import numpy as np

from sbrt.utils.numerics import logsumexp

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params


def _mid_rank(sorted_arr, x: float) -> float:
    lo = bisect.bisect_left(sorted_arr, x)
    hi = bisect.bisect_right(sorted_arr, x)
    return (lo + hi) / 2.0


def _upper_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (n - rank + 0.5) / (n + 1.0)


def _lower_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (rank + 0.5) / (n + 1.0)


class ConformalBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.sorted_e_hist = h0.sorted_e_hist
        self.sorted_abs_e_hist = h0.sorted_abs_e_hist
        self.n_h = h0.n_h
        self.epsilons = list(cfg.conformal.epsilons)
        self._log_k = math.log(len(self.epsilons))
        # A6: ver a nota no topo do módulo. O histórico de referência não é clipado; usar `e_raw`
        # põe a estatística e o nulo dela na mesma escala.
        self.use_raw = getattr(cfg.conformal, "use_raw", False)

        self.L_abs = {eps: 0.0 for eps in self.epsilons}
        self.L_abs_reset = {eps: 0.0 for eps in self.epsilons}
        self.L_right = {eps: 0.0 for eps in self.epsilons}
        self.L_sign = {eps: 0.0 for eps in self.epsilons}

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        z = e_raw if self.use_raw else e
        p_abs = _upper_tail_p(self.sorted_abs_e_hist, abs(z), self.n_h)
        p_right = _upper_tail_p(self.sorted_e_hist, z, self.n_h)
        p_sign = _lower_tail_p(self.sorted_e_hist, z, self.n_h)

        log_p_abs = math.log(p_abs)
        log_p_right = math.log(p_right)
        log_p_sign = math.log(p_sign)

        for eps in self.epsilons:
            log_eps = math.log(eps)
            inc_abs = log_eps + (eps - 1.0) * log_p_abs
            self.L_abs[eps] += inc_abs
            self.L_abs_reset[eps] = max(0.0, self.L_abs_reset[eps] + inc_abs)
            self.L_right[eps] += log_eps + (eps - 1.0) * log_p_right
            self.L_sign[eps] += log_eps + (eps - 1.0) * log_p_sign

    def features(self) -> dict[str, float]:
        return {
            "conformal_logm_abs": logsumexp(list(self.L_abs.values())) - self._log_k,
            "conformal_logm_abs_reset": logsumexp(list(self.L_abs_reset.values())) - self._log_k,
            "conformal_logm_right": logsumexp(list(self.L_right.values())) - self._log_k,
            "conformal_logm_sign": logsumexp(list(self.L_sign.values())) - self._log_k,
        }


def history_null_series(
    e_hist, e_vol_hist, h0: "H0Params", cfg: "Config", restart_every: int,
    max_reps: int = 0, wanted: frozenset = frozenset(),
) -> dict:
    """Réplicas com reinício do próprio ConformalBlock sobre o histórico (F1.b-1). Mesmo contrato de
    `cusum.history_null_series` — ver aquela docstring para o porquê das réplicas.

    **Por que estas features precisam de calibração mais do que qualquer outra.** Sob H0 o incremento
    de cada aposta tem esperança negativa: com p ~ U(0,1) vale E[log p] = −1, logo
    E[inc] = log ε + (ε−1)(−1) = log ε − ε + 1 < 0 para todo ε ≠ 1. As variantes sem reset
    (`abs`, `right`, `sign`) portanto **derivam linearmente em t**, com inclinação que depende da
    série (via o quanto a ECDF do histórico se ajusta às inovações dela). No corte transversal de um
    passo t, o nível dessas features é dominado por t·deriva(série) — uma escala idiossincrática que
    nada tem a ver com quebra, e que a TS-AUC pune diretamente. Daí `kind="cumsum"`.

    **Viés in-sample.** Rodar o bloco sobre o próprio histórico compara cada ponto contra uma ECDF que
    o contém, enquanto o online compara pontos novos. A diferença no p-value de mid-rank é O(1/n_h) —
    com n_h típico de 3.000, ~0,03%. Medido e desprezível frente ao dp do nulo, então não vale a
    complexidade de leave-one-out ou split do histórico."""
    e = np.asarray(e_hist, dtype=np.float64)
    e_vol = np.asarray(e_vol_hist, dtype=np.float64)
    K = int(restart_every)
    n_reps = len(e) // K
    if max_reps > 0:
        n_reps = min(n_reps, int(max_reps))
    if n_reps < 4 or K < 2:
        return {}

    acc: dict = {}
    for r in range(n_reps):
        blk = ConformalBlock()
        blk.reset(h0, cfg)
        base = r * K
        for j in range(K):
            i = base + j
            ev = float(e[i])
            blk.update(ev, ev, float(e_vol[i]), j + 1)
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
