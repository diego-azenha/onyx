"""TrajectoryBlock — a forma da rampa do estatístico, não o valor instantâneo (F4+F9,
docs/BACKLOG_TSAUC.md).

## O que ataca

Uma quebra estrutural produz evidência **sustentada**: a rampa sobe e *fica*. Um outlier transitório
— um draw raro de uma distribuição de cauda pesada mas estacionária — produz um **spike que reverte**.
Nos dois casos o valor instantâneo num dado passo pode ser idêntico; o que os separa é a trajetória.
Distinguir sustentado de transitório é exatamente o problema de falso-positivo (T6/T9 vs T3), e esta
família ataca isso mais diretamente que qualquer feature marginal nova.

Como subproduto, exporta a *idade da evidência* que o compass queria para ponderação condicional à
idade da quebra — **sem** precisar de um segundo modelo (F10).

## Por que este bloco quebra o contrato `StateBlock`, e como

`StateBlock.update(e, e_raw, e_vol, t)` recebe a inovação, não as saídas dos outros blocos. Mas
trajetória é, por definição, meta-estatística *sobre* estatísticos já calculados. O contrato novo
espelha o padrão que `apply_calibration` já estabeleceu — consumir o dict `feats` depois do laço:

    def update_from_feats(self, feats: dict[str, float], t: int) -> None

`StreamScorer.update_features` chama isto **depois** de `apply_calibration`, de propósito: assim a
trajetória é medida sobre estatísticos **já calibrados** contra o nulo da própria série. A inclinação
de um z-score está em unidades de z por passo — comparável entre séries por construção, que é a moeda
da TS-AUC. Medir a trajetória do valor CRU reintroduziria a escala idiossincrática por série que F1
acabou de remover, e as features de trajetória não teriam um nulo próprio para corrigir isso (medi-lo
exigiria replayar o pipeline inteiro sobre o histórico).

Consequência: as features daqui **não** têm versão `_cal`, e não precisam — já nascem em unidades
calibradas.

## Largura

O conjunto rastreado vem do YAML (`trajectory.track`), curto de propósito: cada estatístico
acompanhado multiplica a largura pelo número de descritores. Sob `feature_fraction=0,8`, largura sem
sinal dilui o sorteio (docs/NOTAS_AGENTES.md §7).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.utils.numerics import ewma_update

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


class _Trajectory:
    """Descritores recursivos O(1) da rampa de UM estatístico."""

    __slots__ = ("lam", "thr", "prev", "slope", "mono", "area", "persist", "since_first", "n")

    def __init__(self, lam: float, thr: float):
        self.lam = lam
        self.thr = thr
        self.prev: float | None = None
        self.slope = 0.0
        self.mono = 0.0
        self.area = 0.0          # integrador tipo Page-Hinkley (F9): acumula o excesso sobre o
        self.persist = 0         # limiar e é refletido em 0, então deriva lenta acumula e ruído não
        self.since_first = -1
        self.n = 0

    def update(self, value: float) -> None:
        if not math.isfinite(value):
            return  # estatístico ainda em warm-up: não contamina a trajetória com um zero inventado
        self.n += 1
        if self.prev is not None:
            d = value - self.prev
            self.slope = ewma_update(self.slope, d, self.lam)
            self.mono = ewma_update(self.mono, 1.0 if d > 0 else (-1.0 if d < 0 else 0.0), self.lam)
        self.prev = value

        above = value > self.thr
        self.persist = self.persist + 1 if above else 0
        if above and self.since_first < 0:
            self.since_first = 0
        elif self.since_first >= 0:
            self.since_first += 1
        self.area = max(0.0, self.area + (value - self.thr))

    def features(self, prefix: str, warm: bool) -> dict[str, float]:
        if not warm or self.n < 2:
            return {
                f"traj_{prefix}_slope": _NAN,
                f"traj_{prefix}_mono": _NAN,
                f"traj_{prefix}_persist": _NAN,
                f"traj_{prefix}_since_first": _NAN,
                f"traj_{prefix}_area": _NAN,
            }
        return {
            f"traj_{prefix}_slope": self.slope,
            f"traj_{prefix}_mono": self.mono,
            f"traj_{prefix}_persist": float(self.persist),
            # -1 = nunca excedeu. Um sentinela numérico é justificável aqui (ao contrário da regra
            # geral de NaN): "nunca excedeu" é informação real e ordenável, não ausência de dado.
            f"traj_{prefix}_since_first": float(self.since_first),
            f"traj_{prefix}_area": self.area,
        }


class TrajectoryBlock:
    """Contrato próprio: `update_from_feats` em vez de `update` — ver a docstring do módulo."""

    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        tc = cfg.trajectory
        self.warmup = cfg.features.warmup_min_n
        self.track = tuple(tc.track)  # ((feature, alias), ...) — tupla ordenada, iteração determinística
        self.trackers = {alias: _Trajectory(tc.ewma_lambda, tc.threshold) for _, alias in self.track}
        self.t = 0

    def update_from_feats(self, feats: dict, t: int) -> None:
        self.t = t
        for name, alias in self.track:
            self.trackers[alias].update(feats.get(name, _NAN))

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for _, alias in self.track:
            out.update(self.trackers[alias].features(alias, warm))
        return out
