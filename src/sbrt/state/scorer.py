"""StreamScorer — motor único (plano §15.1, §8.1): o mesmo laço gera as features de treino
(`model/dataset.py`) e roda na inferência real. Nenhuma implementação vetorizada paralela existe —
isso elimina por construção a classe de bug "backtest vetorizado != execução causal" (armadilha
§13.2, docs/PLANO_REPOSITORIO.md §1).

Features #26 (hedge bruto, precisa de x cru), #27 (meta-t) e #28 (meta H0) e #25 (concordância de
localizadores, cruza bayes+cusum) não cabem no contrato `StateBlock` — são calculadas aqui.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.state.accumulators import AccumulatorBlock
from sbrt.state.bayes_filter import BayesFilterBlock
from sbrt.state.conformal import ConformalBlock
from sbrt.state.cusum import CusumBlock
from sbrt.state.h0 import H0Params, seed_lag_buffer, whiten_step
from sbrt.utils.numerics import ewma_update
from sbrt.utils.ring_buffer import RingBuffer
from sbrt.postprocess.monotonicity import apply as apply_monotonicity

if TYPE_CHECKING:
    from sbrt.config import Config


def default_blocks() -> list:
    return [AccumulatorBlock(), CusumBlock(), BayesFilterBlock(), ConformalBlock()]


class StreamScorer:
    def __init__(self, h0: H0Params, blocks: list, ensemble, cfg: "Config"):
        self.h0 = h0
        self.blocks = blocks
        self.ensemble = ensemble
        self.cfg = cfg

        self.lags = seed_lag_buffer(h0)
        self.t = 0
        self._prev_score: float | None = None

        self.use_vol_adjust = h0.rho1_abs_e > cfg.state.vol_adjust["threshold_rho1_abs"]
        self.lambda_v = cfg.state.vol_adjust["lambda_v"]
        self.v = 1.0

        self._hedge_ewma = 0.0
        self._hedge_ring = RingBuffer(cfg.state.hedge_window)
        self._hedge_sum = 0.0
        self._hedge_sumsq = 0.0

        for b in self.blocks:
            b.reset(h0, cfg)

    def update_features(self, x: float) -> dict[str, float]:
        """Um passo: whiten_step -> update() de cada block -> merge + meta-features (t, n_h, nu_hat,
        rho1, ...). MOTOR ÚNICO: usado tanto por update() quanto por model/dataset.py."""
        self.t += 1
        t = self.t
        e, e_raw = whiten_step(x, self.lags, self.h0, self.cfg)

        if self.use_vol_adjust:
            self.v = ewma_update(self.v, e * e, self.lambda_v)
            e_vol = e / math.sqrt(max(self.v, 1e-12))
        else:
            e_vol = e

        feats: dict[str, float] = {}
        for b in self.blocks:
            b.update(e, e_raw, e_vol, t)
            feats.update(b.features())

        self._update_hedge(x)
        wmin = self.cfg.features.warmup_min_n
        n_eff_hedge = min(t, self.cfg.state.hedge_window)
        feats["hedge_ewma_z"] = self._hedge_ewma / max(self.h0.sigma0, 1e-8) if t >= wmin else math.nan
        if t >= wmin and n_eff_hedge > 1:
            mean_w = self._hedge_sum / n_eff_hedge
            var_w = max(self._hedge_sumsq / n_eff_hedge - mean_w * mean_w, 1e-12)
            feats["hedge_window_var_ln"] = math.log(var_w)
        else:
            feats["hedge_window_var_ln"] = math.nan

        feats["meta_t"] = float(t)
        feats["meta_ln1p_t"] = math.log1p(t)

        h0 = self.h0
        feats["meta_h0_n_h"] = float(h0.n_h)
        feats["meta_h0_nu_hat"] = h0.nu_hat
        feats["meta_h0_rho1_e"] = h0.rho1_e
        feats["meta_h0_rho1_abs_e"] = h0.rho1_abs_e
        feats["meta_h0_ar_r2"] = h0.ar_r2
        feats["meta_h0_seasonal_flag"] = 1.0 if h0.seasonal_lag is not None else 0.0
        feats["meta_h0_q99"] = h0.q["0.99"]
        feats["meta_h0_scale_ratio"] = h0.sigma_e_rob / h0.sigma_e

        age_map = feats.get("bayes_age_map_h0100")
        age_cusum = feats.get("cusum_age_mean_pos_d050")
        if age_map is not None and age_cusum is not None and not (math.isnan(age_map) or math.isnan(age_cusum)):
            feats["meta_locator_diff"] = abs(age_map - age_cusum)
            feats["meta_locator_min"] = min(age_map, age_cusum)
        else:
            feats["meta_locator_diff"] = math.nan
            feats["meta_locator_min"] = math.nan

        return feats

    def _update_hedge(self, x: float) -> None:
        self._hedge_ewma = ewma_update(self._hedge_ewma, x - self.h0.mu0, self.cfg.state.hedge_ewma_lambda)
        evicted = self._hedge_ring.push(x)
        if evicted is None:
            self._hedge_sum += x
            self._hedge_sumsq += x * x
        else:
            self._hedge_sum += x - evicted
            self._hedge_sumsq += x * x - evicted * evicted

    def update(self, x: float) -> float:
        """UMA observação -> UM score em [0,1]."""
        from sbrt.model.fallback import fallback_score  # import tardio: evita ciclo state<->model

        feats = self.update_features(x)
        p = self.ensemble.predict_one(feats) if self.ensemble is not None else fallback_score(feats, self.cfg)
        score = apply_monotonicity(p, self._prev_score, self.cfg.postprocess.mode, self.cfg)
        self._prev_score = score
        return score
