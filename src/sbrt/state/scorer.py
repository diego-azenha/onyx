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
from sbrt.state.calibration import apply_calibration
from sbrt.state.conformal import ConformalBlock
from sbrt.state.cusum import CusumBlock
from sbrt.state.dependence import DependenceBlock
from sbrt.state.jumps import JumpBlock
from sbrt.state.lmoments import LMomentBlock
from sbrt.state.mmd import MMDBlock
from sbrt.state.multiscale import MultiScaleBlock
from sbrt.state.rank_twosample import RankTwoSampleBlock
from sbrt.state.varloc import VarLocBlock
from sbrt.state.h0 import H0Params, seed_lag_buffer, whiten_step
from sbrt.utils.numerics import ewma_update
from sbrt.utils.ring_buffer import RingBuffer
from sbrt.postprocess.monotonicity import apply as apply_monotonicity

if TYPE_CHECKING:
    from sbrt.config import Config


def default_blocks() -> list:
    return [
        AccumulatorBlock(),
        CusumBlock(),
        BayesFilterBlock(),
        ConformalBlock(),
        RankTwoSampleBlock(),
        MMDBlock(),          # F3 (proposta V2): MMD de kernel via RFF, marginal e conjunto
        MultiScaleBlock(),   # F4 (proposta V2): energia por escala (Haar diádico causal)
        DependenceBlock(),   # P1 (INVESTIGACAO §4.1): dependência não-linear/multi-lag
        LMomentBlock(),      # P2 (INVESTIGACAO §4.2): forma de cauda dinâmica (L-momentos)
        VarLocBlock(),       # P3 (INVESTIGACAO §3): variância localizada no changepoint
        JumpBlock(),         # P4 (INVESTIGACAO §4.3): bipower/saltos + leverage (precisão T6/T9)
    ]
    # Este é o conjunto de blocos do V4 -- o melhor modelo medido do projeto (docs/HISTORICO.md §1).
    #
    # O BOCPDBlock (state/bocpd.py) esteve aqui no V5, junto com a poda de LMomentBlock e de
    # dependence.windows=[50] (argumento de ROI de latência). O pacote foi medido por R0 e REGREDIU:
    # Delta geral -0,0042 [-0,0095, +0,0006] e o bucket 50<t<=150 significativamente pior
    # (-0,0114, IC exclui 0) -- artifacts/reports/compare_v5_vs_v4.json. Pela regra de decisão de R0
    # (adotar só se o IC excluir 0 A FAVOR), V5 não passa e foi revertido (docs/HISTORICO.md §9).
    #
    # O bloco e o teste continuam em state/bocpd.py, reabríveis: o experimento que separa as duas
    # mudanças empacotadas no V5 (V4 + BOCPD, SEM a poda) nunca foi rodado.


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

        # F2 (proposta V2): impressão digital estendida do regime H0 — constantes por série,
        # calculadas em fit_h0, custo zero por passo. São condicionadores (CE6 nulo como efeito
        # principal), e a família meta_h0 já é a mais usada do modelo (34,3% do |SHAP|).
        for key, value in h0.fingerprint.items():
            feats[f"meta_h0_{key}"] = value

        age_map = feats.get("bayes_age_map_h0100")
        age_cusum = feats.get("cusum_age_mean_pos_d050")
        if age_map is not None and age_cusum is not None and not (math.isnan(age_map) or math.isnan(age_cusum)):
            feats["meta_locator_diff"] = abs(age_map - age_cusum)
            feats["meta_locator_min"] = min(age_map, age_cusum)
        else:
            feats["meta_locator_diff"] = math.nan
            feats["meta_locator_min"] = math.nan

        # F1 (proposta V2): versões `_cal` padronizadas contra o nulo da PRÓPRIA série, medido sobre
        # o histórico em fit_h0. Aplicado por último, depois que todos os blocos já emitiram seus
        # valores crus. Ver a docstring de state/calibration.py para o porquê.
        apply_calibration(feats, self.h0.null_stats, t)

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
