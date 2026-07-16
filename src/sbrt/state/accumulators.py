"""AccumulatorBlock — Welford global + EWMA (média/variância/sinal/excedência) + janelas rodantes
(plano §4.2, tabela §5 linhas #1,2,3,6,7,9,10,12,14,16,17,18,19).

Roteamento de fluxo (plano §3.4): média/dependência/forma usam `e_vol` (vol-ajustado); variância/cauda
usam `e` (escala congelada do histórico) — nunca o inverso, sob pena de o EWMA-vol absorver a própria
quebra de variância (contraexemplo CE2, plano §12.5).

Features #26 (hedge, precisa de x cru) e #27/#28 (meta) não cabem no contrato `StateBlock`
(que só recebe e/e_raw/e_vol/t) — são calculadas por `state/scorer.py` diretamente.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from sbrt.utils.numerics import ewma_update, welford_update
from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


def _evicted_delta(evicted: float | None) -> float:
    return evicted if evicted is not None else 0.0


def _push_and_update_sum(entry: list, value: float) -> None:
    ring: RingBuffer = entry[0]
    evicted = ring.push(value)
    entry[1] = entry[1] + value - _evicted_delta(evicted)


class AccumulatorBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.cfg = cfg
        self.t = 0
        self.warmup_min_n = cfg.features.warmup_min_n

        self.ewma_mean = {lam: 0.0 for lam in cfg.state.ewma_lambdas}          # #2, e_vol
        self.ewma_var = {lam: 1.0 for lam in cfg.state.ewma_lambdas}           # #6, e (v0=1)
        self.welford_mean_evol = (0, 0.0, 0.0)                                  # #1 / #16 denom, e_vol
        self.welford_var_e = (0, 0.0, 0.0)                                     # #9, e

        self.window_sum_evol = {w: [RingBuffer(w), 0.0] for w in cfg.state.window_sizes}   # #3
        self.window_sumsq_e = {w: [RingBuffer(w), 0.0] for w in cfg.state.window_sizes}    # #7
        self.exceed_windows = {w: [RingBuffer(w), 0.0] for w in cfg.state.exceedance_windows}  # #10a
        self.sign_windows = {w: [RingBuffer(w), 0.0] for w in cfg.state.sign_windows}       # #14

        self.ewma_exceed2 = 0.0                                                # #10b
        self._p0_exceed2 = 2.0 * (1.0 - _norm_cdf(2.0))

        self.count_exceed99 = 0                                                # #12
        self.max_abs_eraw = 0.0
        self._q99_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.99))

        self.S_u_global = 0.0                                                  # #16 numerator
        self.prev_evol: float | None = None

        dep_w = cfg.state.dependence_window                                    # #17
        self.dep_u_ring = RingBuffer(dep_w)
        self.dep_u_sum = 0.0
        self.dep_sq_ring = RingBuffer(dep_w)
        self.dep_sq_sum = 0.0

        qw = cfg.state.quantile_crossing_window                                # #18
        self.qc_mid = [RingBuffer(qw), 0.0]
        self.qc_low = [RingBuffer(qw), 0.0]
        self.q25 = h0.q["0.25"]
        self.q75 = h0.q["0.75"]
        self.q10 = h0.q["0.10"]

        sw = cfg.state.skew_window                                             # #19
        self.skew_ring = RingBuffer(sw)
        self.skew_sum3 = 0.0

        # "vol-of-vol": estabilidade do nível de variância numa janela, não o nível em si (todas as
        # #7/#9 já existentes medem nível). Um burst GARCH tem variância-da-variância ALTA (a
        # variância local sobe e desce dentro do burst); uma quebra de variância genuína assenta num
        # novo patamar ESTÁVEL. Feature adicionada após achado de que o modelo rankeava alarmes falsos
        # tipo GARCH acima de quebras sutis reais (T6/T9 vs T3, ver histórico do projeto). Reusa
        # window_sumsq_e[100] (E[e²]) já existente; só precisa de E[e⁴] adicional na mesma janela.
        self.volvol_100 = [RingBuffer(100), 0.0]

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t

        n, mean, m2 = welford_update(*self.welford_mean_evol, e_vol)
        self.welford_mean_evol = (n, mean, m2)

        for lam in self.ewma_mean:
            self.ewma_mean[lam] = ewma_update(self.ewma_mean[lam], e_vol, lam)
        for lam in self.ewma_var:
            self.ewma_var[lam] = ewma_update(self.ewma_var[lam], e * e, lam)

        for w in self.window_sum_evol:
            _push_and_update_sum(self.window_sum_evol[w], e_vol)
        for w in self.window_sumsq_e:
            _push_and_update_sum(self.window_sumsq_e[w], e * e)

        n2, mean2, m2b = welford_update(*self.welford_var_e, e)
        self.welford_var_e = (n2, mean2, m2b)

        ind2 = 1.0 if abs(e) > 2.0 else 0.0
        for w in self.exceed_windows:
            _push_and_update_sum(self.exceed_windows[w], ind2)
        lam_mid = self.cfg.state.ewma_lambdas[1]
        self.ewma_exceed2 = ewma_update(self.ewma_exceed2, ind2, lam_mid)

        if abs(e_raw) > self._q99_abs:
            self.count_exceed99 += 1
        self.max_abs_eraw = max(self.max_abs_eraw, abs(e_raw))

        indpos = 1.0 if e_vol > 0 else 0.0
        for w in self.sign_windows:
            _push_and_update_sum(self.sign_windows[w], indpos)

        if self.prev_evol is not None:
            u_t = e_vol * self.prev_evol
            self.S_u_global += u_t
            evicted_u = self.dep_u_ring.push(u_t)
            self.dep_u_sum += u_t - _evicted_delta(evicted_u)
        evicted_sq = self.dep_sq_ring.push(e_vol * e_vol)
        self.dep_sq_sum += e_vol * e_vol - _evicted_delta(evicted_sq)
        self.prev_evol = e_vol

        in_mid = 1.0 if (self.q25 < e_vol < self.q75) else 0.0
        below_low = 1.0 if e_vol < self.q10 else 0.0
        _push_and_update_sum(self.qc_mid, in_mid)
        _push_and_update_sum(self.qc_low, below_low)

        cube = e_vol ** 3
        evicted_c = self.skew_ring.push(cube)
        self.skew_sum3 += cube - _evicted_delta(evicted_c)

        _push_and_update_sum(self.volvol_100, (e * e) ** 2)

    def features(self) -> dict[str, float]:
        t = self.t
        wmin = self.warmup_min_n
        out: dict[str, float] = {}

        n, mean, m2 = self.welford_mean_evol
        out["accum_welford_mean_z"] = math.sqrt(n) * mean if n >= wmin else _NAN

        for lam in self.ewma_mean:
            var_l = lam / (2.0 - lam)
            key = f"accum_ewma_mean_z_l{round(lam * 1000):03d}"
            out[key] = (self.ewma_mean[lam] / math.sqrt(var_l)) if t >= wmin else _NAN

        for w, (_, s) in self.window_sum_evol.items():
            n_eff = min(t, w)
            key = f"accum_window_mean_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key] = (s / n_eff) * math.sqrt(n_eff)
            else:
                out[key] = _NAN

        for lam in self.ewma_var:
            key = f"accum_ewma_var_ln_l{round(lam * 1000):03d}"
            out[key] = math.log(max(self.ewma_var[lam], 1e-12)) if t >= wmin else _NAN

        for w, (_, s) in self.window_sumsq_e.items():
            n_eff = min(t, w)
            key = f"accum_window_var_ln_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key] = math.log(max(s / n_eff, 1e-12))
            else:
                out[key] = _NAN

        n2, mean2, m2b = self.welford_var_e
        out["accum_welford_var_ln"] = math.log(max(m2b / n2, 1e-12)) if n2 >= wmin else _NAN

        for w, (_, s) in self.exceed_windows.items():
            n_eff = min(t, w)
            key = f"accum_window_exceed2_frac_w{w:03d}"
            out[key] = (s / n_eff) if (t >= wmin and n_eff > 0) else _NAN

        p0 = self._p0_exceed2
        lam_mid = self.cfg.state.ewma_lambdas[1]
        var_ewma = p0 * (1.0 - p0) * lam_mid / (2.0 - lam_mid)
        out["accum_ewma_exceed2_z"] = (
            (self.ewma_exceed2 - p0) / math.sqrt(max(var_ewma, 1e-12)) if t >= wmin else _NAN
        )

        out["accum_global_exceed99_frac"] = (self.count_exceed99 / t) if t >= wmin else _NAN
        out["accum_global_max_abs_eraw"] = self.max_abs_eraw if t >= 1 else _NAN

        for w, (_, s) in self.sign_windows.items():
            n_eff = min(t, w)
            key = f"accum_window_sign_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                p = s / n_eff
                out[key] = (p - 0.5) / math.sqrt(0.25 / n_eff)
            else:
                out[key] = _NAN

        if t >= max(wmin, 3) and m2 > 1e-9:
            r = max(min(self.S_u_global / m2, 0.999), -0.999)
            out["accum_global_rho1_fz"] = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(t - 3, 1))
        else:
            out["accum_global_rho1_fz"] = _NAN

        dep_w = self.cfg.state.dependence_window
        n_eff_dep = min(max(t - 1, 0), dep_w)
        if t >= max(wmin, 3) and self.dep_sq_sum > 1e-9 and n_eff_dep > 3:
            r = max(min(self.dep_u_sum / self.dep_sq_sum, 0.999), -0.999)
            out["accum_window_rho1_fz_w100"] = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n_eff_dep - 3)
        else:
            out["accum_window_rho1_fz_w100"] = _NAN

        qw = self.cfg.state.quantile_crossing_window
        n_eff_qc = min(t, qw)
        if t >= wmin and n_eff_qc > 0:
            out["accum_window_qcross_mid_frac_w100"] = self.qc_mid[1] / n_eff_qc
            out["accum_window_qcross_low_frac_w100"] = self.qc_low[1] / n_eff_qc
        else:
            out["accum_window_qcross_mid_frac_w100"] = _NAN
            out["accum_window_qcross_low_frac_w100"] = _NAN

        sw = self.cfg.state.skew_window
        n_eff_s = min(t, sw)
        if t >= wmin and n_eff_s > 0:
            skew_hat = self.skew_sum3 / n_eff_s
            out["accum_window_skew_z_w250"] = skew_hat * math.sqrt(n_eff_s / 6.0)
        else:
            out["accum_window_skew_z_w250"] = _NAN

        n_eff_vv = min(t, 100)
        if t >= wmin and n_eff_vv > 1 and 100 in self.window_sumsq_e:
            mean_e2 = self.window_sumsq_e[100][1] / n_eff_vv
            mean_e4 = self.volvol_100[1] / n_eff_vv
            var_e2 = max(mean_e4 - mean_e2 * mean_e2, 0.0)
            out["accum_window_volvol_cv_w100"] = math.sqrt(var_e2) / (mean_e2 + 1e-6)
        else:
            out["accum_window_volvol_cv_w100"] = _NAN

        return out


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
