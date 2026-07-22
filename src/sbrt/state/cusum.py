"""CusumBlock — banco de 15 CUSUMs + idades (plano §4.2, tabela §5 linhas #4,#8,#11,#13,#15).

Recursões max O(1), minimax-ótimas para alternativas simples (Page 1954; Moustakides 1986).
Fluxo: média/sinal usam `e_vol` (vol-ajustado); variância usa `e` (frozen, trava anti-absorção
§3.4/CE2); excedência usa `e_raw` contra os quantis do H0; dependência usa `e_vol` normalizado por
sigma_u do histórico. As features saem CRUAS (sem logístico) — a calibração é tarefa do LightGBM
(§5); o mapeamento logístico só existe no fallback puro-estatístico (§8.5).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params


def _fmt(x: float) -> str:
    """0.25 -> '025', 1.5 -> '150' — convenção de sufixo de nome de feature (delta/ratio * 100)."""
    return f"{round(x * 100):03d}"


class CusumBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.cfg = cfg
        self.deltas = list(cfg.cusum.mean_deltas)
        self.ratios_up = list(cfg.cusum.var_ratios_up)
        self.ratio_down = cfg.cusum.var_ratio_down
        self.sigma_u = h0.sigma_u
        self.dep_delta = cfg.state.dependence_delta_u

        self.mean_pos = {d: 0.0 for d in self.deltas}
        self.mean_neg = {d: 0.0 for d in self.deltas}
        self.var_up = {r: 0.0 for r in self.ratios_up}
        self.var_down = 0.0

        self.q95_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.95))
        self.q99_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.99))
        eb = cfg.state.exceedance_bernoulli
        self._eb_q95 = eb["q95"]
        self._eb_q99 = eb["q99"]
        self.exceed_q95 = 0.0
        self.exceed_q99 = 0.0

        sb = cfg.state.sign_bernoulli
        self._sb_p0 = sb["p0"]
        self._sb_p1_pos = sb["p1_pos"]
        self._sb_p1_neg = sb["p1_neg"]
        self.sign_pos = 0.0
        self.sign_neg = 0.0

        self.dep_pos = 0.0
        self.dep_neg = 0.0
        self.prev_evol: float | None = None

        self.ages = {
            "mean_pos": {d: 0 for d in self.deltas},
            "mean_neg": {d: 0 for d in self.deltas},
            "var_up": {r: 0 for r in self.ratios_up},
            "var_down": 0,
            "exceed_q95": 0,
            "exceed_q99": 0,
            "sign_pos": 0,
            "sign_neg": 0,
            "dep_pos": 0,
            "dep_neg": 0,
        }

    @staticmethod
    def _bump_age(current_age: int, new_value: float) -> int:
        return 0 if new_value <= 0.0 else current_age + 1

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        for d in self.deltas:
            self.mean_pos[d] = max(0.0, self.mean_pos[d] + d * e_vol - d * d / 2.0)
            self.ages["mean_pos"][d] = self._bump_age(self.ages["mean_pos"][d], self.mean_pos[d])
            self.mean_neg[d] = max(0.0, self.mean_neg[d] - d * e_vol - d * d / 2.0)
            self.ages["mean_neg"][d] = self._bump_age(self.ages["mean_neg"][d], self.mean_neg[d])

        e2 = e * e
        for r in self.ratios_up:
            inc = 0.5 * ((1.0 - 1.0 / r) * e2 - np.log(r))
            self.var_up[r] = max(0.0, self.var_up[r] + inc)
            self.ages["var_up"][r] = self._bump_age(self.ages["var_up"][r], self.var_up[r])

        r_down = self.ratio_down
        inc_down = 0.5 * ((1.0 - 1.0 / r_down) * e2 - np.log(r_down))
        self.var_down = max(0.0, self.var_down + inc_down)
        self.ages["var_down"] = self._bump_age(self.ages["var_down"], self.var_down)

        b95 = 1.0 if abs(e_raw) > self.q95_abs else 0.0
        p0, p1 = self._eb_q95["p0"], self._eb_q95["p1"]
        inc95 = b95 * np.log(p1 / p0) + (1.0 - b95) * np.log((1.0 - p1) / (1.0 - p0))
        self.exceed_q95 = max(0.0, self.exceed_q95 + inc95)
        self.ages["exceed_q95"] = self._bump_age(self.ages["exceed_q95"], self.exceed_q95)

        b99 = 1.0 if abs(e_raw) > self.q99_abs else 0.0
        p0, p1 = self._eb_q99["p0"], self._eb_q99["p1"]
        inc99 = b99 * np.log(p1 / p0) + (1.0 - b99) * np.log((1.0 - p1) / (1.0 - p0))
        self.exceed_q99 = max(0.0, self.exceed_q99 + inc99)
        self.ages["exceed_q99"] = self._bump_age(self.ages["exceed_q99"], self.exceed_q99)

        b_sign = 1.0 if e_vol > 0 else 0.0
        p0 = self._sb_p0
        p1 = self._sb_p1_pos
        inc_sign_pos = b_sign * np.log(p1 / p0) + (1.0 - b_sign) * np.log((1.0 - p1) / (1.0 - p0))
        self.sign_pos = max(0.0, self.sign_pos + inc_sign_pos)
        self.ages["sign_pos"] = self._bump_age(self.ages["sign_pos"], self.sign_pos)

        p1 = self._sb_p1_neg
        inc_sign_neg = b_sign * np.log(p1 / p0) + (1.0 - b_sign) * np.log((1.0 - p1) / (1.0 - p0))
        self.sign_neg = max(0.0, self.sign_neg + inc_sign_neg)
        self.ages["sign_neg"] = self._bump_age(self.ages["sign_neg"], self.sign_neg)

        if self.prev_evol is not None:
            u_norm = (e_vol * self.prev_evol) / self.sigma_u
            du = self.dep_delta
            self.dep_pos = max(0.0, self.dep_pos + du * u_norm - du * du / 2.0)
            self.ages["dep_pos"] = self._bump_age(self.ages["dep_pos"], self.dep_pos)
            self.dep_neg = max(0.0, self.dep_neg - du * u_norm - du * du / 2.0)
            self.ages["dep_neg"] = self._bump_age(self.ages["dep_neg"], self.dep_neg)
        self.prev_evol = e_vol

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for d in self.deltas:
            out[f"cusum_mean_pos_d{_fmt(d)}"] = self.mean_pos[d]
            out[f"cusum_mean_neg_d{_fmt(d)}"] = self.mean_neg[d]
        for r in self.ratios_up:
            out[f"cusum_var_up_r{_fmt(r)}"] = self.var_up[r]
        out[f"cusum_var_down_r{_fmt(self.ratio_down)}"] = self.var_down
        out["cusum_exceed_q95"] = self.exceed_q95
        out["cusum_exceed_q99"] = self.exceed_q99
        out["cusum_sign_pos"] = self.sign_pos
        out["cusum_sign_neg"] = self.sign_neg
        out["cusum_dep_pos"] = self.dep_pos
        out["cusum_dep_neg"] = self.dep_neg

        # idades: 6 selecionadas (plano §5 #24) — localizadores baratos de tau, usadas também
        # pela concordância de localizadores (#25, calculada em state/scorer.py)
        out["cusum_age_mean_pos_d050"] = float(self.ages["mean_pos"].get(0.5, float("nan")))
        out["cusum_age_mean_neg_d050"] = float(self.ages["mean_neg"].get(0.5, float("nan")))
        out["cusum_age_var_up_r150"] = float(self.ages["var_up"].get(1.5, float("nan")))
        out["cusum_age_sign_pos"] = float(self.ages["sign_pos"])
        out["cusum_age_sign_neg"] = float(self.ages["sign_neg"])
        out["cusum_age_exceed_q95"] = float(self.ages["exceed_q95"])
        return out


def history_null_series(
    e_hist, e_vol_hist, h0: "H0Params", cfg: "Config", restart_every: int,
    max_reps: int = 0, wanted: frozenset = frozenset(),
) -> dict:
    """Réplicas com reinício do PRÓPRIO CusumBlock sobre o histórico (H0 por definição), para a
    calibração de nulo por série (F1.a, state/calibration.py).

    Devolve `{nome: matriz (n_reps, restart_every)}` — linha = uma execução partindo do zero sobre um
    trecho virgem do histórico, coluna j = valor no passo j+1.

    **Por que reiniciar em vez de uma passada contínua** (como fazem `dependence`/`lmoments`): o CUSUM
    é uma recursão que parte de 0 e leva ~75 passos para atingir a distribuição estacionária (medido).
    Uma passada contínua mede só o regime estacionário e não diria nada sobre o nulo em t pequeno —
    justamente onde o modelo é mais fraco. Cada réplica reproduz o que o online realmente faz, já que
    `StreamScorer.__init__` reseta todo bloco no início do segmento online.

    `max_reps` limita o número de réplicas: o erro do nulo cai com sqrt(n_reps) e é depois encolhido
    e suavizado, então passar de ~20 réplicas compra pouco e o custo é linear. Sem o teto, um
    histórico de 5.000 pontos pagaria 50 réplicas para um ganho de precisão desprezível.
    `wanted` restringe as colunas materializadas — o bloco tem de rodar inteiro (a recursão é
    conjunta), mas não há motivo para alocar matriz para 21 features quando a whitelist pede 2.

    Custo: n_reps * restart_every updates, limitado por `max_reps` e por n_h."""
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
        blk = CusumBlock()
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
