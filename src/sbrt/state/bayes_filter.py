"""BayesFilterBlock — filtro bayesiano de troca única, log-espaço (plano §4.3).

Modelo: sob H0, e_t ~ N(0,1) (e = fluxo congelado — o filtro cobre média E variância, "todas
(média+var)" na tabela §5 #20, logo nunca usa o fluxo vol-ajustado, §3.4). Pós-mudança: e_t ~
N(mu,sigma^2), prior conjugado Normal-Inv-chi^2. Hazard constante h, sem morte de regime (a quebra é
permanente). Dois filtros independentes (hazards 1/100 e 1/400) rodam em paralelo dentro do mesmo
bloco (plano tabela §5: "2 filtros (hazards h∈{1/100,1/400}), K=48, protege 8 recentes").
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.utils.numerics import lgamma_cached, logsumexp, welford_update

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params


def _log_student_t(x: float, nu: float, loc: float, scale2: float) -> float:
    scale2 = max(scale2, 1e-12)
    z2 = (x - loc) ** 2 / (nu * scale2)
    return (
        lgamma_cached((nu + 1.0) / 2.0)
        - lgamma_cached(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - ((nu + 1.0) / 2.0) * math.log1p(z2)
    )


class _SingleHazardFilter:
    __slots__ = (
        "log_h", "log_1mh", "mu0", "kappa0", "nu0", "sigma0_sq",
        "max_candidates", "protect_recent", "renorm_threshold", "logw0", "candidates",
    )

    def __init__(self, h: float, prior: dict, max_candidates: int, protect_recent: int, renorm_threshold: float):
        self.log_h = math.log(h)
        self.log_1mh = math.log(1.0 - h)
        self.mu0 = prior["mu0"]
        self.kappa0 = prior["kappa0"]
        self.nu0 = prior["nu0"]
        self.sigma0_sq = prior["sigma0_sq"]
        self.max_candidates = max_candidates
        self.protect_recent = protect_recent
        self.renorm_threshold = renorm_threshold
        self.logw0 = 0.0
        self.candidates: list[dict] = []

    def _log_pred(self, n: float, mean: float, m2: float, x: float) -> float:
        kappa0, mu0, nu0, sigma0_sq = self.kappa0, self.mu0, self.nu0, self.sigma0_sq
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * mean) / kappa_n
        nu_n = nu0 + n
        ssq_n = (nu0 * sigma0_sq + m2 + kappa0 * n * (mean - mu0) ** 2 / kappa_n) / nu_n
        scale2 = ssq_n * (kappa_n + 1.0) / kappa_n
        return _log_student_t(x, nu_n, mu_n, scale2)

    def update(self, e: float, t: int) -> None:
        logpred_new = self._log_pred(0.0, 0.0, 0.0, e)
        logw_new = self.logw0 + self.log_h + logpred_new

        ell0 = -0.5 * math.log(2.0 * math.pi) - 0.5 * e * e
        self.logw0 = self.logw0 + self.log_1mh + ell0

        for c in self.candidates:
            c["logw"] += self._log_pred(c["n"], c["mean"], c["m2"], e)
            n, mean, m2 = welford_update(c["n"], c["mean"], c["m2"], e)
            c["n"], c["mean"], c["m2"] = n, mean, m2

        n, mean, m2 = welford_update(0, 0.0, 0.0, e)
        self.candidates.append({"n": n, "mean": mean, "m2": m2, "logw": logw_new, "birth_t": t})

        if len(self.candidates) > self.max_candidates:
            self.candidates.sort(key=lambda c: c["birth_t"])
            protected = self.candidates[-self.protect_recent:]
            rest = self.candidates[: -self.protect_recent]
            rest.sort(key=lambda c: c["logw"], reverse=True)
            keep = self.max_candidates - len(protected)
            self.candidates = rest[:keep] + protected

        max_logw = max((c["logw"] for c in self.candidates), default=-math.inf)
        overall_max = max(max_logw, self.logw0)
        if abs(overall_max) > self.renorm_threshold:
            self.logw0 -= overall_max
            for c in self.candidates:
                c["logw"] -= overall_max

    def outputs(self, t: int) -> tuple[float, int, float, float]:
        logws = [c["logw"] for c in self.candidates]
        lo = logsumexp(logws) - self.logw0
        map_c = max(self.candidates, key=lambda c: c["logw"])
        age_map = t - map_c["birth_t"]
        n_map, mean_map, m2_map = map_c["n"], map_c["mean"], map_c["m2"]
        map_z_mean = math.sqrt(n_map) * mean_map if n_map > 0 else 0.0
        map_var_ln = math.log(m2_map / n_map + 1e-9) if n_map > 0 else math.log(1e-9)
        return lo, age_map, map_z_mean, map_var_ln


class BayesFilterBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        prior = cfg.bayes.prior
        self.filters = {
            h: _SingleHazardFilter(
                h, prior, cfg.bayes.max_candidates, cfg.bayes.protect_recent, cfg.bayes.logw_renorm_threshold
            )
            for h in cfg.bayes.hazards
        }
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        for f in self.filters.values():
            f.update(e, t)

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for h, f in self.filters.items():
            tag = f"h{round(h * 10000):04d}"
            lo, age_map, map_z_mean, map_var_ln = f.outputs(self.t)
            out[f"bayes_lo_{tag}"] = lo
            out[f"bayes_age_map_{tag}"] = float(age_map)
            out[f"bayes_age_ln1p_{tag}"] = math.log1p(max(age_map, 0))
            out[f"bayes_map_z_mean_{tag}"] = map_z_mean
            out[f"bayes_map_var_ln_{tag}"] = map_var_ln
        return out
