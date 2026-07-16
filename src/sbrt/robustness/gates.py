"""Gates comportamentais da suíte de robustez (plano §10, tabela revisada) — comparação de MEDIANA
entre cenário e controle (ou limiares absolutos), deliberadamente NÃO uma AUC (plano §9.0): são
testes internos com verdade sintética conhecida, não uma tentativa de estimar a TS-AUC oficial.

Nota de contrato: `docs/CONTRACTS.md`/plano copiam `evaluate(scenario_id, scores, tau, cfg)`, mas essa
assinatura não tem como comparar cenário-vs-controle sem receber os dois. Aqui `evaluate` recebe
explicitamente `trajectories` (uma lista de trajetórias de score, uma por seed) e
`control_trajectories` (idem, ou None para cenários sem par) — correção necessária documentada
nesta função e em CONTRACTS.md.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GateResult:
    scenario_id: str
    passed: bool
    details: dict = field(default_factory=dict)


def _at(trajectories: list, t: int) -> np.ndarray:
    """valores de cada trajetória no passo 1-based t (ignora trajetórias mais curtas que t)."""
    return np.array([traj[t - 1] for traj in trajectories if len(traj) >= t], dtype=np.float64)


def _median_at(trajectories: list, t: int) -> float:
    vals = _at(trajectories, t)
    return float(np.median(vals)) if len(vals) else math.nan


def _mean_at(trajectories: list, t: int) -> float:
    vals = _at(trajectories, t)
    return float(np.mean(vals)) if len(vals) else math.nan


def _median_range(trajectories: list, t_from: int, t_to: int):
    """min/max da mediana por passo, para t em [t_from, t_to] (checagem "para todo t>=t_from")."""
    medians = [_median_at(trajectories, t) for t in range(t_from, t_to + 1)]
    medians = [m for m in medians if not math.isnan(m)]
    if not medians:
        return math.nan, math.nan
    return min(medians), max(medians)


def _trend_slope(trajectories: list) -> float:
    max_len = max((len(t) for t in trajectories), default=0)
    means = []
    ts = []
    for t in range(1, max_len + 1):
        vals = _at(trajectories, t)
        if len(vals):
            means.append(float(np.mean(vals)))
            ts.append(t)
    if len(ts) < 2:
        return 0.0
    slope, _ = np.polyfit(np.array(ts, dtype=np.float64), np.array(means, dtype=np.float64), 1)
    return float(slope)


def _has_nan_or_inf(trajectories: list) -> bool:
    for traj in trajectories:
        for v in traj:
            if not math.isfinite(v):
                return True
    return False


def evaluate(
    scenario_id: str,
    trajectories: list,
    control_trajectories: list | None,
    tau: int | None,
    cfg,
) -> GateResult:
    spec = dict(cfg.gates.scenarios.get(scenario_id, {}))
    details: dict = {}

    if _has_nan_or_inf(trajectories):
        return GateResult(scenario_id, False, {"error": "NaN/Inf em trajetórias do cenário"})
    if control_trajectories is not None and _has_nan_or_inf(control_trajectories):
        return GateResult(scenario_id, False, {"error": "NaN/Inf em trajetórias de controle"})

    if scenario_id == "t1":
        lo, _ = _median_range(trajectories, spec["t_from"], max(len(t) for t in trajectories))
        _, ctrl_hi = _median_range(control_trajectories, spec["t_from"], max(len(t) for t in control_trajectories))
        details = {"median_min_observed": lo, "control_median_max_observed": ctrl_hi}
        passed = lo >= spec["median_min"] and ctrl_hi <= spec["control_median_max"]

    elif scenario_id == "t2":
        pre_means = [float(np.mean(traj[: tau - 1])) for traj in trajectories if tau and tau > 1]
        mean_prebreak = float(np.median(pre_means)) if pre_means else math.nan
        slope = _trend_slope([traj[: tau - 1] for traj in trajectories]) if tau and tau > 1 else 0.0
        details = {"mean_prebreak": mean_prebreak, "slope": slope}
        passed = mean_prebreak <= spec["mean_prebreak_max"] and abs(slope) <= cfg.gates.drift_slope_abs_max

    elif scenario_id in ("t3", "t5", "t5b", "t7", "t8"):
        t = tau + spec["t_offset"]
        gap = _median_at(trajectories, t) - _median_at(control_trajectories, t)
        details = {"t": t, "gap": gap}
        passed = gap >= spec["gap_min"]

    elif scenario_id == "t4":
        t = tau + spec["t_offset"]
        med = _median_at(trajectories, t)
        ctrl_med = _median_at(control_trajectories, t)
        details = {"t": t, "median": med, "control_median": ctrl_med}
        passed = med >= spec["median_min"] and ctrl_med <= spec["control_median_max"]

    elif scenario_id in ("t6", "t10"):
        max_len = max(len(t) for t in trajectories)
        final_mean = _mean_at(trajectories, max_len)
        slope = _trend_slope(trajectories)
        details = {"final_mean": final_mean, "slope": slope}
        passed = final_mean <= spec["mean_max"] and abs(slope) <= cfg.gates.drift_slope_abs_max

    elif scenario_id == "t9":
        final_mean = _mean_at(trajectories, 999)
        decay = _mean_at(trajectories, 210) - _mean_at(trajectories, 250)
        details = {"final_mean": final_mean, "decay": decay}
        passed = final_mean <= spec["final_max"] and decay >= spec["decay_min"]

    elif scenario_id == "t11":
        s10 = _mean_at(trajectories, 10)
        s4 = _mean_at(trajectories, 4)
        details = {"s10": s10, "s4": s4}
        passed = s10 > s4

    elif scenario_id in ("t12", "t12b"):
        slope = _trend_slope(trajectories)
        details = {"slope": slope}
        passed = abs(slope) <= cfg.gates.drift_slope_abs_max

    elif scenario_id == "t13":
        decay = _median_at(trajectories, 260) - _median_at(trajectories, 600)
        details = {"decay": decay}
        passed = decay >= spec["decay_min"]

    else:
        raise ValueError(f"cenário desconhecido: {scenario_id!r}")

    return GateResult(scenario_id, bool(passed), details)
