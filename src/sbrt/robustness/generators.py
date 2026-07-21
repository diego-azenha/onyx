"""Geradores de cenários sintéticos T1-T13 (+T5b, T12b) — plano §10. RNG livre aqui (permitido
apenas em robustness/generators.py e tests/, docs/PLANO_REPOSITORIO.md §8 checklist) — é geração de
dado sintético, não caminho de inferência.

`generate(scenario_id, seed, cfg)` devolve (hist, online, tau_index_0based_ou_None). Um scenario_id
com sufixo "_ctrl" (ex.: "t1_ctrl") gera o gêmeo de controle com a MESMA seed, sem a quebra/efeito —
"Controle = gêmeo sem quebra com as mesmas seeds" (plano §10).
"""
from __future__ import annotations

import math

import numpy as np

SCENARIO_IDS = (
    "t1", "t2", "t3", "t4", "t5", "t5b", "t6", "t7", "t8",
    "t9", "t10", "t11", "t12", "t12b", "t13",
)
# cenários com par de controle (gates de gap de mediana precisam do gêmeo sem quebra)
CONTROLLED_SCENARIOS = ("t1", "t3", "t4", "t5", "t5b", "t7", "t8")

# Cenários cujo gate original é um nível ABSOLUTO (assume score calibrado em [0,1]) em vez de um gap
# entre duas trajetórias — exatamente os que o parecer de auditoria (§6-R5, DIAGNOSTICO rec. 1(b))
# identifica como incompatíveis com o calibrador supervisionado (resíduo sem offset, deliberadamente
# não calibrado em escala absoluta, model/predict.py). Para estes, `robustness/gates.py` sabe computar
# um gate RELATIVO (gap contra um painel de referência i.i.d. N(0,1) sem quebra, mesmas seeds/T) quando
# `reference_trajectories` é passado a `evaluate`; sem isso, cai no gate absoluto de sempre (modo
# fallback, que É calibrado em [0,1] por construção, plano §8.5).
RELATIVE_GATE_SCENARIOS = ("t2", "t6", "t9", "t10", "t13")

N_H_DEFAULT = 2000

# comprimento T do segmento online de cada cenário em RELATIVE_GATE_SCENARIOS — o painel de
# referência precisa do mesmo T para que os gaps por passo sejam comparáveis (mesma seed, mesmo t).
_REFERENCE_T = {"t2": 600, "t6": 1000, "t9": 1000, "t10": 1000, "t13": 600}


def _garch11(rng: np.random.RandomState, n: int, omega=0.05, alpha=0.10, beta=0.85, burn=200):
    n_total = n + burn
    h = np.empty(n_total)
    x = np.empty(n_total)
    h[0] = omega / max(1.0 - alpha - beta, 1e-6)
    x[0] = rng.randn() * math.sqrt(h[0])
    for i in range(1, n_total):
        h[i] = omega + alpha * x[i - 1] ** 2 + beta * h[i - 1]
        x[i] = rng.randn() * math.sqrt(h[i])
    return x[burn:]


def _ar1(rng: np.random.RandomState, n: int, phi: float, eps_std: float, start: float = 0.0):
    x = np.empty(n)
    prev = start
    for i in range(n):
        prev = phi * prev + rng.randn() * eps_std
        x[i] = prev
    return x


def _gen_t1(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 600, 3
    online = rng.randn(T)
    if control:
        return hist, online, None
    online[tau - 1:] += 0.8
    return hist, online, tau - 1


def _gen_t2(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T = 600
    tau = T - 5
    online = rng.randn(T)
    if control:
        return hist, online, None
    online[tau - 1:] += 0.8
    return hist, online, tau - 1


def _gen_t3(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 800, 200
    online = rng.randn(T)
    if control:
        return hist, online, None
    online[tau - 1:] += 0.15
    return hist, online, tau - 1


def _gen_t4(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 600, 200
    online = rng.randn(T)
    if control:
        return hist, online, None
    online[tau - 1:] += 1.5
    return hist, online, tau - 1


def _gen_t5(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 600, 200
    if control:
        return hist, rng.randn(T), None
    online = np.empty(T)
    online[: tau - 1] = rng.randn(tau - 1)
    online[tau - 1:] = rng.randn(T - tau + 1) * 1.5
    return hist, online, tau - 1


def _gen_t5b(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau, ramp_len = 600, 200, 200
    if control:
        return hist, rng.randn(T), None
    online = np.empty(T)
    online[: tau - 1] = rng.randn(tau - 1)
    for j in range(T - tau + 1):
        frac = min(j, ramp_len) / ramp_len
        sigma = 1.0 + 0.5 * frac
        online[tau - 1 + j] = rng.randn() * sigma
    return hist, online, tau - 1


def _gen_t6(rng, control: bool):
    hist = _garch11(rng, N_H_DEFAULT)
    online = _garch11(rng, 1000)
    return hist, online, None


def _gen_t7(rng, control: bool):
    phi1, phi2 = 0.2, 0.6
    eps1 = 1.0
    eps2 = eps1 * math.sqrt((1.0 - phi2 ** 2) / (1.0 - phi1 ** 2))
    hist = _ar1(rng, N_H_DEFAULT, phi1, eps1)
    T, tau = 600, 200
    if control:
        online = _ar1(rng, T, phi1, eps1, start=hist[-1])
        return hist, online, None
    pre = _ar1(rng, tau - 1, phi1, eps1, start=hist[-1])
    post = _ar1(rng, T - tau + 1, phi2, eps2, start=pre[-1] if tau > 1 else hist[-1])
    online = np.concatenate([pre, post])
    return hist, online, tau - 1


def _gen_t8(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 600, 200
    if control:
        return hist, rng.randn(T), None
    online = np.empty(T)
    online[: tau - 1] = rng.randn(tau - 1)
    online[tau - 1:] = rng.standard_t(4, size=T - tau + 1) / math.sqrt(2.0)
    return hist, online, tau - 1


def _gen_t9(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T = 1000
    online = rng.randn(T)
    for pos in (50, 180, 420, 700):
        online[pos - 1] = rng.choice([-1.0, 1.0]) * 6.0
    return hist, online, None


def _gen_t10(rng, control: bool):
    period, amplitude, noise_std = 50.0, 1.0, 0.5

    def _seasonal(n):
        t = np.arange(n)
        return amplitude * np.sin(2 * np.pi * t / period) + rng.randn(n) * noise_std

    hist = _seasonal(N_H_DEFAULT)
    online = _seasonal(1000)
    return hist, online, None


def _gen_t11(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T, tau = 10, 5
    online = rng.randn(T)
    if control:
        return hist, online, None
    online[tau - 1:] += 1.0
    return hist, online, tau - 1


def _gen_t12(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT) * 1e-6
    online = rng.randn(1000) * 1e-6
    return hist, online, None


def _gen_t12b(rng, control: bool):
    n_h, T = N_H_DEFAULT, 1000
    hist = np.array([10.0 * (-1) ** i for i in range(n_h)]) + rng.randn(n_h) * 0.1
    online = np.array([10.0 * (-1) ** i for i in range(T)]) + rng.randn(T) * 0.1
    return hist, online, None


def _gen_t13(rng, control: bool):
    hist = rng.randn(N_H_DEFAULT)
    T = 600
    online = rng.randn(T)
    online[199:260] += 1.0  # t=200..260 (1-based), excursão transitória, plano §10 T13
    return hist, online, None


_GENERATORS = {
    "t1": _gen_t1, "t2": _gen_t2, "t3": _gen_t3, "t4": _gen_t4,
    "t5": _gen_t5, "t5b": _gen_t5b, "t6": _gen_t6, "t7": _gen_t7, "t8": _gen_t8,
    "t9": _gen_t9, "t10": _gen_t10, "t11": _gen_t11, "t12": _gen_t12,
    "t12b": _gen_t12b, "t13": _gen_t13,
}


def generate(scenario_id: str, seed: int, cfg=None):
    """T1..T13 (+T5b,T12b). Retorna (hist, online, tau_or_None). `cfg` reservado para overrides
    futuros — hoje os specs vêm hardcoded da tabela §10 (são a especificação do teste, não
    hiperparâmetros do detector)."""
    control = scenario_id.endswith("_ctrl")
    base_id = scenario_id[: -len("_ctrl")] if control else scenario_id
    if base_id not in _GENERATORS:
        raise ValueError(f"cenário desconhecido: {scenario_id!r}")
    rng = np.random.RandomState(seed)
    hist, online, tau = _GENERATORS[base_id](rng, control)
    return np.asarray(hist, dtype=np.float64), np.asarray(online, dtype=np.float64), tau


def generate_reference_panel(scenario_id: str, seed: int, cfg=None):
    """Painel de referência para gates relativos (R5): histórico + online i.i.d. N(0,1), SEM
    quebra/efeito nenhum, com a MESMA seed e o MESMO T do cenário `scenario_id` — mede o piso de
    falso-positivo do calibrador num sinal honestamente vazio, servindo de âncora comparável para
    cenários cujo gate original era um nível absoluto (RELATIVE_GATE_SCENARIOS)."""
    if scenario_id not in _REFERENCE_T:
        raise ValueError(f"sem painel de referência definido para {scenario_id!r}")
    rng = np.random.RandomState(seed)
    hist = rng.randn(N_H_DEFAULT)
    online = rng.randn(_REFERENCE_T[scenario_id])
    return np.asarray(hist, dtype=np.float64), np.asarray(online, dtype=np.float64), None
