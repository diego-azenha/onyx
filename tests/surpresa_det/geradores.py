"""Geradores sintéticos H0 do detector de surpresa (RNG só em tests/, NOTAS_AGENTES §1)."""
import numpy as np


def _garch(rng, n, omega=0.05, alpha=0.10, beta=0.85):
    x = np.empty(n)
    v = omega / (1 - alpha - beta)
    e_ant = 0.0
    for t in range(n):
        v = omega + alpha * e_ant**2 + beta * v
        e_ant = np.sqrt(v) * rng.standard_normal()
        x[t] = e_ant
    return x


def _ar1(rng, n, phi=0.6):
    x = np.empty(n)
    x[0] = rng.standard_normal()
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


GERADORES = {
    "iid": lambda rng, n: rng.standard_normal(n),
    "ar1": _ar1,
    "garch": _garch,
    "t4": lambda rng, n: rng.standard_t(4, n),
}
