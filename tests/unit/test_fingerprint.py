"""F2 (docs/PROPOSTA_FEATURES_V2.md): impressão digital estendida do regime H0."""
from __future__ import annotations

import math

import numpy as np

from sbrt.robustness.generators import generate
from sbrt.state.fingerprint import compute_fingerprint

_LEVELS = ("0.01", "0.25", "0.75", "0.99")


def _q(e: np.ndarray) -> dict:
    return {lv: float(np.quantile(e, float(lv))) for lv in _LEVELS}


def _fp(e: np.ndarray, cfg) -> dict:
    return compute_fingerprint(e, e, _q(e), cfg)


def test_white_noise_gives_reference_values(cfg):
    rng = np.random.RandomState(0)
    e = rng.randn(4000)
    fp = _fp(e, cfg)
    assert all(math.isfinite(v) for v in fp.values())
    assert 0.40 < fp["hurst"] < 0.60          # H=0.5 para ruído branco
    assert abs(fp["acf_e2_l1"]) < 0.08        # sem clustering de volatilidade
    assert abs(fp["spectral_slope"]) < 0.25   # espectro plano


def test_random_walk_has_high_hurst(cfg):
    rng = np.random.RandomState(1)
    e = np.cumsum(rng.randn(4000))
    assert _fp(e, cfg)["hurst"] > 0.75


def test_heavy_tail_raises_hill_xi(cfg):
    rng = np.random.RandomState(2)
    gauss = _fp(rng.randn(4000), cfg)
    heavy = _fp(rng.standard_t(3, size=4000), cfg)
    assert heavy["hill_xi"] > gauss["hill_xi"]


def test_garch_raises_volatility_clustering_descriptors(cfg):
    """A assinatura que o modelo mais precisa distinguir (T6): clustering de vol sem quebra."""
    rng = np.random.RandomState(3)
    white = _fp(rng.randn(4000), cfg)
    garch_hist, _, _ = generate("t6", seed=0, cfg=cfg)
    garch = _fp(garch_hist, cfg)
    assert garch["acf_e2_l1"] > white["acf_e2_l1"]
    assert garch["ljungbox_abs"] > white["ljungbox_abs"]


def test_degenerate_inputs_stay_finite(cfg):
    """Séries constantes/curtas não podem produzir NaN/Inf (T12 da suíte usa escala ~1e-6)."""
    for e in (np.zeros(500), np.ones(500), np.full(500, 1e-9), np.arange(500.0)):
        fp = _fp(e, cfg)
        assert all(math.isfinite(v) for v in fp.values()), f"não-finito em {fp}"


def test_all_expected_keys_present(cfg):
    rng = np.random.RandomState(4)
    fp = _fp(rng.randn(2000), cfg)
    assert set(fp) == {
        "hurst", "hill_xi", "acf_e2_l1", "acf_abs_mass", "acf_decay",
        "spectral_slope", "ljungbox_abs", "volvol", "iqr_tail_ratio",
    }
