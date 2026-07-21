"""F4 (docs/PROPOSTA_FEATURES_V2.md): decomposição causal de energia por escala (Haar diádico)."""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.multiscale import MultiScaleBlock, history_series


def _h0(cfg, seed=5, n=2000):
    rng = np.random.RandomState(seed)
    return fit_h0(rng.randn(n), cfg), rng


def test_energies_near_zero_ln_under_white_noise(cfg):
    """Haar preserva variância: sob H0 unitário, E[d²]=1 em toda escala -> ln E ≈ 0. É isso que
    torna as features aproximadamente comparáveis entre séries já na forma crua."""
    h0, rng = _h0(cfg)
    blk = MultiScaleBlock()
    blk.reset(h0, cfg)
    for t, x in enumerate(rng.randn(4000), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    feats = blk.features()
    for j in range(cfg.multiscale.n_scales):
        assert abs(feats[f"haar_energy_ln_s{j}"]) < 0.6, f"escala {j} longe de 0"


def test_coarse_scales_are_nan_before_enough_coefficients(cfg):
    h0, rng = _h0(cfg)
    blk = MultiScaleBlock()
    blk.reset(h0, cfg)
    for t, x in enumerate(rng.randn(20), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    feats = blk.features()
    # escala 4 precisa de 2^5 * 3 = 96 amostras; com 20 tem de estar NaN
    assert math.isnan(feats["haar_energy_ln_s4"])
    assert not math.isnan(feats["haar_energy_ln_s0"])


def test_variance_level_shift_raises_all_scales(cfg):
    h0, rng = _h0(cfg)
    blk = MultiScaleBlock()
    blk.reset(h0, cfg)
    data = np.concatenate([rng.randn(2000), rng.randn(2000) * 2.0])
    snap = {}
    for t, x in enumerate(data, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (2000, 4000):
            snap[t] = blk.features()
    for j in range(3):  # escalas finas reagem dentro do horizonte do teste
        assert snap[4000][f"haar_energy_ln_s{j}"] > snap[2000][f"haar_energy_ln_s{j}"]


def test_burst_tilts_contrast_toward_fine_scales(cfg):
    """O discriminador que motiva F4: um burst curto de alta frequência deve deslocar o contraste
    fino-vs-grosso para cima em relação a um patamar persistente da mesma energia total."""
    h0, rng = _h0(cfg)

    def run(data):
        blk = MultiScaleBlock()
        blk.reset(h0, cfg)
        for t, x in enumerate(data, start=1):
            e = float(np.clip(x, *cfg.h0.clip_e))
            blk.update(e, x, e, t)
        return blk.features()["haar_contrast_fine_coarse"]

    base = rng.randn(3000)
    # alternância de sinal = energia concentrada na escala mais fina
    burst = base.copy()
    burst[2000:] += 2.0 * np.array([(-1.0) ** i for i in range(1000)])
    # patamar = energia espalhada por todas as escalas
    level = base.copy()
    level[2000:] *= 2.0

    assert run(burst) > run(level)


def test_history_series_matches_online_block(cfg):
    """CRÍTICO (mesma razão de test_mmd): a calibração F1 usa o caminho vetorizado."""
    h0, rng = _h0(cfg)
    data = rng.randn(1200)

    cfg_nowarm = dataclasses.replace(
        cfg, multiscale=dataclasses.replace(cfg.multiscale, warmup_min_coeffs=1)
    )

    blk = MultiScaleBlock()
    blk.reset(h0, cfg_nowarm)
    captured: dict[int, list] = {j: [] for j in range(cfg.multiscale.n_scales)}
    prev_counts = [0] * cfg.multiscale.n_scales
    for t, x in enumerate(data, start=1):
        blk.update(float(x), float(x), float(x), t)
        for j in range(cfg.multiscale.n_scales):
            if blk.count[j] > prev_counts[j]:
                captured[j].append(blk.energy[j])
                prev_counts[j] = blk.count[j]

    vec = history_series(data, cfg_nowarm)
    for j in range(cfg.multiscale.n_scales):
        key = f"haar_energy_ln_s{j}"
        if key not in vec:
            continue
        got = np.log(np.maximum(np.array(captured[j], dtype=float), 1e-12))
        exp = np.asarray(vec[key])
        n = min(len(got), len(exp))
        assert n > 5, f"escala {j} com amostras de menos para o teste"
        assert np.allclose(got[:n], exp[:n], atol=1e-10), f"escala {j} divergiu"
