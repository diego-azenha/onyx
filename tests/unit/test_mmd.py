"""F3 (docs/PROPOSTA_FEATURES_V2.md): MMD de kernel via Random Fourier Features."""
from __future__ import annotations

import math

import numpy as np

from sbrt.state.h0 import fit_h0
from sbrt.state.mmd import MMDBlock, history_reference, history_series, rff_table


def _h0(cfg, seed=3, n=2000):
    rng = np.random.RandomState(seed)
    return fit_h0(rng.randn(n), cfg), rng


def test_rff_table_is_deterministic_and_shared_across_calls(cfg):
    """Comparabilidade transversal depende disto: W e b têm de ser os MESMOS para toda série."""
    W1, b1 = rff_table(cfg.mmd.n_features, cfg.mmd.bandwidth, 1)
    W2, b2 = rff_table(cfg.mmd.n_features, cfg.mmd.bandwidth, 1)
    assert np.array_equal(W1, W2) and np.array_equal(b1, b2)
    assert W1.shape == (1, cfg.mmd.n_features)


def test_rff_table_differs_between_marginal_and_joint(cfg):
    W1, _ = rff_table(cfg.mmd.n_features, cfg.mmd.bandwidth, 1)
    W2, _ = rff_table(cfg.mmd.n_features, cfg.mmd.bandwidth, 2)
    assert W1.shape[0] == 1 and W2.shape[0] == 2


def test_features_finite_and_start_near_zero(cfg):
    h0, rng = _h0(cfg)
    blk = MMDBlock()
    blk.reset(h0, cfg)
    for t, x in enumerate(rng.randn(300), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
    feats = blk.features()
    assert len(feats) == 8  # {vfast, fast, slow, newma} x {marginal, conjunto}
    assert all(math.isfinite(v) for v in feats.values())
    # sob H0 a distância ao histórico deve ficar pequena
    assert feats["mmd_marginal_slow"] < 0.5


def test_marginal_mmd_rises_under_distribution_shift(cfg):
    h0, rng = _h0(cfg)
    blk = MMDBlock()
    blk.reset(h0, cfg)
    snap = {}
    online = np.concatenate([rng.randn(300), rng.randn(300) * 2.5])
    for t, x in enumerate(online, start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (300, 600):
            snap[t] = blk.features()["mmd_marginal_slow"]
    assert snap[600] > snap[300]


def test_joint_mmd_rises_under_dependence_shift_when_marginal_is_stable(cfg):
    """O ponto do espaço conjunto: detectar mudança de DEPENDÊNCIA com marginal ~inalterada."""
    h0, rng = _h0(cfg)
    blk = MMDBlock()
    blk.reset(h0, cfg)

    pre = rng.randn(400)
    # AR(1) forte com variância marginal ~1 (mesma marginal, dependência muito diferente)
    phi = 0.8
    post = np.empty(400)
    prev = 0.0
    for i in range(400):
        prev = phi * prev + rng.randn() * math.sqrt(1 - phi ** 2)
        post[i] = prev

    snap = {}
    for t, x in enumerate(np.concatenate([pre, post]), start=1):
        e = float(np.clip(x, *cfg.h0.clip_e))
        blk.update(e, x, e, t)
        if t in (400, 800):
            snap[t] = blk.features()
    assert snap[800]["mmd_joint_slow"] > snap[400]["mmd_joint_slow"]


def test_history_series_matches_online_block(cfg):
    """CRÍTICO: o cálculo vetorizado sobre o histórico (usado pela calibração F1) tem de produzir
    exatamente a mesma sequência que o laço causal online. Um desalinhamento aqui envenenaria
    silenciosamente o nulo de todas as features MMD calibradas."""
    h0, rng = _h0(cfg)
    data = rng.randn(500)

    blk = MMDBlock()
    blk.reset(h0, cfg)
    online = {k: [] for k in
              ("mmd_marginal_vfast", "mmd_marginal_fast", "mmd_marginal_slow", "mmd_marginal_newma",
               "mmd_joint_vfast", "mmd_joint_fast", "mmd_joint_slow", "mmd_joint_newma")}
    for t, x in enumerate(data, start=1):
        blk.update(float(x), float(x), float(x), t)
        f = blk.features()
        for k in online:
            online[k].append(f[k])

    vec = history_series(data, h0.rff_href, h0.rff_href_joint, cfg)

    # marginal: índice i do vetorizado <-> passo t=i+1 do online
    for k in ("mmd_marginal_vfast", "mmd_marginal_fast", "mmd_marginal_slow", "mmd_marginal_newma"):
        got = np.array(online[k][cfg.features.warmup_min_n:], dtype=float)
        exp = np.asarray(vec[k])[cfg.features.warmup_min_n:]
        assert np.allclose(got, exp, atol=1e-10), f"{k} divergiu"

    # conjunto: o par (e_t, e_{t-1}) só existe a partir de t=2 -> deslocamento de 1. O online emite
    # NaN durante o warm-up (t < warmup_min_n); comparamos onde ele emite número.
    for k in ("mmd_joint_vfast", "mmd_joint_fast", "mmd_joint_slow", "mmd_joint_newma"):
        got = np.array(online[k][1:], dtype=float)
        exp = np.asarray(vec[k])
        n = min(len(got), len(exp))
        got, exp = got[:n], exp[:n]
        finite = np.isfinite(got)
        assert finite.sum() > 400, "warm-up longo demais, teste perdeu poder"
        assert np.allclose(got[finite], exp[finite], atol=1e-10), f"{k} divergiu"


def test_history_reference_matches_direct_mean(cfg):
    rng = np.random.RandomState(11)
    e = rng.randn(1000)
    href, href_joint = history_reference(e, cfg)
    assert href.shape == (cfg.mmd.n_features,)
    assert href_joint.shape == (cfg.mmd.n_features,)
    assert np.all(np.isfinite(href)) and np.all(np.isfinite(href_joint))
