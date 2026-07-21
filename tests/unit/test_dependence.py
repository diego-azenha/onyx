"""P1 (docs/INVESTIGACAO_FALHAS_V3.md): dependência serial não-linear/multi-lag.

O teste que importa é `test_detects_volatility_clustering_that_linear_misses`: valida a premissa da
P1 — que ρ₁ de |e|/e² capta clustering de volatilidade (GARCH) que a autocorrelação linear de e (o
que o banco já tinha, e que estava morto no SHAP) não vê."""
from __future__ import annotations

import math

import numpy as np

from sbrt.robustness.generators import generate
from sbrt.state.dependence import DependenceBlock, _RollingAutocorr
from sbrt.state.h0 import fit_h0


def _run(block, series):
    out = []
    for t, x in enumerate(series, start=1):
        block.update(float(x), float(x), float(x), t)
        out.append(block.features())
    return out


def _new_block(cfg):
    b = DependenceBlock()
    b.reset(None, cfg)
    return b


def test_rolling_autocorr_matches_naive(cfg):
    rng = np.random.RandomState(0)
    u = rng.randn(300)
    W = 50
    ac = _RollingAutocorr(W, 1)
    for v in u:
        ac.update(float(v))
    got = ac.rho(1)
    # referência casando a definição online EXATA: a média/variância são sobre os últimos W valores;
    # o ring de produtos guarda os últimos W produtos p_t = u_t·u_{t-1} (que alcançam um ponto antes
    # da janela de valores -- estimador enviesado auto-consistente, não o naive de janela fechada).
    lo = len(u) - W
    win = u[lo:]
    mean = win.mean()
    var = (win * win).mean() - mean * mean
    prod = u[lo:] * u[lo - 1:-1]  # p_t = u_t·u_{t-1} para os últimos W passos
    autocov = prod.mean() - mean * mean
    exp = autocov / var
    assert abs(got - exp) < 1e-9


def test_features_finite_after_warmup(cfg):
    rng = np.random.RandomState(1)
    feats = _run(_new_block(cfg), rng.randn(300))[-1]
    # absrho1/sqrho1 nas janelas {50,100} + mass_abs + mass_evol (conjunto do V4; a poda de w050
    # veio no V5, que regrediu por R0 e foi revertido — docs/HISTORICO.md §9)
    assert len(feats) == 6
    assert all(math.isfinite(v) for v in feats.values())


def test_features_nan_before_warmup(cfg):
    b = _new_block(cfg)
    b.update(0.1, 0.1, 0.1, 1)
    assert all(math.isnan(v) for v in b.features().values())


def test_linear_autocorr_rises_under_ar1_break(cfg):
    """ρ₁ de e_vol (via massa) sobe quando surge dependência linear AR(1)."""
    rng = np.random.RandomState(2)
    pre = rng.randn(300)
    post = np.empty(300); prev = 0.0
    for i in range(300):
        prev = 0.6 * prev + rng.randn() * math.sqrt(1 - 0.36); post[i] = prev
    rows = _run(_new_block(cfg), np.concatenate([pre, post]))
    assert rows[599]["dep_mass_evol_w100"] > rows[299]["dep_mass_evol_w100"]


def test_detects_volatility_clustering_that_linear_misses(cfg):
    """PREMISSA CENTRAL DA P1. Um processo GARCH tem e SEM autocorrelação linear (ρ₁(e)≈0) mas e²/|e|
    FORTEMENTE autocorrelacionados (clustering de volatilidade). O banco antigo (linear) é cego a
    isso; ρ₁(|e|) e ρ₁(e²) devem acender."""
    garch_hist, garch_online, _ = generate("t6", seed=0, cfg=cfg)  # GARCH puro
    rng = np.random.RandomState(3)
    white = rng.randn(len(garch_online))

    g = _run(_new_block(cfg), garch_online)[-1]
    w = _run(_new_block(cfg), white)[-1]

    # clustering de volatilidade: |e| e e² muito mais autocorrelacionados no GARCH que no ruído branco
    assert g["dep_absrho1_w100"] > w["dep_absrho1_w100"] + 0.05
    assert g["dep_sqrho1_w100"] > w["dep_sqrho1_w100"] + 0.05


def test_calibration_registered_for_dependence(cfg):
    rng = np.random.RandomState(4)
    h0 = fit_h0(rng.randn(3000), cfg)
    names = set(h0.null_stats)
    assert "dep_absrho1_w100" in names and "dep_sqrho1_w100" in names
    assert "dep_mass_abs_w100" in names
    assert "dep_mass_evol_w100" not in names  # e_vol não calibrado (indisponível no histórico)
    assert h0.null_stats["dep_absrho1_w100"].kind == "rho"
    assert h0.null_stats["dep_mass_abs_w100"].kind == "none"


def test_calibration_null_matches_block_over_history(cfg):
    """A calibração roda o próprio bloco sobre o histórico -> o nulo TEM de bater com uma re-execução
    do bloco (garante que não há caminho vetorizado divergente)."""
    from sbrt.state.dependence import history_null_series
    rng = np.random.RandomState(5)
    e_hist = rng.randn(2000)
    series = history_null_series(e_hist, cfg)
    # re-roda o bloco manualmente e compara a media de um alvo
    rows = _run(_new_block(cfg), e_hist)
    manual = [r["dep_absrho1_w100"] for r in rows if math.isfinite(r["dep_absrho1_w100"])]
    from_series = [v for v in series["dep_absrho1_w100"] if math.isfinite(v)]
    assert np.allclose(manual, from_series, atol=1e-12)
