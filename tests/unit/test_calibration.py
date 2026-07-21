"""F1 (docs/PROPOSTA_FEATURES_V2.md): calibração de nulo por série.

O teste que importa é `test_calibration_equalizes_null_scale_across_series`: ele mede diretamente a
premissa da proposta — que estatísticas cruas têm escala nula dependente da série (e portanto são
mal-ordenadas na seção transversal que a TS-AUC avalia), e que a versão `_cal` remove essa
dependência."""
from __future__ import annotations

import math

import numpy as np

from sbrt.robustness.generators import generate
from sbrt.state.calibration import (
    NullSpec,
    _null_at,
    _rolling_mean,
    _upper_tail_p_vec,
    apply_calibration,
)
from sbrt.state.conformal import _upper_tail_p
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def test_rolling_mean_matches_naive(cfg):
    rng = np.random.RandomState(0)
    x = rng.randn(200)
    w = 25
    got = _rolling_mean(x, w)
    exp = np.array([x[i : i + w].mean() for i in range(len(x) - w + 1)])
    assert np.allclose(got, exp)


def test_upper_tail_p_vec_matches_scalar_version(cfg):
    rng = np.random.RandomState(1)
    hist = np.sort(rng.randn(500))
    xs = rng.randn(50)
    got = _upper_tail_p_vec(hist, xs, len(hist))
    exp = np.array([_upper_tail_p(list(hist), float(x), len(hist)) for x in xs])
    assert np.allclose(got, exp)


def test_null_stats_cover_expected_families(cfg):
    rng = np.random.RandomState(2)
    h0 = fit_h0(rng.randn(3000), cfg)
    names = set(h0.null_stats)
    assert any(n.startswith("accum_window_var_ln_") for n in names)
    assert any(n.startswith("ranktwo_wilcoxon_z_") for n in names)
    assert any(n.startswith("ranktwo_dispersion_z_") for n in names)
    assert any(n.startswith("mmd_") for n in names)
    assert any(n.startswith("haar_") for n in names)
    assert any(n.startswith("dep_") for n in names)
    for spec in h0.null_stats.values():
        assert math.isfinite(spec.mu) and math.isfinite(spec.sd) and spec.sd > 0 and spec.min_t >= 1
        assert spec.kind in ("none", "z", "var_ln", "frac", "rho")


def test_apply_calibration_never_invents_numbers(cfg):
    null_stats = {"foo": NullSpec(1.0, 2.0, 50)}
    feats = {"foo": 5.0}
    apply_calibration(feats, null_stats, t=49)          # amostras insuficientes
    assert math.isnan(feats["foo_cal"])
    apply_calibration(feats, null_stats, t=50)
    assert feats["foo_cal"] == (5.0 - 1.0) / 2.0
    feats["foo"] = math.nan                              # cru indisponível
    apply_calibration(feats, null_stats, t=100)
    assert math.isnan(feats["foo_cal"])
    apply_calibration({}, null_stats, t=100)             # feature ausente não quebra


def test_scaling_transport_widens_sd_for_partial_windows(cfg):
    """A correção de diluição: com lei de escala conhecida, a calibração vale antes de a janela
    encher, usando o dp teórico do n efetivo em vez do da janela cheia."""
    w = 100
    spec = NullSpec(mu=-1.0 / w, sd=math.sqrt(2.0 / w), min_t=10, kind="var_ln", window=w)
    mu_full, sd_full = _null_at(spec, t=w)
    mu_part, sd_part = _null_at(spec, t=25)
    assert sd_part > sd_full                     # menos amostras -> nulo mais largo
    assert abs(sd_part - math.sqrt(2.0 / 25)) < 1e-9
    assert abs(mu_part - (-1.0 / 25)) < 1e-9
    # acima da janela cheia nada muda
    assert _null_at(spec, t=10_000) == (spec.mu, spec.sd)


def test_scaling_transport_preserves_series_inflation(cfg):
    """O que se transporta é o FATOR DE INFLAÇÃO da série, não o nível absoluto."""
    w = 100
    inflation = 3.0
    spec = NullSpec(mu=-1.0 / w, sd=inflation * math.sqrt(2.0 / w), min_t=10, kind="var_ln", window=w)
    _, sd_part = _null_at(spec, t=25)
    assert abs(sd_part / math.sqrt(2.0 / 25) - inflation) < 1e-9


def test_z_kind_null_is_independent_of_n(cfg):
    """ranktwo já normaliza por sqrt(n) na origem -> o nulo não deve escalar com n."""
    spec = NullSpec(mu=0.1, sd=1.7, min_t=10, kind="z", window=100)
    assert _null_at(spec, t=15) == _null_at(spec, t=100) == (0.1, 1.7)


def _online_stats(hist, online, cfg, keys):
    h0 = fit_h0(hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)
    acc = {k: [] for k in keys}
    for x in online:
        feats = scorer.update_features(float(x))
        for k in keys:
            v = feats.get(k, math.nan)
            if isinstance(v, float) and math.isfinite(v):
                acc[k].append(v)
    return {k: np.array(v, dtype=float) for k, v in acc.items()}


def test_calibrated_statistic_is_standardized_under_h0(cfg):
    """Rodando sobre dados que continuam H0, a versão `_cal` deve ficar ~N(0,1)."""
    rng = np.random.RandomState(7)
    hist = rng.randn(4000)
    online = rng.randn(3000)
    key = "accum_window_var_ln_w100"
    got = _online_stats(hist, online, cfg, [key, f"{key}_cal"])
    cal = got[f"{key}_cal"]
    assert len(cal) > 1000
    assert abs(float(cal.mean())) < 0.6
    assert 0.5 < float(cal.std(ddof=1)) < 2.0


def test_calibration_equalizes_null_scale_across_series(cfg):
    """PREMISSA CENTRAL DE F1. Duas séries com estruturas de dependência muito diferentes (i.i.d. vs.
    GARCH) produzem escalas nulas muito diferentes na estatística CRUA — é exatamente essa disparidade
    que hoje obriga o modelo a gastar 34% do |SHAP| em `meta_h0_*` para recalibrar. Depois de `_cal`,
    as duas devem ficar na mesma escala."""
    key = "ranktwo_dispersion_z_w100"
    rng = np.random.RandomState(9)

    iid = _online_stats(rng.randn(4000), rng.randn(3000), cfg, [key, f"{key}_cal"])
    g_hist, g_online, _ = generate("t6", seed=1, cfg=cfg)  # GARCH puro, sem quebra
    garch = _online_stats(g_hist, g_online, cfg, [key, f"{key}_cal"])

    raw_ratio = garch[key].std(ddof=1) / iid[key].std(ddof=1)
    cal_ratio = garch[f"{key}_cal"].std(ddof=1) / iid[f"{key}_cal"].std(ddof=1)

    # a estatística crua tem escala nula bem diferente entre as duas séries...
    assert raw_ratio > 1.5 or raw_ratio < 1 / 1.5, f"premissa não se sustenta nestes dados: {raw_ratio:.2f}"
    # ...e a calibrada aproxima as duas escalas
    assert abs(math.log(cal_ratio)) < abs(math.log(raw_ratio)), (
        f"calibração não aproximou as escalas: cru={raw_ratio:.2f}, cal={cal_ratio:.2f}"
    )
