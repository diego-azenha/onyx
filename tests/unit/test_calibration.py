"""F1 (docs/PROPOSTA_FEATURES_V2.md): calibração de nulo por série.

O teste que importa é `test_calibration_equalizes_null_scale_across_series`: ele mede diretamente a
premissa da proposta — que estatísticas cruas têm escala nula dependente da série (e portanto são
mal-ordenadas na seção transversal que a TS-AUC avalia), e que a versão `_cal` remove essa
dependência."""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from sbrt.robustness.generators import generate
from sbrt.state.calibration import (
    NullSpec,
    _null_at,
    _rolling_mean,
    _upper_tail_p_vec,
    apply_calibration,
    history_evol,
)
from sbrt.state.conformal import _upper_tail_p
from sbrt.state.h0 import fit_h0, seed_lag_buffer, whiten_step
from sbrt.state.scorer import StreamScorer, default_blocks
from sbrt.utils.numerics import vol_adjust_step


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


def _scorer_evol_sequence(hist, online, cfg):
    """A sequência `e_vol` que o `StreamScorer` produz internamente, reconstruída pelo MESMO caminho
    que ele usa (whiten_step + o galho de vol-adjust de `update_features`). É o lado "online" da
    equivalência que `history_evol` precisa honrar."""
    h0 = fit_h0(hist, cfg)
    lags = seed_lag_buffer(h0)
    use_vol_adjust = h0.rho1_abs_e > cfg.state.vol_adjust["threshold_rho1_abs"]
    lam = cfg.state.vol_adjust["lambda_v"]
    v = 1.0
    e_seq, evol_seq = [], []
    for x in online:
        e, _ = whiten_step(float(x), lags, h0, cfg)
        if use_vol_adjust:
            v, e_vol = vol_adjust_step(v, e, lam)
        else:
            e_vol = e
        e_seq.append(e)
        evol_seq.append(e_vol)
    return np.array(e_seq), np.array(evol_seq), h0


def test_history_evol_matches_online_recursion_bit_exactly(cfg):
    """F1.0 — o requisito inteiro da função. Alimentada com a MESMA sequência de `e` que o laço
    online viu, `history_evol` tem que devolver exatamente os mesmos `e_vol`, bit a bit. Um desvio
    aqui envenenaria o nulo de toda feature `e_vol`-based sem levantar nenhum erro visível, então o
    teste é de igualdade exata, não de `allclose`."""
    g_hist, g_online, _ = generate("t6", seed=3, cfg=cfg)  # GARCH: liga o ajuste de volatilidade
    e_seq, evol_online, h0 = _scorer_evol_sequence(g_hist, g_online, cfg)
    assert h0.rho1_abs_e > cfg.state.vol_adjust["threshold_rho1_abs"], "cenário não exercita o ramo ligado"

    evol_replay = history_evol(e_seq, h0.rho1_abs_e, cfg)

    assert np.array_equal(evol_replay, evol_online)


def test_history_evol_is_identity_when_adjustment_is_off(cfg):
    """Abaixo do limiar de rho1_abs o `StreamScorer` usa `e_vol = e` — o replay tem que seguir o
    mesmo galho, senão o nulo sai medido numa escala que o online nunca produz."""
    rng = np.random.RandomState(11)
    e = rng.randn(500)
    thr = cfg.state.vol_adjust["threshold_rho1_abs"]
    out = history_evol(e, rho1_abs_e=thr - 0.01, cfg=cfg)
    assert np.array_equal(out, e)
    # e no limiar exato: o online usa `>`, então o ajuste continua DESLIGADO
    assert np.array_equal(history_evol(e, rho1_abs_e=thr, cfg=cfg), e)


def test_history_evol_does_not_mutate_input(cfg):
    rng = np.random.RandomState(12)
    e = rng.randn(200)
    before = e.copy()
    history_evol(e, rho1_abs_e=0.9, cfg=cfg)
    history_evol(e, rho1_abs_e=0.0, cfg=cfg)
    assert np.array_equal(e, before)


def test_history_evol_normalizes_volatility_scale(cfg):
    """O que a função existe para fazer: sob clustering de volatilidade, `e_vol` tem escala local
    estabilizada em relação a `e` cru."""
    rng = np.random.RandomState(13)
    # regime de volatilidade em degrau — o caso que o ajuste deve achatar
    e = np.concatenate([rng.randn(1000), 4.0 * rng.randn(1000)])
    out = history_evol(e, rho1_abs_e=0.9, cfg=cfg)
    ratio_raw = e[1000:].std(ddof=1) / e[:1000].std(ddof=1)
    ratio_adj = out[1000:].std(ddof=1) / out[:1000].std(ddof=1)
    assert ratio_raw > 3.0
    assert ratio_adj < ratio_raw / 2.0


# Os braços F1.a/F1.b-1 foram medidos por R0 e REGREDIRAM (Delta geral -0,0069, IC exclui 0 contra
# — docs/BACKLOG_TSAUC.md), então `calibration.recursive_features` está VAZIO em produção. A
# maquinaria continua no repositório, e estes testes continuam exercitando-a com uma config própria:
# o que foi revertido é a decisão de ligar as colunas, não a corretude do mecanismo, e reabrir um
# braço é acrescentar linhas no YAML.
F1A_RECURSIVE = {
    "cusum_dep_pos": "none",
    "cusum_dep_neg": "none",
    "accum_global_rho1_fz": "none",
    "accum_window_rho1_fz_w100": "none",
    "conformal_logm_abs": "cumsum",
    "conformal_logm_abs_reset": "none",
}


def _cfg_with_recursive(cfg, mapping=None):
    """Config idêntica à de produção, mas com braços recursivos ligados."""
    return replace(cfg, calibration=replace(
        cfg.calibration, recursive_features=mapping or F1A_RECURSIVE))


def test_production_enables_no_recursive_arm(cfg):
    """Trava do resultado do R0: com a whitelist vazia, nenhuma coluna recursiva entra e o conjunto
    de features volta a ser o do V4. Se alguém religar um braço, é uma decisão que precisa de R0
    própria — este teste força a conversa."""
    assert cfg.calibration.recursive_features == ()
    rng = np.random.RandomState(20)
    h0 = fit_h0(rng.randn(3000), cfg)
    assert not any(spec.table for spec in h0.null_stats.values())


def test_calibration_registers_exactly_the_declared_recursive_columns(cfg):
    """Os braços são deliberadamente estreitos — se este teste pegar colunas a mais, a mudança deixou
    de ser atribuível e repete o erro do V5 (empacotar duas mudanças e não isolar nenhuma). A
    whitelist é a única fonte da verdade sobre o que entra."""
    c = _cfg_with_recursive(cfg)
    rng = np.random.RandomState(21)
    h0 = fit_h0(rng.randn(3000), c)
    declared = {n for n, _ in c.calibration.recursive_features}
    assert {n for n, s in h0.null_stats.items() if s.table} == declared
    for name, kind in c.calibration.recursive_features:
        assert h0.null_stats[name].kind == kind, name


def test_transient_table_is_used_below_K_and_stationary_above(cfg):
    spec = NullSpec(mu=9.0, sd=3.0, min_t=1, kind="none", table=((1.0, 2.0), (0.5, 0.7)))
    assert _null_at(spec, t=1) == (1.0, 0.5)
    assert _null_at(spec, t=2) == (2.0, 0.7)
    assert _null_at(spec, t=3) == (9.0, 3.0)          # além da tabela -> estacionário
    assert _null_at(spec, t=10_000) == (9.0, 3.0)


def test_cusum_transient_null_grows_with_t(cfg):
    """A curva de transiente tem que refletir o que foi medido: o CUSUM parte de 0 e sobe até a
    distribuição estacionária. Um nulo achatado (constante em t) é justamente o erro que a tabela
    existe para evitar — ele faria a calibrada acender cedo demais, em todas as séries."""
    c = _cfg_with_recursive(cfg)
    rng = np.random.RandomState(22)
    h0 = fit_h0(rng.randn(4000), c)
    mu_by_t, sd_by_t = h0.null_stats["cusum_dep_pos"].table
    assert len(mu_by_t) == c.calibration.transient_restart_every
    assert mu_by_t[4] < mu_by_t[24] < mu_by_t[-1]
    assert sd_by_t[4] < sd_by_t[-1]
    assert all(v > 0 for v in sd_by_t)


def test_cumsum_kind_removes_the_linear_drift_of_conformal_martingales(cfg):
    """F1.b-1, a premissa. Sob H0 o log-martingale conformal sem reset deriva linearmente em t (o
    incremento tem esperança negativa), então o nível CRU num passo é dominado por t·deriva(série) —
    escala idiossincrática que a TS-AUC pune. A versão `_cal` tem que ficar sem tendência em t."""
    c = _cfg_with_recursive(cfg)
    rng = np.random.RandomState(31)
    key = "conformal_logm_abs"
    assert c.calibration.transient_restart_every < 400  # a janela abaixo excede a tabela de propósito
    got = _online_stats(rng.randn(4000), rng.randn(400), c, [key, f"{key}_cal"])

    raw, cal = got[key], got[f"{key}_cal"]
    steps = np.arange(1, len(raw) + 1, dtype=float)
    slope_raw = np.polyfit(steps, raw, 1)[0]
    slope_cal = np.polyfit(steps[-len(cal):], cal, 1)[0]

    assert slope_raw < -0.05, f"premissa não se sustenta: deriva crua {slope_raw:.4f}/passo"
    # a calibrada é ~N(0,1) e sem tendência: a deriva residual some frente à do cru
    assert abs(slope_cal) < abs(slope_raw) / 20
    assert abs(float(cal.mean())) < 1.5
    assert 0.2 < float(cal.std(ddof=1)) < 3.0


def test_full_window_column_keeps_its_honest_warmup(cfg):
    """`dep_mass_evol_w100` é estatística de JANELA CHEIA, não recursiva: antes de 100 amostras ela
    não existe, e a tabela de transiente não se aplica. Fica NaN em `t<=50` exatamente como a
    `dep_mass_abs_w100` que já vinha do V4 — é o comportamento certo, mas significa que esta coluna
    não ajuda o bucket precoce. Fixado aqui para não ser confundido com regressão."""
    rng = np.random.RandomState(24)
    h0 = fit_h0(rng.randn(3000), cfg)
    # `dep_mass_evol_w100` entrou no braço F1.a e saiu com ele: sem o `e_vol` real, o histórico não
    # reproduz a estatística e calibrá-la seria medir o nulo errado.
    assert "dep_mass_evol_w100" not in h0.null_stats
    assert h0.null_stats["dep_mass_abs_w100"].min_t == cfg.dependence.mass_window


def test_recursive_columns_are_available_and_standardized_in_the_early_bucket(cfg):
    """O ponto inteiro da tabela de transiente: as colunas RECURSIVAS têm que estar VIVAS e ~N(0,1)
    em `t<=50`, não NaN. Uma coluna 100% NaN em t pequeno dilui o sorteio de `feature_fraction=0,8`
    e piora justamente o bucket que este trabalho quer melhorar (docs/NOTAS_AGENTES.md §7)."""
    c = _cfg_with_recursive(cfg)
    keys = [f"{n}_cal" for n, _ in c.calibration.recursive_features]
    per_key = {k: [] for k in keys}
    for s in range(60):                                   # 60 séries H0 -> corte transversal
        r = np.random.RandomState(500 + s)
        h0 = fit_h0(r.randn(2500), c)
        scorer = StreamScorer(h0, default_blocks(), None, c)
        for t in range(1, 51):                            # só o bucket t<=50
            feats = scorer.update_features(float(r.randn()))
            if t >= 25:
                for k in keys:
                    v = feats.get(k, math.nan)
                    if math.isfinite(v):
                        per_key[k].append(v)

    for k in keys:
        vals = np.array(per_key[k], dtype=float)
        assert len(vals) > 500, f"{k} praticamente indisponível em t<=50: {len(vals)} valores"
        assert abs(float(vals.mean())) < 1.0, f"{k} enviesada em t<=50: media={vals.mean():.2f}"
        assert 0.3 < float(vals.std(ddof=1)) < 3.0, f"{k} fora de escala em t<=50"


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
        assert spec.kind in ("none", "z", "var_ln", "frac", "rho", "cumsum")


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
