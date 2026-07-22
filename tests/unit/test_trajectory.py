"""F4+F9 (docs/BACKLOG_TSAUC.md): trajetória do estatístico.

O teste que importa é `test_separates_sustained_break_from_transient_spike`: valida a premissa da
frente — que a FORMA da rampa separa quebra sustentada de spike transitório, que é o problema de
falso-positivo, e que o valor instantâneo não separa (os dois cenários passam pelo mesmo pico)."""
from __future__ import annotations

import math

from sbrt.state.trajectory import TrajectoryBlock


def _run(cfg, values, key="accum_window_var_ln_w100_cal"):
    blk = TrajectoryBlock()
    blk.reset(None, cfg)
    out = []
    for t, v in enumerate(values, start=1):
        feats = {key: v}
        blk.update_from_feats(feats, t)
        out.append(blk.features())
    return out


def test_emits_nan_during_warmup(cfg):
    got = _run(cfg, [0.0] * 3)
    for key, val in got[-1].items():
        assert math.isnan(val), key


def test_tracks_every_configured_alias(cfg):
    got = _run(cfg, [0.5] * 50)[-1]
    for _, alias in cfg.trajectory.track:
        for suffix in ("slope", "mono", "persist", "since_first", "area"):
            assert f"traj_{alias}_{suffix}" in got


def test_nan_statistic_does_not_contaminate_the_trajectory(cfg):
    """Um estatístico ainda em warm-up emite NaN. Tratá-lo como 0,0 inventaria uma queda brusca na
    rampa no exato passo em que a feature acorda."""
    vals = [math.nan] * 20 + [3.0] * 30
    got = _run(cfg, vals)
    assert got[10]["traj_var_slope"] is not None and math.isnan(got[10]["traj_var_slope"])
    final = got[-1]
    assert math.isfinite(final["traj_var_slope"])
    assert final["traj_var_persist"] == 30.0      # contou só os passos reais
    assert abs(final["traj_var_slope"]) < 0.2     # constante em 3.0 -> inclinação ~0, não um salto


def test_separates_sustained_break_from_transient_spike(cfg):
    """PREMISSA CENTRAL DE F4. Os dois cenários atingem o MESMO pico (4,0); só a forma difere.
    Nenhuma feature de valor instantâneo os separa no pico — a trajetória separa."""
    n_pre, n_post = 60, 60
    sustained = [0.0] * n_pre + [4.0] * n_post          # sobe e FICA
    spike = [0.0] * n_pre + [4.0] * 3 + [0.0] * (n_post - 3)  # sobe e REVERTE

    s = _run(cfg, sustained)[-1]
    k = _run(cfg, spike)[-1]

    assert s["traj_var_persist"] > k["traj_var_persist"] + 40
    assert s["traj_var_area"] > k["traj_var_area"] * 5
    # o tempo-desde-a-primeira-excedência é parecido (ambos excederam cedo) -- é a persistência e a
    # área que carregam a distinção, e é isso que a frente promete
    assert s["traj_var_since_first"] > 0 and k["traj_var_since_first"] > 0


def test_slope_and_monotonicity_capture_a_ramp(cfg):
    ramp = [i * 0.1 for i in range(80)]
    flat = [3.0] * 80
    r, f = _run(cfg, ramp)[-1], _run(cfg, flat)[-1]
    assert r["traj_var_slope"] > f["traj_var_slope"] + 0.05
    assert r["traj_var_mono"] > 0.9      # subiu em todo passo
    assert abs(f["traj_var_mono"]) < 0.2


def test_area_is_reflected_at_zero_so_quiet_periods_do_not_go_negative(cfg):
    """O integrador é Page-Hinkley, não uma soma cumulativa livre: um período longo abaixo do limiar
    não pode gerar uma dívida negativa que mascare uma quebra posterior."""
    vals = [-5.0] * 200 + [4.0] * 20
    got = _run(cfg, vals)
    assert got[199]["traj_var_area"] == 0.0
    assert got[-1]["traj_var_area"] > 30.0


def test_never_exceeded_is_a_distinct_ordered_value(cfg):
    got = _run(cfg, [0.0] * 60)[-1]
    assert got["traj_var_since_first"] == -1.0
    assert got["traj_var_persist"] == 0.0
