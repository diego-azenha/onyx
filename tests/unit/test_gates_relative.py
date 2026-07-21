"""Gates relativos (R5, docs/PARECER_AUDITORIA_ONYX.md §6-R5): quando `reference_trajectories` é
passado para `evaluate`, os cenários em RELATIVE_GATE_SCENARIOS (t2/t6/t9/t10/t13) devem gatear no
GAP contra o painel, não no nível absoluto; sem `reference_trajectories` (None), o comportamento
absoluto original deve ficar bit-a-bit inalterado (compatibilidade com o modo fallback e com
tests/robustness/test_gates.py)."""
from __future__ import annotations

from sbrt.robustness.gates import evaluate


def _flat_traj(value: float, n: int) -> list:
    return [value] * n


def test_t6_absolute_mode_unchanged_without_reference(cfg):
    trajs = [_flat_traj(0.30, 1000) for _ in range(5)]
    r_abs = evaluate("t6", trajs, None, None, cfg)
    assert r_abs.passed  # 0.30 <= mean_max (0.40)
    assert "reference_final_mean" not in r_abs.details


def test_t6_relative_gate_uses_gap_not_level(cfg):
    # nível absoluto 0.75 reprovaria o gate absoluto (mean_max=0.40); mas se o painel de referência
    # (score do calibrador sob ruído honesto, sem quebra nenhuma) também flutua perto de 0.70 --
    # calibrador com offset alto mas discriminação boa --, o que importa é o GAP acima do próprio
    # piso, que aqui é pequeno (0.05) e deve passar mesmo que o nível absoluto reprovaria.
    trajs = [_flat_traj(0.75, 1000) for _ in range(5)]
    ref = [_flat_traj(0.70, 1000) for _ in range(5)]
    r_rel = evaluate("t6", trajs, None, None, cfg, reference_trajectories=ref)
    assert abs(r_rel.details["gap"] - 0.05) < 1e-9
    assert r_rel.passed  # gap 0.05 <= mean_max 0.40, apesar do nivel absoluto 0.75 > 0.40


def test_t6_relative_gate_passes_when_gap_small(cfg):
    trajs = [_flat_traj(0.50, 1000) for _ in range(5)]
    ref = [_flat_traj(0.45, 1000) for _ in range(5)]
    r_rel = evaluate("t6", trajs, None, None, cfg, reference_trajectories=ref)
    assert abs(r_rel.details["gap"] - 0.05) < 1e-9
    assert r_rel.passed  # gap 0.05 <= mean_max 0.40


def test_t6_relative_gate_fails_when_gap_large(cfg):
    trajs = [_flat_traj(0.90, 1000) for _ in range(5)]
    ref = [_flat_traj(0.10, 1000) for _ in range(5)]
    r_rel = evaluate("t6", trajs, None, None, cfg, reference_trajectories=ref)
    assert abs(r_rel.details["gap"] - 0.80) < 1e-9
    assert not r_rel.passed  # gap 0.80 > mean_max 0.40


def test_t9_relative_gate_gap(cfg):
    trajs = [_flat_traj(0.5, 1000) for _ in range(3)]
    ref = [_flat_traj(0.45, 1000) for _ in range(3)]
    r_rel = evaluate("t9", trajs, None, None, cfg, reference_trajectories=ref)
    assert abs(r_rel.details["gap"] - 0.05) < 1e-9


def test_t2_relative_gate_gap(cfg):
    trajs = [_flat_traj(0.4, 400) for _ in range(3)]
    ref = [_flat_traj(0.38, 400) for _ in range(3)]
    r_rel = evaluate("t2", trajs, None, tau=200, cfg=cfg, reference_trajectories=ref)
    assert abs(r_rel.details["gap"] - 0.02) < 1e-9
    assert r_rel.passed  # gap 0.02 <= mean_prebreak_max 0.35


def test_t13_relative_gate_gap(cfg):
    import numpy as np

    n = 600
    scenario_traj = np.zeros(n)
    scenario_traj[199:260] = 1.0  # excursao t=200..260, decai a 0
    ref_traj = np.zeros(n)
    ref_traj[199:260] = 0.1  # painel de referencia tem um ruido residual bem menor
    r_rel = evaluate(
        "t13", [scenario_traj.tolist()], None, None, cfg, reference_trajectories=[ref_traj.tolist()]
    )
    assert "reference_decay" in r_rel.details


def test_absolute_mode_matches_reference_none_for_all_relative_scenarios(cfg):
    # sanity: para cada cenario em RELATIVE_GATE_SCENARIOS, reference_trajectories=None reproduz
    # exatamente o gate absoluto (nenhuma chave "gap"/"reference_*" aparece nos details).
    from sbrt.robustness.generators import RELATIVE_GATE_SCENARIOS

    trajs_by_scenario = {
        "t2": [_flat_traj(0.1, 400)],
        "t6": [_flat_traj(0.1, 1000)],
        "t9": [_flat_traj(0.1, 1000)],
        "t10": [_flat_traj(0.1, 1000)],
        "t13": [_flat_traj(0.1, 600)],
    }
    for sid in RELATIVE_GATE_SCENARIOS:
        tau = 200 if sid == "t2" else None
        r = evaluate(sid, trajs_by_scenario[sid], None, tau, cfg)
        assert not any(k.startswith("reference_") or k == "gap" for k in r.details)
