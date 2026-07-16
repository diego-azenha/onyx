"""T1-T13(+T5b,T12b) contra os gates comportamentais (plano §10) — mais lento, por isso fora do
alvo `ci` do Makefile (só `test`). Roda sobre o scorer com o fallback puro-estatístico (§8.5), não
o ensemble supervisionado (que ainda não foi treinado nesta fase do projeto).

Algumas cenários são conhecidamente difíceis para o fallback cru (sem calibração, combina só 3
sinais): T1 (quebra bem no início, baixa informação), T3 (shift sutil), T6 (GARCH sem quebra — o
fallback não tem o condicionamento por ruído que o LightGBM aprenderia), T8 (mudança pura de forma
— o fallback não usa nenhuma feature de forma/quantil) e T9 (outliers isolados sem quebra: com
N_SEEDS=40, o outlier de 6 sigma em t=700 deixa só ~300 passos para decair até t=999 — o candidato
bayesiano nascido do outlier tem variância preditiva inflada pelo prior (kappa0=0.5, sigma0_sq=1.5,
plano §4.3) e por isso decai devagar; medido: final_mean~0.50 contra o limiar 0.35). Isso é esperado
e documentado (docs/PLANO_REPOSITORIO.md §7.3, Checkpoint 2: "gates ... mesmo os reprovados — são a
régua do ganho do modelo"). Este teste é uma trava de regressão: falhas NOVAS além dessa lista são
bugs — reabrir esta lista é uma decisão de calibração (P3), não um ajuste silencioso do teste."""
from __future__ import annotations

import numpy as np

from sbrt.robustness.generators import CONTROLLED_SCENARIOS, SCENARIO_IDS, generate
from sbrt.robustness.gates import evaluate
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks

N_SEEDS = 40
KNOWN_FALLBACK_WEAKNESSES = {"t1", "t3", "t6", "t8", "t9"}


def _run_scorer(hist, online, cfg):
    h0 = fit_h0(hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)
    return [scorer.update(float(x)) for x in online]


def test_robustness_suite_no_unexpected_regressions(cfg):
    results = {}
    for sid in SCENARIO_IDS:
        trajs, tau = [], None
        for seed in range(N_SEEDS):
            hist, online, tau = generate(sid, seed, cfg)
            trajs.append(_run_scorer(hist, online, cfg))

        ctrl_trajs = None
        if sid in CONTROLLED_SCENARIOS:
            ctrl_trajs = [
                _run_scorer(*generate(f"{sid}_ctrl", seed, cfg)[:2], cfg) for seed in range(N_SEEDS)
            ]

        results[sid] = evaluate(sid, trajs, ctrl_trajs, tau, cfg)

    failed = {sid for sid, r in results.items() if not r.passed}
    unexpected = failed - KNOWN_FALLBACK_WEAKNESSES
    details = {sid: results[sid].details for sid in unexpected}
    assert not unexpected, f"cenários com falha NÃO documentada: {details}"
