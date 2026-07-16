#!/usr/bin/env python
"""CLI fina: roda a suíte de robustez T1-T13(+T5b,T12b) sobre o scorer congelado e aplica os gates
comportamentais (plano §10). Gates de MEDIANA/limiar, deliberadamente NÃO TS-AUC (plano §9.0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.robustness.gates import evaluate
from sbrt.robustness.generators import CONTROLLED_SCENARIOS, SCENARIO_IDS, generate
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def _run_scorer(hist, online, cfg, ensemble=None) -> list:
    h0 = fit_h0(hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), ensemble, cfg)
    return [scorer.update(float(x)) for x in online]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--n-seeds", type=int, default=200)
    parser.add_argument("--out", default="artifacts/reports/robustness.json")
    parser.add_argument("--model", default=None, help="path to a trained ModelEnsemble dir; omit for fallback")
    args = parser.parse_args()

    cfg = load_config(args.config)

    ensemble = None
    if args.model:
        from sbrt.model.predict import ModelEnsemble

        ensemble = ModelEnsemble.load(args.model)

    results = {}
    all_passed = True

    for sid in tqdm(SCENARIO_IDS, desc="suíte de robustez"):
        trajs, tau = [], None
        for seed in range(args.n_seeds):
            hist, online, tau = generate(sid, seed, cfg)
            trajs.append(_run_scorer(hist, online, cfg, ensemble))

        ctrl_trajs = None
        if sid in CONTROLLED_SCENARIOS:
            ctrl_trajs = []
            for seed in range(args.n_seeds):
                hist, online, _ = generate(f"{sid}_ctrl", seed, cfg)
                ctrl_trajs.append(_run_scorer(hist, online, cfg, ensemble))

        result = evaluate(sid, trajs, ctrl_trajs, tau, cfg)
        results[sid] = {"passed": result.passed, "details": result.details}
        all_passed = all_passed and result.passed
        print(f"{sid}: {'PASS' if result.passed else 'FAIL'} {result.details}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nresultado geral: {'TODOS OS GATES PASSARAM' if all_passed else 'HÁ GATES REPROVADOS'} -> {out_path}")


if __name__ == "__main__":
    main()
