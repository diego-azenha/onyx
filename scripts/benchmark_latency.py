#!/usr/bin/env python
"""CLI fina: microbenchmark de latência por passo (plano §11). Gate: <= latency_budget_us_per_step
(configs/default.yaml, plano §11.1)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--n-series", type=int, default=100)
    parser.add_argument("--t-per-series", type=int, default=1000)
    parser.add_argument("--out", default="artifacts/reports/latency.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.RandomState(cfg.seed)

    total_steps = 0
    total_time = 0.0
    for _ in tqdm(range(args.n_series), desc="benchmark de latência"):
        hist = rng.randn(2000)
        online = rng.randn(args.t_per_series)
        h0 = fit_h0(hist, cfg)
        scorer = StreamScorer(h0, default_blocks(), None, cfg)

        start = time.perf_counter()
        for x in online:
            scorer.update(float(x))
        total_time += time.perf_counter() - start
        total_steps += len(online)

    us_per_step = (total_time / total_steps) * 1e6
    budget = cfg.gates.latency_budget_us_per_step
    passed = us_per_step <= budget
    print(f"{us_per_step:.2f} us/passo (orçamento: {budget} us/passo) -> {'PASS' if passed else 'FAIL'}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"us_per_step": us_per_step, "budget_us_per_step": budget, "passed": passed}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
