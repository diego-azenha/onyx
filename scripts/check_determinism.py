#!/usr/bin/env python
"""CLI fina: checklist de determinismo pré-submissão (plano §15.2) — re-execução de 30% bit a bit,
mais estrita que a tolerância 1e-8 da plataforma (F9)."""
from __future__ import annotations

import argparse
import sys

import numpy as np

from sbrt.adversarial.determinism import rerun_bitexact
from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--n-series", type=int, default=50)
    parser.add_argument("--fraction", type=float, default=0.3)
    args = parser.parse_args()

    cfg = load_config(args.config)
    rng = np.random.RandomState(cfg.seed)
    sample = [(rng.randn(2000), rng.randn(int(rng.randint(10, 500)))) for _ in range(args.n_series)]

    def factory(hist):
        h0 = fit_h0(hist, cfg)
        return StreamScorer(h0, default_blocks(), None, cfg)

    ok = rerun_bitexact(sample, factory, args.fraction, cfg.seed, progress=True)
    print("determinismo:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
