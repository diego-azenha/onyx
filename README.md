# structural-break-rt

Real-time structural break detector for the ADIA Lab Structural Break Challenge: Real-Time Edition (CrunchDAO).

Causal, per-step break-probability scorer: a deterministic sequential statistics engine (whitening,
CUSUM bank, single-changepoint Bayesian filter, conformal martingales, rolling windows/EWMAs) feeds a
LightGBM calibrator trained on the per-step label `y_t = 1{tau <= t}`.

## Quickstart

```bash
pip install -e ".[dev]"
make ci            # unit + causality + determinism tests
make robustness     # synthetic scenario suite (T1-T13), behavioral gates
make benchmark       # per-step latency microbenchmark
make dataset && make train   # build training rows + fit the LightGBM ensemble
make smoke           # runs adapter/platform.py end to end (crunch test)
```

## Documentation

- [`docs/PLANO_TECNICO.md`](docs/PLANO_TECNICO.md) — theoretical/methodological plan (source of truth for
  formulas, hyperparameters, gates). Referenced throughout the code as `§N`.
- [`docs/PLANO_REPOSITORIO.md`](docs/PLANO_REPOSITORIO.md) — engineering plan (directory layout, module
  contracts, test strategy, execution phases).
- [`docs/CONTRACTS.md`](docs/CONTRACTS.md) — frozen interfaces for `src/sbrt/*`.

## Key design decision

No code in this repository computes or reports a local estimate of TS-AUC as a substitute for the
official score (see `docs/PLANO_TECNICO.md` §9.0). Local tooling verifies **correctness** (causality,
determinism, behavioral sanity on synthetic scenarios) — performance decisions are made by official
submission only.
