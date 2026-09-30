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

All documentation lives in the knowledge base [`knowledge/`](knowledge/README.md) (formerly `docs/`;
the README there has the old-path → new-path redirect table that code comments still refer to).

- [`knowledge/modelo/MODELO.md`](knowledge/modelo/MODELO.md) — the model: formulation, design decisions, formulas,
  hyperparameters, gates. Referenced throughout the code as `§N` (numbering preserved from the
  original technical plan).
- [`knowledge/historico/HISTORICO.md`](knowledge/historico/HISTORICO.md) — every change made over time, with measured results and
  the decision taken (what worked, what was reverted, what not to retry).
- [`knowledge/operacao/NOTAS_AGENTES.md`](knowledge/operacao/NOTAS_AGENTES.md) — operational notes for whoever edits the repo:
  invariants, frozen contracts, commands, measurement protocol, known pitfalls, open issues.
- [`knowledge/diario/`](knowledge/diario/) — dated log of what was done, linking experiment records.

## Key design decision

No code in this repository computes or reports a local estimate of TS-AUC as a substitute for the
official score (see `knowledge/modelo/MODELO.md` §9.0). Local tooling verifies **correctness** (causality,
determinism, behavioral sanity on synthetic scenarios) — performance decisions are made by official
submission only.
