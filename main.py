"""Entry point loaded by the CrunchDAO platform runner (`crunch test` / `crunch push`, default
`--main-file main.py`). All logic lives in `src/sbrt/adapter/platform.py` (plano §15.1) — this file
is a thin re-export so the platform's `load_user_code` finds `train`/`infer` where it expects them."""
import os

from sbrt.adapter.platform import infer, train  # noqa: F401

# Officially supported by the platform (see quickstarter_notebook.ipynb): each series is scored by
# its own independent StreamScorer (no cross-series state), so running infer() across processes is
# safe and doesn't change any single series' result. Matters little for the small reduced test set
# (~34s) but helps on the full private test set (up to ~1e7 steps per plano_structural_break_realtime.md §11).
INFER_PARALLELISM = max(1, os.cpu_count() - 1)
