#!/usr/bin/env python
"""X1 (knowledge/frentes/teto-offline/X1-excursoes.md): famílias de CUSUM calibradas pelo nulo de
excursões da própria série, para as 10k séries de treino. TS-AUC por subconjunto de tipo de quebra e
global, cru (`cus_*`) contra calibrado (`exc_*`)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from surpresa import carregar_config  # noqa: E402
from surpresa.evidencia import Alternativas  # noqa: E402
from surpresa.excursao import pontuar_excursao  # noqa: E402


def _lote(series, cfg):
    alt = Alternativas(cfg)
    out = []
    for sid, h, on in series:
        o = pontuar_excursao(h, on, cfg, alt)
        o["id"] = np.full(len(on), sid, dtype=np.int32)
        o["t"] = np.arange(1, len(on) + 1, dtype=np.int32)
        out.append({k: np.asarray(v, dtype=np.float32) if k not in ("id", "t") else v for k, v in o.items()})
    return out


def main() -> None:
    cfg = carregar_config()
    series = _carregar_series(Path("data"))
    lotes = [series[i:i + 50] for i in range(0, len(series), 50)]
    res = Parallel(n_jobs=3)(delayed(_lote)(l, cfg) for l in lotes)
    df = pd.DataFrame({k: np.concatenate([o[k] for r in res for o in r]) for k in res[0][0]})
    df.to_parquet("artifacts/surpresa/excursoes.parquet")
    print(df.shape)


if __name__ == "__main__":
    main()
