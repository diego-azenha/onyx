#!/usr/bin/env python
"""I8 (knowledge/frentes/estrategias-01-10/README.md): um modelo por faixa de t.
`--split` grava data/processed/train_rows_f{k}.parquet; `--junta` concatena os OOF por faixa e compara
com o E1 da mesma semente (grade do board)."""
import sys
import numpy as np
import pandas as pd
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

FAIXAS = ((1, 50), (51, 150), (151, 400), (401, 10**6))
if sys.argv[1] == "--split":
    rows = pd.read_parquet("data/processed/train_rows.parquet")
    for k, (a, b) in enumerate(FAIXAS):
        rows[(rows.t >= a) & (rows.t <= b)].to_parquet(f"data/processed/train_rows_f{k}.parquet", index=False)
        print(k, a, b, flush=True)
else:
    s = sys.argv[2]
    o = pd.concat([pd.read_parquet(f"artifacts/models/oof_i8f{k}_s{s}.parquet") for k in range(4)]).sort_values(["id", "t"])
    e = pd.read_parquet(f"artifacts/models/oof_e1_s{s}.parquet").sort_values(["id", "t"])
    assert len(o) == len(e) and (o.t.values == e.t.values).all()
    t, y = e.t.values, e.y.values; w = board_grid_multipliers("data/y_train.parquet", t)
    print({"e1": weighted_ts_auc(t, y, e.oof_pred.values, w), "i8_faixas": weighted_ts_auc(t, y, o.oof_pred.values, w)})
    o.to_parquet(f"artifacts/models/oof_i8_s{s}.parquet", index=False)
