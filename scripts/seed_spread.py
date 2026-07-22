#!/usr/bin/env python
"""TS-AUC de cada modelo, semente por semente — a apresentação honesta de um braço de R0.

Um Δ pareado contra um baseline único esconde o que importa depois da medição de 2026-07-22: o
booster é um sorteio, e o desvio entre sorteios (~0,004) é da ordem dos efeitos procurados. Aqui as
duas famílias de modelos aparecem como DISTRIBUIÇÕES — K valores de cada lado — e a decisão passa a
ser sobre médias com dispersão visível, não sobre um par de números.

Uso:
    python scripts/seed_spread.py --group V4 artifacts/models/oof_v4*.parquet \
                                  --group F2 artifacts/models/oof_f2*.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import weighted_ts_auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", nargs="+", action="append", required=True,
                        metavar=("NOME", "PARQUET"),
                        help="nome do grupo seguido dos parquets OOF daquele grupo")
    parser.add_argument("--score-col", default="oof_pred")
    args = parser.parse_args()

    summary = {}
    for spec in args.group:
        name, paths = spec[0], spec[1:]
        vals = []
        print(f"\n{name}")
        for p in paths:
            df = pd.read_parquet(p)
            a = weighted_ts_auc(df["t"].to_numpy(np.float64), df["y"].to_numpy(np.float64),
                                df[args.score_col].to_numpy(np.float64))
            vals.append(a)
            print(f"  {Path(p).name:34s} {a:.4f}")
        v = np.asarray(vals, dtype=np.float64)
        summary[name] = v
        sd = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        print(f"  {'MEDIA':34s} {v.mean():.4f}   (dp entre sementes {sd:.4f}, n={len(v)})")

    names = list(summary)
    if len(names) == 2:
        a, b = summary[names[0]], summary[names[1]]
        diff = b.mean() - a.mean()
        # erro-padrão da diferença de médias, só da componente de SEMENTE (a de séries é comum aos
        # dois lados e o bootstrap pareado do compare_oof.py é quem a mede)
        se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))) if min(len(a), len(b)) > 1 else float("nan")
        print(f"\n{names[1]} - {names[0]}: {diff:+.4f}   (EP da componente de semente: {se:.4f})")
        print("Leitura: |diferença| menor que ~2 EP = indistinguível do sorteio de semente.")
        print("O juiz do efeito ENTRE séries continua sendo compare_oof.py sobre as OOF médias.")


if __name__ == "__main__":
    main()
