#!/usr/bin/env python
"""A2 (BRAINSTORM_RUPTURA_V2.md §1.1): stacking HETEROGÊNEO em espaço de percentil-OOF com pesos
convexos POR BUCKET de t, ajustados por OOF ANINHADO (para o fold k, os pesos vêm dos outros 4 folds).
Membros típicos: bag binário (X1-soft) · rank R3 (K=4) · fallback determinístico. O fallback não tem
efeitos-fixos aprendidos e pode ser mais forte em t≤50, onde o supervisionado é mais fraco — daí pesos
por bucket. Implantação causal = mapas de quantis OOF por membro × bin de t, congelados (offline aqui).
Veredito final por `compare_oof.py`."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc

EDGES = [0, 50, 150, 400, np.inf]
LABELS = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]


def _simplex(M: int, step: float = 0.1):
    n = round(1 / step)

    def comps(tot, k):
        if k == 1:
            yield (tot,)
            return
        for i in range(tot + 1):
            for rest in comps(tot - i, k - 1):
                yield (i,) + rest
    return [np.array(c, dtype=np.float64) / n for c in comps(n, M)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--members", nargs="+", required=True, help="parquets OOF dos membros (id/t/y/oof_pred)")
    p.add_argument("--names", nargs="*", default=None)
    p.add_argument("--step", type=float, default=0.1, help="passo do grid do simplex de pesos")
    p.add_argument("--out", default="artifacts/models/oof_A2_stack.parquet")
    args = p.parse_args()

    cfg = load_config(args.config)
    names = args.names or [Path(m).stem for m in args.members]

    base = pd.read_parquet(args.members[0])[["id", "t", "y"]].copy()
    P = np.zeros((len(base), len(args.members)), dtype=np.float64)
    for j, m in enumerate(args.members):
        d = pd.read_parquet(m)
        if len(d) != len(base) or not d["id"].equals(base["id"]) or not d["t"].equals(base["t"]):
            raise SystemExit(f"{m}: linhas não casam com {args.members[0]}")
        P[:, j] = d.groupby("t")["oof_pred"].rank(method="average", pct=True).to_numpy()

    t = base["t"].to_numpy(); y = base["y"].to_numpy()
    bucket = np.array(LABELS, dtype=object)[pd.cut(t, EDGES, labels=False, right=True).astype(int)]
    fold = np.full(len(base), -1)
    for k, (_, valid) in enumerate(grouped_stratified_kfold(base, cfg.lightgbm.n_folds, cfg.seed)):
        fold[valid] = k
    grid = _simplex(len(args.members), args.step)

    stacked = np.zeros(len(base), dtype=np.float64)
    chosen = {lab: [] for lab in LABELS}
    for k in range(cfg.lightgbm.n_folds):
        for lab in LABELS:
            tr = (fold != k) & (bucket == lab)
            te = (fold == k) & (bucket == lab)
            if not te.any() or not tr.any():
                continue
            best = (None, -9.0)
            for wv in grid:
                sc = P[tr] @ wv
                a = weighted_ts_auc(t[tr], y[tr], sc)
                if a > best[1]:
                    best = (wv, a)
            stacked[te] = P[te] @ best[0]
            chosen[lab].append(best[0])

    print("pesos médios por bucket (nested OOF):")
    for lab in LABELS:
        if chosen[lab]:
            w = np.mean(chosen[lab], axis=0)
            print(f"  {lab:12s} " + " ".join(f"{n}={wi:.2f}" for n, wi in zip(names, w)))

    a0 = weighted_ts_auc(t, y, P[:, 0])  # membro 0 sozinho (percentil) como referência
    a1 = weighted_ts_auc(t, y, stacked)
    print(f"\nTS-AUC membro-0(percentil)={a0:.4f}  stack={a1:.4f}  Δ={a1-a0:+.4f}")

    out = base.copy(); out["oof_pred"] = stacked
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"stack salvo em {args.out} — veredito por compare_oof.py vs incumbente.")


if __name__ == "__main__":
    main()
