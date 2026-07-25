#!/usr/bin/env python
"""B2 + A5 (CAMPANHA_POLIMENTO.md) — dois itens que se respondem sem treinar nada.

B2 — espaco de media do bag. Hoje `scripts/avg_oof.py` faz media de PROBABILIDADES, mas a producao
faz media de LOGITS: `adapter/platform.py` funde os K boosters com `model/fuse.py`, e o fundido
devolve `sigmoid(media dos raws)`. Sao funcoes diferentes e podem ordenar duas series de forma
diferente -- ou seja, o OOF que decide os bracos nao e o objeto que e submetido. Aqui as duas
agregacoes sao medidas na grade do BOARD (A2), lado a lado, e tambem o rank-average intra-t.

A5 — os primeiros passos. Quanto do vetor de features esta em NaN de warmup em t pequeno, e o que a
TS-AUC faz nesse regime.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

M = "artifacts/models"
SEEDS = ["777", "101", "202", "303"]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return np.log(p / (1.0 - p))


def b2(name: str) -> None:
    paths = [f"{M}/oof_{name}_s{s}.parquet" for s in SEEDS]
    paths = [p for p in paths if Path(p).exists()]
    if len(paths) < 2:
        print(f"[B2] {name}: menos de 2 sementes, pulado")
        return
    base = pd.read_parquet(paths[0])[["id", "t", "y"]]
    P = np.column_stack([pd.read_parquet(p)["oof_pred"].to_numpy(np.float64) for p in paths])
    t, y = base["t"].to_numpy(), base["y"].to_numpy()
    wm = board_grid_multipliers("data/y_train.parquet", t)

    agg = {
        "media de PROBABILIDADE (o que avg_oof.py faz)": P.mean(axis=1),
        "media de LOGIT (o que a producao fundida faz)": _logit(P).mean(axis=1),
        "rank-average intra-t": None,
    }
    df = pd.DataFrame({"t": t})
    ranks = np.column_stack([
        df.assign(s=P[:, j]).groupby("t")["s"].rank(pct=True).to_numpy() for j in range(P.shape[1])
    ])
    agg["rank-average intra-t"] = ranks.mean(axis=1)

    print(f"\n=== B2 — {name}, K={len(paths)} sementes (grade do BOARD) ===")
    ref = None
    for label, s in agg.items():
        v = weighted_ts_auc(t, y, s, wm)
        if ref is None:
            ref = v
        print(f"  {label:48s} {v:.5f}   ({v - ref:+.5f})")
    sp = pd.Series(agg["media de PROBABILIDADE (o que avg_oof.py faz)"]).corr(
        pd.Series(agg["media de LOGIT (o que a producao fundida faz)"]), method="spearman")
    print(f"  Spearman(prob, logit) = {sp:.6f}")


def a5(rows_path: str, oof_path: str) -> None:
    f = pq.ParquetFile(rows_path)
    feats = [c for c in f.schema.names if c not in ("id", "t", "y", "thin_weight", "__index_level_0__")]
    # so os passos frios: ler o parquet inteiro so para isso seria desperdicio, mas as colunas sao
    # necessarias todas -- lemos e filtramos, uma vez.
    df = f.read(columns=["t"] + feats).to_pandas()
    print(f"\n=== A5 — warmup: fracao de features em NaN por passo ({len(feats)} colunas) ===")
    small = df[df["t"] <= 20]
    nanfrac = small.groupby("t")[feats].apply(lambda g: g.isna().mean(axis=1).mean())
    for tt in range(1, 21):
        if tt in nanfrac.index:
            bar = "#" * int(round(nanfrac.loc[tt] * 50))
            print(f"  t={tt:2d}  NaN {nanfrac.loc[tt] * 100:5.1f}%  {bar}")
    for tt in [25, 50, 100, 250]:
        sub = df[df["t"] == tt]
        if len(sub):
            print(f"  t={tt:3d} NaN {sub[feats].isna().mean(axis=1).mean() * 100:5.1f}%")

    # colunas que continuam 100% NaN mesmo em t=50: candidatas a diluir feature_fraction de graca
    at50 = df[df["t"] == 50]
    if len(at50):
        dead = [c for c in feats if at50[c].isna().all()]
        print(f"\n  colunas 100% NaN ainda em t=50: {len(dead)}")
        if dead:
            print("   ", sorted(dead)[:20])

    oof = pd.read_parquet(oof_path)
    print("\n=== A5 — TS-AUC por passo, regime frio ===")
    a = oof[oof["t"] <= 20]
    for tt in sorted(a["t"].unique()):
        g = a[a["t"] == tt]
        npos, nneg = int(g["y"].sum()), int((1 - g["y"]).sum())
        if npos and nneg:
            print(f"  t={tt:2d}  n_pos={npos:5d}  AUC_t={weighted_ts_auc(g['t'].to_numpy(), g['y'].to_numpy(), g['oof_pred'].to_numpy()):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", default="b6c6_joint")
    ap.add_argument("--rows", default="data/processed/train_rows_bocpd.parquet")
    ap.add_argument("--oof", default=f"{M}/oof_b6c6_joint_bag4.parquet")
    args = ap.parse_args()
    b2(args.name)
    a5(args.rows, args.oof)


if __name__ == "__main__":
    main()
