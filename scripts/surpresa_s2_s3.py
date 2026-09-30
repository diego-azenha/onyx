#!/usr/bin/env python
"""S2 + S3 da frente de surpresa acumulada (knowledge/frentes/surpresa-acumulada/S2-redundancia.md,
S3-combinacao.md).

S2: correlação de Spearman DENTRO do passo (a única ordenação que a TS-AUC enxerga, invariância C1)
entre o score de surpresa e (a) o OOF do Onyx, (b) as features do Onyx mais próximas.

S3: combinação sem retreino, score = logit(OOF Onyx) + w * surpresa/dp. O peso `w` é escolhido numa
metade das séries (ids pares) e aplicado na outra (ímpares), e vice-versa: nenhuma série é avaliada
com um `w` que a viu. O OOF combinado vai para `compare_oof.py --grid full` contra o OOF do Onyx.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

GRADE_W = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]


def _spearman_xs(df: pd.DataFrame, a: str, b: str, passos: np.ndarray) -> float:
    """Média, sobre os passos em `passos`, da correlação de Spearman entre a e b dentro do passo."""
    vals = []
    for _, g in df[df["t"].isin(passos)].groupby("t"):
        if len(g) < 30:
            continue
        ra, rb = g[a].rank().to_numpy(), g[b].rank().to_numpy()
        if ra.std() > 0 and rb.std() > 0:
            vals.append(np.corrcoef(ra, rb)[0, 1])
    return float(np.mean(vals))


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p) - np.log1p(-p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", default="artifacts/models/oof_b0_vema_bag4.parquet")
    ap.add_argument("--scores", default="artifacts/surpresa/scores.parquet")
    ap.add_argument("--rows", default="data/processed/train_rows.parquet")
    ap.add_argument("--y-train", default="data/y_train.parquet")
    ap.add_argument("--colunas", nargs="+", default=["sr_eta", "sr_cal", "fam_escala_sobe"],
                    help="a PRIMEIRA é a declarada a priori; as outras são exploratórias")
    ap.add_argument("--features-onyx", nargs="*",
                    default=["conformal_logm_abs", "cusum_var_up_r150", "accum_window_var_ln_w100_cal", "bayes_lo_h0100"])
    ap.add_argument("--out-dir", default="artifacts/surpresa")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    oof = pd.read_parquet(args.oof)
    sc = pd.read_parquet(args.scores, columns=["id", "t"] + args.colunas)
    df = oof.merge(sc, on=["id", "t"], how="left", validate="one_to_one")
    assert df[args.colunas].notna().all().all(), "OOF com passos sem score de surpresa"
    import pyarrow.parquet as pq
    existentes = set(pq.read_schema(args.rows).names)
    feats = [f for f in args.features_onyx if f in existentes]
    faltando = sorted(set(args.features_onyx) - set(feats))
    if faltando:
        print("features do Onyx ausentes no dataset (ignoradas):", faltando)
    if feats:
        df = df.merge(pd.read_parquet(args.rows, columns=["id", "t"] + feats), on=["id", "t"], how="left")
    w_mult = board_grid_multipliers(args.y_train, df["t"].to_numpy())
    t, y, ids = df["t"].to_numpy(), df["y"].to_numpy(), df["id"].to_numpy()
    rel: dict = {"oof": args.oof, "n_linhas": int(len(df))}

    # ---------------- S2 ----------------
    passos = np.unique(np.r_[np.arange(10, 101, 10), np.arange(150, 1000, 50)])
    s2 = {}
    for c in tqdm(args.colunas, desc="S2"):
        s2[c] = {"vs_oof": _spearman_xs(df, c, "oof_pred", passos)}
        for f in feats:
            s2[c][f"vs_{f}"] = _spearman_xs(df, c, f, passos)
    rel["S2_spearman_xs"] = s2

    # ---------------- S3 ----------------
    base = _logit(df["oof_pred"].to_numpy(dtype=np.float64))
    rel["base_tsauc"] = weighted_ts_auc(t, y, base, w_mult)
    par = ids % 2 == 0
    s3 = {}
    for c in args.colunas:
        z = df[c].to_numpy(dtype=np.float64)
        z = z / z.std()
        combinado = np.empty_like(base)
        escolhas = {}
        for nome, ajuste, aplica in (("par->impar", par, ~par), ("impar->par", ~par, par)):
            curva = {w: weighted_ts_auc(t[ajuste], y[ajuste], base[ajuste] + w * z[ajuste], w_mult) for w in GRADE_W}
            w_best = max(curva, key=curva.get)
            escolhas[nome] = {"w": w_best, "curva_no_ajuste": curva}
            combinado[aplica] = base[aplica] + w_best * z[aplica]
        cand = df[["id", "t", "y"]].copy()
        cand["oof_pred"] = combinado
        caminho = out_dir / f"oof_b0_mais_{c}.parquet"
        cand.to_parquet(caminho, index=False)
        s3[c] = {"escolhas": escolhas, "tsauc_combinado_crossfit": weighted_ts_auc(t, y, combinado, w_mult),
                 "arquivo": str(caminho)}
    rel["S3"] = s3
    txt = json.dumps(rel, indent=2, default=float)
    (out_dir / "s2_s3.json").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
