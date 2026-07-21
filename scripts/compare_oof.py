#!/usr/bin/env python
"""R0 (docs/PARECER_AUDITORIA_ONYX.md §6): comparador OOF pareado por série, com intervalo de
confiança por bootstrap pareado. Recebe dois parquets de OOF (mesmas séries, colunas id/t/y/score) e
reporta Delta-TS-AUC (candidato - baseline) geral e por bucket de t, com IC 95% por bootstrap
(reamostragem de `id`s com reposição, pareada -- a MESMA amostra de ids em cada réplica é usada para
avaliar baseline e candidato, o que cancela a maior parte da variância comum entre eles).

Isto NÃO substitui a submissão oficial como âncora de leaderboard (docs/PLANO_TECNICO.md §9.0) --
é o instrumento *relativo* que decide, com barra de erro, se uma mudança local vale a pena sondar
oficialmente (parecer §5-D4, §6-R0). Regra de decisão sugerida: adotar a mudança se o IC do Delta
exclui 0 no agregado OU no bucket-alvo declarado a priori (--target-bucket); nunca decidir por um
Delta pontual sem IC.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sbrt.evaluation.ts_auc import weighted_ts_auc

T_BUCKET_EDGES = [0, 50, 150, 400, np.inf]
T_BUCKET_LABELS = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]


def _bucket_of(t: np.ndarray) -> np.ndarray:
    idx = pd.cut(t, bins=T_BUCKET_EDGES, labels=False, right=True)
    return np.array(T_BUCKET_LABELS, dtype=object)[idx.astype(int)]


def _point_estimate(t, y, s_base, s_cand, bucket) -> dict:
    out = {"overall": weighted_ts_auc(t, y, s_cand) - weighted_ts_auc(t, y, s_base)}
    for label in T_BUCKET_LABELS:
        mask = bucket == label
        out[label] = (
            weighted_ts_auc(t[mask], y[mask], s_cand[mask]) - weighted_ts_auc(t[mask], y[mask], s_base[mask])
            if mask.any()
            else float("nan")
        )
    return out


def _one_bootstrap_rep(
    seed: int, id_to_pos: dict, sampled_id_order: np.ndarray, t, y, s_base, s_cand, bucket
) -> dict:
    rng = np.random.default_rng(seed)
    sampled_ids = rng.choice(sampled_id_order, size=len(sampled_id_order), replace=True)
    idx = np.concatenate([id_to_pos[i] for i in sampled_ids])
    return _point_estimate(t[idx], y[idx], s_base[idx], s_cand[idx], bucket[idx])


def paired_bootstrap_compare(
    merged: pd.DataFrame,
    score_col_base: str,
    score_col_cand: str,
    n_boot: int = 500,
    seed: int = 42,
    alpha: float = 0.05,
    n_jobs: int = -1,
) -> dict:
    t = merged["t"].to_numpy(dtype=np.int64)
    y = merged["y"].to_numpy(dtype=np.int64)
    s_base = merged[score_col_base].to_numpy(dtype=np.float64)
    s_cand = merged[score_col_cand].to_numpy(dtype=np.float64)
    bucket = _bucket_of(t)

    point = _point_estimate(t, y, s_base, s_cand, bucket)

    id_to_pos = merged.groupby("id").indices
    unique_ids = np.array(sorted(id_to_pos.keys()))

    rng_master = np.random.default_rng(seed)
    rep_seeds = rng_master.integers(0, 2**31 - 1, size=n_boot)

    reps = Parallel(n_jobs=n_jobs)(
        delayed(_one_bootstrap_rep)(int(s), id_to_pos, unique_ids, t, y, s_base, s_cand, bucket)
        for s in rep_seeds
    )

    result = {}
    for key in ["overall"] + T_BUCKET_LABELS:
        vals = np.array([r[key] for r in reps], dtype=np.float64)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            result[key] = {"point_delta": point[key], "ci_lo": float("nan"), "ci_hi": float("nan"),
                           "boot_mean": float("nan"), "boot_std": float("nan"), "n_boot_valid": 0}
            continue
        lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
        result[key] = {
            "point_delta": point[key],
            "ci_lo": float(lo),
            "ci_hi": float(hi),
            "boot_mean": float(vals.mean()),
            "boot_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n_boot_valid": int(len(vals)),
            "excludes_zero": bool(lo > 0 or hi < 0),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", required=True, help="parquet OOF do modelo baseline (id,t,y,score)")
    parser.add_argument("--candidate", required=True, help="parquet OOF do modelo candidato (id,t,y,score)")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col-baseline", default="oof_pred")
    parser.add_argument("--score-col-candidate", default="oof_pred")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--target-bucket", default=None, choices=T_BUCKET_LABELS,
                         help="bucket declarado a priori para a regra de decisão (opcional)")
    parser.add_argument("--out", default=None, help="caminho opcional para salvar o resultado em JSON")
    args = parser.parse_args()

    base_df = pd.read_parquet(args.baseline)
    cand_df = pd.read_parquet(args.candidate)

    base_df = base_df.rename(columns={args.label_col: "y", args.score_col_baseline: "score_base"})
    cand_df = cand_df.rename(columns={args.label_col: "y", args.score_col_candidate: "score_cand"})

    merged = base_df[["id", "t", "y", "score_base"]].merge(
        cand_df[["id", "t", "y", "score_cand"]], on=["id", "t"], how="inner", suffixes=("", "_cand")
    )
    mismatch = int((merged["y"] != merged["y_cand"]).sum())
    if mismatch:
        raise ValueError(
            f"{mismatch} linhas com rotulo y divergente entre baseline e candidato -- os OOFs nao "
            "sao do mesmo dataset/split; comparacao pareada invalida."
        )
    n_dropped_base = len(base_df) - len(merged)
    n_dropped_cand = len(cand_df) - len(merged)
    if n_dropped_base or n_dropped_cand:
        print(
            f"[compare_oof] aviso: {n_dropped_base} linhas do baseline e {n_dropped_cand} do "
            "candidato nao casaram em (id,t) e foram descartadas do par."
        )

    result = paired_bootstrap_compare(
        merged, "score_base", "score_cand", n_boot=args.n_boot, seed=args.seed,
        alpha=args.alpha, n_jobs=args.n_jobs,
    )

    print(f"\nDelta-TS-AUC (candidato - baseline), {len(merged)} linhas pareadas, "
          f"{merged['id'].nunique()} series, n_boot={args.n_boot}, IC {100 * (1 - args.alpha):.0f}%\n")
    header = f"{'bucket':14s} {'delta':>9s} {'ci_lo':>9s} {'ci_hi':>9s} {'exclui 0':>9s}"
    print(header)
    print("-" * len(header))
    for key in ["overall"] + T_BUCKET_LABELS:
        r = result[key]
        flag = "sim" if r.get("excludes_zero") else "nao"
        print(f"{key:14s} {r['point_delta']:9.4f} {r['ci_lo']:9.4f} {r['ci_hi']:9.4f} {flag:>9s}")

    decide_key = args.target_bucket if args.target_bucket else "overall"
    adopt = result[decide_key].get("excludes_zero", False) and result[decide_key]["point_delta"] > 0
    print(f"\nregra de decisao (bucket-alvo={decide_key}): "
          f"{'ADOTAR' if adopt else 'NAO adotar (ou inconclusivo)'} a mudanca.")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nresultado salvo em {out_path}")


if __name__ == "__main__":
    main()
