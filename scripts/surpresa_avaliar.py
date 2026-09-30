#!/usr/bin/env python
"""S1 (knowledge/frentes/surpresa-acumulada/S1-detector-sozinho.md): pontua as 10.000 séries de treino
com o detector de surpresa acumulada e mede a TS-AUC de cada saída SOZINHA na grade do board.

O detector não usa rótulo nenhum (é ajustado no histórico de cada série), então medir nas 10k séries
é honesto enquanto `configs/surpresa.yaml` ficar congelada. A grade é a de `y_train.parquet` (todos
os passos online), isto é, a ponderação do board sem multiplicadores. Controle de seleção: metades
par/ímpar de ids. IC do score primário por bootstrap de séries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc
from surpresa import carregar_config
from surpresa.evidencia import Alternativas
from surpresa.stream import pontuar_serie

BUCKETS = [("t<=50", 0, 50), ("50<t<=150", 50, 150), ("150<t<=400", 150, 400), ("t>400", 400, 10**9)]


def _carregar_series(data_dir: Path):
    X = pd.read_parquet(data_dir / "X_train.parquet")
    ids = X.index.get_level_values(0).to_numpy()
    per = X["period"].to_numpy()
    val = X["value"].to_numpy(dtype=np.float64)
    cortes = np.flatnonzero(np.diff(ids)) + 1
    inicios = np.r_[0, cortes]
    fins = np.r_[cortes, len(ids)]
    series = []
    for a, b in zip(inicios, fins):
        p = per[a:b]
        series.append((int(ids[a]), val[a:b][p == 1], val[a:b][p == 2]))
    return series


def _lote(series, cfg):
    alt = Alternativas(cfg)
    out = []
    for sid, h, on in series:
        o = pontuar_serie(h, on, cfg, alt)
        o["id"] = np.full(len(on), sid, dtype=np.int32)
        o["t"] = np.arange(1, len(on) + 1, dtype=np.int32)
        out.append(o)
    return out


def _tsauc_buckets(t, y, s, w_mult=None) -> dict:
    res = {"geral": weighted_ts_auc(t, y, s, w_mult)}
    for nome, lo, hi in BUCKETS:
        m = (t > lo) & (t <= hi)
        res[nome] = weighted_ts_auc(t[m], y[m], s[m], w_mult)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="artifacts/surpresa")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=100)
    ap.add_argument("--primario", default="sr_eta")
    ap.add_argument("--onyx-rows", default=None,
                    help="train_rows.parquet do Onyx: mede conformal_logm_abs sozinho como referência")
    ap.add_argument("--reusar", action="store_true", help="pula a pontuação se scores.parquet existe")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caminho_scores = out_dir / "scores.parquet"

    if args.reusar and caminho_scores.exists():
        df = pd.read_parquet(caminho_scores)
    else:
        cfg = carregar_config()
        series = _carregar_series(Path(args.data_dir))
        lotes = [series[i:i + 50] for i in range(0, len(series), 50)]
        res = Parallel(n_jobs=args.n_jobs)(delayed(_lote)(l, cfg) for l in tqdm(lotes, desc="pontuando lotes de 50"))
        cols = {k: np.concatenate([o[k] for r in res for o in r]) for k in res[0][0]}
        df = pd.DataFrame(cols)
        y = pd.read_parquet(Path(args.data_dir) / "y_train.parquet").reset_index().sort_values(["id", "time"])
        y["t"] = y.groupby("id").cumcount() + 1
        df = df.merge(y[["id", "t", "target"]].rename(columns={"target": "y"}), on=["id", "t"], how="inner",
                      validate="one_to_one")
        assert len(df) == len(y), (len(df), len(y))
        df.to_parquet(caminho_scores)

    t, yy, ids = df["t"].to_numpy(), df["y"].to_numpy(), df["id"].to_numpy()
    scores = [c for c in df.columns if c not in ("id", "t", "y")]
    relatorio: dict = {"n_linhas": int(len(df)), "n_series": int(df["id"].nunique()), "grade": "full (y_train)"}
    relatorio["tsauc"] = {c: _tsauc_buckets(t, yy, df[c].to_numpy()) for c in tqdm(scores, desc="TS-AUC")}

    par = ids % 2 == 0
    relatorio["metades"] = {c: {"par": weighted_ts_auc(t[par], yy[par], df[c].to_numpy()[par]),
                                "impar": weighted_ts_auc(t[~par], yy[~par], df[c].to_numpy()[~par])}
                            for c in scores}

    # bootstrap de séries para o primário
    rng = np.random.default_rng(42)
    uids, inicio, cont = np.unique(ids, return_index=True, return_counts=True)
    assert (np.diff(ids) >= 0).all(), "scores.parquet deveria estar ordenado por id"
    pos = {u: np.arange(a, a + n) for u, a, n in zip(uids, inicio, cont)}
    s = df[args.primario].to_numpy()
    boots = []
    for _ in tqdm(range(args.n_boot), desc="bootstrap"):
        amostra = rng.choice(uids, size=len(uids), replace=True)
        # reamostra séries com reposição; cópias viram séries distintas (novo id) para o rank por passo
        idx = np.concatenate([pos[u] for u in amostra])
        boots.append(weighted_ts_auc(t[idx], yy[idx], s[idx]))
    if boots:
        relatorio["ic95_primario"] = {"score": args.primario,
                                      "lo": float(np.quantile(boots, 0.025)), "hi": float(np.quantile(boots, 0.975))}

    if args.onyx_rows:
        rows = pd.read_parquet(args.onyx_rows, columns=["id", "t", "y", "conformal_logm_abs"])
        w_mult = board_grid_multipliers(str(Path(args.data_dir) / "y_train.parquet"), rows["t"].to_numpy())
        v = rows["conformal_logm_abs"].to_numpy(dtype=np.float64)
        v = np.where(np.isnan(v), -np.inf, v)
        relatorio["referencia_conformal_logm_abs"] = _tsauc_buckets(
            rows["t"].to_numpy(), rows["y"].to_numpy(), v, w_mult)
        # a mesma régua (grade thin + multiplicadores) para o primário, para comparar maçã com maçã
        m = df.merge(rows[["id", "t"]], on=["id", "t"], how="inner")
        relatorio["primario_na_grade_thin"] = _tsauc_buckets(
            m["t"].to_numpy(), m["y"].to_numpy(), m[args.primario].to_numpy(), w_mult)

    txt = json.dumps(relatorio, indent=2, default=float)
    (out_dir / "s1.json").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
