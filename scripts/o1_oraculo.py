#!/usr/bin/env python
"""O1 (knowledge/frentes/oraculo-destilado/README.md): o ORÁCULO que vê a série inteira.

Para cada linha (série i, passo t) do train_rows, features do Onyx (top-K por ganho) em t e em
snapshots FUTUROS da mesma série (t+50, t+200, fim da série), mais log t, log T e log(T-t+1).
Alvo: y_t = 1{tau <= t}. Um GBM estima P(tau <= t | série inteira).

Modos:
- `--sonda`: treina nos folds != 0 e mede a TS-AUC do oráculo no fold 0, contra o E1.
- `--alvos`: NESTED cross-fit com os folds do Onyx. Para cada fold f do aluno, oráculo ajustado nas
  séries fora de f, com CV interna de 4 partes para dar a previsão dessas séries. Grava
  `y_soft_f0..f4` (probabilidade do oráculo) no parquet do aluno, que usa `soft_label` em `train.py`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc

DELTAS = (50, 200)


def montar(rows: pd.DataFrame, cols: list[str], cols_agora: list[str] | None = None) -> pd.DataFrame:
    """Features do oráculo, alinhadas a `rows` (ordenado por id, t). `cols` entram em t e nos snapshots
    futuros; `cols_agora` (default = cols) entram só em t. Para a Rao-Blackwellização valer, a
    informação do oráculo em t tem de CONTER a do aluno: `cols_agora` = todas as features do aluno."""
    ids = rows["id"].to_numpy(); t = rows["t"].to_numpy()
    F = rows[cols].to_numpy(dtype=np.float32)
    F_agora = rows[cols_agora].to_numpy(dtype=np.float32) if cols_agora is not None else F
    nomes_agora = cols_agora if cols_agora is not None else cols
    cortes = np.flatnonzero(np.diff(ids)) + 1
    ini = np.r_[0, cortes]; fim = np.r_[cortes, len(ids)]
    blocos = {d: np.empty(len(ids), dtype=np.int64) for d in DELTAS}
    ultimo = np.empty(len(ids), dtype=np.int64)
    T_len = np.empty(len(ids), dtype=np.float32)
    for a, b in zip(ini, fim):
        tt = t[a:b]
        for d in DELTAS:
            k = np.searchsorted(tt, tt + d, side="left")
            blocos[d][a:b] = a + np.minimum(k, b - a - 1)
        ultimo[a:b] = b - 1
        T_len[a:b] = tt[-1]
    partes = [pd.DataFrame(F_agora, columns=[f"o_{c}" for c in nomes_agora])]
    for d in DELTAS:
        partes.append(pd.DataFrame(F[blocos[d]], columns=[f"o{d}_{c}" for c in cols]))
    partes.append(pd.DataFrame(F[ultimo], columns=[f"oT_{c}" for c in cols]))
    X = pd.concat(partes, axis=1)
    X["log_t"] = np.log(t.astype(np.float32)); X["log_T"] = np.log(T_len)
    X["log_resto"] = np.log(T_len - t + 1)
    return X


PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.5,
              bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, extra_trees=True, verbose=-1, seed=0,
              num_threads=6, deterministic=True, force_row_wise=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", default="data/processed/train_rows_p1_top65.parquet")
    ap.add_argument("--full-rows", default="data/processed/train_rows.parquet")
    ap.add_argument("--sonda", action="store_true")
    ap.add_argument("--alvos", action="store_true")
    ap.add_argument("--fold-seed", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--sub", type=int, default=2, help="usa 1 a cada `sub` linhas no treino do oráculo")
    ap.add_argument("--todas-em-t", action="store_true",
                    help="em t, o oráculo vê TODAS as features do aluno (lidas de --full-rows); snapshots futuros só das top-K")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    seed = args.fold_seed if args.fold_seed is not None else cfg.seed
    art = Path("artifacts/oraculo"); art.mkdir(parents=True, exist_ok=True)

    rows = pd.read_parquet(args.rows).sort_values(["id", "t"]).reset_index(drop=True)
    cols = [c for c in rows.columns if c not in ("id", "t", "y", "thin_weight")]
    if args.todas_em_t:
        full0 = pd.read_parquet(args.full_rows).sort_values(["id", "t"]).reset_index(drop=True)
        assert (full0[["id", "t"]].to_numpy() == rows[["id", "t"]].to_numpy()).all()
        agora = [c for c in full0.columns if c not in ("id", "t", "y", "thin_weight") and not c.startswith("y_soft")]
        for c in agora:
            if c not in rows.columns:
                rows[c] = full0[c].to_numpy()
        del full0
        X = montar(rows, cols, agora)
    else:
        X = montar(rows, cols)
    print("features do oráculo:", X.shape, flush=True)
    y = rows["y"].to_numpy()
    folds = list(grouped_stratified_kfold(rows[["id", "t", "y"]], cfg.lightgbm.n_folds, seed))
    fold_de = np.empty(len(rows), dtype=np.int64)
    for f, (_, va) in enumerate(folds):
        fold_de[va] = f
    w = board_grid_multipliers("data/y_train.parquet", rows["t"].to_numpy())
    passo = np.arange(len(rows)) % args.sub == 0

    if args.sonda:
        tr = (fold_de != 0) & passo; va = fold_de == 0
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), args.rounds)
        p = m.predict(X[va])
        e1 = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet")
        v = rows.loc[va, ["id", "t", "y"]].assign(oraculo=p).merge(e1, on=["id", "t", "y"])
        rel = {"tsauc_oraculo_fold0": weighted_ts_auc(v.t.values, v.y.values, v.oraculo.values, w),
               "tsauc_e1_fold0": weighted_ts_auc(v.t.values, v.y.values, v.oof_pred.values, w)}
        print(json.dumps(rel, indent=2))
        (art / f"sonda{args.tag}.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
        imp = pd.Series(m.feature_importance("gain"), index=X.columns).sort_values(ascending=False)
        print(imp.head(15))
        return

    if args.alvos:
        ids = rows["id"].to_numpy()
        soft = {}
        rel = {"seed": seed, "tsauc_oraculo_interno": []}
        for f in range(len(folds)):
            fora = fold_de != f
            ids_f = np.unique(ids[fora])
            perm = np.random.default_rng(500 + f).permutation(ids_f)
            partes = np.array_split(perm, 4)
            nota = np.zeros(len(rows), dtype=np.float32)
            for parte in partes:
                te = np.isin(ids, parte)
                trm = fora & ~te & passo
                m = lgb.train({**PARAMS, "seed": f}, lgb.Dataset(X[trm], y[trm]), args.rounds)
                nota[te] = m.predict(X[te])
            rel["tsauc_oraculo_interno"].append(
                weighted_ts_auc(rows.t.values[fora], y[fora], nota[fora], w))
            soft[f"y_soft_f{f}"] = nota
            print(f"fold {f}: TS-AUC do oráculo (interno, fora de f) {rel['tsauc_oraculo_interno'][-1]:.4f}", flush=True)
        full = pd.read_parquet(args.full_rows).sort_values(["id", "t"]).reset_index(drop=True)
        assert (full[["id", "t"]].to_numpy() == rows[["id", "t"]].to_numpy()).all()
        for k, v in soft.items():
            full[k] = v
        out = Path(f"data/processed/train_rows_oraculo{args.tag}_s{seed}.parquet")
        full.to_parquet(out)
        (art / f"alvos{args.tag}_s{seed}.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
        print(json.dumps(rel, indent=2), "\ngravado", out)


if __name__ == "__main__":
    main()
