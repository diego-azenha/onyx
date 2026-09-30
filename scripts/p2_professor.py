#!/usr/bin/env python
"""Passo 2 do roteiro de 30/09 (knowledge/frentes/roteiro-30-09/P2-professor-aluno.md): o PROFESSOR.

Para cada série e cada m numa grade, features privilegiadas (o trecho exato [tau, tau+m) e o trecho
pós-quebra inteiro, contra o histórico). Um GBM separa positivos de negativos (pseudo-tau). A nota é
a chance de a quebra já ser perceptível em m.

Nested cross-fit com os folds do Onyx: para cada fold f do aluno, professor ajustado só nas séries
fora de f, com CV interna de 4 partes para dar a nota dessas séries. Grava `train_rows` + y_soft_f0..f4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from offline.features import contexto_hist, features  # noqa: E402
from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.evaluation.splits import grouped_stratified_kfold  # noqa: E402

GRADE_M = (2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800)


def _linhas_serie(sid: int, y: int, ini: int, h: np.ndarray, on: np.ndarray) -> list[dict]:
    ctx = contexto_hist(h)
    antes = np.concatenate([h, on[:ini]])[-5:]
    pos = on[ini:]
    fut = {f"f_{k}": v for k, v in features(ctx, pos, antes).items()} if len(pos) >= 2 else {}
    out = []
    for m in GRADE_M:
        if m > len(pos):
            break
        r = {f"a_{k}": v for k, v in features(ctx, pos[:m], antes).items()}
        r.update(fut)
        r.update(id=sid, y=y, m=m, log_m=np.log(m))
        out.append(r)
    return out


def _gbm(X, y, seed=0):
    p = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, extra_trees=True, verbose=-1, seed=seed,
             num_threads=4, deterministic=True)
    return lgb.train(p, lgb.Dataset(X, y), 300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold-seed", type=int, default=None)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--rows", default="data/processed/train_rows.parquet")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    seed = args.fold_seed if args.fold_seed is not None else cfg.seed
    out_rows = Path(args.out or f"data/processed/train_rows_prof_s{seed}.parquet")
    art = Path("artifacts/roteiro/p2"); art.mkdir(parents=True, exist_ok=True)

    yi = pd.read_parquet("data/y_train_index.parquet")
    rng = np.random.default_rng(2027)
    tarefas = []
    for sid, h, on in _carregar_series(Path("data")):
        tau = int(yi.loc[sid, "tau_index"])
        if tau >= 0:
            tarefas.append((sid, 1, tau, h, on))
        elif len(on) >= 3:
            tarefas.append((sid, 0, int(rng.integers(0, len(on) - 1)), h, on))
    cache = art / "linhas_professor.parquet"
    if cache.exists():
        G = pd.read_parquet(cache)
    else:
        res = Parallel(n_jobs=args.n_jobs, batch_size=16)(delayed(_linhas_serie)(*t) for t in tarefas)
        G = pd.DataFrame([r for rr in res for r in rr])
        G.to_parquet(cache)
    feats = [c for c in G.columns if c not in ("id", "y", "m")]
    print("linhas do professor:", G.shape, "positivas", int(G.y.sum()), flush=True)

    meta = pd.read_parquet(args.rows, columns=["id", "t", "y"])
    folds = list(grouped_stratified_kfold(meta, cfg.lightgbm.n_folds, seed))
    fold_ids = [np.unique(meta["id"].to_numpy()[va]) for _, va in folds]
    ids_series = G["id"].to_numpy()
    notas = {}                                   # f -> Series(index=(id,m)) nota do professor
    rel = {"seed": seed, "auc_professor_externo": [], "auc_professor_interno": []}
    for f, va_ids in enumerate(fold_ids):
        fora = ~np.isin(ids_series, va_ids)
        Gf = G[fora]
        ids_f = np.unique(Gf["id"])
        perm = np.random.default_rng(100 + f).permutation(ids_f)
        partes = np.array_split(perm, 4)
        nota_f = pd.Series(np.nan, index=Gf.index)
        for parte in partes:
            te = Gf["id"].isin(parte).to_numpy()
            m = _gbm(Gf.loc[~te, feats], Gf.loc[~te, "y"], seed=f)
            nota_f[te] = m.predict(Gf.loc[te, feats])
        rel["auc_professor_interno"].append(float(roc_auc_score(Gf["y"], nota_f)))
        # professor "externo" (todas as séries fora de f) avaliado nas séries de f: diagnóstico apenas
        m_ext = _gbm(Gf[feats], Gf["y"], seed=f)
        Gv = G[~fora]
        rel["auc_professor_externo"].append(float(roc_auc_score(Gv["y"], m_ext.predict(Gv[feats]))))
        notas[f] = pd.DataFrame({"id": Gf["id"].to_numpy(), "m": Gf["m"].to_numpy(), "nota": nota_f.to_numpy()})
        print(f"fold {f}: AUC interno {rel['auc_professor_interno'][-1]:.4f} externo {rel['auc_professor_externo'][-1]:.4f}", flush=True)

    rows = pd.read_parquet(args.rows)
    tau = rows["id"].map(yi["tau_index"]).to_numpy()
    m_row = rows["t"].to_numpy() - tau                       # pontos pós-quebra observados (>=1 se y=1)
    pos = rows["y"].to_numpy() == 1
    logm_row = np.log(np.clip(m_row, 2, None).astype(float))
    for f in range(len(folds)):
        tab = notas[f]
        col = np.zeros(len(rows), dtype=np.float32)
        por_id = {i: g.sort_values("m") for i, g in tab.groupby("id")}
        idx_pos = np.flatnonzero(pos)
        ids_pos = rows["id"].to_numpy()[idx_pos]
        for i in np.unique(ids_pos):
            g = por_id.get(int(i))
            if g is None:
                continue                                      # série do fold f (validação): fica 0, não é usada
            sel = idx_pos[ids_pos == i]
            col[sel] = np.interp(logm_row[sel], np.log(g["m"].to_numpy(float)), g["nota"].to_numpy(float))
        rows[f"y_soft_f{f}"] = col
        rel[f"media_y_soft_pos_f{f}"] = float(col[pos & (col > 0)].mean())
    rows.to_parquet(out_rows)
    (art / f"professor_s{seed}.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
