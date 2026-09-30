#!/usr/bin/env python
"""W1 (knowledge/frentes/teto-offline/W1-modelo-janelas.md): segundo modelo, de construção diferente, para ensemble.

Features de duas amostras (src/offline/features.py) calculadas CAUSALMENTE sobre janelas do online
(prefixo inteiro, últimas 50 e últimas 200 observações) contra o histórico, numa grade grossa de t, e
mantidas (forward-fill) nas linhas do train_rows até o próximo recálculo, como faria o tempo real. GBM
com os folds do Onyx (partição 42) → OOF nas linhas do train_rows → TS-AUC sozinho e blend
cross-fit com o E1 (peso ajustado em ids pares e aplicado nos ímpares, e vice-versa).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from offline.features import contexto_hist, features  # noqa: E402
from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.evaluation.splits import grouped_stratified_kfold  # noqa: E402
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc  # noqa: E402

GRADE_T = (10, 20, 30, 45, 65, 90, 120, 160, 210, 280, 360, 460, 580, 720, 880, 999)
JANELAS = (None, 50, 200)          # None = prefixo inteiro


def _serie(sid, h, on):
    ctx = contexto_hist(h)
    out = []
    for t in GRADE_T:
        if t > len(on):
            break
        r = {"id": sid, "t_calc": t}
        for W in JANELAS:
            ini = 0 if W is None else max(0, t - W)
            nome = "p" if W is None else f"w{W}"
            antes = np.concatenate([h, on[:ini]])[-5:]
            f = features(ctx, on[ini:t], antes)
            r.update({f"{nome}_{k}": v for k, v in f.items() if not k.startswith("h_")})
        r.update({k: v for k, v in ctx.resumo.items()})
        out.append(r)
    return out


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    art = Path("artifacts/w1"); art.mkdir(parents=True, exist_ok=True)
    cache = art / "feats_grade.parquet"
    if cache.exists():
        G = pd.read_parquet(cache)
    else:
        res = Parallel(n_jobs=4, batch_size=8)(delayed(_serie)(*s) for s in _carregar_series(Path("data")))
        G = pd.DataFrame([r for rr in res for r in rr]); G.to_parquet(cache)
    fc = [c for c in G.columns if c not in ("id", "t_calc")]
    G[fc] = G[fc].astype(np.float32)
    print("grade:", G.shape, flush=True)
    rows = pd.read_parquet("data/processed/train_rows.parquet", columns=["id", "t", "y"]).sort_values(["id", "t"])
    rows["t"] = rows["t"].astype(np.int64); rows["id"] = rows["id"].astype(np.int64)
    G["t_calc"] = G["t_calc"].astype(np.int64); G["id"] = G["id"].astype(np.int64)
    # forward-fill: cada linha (id, t) recebe o último recálculo com t_calc <= t (linhas com t < 10 ficam NaN)
    G = G.sort_values(["id", "t_calc"])
    M = pd.merge_asof(rows.sort_values("t"), G.rename(columns={"t_calc": "t"}).sort_values("t"),
                      on="t", by="id", direction="backward").sort_values(["id", "t"]).reset_index(drop=True)
    fcols = [c for c in M.columns if c not in ("id", "t", "y")]
    M["log_t"] = np.log(M["t"].astype(float)); fcols.append("log_t")
    y = M["y"].to_numpy()
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.6,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, extra_trees=True, verbose=-1, seed=0,
                  num_threads=6, deterministic=True, force_row_wise=True)
    g = M.groupby("t")["y"]
    npos = M["t"].map(g.sum()); nneg = M["t"].map(g.size() - g.sum())
    w = np.where(y == 1, nneg, npos).astype(float); w /= w.mean()
    oof = np.full(len(M), np.nan)
    sub = np.arange(len(M)) % 2 == 0
    for tr, va in grouped_stratified_kfold(M[["id", "t", "y"]], cfg.lightgbm.n_folds, cfg.seed):
        trs = tr[sub[tr]]
        m = lgb.train(params, lgb.Dataset(M.loc[trs, fcols], y[trs], weight=w[trs]), 500)
        oof[va] = m.predict(M.loc[va, fcols])
    M["w1"] = oof
    e1 = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet")
    D = M[["id", "t", "y", "w1"]].merge(e1, on=["id", "t", "y"])
    wm = board_grid_multipliers("data/y_train.parquet", D["t"].to_numpy())
    t, yy = D["t"].to_numpy(), D["y"].to_numpy()
    lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
    a, b = lg(D["oof_pred"].to_numpy()), lg(D["w1"].to_numpy())
    rel = {"tsauc_w1": weighted_ts_auc(t, yy, b, wm), "tsauc_e1": weighted_ts_auc(t, yy, a, wm)}
    par = (D["id"] % 2 == 0).to_numpy(); comb = np.empty_like(a); esc = {}
    for nome, aj, ap in (("par", par, ~par), ("impar", ~par, par)):
        curva = {wt: weighted_ts_auc(t[aj], yy[aj], (1 - wt) * a[aj] + wt * b[aj], wm) for wt in (0, .1, .2, .3, .4, .5)}
        wt = max(curva, key=curva.get); esc[nome] = wt; comb[ap] = (1 - wt) * a[ap] + wt * b[ap]
    rel["peso_escolhido"] = esc; rel["tsauc_blend"] = weighted_ts_auc(t, yy, comb, wm)
    D.assign(oof_pred=comb)[["id", "t", "y", "oof_pred"]].to_parquet(art / "oof_blend_e1_w1.parquet", index=False)
    (art / "w1.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
