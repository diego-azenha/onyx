#!/usr/bin/env python
"""T1 (knowledge/frentes/teto-offline/T1-teto-offline.md): teto offline histórico × trecho.

Uma linha por série. Positivo: trecho a partir de τ (oráculo) ou prefixo inteiro até τ+L
(contaminado). Negativo: o mesmo com um pseudo-τ sorteado da distribuição real de τ. LightGBM em CV de
5 folds -> AUC. Referência: OOF do Onyx (B0) no passo τ+L / p+L, no passo retido mais próximo.
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
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from offline.features import contexto_hist, features  # noqa: E402


def _amostras(series, yi, L, rng, min_ini=0):
    """Positivo: tau real. Negativo: pseudo-tau ~ U[min_ini, len-L], que é a distribuição
    condicional do tau dos positivos dado o comprimento (tau/len é uniforme, premissa 4). A primeira
    versão sorteava da distribuição AGREGADA de tau, que favorece tau pequeno, e isso vazava
    comprimento e t entre as classes (contaminado > oráculo, impossível sem vazamento)."""
    out = []
    for sid, h, on in series:
        tau = int(yi.loc[sid, "tau_index"])
        if tau >= 0:
            if len(on) - tau < L or tau < min_ini:
                continue
            ini, y = tau, 1
        else:
            if len(on) - L < min_ini:
                continue
            ini, y = int(rng.integers(min_ini, len(on) - L + 1)), 0
        out.append((sid, y, ini))
    return out


def _pre(h, on, ini):
    ctx = contexto_hist(h)
    return features(ctx, on[:ini], h[-5:])


def _uma(h, on, ini, L):
    ctx = contexto_hist(h)
    antes = np.concatenate([h, on[:ini]])[-5:]
    f_or = features(ctx, on[ini:ini + L], antes)
    f_ct = features(ctx, on[:ini + L], h[-5:])
    return f_or, f_ct


def _cv_auc(X: pd.DataFrame, y: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray]:
    oof = np.zeros(len(y))
    params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=40,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, seed=seed, num_threads=4, deterministic=True)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=400)
        oof[te] = m.predict(X.iloc[te])
    return roc_auc_score(y, oof), oof


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--Ls", nargs="+", type=int, default=[100, 200, 400])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--oof", default="artifacts/models/oof_b0_vema_bag4.parquet")
    ap.add_argument("--out-dir", default="artifacts/offline")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    series = _carregar_series(Path("data"))
    por_id = {sid: (h, on) for sid, h, on in series}
    yi = pd.read_parquet("data/y_train_index.parquet")
    oof = pd.read_parquet(args.oof)
    oof_t = {sid: g for sid, g in oof.groupby("id")[["t", "oof_pred"]]}
    res = {}
    for L in args.Ls:
        rng = np.random.default_rng(1000 + L)
        am = _amostras(series, yi, L, rng)
        feats = Parallel(n_jobs=args.n_jobs, batch_size=64)(
            delayed(_uma)(por_id[sid][0], por_id[sid][1], ini, L) for sid, _, ini in am)
        y = np.array([a[1] for a in am])
        Xo = pd.DataFrame([f[0] for f in feats]); Xc = pd.DataFrame([f[1] for f in feats])
        Xo.assign(id=[a[0] for a in am], y=y).to_parquet(out_dir / f"t1_oraculo_L{L}.parquet")
        Xc.assign(id=[a[0] for a in am], y=y).to_parquet(out_dir / f"t1_contaminado_L{L}.parquet")
        auc_or, _ = _cv_auc(Xo, y)
        auc_ct, _ = _cv_auc(Xc, y)
        # referência Onyx: score no passo ini+L (1-based t = ini+L), passo retido mais próximo <= alvo
        s_on = []
        for sid, _, ini in am:
            g = oof_t[sid]
            tt = g["t"].to_numpy()
            k = np.searchsorted(tt, ini + L, side="right") - 1
            s_on.append(g["oof_pred"].to_numpy()[max(k, 0)])
        auc_onyx = roc_auc_score(y, s_on)
        res[L] = {"n": int(len(y)), "taxa_pos": float(y.mean()), "auc_oraculo": auc_or,
                  "auc_contaminado": auc_ct, "auc_onyx_mesmo_ponto": auc_onyx}
        print(L, json.dumps(res[L]), flush=True)
    # O trecho online ANTES da quebra já difere do histórico? (positivos com tau>=50; negativos com
    # pseudo-tau na mesma distribuição). Se AUC >> 0,5, há informação causal pré-quebra.
    rng = np.random.default_rng(7)
    am = _amostras(series, yi, 1, rng, min_ini=50)
    feats = Parallel(n_jobs=args.n_jobs, batch_size=64)(
        delayed(_pre)(por_id[sid][0], por_id[sid][1], ini) for sid, _, ini in am)
    y = np.array([a[1] for a in am]); Xp = pd.DataFrame(feats)
    auc_pre, _ = _cv_auc(Xp.drop(columns=["n"]), y)
    res["pre_quebra"] = {"n": int(len(y)), "auc": auc_pre}
    print("pre", json.dumps(res["pre_quebra"]), flush=True)
    (out_dir / "t1.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
