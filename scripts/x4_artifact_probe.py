#!/usr/bin/env python
"""X4 probe de artefato OBRIGATÓRIO (BRAINSTORM_RUPTURA_TSAUC.md §3): antes de qualquer braço, treinar
um discriminador real-vs-injetado SOB H0. AUC>0,6 ⇒ o gerador (ou a construção do portador) vaza
assinatura ⇒ consertar antes de usar os sintéticos, senão o modelo aprende a detectar 'injeção', não
'quebra'.

Aqui H0 = sem quebra. Classe REAL(0) = linhas H0 reais já construídas (train_rows, y==0). Classe
SINTÉTICA(1) = portadores do gerador na família 'none' (cauda do histórico como pseudo-online,
intocada), passados pelo MESMO motor de features. Se o discriminador separa, a diferença está na
CONSTRUÇÃO DO PORTADOR (fit_h0 em histórico mais curto, cauda como online), não na quebra -- e isso
contaminaria todo braço. Reporta AUC por 5-fold."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.dataset import build_training_rows
from sbrt.adversarial.inject import make_record

_NON_FEATURE = {"id", "t", "y", "thin_weight"}
_DROP = ("spec_", "ord_", "mrep_")


def _load_histories(data_dir: Path, min_len: int, n_series: int, seed: int) -> dict:
    X = pd.read_parquet(data_dir / "X_train.parquet")
    out = {}
    for dataset_id, group in X.groupby(level="id"):
        g = group.reset_index()
        h = g.loc[g["period"] == 1, "value"].to_numpy(dtype="float64")
        if len(h) >= min_len:
            out[int(dataset_id)] = h
    ids = sorted(out)
    rng = np.random.default_rng(seed)
    pick = rng.choice(ids, size=min(n_series, len(ids)), replace=False)
    return {int(i): out[int(i)] for i in pick}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--data-dir", default="data")
    p.add_argument("--rows", default="data/processed/train_rows_3eixos.parquet")
    p.add_argument("--n-online", type=int, default=200)
    p.add_argument("--n-carriers", type=int, default=500)
    p.add_argument("--n-real", type=int, default=60000, help="linhas H0 reais amostradas")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = load_config(args.config)

    print(f"[probe] gerando {args.n_carriers} portadores sintéticos H0 (família 'none')...")
    # C2: exige história real longa o bastante para pseudo-história >=1000 (suporte real).
    hist = _load_histories(Path(args.data_dir), args.n_online + 1000, args.n_carriers, args.seed)
    recs = [make_record(i, h, "none", 0.0, args.n_online) for i, h in hist.items()]
    recs = [r for r in recs if r is not None]
    synth = build_training_rows(recs, cfg, progress=True, n_jobs=-1)
    synth = synth[synth["y"] == 0]
    feats = sorted(c for c in synth.columns if c not in _NON_FEATURE and not c.startswith(_DROP))
    print(f"[probe] linhas sintéticas H0: {len(synth)}; features: {len(feats)}")

    print(f"[probe] amostrando {args.n_real} linhas H0 reais de {args.rows}...")
    real = pd.read_parquet(args.rows, columns=feats + ["y"])
    real = real[real["y"] == 0].drop(columns=["y"])
    real = real.sample(n=min(args.n_real, len(real)), random_state=args.seed)

    n = min(len(synth), len(real))
    Xs = synth[feats].sample(n=n, random_state=args.seed).to_numpy(np.float32)
    Xr = real[feats].sample(n=n, random_state=args.seed).to_numpy(np.float32)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.zeros(n), np.ones(n)])
    print(f"[probe] discriminador: {n} reais vs {n} sintéticas, {len(feats)} features")

    aucs = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=args.seed).split(X, y):
        d = lgb.Dataset(X[tr], label=y[tr])
        b = lgb.train({"objective": "binary", "num_leaves": 31, "learning_rate": 0.05,
                       "min_data_in_leaf": 100, "verbose": -1, "seed": 42},
                      d, num_boost_round=200)
        aucs.append(roc_auc_score(y[va], b.predict(X[va])))
    auc = float(np.mean(aucs))
    print(f"\n[probe] AUC real-vs-sintético (5-fold) = {auc:.4f}  "
          f"({'FALHA (>0,6): gerador vaza, consertar' if auc > 0.6 else 'PASSA (<=0,6): H0 indistinguível — GO para os braços X4'})")


if __name__ == "__main__":
    main()
