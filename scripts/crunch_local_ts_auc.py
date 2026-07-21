#!/usr/bin/env python
"""Teste local de TS-AUC no MOLDE CRUNCH sobre o conjunto de teste reduzido (data/X_test.reduced).

Roda o CAMINHO REAL DE SUBMISSÃO (`sbrt.adapter.platform.infer` — o mesmo `def infer` que a plataforma
chama, via `GeneratorWrapper`) e pontua com a fórmula oficial de TS-AUC (idêntica à célula "Computing
TS-AUC locally" do quickstarter e a scripts/local_ts_auc.py): AUC por passo online, ponderada por
n_pos·n_neg. É um conjunto HELD-OUT (100 séries), separado do OOF de treino -- portanto uma medida de
generalização honesta, não um número in-sample.

Diferente do OOF (§9.0/D4): não é o comparador pareado por série (esse é scripts/compare_oof.py sobre o
treino) -- é a régua mais próxima do leaderboard que dá para rodar localmente. Compara N modelos e
imprime o TS-AUC de cada um sobre exatamente as mesmas séries."""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sbrt.config import load_config
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


def _load_series(data_dir: str):
    X = pd.read_parquet(f"{data_dir}/X_test.reduced.parquet")
    ids = list(X.index.get_level_values("id").unique())
    datasets = []
    online_index = []  # (id, time) de cada ponto online, na ordem em que os scores serão emitidos
    for sid in ids:
        g = X.loc[sid]
        hist = g.loc[g["period"] == 1, "value"].to_numpy(dtype=np.float64)
        online_rows = g.loc[g["period"] == 2, "value"]
        datasets.append((hist.tolist(), online_rows.to_numpy(dtype=np.float64).tolist()))
        for time in online_rows.index:
            online_index.append((sid, time))
    return datasets, online_index


def _infer(model_dir: str, datasets, online_index, cfg) -> pd.Series:
    """Inferência pelo MESMO caminho da submissão: `StreamScorer` + `ModelEnsemble.predict_one` por
    passo (o que `platform.infer` faz em modo supervised). Carrega o ensemble explicitamente porque os
    modelos foram salvos por `ModelEnsemble.save()`, sem o marcador `model.joblib` que `platform.infer`
    usa para decidir supervised×fallback -- carregá-lo aqui evita o fallback silencioso."""
    ensemble = None
    if model_dir and os.path.exists(os.path.join(model_dir, "boosters.joblib")):
        from sbrt.model.predict import ModelEnsemble

        ensemble = ModelEnsemble.load(model_dir)

    scores = []
    for hist, online in datasets:
        h0 = fit_h0(np.asarray(hist, dtype=np.float64), cfg)
        scorer = StreamScorer(h0, default_blocks(), ensemble, cfg)
        for x in online:
            scores.append(float(scorer.update(float(x))))
    idx = pd.MultiIndex.from_tuples(online_index, names=["id", "time"])
    return pd.Series(scores, index=idx, name="prediction")


def official_ts_auc(prediction: pd.Series, y_test: pd.DataFrame) -> float:
    """Fórmula oficial (quickstarter / scripts/local_ts_auc.py): AUC por passo online, peso n_pos·n_neg."""
    merged = prediction.to_frame().join(y_test, how="left")
    merged["time_online"] = merged.groupby(level="id").cumcount()
    wsum = tot = 0.0
    for _, grp in merged.groupby("time_online"):
        labels = grp["target"].to_numpy()
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        if n_pos == 0 or n_neg == 0:
            continue
        auc = roc_auc_score(labels, grp["prediction"].to_numpy())
        w = float(n_pos * n_neg)
        wsum += w * auc
        tot += w
    return wsum / tot if tot > 0 else 0.5


def _ts_auc_by_bucket(prediction: pd.Series, y_test: pd.DataFrame) -> dict:
    merged = prediction.to_frame().join(y_test, how="left")
    merged["t"] = merged.groupby(level="id").cumcount() + 1
    edges = [0, 50, 150, 400, np.inf]
    labels = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]
    merged["bucket"] = pd.cut(merged["t"], bins=edges, labels=labels, right=True)
    out = {}
    for lb in labels:
        sub = merged[merged["bucket"] == lb]
        wsum = tot = 0.0
        for _, grp in sub.groupby("t"):
            yv = grp["target"].to_numpy()
            n_pos, n_neg = int(yv.sum()), int((1 - yv).sum())
            if n_pos == 0 or n_neg == 0:
                continue
            wsum += n_pos * n_neg * roc_auc_score(yv, grp["prediction"].to_numpy())
            tot += n_pos * n_neg
        out[lb] = wsum / tot if tot > 0 else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models", nargs="+", required=True,
                        help="pares nome=caminho, ex.: baseline=artifacts/models/v1_preV2 final=artifacts/models/v4")
    args = parser.parse_args()

    cfg = load_config()
    datasets, online_index = _load_series(args.data_dir)
    y_test = pd.read_parquet(f"{args.data_dir}/y_test.reduced.parquet")
    n_series = len({sid for sid, _ in online_index})
    print(f"molde crunch: {n_series} séries held-out, {len(online_index)} pontos online\n")

    results = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        pred = _infer(path, datasets, online_index, cfg)
        overall = official_ts_auc(pred, y_test)
        buckets = _ts_auc_by_bucket(pred, y_test)
        results[name] = (overall, buckets)
        print(f"[{name}] TS-AUC oficial (molde crunch): {overall:.4f}")
        for lb, v in buckets.items():
            print(f"    {lb:12s} {v:.4f}")
        print()

    if len(results) >= 2:
        names = list(results)
        base = results[names[0]][0]
        print("=== deltas vs. primeiro modelo ===")
        for nm in names[1:]:
            print(f"  {nm} - {names[0]}: {results[nm][0] - base:+.4f} (geral)")


if __name__ == "__main__":
    main()
