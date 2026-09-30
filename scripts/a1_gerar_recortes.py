#!/usr/bin/env python
"""A1 (knowledge/frentes/teto-offline/A1-recorte.md): séries novas por re-corte das séries reais.

Para cada série original e cada cópia k: S = hist ++ online (quebra em b = n_h + tau, se houver).
Sorteia n_on' ~ U[10, 999] e tau' ~ U[0, n_on') (positivos) respeitando o que existe em S, corta o
novo online em c = b - tau' e o novo histórico em S[c - n_h' : c], com n_h' ~ U[1000, 5000] limitado ao
disponível. Re-padroniza tudo pela média/dp do novo histórico. Negativos: corte livre dentro de S.
As linhas saem pelo MESMO motor do Onyx (`build_training_rows`), com id = 10000*k + id_origem.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.model.dataset import SeriesRecord, build_training_rows  # noqa: E402

MIN_H, MAX_H, MIN_ON, MAX_ON = 1000, 5000, 10, 999


def recortar(sid: int, h: np.ndarray, on: np.ndarray, tau: int, rng: np.random.Generator, k: int):
    S = np.concatenate([h, on])
    n = len(S)
    if tau >= 0:
        b = len(h) + tau
        pos_disp = n - b                                   # pontos pós-quebra disponíveis
        for _ in range(20):
            n_on = int(rng.integers(MIN_ON, MAX_ON + 1))
            tau2 = int(rng.integers(0, n_on))
            if n_on - tau2 > pos_disp:                     # não há pós-quebra suficiente
                continue
            c = b - tau2
            if c < MIN_H:                                  # histórico novo precisa de >= 1000 pontos H0
                continue
            break
        else:
            return None
    else:
        tau2 = None
        n_on = int(rng.integers(MIN_ON, min(MAX_ON, n - MIN_H) + 1))
        c = int(rng.integers(MIN_H, n - n_on + 1))
    n_h = int(rng.integers(MIN_H, min(MAX_H, c) + 1))
    hist = S[c - n_h:c]
    online = S[c:c + n_on]
    mu, sd = hist.mean(), hist.std()
    sd = sd if sd > 0 else 1.0
    return SeriesRecord(dataset_id=10000 * k + sid, x_hist=(hist - mu) / sd, x_online=(online - mu) / sd,
                        tau_index=tau2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1, help="índice da cópia (id = 10000*k + origem)")
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out or f"data/processed/aug_recorte_k{args.k}.parquet")
    cfg = load_config(DEFAULT_CONFIG_PATH)
    yi = pd.read_parquet("data/y_train_index.parquet")
    rng = np.random.default_rng(20260930 + args.k)
    recs = []
    for sid, h, on in _carregar_series(Path("data")):
        r = recortar(sid, h, on, int(yi.loc[sid, "tau_index"]), rng, args.k)
        if r is not None:
            recs.append(r)
    pos = sum(r.tau_index is not None for r in recs)
    print(f"{len(recs)} séries re-cortadas ({pos} com quebra)", flush=True)
    rows = build_training_rows(recs, cfg, progress=True, n_jobs=args.n_jobs)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out)
    print(f"gravado {len(rows)} linhas em {out}")


if __name__ == "__main__":
    main()
