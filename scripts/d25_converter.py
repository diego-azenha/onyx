#!/usr/bin/env python
"""D25 (knowledge/frentes/teto-offline/D25-dados-2025.md): séries da edição 2025 convertidas para o
formato de 2026.

2025: cada série tem period 0 (sempre H0) e period 1; `structural_breakpoint=True` significa que a
quebra acontece exatamente na fronteira. Conversão (mesma lógica do re-corte do A1):
- positivo: novo online começa em c dentro do period 0, com τ' = n0 − c; histórico = P0[c − n_h' : c]
  com n_h' ≥ 1000; online = P0[c:] ++ P1, cortado em n_on' ~ U[10, 999], com τ' ~ U[0, n_on');
- negativo: P0 ++ P1 é H0 contínuo, então o corte é livre.
Tudo padronizado pela média e dp do novo histórico. Os ids ficam em 1.000.000 + 10000·k + id_2025:
externos, entram no treino de TODOS os folds (`a1_treino.py`) e nunca são validados.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.model.dataset import SeriesRecord

MIN_H, MAX_H, MIN_ON, MAX_ON = 1000, 5000, 10, 999
BASE_ID = 1_000_000


def carregar_2025(data_dir: Path = Path("data/2025")):
    X = pd.read_parquet(data_dir / "X_train_2025.parquet")
    y = pd.read_parquet(data_dir / "y_train_2025.parquet")["structural_breakpoint"]
    ids = X.index.get_level_values(0).to_numpy()
    per = X["period"].to_numpy()
    val = X["value"].to_numpy(dtype=np.float64)
    cortes = np.flatnonzero(np.diff(ids)) + 1
    out = []
    for a, b in zip(np.r_[0, cortes], np.r_[cortes, len(ids)]):
        sid = int(ids[a])
        out.append((sid, val[a:b][per[a:b] == 0], val[a:b][per[a:b] == 1], bool(y.loc[sid])))
    return out


def converter(sid: int, p0: np.ndarray, p1: np.ndarray, quebra: bool, rng: np.random.Generator, k: int):
    S = np.concatenate([p0, p1])
    n0 = len(p0)
    if quebra:
        for _ in range(20):
            n_on = int(rng.integers(MIN_ON, MAX_ON + 1))
            tau2 = int(rng.integers(0, n_on))
            if n_on - tau2 > len(p1) or n0 - tau2 < MIN_H:
                continue
            c = n0 - tau2
            break
        else:
            return None
    else:
        tau2 = None
        n_on = int(rng.integers(MIN_ON, min(MAX_ON, len(S) - MIN_H) + 1))
        c = int(rng.integers(MIN_H, len(S) - n_on + 1))
    n_h = int(rng.integers(MIN_H, min(MAX_H, c) + 1))
    hist, online = S[c - n_h:c], S[c:c + n_on]
    mu, sd = hist.mean(), hist.std()
    sd = sd if sd > 0 else 1.0
    return SeriesRecord(dataset_id=BASE_ID + 10000 * k + sid, x_hist=(hist - mu) / sd,
                        x_online=(online - mu) / sd, tau_index=tau2)


def registros(k: int, seed: int = 2025) -> list:
    rng = np.random.default_rng(seed * 100 + k)
    recs = []
    for sid, p0, p1, q in carregar_2025():
        r = converter(sid, p0, p1, q, rng, k)
        if r is not None:
            recs.append(r)
    return recs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    from sbrt.config import DEFAULT_CONFIG_PATH, load_config
    from sbrt.model.dataset import build_training_rows
    recs = registros(args.k)
    print(f"{len(recs)} séries de 2025 convertidas ({sum(r.tau_index is not None for r in recs)} com quebra)", flush=True)
    rows = build_training_rows(recs, load_config(DEFAULT_CONFIG_PATH), progress=True, n_jobs=args.n_jobs)
    out = Path(f"data/processed/aug_2025_k{args.k}.parquet")
    rows.to_parquet(out)
    print(f"gravado {len(rows)} linhas em {out}")


if __name__ == "__main__":
    main()
