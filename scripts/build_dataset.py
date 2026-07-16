#!/usr/bin/env python
"""CLI fina: constrói o dataset de treino por passo a partir de data/X_train.parquet +
data/y_train_index.parquet (plano §8.1). Nenhuma lógica nova aqui — a lógica vive em
src/sbrt/model/dataset.py (docs/PLANO_REPOSITORIO.md §5, regra tqdm)."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.dataset import SeriesRecord, build_training_rows


def _load_series_records(data_dir: Path) -> list:
    X = pd.read_parquet(data_dir / "X_train.parquet")
    y_index = pd.read_parquet(data_dir / "y_train_index.parquet")

    records = []
    for dataset_id, group in tqdm(X.groupby(level="id"), desc="lendo séries de X_train.parquet"):
        g = group.reset_index()
        hist_vals = g.loc[g["period"] == 1, "value"].to_numpy(dtype="float64")
        online_vals = g.loc[g["period"] == 2, "value"].to_numpy(dtype="float64")
        tau = int(y_index.loc[dataset_id, "tau_index"])
        tau_index = tau if tau >= 0 else None
        records.append(
            SeriesRecord(dataset_id=int(dataset_id), x_hist=hist_vals, x_online=online_vals, tau_index=tau_index)
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="data/processed/train_rows.parquet")
    parser.add_argument("--n-jobs", type=int, default=None, help="override cfg.model.dataset_n_jobs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_jobs = args.n_jobs if args.n_jobs is not None else cfg.model.dataset_n_jobs
    records = _load_series_records(Path(args.data_dir))
    rows = build_training_rows(records, cfg, progress=True, n_jobs=n_jobs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(out_path)
    print(f"gravado {len(rows)} linhas ({rows['id'].nunique()} séries) em {out_path}")


if __name__ == "__main__":
    main()
