#!/usr/bin/env python
"""R2 (docs/PARECER_AUDITORIA_ONYX.md §6-R2, segunda metade): mini-sweep de hiperparâmetros
julgado pelo juiz de treino novo (feval `ts_auc_by_t`, model/train.py). Grade pequena e registrada
(nao um sweep exaustivo): lr in {0.05, 0.02}, min_data_in_leaf in {200, 50}, lambda_l2 in {5, 1} ->
8 celulas. Reusa o dataset ja construido (data/processed/train_rows.parquet) -- nao reconstroi o
dataset a cada celula.

Cada celula treina o ensemble completo (5 folds) e salva seu OOF em <out-dir>/oof_<cell_id>.parquet.
O veredicto de qual celula "ganhou" NAO e decidido aqui por um numero pontual (plano §9.0) -- rode
scripts/compare_oof.py entre a celula default e cada candidata (ou entre pares de candidatas) para
decidir com intervalo de confianca (R0)."""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights

DEFAULT_GRID = {
    "learning_rate": [0.05, 0.02],
    "min_data_in_leaf": [200, 50],
    "lambda_l2": [5.0, 1.0],
}


def _cell_id(params: dict) -> str:
    return "_".join(f"{k}{v}".replace(".", "p") for k, v in params.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--out-dir", default="artifacts/sweep")
    parser.add_argument("--grid", default=None, help="JSON dict de listas, sobrescreve DEFAULT_GRID")
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = pd.read_parquet(args.rows)
    weights = compute_row_weights(rows, cfg)

    grid = json.loads(args.grid) if args.grid else DEFAULT_GRID
    keys = list(grid.keys())
    cells = [dict(zip(keys, combo)) for combo in product(*[grid[k] for k in keys])]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    for i, cell_params in enumerate(cells, start=1):
        cell_id = _cell_id(cell_params)
        print(f"\n{'=' * 70}\n[sweep] célula {i}/{len(cells)}: {cell_params} (id={cell_id})\n{'=' * 70}")
        t0 = time.time()

        lgb_cfg = dataclasses.replace(cfg.lightgbm, **cell_params)
        cell_cfg = dataclasses.replace(cfg, lightgbm=lgb_cfg)

        ensemble, oof_pred = train_ensemble(rows, weights, cell_cfg, progress=True)
        dt = time.time() - t0

        best_iters = [len(fe.get("valid_0", {}).get("ts_auc_by_t", [])) for fe in ensemble.fold_evals]
        oof_path = out_dir / f"oof_{cell_id}.parquet"
        oof_df = rows[["id", "t", "y"]].copy()
        oof_df["oof_pred"] = np.asarray(oof_pred, dtype=np.float64)
        oof_df.to_parquet(oof_path)

        print(f"[sweep] célula {cell_id}: {dt:.0f}s, best_iters={best_iters} -> {oof_path}")
        manifest.append({
            "cell_id": cell_id, "params": cell_params, "seconds": dt,
            "best_iters": best_iters, "oof_path": str(oof_path),
        })

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[sweep] {len(cells)} células concluídas. manifesto em {manifest_path}")
    print("[sweep] compare cada oof_*.parquet contra o baseline com scripts/compare_oof.py (R0) "
          "antes de adotar qualquer célula.")


if __name__ == "__main__":
    main()
