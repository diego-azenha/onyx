#!/usr/bin/env python
"""CLI fina: treina o ensemble LightGBM (plano §8.3) a partir de data/processed/train_rows.parquet.
Salva também as predições out-of-fold (id, t, y, tau_index-derivado, oof_pred) para diagnósticos
(A4: resposta ao degrau alinhada em tau, plano_acao_v1_para_v2.md)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--out", default="artifacts/models/v1")
    parser.add_argument("--oof-out", default=None, help="default: <out>/../oof_<basename(out)>.parquet")
    parser.add_argument("--drop-prefix", nargs="*", default=[], metavar="PREFIXO",
                        help="remove colunas de feature com estes prefixos antes de treinar. Existe "
                             "para separar BRAÇOS de R0 sem repetir o build: um único dataset com "
                             "várias famílias novas rende um braço por família, cada um com sua R0, "
                             "porque `model/train.py` deriva `feature_order` das colunas presentes. "
                             "A disciplina de 'um braço por R0' é sobre o MODELO, não sobre o build.")
    parser.add_argument("--boost-seed", type=int, default=None,
                        help="sobrescreve lightgbm.boost_seed (sorteio interno do LightGBM; NÃO muda "
                             "os folds, que vêm de cfg.seed). MEDIDO 2026-07-22: trocar só isto move "
                             "a TS-AUC OOF em -0,0037 [-0,0088, +0,0012] — maior que o Delta de dois "
                             "dos três braços já descartados. Por isso todo braço de R0 agora se mede "
                             "com VÁRIAS sementes por lado; ver docs/BACKLOG_TSAUC.md.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.boost_seed is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, boost_seed=args.boost_seed))
        print(f"boost_seed sobrescrito para {args.boost_seed} (folds inalterados, cfg.seed={cfg.seed})")
    rows = pd.read_parquet(args.rows)
    if args.drop_prefix:
        dropped = [c for c in rows.columns if c.startswith(tuple(args.drop_prefix))]
        rows = rows.drop(columns=dropped)
        print(f"removidas {len(dropped)} colunas por --drop-prefix {args.drop_prefix}: {sorted(dropped)}")
    weights = compute_row_weights(rows, cfg)
    ensemble, oof_pred = train_ensemble(rows, weights, cfg, progress=True)
    ensemble.save(args.out)
    print(f"ensemble salvo em {args.out} ({len(ensemble.boosters)} folds, {len(ensemble.feature_order)} features)")

    out_dir = Path(args.out)
    oof_out = Path(args.oof_out) if args.oof_out else out_dir.parent / f"oof_{out_dir.name}.parquet"
    oof_df = rows[["id", "t", "y"]].copy()
    oof_df["oof_pred"] = np.asarray(oof_pred, dtype=np.float64)
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_out)
    print(f"predições out-of-fold salvas em {oof_out} ({len(oof_df)} linhas)")


if __name__ == "__main__":
    main()
