#!/usr/bin/env python
"""A2: gera o score do FALLBACK determinístico (bayes + banco CUSUM + martingale conformal) por linha
(id,t), para entrar como MEMBRO do stacking heterogêneo. O fallback é uma função fechada das features
(model/fallback.py:fallback_score) — não tem efeitos-fixos aprendidos, então pode ser relativamente
mais forte em t≤50 onde o supervisionado é mais fraco. Sem treino, sem fold: é determinístico das
features. Vetorizado sobre o parquet de treino."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.fallback import _CUSUM_BANK_KEYS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--rows", default="data/processed/train_rows_3eixos.parquet")
    p.add_argument("--out", default="artifacts/models/oof_fallback.parquet")
    args = p.parse_args()

    cfg = load_config(args.config)
    fb = cfg.fallback
    cols = ["id", "t", "y", "bayes_lo_h0025", "conformal_logm_abs_reset", *list(_CUSUM_BANK_KEYS)]
    df = pd.read_parquet(args.rows, columns=cols)

    lo = df["bayes_lo_h0025"].to_numpy(np.float64)
    logm = df["conformal_logm_abs_reset"].to_numpy(np.float64)
    bank = df[list(_CUSUM_BANK_KEYS)].to_numpy(np.float64)
    max_cusum = np.nanmax(np.where(np.isnan(bank), 0.0, bank), axis=1)
    max_cusum = np.clip(max_cusum, 0.0, None)
    z_cusum = np.sqrt(2.0 * max_cusum)

    logit = fb.w_lo * np.nan_to_num(lo) + fb.w_cusum * z_cusum + fb.w_conformal * np.nan_to_num(logm) - fb.bias
    score = 1.0 / (1.0 + np.exp(-logit))

    out = df[["id", "t", "y"]].copy()
    out["oof_pred"] = score
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    from sbrt.evaluation.ts_auc import weighted_ts_auc
    print(f"fallback OOF salvo em {args.out} ({len(out)} linhas); "
          f"TS-AUC do fallback sozinho: {weighted_ts_auc(out['t'].values, out['y'].values, out['oof_pred'].values):.4f}")


if __name__ == "__main__":
    main()
