#!/usr/bin/env python
"""A1 (BRAINSTORM_RUPTURA_V2.md §2.2): sonda offline do EWMA assimétrico no pós-processamento. Aplica a
transformação causal por série sobre as trajetórias OOF do incumbente e mede o Δ TS-AUC. Grade com
SELEÇÃO em 3 folds / CONFIRMAÇÃO nos 2 restantes (com barra fina, grid sem holdout fabrica ganho).
Diagnóstico; a adoção final passa por `scripts/compare_oof.py` (IC) sobre o OOF transformado do melhor
cel."""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc
from sbrt.postprocess.monotonicity import ema_asym_step


def _apply_asym(oof: pd.DataFrame, a_up: float, a_dn: float) -> np.ndarray:
    """EWMA assimétrico causal por série (assume oof ordenado por id,t)."""
    s = oof["oof_pred"].to_numpy(dtype=np.float64)
    ids = oof["id"].to_numpy()
    out = np.empty_like(s)
    prev = None
    cur_id = None
    for i in range(len(s)):
        if ids[i] != cur_id:
            cur_id = ids[i]
            prev = None
        out[i] = s[i] if prev is None else ema_asym_step(s[i], prev, a_up, a_dn)
        prev = out[i]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--oof", default="artifacts/models/oof_x1_soft_bag4.parquet")
    p.add_argument("--up", nargs="+", type=float, default=[0.6, 0.8])
    p.add_argument("--down", nargs="+", type=float, default=[0.05, 0.1, 0.2])
    p.add_argument("--out", default="artifacts/models/oof_A1_vema.parquet")
    args = p.parse_args()

    cfg = load_config(args.config)
    oof = pd.read_parquet(args.oof).sort_values(["id", "t"]).reset_index(drop=True)

    fold_of = {}
    for k, (_, valid) in enumerate(grouped_stratified_kfold(oof, cfg.lightgbm.n_folds, cfg.seed)):
        for i in oof.iloc[valid]["id"].unique():
            fold_of[int(i)] = k
    fold = oof["id"].map(fold_of).to_numpy()
    sel = np.isin(fold, [0, 1, 2])
    conf = np.isin(fold, [3, 4])

    t, y = oof["t"].to_numpy(), oof["y"].to_numpy()
    base_sel = weighted_ts_auc(t[sel], y[sel], oof["oof_pred"].to_numpy()[sel])
    base_conf = weighted_ts_auc(t[conf], y[conf], oof["oof_pred"].to_numpy()[conf])

    print(f"baseline TS-AUC — seleção(3 folds) {base_sel:.4f} · confirmação(2 folds) {base_conf:.4f}\n")
    print(f"{'a_up':>6} {'a_down':>7} {'Δ_seleção':>11}")
    best = (None, -9.0, None)
    cache = {}
    for a_up, a_dn in product(args.up, args.down):
        if a_up < a_dn:
            continue
        tr = _apply_asym(oof, a_up, a_dn)
        cache[(a_up, a_dn)] = tr
        d_sel = weighted_ts_auc(t[sel], y[sel], tr[sel]) - base_sel
        print(f"{a_up:6.2f} {a_dn:7.2f} {d_sel:+11.4f}")
        if d_sel > best[1]:
            best = ((a_up, a_dn), d_sel, tr)

    (a_up, a_dn), d_sel, tr = best
    d_conf = weighted_ts_auc(t[conf], y[conf], tr[conf]) - base_conf
    print(f"\nmelhor na seleção: a_up={a_up}, a_down={a_dn} (Δ_sel {d_sel:+.4f})")
    print(f"CONFIRMAÇÃO nos 2 folds retidos: Δ_conf {d_conf:+.4f}  "
          f"({'replica — vale medir com IC' if d_conf > 0 else 'NÃO replica — kill'})")

    out = oof[["id", "t", "y"]].copy()
    out["oof_pred"] = tr
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"\nOOF transformado (melhor cel) salvo em {args.out} — veredito final por compare_oof.py.")


if __name__ == "__main__":
    main()
