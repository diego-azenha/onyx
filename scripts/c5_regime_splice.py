#!/usr/bin/env python
"""C5-lite (BRAINSTORM_RUPTURA_V2.md, item C5): modelo por REGIME de t sem treinar nada.

O C5 original propunha dois modelos (t<=64 / t>64) com calibradores e HPs proprios. A campanha B
entregou o mesmo conteudo de graca: o braco B6 (`feature_contri=0.6` em `meta_h0_*`) GANHA em
150<t<=400 (+0,0049) e t>400 (+0,0048), mas PERDE em 50<t<=150 (-0,0029). Isto e exatamente o
conflito de regimes que o C5 postulava -- e como a TS-AUC e calculada DENTRO de cada t e so depois
agregada com pesos (`evaluation/ts_auc.weighted_ts_auc`), costurar dois OOFs por faixa de t e
metricamente legitimo: nenhuma AUC compara scores de t diferentes, entao as escalas dos dois modelos
nunca se encontram. Nao ha o problema de comparabilidade que matou X3/A3.

O ponto de costura (150) e uma BORDA DE BUCKET PRE-REGISTRADA (T_BUCKET_EDGES), nao um corte
ajustado aos dados. Ainda assim a decisao de "qual lado ganha" saiu deste mesmo OOF, entao o script
faz o teste honesto: escolhe o lado vencedor em 3 folds e CONFIRMA nos 2 restantes (mesma disciplina
do A1), alem do compare pareado no conjunto todo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc


def splice(base: pd.DataFrame, cand: pd.DataFrame, cut: float) -> pd.DataFrame:
    """base para t<=cut, cand para t>cut (linhas alinhadas por id/t)."""
    out = base.copy()
    hi = out["t"].to_numpy() > cut
    out.loc[hi, "oof_pred"] = cand.loc[hi, "oof_pred"].to_numpy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="artifacts/models/oof_x1_soft_bag4.parquet")
    ap.add_argument("--cand", default="artifacts/models/oof_b6_contri06_bag4.parquet")
    ap.add_argument("--cut", type=float, default=150.0)
    ap.add_argument("--out", default="artifacts/models/oof_c5_splice_bag4.parquet")
    args = ap.parse_args()

    base = pd.read_parquet(args.base).sort_values(["id", "t"]).reset_index(drop=True)
    cand = pd.read_parquet(args.cand).sort_values(["id", "t"]).reset_index(drop=True)
    assert (base["id"].to_numpy() == cand["id"].to_numpy()).all()
    assert (base["t"].to_numpy() == cand["t"].to_numpy()).all()

    sp = splice(base, cand, args.cut)
    t = base["t"].to_numpy(np.float64)
    y = base["y"].to_numpy(np.float64)

    print(f"corte t={args.cut:g}  (base para t<=corte, cand para t>corte)")
    for name, df in (("base", base), ("cand", cand), ("splice", sp)):
        print(f"  {name:7s} TS-AUC = {weighted_ts_auc(t, y, df['oof_pred'].to_numpy(np.float64)):.4f}")

    # --- teste honesto: escolhe em 3 folds, confirma em 2 ---------------------------------------
    cfg = load_config()
    meta = base[["id", "t", "y"]]
    folds = list(grouped_stratified_kfold(meta, cfg.lightgbm.n_folds, cfg.seed))
    fold_of = np.full(len(base), -1, dtype=np.int64)
    for f, (_, valid_idx) in enumerate(folds):
        fold_of[valid_idx] = f
    sel = np.isin(fold_of, [0, 1, 2])
    con = np.isin(fold_of, [3, 4])

    print("\nselecao (folds 0-2) / confirmacao (folds 3-4):")
    for split_name, m in (("selecao", sel), ("confirm", con)):
        d_sp = weighted_ts_auc(t[m], y[m], sp["oof_pred"].to_numpy(np.float64)[m])
        d_ba = weighted_ts_auc(t[m], y[m], base["oof_pred"].to_numpy(np.float64)[m])
        print(f"  {split_name}: splice {d_sp:.4f}  base {d_ba:.4f}  delta {d_sp - d_ba:+.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sp.to_parquet(args.out, index=False)
    print(f"\nOOF costurado -> {args.out}  (rode compare_oof.py para o IC pareado)")


if __name__ == "__main__":
    main()
