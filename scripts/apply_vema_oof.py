#!/usr/bin/env python
"""Aplica o EWMA assimetrico do A1 (`postprocess/monotonicity.ema_asym_step`) sobre um OOF qualquer e
grava o parquet transformado, para empilhar o A1 sobre outros bracos (X1-soft, B6, costura C5) sem
retreinar. A sonda `postproc_vema_probe.py` faz a GRADE (selecao/confirmacao); aqui os alphas ja vem
escolhidos -- este script so materializa o OOF para o `compare_oof.py` medir com IC.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from postproc_vema_probe import _apply_asym


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", required=True)
    ap.add_argument("--a-up", type=float, default=0.8)
    ap.add_argument("--a-dn", type=float, default=0.2)
    ap.add_argument("--min-t", type=float, default=None,
                    help="aplica o v-ema so em t>min-t (a recursao recomeca em cada serie na "
                         "primeira linha acima do corte). O A1 e um braco de t ALTO (+0,0009/+0,0012 "
                         "em 150<t<=400 / t>400) que PERDE em t<=50 (-0,0014); o gate evita pagar "
                         "esse pedagio. Mesma logica de regime do c5_regime_splice.py.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    oof = pd.read_parquet(args.oof).sort_values(["id", "t"]).reset_index(drop=True)
    if args.min_t is None:
        oof["oof_pred"] = _apply_asym(oof, args.a_up, args.a_dn)
    else:
        hi = oof["t"].to_numpy() > args.min_t
        sub = oof.loc[hi].reset_index(drop=True)  # ja ordenado por id,t; a recursao reinicia por id
        vals = oof["oof_pred"].to_numpy(dtype=float).copy()
        vals[hi] = _apply_asym(sub, args.a_up, args.a_dn)
        oof["oof_pred"] = vals
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(args.out, index=False)
    print(f"v-ema(a_up={args.a_up}, a_dn={args.a_dn}) aplicado: {args.oof} -> {args.out}")


if __name__ == "__main__":
    main()
