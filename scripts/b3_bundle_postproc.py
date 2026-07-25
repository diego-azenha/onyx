#!/usr/bin/env python
"""B3 (CAMPANHA_POLIMENTO.md) — o bundle de pos-processo, reotimizado sob a ponderacao do BOARD.

Duas coisas mudam em relacao a `postproc_vema_probe.py`:

1. **A grade.** O gate `t>150` do A1 foi escolhido com a justificativa registrada em
   `scripts/apply_vema_oof.py`: "+0,0009/+0,0012 em 150<t<=400 / t>400, PERDE em t<=50 (-0,0014)".
   Sob os pesos do board `t>400` vale 31,9% (nao 16,5%) e `t<=50` vale 3,9% (nao 8,1%) -- ou seja o
   ganho foi subcreditado e o pedagio, supercreditado. O corte otimo tem de ser reencontrado.
2. **O bundle e otimizado JUNTO** (a_up, a_dn, min_t), nao um parametro por vez, e o vencedor e
   confirmado em folds que nao participaram da selecao -- com barra fina, grade sem holdout fabrica
   ganho (a licao do proprio probe).

A varredura roda nas DUAS particoes quando os dois OOFs sao passados, e so reporta como candidato o
que ganha nas duas: a variancia de particao (~0,007) e 5x a barra.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc
from postproc_vema_probe import _apply_asym


def _gated(oof: pd.DataFrame, a_up: float, a_dn: float, min_t: float | None) -> np.ndarray:
    """v-EMA aplicado so em t>min_t; a recursao reinicia por serie na primeira linha acima do corte
    (mesma semantica de scripts/apply_vema_oof.py, para que o resultado seja implantavel tal e qual)."""
    if min_t is None:
        return _apply_asym(oof, a_up, a_dn)
    hi = oof["t"].to_numpy() > min_t
    vals = oof["oof_pred"].to_numpy(dtype=np.float64).copy()
    if hi.any():
        vals[hi] = _apply_asym(oof.loc[hi].reset_index(drop=True), a_up, a_dn)
    return vals


def varrer(oof_path: str, cfg, ups, downs, cortes, y_train: str) -> pd.DataFrame:
    oof = pd.read_parquet(oof_path).sort_values(["id", "t"]).reset_index(drop=True)
    wm = board_grid_multipliers(y_train, oof["t"].to_numpy())

    fold_of: dict[int, int] = {}
    for k, (_, valid) in enumerate(grouped_stratified_kfold(oof, cfg.lightgbm.n_folds, cfg.seed)):
        for i in oof.iloc[valid]["id"].unique():
            fold_of[int(i)] = k
    fold = oof["id"].map(fold_of).to_numpy()
    sel, conf = np.isin(fold, [0, 1, 2]), np.isin(fold, [3, 4])

    t, y = oof["t"].to_numpy(), oof["y"].to_numpy()
    s0 = oof["oof_pred"].to_numpy()
    base = {"sel": weighted_ts_auc(t[sel], y[sel], s0[sel], wm),
            "conf": weighted_ts_auc(t[conf], y[conf], s0[conf], wm),
            "tudo": weighted_ts_auc(t, y, s0, wm)}
    print(f"  base (grade do board): sel {base['sel']:.5f} · conf {base['conf']:.5f} · tudo {base['tudo']:.5f}")

    linhas = []
    for a_up, a_dn, corte in product(ups, downs, cortes):
        if a_dn > a_up:
            continue
        v = _gated(oof, a_up, a_dn, corte)
        linhas.append({
            "a_up": a_up, "a_dn": a_dn, "min_t": -1 if corte is None else corte,
            "d_sel": weighted_ts_auc(t[sel], y[sel], v[sel], wm) - base["sel"],
            "d_conf": weighted_ts_auc(t[conf], y[conf], v[conf], wm) - base["conf"],
            "d_tudo": weighted_ts_auc(t, y, v, wm) - base["tudo"],
        })
    return pd.DataFrame(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--oof-a", default="artifacts/models/oof_b6c6_joint_bag4.parquet",
                    help="OOF da particao 42")
    ap.add_argument("--oof-b", default=None, help="OOF da particao 43 (opcional, mas exigido para adotar)")
    ap.add_argument("--up", nargs="+", type=float, default=[0.6, 0.8, 1.0])
    ap.add_argument("--down", nargs="+", type=float, default=[0.05, 0.1, 0.2, 0.4])
    ap.add_argument("--cortes", nargs="+", default=["none", "50", "100", "150", "250", "400"])
    ap.add_argument("--y-train", default="data/y_train.parquet")
    ap.add_argument("--out", default="artifacts/reports/b3_bundle.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cortes = [None if c == "none" else float(c) for c in args.cortes]

    print("=== particao 42 ===")
    a = varrer(args.oof_a, cfg, args.up, args.down, cortes, args.y_train)
    print("\n  top 8 por d_sel (e o que a confirmacao diz deles):")
    print(a.sort_values("d_sel", ascending=False).head(8).to_string(index=False,
          float_format=lambda v: f"{v:+.5f}"))

    melhor = a.sort_values("d_sel", ascending=False).iloc[0]
    print(f"\n  escolhido na SELECAO: a_up={melhor.a_up} a_dn={melhor.a_dn} min_t={melhor.min_t}")
    print(f"  -> na CONFIRMACAO: {melhor.d_conf:+.5f}  (se negativo, a grade fabricou o ganho)")

    if args.oof_b:
        print("\n=== particao 43 ===")
        b = varrer(args.oof_b, cfg, args.up, args.down, cortes, args.y_train)
        j = a.merge(b, on=["a_up", "a_dn", "min_t"], suffixes=("_42", "_43"))
        j["min_das_duas"] = j[["d_tudo_42", "d_tudo_43"]].min(axis=1)
        print("\n  top 8 pelo PIOR das duas particoes (o criterio que a campanha exige):")
        print(j.sort_values("min_das_duas", ascending=False)
               .head(8)[["a_up", "a_dn", "min_t", "d_tudo_42", "d_tudo_43", "min_das_duas"]]
               .to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
        j.to_csv(args.out, index=False)
    else:
        a.to_csv(args.out, index=False)
        print("\n  (sem particao 43: nada aqui pode ser adotado -- so triado)")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
