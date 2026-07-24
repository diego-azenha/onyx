#!/usr/bin/env python
"""D1 (BRAINSTORM_RUPTURA_TSAUC.md §3, gate de X2): estratifica a TS-AUC da família de MÉDIA por
tercis de PERSISTÊNCIA (`ar_r2` do censo A1). Testa o mecanismo do Defeito 1 de X2 antes de qualquer
código: em séries persistentes (Σφ→1, `ar_r2` alto) um degrau de média vira um PULSO de ~p passos no
canal branqueado e satura em `e_vol` -- o detector de média fica cego. Em séries de baixa persistência
o mesmo degrau aparece cheio.

Predição registrada a priori: a cegueira concentra-se no tercil PERSISTENTE (ar_r2 alto -> AUC menor);
o tercil de baixa persistência deve estar visivelmente acima de 0,5, sobretudo no subconjunto de
quebras de média claras (|delta_mean_e| > thresh). Se a estratificação sair PLANA, o mecanismo está
errado e X2 morre antes de nascer (nenhum build).

Método: a TS-AUC por passo compara, em cada t, os positivos (y=1) contra o pool de vivos (y=0). Aqui
os positivos são restringidos por tercil de ar_r2 da SÉRIE quebrada; os negativos são o pool inteiro.
Só precisa do censo (ar_r2 por série quebrada) + a OOF -- não lê o parquet de features."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import weighted_ts_auc


def _auc_for_positives(oof: pd.DataFrame, pos_ids: set) -> tuple[float, int]:
    """TS-AUC onde os positivos (y=1) vêm só de `pos_ids`; negativos = todos os vivos (y=0)."""
    keep = ((oof["y"].to_numpy() == 1) & oof["id"].isin(pos_ids).to_numpy()) | (oof["y"].to_numpy() == 0)
    sub = oof.loc[keep]
    auc = weighted_ts_auc(sub["t"].to_numpy(np.float64), sub["y"].to_numpy(np.float64),
                          sub["oof_pred"].to_numpy(np.float64))
    n_pos_series = len(pos_ids)
    return float(auc), n_pos_series


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oof", default="artifacts/models/oof_x0_a_bag4.parquet")
    p.add_argument("--census", default="artifacts/reports/break_type_census.csv")
    p.add_argument("--ar-col", default="ar_r2")
    p.add_argument("--mean-col", default="delta_mean_e")
    p.add_argument("--mean-thresh", type=float, default=0.3, help="|delta_mean_e| acima disto = quebra de média clara (~6,8% das séries)")
    args = p.parse_args()

    oof = pd.read_parquet(args.oof)
    census = pd.read_csv(args.census)
    census = census.dropna(subset=[args.ar_col])

    # tercis de persistência sobre as séries quebradas
    census["tercil"] = pd.qcut(census[args.ar_col], 3, labels=["baixo", "médio", "alto"])
    mean_break = census[args.mean_col].abs() > args.mean_thresh
    print(f"séries quebradas no censo: {len(census)}; com |{args.mean_col}|>{args.mean_thresh}: "
          f"{int(mean_break.sum())} ({100*mean_break.mean():.1f}%)\n")

    header = f"{'tercil ar_r2':14s} {'ar_r2 faixa':>18s} {'n_pos':>7s} {'AUC(todas)':>11s} {'n_mean':>7s} {'AUC(média)':>11s}"
    print(header)
    print("-" * len(header))
    for terc in ["baixo", "médio", "alto"]:
        sub = census[census["tercil"] == terc]
        lo, hi = sub[args.ar_col].min(), sub[args.ar_col].max()
        auc_all, n_all = _auc_for_positives(oof, set(sub["id"]))
        sub_mean = sub[mean_break.loc[sub.index]]
        if len(sub_mean):
            auc_mean, n_mean = _auc_for_positives(oof, set(sub_mean["id"]))
        else:
            auc_mean, n_mean = float("nan"), 0
        print(f"{terc:14s} {f'[{lo:.3f},{hi:.3f}]':>18s} {n_all:7d} {auc_all:11.4f} {n_mean:7d} {auc_mean:11.4f}")

    print("\nleitura: AUC(média) DECRESCENTE de 'baixo'->'alto' ar_r2 confirma o Defeito 1 (cegueira "
          "concentrada em séries persistentes) -> GO para X2. Estratificação plana -> mecanismo errado, "
          "X2 morre. O 'baixo' precisa ficar visivelmente > 0,5.")


if __name__ == "__main__":
    main()
