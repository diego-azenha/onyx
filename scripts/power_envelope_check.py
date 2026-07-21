#!/usr/bin/env python
"""R6-iii (docs/PARECER_AUDITORIA_ONYX.md §6-R6): compara a TS-AUC OOF observada por bucket de t
contra o envelope de POTÊNCIA de um teste z de shift de média, Phi(delta*sqrt(m) - z_alpha), usando
a magnitude de shift medida pelo censo A1 (scripts/break_type_census.py) e o número de pontos
pós-quebra `m` observado no próprio OOF.

Isto é uma aproximação DELIBERADA, não uma equivalência formal: a TS-AUC agrega sinal de
média+variância+dependência+forma (parecer §3.1), enquanto o envelope aqui modela só um shift de
média sob um teste z simples (o "H1: ruído amostral" do DIAGNOSTICO usa a mesma família). O objetivo
não é uma igualdade numérica -- é responder "este bucket está perto do teto de informação de um
detector de shift-de-média simples, ou a AUC observada está bem abaixo do que a magnitude medida
permitiria?" (parecer §4.1, "t<=50 pode estar perto do teto de informação"). Uma AUC observada muito
menor que o envelope sugere gargalo de EXTRAÇÃO (R1-R4); uma AUC próxima do envelope sugere teto de
INFORMAÇÃO para esta família de efeito."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats as spstats

from sbrt.evaluation.ts_auc import weighted_ts_auc

T_BUCKET_EDGES = [0, 50, 150, 400, np.inf]
T_BUCKET_LABELS = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oof", default="artifacts/models/oof_v1.parquet")
    parser.add_argument("--census", default="artifacts/reports/break_type_census.csv")
    parser.add_argument("--label-col", default="y")
    parser.add_argument("--score-col", default="oof_pred")
    parser.add_argument("--alpha", type=float, default=0.05, help="nivel do teste z hipotetico (one-sided)")
    parser.add_argument("--delta-col", default="delta_mean_e",
                         help="coluna do censo usada como magnitude de shift (unidades de inovacao)")
    args = parser.parse_args()

    oof = pd.read_parquet(args.oof)
    census = pd.read_csv(args.census)

    delta = float(census[args.delta_col].abs().median())
    z_alpha = float(spstats.norm.ppf(1.0 - args.alpha))

    oof = oof.copy()
    idx = pd.cut(oof["t"], bins=T_BUCKET_EDGES, labels=False, right=True)
    oof["t_bucket"] = np.array(T_BUCKET_LABELS, dtype=object)[idx.to_numpy(dtype=np.int64)]

    has_break = oof.groupby("id")[args.label_col].transform("max") > 0
    broken = oof.loc[has_break].copy()
    tau_first_t = broken.loc[broken[args.label_col] == 1].groupby("id")["t"].min()
    broken["tau_first_t"] = broken["id"].map(tau_first_t)
    broken["m_post_break"] = broken["t"] - broken["tau_first_t"] + 1
    broken = broken.loc[(broken[args.label_col] == 1) & (broken["m_post_break"] > 0)]

    print(f"delta (mediana |{args.delta_col}| do censo A1): {delta:.4f}")
    print(f"z_alpha (alpha={args.alpha}, one-sided): {z_alpha:.4f}\n")

    header = f"{'bucket':14s} {'auc_obs':>9s} {'median_m':>9s} {'power_env':>10s} {'gap':>9s}"
    print(header)
    print("-" * len(header))
    for label in T_BUCKET_LABELS:
        sub = oof[oof["t_bucket"] == label]
        auc_obs = weighted_ts_auc(
            sub["t"].to_numpy(), sub[args.label_col].to_numpy(), sub[args.score_col].to_numpy()
        )
        m_sub = broken.loc[broken["t_bucket"] == label, "m_post_break"]
        median_m = float(m_sub.median()) if len(m_sub) else float("nan")
        power_env = float(spstats.norm.cdf(delta * np.sqrt(max(median_m, 0.0)) - z_alpha)) if median_m == median_m else float("nan")
        gap = power_env - auc_obs if power_env == power_env else float("nan")
        print(f"{label:14s} {auc_obs:9.4f} {median_m:9.1f} {power_env:10.4f} {gap:9.4f}")

    print("\nleitura: 'gap' grande e positivo -> AUC bem abaixo do envelope de um shift-de-media "
          "simples (indicio de gargalo de EXTRACAO, R1-R4); 'gap' perto de 0 -> bucket proximo do "
          "teto de informacao para esta familia de efeito (nao e o mesmo que 'teto de informacao "
          "absoluto', ja que a TS-AUC tambem premia shifts de variancia/forma nao capturados aqui).")


if __name__ == "__main__":
    main()
