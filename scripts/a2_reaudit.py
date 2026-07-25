#!/usr/bin/env python
"""A2 (CAMPANHA_POLIMENTO.md) — reauditoria das decisoes ja tomadas sob a ponderacao do BOARD.

Todas as adocoes e rejeicoes do projeto foram julgadas por `compare_oof.py` na grade com thinning,
que da a `t>400` 16,5%% do peso quando o board lhe da 31,9%% (e a `t<=50` 8,1%% contra 3,9%%). Um braco
cujo ganho se concentra em `t` alto foi sistematicamente subcreditado por ~2x; um que ganha em `t`
baixo, supercreditado. Isto reroda os pares (baseline, candidato) que decidiram o projeto, nas DUAS
ponderacoes, e imprime as duas colunas lado a lado.

Custo: so bootstrap sobre OOFs que ja existem -- nenhum treino.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc
from compare_oof import paired_bootstrap_compare

M = "artifacts/models"

# (rotulo, baseline, candidato, o que foi decidido na epoca)
PARES = [
    # O par tem de ser 183-vs-183: o toco foi treinado com `--drop-prefix spec_ ord_ mrep_`, entao
    # comparar contra `b6c6_joint` (187) conflacionaria as interacoes com o ganho do BOCPD.
    ("E0c: interacoes (B6 vs toco d=1, ambos 183 feat)",
     f"{M}/oof_e0c_stump_s777.parquet", f"{M}/oof_b6_contri06_s777.parquet",
     "EXP0 §2.1: +0,0039, IC [-0,0015; +0,0100] incluia 0 => 'g saturado'"),
    ("B6: feature_contri=0,6 em meta_h0_* (part. 42)",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_b6_contri06_bag4.parquet",
     "ADOTADO (+0,0023, agregado incluia 0)"),
    ("B6: idem, particao 43",
     f"{M}/oof_c3_partB_bag4.parquet", f"{M}/oof_b6_partB_bag4.parquet",
     "ADOTADO (+0,0049, IC excluia 0)"),
    ("C6: BOCPD acrescentado ao B6",
     f"{M}/oof_b6_contri06_bag4.parquet", f"{M}/oof_b6c6_joint_bag4.parquet",
     "ligado por decisao explicita, abaixo da barra (+0,0014)"),
    ("X1-soft: pesos por detectabilidade",
     f"{M}/oof_x0_a_bag4.parquet", f"{M}/oof_x1_soft_bag4.parquet",
     "ADOTADO (+0,0040, IC excluia 0)"),
    ("B1: wt-align (massa por passo = w_t)",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_b1_wtalign_bag4.parquet",
     "rejeitado"),
    ("B5: monotone_constraints nos LLR",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_b5_mono_bag4.parquet",
     "rejeitado"),
    ("B2: rampa (t-tau) nos pesos",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_b2_ramp_bag4.parquet",
     "rejeitado"),
    ("B3: rebaixar negativos ruidosos",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_b3_noisyneg_bag4.parquet",
     "rejeitado"),
    ("C5: costura por regime",
     f"{M}/oof_x1_soft_bag4.parquet", f"{M}/oof_c5_splice_bag4.parquet",
     "adotado e RETRATADO em 24h"),
    ("V5 (BOCPD + poda) vs V4",
     f"{M}/oof_v4.parquet", f"{M}/oof_v5.parquet",
     "rejeitado (-0,0042)"),
    ("F1: calibracao recursiva vs V4",
     f"{M}/oof_v4.parquet", f"{M}/oof_f1.parquet",
     "rejeitado (-0,0069)"),
    ("F2: mismatch/portmanteau (K=4)",
     f"{M}/oof_v4_k4.parquet", f"{M}/oof_f2_k4.parquet",
     "rejeitado (-0,0024)"),
    ("A4: v4_k4 vs incumbente B6+C6",
     f"{M}/oof_b6c6_joint_bag4.parquet", f"{M}/oof_v4_k4.parquet",
     "v4_k4 lia 0,6133 > 0,6125 do incumbente"),
    ("mrep (adotado em producao) vs V4",
     f"{M}/oof_v4_k4.parquet", f"{M}/oof_mrep_k4.parquet",
     "ADOTADO (+0,0042, IC excluia 0)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="workers do bootstrap. Default BAIXO de proposito: com `-1` este script "
                         "sobe ~13 workers e, concorrendo com um treino, ja deadlocou o LightGBM "
                         "(NOTAS_AGENTES.md §7). 2-3 convivem; use -1 so com a maquina livre.")
    ap.add_argument("--only", nargs="*", default=None,
                    help="subconjunto de rotulos (casamento por substring), para retomar apos uma "
                         "interrupcao sem refazer o que ja saiu.")
    ap.add_argument("--out", default="artifacts/reports/a2_reaudit.json")
    args = ap.parse_args()

    report = []
    for label, base_p, cand_p, nota in PARES:
        if args.only and not any(o.lower() in label.lower() for o in args.only):
            continue
        if not (Path(base_p).exists() and Path(cand_p).exists()):
            print(f"[pulado] {label}: falta {base_p if not Path(base_p).exists() else cand_p}")
            continue
        b = pd.read_parquet(base_p).rename(columns={"oof_pred": "score_base"})
        c = pd.read_parquet(cand_p).rename(columns={"oof_pred": "score_cand"})
        m = b[["id", "t", "y", "score_base"]].merge(c[["id", "t", "score_cand"]], on=["id", "t"])
        if len(m) == 0:
            print(f"[pulado] {label}: grades incompativeis")
            continue
        wm = board_grid_multipliers("data/y_train.parquet", m["t"].to_numpy())

        t, y = m["t"].to_numpy(), m["y"].to_numpy()
        lv = {g: (weighted_ts_auc(t, y, m["score_base"].to_numpy(), w),
                  weighted_ts_auc(t, y, m["score_cand"].to_numpy(), w))
              for g, w in (("thin", None), ("full", wm))}

        r_thin = paired_bootstrap_compare(m, "score_base", "score_cand", n_boot=args.n_boot,
                                          n_jobs=args.n_jobs, w_mult=None)
        r_full = paired_bootstrap_compare(m, "score_base", "score_cand", n_boot=args.n_boot,
                                          n_jobs=args.n_jobs, w_mult=wm)

        print(f"\n=== {label}")
        print(f"    na epoca: {nota}")
        print(f"    niveis  thin: {lv['thin'][0]:.4f} -> {lv['thin'][1]:.4f}   "
              f"full: {lv['full'][0]:.4f} -> {lv['full'][1]:.4f}")
        print(f"    {'':12s} {'THINNING (como foi julgado)':>32s} {'BOARD (correto)':>32s}")
        for k in ["overall", "t>400"]:
            a, bb = r_thin[k], r_full[k]
            fa = "exclui 0" if a.get("excludes_zero") else "        "
            fb = "exclui 0" if bb.get("excludes_zero") else "        "
            print(f"    {k:12s} {a['point_delta']:+8.4f} [{a['ci_lo']:+.4f},{a['ci_hi']:+.4f}] {fa}"
                  f" {bb['point_delta']:+8.4f} [{bb['ci_lo']:+.4f},{bb['ci_hi']:+.4f}] {fb}")
        virou = (r_thin["overall"].get("excludes_zero") != r_full["overall"].get("excludes_zero"))
        if virou:
            print("    *** O VEREDITO MUDA DE GRADE ***")
        report.append({"label": label, "nota": nota, "niveis": lv,
                       "thin": r_thin, "full": r_full, "veredito_muda": virou})

    Path(args.out).write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nsalvo em {args.out}")
    mudaram = [r["label"] for r in report if r["veredito_muda"]]
    print(f"\nbracos cujo veredito MUDA ao corrigir a grade ({len(mudaram)}):")
    for x in mudaram:
        print("  -", x)


if __name__ == "__main__":
    main()
