#!/usr/bin/env python
"""Mede todos os bracos `p*_bag2` da triagem contra o incumbente, na grade do BOARD (A2).

Descobre sozinho quais bracos ja existem, entao pode rodar a qualquer momento durante a campanha --
inclusive depois de uma interrupcao. Imprime nivel, Delta e IC por bucket, ordenado por Delta.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc
from compare_oof import paired_bootstrap_compare

M = Path("artifacts/models")
BARRA = 0.0014  # a barra de decisao do projeto


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", default=str(M / "oof_b6c6_joint_bag2.parquet"))
    ap.add_argument("--glob", default="oof_p*_bag2.parquet")
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=2,
                    help="workers do bootstrap. Default BAIXO: com `-1` este script disputa a maquina "
                         "com um treino em curso e a contencao ja deadlocou o LightGBM duas vezes "
                         "(NOTAS_AGENTES.md §7). Use -1 so com a maquina livre.")
    ap.add_argument("--out", default="artifacts/reports/polimento_triagem.json")
    args = ap.parse_args()

    base = pd.read_parquet(args.baseline).rename(columns={"oof_pred": "score_base"})
    wm = board_grid_multipliers("data/y_train.parquet", base["t"].to_numpy())
    t, y = base["t"].to_numpy(), base["y"].to_numpy()
    lvl_base = weighted_ts_auc(t, y, base["score_base"].to_numpy(), wm)
    print(f"baseline: {Path(args.baseline).stem}  TS-AUC (grade do board) = {lvl_base:.5f}\n")

    out = []
    for p in sorted(M.glob(args.glob)):
        cand = pd.read_parquet(p).rename(columns={"oof_pred": "score_cand"})
        m = base[["id", "t", "y", "score_base"]].merge(cand[["id", "t", "score_cand"]], on=["id", "t"])
        lvl = weighted_ts_auc(t, y, m["score_cand"].to_numpy(), wm)
        r = paired_bootstrap_compare(m, "score_base", "score_cand", n_boot=args.n_boot,
                                     n_jobs=args.n_jobs, w_mult=wm)
        o = r["overall"]
        veredito = ("ADOTAR" if o["excludes_zero"] and o["point_delta"] > 0 else
                    "REJEITAR" if o["excludes_zero"] else "inconclusivo")
        out.append({"braco": p.stem, "nivel": lvl, "delta": o["point_delta"],
                    "ci": [o["ci_lo"], o["ci_hi"]], "veredito": veredito, "buckets": r})
        print(f"=== {p.stem}")
        print(f"    nivel {lvl:.5f} ({lvl - lvl_base:+.5f} vs baseline)")
        for k in ["overall", "t<=50", "50<t<=150", "150<t<=400", "t>400"]:
            b = r[k]
            flag = "exclui 0" if b.get("excludes_zero") else ""
            print(f"    {k:12s} {b['point_delta']:+8.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] {flag}")
        print(f"    -> {veredito} (barra {BARRA:+.4f}; ganho = {o['point_delta'] / BARRA:.1f}x a barra)\n")

    out.sort(key=lambda d: -d["delta"])
    print("=== RANKING (grade do board) ===")
    for d in out:
        print(f"  {d['delta']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]  {d['braco']:28s} {d['veredito']}")
    Path(args.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"\nsalvo em {args.out}")


if __name__ == "__main__":
    main()
