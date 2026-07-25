#!/usr/bin/env python
"""Exp0 (DIAGNOSTICO_ESTRUTURAL.md §5): junta as quatro medições numa tabela só e aplica a tabela de
decisão registrada antes da execução.

Compara, TODAS na MESMA grade de linhas (id,t) — a grade do OOF do incumbente, senão a comparação
mede a grade e não o score:

  E0a/E0b  o teto informacional (scripts/e0_skyline.py: oráculo de τ e scan causal)
  E0c      o toco (GBM de profundidade 1: a base sem NENHUMA interação)
  E0d      o fallback puro (estatísticas cruas, sem aprendizado)
  ref      o incumbente e o que mais for passado em --extra

Decompõe por bucket de t e, para os positivos, por tercil de MAGNITUDE da quebra (divergência L2 dos
eixos do censo A1, idêntica a model/detectability.py) — mantendo TODOS os negativos em cada tercil,
porque a AUC é uma comparação par a par: restringir positivos por magnitude e comparar contra o mesmo
conjunto de negativos responde "quanto do teto vem das quebras grandes".
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import weighted_ts_auc

T_EDGES = [0, 50, 150, 400, np.inf]
T_LABELS = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]
AXES = ("delta_logvar_e", "delta_rho1", "delta_kurt", "delta_exceed")


def _row(df: pd.DataFrame, col: str, bucket: pd.Series) -> dict:
    t, y, s = df["t"].to_numpy(), df["y"].to_numpy(), df[col].to_numpy(np.float64)
    out = {"geral": weighted_ts_auc(t, y, s)}
    for lbl in T_LABELS:
        m = (bucket == lbl).to_numpy()
        out[lbl] = weighted_ts_auc(t[m], y[m], s[m]) if m.any() else np.nan
    return out


def _print_table(title: str, rows: dict) -> None:
    print(f"\n{title}")
    print(f"  {'score':<22s}" + "".join(f"{c:>13s}" for c in ["geral", *T_LABELS]))
    for name, vals in rows.items():
        print(f"  {name:<22s}" + "".join(f"{vals[c]:>13.4f}" for c in ["geral", *T_LABELS]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skyline", default="artifacts/models/oof_e0_skyline.parquet")
    ap.add_argument("--incumbent", default="artifacts/models/oof_b6_contri06_s777.parquet")
    ap.add_argument("--fallback", default="artifacts/models/oof_e0d_fallback.parquet")
    ap.add_argument("--stump", default="artifacts/models/oof_e0c_stump_s777.parquet")
    ap.add_argument("--census", default="artifacts/reports/break_type_census.csv")
    ap.add_argument("--extra", nargs="*", default=[], metavar="NOME=CAMINHO")
    args = ap.parse_args()

    base = pd.read_parquet(args.incumbent).rename(columns={"oof_pred": "incumbente"})
    print(f"grade de referência: {args.incumbent} ({len(base)} linhas, {base['id'].nunique()} séries)")

    named = [("fallback (E0d)", args.fallback), ("toco d=1 (E0c)", args.stump)]
    named += [(s.split("=", 1)[0], s.split("=", 1)[1]) for s in args.extra]
    for name, path in named:
        try:
            d = pd.read_parquet(path)[["id", "t", "oof_pred"]].rename(columns={"oof_pred": name})
        except (FileNotFoundError, OSError):
            print(f"  [ausente] {name}: {path}")
            continue
        base = base.merge(d, on=["id", "t"], how="left")

    sky = pd.read_parquet(args.skyline)
    base = base.merge(sky.drop(columns=["y"]), on=["id", "t"], how="left")
    score_cols = [c for c in base.columns if c not in ("id", "t", "y")]
    miss = base[score_cols].isna().sum()
    if miss.any():
        print(f"  aviso: NaN por coluna -> {dict(miss[miss > 0])} (preenchidos com a mediana)")
        base[score_cols] = base[score_cols].fillna(base[score_cols].median())

    bucket = pd.cut(base["t"], bins=T_EDGES, labels=T_LABELS, right=True).astype(object)
    _print_table("TS-AUC por bucket de t (mesma grade de linhas para todos)",
                 {c: _row(base, c, bucket) for c in score_cols})

    # --- magnitude: positivos por tercil de divergência, negativos inteiros em cada tercil --------
    try:
        cen = pd.read_csv(args.census)
    except (FileNotFoundError, OSError):
        print("\n(censo ausente: pulei a decomposição por magnitude)")
        cen = None
    if cen is not None:
        z = np.zeros(len(cen))
        for ax in AXES:
            v = cen[ax].to_numpy(np.float64)
            sd = np.nanstd(v)
            if sd > 0:
                z += np.nan_to_num((v / sd) ** 2)
        cen["divergence"] = np.sqrt(z)
        q = cen["divergence"].quantile([1 / 3, 2 / 3]).to_numpy()
        cen["mag"] = np.where(cen["divergence"] <= q[0], "baixa",
                              np.where(cen["divergence"] <= q[1], "média", "alta"))
        mag_of = dict(zip(cen["id"].astype(int), cen["mag"]))
        base["mag"] = base["id"].map(mag_of)
        neg = base[base["y"] == 0]
        print("\nTS-AUC com os positivos restritos por magnitude da quebra (censo A1; "
              "negativos sempre inteiros)")
        print(f"  {'score':<22s}" + "".join(f"{c:>13s}" for c in ["baixa", "média", "alta"]))
        for c in score_cols:
            cells = []
            for m in ("baixa", "média", "alta"):
                sub = pd.concat([neg, base[(base["y"] == 1) & (base["mag"] == m)]])
                cells.append(weighted_ts_auc(sub["t"].to_numpy(), sub["y"].to_numpy(),
                                             sub[c].to_numpy(np.float64)))
            print(f"  {c:<22s}" + "".join(f"{v:>13.4f}" for v in cells))

    # --- tabela de decisão registrada em §5 -------------------------------------------------------
    if "e0a_aug" in base.columns:
        e0a = weighted_ts_auc(base["t"].to_numpy(), base["y"].to_numpy(),
                              base["e0a_aug"].to_numpy(np.float64))
        e0b = weighted_ts_auc(base["t"].to_numpy(), base["y"].to_numpy(),
                              base["e0b_scan"].to_numpy(np.float64))
        print("\n--- tabela de decisão (§5, registrada antes da execução) ---")
        print(f"  E0a (aug) = {e0a:.4f} · E0b = {e0b:.4f}")
        if e0a <= 0.66:
            print("  => TETO REAL: o platô é da tarefa. Caminho B (§7-B): consistência.")
        elif e0a >= 0.72 and e0b >= 0.68:
            print("  => GARGALO DE EXTRAÇÃO: as portas da §6 recebem as semanas restantes (caminho A).")
        else:
            print("  => ZONA INTERMEDIÁRIA: decidir por bucket; 150–400 é 49% do peso.")


if __name__ == "__main__":
    main()
