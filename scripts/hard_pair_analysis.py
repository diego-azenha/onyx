#!/usr/bin/env python
"""T0.1 (BRAINSTORM_RUPTURA_V2.md §1.3): mineração de pares confusáveis. A TS-AUC é a fração GLOBAL de
pares concordantes sobre todos os triplos (t, positivo, negativo) — logo a perda é uma soma sobre pares
invertidos, e pode estar concentrada em poucos negativos "quentes". Este script mede essa concentração
e caracteriza os culpados. Diagnóstico OFFLINE, sem retreino; des-arrisca B3/C5/B6.

(a) NEGATIVOS QUENTES — massa de pares invertidos por série. Para cada linha y=0 no passo t, quantos
    positivos do mesmo t ela supera (score do negativo >= score do positivo = par invertido). Soma por
    série ⇒ ranking + curva de Lorenz. Se o topo carrega muito da perda, existe alvo pequeno.
(b) SUSPEITA DE RÓTULO (só nos quentes, e só de estatística ANTICAUSAL independente — nunca do score,
    anti-circularidade): divergência entre as duas metades do online (|Δmédia|/σ + |Δlogvar|). Um
    "negativo" que o modelo insiste em pontuar alto E que muda de regime no meio é candidato a rótulo
    errado (o changelog W23/2026 já corrigiu 29 séries mal-rotuladas).

Saída: `artifacts/reports/hard_pairs_negcarga.csv` (todas as séries y=0) e, com --with-labels,
`artifacts/reports/hard_pairs_flags.csv` (top quentes + divergência anticausal)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def inverted_mass_per_series(oof: pd.DataFrame) -> pd.DataFrame:
    """Para cada linha y=0, nº de positivos do MESMO t com score <= score do negativo (par invertido).
    Agrega por série. Vetorizado por passo com searchsorted."""
    inv = np.zeros(len(oof), dtype=np.float64)
    y = oof["y"].to_numpy()
    s = oof["oof_pred"].to_numpy(dtype=np.float64)
    idx_by_t = oof.groupby("t").indices
    for t, idx in idx_by_t.items():
        yy, ss = y[idx], s[idx]
        pos = np.sort(ss[yy == 1])
        if len(pos) == 0:
            continue
        neg_local = np.where(yy == 0)[0]
        if len(neg_local) == 0:
            continue
        counts = np.searchsorted(pos, ss[neg_local], side="right")  # positivos <= score do neg (local)
        inv[idx[neg_local]] = counts
    oof = oof.assign(inv_mass=inv)
    per = oof[oof["y"] == 0].groupby("id")["inv_mass"].sum().rename("inv_mass")
    has_break = oof.groupby("id")["y"].max().rename("has_break")
    n_neg_rows = oof[oof["y"] == 0].groupby("id").size().rename("n_neg_rows")
    return pd.concat([per, has_break, n_neg_rows], axis=1).reset_index().sort_values("inv_mass", ascending=False)


def _anticausal_divergence(x: np.ndarray) -> float:
    if len(x) < 40:
        return np.nan
    h1, h2 = x[: len(x) // 2], x[len(x) // 2:]
    sd = h1.std(ddof=1)
    dmean = abs(h2.mean() - h1.mean()) / max(sd, 1e-8)
    dlogvar = abs(np.log(max(h2.var(ddof=1), 1e-12)) - np.log(max(h1.var(ddof=1), 1e-12)))
    return float(np.sqrt(dmean ** 2 + dlogvar ** 2))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--oof", default="artifacts/models/oof_x1_soft_bag4.parquet")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--top", type=int, default=300, help="quantos negativos quentes caracterizar em (b)")
    p.add_argument("--with-labels", action="store_true", help="fase (b): carrega X_train e mede divergência anticausal")
    p.add_argument("--out-dir", default="artifacts/reports")
    args = p.parse_args()

    oof = pd.read_parquet(args.oof)
    per = inverted_mass_per_series(oof)
    total = per["inv_mass"].sum()

    # Lorenz: fração da massa nos negativos mais quentes
    m = per["inv_mass"].to_numpy()
    cum = np.cumsum(m) / total
    print(f"massa total de pares invertidos: {total:.3g}  ({int((oof['y']==0).sum())} linhas y=0, "
          f"{len(per)} séries)")
    for frac in (0.001, 0.005, 0.01, 0.05, 0.10):
        k = max(1, int(frac * len(per)))
        print(f"  top {frac*100:>4.1f}% séries ({k:5d}) carregam {cum[k-1]*100:5.1f}% da perda")
    ctrl = per[per["has_break"] == 0]
    print(f"\ndestas, {len(ctrl)} são CONTROLES (sem quebra); os {min(args.top,len(ctrl))} mais quentes "
          f"controles carregam {ctrl['inv_mass'].head(args.top).sum()/total*100:.1f}% da perda total")

    out_dir = Path(args.out_dir)
    per.to_csv(out_dir / "hard_pairs_negcarga.csv", index=False)
    print(f"\nsalvo: {out_dir/'hard_pairs_negcarga.csv'}")

    if args.with_labels:
        top_ctrl = ctrl.head(args.top)["id"].astype(int).tolist()
        print(f"\n(b) carregando online de {len(top_ctrl)} controles quentes para divergência anticausal...")
        X = pd.read_parquet(Path(args.data_dir) / "X_train.parquet")
        rows = []
        for cid in top_ctrl:
            g = X.loc[cid].reset_index()
            online = g.loc[g["period"] == 2, "value"].to_numpy(dtype="float64")
            rows.append({"id": cid, "anticausal_div": _anticausal_divergence(online), "n_online": len(online)})
        flags = per.merge(pd.DataFrame(rows), on="id", how="right")
        thr = np.nanquantile(flags["anticausal_div"], 0.8)
        flags["label_suspect"] = flags["anticausal_div"] > thr
        flags.sort_values("inv_mass", ascending=False).to_csv(out_dir / "hard_pairs_flags.csv", index=False)
        n_flag = int(flags["label_suspect"].sum())
        print(f"flag (div anticausal > p80={thr:.2f} entre os quentes): {n_flag}/{len(flags)} controles quentes")
        print(f"  = {n_flag/oof['id'].nunique()*100:.2f}% do total de séries (gate B3: >0,5%?)")
        print(f"salvo: {out_dir/'hard_pairs_flags.csv'}")


if __name__ == "__main__":
    main()
