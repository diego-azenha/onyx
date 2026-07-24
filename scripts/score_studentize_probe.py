#!/usr/bin/env python
"""X3 (BRAINSTORM_RUPTURA_TSAUC.md §3): studentização DISTRIBUCIONAL do score por série -- sonda
offline, sem retreino. Morreu a centragem ADITIVA (subtrair o nível, −0,0014 a −0,0018). Aqui a
transformação é pela CDF NULA da própria série: cada score online vira o percentil dele na
distribuição H0 da série (equaliza escala E forma, não só nível). Por-série ⇒ reordena transversal
(não é a recalibração global C1-neutra).

Reusa `artifacts/reports/centering_baselines.parquet` (score CRU do modelo sobre a cauda H0 do
histórico, t=1..200, 10k séries) -- o mesmo passe de baseline que a frente de centragem já pagou.
Null por série = baseline_raw no regime assentado (t>=`null_t_min`, sem warm-up). Gate registrado:
corr_xs(raw_online, studentizado) < 0,95 (>0,95 = não reordena, morta). Kill: gate, ou Δ≤0 pareado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import weighted_ts_auc
from sbrt.model.base_rate import predict_base_rate_logit


def _corr_xs(df: pd.DataFrame, a: str, b: str, at_t=(30, 60, 120)) -> float:
    cs = []
    for t in at_t:
        s = df[df["t"] == t]
        if len(s) > 30:
            cs.append(abs(np.corrcoef(s[a], s[b])[0, 1]))
    return float(np.mean(cs)) if cs else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baselines", default="artifacts/reports/centering_baselines.parquet")
    p.add_argument("--oof", default="artifacts/models/oof_x0_a_bag4.parquet")
    p.add_argument("--model", default="artifacts/models/b183_s777", help="dir com base_rate_curve.json")
    p.add_argument("--null-t-min", type=int, default=50, help="usa baseline_raw com t>=isto como nulo (evita warm-up)")
    p.add_argument("--out", default="artifacts/models/oof_x3_studentized.parquet")
    args = p.parse_args()

    bl = pd.read_parquet(args.baselines)
    bl = bl[bl["t"] >= args.null_t_min]
    null_by_id = {i: np.sort(g.to_numpy(dtype=np.float64)) for i, g in bl.groupby("id")["baseline_raw"]}

    base_rate_curve = json.loads((Path(args.model) / "base_rate_curve.json").read_text(encoding="utf-8"))
    oof = pd.read_parquet(args.oof).copy()
    pcl = oof["oof_pred"].to_numpy(dtype=np.float64).clip(1e-9, 1 - 1e-9)
    oof["raw_online"] = np.log(pcl / (1 - pcl)) - predict_base_rate_logit(oof["t"].to_numpy(), base_rate_curve)

    # studentização: percentil de raw_online na CDF nula da própria série
    stud = np.full(len(oof), np.nan, dtype=np.float64)
    ids = oof["id"].to_numpy()
    raw = oof["raw_online"].to_numpy(dtype=np.float64)
    for i, idx in oof.groupby("id").indices.items():
        nn = null_by_id.get(i)
        if nn is None or len(nn) == 0:
            stud[idx] = raw[idx]  # sem nulo: mantém (neutro)
            continue
        stud[idx] = np.searchsorted(nn, raw[idx], side="right") / len(nn)
    oof["studentized"] = stud

    corr = _corr_xs(oof, "raw_online", "studentized")
    print(f"[X3] corr_xs(raw_online, studentizado) = {corr:.3f}  "
          f"({'GATE FALHA (>0,95, não reordena)' if corr > 0.95 else 'GATE PASSA (<0,95) — vale medir'})")

    a0 = weighted_ts_auc(oof["t"].to_numpy(), oof["y"].to_numpy(), oof["oof_pred"].to_numpy())
    a1 = weighted_ts_auc(oof["t"].to_numpy(), oof["y"].to_numpy(), oof["studentized"].to_numpy())
    print(f"[X3] TS-AUC baseline={a0:.4f}  studentizado={a1:.4f}  Δpontual={a1-a0:+.4f}")

    out = oof[["id", "t", "y"]].copy()
    out["oof_pred"] = oof["studentized"].to_numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[X3] OOF studentizada salva em {args.out} — veredito por compare_oof.py (pareado).")


if __name__ == "__main__":
    main()
