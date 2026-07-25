#!/usr/bin/env python
"""A2 (CAMPANHA_POLIMENTO.md) — réplica da métrica do board sobre a GRADE CHEIA.

O OOF vive na grade com thinning (`configs/default.yaml:thinning` — todos os t ate 100, passo 2 em
101-400, passo 4 em 401+): 399 valores de t contra os ~996 que o board avalia. A TS-AUC agregada e
`sum_t w_t * AUC_t / sum_t w_t` com `w_t = n_pos(t) * n_neg(t)`; restringir o somatorio a um
subconjunto de t NAO e neutro, porque cada t retido carrega apenas o proprio w_t. Uma regiao
subamostrada 4x entra no agregado com 1/4 da massa que o board lhe da.

Duas coisas importam e sao diferentes:
  (a) `AUC_t` medido no OOF em cada t retido e EXATO -- o thinning descarta passos, nunca series,
      entao a populacao viva em cada t retido esta inteira (verificado abaixo contra y_train).
  (b) o agregado e enviesado, porque os pesos somam sobre a grade errada.

Logo a correcao nao exige rodar nada de novo: basta reponderar os `AUC_t` que ja temos com os `w_t`
da grade cheia, interpolando `AUC_t` nos passos ausentes (AUC_t e suave em t -- o residuo dessa
interpolacao e medido aqui por holdout nos t retidos).

`data/y_train.parquet` e a grade cheia (5.036.517 linhas, todos os passos online), entao os `w_t`
verdadeiros saem dele sem depender de nenhuma feature.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def auc_by_t(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """AUC_t e w_t por passo, na grade em que `df` vive. Mann-Whitney com rank medio (empates=0,5)."""
    d = df[["t", "y", score_col]].copy()
    d["rank"] = d.groupby("t")[score_col].rank(method="average")
    g = d.groupby("t")
    n = g["y"].size()
    n_pos = g["y"].sum()
    n_neg = n - n_pos
    r_pos = d.loc[d["y"] == 1].groupby("t")["rank"].sum().reindex(n.index, fill_value=0.0)
    valid = (n_pos > 0) & (n_neg > 0)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    out = pd.DataFrame({"t": n.index, "n_pos": n_pos.values, "n_neg": n_neg.values,
                        "auc_t": auc.values, "valid": valid.values})
    out["w_t"] = (out["n_pos"] * out["n_neg"]).astype(np.float64)
    return out.loc[out["valid"]].drop(columns="valid").reset_index(drop=True)


def full_grid_weights(y_train_path: str) -> pd.DataFrame:
    """w_t = n_pos(t)*n_neg(t) sobre TODOS os passos online, direto do rotulo oficial."""
    y = pd.read_parquet(y_train_path).reset_index()
    # `time` e o indice absoluto da serie; o passo online 1-based e o rank de `time` dentro do id.
    y = y.sort_values(["id", "time"])
    y["t"] = y.groupby("id").cumcount() + 1
    g = y.groupby("t")["target"]
    n = g.size()
    n_pos = g.sum()
    n_neg = n - n_pos
    out = pd.DataFrame({"t": n.index, "n_pos": n_pos.values, "n_neg": n_neg.values})
    out["w_t"] = (out["n_pos"] * out["n_neg"]).astype(np.float64)
    return out[out["w_t"] > 0].reset_index(drop=True)


BUCKETS = [(1, 50), (51, 150), (151, 400), (401, 10 ** 9)]


def bucket_table(w: pd.DataFrame, auc_col: str | None = None) -> pd.DataFrame:
    rows = []
    tot = w["w_t"].sum()
    for lo, hi in BUCKETS:
        m = (w["t"] >= lo) & (w["t"] <= hi)
        r = {"bucket": f"{lo}-{hi if hi < 10 ** 9 else '+'}", "peso": w.loc[m, "w_t"].sum() / tot}
        if auc_col:
            ww = w.loc[m, "w_t"]
            r["ts_auc"] = float((w.loc[m, auc_col] * ww).sum() / ww.sum()) if ww.sum() > 0 else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof", nargs="+", required=True, help="parquets de OOF (id,t,y,oof_pred)")
    ap.add_argument("--y-train", default="data/y_train.parquet")
    ap.add_argument("--score-col", default="oof_pred")
    ap.add_argument("--out", default="artifacts/reports/a2_full_grid.json")
    args = ap.parse_args()

    full = full_grid_weights(args.y_train)
    print(f"grade cheia: {len(full)} passos com w_t>0, t in [{full['t'].min()}, {full['t'].max()}]")

    report: dict = {"grade_cheia_n_t": int(len(full))}

    for path in args.oof:
        name = Path(path).stem
        oof = pd.read_parquet(path)
        a = auc_by_t(oof, args.score_col)
        thin_ts = a["t"].values
        print(f"\n=== {name} ===")
        print(f"grade do OOF: {len(a)} passos")

        # --- (a) a populacao viva em cada t retido esta inteira? ---
        chk = full.merge(a[["t", "n_pos", "n_neg"]], on="t", suffixes=("_full", "_oof"))
        dp = (chk["n_pos_full"] - chk["n_pos_oof"]).abs().max()
        dn = (chk["n_neg_full"] - chk["n_neg_oof"]).abs().max()
        print(f"populacao por passo, OOF vs rotulo oficial: max|dn_pos|={dp}, max|dn_neg|={dn}")

        # --- TS-AUC como e medida hoje (grade com thinning) ---
        ts_thin = float((a["auc_t"] * a["w_t"]).sum() / a["w_t"].sum())

        # --- TS-AUC na grade cheia: mesmos AUC_t, pesos do board, AUC_t interpolado onde falta ---
        auc_full = np.interp(full["t"].values, a["t"].values, a["auc_t"].values)
        f = full.copy()
        f["auc_t"] = auc_full
        ts_full = float((f["auc_t"] * f["w_t"]).sum() / f["w_t"].sum())

        # --- residuo da interpolacao, medido por holdout nos proprios t retidos ---
        # esconde os t de indice impar da grade retida, interpola dos vizinhos, compara.
        idx = np.arange(len(a))
        hid, kep = idx[1::2], idx[0::2]
        pred = np.interp(a["t"].values[hid], a["t"].values[kep], a["auc_t"].values[kep])
        err = np.abs(pred - a["auc_t"].values[hid])
        w_h = a["w_t"].values[hid]
        err_w = float((err * w_h).sum() / w_h.sum())
        print(f"erro da interpolacao de AUC_t (holdout nos t retidos): medio ponderado {err_w:.5f}, "
              f"max {err.max():.5f}")

        print(f"TS-AUC grade THINNING (como e medida hoje): {ts_thin:.4f}")
        print(f"TS-AUC grade CHEIA   (como o board mede)  : {ts_full:.4f}")
        print(f"delta (cheia - thinning): {ts_full - ts_thin:+.4f}")

        print("\npesos por bucket:")
        bt_thin = bucket_table(a, "auc_t").rename(columns={"peso": "peso_thin", "ts_auc": "auc_thin"})
        bt_full = bucket_table(f, "auc_t").rename(columns={"peso": "peso_full", "ts_auc": "auc_full"})
        bt = bt_thin.merge(bt_full, on="bucket")
        bt["razao_peso"] = bt["peso_full"] / bt["peso_thin"]
        print(bt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        report[name] = {
            "ts_auc_thin": ts_thin, "ts_auc_full": ts_full, "delta": ts_full - ts_thin,
            "interp_err_w": err_w, "interp_err_max": float(err.max()),
            "n_t_oof": int(len(a)), "max_dn_pos": int(dp), "max_dn_neg": int(dn),
            "buckets": bt.to_dict(orient="records"),
        }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nrelatorio salvo em {args.out}")


if __name__ == "__main__":
    main()
