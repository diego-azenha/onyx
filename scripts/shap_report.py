#!/usr/bin/env python
"""Relatório de SHAP por feature, por FAMÍLIA e por BUCKET DE t (docs/PROPOSTA_FEATURES_V2.md §5).

Usa `booster.predict(pred_contrib=True)` (TreeSHAP exato do LightGBM) sobre uma subamostra
determinística das linhas de treino, média dos folds do ensemble. Diagnóstico local de "no que o
modelo se apoia" -- NUNCA uma estimativa de TS-AUC (plano §9.0).

## Duas medidas, e por que a segunda é a que importa

**`mean_abs_shap`** = média de |contribuição| sobre TODAS as linhas. É a medida convencional -- e,
para esta competição, ela **engana**. A TS-AUC compara séries *dentro do mesmo passo t* (invariância
C1, plano §1.2): qualquer componente do score que seja constante dentro de um passo desloca todas as
séries vivas igualmente e é **exatamente neutro** para a métrica. Mas `mean_abs_shap` mistura
variação entre passos com variação entre séries, então uma feature como `meta_t` -- que varia de 1 a
1000+ entre linhas e é constante dentro de cada passo -- pontua altíssimo enquanto contribui **zero**
para o ranking transversal.

**`xs_shap`** (cross-sectional) = desvio padrão da contribuição **dentro de cada passo t**, agregado
sobre t com o mesmo peso da métrica oficial (w_t = n_pos(t)·n_neg(t)). Isola precisamente a parte da
explicação que move a TS-AUC. É esta a coluna a usar para decidir onde investir.

A decomposição por bucket de t existe para testar a assimetria batch x tempo-real da proposta §3:
uma família que só contribui em t alto (onde há janela cheia) tem valor diferente de uma que
contribui em t baixo (onde somos fracos)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.predict import ModelEnsemble

T_BUCKET_EDGES = [0, 50, 150, 400, np.inf]
T_BUCKET_LABELS = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]


def family_of(name: str) -> str:
    """Prefixo -> família. `meta_h0_*` é separado de `meta_t`/`meta_locator_*` porque são coisas
    conceitualmente diferentes (impressão digital da série vs. relógio/localizador)."""
    if name.startswith("meta_h0_"):
        return "meta_h0 (constante por serie)"
    if name.startswith("meta_"):
        return "meta_tempo/locator"
    return name.split("_", 1)[0]


def _cross_sectional_shap(signed: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Desvio padrão da contribuição DENTRO de cada passo t, agregado sobre t com o peso da métrica
    oficial (w_t = n_pos·n_neg). Componentes constantes dentro do passo dão desvio 0 -- que é
    exatamente o seu efeito sobre a TS-AUC (invariância C1).

    Implementado por `reduceat` sobre as linhas ordenadas por t (evita um groupby do pandas sobre
    uma matriz de ~10^5 x ~10^2)."""
    order = np.argsort(t, kind="stable")
    ts, s, ys = t[order], signed[order], y[order]
    starts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1]])
    counts = np.diff(np.r_[starts, len(ts)]).astype(np.float64)

    sum_ = np.add.reduceat(s, starts, axis=0)
    sum_sq = np.add.reduceat(s * s, starts, axis=0)
    mean = sum_ / counts[:, None]
    var = np.maximum(sum_sq / counts[:, None] - mean * mean, 0.0)
    sd = np.sqrt(var)  # (n_steps, n_features)

    n_pos = np.add.reduceat(ys.astype(np.float64), starts)
    w = n_pos * (counts - n_pos)  # peso oficial do passo; 0 se o passo nao tem os dois rotulos
    if w.sum() <= 0:
        return sd.mean(axis=0)
    return (sd * w[:, None]).sum(axis=0) / w.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--model", default="artifacts/models/v1")
    parser.add_argument("--sample", type=int, default=200_000, help="linhas amostradas (SHAP e caro)")
    parser.add_argument("--seed", type=int, default=None, help="default: cfg.seed")
    parser.add_argument("--out", default="artifacts/reports/shap_feature_importance.csv")
    parser.add_argument("--out-by-bucket", default="artifacts/reports/shap_by_t_bucket.csv")
    parser.add_argument("--highlight", default=None,
                        help="prefixo de familia para destacar no console (ex.: ranktwo)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    ensemble = ModelEnsemble.load(args.model)
    rows = pd.read_parquet(args.rows)

    n = len(rows)
    if args.sample and n > args.sample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, size=args.sample, replace=False))
        rows = rows.iloc[idx]
    print(f"SHAP sobre {len(rows)} linhas, {len(ensemble.feature_order)} features, "
          f"{len(ensemble.boosters)} folds", flush=True)

    feature_order = list(ensemble.feature_order)
    X = rows[feature_order].to_numpy(dtype=np.float32)

    # pred_contrib devolve (n, n_features+1): ultima coluna e o base value, descartada.
    # `signed` e necessario para a medida transversal (dispersao dentro do passo); `total` (|.|)
    # para a medida convencional.
    signed = np.zeros((len(rows), len(feature_order)), dtype=np.float64)
    total = np.zeros((len(rows), len(feature_order)), dtype=np.float64)
    for i, booster in enumerate(ensemble.boosters):
        contrib = np.asarray(
            booster.predict(X, pred_contrib=True, num_threads=cfg.lightgbm.predict_num_threads)
        )[:, :-1]
        signed += contrib
        total += np.abs(contrib)
        print(f"  fold {i} ok", flush=True)
    signed /= len(ensemble.boosters)
    total /= len(ensemble.boosters)

    mean_abs = total.mean(axis=0)
    xs = _cross_sectional_shap(signed, rows["t"].to_numpy(), rows["y"].to_numpy())
    df = pd.DataFrame({"feature": feature_order, "mean_abs_shap": mean_abs, "xs_shap": xs})
    df["family"] = df["feature"].map(family_of)
    df = df.sort_values("xs_shap", ascending=False).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    tot_abs = df["mean_abs_shap"].sum()
    tot_xs = df["xs_shap"].sum()

    print("\n=== share por familia: XS (transversal, o que move a TS-AUC) vs. convencional ===")
    g = df.groupby("family")[["xs_shap", "mean_abs_shap"]].sum().sort_values("xs_shap", ascending=False)
    print(f"  {'familia':42s} {'XS%':>7s} {'conv%':>7s}")
    for fam, r in g.iterrows():
        print(f"  {fam:42s} {100 * r['xs_shap'] / tot_xs:6.1f}% {100 * r['mean_abs_shap'] / tot_abs:6.1f}%")

    print("\n=== top 25 features por SHAP transversal ===")
    print(f"  {'feature':40s} {'XS%':>7s} {'conv%':>7s}")
    for _, r in df.head(25).iterrows():
        print(f"  {r['feature']:40s} {100 * r['xs_shap'] / tot_xs:6.2f}% "
              f"{100 * r['mean_abs_shap'] / tot_abs:6.2f}%  [{r['family']}]")

    if args.highlight:
        sub = df[df["feature"].str.startswith(args.highlight)]
        print(f"\n=== familia destacada: {args.highlight}* ===")
        if sub.empty:
            print("  (nenhuma feature com esse prefixo)")
        else:
            for _, r in sub.iterrows():
                rank = int(df.index[df["feature"] == r["feature"]][0]) + 1
                print(f"  {r['feature']:40s} {100 * r['xs_shap'] / tot_xs:6.2f}% (XS, rank {rank}/{len(df)})")
            print(f"  TOTAL da familia: {100 * sub['xs_shap'].sum() / tot_xs:.2f}% (XS)")

    # --- por bucket de t (testa a assimetria batch x tempo-real, proposta §3) ---
    idx_bucket = pd.cut(rows["t"], bins=T_BUCKET_EDGES, labels=False, right=True).to_numpy(dtype=np.int64)
    bucket_rows = []
    for b, label in enumerate(T_BUCKET_LABELS):
        mask = idx_bucket == b
        if not mask.any():
            continue
        share = _cross_sectional_shap(signed[mask], rows["t"].to_numpy()[mask], rows["y"].to_numpy()[mask])
        for feat, v in zip(feature_order, share):
            bucket_rows.append({"t_bucket": label, "feature": feat, "family": family_of(feat), "xs_shap": v})
    bdf = pd.DataFrame(bucket_rows)
    Path(args.out_by_bucket).parent.mkdir(parents=True, exist_ok=True)
    bdf.to_csv(args.out_by_bucket, index=False)

    print("\n=== share de SHAP TRANSVERSAL por familia x bucket de t (%) ===")
    pivot = bdf.pivot_table(index="family", columns="t_bucket", values="xs_shap", aggfunc="sum")
    pivot = pivot[[c for c in T_BUCKET_LABELS if c in pivot.columns]]
    pivot = 100 * pivot / pivot.sum(axis=0)
    print(pivot.round(1).to_string())

    print(f"\nsalvo em {out_path} e {args.out_by_bucket}")


if __name__ == "__main__":
    main()
