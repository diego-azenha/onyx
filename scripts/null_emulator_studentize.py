#!/usr/bin/env python
"""A3 (BRAINSTORM_RUPTURA_V2.md §1.4): studentização do score contra um emulador de nulo com ALVO
H0-REAL. O X3 morreu (−0,0294) porque o nulo vinha da cauda do histórico (viés de regime). Aqui o alvo
do nulo são as linhas H0 do ONLINE real (t<τ + séries sem quebra) — milhões de amostras no regime
certo. Ajusta um regressor quantílico (LGBM pinball, q10/q50/q90) do score-nulo em função de (meta_h0, t),
OUT-OF-FOLD (emulador dos folds≠k aplicado ao fold k), e studentiza:

    studentizado = (raw_online − q50_nulo) / (q90_nulo − q10_nulo)

que remove a LOCAÇÃO e a ESCALA do nulo por série (o que a centragem aditiva não tocava), é monótono em
raw_online (preserva ordem, sem empate de cauda) e só reordena transversalmente (não é a recalibração
global C1-neutra). Seleção de linhas H0 por τ = informação privilegiada no TREINO do emulador (licença
do X1). CLÁUSULA: se falhar, o eixo de comparabilidade autorreferente fecha com prejuízo (3ª morte, nulo
certo). Diagnóstico offline; veredito final por `compare_oof.py`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc
from sbrt.model.base_rate import predict_base_rate_logit

_QUANTILES = (0.1, 0.5, 0.9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--oof", default="artifacts/models/oof_x1_soft_bag4.parquet")
    p.add_argument("--rows", default="data/processed/train_rows_3eixos.parquet")
    p.add_argument("--model", default="artifacts/models/b183_s777", help="dir com base_rate_curve.json")
    p.add_argument("--fit-sample", type=int, default=300000, help="linhas H0 amostradas por fold p/ ajustar o emulador")
    p.add_argument("--out", default="artifacts/models/oof_A3_studentized.parquet")
    args = p.parse_args()

    cfg = load_config(args.config)
    oof = pd.read_parquet(args.oof).reset_index(drop=True)

    # raw_online = logit(oof_pred) − init_score (mesmo espaço do X3/centragem)
    base_rate = json.loads((Path(args.model) / "base_rate_curve.json").read_text(encoding="utf-8"))
    pcl = oof["oof_pred"].to_numpy(np.float64).clip(1e-9, 1 - 1e-9)
    oof["raw"] = np.log(pcl / (1 - pcl)) - predict_base_rate_logit(oof["t"].to_numpy(), base_rate)

    # meta_h0 por série (constantes) do parquet de features -- lê só o schema para achar as colunas
    import pyarrow.parquet as pq
    meta_cols = [c for c in pq.ParquetFile(args.rows).schema_arrow.names if c.startswith("meta_h0")]
    meta = pd.read_parquet(args.rows, columns=["id"] + meta_cols).groupby("id").first().reset_index()
    df = oof.merge(meta, on="id", how="left")
    feat = meta_cols + ["t"]
    X = df[feat].to_numpy(np.float32)
    raw = df["raw"].to_numpy(np.float64)

    folds = list(grouped_stratified_kfold(oof, cfg.lightgbm.n_folds, cfg.seed))
    qpred = {q: np.full(len(df), np.nan) for q in _QUANTILES}
    rng = np.random.default_rng(cfg.seed)
    for k, (train_idx, valid_idx) in enumerate(folds):
        h0 = train_idx[df["y"].to_numpy()[train_idx] == 0]  # linhas H0 do treino (t<τ + controles)
        if len(h0) > args.fit_sample:
            h0 = rng.choice(h0, size=args.fit_sample, replace=False)
        dtrain = lgb.Dataset(X[h0], label=raw[h0])
        for q in _QUANTILES:
            b = lgb.train({"objective": "quantile", "alpha": q, "num_leaves": 63, "learning_rate": 0.05,
                           "min_data_in_leaf": 200, "verbose": -1, "seed": 42}, dtrain, num_boost_round=200)
            qpred[q][valid_idx] = b.predict(X[valid_idx])
        print(f"  fold {k}: emulador ajustado em {len(h0)} linhas H0")

    spread = (qpred[0.9] - qpred[0.1])
    df["studentized"] = (raw - qpred[0.5]) / np.where(spread > 1e-6, spread, 1e-6)

    a0 = weighted_ts_auc(df["t"].to_numpy(), df["y"].to_numpy(), df["oof_pred"].to_numpy())
    a1 = weighted_ts_auc(df["t"].to_numpy(), df["y"].to_numpy(), df["studentized"].to_numpy())
    print(f"\n[A3] TS-AUC baseline={a0:.4f}  studentizado={a1:.4f}  Δpontual={a1-a0:+.4f}")

    out = df[["id", "t", "y"]].copy()
    out["oof_pred"] = df["studentized"].to_numpy()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[A3] OOF studentizada salva em {args.out} — veredito por compare_oof.py (CLÁUSULA de fechamento).")


if __name__ == "__main__":
    main()
