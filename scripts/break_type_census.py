#!/usr/bin/env python
"""A1 (plano_acao_v1_para_v2.md §4): censo de tipos de quebra, model-free. Para cada série
rotulada com quebra, mede o que DE FATO muda em tau — sem depender de nenhum modelo treinado ou do
comportamento do scorer supervisionado. Usa apenas a fase-histórico (fit_h0) e o whitening causal
(whiten_step) já implementados, aplicados uma vez sobre o segmento online completo de cada série
(não é o motor único de inferência — aqui rodamos post-hoc sobre dados já rotulados para medir a
verdade dos dados, não para pontuar)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats
from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.state.h0 import fit_h0, seed_lag_buffer, whiten_step

MIN_SEGMENT = 10  # pontos mínimos de cada lado de tau para uma estimativa de delta minimamente estável


def _acf1(x: np.ndarray) -> float:
    if len(x) < 3:
        return np.nan
    xc = x - x.mean()
    den = np.dot(xc, xc)
    return float(np.dot(xc[:-1], xc[1:]) / den) if den > 0 else np.nan


def analyze_series(x_hist: np.ndarray, x_online: np.ndarray, tau_index: int, cfg) -> dict | None:
    if tau_index < MIN_SEGMENT or (len(x_online) - tau_index) < MIN_SEGMENT:
        return None

    h0 = fit_h0(x_hist, cfg)
    lags = seed_lag_buffer(h0)
    e_vals = np.empty(len(x_online))
    for i, x in enumerate(x_online):
        e, _ = whiten_step(float(x), lags, h0, cfg)
        e_vals[i] = e

    pre_x, post_x = x_online[:tau_index], x_online[tau_index:]
    pre_e, post_e = e_vals[:tau_index], e_vals[tau_index:]

    pre_x_std = pre_x.std(ddof=1) if len(pre_x) > 1 else 1.0
    pre_e_std = pre_e.std(ddof=1) if len(pre_e) > 1 else 1.0
    delta_mean_x = (post_x.mean() - pre_x.mean()) / max(pre_x_std, 1e-8)
    delta_mean_e = (post_e.mean() - pre_e.mean()) / max(pre_e_std, 1e-8)

    var_pre_x = max(pre_x.var(ddof=1), 1e-12)
    var_post_x = max(post_x.var(ddof=1), 1e-12)
    var_pre_e = max(pre_e.var(ddof=1), 1e-12)
    var_post_e = max(post_e.var(ddof=1), 1e-12)
    delta_logvar_x = float(np.log(var_post_x) - np.log(var_pre_x))
    delta_logvar_e = float(np.log(var_post_e) - np.log(var_pre_e))

    delta_rho1 = _acf1(post_e) - _acf1(pre_e)
    kurt_pre = spstats.kurtosis(pre_e, fisher=True, bias=False) if len(pre_e) > 3 else np.nan
    kurt_post = spstats.kurtosis(post_e, fisher=True, bias=False) if len(post_e) > 3 else np.nan
    delta_kurt = float(kurt_post - kurt_pre)
    delta_exceed = float(np.mean(np.abs(post_e) > 2) - np.mean(np.abs(pre_e) > 2))

    return {
        "delta_mean_x": float(delta_mean_x),
        "delta_mean_e": float(delta_mean_e),
        "delta_logvar_x": delta_logvar_x,
        "delta_logvar_e": delta_logvar_e,
        "delta_rho1": float(delta_rho1),
        "delta_kurt": delta_kurt,
        "delta_exceed": delta_exceed,
        "sum_phi": float(np.sum(h0.phi)),
        "ar_r2": float(h0.ar_r2),
        "n_pre": len(pre_x),
        "n_post": len(post_x),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts/reports/break_type_census.csv")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)

    X = pd.read_parquet(data_dir / "X_train.parquet")
    y_index = pd.read_parquet(data_dir / "y_train_index.parquet")

    rows = []
    skipped = 0
    ids = list(X.index.get_level_values("id").unique())
    for dataset_id in tqdm(ids, desc="censo de tipos de quebra"):
        tau = int(y_index.loc[dataset_id, "tau_index"])
        if tau < 0:
            continue
        group = X.loc[dataset_id].reset_index()
        hist_vals = group.loc[group["period"] == 1, "value"].to_numpy(dtype="float64")
        online_vals = group.loc[group["period"] == 2, "value"].to_numpy(dtype="float64")

        result = analyze_series(hist_vals, online_vals, tau, cfg)
        if result is None:
            skipped += 1
            continue
        result["id"] = int(dataset_id)
        result["tau_index"] = tau
        rows.append(result)

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{len(df)} séries analisadas ({skipped} puladas por segmento curto < {MIN_SEGMENT})")
    print("\nMedianas dos deltas:")
    cols = ["delta_mean_x", "delta_mean_e", "delta_logvar_x", "delta_logvar_e", "delta_rho1", "delta_kurt", "delta_exceed", "sum_phi"]
    print(df[cols].median().to_string())
    print("\nFração com |delta_mean_e| > 0.3 (shift de média não-trivial nas inovações):",
          round((df["delta_mean_e"].abs() > 0.3).mean(), 3))
    print("Fração com |delta_logvar_e| > 0.3 (shift de variância não-trivial):",
          round((df["delta_logvar_e"].abs() > 0.3).mean(), 3))
    print("\nCorrelação entre eixos:")
    print(df[cols].corr().round(2).to_string())
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
