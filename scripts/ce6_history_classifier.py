#!/usr/bin/env python
"""A3 / CE6 (plano_structural_break_realtime.md §12.2, nunca executado; plano_acao_v1_para_v2.md
§4 A3): treina um classificador usando SÓ as features do H0 (nenhum ponto online) para prever se a
série tem quebra em algum momento, via CV padrão. Decide o destino das features `meta_h0_*`:

- Se prevê melhor que a taxa-base (~50%) -> o gerador vaza o rótulo pelo histórico; meta_h0_* fazem
  trabalho legítimo (mesmo como efeito principal) e a política de não explorar deliberadamente
  (§12.2) precisa ser reavaliada explicitamente com o usuário.
- Se não prevê nada -> qualquer uso de meta_h0_* como EFEITO PRINCIPAL (não interação) injeta um
  offset por série sem informação sobre quebra -- ativamente prejudicial para o ranking transversal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.state.h0 import fit_h0

_BASE_NAMES = [
    "mu0", "sigma0", "sigma_e", "sigma_e_rob", "nu_hat", "rho1_e", "rho1_abs_e",
    "ar_r2", "n_h", "sigma_u", "q01", "q05", "q10", "q25", "q75", "q90", "q95", "q99", "has_seasonal",
]
# F2 (docs/PROPOSTA_FEATURES_V2.md): a impressão digital estendida também é função APENAS do
# histórico, então entra obrigatoriamente nesta checagem -- de nada adianta CE6 continuar ≈0,5 sobre
# as 19 features antigas se as 9 novas vazarem o rótulo.
_FINGERPRINT_NAMES = [
    "hurst", "hill_xi", "acf_e2_l1", "acf_abs_mass", "acf_decay",
    "spectral_slope", "ljungbox_abs", "volvol", "iqr_tail_ratio",
]
FEATURE_NAMES = _BASE_NAMES + [f"fp_{n}" for n in _FINGERPRINT_NAMES]


def h0_to_row(h0) -> list:
    q = h0.q
    base = [
        h0.mu0, h0.sigma0, h0.sigma_e, h0.sigma_e_rob, h0.nu_hat, h0.rho1_e, h0.rho1_abs_e,
        h0.ar_r2, float(h0.n_h), h0.sigma_u,
        q["0.01"], q["0.05"], q["0.10"], q["0.25"], q["0.75"], q["0.90"], q["0.95"], q["0.99"],
        1.0 if h0.seasonal_lag is not None else 0.0,
    ]
    return base + [float(h0.fingerprint.get(n, 0.0)) for n in _FINGERPRINT_NAMES]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out", default="artifacts/reports/ce6_history_classifier.csv")
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(args.data_dir)

    X_raw = pd.read_parquet(data_dir / "X_train.parquet")
    y_index = pd.read_parquet(data_dir / "y_train_index.parquet")

    rows, labels, ids = [], [], []
    for dataset_id in tqdm(list(X_raw.index.get_level_values("id").unique()), desc="ajustando H0 por série (CE6)"):
        group = X_raw.loc[dataset_id]
        hist_vals = group.loc[group["period"] == 1, "value"].to_numpy(dtype="float64")
        h0 = fit_h0(hist_vals, cfg)
        rows.append(h0_to_row(h0))
        tau = int(y_index.loc[dataset_id, "tau_index"])
        labels.append(1 if tau >= 0 else 0)
        ids.append(dataset_id)

    X = np.array(rows, dtype=np.float64)
    y = np.array(labels, dtype=np.int32)

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=cfg.seed)
    oof = np.zeros(len(y), dtype=np.float64)
    for train_idx, valid_idx in skf.split(X, y):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
        clf.fit(X[train_idx], y[train_idx])
        oof[valid_idx] = clf.predict_proba(X[valid_idx])[:, 1]

    auc = roc_auc_score(y, oof)

    out_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    out_df["id"] = ids
    out_df["has_break"] = y
    out_df["oof_pred"] = oof
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nCE6 — classificador só-histórico, {len(y)} séries, {args.n_folds}-fold CV")
    print(f"base rate (fração com quebra): {y.mean():.4f}")
    print(f"OOF AUC (série-nível, prever SE quebra, não quando): {auc:.4f}")
    if auc > 0.55:
        print("-> ACIMA da taxa-base: o histórico carrega sinal sobre se a série vai quebrar. "
              "meta_h0_* fazem trabalho legítimo mesmo como efeito principal — mas reavaliar com "
              "o usuário a política de não explorar isso deliberadamente (plano §12.2).")
    else:
        print("-> Ao redor da taxa-base: o histórico sozinho não prevê quebra. Uso de meta_h0_* "
              "como efeito PRINCIPAL (não interação/condicionamento) seria prejudicial ao ranking.")
    print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
