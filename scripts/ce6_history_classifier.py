#!/usr/bin/env python
"""A3 / CE6 (plano_structural_break_realtime.md §12.2, nunca executado; plano_acao_v1_para_v2.md
§4 A3): treina um classificador usando SÓ as features do H0 (nenhum ponto online) para prever se a
série tem quebra em algum momento, via CV padrão. Decide o destino das features `meta_h0_*`:

- Se prevê melhor que a taxa-base (~50%) -> o gerador vaza o rótulo pelo histórico; meta_h0_* fazem
  trabalho legítimo (mesmo como efeito principal) e a política de não explorar deliberadamente
  (§12.2) precisa ser reavaliada explicitamente com o usuário.
- Se não prevê nada -> qualquer uso de meta_h0_* como EFEITO PRINCIPAL (não interação) injeta um
  offset por série sem informação sobre quebra -- ativamente prejudicial para o ranking transversal.

## F0.d: o segundo alvo, e por que ele é o gate da frente de precursores

O alvo acima pergunta "esta série QUEBRA?". A frente F5 (docs/BACKLOG_TSAUC.md) aposta em outra coisa:
que a cauda do histórico carrega precursores de *critical slowing down* -- AC(1) e variância subindo
antes da transição -- e que por isso daria para emitir sinal já no passo 1, quando não há observação
online nenhuma.

Essa é uma pergunta DIFERENTE: prever *quando*, não *se*. Uma série pode ter histórico
indistinguível de outra e ainda assim quebrar cedo. Por isso o segundo alvo: entre as séries que
quebram, prever `1{tau_index <= k}` a partir SÓ do histórico.

O veredito é operacional. Se as duas formulações ficam ≈0,50 com os 28 descritores atuais, F5 está
apostando contra uma medição direta e não deve ser construído antes de haver um descritor que mova
este número. Rodar isto custa minutos; construir F5 custa uma frente inteira.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
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
from sbrt.state.fingerprint import compute_precursors
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

# F0.d: os precursores de critical slowing down (state/fingerprint.py:compute_precursors). Entram
# só com --with-precursors, e não são features de produção — este script é o gate que decide se
# chegam a ser. Ver a docstring do módulo.
_PRECURSOR_NAMES = [
    "precursor_ac1_slope", "precursor_var_slope",
    "precursor_skew_slope", "precursor_ac1_last_minus_first",
]


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
    parser.add_argument("--tau-threshold", type=int, default=50,
                        help="F0.d: limiar de 'quebra precoce' para o segundo alvo (default: o bucket t<=50)")
    parser.add_argument("--reuse", action="store_true",
                        help="reaproveita a matriz de features de --out em vez de refazer fit_h0 em "
                             "todas as séries (só vale se os descritores não mudaram; verificado)")
    parser.add_argument("--with-precursors", action="store_true",
                        help="F0.d: acrescenta os precursores de critical slowing down da cauda do "
                             "histórico (state/fingerprint.py:compute_precursors) — o gate de F5")
    args = parser.parse_args()

    cfg = load_config(args.config)
    feature_names = list(FEATURE_NAMES) + (_PRECURSOR_NAMES if args.with_precursors else [])
    # o CE6 só lê escalares do H0; `null_stats` nunca é tocado. Desligar a calibração torna o
    # fit_h0 desta varredura várias vezes mais barato sem mudar nenhum número que este script usa.
    cfg = replace(cfg, calibration=replace(cfg.calibration, enabled=False))
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)

    y_index = pd.read_parquet(data_dir / "y_train_index.parquet")

    cached = None
    if args.reuse and out_path.exists():
        cached = pd.read_csv(out_path)
        if not set(feature_names).issubset(cached.columns):
            missing = sorted(set(feature_names) - set(cached.columns))
            print(f"[ce6] cache em {out_path} não tem {len(missing)} descritor(es) atual(is) "
                  f"({missing[:3]}...) — recalculando do zero.")
            cached = None

    if cached is not None:
        ids = cached["id"].tolist()
        X = cached[feature_names].to_numpy(dtype=np.float64)
        print(f"[ce6] reaproveitando a matriz de features de {out_path} ({len(ids)} séries)")
    else:
        X_raw = pd.read_parquet(data_dir / "X_train.parquet")
        rows, ids = [], []
        for dataset_id in tqdm(list(X_raw.index.get_level_values("id").unique()),
                               desc="ajustando H0 por série (CE6)"):
            group = X_raw.loc[dataset_id]
            hist_vals = group.loc[group["period"] == 1, "value"].to_numpy(dtype="float64")
            h0 = fit_h0(hist_vals, cfg)
            row = h0_to_row(h0)
            if args.with_precursors:
                pre = compute_precursors(h0.e_hist, cfg)
                row = row + [pre[n] for n in _PRECURSOR_NAMES]
            rows.append(row)
            ids.append(dataset_id)
        X = np.array(rows, dtype=np.float64)

    tau = np.array([int(y_index.loc[i, "tau_index"]) for i in ids], dtype=np.int64)
    y = (tau >= 0).astype(np.int32)

    def cv_auc(features: np.ndarray, target: np.ndarray) -> tuple:
        skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=cfg.seed)
        oof = np.zeros(len(target), dtype=np.float64)
        for train_idx, valid_idx in skf.split(features, target):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
            clf.fit(features[train_idx], target[train_idx])
            oof[valid_idx] = clf.predict_proba(features[valid_idx])[:, 1]
        return roc_auc_score(target, oof), oof

    auc, oof = cv_auc(X, y)

    out_df = pd.DataFrame(X, columns=feature_names)
    out_df["id"] = ids
    out_df["has_break"] = y
    out_df["tau_index"] = tau
    out_df["oof_pred"] = oof
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"\nCE6 — classificador só-histórico, {len(y)} séries, {args.n_folds}-fold CV, "
          f"{len(feature_names)} descritores")
    print(f"\n[alvo 1] a série QUEBRA?")
    print(f"  base rate: {y.mean():.4f}   OOF AUC: {auc:.4f}")
    if auc > 0.55:
        print("  -> ACIMA da taxa-base: o histórico carrega sinal sobre se a série vai quebrar. "
              "meta_h0_* fazem trabalho legítimo mesmo como efeito principal — mas reavaliar com "
              "o usuário a política de não explorar isso deliberadamente (plano §12.2).")
    else:
        print("  -> Ao redor da taxa-base: o histórico sozinho não prevê quebra. Uso de meta_h0_* "
              "como efeito PRINCIPAL (não interação/condicionamento) seria prejudicial ao ranking.")

    # --- F0.d: entre as que quebram, a quebra é PRECOCE? (o gate de F5) ---
    k = args.tau_threshold
    broke = y == 1
    early = (tau[broke] <= k).astype(np.int32)
    print(f"\n[alvo 2 — F0.d, gate de F5] a quebra é PRECOCE (tau_index <= {k})?")
    if early.sum() < 50 or (1 - early).sum() < 50:
        print(f"  amostra insuficiente ({early.sum()} precoces de {broke.sum()}) — inconclusivo.")
        return
    auc_early, _ = cv_auc(X[broke], early)
    print(f"  base rate: {early.mean():.4f}   OOF AUC: {auc_early:.4f}   "
          f"({early.sum()} precoces de {broke.sum()} séries com quebra)")
    if auc_early > 0.55:
        print("  -> O histórico ANTECIPA quebra precoce. É o achado que F5 precisa: vale construir "
               "os precursores condicionados (docs/BACKLOG_TSAUC.md F5).")
    else:
        print("  -> Ao redor da taxa-base: o histórico não antecipa quebra precoce com os descritores\n"
              "     atuais. F5 (precursores de critical slowing down) está apostando contra uma\n"
              "     medição direta — NÃO construir a frente antes de um descritor que mova este número.")
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
