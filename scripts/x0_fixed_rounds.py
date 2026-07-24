#!/usr/bin/env python
"""X0 (BRAINSTORM_RUPTURA_TSAUC.md §3): rodadas fixas por fold pela curva de validação MÉDIA ENTRE
SEMENTES. Encolhe o σ_seed que a parada antecipada erráica (51->103 árvores no mesmo fold só trocando
a semente) injeta na régua -- a maior alavanca disponível hoje e não é feature (docs/BACKLOG_TSAUC.md,
"O nulo da regra de decisão").

Insumo: K ensembles JÁ treinados com ES DESLIGADO e cap alto (curvas COMPLETAS por rodada), p.ex.
    for S in 777 101 202 303; do \
      python scripts/train.py --rows data/processed/train_rows_3eixos.parquet \
        --drop-prefix spec_ ord_ mrep_ --boost-seed $S \
        --early-stopping-rounds 0 --n-estimators-cap 600 --out artifacts/models/x0_s$S ; done

Este script NÃO retreina. Para cada métrica (ts_auc_by_t: maior-melhor; binary_logloss_diag:
menor-melhor) computa a curva média entre sementes POR FOLD, tira o argmax -> nº de rodadas fixo por
fold, e reconstrói a OOF de cada semente FATIANDO os boosters já treinados em `num_iteration`
(mesma aritmética de model/train.py:173-175: raw + init_score -> sigmoide). Escreve uma OOF por
semente por política e a média K das sementes (o objeto que o R0 compara). O veredito é do
compare_oof.py/seed_spread.py sobre essas OOF, nunca deste script.

Hipótese registrada a priori: a média entre sementes remove a maldição do vencedor que matou o R2 e o
`ts_auc_by_t` como critério de parada (aquilo otimizava UMA realização ruidosa; a média de K é outra
estatística). Kill de X0: σ_seed não cai E Δ≤0 nas três políticas (esta (b)/(c) vs. status quo (a))."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.model.base_rate import predict_base_rate_logit
from sbrt.model.predict import ModelEnsemble

_NON_FEATURE_COLS = {"id", "t", "y", "thin_weight"}

# métrica -> (chave no fold_evals["valid_0"], maior-é-melhor)
_METRICS = {
    "logloss": ("binary_logloss_diag", False),
    "tsauc": ("ts_auc_by_t", True),
}


def _mean_curve_argmax(fold_evals_by_seed: list[list[dict]], n_folds: int, key: str, higher_better: bool):
    """Para cada fold, média entre sementes da curva por rodada -> índice do melhor round.
    Retorna (best_rounds_1indexed[n_folds], mean_curves[n_folds])."""
    best_rounds, mean_curves = [], []
    for f in range(n_folds):
        curves = [np.asarray(fe[f]["valid_0"][key], dtype=np.float64) for fe in fold_evals_by_seed]
        m = min(len(c) for c in curves)  # com ES off todas têm cap; min protege contra qualquer diferença
        mean_curve = np.mean([c[:m] for c in curves], axis=0)
        best_idx = int(np.argmax(mean_curve) if higher_better else np.argmin(mean_curve))
        best_rounds.append(best_idx + 1)  # num_iteration = índice(0-based) + 1
        mean_curves.append(mean_curve)
    return best_rounds, mean_curves


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    p.add_argument("--rows", default="data/processed/train_rows_3eixos.parquet")
    p.add_argument("--drop-prefix", nargs="*", default=["spec_", "ord_", "mrep_"], metavar="PREFIXO",
                   help="mesma remoção usada no treino das sementes (default = V4 legítimo).")
    p.add_argument("--ensembles", nargs="+", required=True,
                   help="dirs dos K ensembles X0 (ES off, cap alto), p.ex. artifacts/models/x0_s777 ...")
    p.add_argument("--out-dir", default="artifacts/models")
    p.add_argument("--tag", default="x0", help="prefixo dos parquets de saída (oof_<tag>_<pol>_s<seed>).")
    args = p.parse_args()

    cfg = load_config(args.config)
    rows = pd.read_parquet(args.rows)
    if args.drop_prefix:
        dropped = [c for c in rows.columns if c.startswith(tuple(args.drop_prefix))]
        rows = rows.drop(columns=dropped)
        print(f"removidas {len(dropped)} colunas por --drop-prefix {args.drop_prefix}")

    ensembles = [ModelEnsemble.load(d) for d in args.ensembles]
    feature_order = ensembles[0].feature_order
    for d, e in zip(args.ensembles, ensembles):
        if e.feature_order != feature_order:
            raise SystemExit(f"{d}: feature_order difere do primeiro ensemble — sementes incompatíveis.")
    n_folds = cfg.lightgbm.n_folds
    for d, e in zip(args.ensembles, ensembles):
        if len(e.boosters) != n_folds or len(e.fold_evals) != n_folds:
            raise SystemExit(f"{d}: esperados {n_folds} folds, achei {len(e.boosters)} boosters / "
                             f"{len(e.fold_evals)} fold_evals.")

    X = rows[list(feature_order)].to_numpy(dtype=np.float32)
    t_values = rows["t"].to_numpy(dtype=np.float64)
    # folds idênticos ao treino: mesma função, mesma cfg.seed, mesma ordem de linhas do parquet.
    folds = list(grouped_stratified_kfold(rows, n_folds, cfg.seed))

    fold_evals_by_seed = [e.fold_evals for e in ensembles]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pol, (key, higher_better) in _METRICS.items():
        best_rounds, _ = _mean_curve_argmax(fold_evals_by_seed, n_folds, key, higher_better)
        print(f"\n[política {pol}] rodadas fixas por fold ({key}, {'max' if higher_better else 'min'} "
              f"da média entre {len(ensembles)} sementes): {best_rounds}")

        per_seed_paths = []
        for d, e in zip(args.ensembles, ensembles):
            seed_tag = Path(d).name.split("_s")[-1]
            oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
            init_full = predict_base_rate_logit(t_values, e.base_rate_curve)
            for f, (_train_idx, valid_idx) in enumerate(folds):
                raw = e.boosters[f].predict(X[valid_idx], raw_score=True, num_iteration=best_rounds[f])
                logit = raw + init_full[valid_idx]
                oof_pred[valid_idx] = 1.0 / (1.0 + np.exp(-logit))
            oof_df = rows[["id", "t", "y"]].copy()
            oof_df["oof_pred"] = oof_pred
            sp = out_dir / f"oof_{args.tag}_{pol}_s{seed_tag}.parquet"
            oof_df.to_parquet(sp)
            per_seed_paths.append(sp)
            print(f"  s{seed_tag}: OOF fatiada -> {sp}")

        # média K (espaço de probabilidade — mesma agregação que avg_oof.py; NÃO o bag fundido em logit)
        base = pd.read_parquet(per_seed_paths[0])
        acc = base["oof_pred"].to_numpy(dtype=np.float64).copy()
        for sp in per_seed_paths[1:]:
            acc += pd.read_parquet(sp)["oof_pred"].to_numpy(dtype=np.float64)
        acc /= len(per_seed_paths)
        bag = base.copy()
        bag["oof_pred"] = acc
        bp = out_dir / f"oof_{args.tag}_{pol}_bag{len(per_seed_paths)}.parquet"
        bag.to_parquet(bp)
        print(f"  média de {len(per_seed_paths)} sementes -> {bp}")

    print("\nPróximo: seed_spread.py (σ_seed novo) e compare_oof.py de cada bag (b)/(c) contra o bag (a) "
          "status-quo = avg_oof(oof_b183_s{777,101,202,303}).")


if __name__ == "__main__":
    main()
