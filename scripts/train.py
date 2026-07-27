#!/usr/bin/env python
"""CLI fina: treina o ensemble LightGBM (plano §8.3) a partir de data/processed/train_rows.parquet.
Salva também as predições out-of-fold (id, t, y, tau_index-derivado, oof_pred) para diagnósticos
(A4: resposta ao degrau alinhada em tau, plano_acao_v1_para_v2.md)."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--rows", default="data/processed/train_rows.parquet")
    parser.add_argument("--out", default="artifacts/models/v1")
    parser.add_argument("--oof-out", default=None, help="default: <out>/../oof_<basename(out)>.parquet")
    parser.add_argument("--drop-prefix", nargs="*", default=[], metavar="PREFIXO",
                        help="remove colunas de feature com estes prefixos antes de treinar. Existe "
                             "para separar BRAÇOS de R0 sem repetir o build: um único dataset com "
                             "várias famílias novas rende um braço por família, cada um com sua R0, "
                             "porque `model/train.py` deriva `feature_order` das colunas presentes. "
                             "A disciplina de 'um braço por R0' é sobre o MODELO, não sobre o build.")
    parser.add_argument("--boost-seed", type=int, default=None,
                        help="sobrescreve lightgbm.boost_seed (sorteio interno do LightGBM; NÃO muda "
                             "os folds, que vêm de cfg.seed). MEDIDO 2026-07-22: trocar só isto move "
                             "a TS-AUC OOF em -0,0037 [-0,0088, +0,0012] — maior que o Delta de dois "
                             "dos três braços já descartados. Por isso todo braço de R0 agora se mede "
                             "com VÁRIAS sementes por lado; ver docs/BACKLOG_TSAUC.md.")
    parser.add_argument("--early-stopping-rounds", type=int, default=None,
                        help="sobrescreve lightgbm.early_stopping_rounds. 0 ou negativo DESLIGA a "
                             "parada antecipada e treina até n_estimators_cap — necessário para X0 "
                             "(curva por rodada COMPLETA, para o argmax da média entre sementes).")
    parser.add_argument("--n-estimators-cap", type=int, default=None,
                        help="sobrescreve lightgbm.n_estimators_cap (num_boost_round). X0 usa ~600/"
                             "fold com ES desligado para conter o argmax da curva média.")
    parser.add_argument("--detectability-mode", default="none", choices=["none", "hard", "soft", "ramp"],
                        help="X1: pondera positivos por detectabilidade (detectability.csv). none=V4 "
                             "atual; hard=dropa decil inferior; soft=clip(d/q95,0.3,1); ramp=(t−τ).")
    parser.add_argument("--wt-align", action="store_true",
                        help="B1/F3: alinhar a massa de peso por passo t com w_t=n_pos·n_neg da métrica.")
    parser.add_argument("--noisy-neg-flags", default=None,
                        help="B3: CSV com id,label_suspect; rebaixa negativos das séries flagadas.")
    parser.add_argument("--pairwise-alpha", type=float, default=None,
                        help="B4: peso do termo pairwise intra-t no objetivo custom (0.1/0.3).")
    parser.add_argument("--fold-seed", type=int, default=None,
                        help="C3: sobrescreve cfg.seed (a PARTIÇÃO de folds). Bagging de partição = "
                             "treinar com folds diferentes e fundir OOF honesto.")
    parser.add_argument("--monotone-llr", action="store_true",
                        help="B5: monotone_constraints=+1 nos acumuladores de evidência (LLR).")
    parser.add_argument("--feature-contri-meta", type=float, default=None,
                        help="B6: multiplicador de ganho (0.5-0.7) nas colunas meta_h0_*.")
    parser.add_argument("--detectability-path", default="artifacts/reports/detectability.csv",
                        help="C2(ii): CSV de d_i lido offline. O default e o do X1 adotado; o braco do "
                             "delta_mean aponta para um CSV gerado com --extra-axes delta_mean_e.")
    parser.add_argument("--contri-extra-cols", nargs="*", default=[], metavar="COLUNA",
                        help="C1(ii): estende a penalizacao de feature_contri a estas colunas, por "
                             "NOME EXATO (ex.: conformal_logm_abs mmd_joint_slow_cal). Sem elas, no-op.")
    parser.add_argument("--contri-extra", type=float, default=1.0,
                        help="C1(ii): multiplicador aplicado a --contri-extra-prefixes (default 1.0 "
                             "= sem penalizacao; independente do --feature-contri-meta).")
    parser.add_argument("--num-leaves", type=int, default=None,
                        help="E0c (DIAGNOSTICO_ESTRUTURAL.md §5): sobrescreve lightgbm.num_leaves. "
                             "Com 2 (+ --max-depth 1) o aprendiz vira um TOCO: modelo aditivo puro, "
                             "sem NENHUMA interacao entre as 183 features. Se a TS-AUC empatar com o "
                             "incumbente, as interacoes nao contribuem e a base saturou o aprendiz.")
    parser.add_argument("--max-depth", type=int, default=None,
                        help="E0c: sobrescreve lightgbm.max_depth (1 = toco; -1 = sem limite).")
    parser.add_argument("--base-rate-weighted", action="store_true",
                        help="A3b: ajusta a curva de taxa-base (init_score) sob os MESMOS pesos de "
                             "linha do treino. Os pesos pareados do R1 equalizam as classes dentro de "
                             "cada t (taxa ponderada ~0,5), entao a curva nao-ponderada injeta um "
                             "offset de ate +3,81 em log-odds em vez de remover um.")
    parser.add_argument("--linear-tree", action="store_true",
                        help="D1: folhas LINEARES (modelo linear por folha em vez de constante). O E0c "
                             "mostrou que o problema e aditivo e suave; arvores de degraus aproximam "
                             "curvas suaves com escadinhas.")
    parser.add_argument("--max-bin", type=int, default=None,
                        help="D2: sobrescreve lightgbm.max_bin (255 -> 511/1023). Resolucao do "
                             "histograma nas caudas dos LLRs, onde a evidencia forte vive.")
    parser.add_argument("--detect-floor", type=float, default=None,
                        help="C2: piso do multiplicador de detectabilidade do X1 (w_pos ×= "
                             "clip(d/q95, floor, 1)). O default 0,3 vinha da assinatura de "
                             "compute_row_weights, NÃO de cfg.weights.detect_floor -- offline o YAML "
                             "não alcançava este número, então a vizinhança nunca pôde ser varrida.")
    parser.add_argument("--set", nargs="*", default=[], metavar="CHAVE=VALOR",
                        help="D3/D5: sobrescreve qualquer campo escalar de lightgbm (ex.: "
                             "learning_rate=0.03 min_data_in_leaf=100). Tipado pelo valor atual.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.boost_seed is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, boost_seed=args.boost_seed))
        print(f"boost_seed sobrescrito para {args.boost_seed} (folds inalterados, cfg.seed={cfg.seed})")
    if args.early_stopping_rounds is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, early_stopping_rounds=args.early_stopping_rounds))
        print(f"early_stopping_rounds sobrescrito para {args.early_stopping_rounds}"
              f"{' (parada DESLIGADA)' if args.early_stopping_rounds <= 0 else ''}")
    if args.n_estimators_cap is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, n_estimators_cap=args.n_estimators_cap))
        print(f"n_estimators_cap sobrescrito para {args.n_estimators_cap}")
    if args.pairwise_alpha is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, pairwise_alpha=args.pairwise_alpha))
        print(f"B4: objetivo pairwise intra-t com alpha={args.pairwise_alpha}")
    if args.fold_seed is not None:
        cfg = replace(cfg, seed=args.fold_seed)
        print(f"C3: partição de folds com cfg.seed={args.fold_seed}")
    if args.monotone_llr:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, monotone_llr=True))
        print("B5: monotone_constraints=+1 nos acumuladores de LLR")
    if args.feature_contri_meta is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, feature_contri_meta=args.feature_contri_meta))
        print(f"B6: feature_contri={args.feature_contri_meta} em meta_h0_*")
    if args.contri_extra_cols:
        cfg = replace(cfg, lightgbm=replace(
            cfg.lightgbm,
            feature_contri_extra_cols=tuple(args.contri_extra_cols),
            feature_contri_extra=args.contri_extra,
        ))
        print(f"C1(ii): feature_contri={args.contri_extra} em {list(args.contri_extra_cols)}")
    if args.num_leaves is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, num_leaves=args.num_leaves))
        print(f"num_leaves sobrescrito para {args.num_leaves}")
    if args.max_depth is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, max_depth=args.max_depth))
        print(f"max_depth sobrescrito para {args.max_depth}"
              f"{' (TOCO: modelo aditivo, sem interacoes)' if args.max_depth == 1 else ''}")
    if args.base_rate_weighted:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, base_rate_weighted=True))
        print("A3b: curva de taxa-base (init_score) ajustada sob os pesos de treino")
    if args.linear_tree:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, linear_tree=True))
        print("D1: linear_tree=true (folhas lineares)")
    if args.max_bin is not None:
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, max_bin=args.max_bin))
        print(f"D2: max_bin={args.max_bin}")
    for kv in args.set:
        k, _, v = kv.partition("=")
        cur = getattr(cfg.lightgbm, k)  # KeyError explicito se o campo nao existe
        cast = type(cur) if cur is not None else float
        val = (v.lower() in ("1", "true", "yes")) if cast is bool else cast(v)
        cfg = replace(cfg, lightgbm=replace(cfg.lightgbm, **{k: val}))
        print(f"--set {k}={val!r} (era {cur!r})")
    rows = pd.read_parquet(args.rows)
    if args.drop_prefix:
        dropped = [c for c in rows.columns if c.startswith(tuple(args.drop_prefix))]
        rows = rows.drop(columns=dropped)
        print(f"removidas {len(dropped)} colunas por --drop-prefix {args.drop_prefix}: {sorted(dropped)}")
    noisy_ids = None
    if args.noisy_neg_flags:
        _f = pd.read_csv(args.noisy_neg_flags)
        noisy_ids = set(_f.loc[_f["label_suspect"] == True, "id"].astype(int))  # noqa: E712
        print(f"B3: rebaixando negativos de {len(noisy_ids)} séries flagadas")
    weights = compute_row_weights(rows, cfg, detectability_mode=args.detectability_mode,
                                  detectability_path=args.detectability_path,
                                  detect_floor=(args.detect_floor if args.detect_floor is not None
                                                else cfg.weights.detect_floor),
                                  wt_align=args.wt_align, noisy_neg_ids=noisy_ids)
    if args.detect_floor is not None:
        print(f"C2: detect_floor={args.detect_floor}")
    if args.detectability_mode != "none":
        print(f"X1: pesos de detectabilidade modo={args.detectability_mode}")
    if args.wt_align:
        print("B1/F3: massa de peso alinhada a w_t=n_pos·n_neg")
    ensemble, oof_pred = train_ensemble(rows, weights, cfg, progress=True)
    ensemble.save(args.out)
    print(f"ensemble salvo em {args.out} ({len(ensemble.boosters)} folds, {len(ensemble.feature_order)} features)")

    out_dir = Path(args.out)
    oof_out = Path(args.oof_out) if args.oof_out else out_dir.parent / f"oof_{out_dir.name}.parquet"
    oof_df = rows[["id", "t", "y"]].copy()
    oof_df["oof_pred"] = np.asarray(oof_pred, dtype=np.float64)
    oof_out.parent.mkdir(parents=True, exist_ok=True)
    oof_df.to_parquet(oof_out)
    print(f"predições out-of-fold salvas em {oof_out} ({len(oof_df)} linhas)")


if __name__ == "__main__":
    main()
