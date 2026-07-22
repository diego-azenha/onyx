#!/usr/bin/env python
"""Média de predições OOF sobre várias sementes do LightGBM — o lado do protocolo que o bootstrap
pareado não cobre.

## Por que isto existe

`compare_oof.py` reamostra SÉRIES e trata as predições de cada modelo como fixas. Mas o booster é um
sorteio: com `feature_fraction=0,8` e `bagging_fraction=0,8`, mudar o conjunto de colunas embaralha
todos os sorteios, então o candidato não é "o baseline mais uma coluna" — é um modelo novo tirado da
mesma distribuição.

MEDIDO (2026-07-22, `artifacts/reports/compare_null_boostseed.json`): retreinar o V4 contra ele
mesmo — mesmos dados, mesmas 183 features, MESMOS folds, só `lightgbm.boost_seed` diferente — dá
Delta -0,0037 [-0,0088, +0,0012]. Isso é **maior em módulo** que o Delta do F2 (-0,0024) e igual ao
do V5 (-0,0042), dois braços descartados por esse número.

O sinal negativo não é acaso: `oof_v4` é o incumbente porque foi o sorteio que ficou, então qualquer
sorteio novo regride à média. É maldição do vencedor no nível do artefato — o mesmo mecanismo que
`config.py:LightGBMConfig.early_stopping_metric` já documenta para a escolha da rodada de boosting,
aplicado ao modelo inteiro.

## O que este script faz

Média das colunas `oof_pred` de K parquets treinados com sementes diferentes e **os mesmos folds**
(`cfg.seed` fixo => `grouped_stratified_kfold` idêntico => as linhas casam por (id, t)). O resultado
é ao mesmo tempo:

- um estimador com ruído de semente ~1/sqrt(K) — o que torna o R0 capaz de resolver efeitos da ordem
  de 0,004, que hoje ele não resolve;
- um modelo que se poderia de fato submeter (bagging de sementes), não um artifício de medição.

Os dois lados da comparação precisam do MESMO K, senão a diferença de suavização vira efeito.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="parquets OOF (mesmas linhas, sementes distintas)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--score-col", default="oof_pred")
    args = parser.parse_args()

    base = pd.read_parquet(args.inputs[0])
    key = base[["id", "t"]]
    acc = base[args.score_col].to_numpy(dtype=np.float64).copy()

    for path in args.inputs[1:]:
        df = pd.read_parquet(path)
        if len(df) != len(base) or not df["id"].equals(base["id"]) or not df["t"].equals(base["t"]):
            raise SystemExit(
                f"{path}: linhas não casam com {args.inputs[0]}. As sementes têm de compartilhar os "
                f"MESMOS folds (cfg.seed fixo) e o mesmo dataset — senão a média não é do mesmo objeto."
            )
        acc += df[args.score_col].to_numpy(dtype=np.float64)
    acc /= len(args.inputs)

    out = base.copy()
    out[args.score_col] = acc
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"média de {len(args.inputs)} sementes salva em {args.out} ({len(out)} linhas)")
    print(f"  ruído de semente esperado: ~1/sqrt({len(args.inputs)}) = {1/np.sqrt(len(args.inputs)):.2f}x o de uma só")
    _ = key


if __name__ == "__main__":
    main()
