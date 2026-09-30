#!/usr/bin/env python
"""S6 (knowledge/frentes/teto-offline/S6-especialistas-por-tipo.md): especialistas por TIPO de quebra.

No treino conhecemos o tipo de cada quebra (log-razão de variância do pós contra o histórico:
sobe > 0,3, desce < -0,3, fixa no meio; as quebras curtas vão para fixa). Cada especialista é a receita
do E1 treinada com os positivos do SEU tipo contra todos os negativos. As linhas positivas dos outros
tipos saem do TREINO, e a validação fica com todas as linhas (OOF completo). É supervisão extra por
evento para um aprendiz limitado por amostra.

Wrapper do scripts/train.py: variável de ambiente S6_TIPO in {sobe, desce, fixa}.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbrt.model.train as mt  # noqa: E402
import train as cli  # noqa: E402

_original = mt.grouped_stratified_kfold
TIPO = os.environ["S6_TIPO"]
GRUPO = {"sobe": {"sobe"}, "desce": {"desce"}, "fixa": {"fixa", "curta"}}[TIPO]


def _folds(meta: pd.DataFrame, k: int, seed: int):
    tipo = pd.read_parquet("artifacts/roteiro/tipo_quebra.parquet")["tipo"]
    t_row = meta["id"].map(tipo).to_numpy()
    fora = (meta["y"].to_numpy() == 1) & ~np.isin(t_row, list(GRUPO))
    for tr, va in _original(meta, k, seed):
        yield tr[~fora[tr]], va


def main() -> None:
    mt.grouped_stratified_kfold = _folds
    print(f"S6: especialista '{TIPO}' (tipos {sorted(GRUPO)})")
    cli.main()


if __name__ == "__main__":
    main()
