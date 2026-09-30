#!/usr/bin/env python
"""N1 (knowledge/frentes/teto-offline/N1-subamostra-negativos.md): treino com negativos subamostrados.

Wrapper fino do `scripts/train.py`: os folds são os do Onyx; em cada fold, as séries NEGATIVAS com
hash(id) fora da fração `N1_FRAC` (variável de ambiente, default 0,5) saem do TREINO e continuam na
VALIDAÇÃO. O OOF sai para todas as séries.
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
FRAC = float(os.environ.get("N1_FRAC", "0.5"))


def _mantem(ids: np.ndarray) -> np.ndarray:
    # hash determinístico do id em [0, 1)
    return ((ids.astype(np.int64) * 2654435761) % 1000) / 1000.0 < FRAC


def _folds(meta: pd.DataFrame, k: int, seed: int):
    ids = meta["id"].to_numpy()
    neg_serie = meta.groupby("id")["y"].transform("max").to_numpy() == 0
    fora = neg_serie & ~_mantem(ids)
    for tr, va in _original(meta, k, seed):
        yield tr[~fora[tr]], va


def main() -> None:
    mt.grouped_stratified_kfold = _folds
    print(f"N1: negativos mantidos no treino = {FRAC}")
    cli.main()


if __name__ == "__main__":
    main()
