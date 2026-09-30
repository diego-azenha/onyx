#!/usr/bin/env python
"""A1 (knowledge/frentes/teto-offline/A1-recorte.md): treino do Onyx com séries aumentadas, sem vazamento.

Wrapper fino do `scripts/train.py`: troca `grouped_stratified_kfold` (em `sbrt.model.train`) por uma
versão que calcula os folds SÓ nas séries originais (id < 10000) e manda cada série aumentada
(id = 10000*k + origem) para o TREINO do fold em que a origem é treino, nunca para a validação.
Todo o resto é o `train.py` intocado. O OOF final é filtrado para as séries originais.

Uso: python scripts/a1_treino.py --rows <originais+aumentadas.parquet> <flags do train.py...>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sbrt.model.train as mt  # noqa: E402
import train as cli  # noqa: E402  (scripts/train.py)

_original = mt.grouped_stratified_kfold


def _folds_presos_a_origem(meta: pd.DataFrame, k: int, seed: int):
    ids = meta["id"].to_numpy()
    orig = ids < 10000
    pos_orig = np.flatnonzero(orig)
    pos_aug = np.flatnonzero(~orig)
    src_aug = ids[pos_aug] % 10000
    meta_orig = meta.iloc[pos_orig].reset_index(drop=True)
    for tr, va in _original(meta_orig, k, seed):
        ids_tr = np.unique(meta_orig["id"].to_numpy()[tr])
        aug_tr = pos_aug[np.isin(src_aug, ids_tr)]
        yield np.concatenate([pos_orig[tr], aug_tr]), pos_orig[va]


def main() -> None:
    mt.grouped_stratified_kfold = _folds_presos_a_origem
    cli.main()
    # filtra o OOF para as séries originais (as aumentadas nunca foram validadas: NaN)
    argv = sys.argv
    out = argv[argv.index("--out") + 1]
    oof_path = Path(argv[argv.index("--oof-out") + 1]) if "--oof-out" in argv else \
        Path(out).parent / f"oof_{Path(out).name}.parquet"
    oof = pd.read_parquet(oof_path)
    oof = oof[oof["id"] < 10000].reset_index(drop=True)
    assert oof["oof_pred"].notna().all(), "série original sem predição OOF"
    oof.to_parquet(oof_path)
    print(f"OOF filtrado para {oof['id'].nunique()} séries originais ({len(oof)} linhas)")


if __name__ == "__main__":
    main()
