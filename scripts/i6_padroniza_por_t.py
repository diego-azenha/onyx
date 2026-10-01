#!/usr/bin/env python
"""I6 (knowledge/frentes/estrategias-01-10/README.md): features padronizadas DENTRO de cada passo t.

Lição do I1: um GBM agregado perde ordenação por passo até com (z, log t). A métrica compara séries no
mesmo t, mas as features do Onyx mudam de escala com t (janelas que enchem, acumuladores que crescem), e
as árvores gastam cortes nessa escala. Aqui cada feature vira (x − média_t) / dp_t, com média e dp por
faixa de t estimados nas linhas de TREINO (mapa congelado, aplicável em produção, causal porque t é
conhecido). Grava data/processed/train_rows_i6.parquet; o braço roda com a receita do E1.

Nota: as estatísticas por t usam todas as séries do treino, o que inclui as de validação no OOF. É só a
escala marginal de cada feature por passo (sem rótulo), então o efeito de vazamento é desprezível; em
produção o mapa vem do treino inteiro, como aqui.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def faixa(t: np.ndarray) -> np.ndarray:
    """Faixas de t: exatas até 100, de 4 em 4 até 400, de 16 em 16 depois (estatística estável)."""
    return np.where(t <= 100, t, np.where(t <= 400, 100 + (t - 100) // 4, 175 + (t - 400) // 16))


def main() -> None:
    rows = pd.read_parquet("data/processed/train_rows.parquet")
    cols = [c for c in rows.columns if c not in ("id", "t", "y", "thin_weight")]
    b = faixa(rows["t"].to_numpy())
    for i, c in enumerate(cols):
        x = rows[c].astype(np.float64)
        g = x.groupby(b)
        mu = g.transform("mean"); sd = g.transform("std")
        rows[c] = ((x - mu) / (sd + 1e-9)).astype(np.float32)
        if i % 40 == 0:
            print(i, c, flush=True)
    rows.to_parquet("data/processed/train_rows_i6.parquet", index=False)
    print("gravado", rows.shape)


if __name__ == "__main__":
    main()
