"""A2 (CAMPANHA_POLIMENTO.md) — a reponderação para a grade do board.

O que estes testes travam é a propriedade que torna a correção legítima: `w_mult` é uma função da
GRADE, não da amostra, então compõe com a reamostragem do bootstrap pareado; e reponderar uma grade
que já é a cheia não muda nada.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc


def _toy(n_series: int = 60, max_t: int = 12, seed: int = 0):
    """Painel sintético com comprimentos variáveis e metade das séries positivas a partir de tau."""
    rng = np.random.default_rng(seed)
    rows = []
    for sid in range(n_series):
        T = int(rng.integers(4, max_t + 1))
        tau = int(rng.integers(1, T + 1)) if sid % 2 == 0 else None
        for t in range(1, T + 1):
            rows.append((sid, t, 1 if (tau is not None and t >= tau) else 0))
    return pd.DataFrame(rows, columns=["id", "t", "y"])


def test_w_mult_none_reproduz_comportamento_historico():
    df = _toy()
    s = np.linspace(0, 1, len(df))
    assert weighted_ts_auc(df.t.values, df.y.values, s) == weighted_ts_auc(
        df.t.values, df.y.values, s, None
    )


def test_w_mult_unitario_e_neutro():
    df = _toy()
    s = np.linspace(0, 1, len(df))
    ones = {int(t): 1.0 for t in df.t.unique()}
    assert weighted_ts_auc(df.t.values, df.y.values, s, ones) == pytest.approx(
        weighted_ts_auc(df.t.values, df.y.values, s)
    )


def test_multiplicadores_somam_a_massa_do_board(tmp_path):
    """Cada passo do board é atribuído a exatamente um passo do OOF, então `Σ w_mult·w_t` sobre a
    grade do OOF tem de reproduzir `Σ w_t` do board — nenhuma massa criada nem perdida."""
    df = _toy(n_series=80, max_t=16, seed=1)
    y_path = tmp_path / "y.parquet"
    df.assign(time=df.t, target=df.y).set_index(["id", "time"])[["target"]].to_parquet(y_path)

    grade_oof = np.array([t for t in sorted(df.t.unique()) if t % 2 == 1])  # thinning artificial
    mult = board_grid_multipliers(str(y_path), grade_oof)

    g = df.groupby("t")["y"]
    w_board = (g.sum() * (g.size() - g.sum())).astype(float)
    w_board = w_board[w_board > 0]

    massa_reponderada = sum(mult[int(t)] * w_board[t] for t in grade_oof if t in w_board.index)
    assert massa_reponderada == pytest.approx(w_board.sum())


def test_grade_cheia_tem_multiplicadores_unitarios(tmp_path):
    """Se o OOF já vive na grade do board, reponderar não pode mudar nada."""
    df = _toy(seed=2)
    y_path = tmp_path / "y.parquet"
    df.assign(time=df.t, target=df.y).set_index(["id", "time"])[["target"]].to_parquet(y_path)

    mult = board_grid_multipliers(str(y_path), np.array(sorted(df.t.unique())))
    assert all(v == pytest.approx(1.0) for v in mult.values())

    s = np.linspace(0, 1, len(df))
    assert weighted_ts_auc(df.t.values, df.y.values, s, mult) == pytest.approx(
        weighted_ts_auc(df.t.values, df.y.values, s)
    )


def test_reponderacao_recupera_o_agregado_da_grade_cheia(tmp_path):
    """O teste que importa: pontuar só nos passos ímpares COM os multiplicadores tem de chegar perto
    do agregado sobre todos os passos -- que é a afirmação inteira do A2. Exato quando `AUC_t` é
    constante em t (aqui, um score que só depende de y)."""
    df = _toy(n_series=200, max_t=20, seed=3)
    y_path = tmp_path / "y.parquet"
    df.assign(time=df.t, target=df.y).set_index(["id", "time"])[["target"]].to_parquet(y_path)

    rng = np.random.default_rng(7)
    s = df.y.values * 0.6 + rng.uniform(0, 0.4, len(df))  # AUC_t idêntica em todo t

    cheio = weighted_ts_auc(df.t.values, df.y.values, s)
    m = df.t.values % 2 == 1
    grade = np.array(sorted(df.t[m].unique()))
    mult = board_grid_multipliers(str(y_path), grade)
    reponderado = weighted_ts_auc(df.t.values[m], df.y.values[m], s[m], mult)

    assert reponderado == pytest.approx(cheio, abs=0.02)
