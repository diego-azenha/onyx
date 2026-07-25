"""A fusão de boosters tem de valer também para `linear_tree=true` (D1, adotado 2026-07-25).

## O defeito que este teste trava

`fuse_boosters` funde K boosters concatenando as árvores e dividindo as folhas por K — exato
enquanto a folha é uma CONSTANTE, porque o raw do LightGBM é uma soma sobre árvores.

Com `linear_tree=true` a folha passa a valer `leaf_const + Σ leaf_coeff_i · x_i`, e é por esses
campos que o LightGBM prediz quando `is_linear=1`. A versão original de `_scale_leaves` escalava
só `leaf_value`, deixando `leaf_const` e ~35 mil coeficientes por booster intactos: o fundido
devolvia algo com erro de **1,1e+02** contra a média dos raws.

O defeito era invisível no OOF — `scripts/train.py` faz média das PREDIÇÕES por semente
(`avg_oof.py`) e nunca chama `fuse_boosters` — e só aparecia no caminho de PRODUÇÃO
(`adapter/platform.py:train`, que roda na nuvem). Foi a guarda numérica do próprio `fuse_boosters`
que o pegou, ao regenerar o notebook de submissão. É a mesma classe do achado A3 ("o que é medido
não é o que é implantado"), e por isso vira teste em vez de nota de rodapé.
"""
from __future__ import annotations

import numpy as np
import pytest

lgb = pytest.importorskip("lightgbm")

from sbrt.model.fuse import fuse_boosters


def _treinar(linear: bool, seed: int, n_feat: int = 6):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(400, n_feat))
    # alvo com componente LINEAR de verdade: sem ele o LightGBM pode devolver folhas degeneradas
    # (coeficientes vazios) e o teste passaria sem exercitar o caminho que importa.
    y = (x[:, 0] * 1.5 + x[:, 1] - 0.5 * x[:, 2] + rng.normal(scale=0.3, size=400) > 0).astype(int)
    return lgb.train(
        {
            "objective": "binary", "verbose": -1, "num_leaves": 7, "min_data_in_leaf": 20,
            "learning_rate": 0.1, "seed": seed, "deterministic": True, "force_row_wise": True,
            "linear_tree": linear,
        },
        lgb.Dataset(x, label=y),
        num_boost_round=12,
    )


@pytest.mark.parametrize("linear", [False, True])
def test_fusao_reproduz_a_media_dos_raws(linear: bool) -> None:
    """O contrato de `fuse_boosters`: raw do fundido == média dos raws, nos dois modos de folha."""
    boosters = [_treinar(linear, s) for s in (11, 22, 33)]
    fundido = fuse_boosters(boosters)  # verify=True levanta sozinho se a fusão não bater

    x = np.linspace(-3.0, 3.0, 64 * 6).reshape(64, 6)
    esperado = np.column_stack(
        [b.predict(x, raw_score=True, num_threads=1) for b in boosters]
    ).mean(axis=1)
    obtido = fundido.predict(x, raw_score=True, num_threads=1)
    assert np.abs(esperado - obtido).max() < 1e-9


def test_arvores_lineares_tem_coeficientes_de_verdade() -> None:
    """Guarda do próprio teste: se o LightGBM degenerasse as folhas para constantes, o teste acima
    passaria sem nunca exercitar `leaf_coeff` — e o defeito voltaria despercebido."""
    texto = _treinar(True, 11).model_to_string()
    assert "is_linear=1" in texto
    coeffs = [l for l in texto.splitlines() if l.startswith("leaf_coeff=") and l[len("leaf_coeff="):].strip()]
    assert coeffs, "nenhuma folha linear com coeficientes: o teste não está cobrindo o caminho do D1"
