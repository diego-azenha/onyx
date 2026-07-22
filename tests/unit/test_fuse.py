import numpy as np
import pytest

from sbrt.model.fuse import fuse_boosters


def _boosters(k=3, n=400, n_feat=6, rounds=12):
    import lightgbm as lgb

    rng = np.random.RandomState(0)
    X = rng.randn(n, n_feat)
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.3 * rng.randn(n) > 0).astype(int)
    out = []
    for s in range(k):
        d = lgb.Dataset(X, label=y)
        out.append(lgb.train({"objective": "binary", "verbose": -1, "seed": 100 + s,
                              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                              "num_leaves": 7, "deterministic": True}, d, num_boost_round=rounds))
    return out, X


def test_fused_raw_is_the_mean_of_raws():
    """A propriedade que autoriza a fusão: a predição raw é uma SOMA sobre árvores, então concatenar
    K modelos com folhas/K dá exatamente a média dos raws."""
    bs, X = _boosters()
    fused = fuse_boosters(bs)
    esperado = np.column_stack([b.predict(X, raw_score=True) for b in bs]).mean(axis=1)
    assert np.abs(esperado - fused.predict(X, raw_score=True)).max() < 1e-9
    assert fused.num_trees() == sum(b.num_trees() for b in bs)


def test_single_booster_is_returned_untouched():
    bs, _ = _boosters(k=1)
    assert fuse_boosters(bs) is bs[0]


def test_verification_rejects_a_corrupted_fusion(monkeypatch):
    """O guarda existe porque a fusão reescreve `tree_sizes` por regex: se o padrão nao casar, o
    `re.sub` devolve a string INALTERADA e o modelo corrompido carrega em silêncio. Este caminho roda
    na nuvem, onde ninguém está olhando — falhar alto é melhor que submeter lixo.

    Aqui a corrupção é simulada sabotando a divisão das folhas (esquecendo o /K), que é o outro modo
    de falha possível e produz um modelo cujo raw é a SOMA, não a média."""
    import sbrt.model.fuse as mod

    monkeypatch.setattr(mod, "_scale_leaves", lambda block, k: block)
    bs, _ = _boosters()
    with pytest.raises(RuntimeError, match="fus.o INV.LIDA|INVALIDA"):
        mod.fuse_boosters(bs)


def test_missing_tree_sizes_raises_instead_of_silently_corrupting(monkeypatch):
    import re as _re

    import sbrt.model.fuse as mod

    real_subn = _re.subn
    monkeypatch.setattr(mod.re, "subn", lambda *a, **k: (real_subn(*a, **k)[0], 0))
    bs, _ = _boosters()
    with pytest.raises(RuntimeError, match="tree_sizes"):
        mod.fuse_boosters(bs)
