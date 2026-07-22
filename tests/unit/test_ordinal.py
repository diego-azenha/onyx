import math
from itertools import permutations

import numpy as np

from sbrt.state.ordinal import _REV, OrdinalBlock, pattern_index


def _run(cfg, x):
    blk = OrdinalBlock()
    blk.reset(None, cfg)
    for t, v in enumerate(x, start=1):
        blk.update(float(v), float(v), float(v), t)
    return blk.features()


def test_pattern_index_is_a_bijection():
    for m in (3, 4):
        idx = sorted(pattern_index(p, m) for p in permutations(range(m)))
        assert idx == list(range(math.factorial(m)))


def test_reversal_map_is_an_involution():
    for m, rev in _REV.items():
        assert sorted(rev) == list(range(math.factorial(m)))
        assert all(rev[rev[i]] == i for i in range(len(rev)))


def test_white_noise_gives_maximal_permutation_entropy(cfg):
    """i.i.d. => os m! padrões são equiprováveis => entropia normalizada ~1. É o valor de referência
    sob H0, igual para qualquer série independentemente de variância ou cauda."""
    x = np.random.RandomState(0).randn(3000)
    f = _run(cfg, x)
    assert 0.97 < f["ord_pe_m3_w100"] <= 1.0
    assert 0.93 < f["ord_pe_m4_w100"] <= 1.0


def test_monotone_ramp_collapses_entropy(cfg):
    """Uma rampa só produz um padrão ordinal => entropia 0 e 5/6 dos padrões proibidos."""
    f = _run(cfg, np.arange(300.0))
    assert f["ord_pe_m3_w100"] == 0.0
    assert f["ord_pe_m4_w100"] == 0.0
    assert f["ord_forbidden_m4_w100"] == 23 / 24


def test_invariant_to_any_monotone_transform(cfg):
    """A propriedade central: o padrão ordinal não vê valores, só ordem. Variância, cauda e escala
    são invisíveis aqui — é o que torna este eixo ortogonal à família que domina o modelo."""
    x = np.random.RandomState(1).randn(1000)
    a = _run(cfg, x)
    b = _run(cfg, np.exp(2.0 * x))  # estritamente monótona
    assert a == b


def test_irreversibility_zero_for_reversible_process_positive_for_asymmetric(cfg):
    """Todo processo linear gaussiano é reversível no tempo => a estatística é ~0 em esperança.
    Um dente-de-serra (subida lenta, queda brusca) é o caso oposto. Nenhuma feature do banco
    distingue os dois: variância, |e|, e², ρ_k e Haar são simétricos no tempo por construção."""
    rng = np.random.RandomState(2)
    e = rng.randn(4000)
    ar = np.zeros(4000)
    for i in range(1, 4000):
        ar[i] = 0.5 * ar[i - 1] + e[i]
    saw = np.array([(i % 7) - 3.0 for i in range(4000)]) + 0.05 * e

    rev = _run(cfg, ar)["ord_irrev_m3_w100"]
    asym = _run(cfg, saw)["ord_irrev_m3_w100"]
    assert rev < 0.15
    assert asym > 0.5


def test_partial_window_emits_before_it_is_full(cfg):
    """Emitir com janela parcial é decisão de desenho: o viés de contagem depende só de n, e n é
    igual para todas as séries no mesmo t — neutro sob a invariância C1 da TS-AUC. Esperar a janela
    de 100 encher deixaria o bucket 51-150 inteiro em NaN de graça."""
    x = np.random.RandomState(3).randn(60)
    blk = OrdinalBlock()
    blk.reset(None, cfg)
    for t, v in enumerate(x, start=1):
        blk.update(float(v), float(v), float(v), t)
    assert math.isfinite(blk.features()["ord_pe_m3_w100"])
