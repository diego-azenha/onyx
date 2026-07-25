"""B3 (2026-07-25): prova que o gate `postprocess.min_t` no caminho ONLINE reproduz bit-a-bit a
transformacao OFFLINE com que o braco foi medido.

Por que este teste existe, e por que ele e o teste critico do B3:

O +0,0009 [+0,0002; +0,0013] na particao 42 e o +0,00094 [+0,00039; +0,00143] na 43 foram medidos por
`scripts/apply_vema_oof.py --min-t 50`, que opera sobre um parquet de OOF: fatia as linhas com
`t > 50` e roda o EWMA assimetrico do ZERO dentro da fatia, por serie. A producao, em contraste, e um
laco online em `state/scorer.py` que carrega `_prev_score` desde `t=1`.

As duas coisas so coincidem se a recursao RECOMECAR na primeira observacao acima do corte. Sem esse
reinicio o `prev` do regime frio (t<=50) vaza para dentro do regime quente, e a producao passa a
computar algo que NENHUMA das duas medicoes viu -- o numero submetido deixaria de ser o numero
medido, silenciosamente e sem erro.
"""
from __future__ import annotations

import numpy as np
import pytest

from sbrt.postprocess.monotonicity import apply as apply_monotonicity, ema_asym_step


class _FakePost:
    def __init__(self, mode, up, dn, min_t):
        self.mode, self.ema_up_alpha, self.ema_down_alpha, self.min_t = mode, up, dn, min_t
        self.soft_decay, self.ema_alpha = 0.02, 0.7


class _FakeCfg:
    def __init__(self, post):
        self.postprocess = post


def _offline(scores: np.ndarray, a_up: float, a_dn: float, min_t: float | None) -> np.ndarray:
    """Replica de `apply_vema_oof.py` para UMA serie com t = 1..n (a mesma fatia + EWMA do zero)."""
    t = np.arange(1, len(scores) + 1)
    out = scores.astype(np.float64).copy()
    hi = t > min_t if min_t is not None else np.ones(len(scores), dtype=bool)
    sub = scores[hi]
    acc = np.empty_like(sub, dtype=np.float64)
    prev = None
    for i in range(len(sub)):  # `_apply_asym`: prev=None no inicio de cada serie/fatia
        acc[i] = sub[i] if prev is None else ema_asym_step(sub[i], prev, a_up, a_dn)
        prev = acc[i]
    out[hi] = acc
    return out


def _online(scores: np.ndarray, a_up: float, a_dn: float, min_t: float | None) -> np.ndarray:
    """Replica do laco de `state/scorer.py:update` -- prev carregado desde t=1, gate via `t`."""
    cfg = _FakeCfg(_FakePost("ema_asym", a_up, a_dn, min_t))
    out = np.empty(len(scores), dtype=np.float64)
    prev = None
    for i, p in enumerate(scores):
        t = i + 1  # scorer.t e 1-based
        s = apply_monotonicity(float(p), prev, "ema_asym", cfg, t)
        out[i] = s
        prev = s
    return out


@pytest.mark.parametrize("min_t", [50, 10, 1, 150])
def test_online_reproduz_offline_com_gate(min_t):
    rng = np.random.default_rng(20260725)
    scores = rng.uniform(0.0, 1.0, size=400)
    off = _offline(scores, 1.0, 0.2, min_t)
    on = _online(scores, 1.0, 0.2, min_t)
    np.testing.assert_array_equal(on, off)  # bit-a-bit, nao `allclose`


def test_sem_gate_tambem_coincide():
    """`min_t=None` = sem gate: o online tem de bater com o EWMA puro desde t=1."""
    rng = np.random.default_rng(1)
    scores = rng.uniform(0.0, 1.0, size=200)
    np.testing.assert_array_equal(_online(scores, 1.0, 0.2, None), _offline(scores, 1.0, 0.2, None))


def test_abaixo_do_corte_e_identidade():
    """A assinatura que as duas medicoes mostraram: `t<=50` da EXATAMENTE 0,0000 de delta."""
    rng = np.random.default_rng(7)
    scores = rng.uniform(0.0, 1.0, size=60)
    on = _online(scores, 1.0, 0.2, 50)
    np.testing.assert_array_equal(on[:50], scores[:50])
    assert on[50] == scores[50], "t=51 e o REINICIO da recursao: tem de sair intacto"


def test_com_a_up_1_so_as_DESCIDAS_sao_suavizadas():
    """Propriedade do par (a_up=1,0 / a_dn=0,2) adotado, que nao e obvia e quase me enganou ao
    escrever este arquivo: com alpha=1 na subida, `1,0*p + 0,0*prev = p`, entao TODO passo de subida
    e identidade. O pos-processo age exclusivamente nas descidas -- e por isso e um max-hold suave,
    e nao um EWMA simetrico. Se alguem trocar `ema_up_alpha` por <1, esta invariante quebra e o
    braco medido deixa de ser o braco implantado."""
    scores = np.array([0.10, 0.90, 0.20, 0.95, 0.30], dtype=np.float64)  # t=1..5
    on = _online(scores, 1.0, 0.2, None)
    assert on[0] == scores[0]                       # t=1: sem prev
    assert on[1] == scores[1]                       # subida 0,10 -> 0,90: intacta
    assert on[2] == pytest.approx(0.2 * 0.20 + 0.8 * 0.90)   # descida: suavizada
    assert on[3] == scores[3]                       # subida: intacta de novo
    assert on[4] < scores[3] and on[4] > scores[4]  # descida: fica entre prev e p


def test_o_reinicio_importa():
    """Guarda-costas: uma implementacao SEM o reinicio em t=min_t+1 passaria nos outros testes se o
    corte fosse trivial, mas diverge aqui. Prova que (b) no gate nao e decorativo."""
    rng = np.random.default_rng(99)
    scores = rng.uniform(0.0, 1.0, size=120)
    correto = _online(scores, 1.0, 0.2, 50)

    # variante ERRADA: aplica o EWMA em t>50 mas herda o prev do regime frio
    errado = np.empty_like(scores, dtype=np.float64)
    prev = None
    for i, p in enumerate(scores):
        t = i + 1
        errado[i] = p if t <= 50 or prev is None else ema_asym_step(p, prev, 1.0, 0.2)
        prev = errado[i]

    assert not np.array_equal(correto, errado), (
        "sem o reinicio a serie diverge -- se este assert falhar, o gate perdeu a condicao (b)"
    )
