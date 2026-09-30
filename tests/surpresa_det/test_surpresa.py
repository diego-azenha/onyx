"""Testes do detector de surpresa acumulada (knowledge/frentes/surpresa-acumulada/S0-sanidade-sintetica.md)."""
import numpy as np
import pytest

from surpresa import carregar_config
from surpresa.evidencia import Alternativas
from surpresa.modelo import ajustar, fluxos
from surpresa.stream import SurpresaStream, pontuar_serie

from .geradores import GERADORES, _garch


@pytest.fixture(scope="module")
def cfg():
    return carregar_config()


@pytest.fixture(scope="module")
def alt(cfg):
    return Alternativas(cfg)


@pytest.mark.parametrize("tipo", sorted(GERADORES))
def test_nulo_universal(cfg, tipo):
    """Sob H0, g_A e g_B online são ~N(0,1) em qualquer tipo de série."""
    rng = np.random.default_rng(7)
    ga, gb = [], []
    for _ in range(40):
        x = GERADORES[tipo](rng, 2600)
        f = fluxos(ajustar(x[:2000], cfg), x[2000:])
        ga.append(f["g_a"])
        gb.append(f["g_b"])
    ga, gb = np.concatenate(ga), np.concatenate(gb)
    for g in (ga, gb):
        assert abs(g.mean()) < 0.05
        assert 0.93 < g.std() < 1.07


def test_media_nula_bate_monte_carlo(alt):
    rng = np.random.default_rng(1)
    g = rng.standard_normal(400_000)
    from scipy.special import ndtr
    L = alt.incrementos(g[1:], ndtr(g[1:]), g[1:], g[:-1])
    np.testing.assert_allclose(L.mean(axis=0), alt.media_nula(60), atol=4e-3)


def test_stream_igual_lote(cfg, alt):
    rng = np.random.default_rng(3)
    h, on = _garch(rng, 1500), _garch(rng, 300)
    lote = pontuar_serie(h, on, cfg, alt)
    s = SurpresaStream(h, cfg, alt)
    for i, x in enumerate(on):
        o = s.update(x)
        for k, v in o.items():
            assert abs(v - lote[k][i]) < 1e-9, (k, i)


def test_causal_prefixo(cfg, alt):
    """O score em t não depende de pontos depois de t."""
    rng = np.random.default_rng(4)
    h, on = rng.standard_normal(1200), rng.standard_normal(400)
    cheio = pontuar_serie(h, on, cfg, alt)
    for k in (1, 57, 250):
        pref = pontuar_serie(h, on[:k], cfg, alt)
        for nome in cheio:
            np.testing.assert_array_equal(pref[nome], cheio[nome][:k])


def test_poder_escala_1_2(cfg, alt):
    """sigma 1,0 -> 1,2 desde t=1: a família escala_sobe separa quebra de controle até t=150."""
    rng = np.random.default_rng(5)
    q, c = [], []
    for _ in range(60):
        h = rng.standard_normal(2000)
        on = rng.standard_normal(150)
        c.append(pontuar_serie(h, on, cfg, alt)["fam_escala_sobe"][-1])
        q.append(pontuar_serie(h, on * 1.2, cfg, alt)["fam_escala_sobe"][-1])
    assert np.median(q) > np.quantile(c, 0.95)
