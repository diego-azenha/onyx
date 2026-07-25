"""A6 (CAMPANHA_POLIMENTO.md) — a estatística e o nulo dela têm de viver na mesma escala.

Dois defeitos, dois flags. O que estes testes travam é: (a) os defaults reproduzem o comportamento
histórico byte-a-byte, e (b) cada flag de fato muda o que se propôs mudar, na direção prevista.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from sbrt.config import load_config
from sbrt.state.conformal import ConformalBlock
from sbrt.state.h0 import fit_h0, whiten_step
from sbrt.utils.ring_buffer import RingBuffer


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def hist():
    """Histórico de cauda pesada, para que o clip em [-8, 8] tenha o que morder."""
    rng = np.random.default_rng(11)
    return rng.standard_t(df=3, size=2000).astype(np.float64)


def test_defaults_permanecem_desligados(cfg):
    assert cfg.conformal.use_raw is False
    assert cfg.h0.null_clip_match is False


def test_historico_de_referencia_do_conformal_nao_e_clipado(cfg, hist):
    """A premissa do A6.1: `sorted_abs_e_hist` preserva extremos além de `clip_e`, então comparar um
    `e` clipado contra ele é assimétrico. Se este teste falhar, o defeito foi corrigido em outro
    lugar e a nota do módulo precisa ser reescrita."""
    h0 = fit_h0(hist, cfg)
    assert h0.sorted_abs_e_hist.max() > cfg.h0.clip_e[1], (
        "o histórico de referência passou a ser clipado — a assimetria do A6.1 mudou de forma"
    )


def test_use_raw_muda_o_p_value_so_alem_do_clip(cfg, hist):
    """Dentro de [-8, 8] as duas variantes têm de ser idênticas (e == e_raw); além, a versão `raw`
    tem de dar um log-martingale ESTRITAMENTE MAIOR — mais evidência, que é o ponto."""
    h0 = fit_h0(hist, cfg)
    cfg_raw = replace(cfg, conformal=replace(cfg.conformal, use_raw=True))

    lo, hi = cfg.h0.clip_e
    dentro, fora = 2.5, 40.0

    def _logm(c, valores):
        b = ConformalBlock()
        b.reset(h0, c)
        for v in valores:
            e_clip = float(np.clip(v, lo, hi))
            b.update(e_clip, float(v), e_clip, 1)
        return b.features()["conformal_logm_abs"]

    assert _logm(cfg, [dentro] * 5) == pytest.approx(_logm(cfg_raw, [dentro] * 5))
    assert _logm(cfg_raw, [fora] * 5) > _logm(cfg, [fora] * 5)


def test_null_clip_match_muda_o_nulo_das_colunas_cal(cfg, hist):
    """O nulo por série é estimado por replay sobre o histórico. Com o flag ligado o replay vê o
    mesmo histórico CLIPADO que a produção vê online, então as estatísticas de escala do nulo mudam
    nas séries de cauda pesada."""
    base = fit_h0(hist, cfg)
    casado = fit_h0(hist, replace(cfg, h0=replace(cfg.h0, null_clip_match=True)))

    assert set(base.null_stats) == set(casado.null_stats)
    difs = [k for k in base.null_stats if base.null_stats[k] != casado.null_stats[k]]
    assert difs, "clipar o histórico do replay não mudou nulo nenhum — o flag não está ligado ao replay"


def test_clip_de_whiten_step_e_o_que_a_producao_ve(cfg, hist):
    """Sanidade do contrato que o A6 audita: `whiten_step` devolve (clipado, cru) e só o primeiro é
    o que a maioria dos blocos consome."""
    h0 = fit_h0(hist, cfg)
    lags = RingBuffer(max(len(h0.phi), h0.seasonal_lag or 0))
    for v in h0.lag_seed:
        lags.push(float(v))
    e_clip, e_raw = whiten_step(1e6, lags, h0, cfg)
    assert e_clip == pytest.approx(cfg.h0.clip_e[1])
    assert e_raw > cfg.h0.clip_e[1] * 10
