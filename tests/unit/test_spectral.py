import math

import numpy as np

from sbrt.state.spectral import SpectralBlock

# Níveis de H0 MEDIDOS (40 séries i.i.d. de 2000 pontos, config default):
#   spec_centroid_slow  0,492 +- 0,039
#   spec_entropy_slow   0,982 +- 0,012
#   spec_lowratio_slow  0,346 +- 0,060
# Os limiares abaixo saem daí, não de intuição. A dispersão pequena é consequência direta da média
# de Welch: sem ela a entropia de H0 é 0,81 +- 0,10 e o teste de sinal REPROVA (ver a docstring de
# state/spectral.py).


def _run(cfg, x):
    blk = SpectralBlock()
    blk.reset(None, cfg)
    for t, v in enumerate(x, start=1):
        blk.update(float(v), float(v), float(v), t)
    return blk.features()


def _ar(phi, n, rng):
    e = rng.randn(n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def test_white_noise_gives_flat_spectrum(cfg):
    """Sob H0 o `e` é branco (o AR(10) do histórico o branqueou): espectro chato => entropia
    normalizada perto de 1 e centroide perto de ½. É o ponto de referência que torna as colunas
    comparáveis ENTRE séries sem calibrar nada — e portanto sem injetar ruído transversal."""
    x = np.random.RandomState(0).randn(2000)
    f = _run(cfg, x)
    assert 0.94 < f["spec_entropy_slow"] <= 1.0
    assert abs(f["spec_centroid_slow"] - 0.5) < 0.12


def test_positive_persistence_lowers_centroid_and_entropy(cfg):
    """AR(1) com phi>0 concentra potência em baixa frequência. Medido: centroide 0,297 contra 0,492
    de H0 — 5 desvios."""
    x = _ar(0.6, 2000, np.random.RandomState(1))
    f = _run(cfg, x)
    assert f["spec_centroid_slow"] < 0.38
    assert f["spec_entropy_slow"] < 0.92
    assert f["spec_lowratio_slow"] > 0.50


def test_negative_persistence_raises_centroid(cfg):
    """phi<0 (alternância) empurra potência para alta frequência — o oposto do caso acima. É esta
    assimetria de SINAL que distingue este bloco de `dep_mass`/`mismatch_white`, que somam ρ_k² e
    portanto não sabem para que lado a dependência foi."""
    x = _ar(-0.6, 2000, np.random.RandomState(2))
    f = _run(cfg, x)
    assert f["spec_centroid_slow"] > 0.62
    assert f["spec_lowratio_slow"] < 0.20


def test_invariant_to_scale(cfg):
    """A propriedade que torna este eixo ORTOGONAL ao banco: uma quebra pura de variância multiplica
    todo o fluxo por uma constante e não move nenhuma destas colunas. Elas só podem contribuir onde
    a família de variância (que domina o xs-SHAP) é cega."""
    x = np.random.RandomState(3).randn(2000)
    a = _run(cfg, x)
    b = _run(cfg, 7.5 * x)
    for k in a:
        assert math.isclose(a[k], b[k], rel_tol=1e-9, abs_tol=1e-12), k


def test_warmup_emits_nan_then_finite(cfg):
    x = np.random.RandomState(4).randn(200)
    blk = SpectralBlock()
    blk.reset(None, cfg)
    blk.update(float(x[0]), float(x[0]), float(x[0]), 1)
    assert all(math.isnan(v) for v in blk.features().values())
    for t, v in enumerate(x[1:], start=2):
        blk.update(float(v), float(v), float(v), t)
    assert all(math.isfinite(v) for v in blk.features().values())
