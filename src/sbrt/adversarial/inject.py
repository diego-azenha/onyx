"""X4 (BRAINSTORM_RUPTURA_TSAUC.md §3): injeção sintética de quebras em PORTADORES REAIS.

A cauda do histórico de cada série é H0 real e grátis: split em pseudo-história + pseudo-online,
injeta uma quebra em τ′ conhecido, e o mesmo motor de features (`build_training_rows`) produz linhas
supervisionadas sem gap de domínio no portador. Ataca o gargalo silencioso n_eff ≈ 10⁴.

Este módulo cobre as famílias com H0 LIMPO por construção (transformação pontual do portador real, que
em m=0 devolve o portador intocado -- nenhum artefato possível sob H0): média (degrau e rampa) e
variância (escala). Famílias que REGERAM a série (dependência/cauda via modelo ajustado) ficam de fora
até o probe de artefato passar para elas -- elas são o motivo de o probe ser obrigatório.

Determinismo por hash(série, braço): reprodutível e independente de ordem. Magnitudes amostradas das
distribuições `delta_*` do censo A1 (break_type_census.csv), em unidades CRUAS de x (delta_mean_x,
delta_logvar_x) -- sem acoplar com σ_e por-série."""
from __future__ import annotations

import hashlib

import numpy as np

from sbrt.model.dataset import SeriesRecord

FAMILIES = ("mean_step", "mean_ramp", "var", "none")


def rng_for(series_id: int, arm: str, salt: int = 0) -> np.random.Generator:
    h = hashlib.sha256(f"{series_id}:{arm}:{salt}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def _carrier(series_id: int, hist: np.ndarray, n_online: int, min_hist: int, arm: str) -> tuple | None:
    """C2 (BRAINSTORM_RUPTURA_V2.md §1.7): split com comprimento de pseudo-história SORTEADO dentro do
    suporte real. O X4 original cortava sempre `len(hist)−n_online`, empurrando `n_h` para valores fora
    da distribuição (histórias reais têm n≥~1000; um portador de 800 é impossível) — assinatura provável
    do probe 0,9998. Aqui a pseudo-história tem comprimento uniforme em [min_hist, len−n_online], então a
    distribuição de comprimentos fica dentro do suporte. Determinístico por hash. None se não couber."""
    max_hist = len(hist) - n_online
    if max_hist < min_hist:
        return None
    rng = rng_for(series_id, arm + "_carrier")
    hlen = int(rng.integers(min_hist, max_hist + 1))
    return hist[:hlen].copy(), hist[hlen: hlen + n_online].copy()


def _tau_index(rng: np.random.Generator, n_online: int) -> int:
    """τ′ uniforme em [0, n_online-1] INCLUINDO τ′=0 (o caso 'quebra desde o primeiro passo online')."""
    return int(rng.integers(0, n_online))


def make_record(
    series_id: int, hist: np.ndarray, family: str, mag: float, n_online: int, arm: str = "x4",
    min_hist: int = 1000,
) -> SeriesRecord | None:
    """Um portador sintético. `mag` em unidades cruas de x (degrau/±ratio). family='none' = controle-
    espelho (m=0): o portador real intocado, rotulado sem quebra -- neutraliza o efeito-fixo das
    constantes `meta_h0`. `min_hist` (C2): pseudo-história >= isto, no suporte real. None se não couber."""
    carrier = _carrier(series_id, hist, n_online, min_hist, arm)
    if carrier is None:
        return None
    x_hist, x_online = carrier
    rng = rng_for(series_id, arm)

    if family == "none":
        return SeriesRecord(dataset_id=series_id, x_hist=x_hist, x_online=x_online, tau_index=None)

    tau = _tau_index(rng, n_online)
    xo = x_online.copy()
    if family == "mean_step":
        xo[tau:] += mag
    elif family == "mean_ramp":
        ramp = np.linspace(0.0, mag, num=len(xo) - tau)
        xo[tau:] += ramp
    elif family == "var":
        ratio = mag  # >0; passado como razão de desvio
        mu = float(x_online[:tau].mean()) if tau > 0 else float(x_online.mean())
        xo[tau:] = mu + ratio * (x_online[tau:] - mu)
    else:
        raise ValueError(f"família desconhecida: {family!r}")
    return SeriesRecord(dataset_id=series_id, x_hist=x_hist, x_online=xo, tau_index=tau)


def sample_magnitude(rng: np.random.Generator, census_col: np.ndarray, family: str) -> float:
    """Amostra a magnitude da distribuição empírica do censo (|delta| reamostrado). var: exp(±|logvar|)
    vira razão de desvio (>1 ou <1); média: ±|delta_mean_x|."""
    v = float(abs(rng.choice(census_col)))
    if family == "var":
        return float(np.exp(0.5 * (v if rng.random() < 0.5 else -v)))  # razão de DESVIO = exp(0.5*logvar)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    return sign * v
