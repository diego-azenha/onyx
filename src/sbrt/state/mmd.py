"""MMDBlock — Maximum Mean Discrepancy de kernel via Random Fourier Features, online
(docs/PROPOSTA_FEATURES_V2.md F3; Rahimi & Recht 2007; Keriven et al., "NEWMA", 2020).

Motivação (proposta §4-F3): todo o banco atual resume o fluxo de inovações em **momentos de baixa
ordem** (média, variância, curtose via nu_hat) ou em **funcionais do contraste de CDFs empíricas**
(R4: Wilcoxon, chi²-forma). Um MMD com kernel característico (gaussiano) captura *todas* as
diferenças distribucionais num único escalar — é diferente em espécie, não mais um funcional do
mesmo objeto.

Como funciona. z(x) = sqrt(2/D)·cos(Wᵀx + b) aproxima o mapa de características do kernel gaussiano
(k(x,y) ≈ z(x)·z(y)). Então MMD²(P,Q) ≈ ‖E_P z − E_Q z‖². Aqui:
- E_Q z = `h0.rff_href`, a média de z sobre o **histórico congelado** (calculada uma vez em `fit_h0`);
- E_P z = duas EWMAs de z(e_t) com fatores de esquecimento diferentes (rápido ≈ janela 1/λ_fast,
  lento ≈ 1/λ_slow), sem armazenar amostra nenhuma — O(D) por passo, O(D) de memória.

Três estatísticas por espaço: distância da EWMA rápida ao histórico, da lenta ao histórico, e entre
as duas EWMAs (o estatístico NEWMA propriamente dito, que compara "recente" com "menos recente" sem
referência externa e por isso reage a mudanças graduais que o histórico congelado dilui).

Dois espaços:
- **marginal**, sobre e_t: sensível a qualquer mudança da distribuição marginal (média, variância,
  assimetria, cauda) de uma vez só;
- **conjunto**, sobre o par (e_t, e_{t−1}): sensível a mudanças da distribuição CONJUNTA, isto é, da
  estrutura de **dependência** — um detector não-paramétrico de quebra de dependência que nenhuma
  feature atual cobre (as existentes são ρ₁ de Fisher-z e CUSUMs de produto defasado, ambos
  paramétricos e de segunda ordem).

Opera sobre `e` (escala congelada do histórico), não `e_vol`: a referência `rff_href` foi construída
a partir do resíduo do histórico — comparar `e_vol` contra ela seria inconsistente de escala quando
o ajuste de volatilidade está ativo (mesma convenção de `ConformalBlock`/`RankTwoSampleBlock`).

**Comparabilidade transversal (crítico):** W e b são sorteados UMA vez a partir de uma seed fixa de
módulo (`_RFF_SEED`) e compartilhados por TODAS as séries e execuções — são uma tabela de constantes
determinística, não um sorteio por série. Se cada série tivesse seu próprio W, os MMDs não seriam
comparáveis entre séries e a TS-AUC (que ordena séries no mesmo passo) ficaria envenenada. A seed é
uma constante dedicada, deliberadamente NÃO `cfg.seed`, para que trocar a seed de CV do modelo não
mude silenciosamente a definição das features.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter  # topo, não dentro da função: o conversor do Crunch avisa em
                                  # import aninhado (pode não virar requirement do submission)

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

# Constante dedicada (ver docstring): a tabela de RFF é parte da DEFINIÇÃO das features.
_RFF_SEED = 20260720

_TABLE_CACHE: dict = {}


def rff_table(n_features: int, bandwidth: float, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """(W, b) determinísticos e cacheados por (D, bandwidth, dim). W: (dim, D); b: (D,)."""
    key = (n_features, float(bandwidth), int(dim))
    cached = _TABLE_CACHE.get(key)
    if cached is None:
        rng = np.random.default_rng(_RFF_SEED + 1_000_003 * dim + n_features)
        W = rng.normal(0.0, 1.0 / max(bandwidth, 1e-12), size=(dim, n_features))
        b = rng.uniform(0.0, 2.0 * math.pi, size=n_features)
        cached = (W, b)
        _TABLE_CACHE[key] = cached
    return cached


def rff_map(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """x: (n, dim) -> z: (n, D), com z(x) = sqrt(2/D)·cos(Wᵀx + b)."""
    D = W.shape[1]
    return math.sqrt(2.0 / D) * np.cos(np.asarray(x, dtype=np.float64) @ W + b)


def history_reference(e_hist: np.ndarray, cfg: "Config") -> tuple[np.ndarray, np.ndarray]:
    """Médias de z sobre o histórico, marginal e conjunta — a referência congelada de H0.
    Chamado uma vez por série em `fit_h0`."""
    m_cfg = cfg.mmd
    e = np.asarray(e_hist, dtype=np.float64)

    W1, b1 = rff_table(m_cfg.n_features, m_cfg.bandwidth, 1)
    href = rff_map(e.reshape(-1, 1), W1, b1).mean(axis=0)

    W2, b2 = rff_table(m_cfg.n_features, m_cfg.bandwidth, 2)
    if len(e) >= 2:
        pairs = np.column_stack([e[1:], e[:-1]])
        href_joint = rff_map(pairs, W2, b2).mean(axis=0)
    else:
        href_joint = np.zeros(m_cfg.n_features, dtype=np.float64)
    return href, href_joint


class MMDBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        m_cfg = cfg.mmd
        self.D = m_cfg.n_features
        self.lam_vfast = m_cfg.lambda_vfast
        self.lam_fast = m_cfg.lambda_fast
        self.lam_slow = m_cfg.lambda_slow
        self.warmup_min_n = cfg.features.warmup_min_n
        self.t = 0

        self.W1, self.b1 = rff_table(self.D, m_cfg.bandwidth, 1)
        self.W2, self.b2 = rff_table(self.D, m_cfg.bandwidth, 2)
        self.W1_flat = self.W1[0]  # (D,) — entrada escalar, evita matmul por passo
        self.scale = math.sqrt(2.0 / self.D)

        self.href = np.asarray(h0.rff_href, dtype=np.float64)
        self.href_joint = np.asarray(h0.rff_href_joint, dtype=np.float64)

        # EWMAs inicializadas NA referência: o estatístico começa em 0 (nenhuma evidência de
        # desvio) e cresce à medida que a distribuição recente se afasta do H0 — mais honesto no
        # warm-up do que inicializar em 0, que fabricaria um MMD grande espúrio no passo 1.
        self.m_vfast = self.href.copy()
        self.m_fast = self.href.copy()
        self.m_slow = self.href.copy()
        self.mj_vfast = self.href_joint.copy()
        self.mj_fast = self.href_joint.copy()
        self.mj_slow = self.href_joint.copy()

        self.prev_e: float | None = None

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t

        z = self.scale * np.cos(self.W1_flat * e + self.b1)
        self.m_vfast += self.lam_vfast * (z - self.m_vfast)
        self.m_fast += self.lam_fast * (z - self.m_fast)
        self.m_slow += self.lam_slow * (z - self.m_slow)

        if self.prev_e is not None:
            zj = self.scale * np.cos(self.W2[0] * e + self.W2[1] * self.prev_e + self.b2)
            self.mj_vfast += self.lam_vfast * (zj - self.mj_vfast)
            self.mj_fast += self.lam_fast * (zj - self.mj_fast)
            self.mj_slow += self.lam_slow * (zj - self.mj_slow)
        self.prev_e = e

    def features(self) -> dict[str, float]:
        names = (
            "mmd_marginal_vfast", "mmd_marginal_fast", "mmd_marginal_slow", "mmd_marginal_newma",
            "mmd_joint_vfast", "mmd_joint_fast", "mmd_joint_slow", "mmd_joint_newma",
        )
        if self.t < self.warmup_min_n:
            return {n: math.nan for n in names}

        def _dist(a: np.ndarray, b: np.ndarray) -> float:
            d = a - b
            return float(math.sqrt(max(float(d @ d), 0.0)))

        return {
            "mmd_marginal_vfast": _dist(self.m_vfast, self.href),
            "mmd_marginal_fast": _dist(self.m_fast, self.href),
            "mmd_marginal_slow": _dist(self.m_slow, self.href),
            "mmd_marginal_newma": _dist(self.m_fast, self.m_slow),
            "mmd_joint_vfast": _dist(self.mj_vfast, self.href_joint),
            "mmd_joint_fast": _dist(self.mj_fast, self.href_joint),
            "mmd_joint_slow": _dist(self.mj_slow, self.href_joint),
            "mmd_joint_newma": _dist(self.mj_fast, self.mj_slow),
        }


def history_series(e_hist: np.ndarray, href: np.ndarray, href_joint: np.ndarray, cfg: "Config") -> dict:
    """As MESMAS seis estatísticas, calculadas sobre o histórico de forma vetorizada, para a
    calibração de nulo por série (F1, `state/calibration.py`).

    Não é o "backtest vetorizado" proibido pelo §13.2 do plano: não produz features de treino nem
    scores — produz uma constante por série (média/desvio do nulo) a partir de dados que já são H0
    por definição. Ainda assim, a equivalência com o laço online é verificada em
    `tests/unit/test_mmd.py::test_history_series_matches_online_block`."""
    m_cfg = cfg.mmd
    e = np.asarray(e_hist, dtype=np.float64)
    n = len(e)
    if n < 3:
        return {}

    W1, b1 = rff_table(m_cfg.n_features, m_cfg.bandwidth, 1)
    W2, b2 = rff_table(m_cfg.n_features, m_cfg.bandwidth, 2)

    def _ewma(x: np.ndarray, lam: float, init: np.ndarray) -> np.ndarray:
        # y[t] = (1-lam) y[t-1] + lam x[t], com y[-1] = init (estado zi = (1-lam)*init)
        zi = np.tile((1.0 - lam) * init, (1, 1))
        y, _ = lfilter([lam], [1.0, -(1.0 - lam)], x, axis=0, zi=zi)
        return y

    z = rff_map(e.reshape(-1, 1), W1, b1)
    m_vfast = _ewma(z, m_cfg.lambda_vfast, href)
    m_fast = _ewma(z, m_cfg.lambda_fast, href)
    m_slow = _ewma(z, m_cfg.lambda_slow, href)

    pairs = np.column_stack([e[1:], e[:-1]])
    zj = rff_map(pairs, W2, b2)
    mj_vfast = _ewma(zj, m_cfg.lambda_vfast, href_joint)
    mj_fast = _ewma(zj, m_cfg.lambda_fast, href_joint)
    mj_slow = _ewma(zj, m_cfg.lambda_slow, href_joint)

    def _norm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(((a - b) ** 2).sum(axis=1), 0.0))

    return {
        "mmd_marginal_vfast": _norm(m_vfast, href),
        "mmd_marginal_fast": _norm(m_fast, href),
        "mmd_marginal_slow": _norm(m_slow, href),
        "mmd_marginal_newma": _norm(m_fast, m_slow),
        "mmd_joint_vfast": _norm(mj_vfast, href_joint),
        "mmd_joint_fast": _norm(mj_fast, href_joint),
        "mmd_joint_slow": _norm(mj_slow, href_joint),
        "mmd_joint_newma": _norm(mj_fast, mj_slow),
    }
