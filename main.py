#%pip install crunch-cli lightgbm scikit-learn scipy joblib tqdm pyyaml --upgrade --quiet --progress-bar off
#!crunch setup-notebook structural-break-real-time COLE_SEU_TOKEN_AQUI


import crunch

#crunch_tools = crunch.load_notebook()


# @crunch/keep:on
# Configuração (configs/default.yaml embutida) — única fonte de números do pipeline.
_EMBEDDED_YAML = r'''
# Single source of truth for every number used by src/sbrt (plan checklist §8: "nenhum número
# mágico no código"). References `§N` point at docs/PLANO_TECNICO.md.
seed: 42

h0:                                    # plano §3.1
  ar_order: 10
  min_hist_len: 50
  seasonal_acf_threshold: 0.25
  seasonal_lag_range: [6, 128]
  ar_r2_min_reduction: 0.02            # accept AR(p) only if var(resid)/var(x) <= 1 - this
  nu_clip: [5.0, 50.0]
  quantile_levels: [0.01, 0.05, 0.10, 0.25, 0.75, 0.90, 0.95, 0.99]
  clip_e: [-8.0, 8.0]

state:                                 # plano §4.2
  ewma_lambdas: [0.05, 0.10, 0.30]
  window_sizes: [10, 25, 50, 100, 250]
  exceedance_windows: [50, 250]
  sign_windows: [50, 250]
  vol_adjust: {threshold_rho1_abs: 0.15, lambda_v: 0.06}
  sign_bernoulli: {p0: 0.5, p1_pos: 0.65, p1_neg: 0.35}
  exceedance_bernoulli: {q95: {p0: 0.05, p1: 0.15}, q99: {p0: 0.01, p1: 0.05}}
  dependence_delta_u: 0.3
  skew_window: 250
  quantile_crossing_window: 100
  dependence_window: 100
  hedge_window: 100
  hedge_ewma_lambda: 0.1

cusum:                                 # plano §4.2, tabela #4/#8/#11/#13/#15
  mean_deltas: [0.25, 0.5, 1.0]
  var_ratios_up: [1.5, 2.5]
  var_ratio_down: 0.5
  protected_recent_ages: 8             # not used directly here, kept for symmetry with bayes pruning docs

bayes:                                 # plano §4.3
  hazards: [0.02, 0.01, 0.0025]        # 1/50, 1/100, 1/400 -- 1/50 acrescentado (docs/DIAGNOSTICO_TS_AUC.md,
                                        # direcao 4): reacao mais rapida em t baixo, onde a mediana de
                                        # tau_index de treino e 184 e o quartil inferior e 67
  max_candidates: 48
  protect_recent: 8
  prior: {mu0: 0.0, kappa0: 0.5, nu0: 2.0, sigma0_sq: 1.5}
  logw_renorm_threshold: 600.0

conformal:                            # plano §4.2 #23
  epsilons: [0.05, 0.1, 0.2, 0.4]
  reset_epsilons: [0.05, 0.1, 0.2, 0.4]

rank_twosample:                       # R4 (docs/PARECER_AUDITORIA_ONYX.md §6-R4): duas amostras
  windows: [25, 100]                  # rank-based janela-vs-histórico -- analogo causal de 2025

dependence:                           # P1 (docs/INVESTIGACAO_FALHAS_V3.md): dependencia nao-linear
  windows: [100]                      # rho1 de |e| e e^2 -- w050 PODADA (morta no SHAP do V4, rank 164+)
  mass_window: 100                    # massa multi-lag Sum rho_k^2
  mass_max_lag: 5

lmoments:                             # P2 -- PODADA do pipeline (0,51% SHAP por ~65us; ver scorer.py).
  windows: [50, 100]                  # config mantida para o bloco/teste standalone reabrivel

bocpd:                                # BOCPD (Adams-MacKay): posterior de run-length de variancia
  r_max: 256                          # ~30 us/passo (medido); localizacao principiada (vs varloc heuristico)
  hazard_lambda: 250.0                # H = 1/lambda
  alpha0: 2.0                         # prior IG -> preditiva t_4 (variancia 1, robusta a cauda)
  beta0: 1.0
  recent_k: 5                         # cp_prob = P(run-length < 5)

varloc:                               # P3 (docs/INVESTIGACAO_FALHAS_V3.md): variancia localizada
  scales: [10, 25, 50, 100, 250]      # max/min do z de variancia sobre escalas (localiza o changepoint)
  recent: 25                          # contraste recente-vs-defasado
  lagged: 100

jumps:                                # P4 (docs/INVESTIGACAO_FALHAS_V3.md): bipower/saltos + leverage
  windows: [50, 100]                  # RV/BV (razao de salto), semivariancia RS+/RS-, leverage

# --- proposta V2 (docs/PROPOSTA_FEATURES_V2.md) ---
mmd:                                  # F3: MMD de kernel via Random Fourier Features (NEWMA)
  n_features: 64                      # D; custo medido ~9 us/passo (D=64), ~11 us (D=128)
  bandwidth: 1.0                      # e ja e padronizado por sigma_e -> sigma=1 e a escala natural
  lambda_vfast: 0.08                  # janela efetiva ~12 passos -- cobre o regime t<=50, onde
                                      # fast/slow ainda nao aqueceram (docs/RESULTADOS_FEATURES_V2 §3)
  lambda_fast: 0.02                   # janela efetiva ~50 passos
  lambda_slow: 0.005                  # janela efetiva ~200 passos

multiscale:                           # F4: energia por escala (Haar diadico causal), ~4 us/passo
  n_scales: 5                         # escala j produz um coeficiente a cada 2^(j+1) amostras
  ewma_lambda: 0.05
  warmup_min_coeffs: 3                # NaN ate a escala ter 3 coeficientes (honesto, nao inventado)

h0_fingerprint:                       # F2: descritores estendidos de H0, custo ZERO por passo
  hill_frac: 0.05                     # fracao superior de |e| usada pelo estimador de Hill
  acf_max_lag: 10
  hurst_scales: [1, 2, 4, 8, 16]
  volvol_window: 50

calibration:                          # F1: nulo por serie medido sobre o proprio historico
  enabled: true
  shrink_pseudo: 10.0                 # encolhe sd empirico -> teorico i.i.d. com peso n_eff/(n_eff+10)

features:
  warmup_min_n: 5                     # below this, window/EWMA-derived stats emit NaN

lightgbm:                              # plano §8.3
  learning_rate: 0.05
  num_leaves: 63
  max_depth: -1
  min_data_in_leaf: 200
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  lambda_l2: 5.0
  n_estimators_cap: 1500
  early_stopping_rounds: 100
  max_bin: 255
  deterministic: true
  force_row_wise: true
  train_num_threads: 8
  predict_num_threads: 1
  n_folds: 5
  feval_max_valid_rows: null           # R2 (parecer §6-R2): subamostra determinística do feval de
                                        # AUC-por-passo. MEDIDO (retreino real, 2026-07-20): uma
                                        # subamostra fixa de 150k linhas usada como criterio de
                                        # parada por ~150-200 rodadas sofre "winner's curse" --
                                        # o round-argmax fica otimisticamente enviesado para ESSA
                                        # subamostra fixa e nao generaliza para o fold inteiro.
                                        # Usar o fold inteiro reduz essa fonte de ruido; reativar
                                        # (>0) só se o custo por rodada em datasets maiores exigir e
                                        # com subamostra bem maior (>= algumas centenas de milhares).
  early_stopping_metric: logloss       # ver comentário em src/sbrt/config.py:LightGBMConfig --
                                        # "ts_auc_by_t" sozinho (mesmo sem subamostra) regrediu a
                                        # TS-AUC OOF real (Delta -0.0099, IC exclui 0); "logloss"
                                        # reproduz o comportamento original, validado neste retreino
                                        # (Delta -0.0014, IC inclui 0 -- indistinguível do baseline).

rank:                                   # R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): objetivo de
  objective: lambdarank                 # ranking por grupo t, membro PARALELO do ensemble binario
  label_gain: [0, 1]                    # (nao substitui) -- ver model/train.py:train_rank
  truncation_level_cap: 300             # ver comentario em config.py:RankObjectiveConfig -- grupos
                                         # de t<=100 chegam a ~8000 linhas/fold; sem cap o treino
                                         # trava (custo ~group_size*truncation_level por grupo)

thinning:                              # plano §8.1
  full_until: 100
  step_101_400: 2
  step_401_plus: 4

model:
  mode: supervised                     # fallback | supervised — supervised requires artifacts/models/vN
  dataset_n_jobs: -1                   # paralelismo entre séries ao construir o dataset de treino (§8.1);
                                        # -1 = todos os núcleos (joblib); 1 = serial (comportamento original)

fallback:                              # plano §8.5 — caminho de emergência determinístico
  w_lo: 0.9                            # peso do log-odds bayesiano (hazard 1/400)
  w_cusum: 0.4                         # peso do max do banco de CUSUM (via sqrt(2*LLR), escala z)
  w_conformal: 0.3                     # peso do log-martingale conformal (variante reset)
  bias: 0.0                            # a calibrar no treino para mediana 0.5 em séries sem quebra (§8.5)

gates:                                  # plano §10 (tabela revisada) e §11.1 — COMPORTAMENTAIS, nunca TS-AUC (§9.0)
  drift_slope_abs_max: 1.0e-4
  # 300 era um alvo de engenharia conservador, não o orçamento real. Medido: build de features
  # ~233us/passo + predict ~330us estimado = ~560us/passo; orçamento real (15h/semana / 1e7 passos)
  # = ~5400us/passo -> ~16x de folga. Ver plano_acao_v1_para_v2.md §1.4. Subir este número não é
  # licença para inflar custo sem necessidade, mas 300 estava barrando trabalho legítimo.
  latency_budget_us_per_step: 1500
  t1:  {median_min: 0.65, control_median_max: 0.25, t_from: 50}
  t2:  {mean_prebreak_max: 0.35}
  t3:  {gap_min: 0.15, t_offset: 200}
  t4:  {median_min: 0.75, control_median_max: 0.20, t_offset: 15}
  t5:  {gap_min: 0.35, t_offset: 100}
  t5b: {gap_min: 0.20, t_offset: 200}
  t6:  {mean_max: 0.40}
  t7:  {gap_min: 0.20, t_offset: 150}
  t8:  {gap_min: 0.15, t_offset: 200}
  t9:  {final_max: 0.35, decay_min: 0.1}
  t10: {mean_max: 0.40}
  t11: {}
  t12: {}
  t12b: {}
  t13: {decay_min: 0.15}

submission:                             # plano §9.3 — G-0/G-mono/G-peso decididos por submissão oficial,
  log_path: "artifacts/reports/submission_log.md"   # nunca por número calculado localmente

postprocess:
  mode: free                           # free | hold | soft | ema — plano §7
  soft_decay: 0.02
  ema_alpha: 0.7

'''


# @crunch/keep:on
# ============================== sbrt/utils/numerics.py ==============================
"""Primitivas numéricas reaproveitadas por múltiplos state blocks (plano §4, docs/CONTRACTS.md).

NUNCA reimplementar Welford ou logsumexp localmente em outro arquivo — é a defesa contra a classe de
bug "duas implementações que divergem depois de 500 passos por causa de um `if` diferente"
(docs/PLANO_REPOSITORIO.md §1).
"""

import math
from typing import Sequence


def welford_update(n: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
    """Recursão de Welford (1962) — atualização estável de média e soma de quadrados."""
    n_new = n + 1
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2 + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def logsumexp(values: Sequence[float]) -> float:
    """log(sum(exp(values))), estável numericamente. Ignora -inf (candidatos mortos)."""
    finite_max = -math.inf
    for v in values:
        if v > finite_max:
            finite_max = v
    if finite_max == -math.inf:
        return -math.inf
    total = 0.0
    for v in values:
        if v != -math.inf:
            total += math.exp(v - finite_max)
    return finite_max + math.log(total)


_LGAMMA_CACHE: dict = {}


def lgamma_cached(x: float) -> float:
    """math.lgamma com cache — plano §4.3: nu_n = nu0 + n_j cresce em passos inteiros, então os
    mesmos argumentos reaparecem massivamente entre candidatos e passos.

    Memo explícito em vez de `@lru_cache`: no notebook de submissão o pipeline é achatado em
    `__main__`, e `functools._lru_cache_wrapper` se serializa POR REFERÊNCIA (nome qualificado). O
    worker do loky (spawn) não tem esse nome no seu `__main__` e o `train()` paralelo morre com
    `Can't get attribute 'lgamma_cached'`. Uma função normal + dict são serializados por VALOR pelo
    cloudpickle e atravessam o fork/spawn intactos. Valores são idênticos aos do lru_cache."""
    hit = _LGAMMA_CACHE.get(x)
    if hit is not None:
        return hit
    if len(_LGAMMA_CACHE) >= 4096:  # mesmo teto do lru_cache anterior
        _LGAMMA_CACHE.clear()
    value = _LGAMMA_CACHE[x] = math.lgamma(x)
    return value


def ewma_update(prev: float, x: float, lam: float) -> float:
    return (1.0 - lam) * prev + lam * x


# @crunch/keep:on
# ============================== sbrt/utils/ring_buffer.py ==============================
"""Buffer circular O(1) — usado por accumulators.py (janelas rodantes) e h0.py (lags do AR,
atravessando a fronteira histórico->online, plano §3.1 item 8 / armadilha §13.3)."""


class RingBuffer:
    """Capacidade fixa. `push` é O(1); `peek(age)` lê o valor `age` passos atrás sem alocar.

    Convenção de idade: logo após um `push(x)`, `peek(0) == x` (o mais recente), `peek(1)` é o
    penúltimo, etc. Para janelas rodantes de tamanho w, o valor que *sai* da janela ao inserir uma
    nova observação é `peek(w - 1)` chamado ANTES do push (ver state/accumulators.py).
    """

    __slots__ = ("capacity", "_buf", "_head", "_count")

    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity deve ser >= 1")
        self.capacity = capacity
        self._buf = [0.0] * capacity
        self._head = -1
        self._count = 0

    def push(self, x: float) -> float | None:
        """Insere x; retorna o valor expulso (se o buffer já estava cheio) ou None."""
        self._head = (self._head + 1) % self.capacity
        evicted = self._buf[self._head] if self._count >= self.capacity else None
        self._buf[self._head] = x
        if self._count < self.capacity:
            self._count += 1
        return evicted

    def peek(self, age: int) -> float:
        if age < 0 or age >= self.capacity:
            raise IndexError(f"age {age} fora de [0, {self.capacity})")
        idx = (self._head - age) % self.capacity
        return self._buf[idx]

    def __len__(self) -> int:
        return self._count


# @crunch/keep:on
# ============================== sbrt/config.py ==============================
"""Typed loader for configs/*.yaml — the single source of truth for every hyperparameter (plan §4)."""

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = None


@dataclass(frozen=True)
class H0Config:
    ar_order: int
    min_hist_len: int
    seasonal_acf_threshold: float
    seasonal_lag_range: tuple
    ar_r2_min_reduction: float
    nu_clip: tuple
    quantile_levels: tuple
    clip_e: tuple


@dataclass(frozen=True)
class StateConfig:
    ewma_lambdas: tuple
    window_sizes: tuple
    exceedance_windows: tuple
    sign_windows: tuple
    vol_adjust: dict
    sign_bernoulli: dict
    exceedance_bernoulli: dict
    dependence_delta_u: float
    skew_window: int
    quantile_crossing_window: int
    dependence_window: int
    hedge_window: int
    hedge_ewma_lambda: float


@dataclass(frozen=True)
class CusumConfig:
    mean_deltas: tuple
    var_ratios_up: tuple
    var_ratio_down: float
    protected_recent_ages: int


@dataclass(frozen=True)
class BayesConfig:
    hazards: tuple
    max_candidates: int
    protect_recent: int
    prior: dict
    logw_renorm_threshold: float


@dataclass(frozen=True)
class ConformalConfig:
    epsilons: tuple
    reset_epsilons: tuple


@dataclass(frozen=True)
class RankTwoSampleConfig:
    windows: tuple  # R4 (docs/PARECER_AUDITORIA_ONYX.md §6-R4): janelas para os testes de duas
    # amostras rank-based janela-vs-histórico (localização/dispersão Wilcoxon-like + forma chi2)


@dataclass(frozen=True)
class DependenceConfig:
    """P1 (docs/INVESTIGACAO_FALHAS_V3.md): dependência serial não-linear/multi-lag."""
    windows: tuple      # janelas para ρ₁ de |e| e e² (clustering de volatilidade)
    mass_window: int    # janela para a massa multi-lag Σρ_k²
    mass_max_lag: int   # L da massa multi-lag


@dataclass(frozen=True)
class LMomentConfig:
    """P2 (docs/INVESTIGACAO_FALHAS_V3.md): forma de cauda dinâmica via L-momentos."""
    windows: tuple      # janelas para L-skewness/L-kurtosis


@dataclass(frozen=True)
class VarLocConfig:
    """P3 (docs/INVESTIGACAO_FALHAS_V3.md): variância localizada no changepoint."""
    scales: tuple       # escalas de janela para o max/min do z de variância
    recent: int         # janela recente do contraste recente-vs-defasado
    lagged: int         # comprimento da janela defasada


@dataclass(frozen=True)
class JumpConfig:
    """P4 (docs/INVESTIGACAO_FALHAS_V3.md): bipower/saltos + leverage."""
    windows: tuple      # janelas para RV/BV, semivariância, leverage


@dataclass(frozen=True)
class BOCPDConfig:
    """BOCPD (Adams-MacKay 2007): posterior de run-length de variância (docs/RESULTADOS_P1_P4.md)."""
    r_max: int          # truncagem do run-length (O(R_max)/passo)
    hazard_lambda: float  # hazard H = 1/lambda (prior geométrico no run-length)
    alpha0: float       # prior Inverse-Gamma da variância do regime
    beta0: float
    recent_k: int       # cp_prob = soma de p(r) para r < recent_k


@dataclass(frozen=True)
class RankObjectiveConfig:
    objective: str  # "lambdarank" ou "rank_xendcg" -- R3 (parecer §6-R3), membro paralelo do
    # ensemble binário, query=passo t.
    label_gain: tuple
    truncation_level_cap: int  # `lambdarank_truncation_level` = min(maior grupo do fold, este cap).
    # MEDIDO (retreino real, 2026-07-20): t<=100 mantém TODAS as ~10000 séries vivas (thinning só
    # começa em t>100, configs/default.yaml:thinning), então o maior grupo de um fold chega a ~8000
    # linhas -- truncation_level sem cap (a recomendação literal do parecer, "≥ tamanho máximo de
    # grupo") faz o custo por grupo escalar ~group_size×truncation_level e trava o treino (processo
    # rodou >4h sem terminar, matado manualmente). Um cap moderado ainda cobre a imensa maioria dos
    # grupos por inteiro (grupos ficam pequenos rapidamente após o thinning) e mantém o treino
    # tratável; grupos maiores que o cap ficam com gradiente pleno só no topo -- risco documentado,
    # aceito por tratabilidade (ver docstring de model/train.py:train_rank).


@dataclass(frozen=True)
class MMDConfig:
    """F3 (docs/PROPOSTA_FEATURES_V2.md): MMD de kernel via Random Fourier Features."""
    n_features: int
    bandwidth: float
    lambda_vfast: float  # janela efetiva curta -- existe para o regime de t pequeno, onde
    # `lambda_fast`/`lambda_slow` ainda não aqueceram e a família ficava 100% NaN
    lambda_fast: float
    lambda_slow: float


@dataclass(frozen=True)
class MultiScaleConfig:
    """F4: decomposição causal de energia por escala (Haar diádico)."""
    n_scales: int
    ewma_lambda: float
    warmup_min_coeffs: int


@dataclass(frozen=True)
class H0FingerprintConfig:
    """F2: descritores estendidos do regime H0 (state/fingerprint.py)."""
    hill_frac: float
    acf_max_lag: int
    hurst_scales: tuple
    volvol_window: int


@dataclass(frozen=True)
class CalibrationConfig:
    """F1: calibração de nulo por série (state/calibration.py). `shrink_pseudo` é a pseudo-contagem
    de encolhimento do desvio empírico para o teórico i.i.d. — necessária porque uma janela w sobre
    um histórico n_h só tem ~n_h/w janelas independentes."""
    enabled: bool
    shrink_pseudo: float


@dataclass(frozen=True)
class FeaturesConfig:
    warmup_min_n: int


@dataclass(frozen=True)
class LightGBMConfig:
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_data_in_leaf: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    n_estimators_cap: int
    early_stopping_rounds: int
    max_bin: int
    deterministic: bool
    force_row_wise: bool
    train_num_threads: int
    predict_num_threads: int
    n_folds: int
    feval_max_valid_rows: int | None = None  # R2 (parecer §6-R2): subamostra determinística do fold
    # de validação usada pelo feval de AUC-por-passo a cada rodada de boosting; None = fold inteiro.
    early_stopping_metric: str = "logloss"  # "logloss" ou "ts_auc_by_t" -- qual das duas métricas do
    # feval (model/train.py:_make_fold_feval) governa a parada via first_metric_only. MEDIDO
    # (retreino real, 2026-07-20): "ts_auc_by_t" sozinho treina 100-236 rodadas (vs. 61-89 com
    # logloss) perseguindo o argmax de uma métrica rank-based cujo ruído entre rodadas é dominado
    # pelo n efetivo de ~10^4 séries (não pelo número de linhas) -- isso produziu uma regressão real
    # e estatisticamente significativa na TS-AUC OOF completa (Delta -0.0099, IC exclui 0) mesmo
    # usando o fold de validação inteiro no feval (sem subamostra). "logloss" (default) reproduz o
    # comportamento original, validado; "ts_auc_by_t" fica disponível para experimentação futura com
    # estabilização adicional (ex.: min_delta, suavização), não para uso direto.


@dataclass(frozen=True)
class ThinningConfig:
    full_until: int
    step_101_400: int
    step_401_plus: int


@dataclass(frozen=True)
class ModelConfig:
    mode: str
    dataset_n_jobs: int  # paralelismo entre séries em model/dataset.py; -1 = todos os núcleos (joblib)


@dataclass(frozen=True)
class FallbackConfig:
    w_lo: float
    w_cusum: float
    w_conformal: float
    bias: float


@dataclass(frozen=True)
class GatesConfig:
    drift_slope_abs_max: float
    latency_budget_us_per_step: float
    scenarios: dict  # keyed by scenario id (t1, t2, ..., t12b, t13) -> dict of thresholds


@dataclass(frozen=True)
class SubmissionConfig:
    log_path: str


@dataclass(frozen=True)
class PostprocessConfig:
    mode: str
    soft_decay: float
    ema_alpha: float


@dataclass(frozen=True)
class Config:
    seed: int
    h0: H0Config
    state: StateConfig
    cusum: CusumConfig
    bayes: BayesConfig
    conformal: ConformalConfig
    rank_twosample: RankTwoSampleConfig
    dependence: DependenceConfig
    lmoments: LMomentConfig
    varloc: VarLocConfig
    jumps: JumpConfig
    bocpd: BOCPDConfig
    mmd: MMDConfig
    multiscale: MultiScaleConfig
    h0_fingerprint: H0FingerprintConfig
    calibration: CalibrationConfig
    features: FeaturesConfig
    lightgbm: LightGBMConfig
    rank: RankObjectiveConfig
    thinning: ThinningConfig
    model: ModelConfig
    fallback: FallbackConfig
    gates: GatesConfig
    submission: SubmissionConfig
    postprocess: PostprocessConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    raw = yaml.safe_load(_EMBEDDED_YAML)

    gates_raw = dict(raw["gates"])
    drift_slope_abs_max = gates_raw.pop("drift_slope_abs_max")
    latency_budget_us_per_step = gates_raw.pop("latency_budget_us_per_step")

    return Config(
        seed=raw["seed"],
        h0=H0Config(**raw["h0"]),
        state=StateConfig(**raw["state"]),
        cusum=CusumConfig(**raw["cusum"]),
        bayes=BayesConfig(**raw["bayes"]),
        conformal=ConformalConfig(**raw["conformal"]),
        rank_twosample=RankTwoSampleConfig(**raw["rank_twosample"]),
        dependence=DependenceConfig(**raw["dependence"]),
        lmoments=LMomentConfig(**raw["lmoments"]),
        varloc=VarLocConfig(**raw["varloc"]),
        jumps=JumpConfig(**raw["jumps"]),
        bocpd=BOCPDConfig(**raw["bocpd"]),
        mmd=MMDConfig(**raw["mmd"]),
        multiscale=MultiScaleConfig(**raw["multiscale"]),
        h0_fingerprint=H0FingerprintConfig(**raw["h0_fingerprint"]),
        calibration=CalibrationConfig(**raw["calibration"]),
        features=FeaturesConfig(**raw["features"]),
        lightgbm=LightGBMConfig(**raw["lightgbm"]),
        rank=RankObjectiveConfig(**raw["rank"]),
        thinning=ThinningConfig(**raw["thinning"]),
        model=ModelConfig(**raw["model"]),
        fallback=FallbackConfig(**raw["fallback"]),
        gates=GatesConfig(
            drift_slope_abs_max=drift_slope_abs_max,
            latency_budget_us_per_step=latency_budget_us_per_step,
            scenarios=gates_raw,
        ),
        submission=SubmissionConfig(**raw["submission"]),
        postprocess=PostprocessConfig(**raw["postprocess"]),
    )


# @crunch/keep:on
# ============================== sbrt/features/assembly.py ==============================
"""Ordem canônica das features + schema (plano §5). A ordem canônica é `sorted(feats.keys())` —
determinística e estável sem precisar de um "primeiro run" especial; persistida junto do modelo
para que `model/predict.py` monte o vetor na mesma ordem usada em `model/dataset.py` (motor único,
docs/PLANO_REPOSITORIO.md §1)."""

import json
from pathlib import Path

import numpy as np


def build_feature_order(feats: dict) -> tuple:
    return tuple(sorted(feats.keys()))


def to_array(feats: dict, order: tuple) -> np.ndarray:
    """Materializa na ordem canônica; chave ausente (warm-up) -> NaN. LightGBM trata NaN
    nativamente — nunca usar sentinela numérica."""
    return np.array([feats.get(k, np.nan) for k in order], dtype=np.float64)


def save_schema(order: tuple, path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(order), indent=2), encoding="utf-8")


def load_schema(path: str | Path) -> tuple:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")))


# @crunch/keep:on
# ============================== sbrt/state/accumulators.py ==============================
"""AccumulatorBlock — Welford global + EWMA (média/variância/sinal/excedência) + janelas rodantes
(plano §4.2, tabela §5 linhas #1,2,3,6,7,9,10,12,14,16,17,18,19).

Roteamento de fluxo (plano §3.4): média/dependência/forma usam `e_vol` (vol-ajustado); variância/cauda
usam `e` (escala congelada do histórico) — nunca o inverso, sob pena de o EWMA-vol absorver a própria
quebra de variância (contraexemplo CE2, plano §12.5).

Features #26 (hedge, precisa de x cru) e #27/#28 (meta) não cabem no contrato `StateBlock`
(que só recebe e/e_raw/e_vol/t) — são calculadas por `state/scorer.py` diretamente.
"""

import math
from typing import TYPE_CHECKING

import numpy as np


_NAN = math.nan


def _evicted_delta(evicted: float | None) -> float:
    return evicted if evicted is not None else 0.0


def _push_and_update_sum(entry: list, value: float) -> None:
    ring: RingBuffer = entry[0]
    evicted = ring.push(value)
    entry[1] = entry[1] + value - _evicted_delta(evicted)


class AccumulatorBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.cfg = cfg
        self.t = 0
        self.warmup_min_n = cfg.features.warmup_min_n

        self.ewma_mean = {lam: 0.0 for lam in cfg.state.ewma_lambdas}          # #2, e_vol
        self.ewma_var = {lam: 1.0 for lam in cfg.state.ewma_lambdas}           # #6, e (v0=1)
        self.welford_mean_evol = (0, 0.0, 0.0)                                  # #1 / #16 denom, e_vol
        self.welford_var_e = (0, 0.0, 0.0)                                     # #9, e

        self.window_sum_evol = {w: [RingBuffer(w), 0.0] for w in cfg.state.window_sizes}   # #3
        self.window_sumsq_e = {w: [RingBuffer(w), 0.0] for w in cfg.state.window_sizes}    # #7
        self.exceed_windows = {w: [RingBuffer(w), 0.0] for w in cfg.state.exceedance_windows}  # #10a
        self.sign_windows = {w: [RingBuffer(w), 0.0] for w in cfg.state.sign_windows}       # #14

        self.ewma_exceed2 = 0.0                                                # #10b
        self._p0_exceed2 = 2.0 * (1.0 - _norm_cdf(2.0))

        self.count_exceed99 = 0                                                # #12
        self.max_abs_eraw = 0.0
        self._q99_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.99))

        self.S_u_global = 0.0                                                  # #16 numerator
        self.prev_evol: float | None = None

        dep_w = cfg.state.dependence_window                                    # #17
        self.dep_u_ring = RingBuffer(dep_w)
        self.dep_u_sum = 0.0
        self.dep_sq_ring = RingBuffer(dep_w)
        self.dep_sq_sum = 0.0

        qw = cfg.state.quantile_crossing_window                                # #18
        self.qc_mid = [RingBuffer(qw), 0.0]
        self.qc_low = [RingBuffer(qw), 0.0]
        self.q25 = h0.q["0.25"]
        self.q75 = h0.q["0.75"]
        self.q10 = h0.q["0.10"]

        sw = cfg.state.skew_window                                             # #19
        self.skew_ring = RingBuffer(sw)
        self.skew_sum3 = 0.0

        # "vol-of-vol": estabilidade do nível de variância numa janela, não o nível em si (todas as
        # #7/#9 já existentes medem nível). Um burst GARCH tem variância-da-variância ALTA (a
        # variância local sobe e desce dentro do burst); uma quebra de variância genuína assenta num
        # novo patamar ESTÁVEL. Feature adicionada após achado de que o modelo rankeava alarmes falsos
        # tipo GARCH acima de quebras sutis reais (T6/T9 vs T3, ver histórico do projeto). Reusa
        # window_sumsq_e[100] (E[e²]) já existente; só precisa de E[e⁴] adicional na mesma janela.
        self.volvol_100 = [RingBuffer(100), 0.0]

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t

        n, mean, m2 = welford_update(*self.welford_mean_evol, e_vol)
        self.welford_mean_evol = (n, mean, m2)

        for lam in self.ewma_mean:
            self.ewma_mean[lam] = ewma_update(self.ewma_mean[lam], e_vol, lam)
        for lam in self.ewma_var:
            self.ewma_var[lam] = ewma_update(self.ewma_var[lam], e * e, lam)

        for w in self.window_sum_evol:
            _push_and_update_sum(self.window_sum_evol[w], e_vol)
        for w in self.window_sumsq_e:
            _push_and_update_sum(self.window_sumsq_e[w], e * e)

        n2, mean2, m2b = welford_update(*self.welford_var_e, e)
        self.welford_var_e = (n2, mean2, m2b)

        ind2 = 1.0 if abs(e) > 2.0 else 0.0
        for w in self.exceed_windows:
            _push_and_update_sum(self.exceed_windows[w], ind2)
        lam_mid = self.cfg.state.ewma_lambdas[1]
        self.ewma_exceed2 = ewma_update(self.ewma_exceed2, ind2, lam_mid)

        if abs(e_raw) > self._q99_abs:
            self.count_exceed99 += 1
        self.max_abs_eraw = max(self.max_abs_eraw, abs(e_raw))

        indpos = 1.0 if e_vol > 0 else 0.0
        for w in self.sign_windows:
            _push_and_update_sum(self.sign_windows[w], indpos)

        if self.prev_evol is not None:
            u_t = e_vol * self.prev_evol
            self.S_u_global += u_t
            evicted_u = self.dep_u_ring.push(u_t)
            self.dep_u_sum += u_t - _evicted_delta(evicted_u)
        evicted_sq = self.dep_sq_ring.push(e_vol * e_vol)
        self.dep_sq_sum += e_vol * e_vol - _evicted_delta(evicted_sq)
        self.prev_evol = e_vol

        in_mid = 1.0 if (self.q25 < e_vol < self.q75) else 0.0
        below_low = 1.0 if e_vol < self.q10 else 0.0
        _push_and_update_sum(self.qc_mid, in_mid)
        _push_and_update_sum(self.qc_low, below_low)

        cube = e_vol ** 3
        evicted_c = self.skew_ring.push(cube)
        self.skew_sum3 += cube - _evicted_delta(evicted_c)

        _push_and_update_sum(self.volvol_100, (e * e) ** 2)

    def features(self) -> dict[str, float]:
        t = self.t
        wmin = self.warmup_min_n
        out: dict[str, float] = {}

        n, mean, m2 = self.welford_mean_evol
        out["accum_welford_mean_z"] = math.sqrt(n) * mean if n >= wmin else _NAN

        for lam in self.ewma_mean:
            var_l = lam / (2.0 - lam)
            key = f"accum_ewma_mean_z_l{round(lam * 1000):03d}"
            out[key] = (self.ewma_mean[lam] / math.sqrt(var_l)) if t >= wmin else _NAN

        for w, (_, s) in self.window_sum_evol.items():
            n_eff = min(t, w)
            key = f"accum_window_mean_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key] = (s / n_eff) * math.sqrt(n_eff)
            else:
                out[key] = _NAN

        for lam in self.ewma_var:
            key = f"accum_ewma_var_ln_l{round(lam * 1000):03d}"
            out[key] = math.log(max(self.ewma_var[lam], 1e-12)) if t >= wmin else _NAN

        for w, (_, s) in self.window_sumsq_e.items():
            n_eff = min(t, w)
            key = f"accum_window_var_ln_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key] = math.log(max(s / n_eff, 1e-12))
            else:
                out[key] = _NAN

        n2, mean2, m2b = self.welford_var_e
        out["accum_welford_var_ln"] = math.log(max(m2b / n2, 1e-12)) if n2 >= wmin else _NAN

        for w, (_, s) in self.exceed_windows.items():
            n_eff = min(t, w)
            key = f"accum_window_exceed2_frac_w{w:03d}"
            out[key] = (s / n_eff) if (t >= wmin and n_eff > 0) else _NAN

        p0 = self._p0_exceed2
        lam_mid = self.cfg.state.ewma_lambdas[1]
        var_ewma = p0 * (1.0 - p0) * lam_mid / (2.0 - lam_mid)
        out["accum_ewma_exceed2_z"] = (
            (self.ewma_exceed2 - p0) / math.sqrt(max(var_ewma, 1e-12)) if t >= wmin else _NAN
        )

        out["accum_global_exceed99_frac"] = (self.count_exceed99 / t) if t >= wmin else _NAN
        out["accum_global_max_abs_eraw"] = self.max_abs_eraw if t >= 1 else _NAN

        for w, (_, s) in self.sign_windows.items():
            n_eff = min(t, w)
            key = f"accum_window_sign_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                p = s / n_eff
                out[key] = (p - 0.5) / math.sqrt(0.25 / n_eff)
            else:
                out[key] = _NAN

        if t >= max(wmin, 3) and m2 > 1e-9:
            r = max(min(self.S_u_global / m2, 0.999), -0.999)
            out["accum_global_rho1_fz"] = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(max(t - 3, 1))
        else:
            out["accum_global_rho1_fz"] = _NAN

        dep_w = self.cfg.state.dependence_window
        n_eff_dep = min(max(t - 1, 0), dep_w)
        if t >= max(wmin, 3) and self.dep_sq_sum > 1e-9 and n_eff_dep > 3:
            r = max(min(self.dep_u_sum / self.dep_sq_sum, 0.999), -0.999)
            out["accum_window_rho1_fz_w100"] = 0.5 * math.log((1.0 + r) / (1.0 - r)) * math.sqrt(n_eff_dep - 3)
        else:
            out["accum_window_rho1_fz_w100"] = _NAN

        qw = self.cfg.state.quantile_crossing_window
        n_eff_qc = min(t, qw)
        if t >= wmin and n_eff_qc > 0:
            out["accum_window_qcross_mid_frac_w100"] = self.qc_mid[1] / n_eff_qc
            out["accum_window_qcross_low_frac_w100"] = self.qc_low[1] / n_eff_qc
        else:
            out["accum_window_qcross_mid_frac_w100"] = _NAN
            out["accum_window_qcross_low_frac_w100"] = _NAN

        sw = self.cfg.state.skew_window
        n_eff_s = min(t, sw)
        if t >= wmin and n_eff_s > 0:
            skew_hat = self.skew_sum3 / n_eff_s
            out["accum_window_skew_z_w250"] = skew_hat * math.sqrt(n_eff_s / 6.0)
        else:
            out["accum_window_skew_z_w250"] = _NAN

        n_eff_vv = min(t, 100)
        if t >= wmin and n_eff_vv > 1 and 100 in self.window_sumsq_e:
            mean_e2 = self.window_sumsq_e[100][1] / n_eff_vv
            mean_e4 = self.volvol_100[1] / n_eff_vv
            var_e2 = max(mean_e4 - mean_e2 * mean_e2, 0.0)
            out["accum_window_volvol_cv_w100"] = math.sqrt(var_e2) / (mean_e2 + 1e-6)
        else:
            out["accum_window_volvol_cv_w100"] = _NAN

        return out


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# @crunch/keep:on
# ============================== sbrt/state/cusum.py ==============================
"""CusumBlock — banco de 15 CUSUMs + idades (plano §4.2, tabela §5 linhas #4,#8,#11,#13,#15).

Recursões max O(1), minimax-ótimas para alternativas simples (Page 1954; Moustakides 1986).
Fluxo: média/sinal usam `e_vol` (vol-ajustado); variância usa `e` (frozen, trava anti-absorção
§3.4/CE2); excedência usa `e_raw` contra os quantis do H0; dependência usa `e_vol` normalizado por
sigma_u do histórico. As features saem CRUAS (sem logístico) — a calibração é tarefa do LightGBM
(§5); o mapeamento logístico só existe no fallback puro-estatístico (§8.5).
"""

from typing import TYPE_CHECKING

import numpy as np

def _fmt(x: float) -> str:
    """0.25 -> '025', 1.5 -> '150' — convenção de sufixo de nome de feature (delta/ratio * 100)."""
    return f"{round(x * 100):03d}"


class CusumBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.cfg = cfg
        self.deltas = list(cfg.cusum.mean_deltas)
        self.ratios_up = list(cfg.cusum.var_ratios_up)
        self.ratio_down = cfg.cusum.var_ratio_down
        self.sigma_u = h0.sigma_u
        self.dep_delta = cfg.state.dependence_delta_u

        self.mean_pos = {d: 0.0 for d in self.deltas}
        self.mean_neg = {d: 0.0 for d in self.deltas}
        self.var_up = {r: 0.0 for r in self.ratios_up}
        self.var_down = 0.0

        self.q95_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.95))
        self.q99_abs = float(np.quantile(h0.sorted_abs_e_hist, 0.99))
        eb = cfg.state.exceedance_bernoulli
        self._eb_q95 = eb["q95"]
        self._eb_q99 = eb["q99"]
        self.exceed_q95 = 0.0
        self.exceed_q99 = 0.0

        sb = cfg.state.sign_bernoulli
        self._sb_p0 = sb["p0"]
        self._sb_p1_pos = sb["p1_pos"]
        self._sb_p1_neg = sb["p1_neg"]
        self.sign_pos = 0.0
        self.sign_neg = 0.0

        self.dep_pos = 0.0
        self.dep_neg = 0.0
        self.prev_evol: float | None = None

        self.ages = {
            "mean_pos": {d: 0 for d in self.deltas},
            "mean_neg": {d: 0 for d in self.deltas},
            "var_up": {r: 0 for r in self.ratios_up},
            "var_down": 0,
            "exceed_q95": 0,
            "exceed_q99": 0,
            "sign_pos": 0,
            "sign_neg": 0,
            "dep_pos": 0,
            "dep_neg": 0,
        }

    @staticmethod
    def _bump_age(current_age: int, new_value: float) -> int:
        return 0 if new_value <= 0.0 else current_age + 1

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        for d in self.deltas:
            self.mean_pos[d] = max(0.0, self.mean_pos[d] + d * e_vol - d * d / 2.0)
            self.ages["mean_pos"][d] = self._bump_age(self.ages["mean_pos"][d], self.mean_pos[d])
            self.mean_neg[d] = max(0.0, self.mean_neg[d] - d * e_vol - d * d / 2.0)
            self.ages["mean_neg"][d] = self._bump_age(self.ages["mean_neg"][d], self.mean_neg[d])

        e2 = e * e
        for r in self.ratios_up:
            inc = 0.5 * ((1.0 - 1.0 / r) * e2 - np.log(r))
            self.var_up[r] = max(0.0, self.var_up[r] + inc)
            self.ages["var_up"][r] = self._bump_age(self.ages["var_up"][r], self.var_up[r])

        r_down = self.ratio_down
        inc_down = 0.5 * ((1.0 - 1.0 / r_down) * e2 - np.log(r_down))
        self.var_down = max(0.0, self.var_down + inc_down)
        self.ages["var_down"] = self._bump_age(self.ages["var_down"], self.var_down)

        b95 = 1.0 if abs(e_raw) > self.q95_abs else 0.0
        p0, p1 = self._eb_q95["p0"], self._eb_q95["p1"]
        inc95 = b95 * np.log(p1 / p0) + (1.0 - b95) * np.log((1.0 - p1) / (1.0 - p0))
        self.exceed_q95 = max(0.0, self.exceed_q95 + inc95)
        self.ages["exceed_q95"] = self._bump_age(self.ages["exceed_q95"], self.exceed_q95)

        b99 = 1.0 if abs(e_raw) > self.q99_abs else 0.0
        p0, p1 = self._eb_q99["p0"], self._eb_q99["p1"]
        inc99 = b99 * np.log(p1 / p0) + (1.0 - b99) * np.log((1.0 - p1) / (1.0 - p0))
        self.exceed_q99 = max(0.0, self.exceed_q99 + inc99)
        self.ages["exceed_q99"] = self._bump_age(self.ages["exceed_q99"], self.exceed_q99)

        b_sign = 1.0 if e_vol > 0 else 0.0
        p0 = self._sb_p0
        p1 = self._sb_p1_pos
        inc_sign_pos = b_sign * np.log(p1 / p0) + (1.0 - b_sign) * np.log((1.0 - p1) / (1.0 - p0))
        self.sign_pos = max(0.0, self.sign_pos + inc_sign_pos)
        self.ages["sign_pos"] = self._bump_age(self.ages["sign_pos"], self.sign_pos)

        p1 = self._sb_p1_neg
        inc_sign_neg = b_sign * np.log(p1 / p0) + (1.0 - b_sign) * np.log((1.0 - p1) / (1.0 - p0))
        self.sign_neg = max(0.0, self.sign_neg + inc_sign_neg)
        self.ages["sign_neg"] = self._bump_age(self.ages["sign_neg"], self.sign_neg)

        if self.prev_evol is not None:
            u_norm = (e_vol * self.prev_evol) / self.sigma_u
            du = self.dep_delta
            self.dep_pos = max(0.0, self.dep_pos + du * u_norm - du * du / 2.0)
            self.ages["dep_pos"] = self._bump_age(self.ages["dep_pos"], self.dep_pos)
            self.dep_neg = max(0.0, self.dep_neg - du * u_norm - du * du / 2.0)
            self.ages["dep_neg"] = self._bump_age(self.ages["dep_neg"], self.dep_neg)
        self.prev_evol = e_vol

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for d in self.deltas:
            out[f"cusum_mean_pos_d{_fmt(d)}"] = self.mean_pos[d]
            out[f"cusum_mean_neg_d{_fmt(d)}"] = self.mean_neg[d]
        for r in self.ratios_up:
            out[f"cusum_var_up_r{_fmt(r)}"] = self.var_up[r]
        out[f"cusum_var_down_r{_fmt(self.ratio_down)}"] = self.var_down
        out["cusum_exceed_q95"] = self.exceed_q95
        out["cusum_exceed_q99"] = self.exceed_q99
        out["cusum_sign_pos"] = self.sign_pos
        out["cusum_sign_neg"] = self.sign_neg
        out["cusum_dep_pos"] = self.dep_pos
        out["cusum_dep_neg"] = self.dep_neg

        # idades: 6 selecionadas (plano §5 #24) — localizadores baratos de tau, usadas também
        # pela concordância de localizadores (#25, calculada em state/scorer.py)
        out["cusum_age_mean_pos_d050"] = float(self.ages["mean_pos"].get(0.5, float("nan")))
        out["cusum_age_mean_neg_d050"] = float(self.ages["mean_neg"].get(0.5, float("nan")))
        out["cusum_age_var_up_r150"] = float(self.ages["var_up"].get(1.5, float("nan")))
        out["cusum_age_sign_pos"] = float(self.ages["sign_pos"])
        out["cusum_age_sign_neg"] = float(self.ages["sign_neg"])
        out["cusum_age_exceed_q95"] = float(self.ages["exceed_q95"])
        return out


# @crunch/keep:on
# ============================== sbrt/state/bayes_filter.py ==============================
"""BayesFilterBlock — filtro bayesiano de troca única, log-espaço (plano §4.3).

Modelo: sob H0, e_t ~ N(0,1) (e = fluxo congelado — o filtro cobre média E variância, "todas
(média+var)" na tabela §5 #20, logo nunca usa o fluxo vol-ajustado, §3.4). Pós-mudança: e_t ~
N(mu,sigma^2), prior conjugado Normal-Inv-chi^2. Hazard constante h, sem morte de regime (a quebra é
permanente). Três filtros independentes (hazards 1/50, 1/100 e 1/400, configs/default.yaml
bayes.hazards) rodam em paralelo dentro do mesmo bloco — 1/50 acrescentado para reação mais rápida
em t baixo (docs/DIAGNOSTICO_TS_AUC.md, direção 4; plano tabela §5 original previa só 1/100 e 1/400).
"""

import math
from typing import TYPE_CHECKING


def _log_student_t(x: float, nu: float, loc: float, scale2: float) -> float:
    scale2 = max(scale2, 1e-12)
    z2 = (x - loc) ** 2 / (nu * scale2)
    return (
        lgamma_cached((nu + 1.0) / 2.0)
        - lgamma_cached(nu / 2.0)
        - 0.5 * math.log(nu * math.pi * scale2)
        - ((nu + 1.0) / 2.0) * math.log1p(z2)
    )


class _SingleHazardFilter:
    __slots__ = (
        "log_h", "log_1mh", "mu0", "kappa0", "nu0", "sigma0_sq",
        "max_candidates", "protect_recent", "renorm_threshold", "logw0", "candidates",
    )

    def __init__(self, h: float, prior: dict, max_candidates: int, protect_recent: int, renorm_threshold: float):
        self.log_h = math.log(h)
        self.log_1mh = math.log(1.0 - h)
        self.mu0 = prior["mu0"]
        self.kappa0 = prior["kappa0"]
        self.nu0 = prior["nu0"]
        self.sigma0_sq = prior["sigma0_sq"]
        self.max_candidates = max_candidates
        self.protect_recent = protect_recent
        self.renorm_threshold = renorm_threshold
        self.logw0 = 0.0
        self.candidates: list[dict] = []

    def _log_pred(self, n: float, mean: float, m2: float, x: float) -> float:
        kappa0, mu0, nu0, sigma0_sq = self.kappa0, self.mu0, self.nu0, self.sigma0_sq
        kappa_n = kappa0 + n
        mu_n = (kappa0 * mu0 + n * mean) / kappa_n
        nu_n = nu0 + n
        ssq_n = (nu0 * sigma0_sq + m2 + kappa0 * n * (mean - mu0) ** 2 / kappa_n) / nu_n
        scale2 = ssq_n * (kappa_n + 1.0) / kappa_n
        return _log_student_t(x, nu_n, mu_n, scale2)

    def update(self, e: float, t: int) -> None:
        logpred_new = self._log_pred(0.0, 0.0, 0.0, e)
        logw_new = self.logw0 + self.log_h + logpred_new

        ell0 = -0.5 * math.log(2.0 * math.pi) - 0.5 * e * e
        self.logw0 = self.logw0 + self.log_1mh + ell0

        for c in self.candidates:
            c["logw"] += self._log_pred(c["n"], c["mean"], c["m2"], e)
            n, mean, m2 = welford_update(c["n"], c["mean"], c["m2"], e)
            c["n"], c["mean"], c["m2"] = n, mean, m2

        n, mean, m2 = welford_update(0, 0.0, 0.0, e)
        self.candidates.append({"n": n, "mean": mean, "m2": m2, "logw": logw_new, "birth_t": t})

        if len(self.candidates) > self.max_candidates:
            self.candidates.sort(key=lambda c: c["birth_t"])
            protected = self.candidates[-self.protect_recent:]
            rest = self.candidates[: -self.protect_recent]
            rest.sort(key=lambda c: c["logw"], reverse=True)
            keep = self.max_candidates - len(protected)
            self.candidates = rest[:keep] + protected

        max_logw = max((c["logw"] for c in self.candidates), default=-math.inf)
        overall_max = max(max_logw, self.logw0)
        if abs(overall_max) > self.renorm_threshold:
            self.logw0 -= overall_max
            for c in self.candidates:
                c["logw"] -= overall_max

    def outputs(self, t: int) -> tuple[float, int, float, float]:
        logws = [c["logw"] for c in self.candidates]
        lo = logsumexp(logws) - self.logw0
        map_c = max(self.candidates, key=lambda c: c["logw"])
        age_map = t - map_c["birth_t"]
        n_map, mean_map, m2_map = map_c["n"], map_c["mean"], map_c["m2"]
        map_z_mean = math.sqrt(n_map) * mean_map if n_map > 0 else 0.0
        map_var_ln = math.log(m2_map / n_map + 1e-9) if n_map > 0 else math.log(1e-9)
        return lo, age_map, map_z_mean, map_var_ln


class BayesFilterBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        prior = cfg.bayes.prior
        self.filters = {
            h: _SingleHazardFilter(
                h, prior, cfg.bayes.max_candidates, cfg.bayes.protect_recent, cfg.bayes.logw_renorm_threshold
            )
            for h in cfg.bayes.hazards
        }
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        for f in self.filters.values():
            f.update(e, t)

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for h, f in self.filters.items():
            tag = f"h{round(h * 10000):04d}"
            lo, age_map, map_z_mean, map_var_ln = f.outputs(self.t)
            out[f"bayes_lo_{tag}"] = lo
            out[f"bayes_age_map_{tag}"] = float(age_map)
            out[f"bayes_age_ln1p_{tag}"] = math.log1p(max(age_map, 0))
            out[f"bayes_map_z_mean_{tag}"] = map_z_mean
            out[f"bayes_map_var_ln_{tag}"] = map_var_ln
        return out


# @crunch/keep:on
# ============================== sbrt/state/conformal.py ==============================
"""ConformalBlock — martingales conformais sobre p-values causais das inovações contra a
distribuição do histórico (plano §4.2 #23; Vovk et al. 2005; Volkhonskiy et al. 2017).

Evidência livre de distribuição, O(log n_h)/passo via busca binária nos arrays ordenados do
histórico (`H0Params.sorted_e_hist` / `sorted_abs_e_hist`). Opera sempre sobre `e` (escala congelada)
porque os arrays ordenados de referência foram construídos a partir do resíduo/sigma_e do histórico —
comparar `e_vol` contra eles seria inconsistente de escala quando o ajuste de volatilidade está ativo.

Três variantes de p-value (todas via mid-rank, para lidar com empates):
- abs: cauda superior de |e_t| contra |e| do histórico — sensível a variância/cauda.
- right: cauda superior de e_t (com sinal) — sensível a shift positivo/skew à direita.
- sign: cauda inferior de e_t (com sinal) — sensível a shift negativo/skew à esquerda.

Cada uma vira um log-martingale (mistura sobre epsilons, "apostas" de Vovk); "6->4 usadas" (plano
tabela §5 #23): abs tem variante acumulada E com reset (SR-like), right/sign só acumuladas.
"""

import bisect
import math
from typing import TYPE_CHECKING


def _mid_rank(sorted_arr, x: float) -> float:
    lo = bisect.bisect_left(sorted_arr, x)
    hi = bisect.bisect_right(sorted_arr, x)
    return (lo + hi) / 2.0


def _upper_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (n - rank + 0.5) / (n + 1.0)


def _lower_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (rank + 0.5) / (n + 1.0)


class ConformalBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.sorted_e_hist = h0.sorted_e_hist
        self.sorted_abs_e_hist = h0.sorted_abs_e_hist
        self.n_h = h0.n_h
        self.epsilons = list(cfg.conformal.epsilons)
        self._log_k = math.log(len(self.epsilons))

        self.L_abs = {eps: 0.0 for eps in self.epsilons}
        self.L_abs_reset = {eps: 0.0 for eps in self.epsilons}
        self.L_right = {eps: 0.0 for eps in self.epsilons}
        self.L_sign = {eps: 0.0 for eps in self.epsilons}

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        p_abs = _upper_tail_p(self.sorted_abs_e_hist, abs(e), self.n_h)
        p_right = _upper_tail_p(self.sorted_e_hist, e, self.n_h)
        p_sign = _lower_tail_p(self.sorted_e_hist, e, self.n_h)

        log_p_abs = math.log(p_abs)
        log_p_right = math.log(p_right)
        log_p_sign = math.log(p_sign)

        for eps in self.epsilons:
            log_eps = math.log(eps)
            inc_abs = log_eps + (eps - 1.0) * log_p_abs
            self.L_abs[eps] += inc_abs
            self.L_abs_reset[eps] = max(0.0, self.L_abs_reset[eps] + inc_abs)
            self.L_right[eps] += log_eps + (eps - 1.0) * log_p_right
            self.L_sign[eps] += log_eps + (eps - 1.0) * log_p_sign

    def features(self) -> dict[str, float]:
        return {
            "conformal_logm_abs": logsumexp(list(self.L_abs.values())) - self._log_k,
            "conformal_logm_abs_reset": logsumexp(list(self.L_abs_reset.values())) - self._log_k,
            "conformal_logm_right": logsumexp(list(self.L_right.values())) - self._log_k,
            "conformal_logm_sign": logsumexp(list(self.L_sign.values())) - self._log_k,
        }


# @crunch/keep:on
# ============================== sbrt/state/rank_twosample.py ==============================
"""RankTwoSampleBlock — duas amostras rank-based janela-vs-histórico (parecer de auditoria
docs/PARECER_AUDITORIA_ONYX.md §3.6/§6-R4: "o análogo causal do que venceu 2025").

O banco atual é forte em detectores sequenciais paramétricos (CUSUM, filtro bayesiano) e fraco no
paradigma que dominou a edição batch: comparação distribucional direta janela-vs-histórico. Os
p-values conformais de `ConformalBlock` (rank de e_t contra o histórico ordenado, livres de
distribuição e comparáveis entre séries por construção) são reaproveitados aqui como a "moeda" de
um agregador de POTÊNCIA (médias de janela) em vez do agregador de martingale de Vovk (otimizado
para controle de erro tipo Ville sob H0, não para ranking) — parecer §3.6.

Três famílias, cada uma O(log n_h) por passo via busca binária nos arrays ordenados do histórico
(`H0Params.sorted_e_hist` / `sorted_abs_e_hist`), sempre sobre `e` (escala congelada, mesma
convenção de `ConformalBlock` — os arrays de referência foram construídos a partir do resíduo do
histórico, comparar `e_vol` seria inconsistente de escala):

- localização: z de Wilcoxon de janela = média_w(p_right − ½)·√(12·n_eff) — estatística de
  Mann-Whitney da janela contra o histórico, robusta a caudas pesadas (parecer §3.6). CONVENÇÃO DE
  SINAL herdada de `p_right`/`p_abs` (cauda superior, mesma de ConformalBlock): p encolhe para perto
  de 0 quando o valor observado é extremo à direita, então um shift POSITIVO faz este z ficar mais
  NEGATIVO (não positivo) — sinal válido e monótono para uma árvore, só invertido do que a
  nomenclatura "Wilcoxon" sugeriria ingenuamente.
- dispersão/cauda: o mesmo, sobre p_abs em vez de p_right (tipo Ansari-Bradley) — sensível a
  quebras de variância/forma sem depender de momentos; mesma convenção de sinal (mais negativo sob
  variância maior, já que p_abs também é cauda superior).
- forma: chi²-de-janela sobre 4 bins de quantis do histórico (quartis) — frações observadas vs.
  nominais (25% cada), generaliza o quantile-crossing existente (accumulators.py #18).
"""

import bisect
import math
from typing import TYPE_CHECKING

import numpy as np


def _mid_rank(sorted_arr, x: float) -> float:
    lo = bisect.bisect_left(sorted_arr, x)
    hi = bisect.bisect_right(sorted_arr, x)
    return (lo + hi) / 2.0


def _upper_tail_p(sorted_arr, x: float, n: int) -> float:
    rank = _mid_rank(sorted_arr, x)
    return (n - rank + 0.5) / (n + 1.0)


class _WindowSum:
    """RingBuffer(w) + soma incremental — mesmo padrão de accumulators.py."""

    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, value: float) -> None:
        evicted = self.ring.push(value)
        self.total += value - (evicted if evicted is not None else 0.0)


class RankTwoSampleBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        self.warmup_min_n = cfg.features.warmup_min_n
        self.sorted_e_hist = h0.sorted_e_hist
        self.sorted_abs_e_hist = h0.sorted_abs_e_hist
        self.n_h = h0.n_h
        self.windows = list(cfg.rank_twosample.windows)
        self.t = 0

        self.loc_sum = {w: _WindowSum(w) for w in self.windows}
        self.disp_sum = {w: _WindowSum(w) for w in self.windows}

        # quartis do histórico (a partir de e_hist ordenado -- não depende de h0.q, que não inclui
        # a mediana); 4 bins: (-inf,q25], (q25,q50], (q50,q75], (q75,inf), nominal 25% cada.
        self._q25, self._q50, self._q75 = (float(v) for v in np.quantile(self.sorted_e_hist, [0.25, 0.5, 0.75]))
        self.bin_counts = {w: [0.0, 0.0, 0.0, 0.0] for w in self.windows}
        self.bin_ring = {w: RingBuffer(w) for w in self.windows}

    def _bin_of(self, e: float) -> int:
        if e <= self._q25:
            return 0
        if e <= self._q50:
            return 1
        if e <= self._q75:
            return 2
        return 3

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        p_right = _upper_tail_p(self.sorted_e_hist, e, self.n_h)
        p_abs = _upper_tail_p(self.sorted_abs_e_hist, abs(e), self.n_h)

        for w in self.windows:
            self.loc_sum[w].push(p_right - 0.5)
            self.disp_sum[w].push(p_abs - 0.5)

        b = self._bin_of(e)
        for w in self.windows:
            ring = self.bin_ring[w]
            counts = self.bin_counts[w]
            counts[b] += 1.0
            evicted = ring.push(float(b))
            if evicted is not None:
                counts[int(evicted)] -= 1.0

    def features(self) -> dict[str, float]:
        t = self.t
        wmin = self.warmup_min_n
        out: dict[str, float] = {}

        for w in self.windows:
            n_eff = min(t, w)
            key_loc = f"ranktwo_wilcoxon_z_w{w:03d}"
            key_disp = f"ranktwo_dispersion_z_w{w:03d}"
            if t >= wmin and n_eff > 0:
                out[key_loc] = (self.loc_sum[w].total / n_eff) * math.sqrt(12.0 * n_eff)
                out[key_disp] = (self.disp_sum[w].total / n_eff) * math.sqrt(12.0 * n_eff)
            else:
                out[key_loc] = math.nan
                out[key_disp] = math.nan

            key_shape = f"ranktwo_shape_chi2_w{w:03d}"
            if t >= wmin and n_eff > 0:
                expected = n_eff / 4.0
                counts = self.bin_counts[w]
                out[key_shape] = sum((c - expected) ** 2 / expected for c in counts) if expected > 0 else math.nan
            else:
                out[key_shape] = math.nan

        return out


# @crunch/keep:on
# ============================== sbrt/state/mmd.py ==============================
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

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter  # topo, não dentro da função: o conversor do Crunch avisa em
                                  # import aninhado (pode não virar requirement do submission)

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


def mmd_history_reference(e_hist: np.ndarray, cfg: "Config") -> tuple[np.ndarray, np.ndarray]:
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


def mmd_history_series(e_hist: np.ndarray, href: np.ndarray, href_joint: np.ndarray, cfg: "Config") -> dict:
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


# @crunch/keep:on
# ============================== sbrt/state/multiscale.py ==============================
"""MultiScaleBlock — decomposição causal de energia por escala (Haar diádico)
(docs/PROPOSTA_FEATURES_V2.md F4).

Motivação. O parecer de auditoria classificou o confundimento CE2×T6 como "indecidível no detector"
(§4.3): um patamar novo de variância e um cluster GARCH longo são indistinguíveis *numa janela*. Mas
eles não são indistinguíveis *entre escalas* — e é isso que nenhuma feature atual mede:

- um **patamar persistente** de variância eleva a energia em TODAS as escalas aproximadamente igual;
- um **burst GARCH** (oscilação rápida de volatilidade) concentra energia nas escalas FINAS;
- um **drift lento** concentra nas escalas GROSSAS.

Logo, o *formato* da curva energia-vs-escala discrimina o que o nível sozinho não discrimina. As
janelas existentes (`accum_window_var_ln_w010..w250`) são **suavizações do mesmo nível** em
comprimentos diferentes, não uma decomposição de escala: todas medem E[e²] numa janela, apenas com
mais ou menos suavização. A transformada de Haar separa a energia em bandas de frequência
*disjuntas* — informação genuinamente diferente.

Implementação. Cascata diádica causal e O(1) amortizado: em cada escala j mantém-se no máximo um
valor pendente; quando o par (a, b) fica completo emite-se o detalhe d = (a−b)/√2 (banda daquela
escala) e a aproximação s = (a+b)/√2, que sobe para a escala j+1. A energia por escala é uma EWMA de
d². Um coeficiente da escala j nasce a cada 2^(j+1) amostras — por isso as escalas grossas ficam em
NaN por muitos passos (tratado nativamente pelo LightGBM; é honesto: não há informação multi-escala
antes de haver amostras).

Sob H0 (ruído branco de variância unitária) a transformada de Haar preserva variância, então
E[d²] = 1 em toda escala e `haar_energy_ln_s*` ≈ 0 — as features já nascem aproximadamente
comparáveis entre séries, e a calibração de nulo por série (F1) corrige o resíduo de dependência.

Consome `e` (escala congelada do histórico), não `e_vol`: é família de variância — a trava
anti-absorção CE2 do plano §3.4 vale aqui igual às demais.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import lfilter  # topo, não dentro da função: o conversor do Crunch avisa em
                                  # import aninhado (pode não virar requirement do submission)

_SQRT2 = math.sqrt(2.0)


class MultiScaleBlock:
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        ms = cfg.multiscale
        self.J = ms.n_scales
        self.lam = ms.ewma_lambda
        self.min_coeffs = ms.warmup_min_coeffs

        self.pending: list[float | None] = [None] * self.J
        self.energy = [1.0] * self.J  # prior H0: Haar preserva variância -> E[d²]=1
        self.count = [0] * self.J

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        x = e
        for j in range(self.J):
            p = self.pending[j]
            if p is None:
                self.pending[j] = x
                return  # nada sobe para as escalas seguintes neste passo
            self.pending[j] = None
            d = (p - x) / _SQRT2
            s = (p + x) / _SQRT2
            self.energy[j] += self.lam * (d * d - self.energy[j])
            self.count[j] += 1
            x = s

    def features(self) -> dict[str, float]:
        out: dict[str, float] = {}
        ln_e: list[float] = []
        for j in range(self.J):
            if self.count[j] >= self.min_coeffs:
                v = math.log(max(self.energy[j], 1e-12))
            else:
                v = math.nan
            out[f"haar_energy_ln_s{j}"] = v
            ln_e.append(v)

        fine, mid, coarse = ln_e[0], ln_e[min(2, self.J - 1)], ln_e[self.J - 1]
        out["haar_contrast_fine_coarse"] = (
            fine - coarse if not (math.isnan(fine) or math.isnan(coarse)) else math.nan
        )
        out["haar_contrast_fine_mid"] = fine - mid if not (math.isnan(fine) or math.isnan(mid)) else math.nan
        return out


def multiscale_history_series(e_hist: np.ndarray, cfg: "Config") -> dict:
    """As mesmas estatísticas sobre o histórico, vetorizadas, para a calibração de nulo por série
    (F1). O pareamento é idêntico ao do laço online (pares consecutivos disjuntos, aproximação
    subindo de escala) — equivalência verificada em
    `tests/unit/test_multiscale.py::test_history_series_matches_online_block`."""
    ms = cfg.multiscale
    e = np.asarray(e_hist, dtype=np.float64)
    out: dict = {}
    ln_by_scale: list[np.ndarray] = []

    s = e
    for j in range(ms.n_scales):
        n_pairs = len(s) // 2
        if n_pairs < 1:
            break
        a = s[: 2 * n_pairs : 2]
        b = s[1 : 2 * n_pairs : 2]
        d = (a - b) / _SQRT2
        s = (a + b) / _SQRT2

        zi = [(1.0 - ms.ewma_lambda) * 1.0]  # energia inicial 1.0, igual ao online
        energy, _ = lfilter([ms.ewma_lambda], [1.0, -(1.0 - ms.ewma_lambda)], d * d, zi=zi)
        valid = energy[ms.warmup_min_coeffs - 1:] if ms.warmup_min_coeffs > 0 else energy
        ln_v = np.log(np.maximum(valid, 1e-12))
        out[f"haar_energy_ln_s{j}"] = ln_v
        ln_by_scale.append(ln_v)

    if len(ln_by_scale) >= 2:
        fine = ln_by_scale[0]
        coarse = ln_by_scale[-1]
        mid = ln_by_scale[min(2, len(ln_by_scale) - 1)]
        n_fc = min(len(fine), len(coarse))
        n_fm = min(len(fine), len(mid))
        # Escalas grossas têm menos coeficientes; para a distribuição nula basta parear os
        # primeiros n comuns (a média/desvio não dependem do alinhamento temporal exato).
        out["haar_contrast_fine_coarse"] = fine[:n_fc] - coarse[:n_fc]
        out["haar_contrast_fine_mid"] = fine[:n_fm] - mid[:n_fm]
    return out


# @crunch/keep:on
# ============================== sbrt/state/dependence.py ==============================
"""DependenceBlock — dependência serial não-linear e multi-lag (docs/INVESTIGACAO_FALHAS_V3.md P1).

## Por que existe

O cruzamento censo×OOF (INVESTIGACAO §1) mostrou que quebras *puras* de dependência (Δρ₁ alto, Δlogvar
baixo) têm detectabilidade **0,492 — abaixo do acaso** — apesar de o limite de Neyman-Pearson (§2.1)
mostrar que uma mudança de ρ₁ de magnitude moderada é altamente detectável (0,81–0,99 com janela
média/longa). É o maior ponto cego do modelo, e um eixo de sinal *independente* da variância (β=+0,04,
corr 0,14 com variância).

O banco só media dependência **linear lag-1**: `accum_*_rho1_fz` (Fisher-z de ρ₁ de e_vol), `cusum_dep`
(produto defasado), `mmd_joint` (conjunta lag-1). O SHAP transversal (INVESTIGACAO §4.1) mostrou os
lineares clássicos **mortos** (0,1–0,6%); só o MMD-joint vive (~10%), e mesmo ele não crava as quebras
de dependência. Este bloco cobre o que faltava:

- **Clustering de volatilidade** (ρ₁ de |e| e de e²): uma quebra pode mudar a *persistência* da
  volatilidade sem mudar seu nível médio. Nada online via isso (só o `meta_h0_acf_e2_l1` estático, da
  F2). Bônus: separa GARCH de quebra-de-nível de variância — um cluster GARCH tem ρ₁(e²) alto
  *persistente* (no histórico e no online), então a versão calibrada (contra o nulo da própria série)
  fica baixa; uma quebra de nível dá um ρ₁(e²) transitório *em excesso* sobre o nulo — ataca T6.
- **Massa multi-lag** (Σ_{k=1}^{L} ρ_k²): dependência em lags > 1 que o lag-1 sozinho perde.

Roteamento (plano §3.4): |e| e e² usam `e` (escala congelada — família de variância/cauda, trava CE2);
a massa linear usa `e_vol` (vol-ajustado — dependência de média/forma), consistente com o resto do banco.
Custo medido: ~1 µs/passo (produtos defasados incrementais, O(L)).
"""

import math
from collections import deque
from typing import TYPE_CHECKING


_NAN = math.nan


def _d(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RollingAutocorr:
    """Autocorrelação de janela, O(L) por passo, para lags 1..max_lag. Estimador enviesado padrão:
    ρ_k = (E_jan[v_t·v_{t-k}] − v̄²) / (E_jan[v²] − v̄²), tudo sobre a mesma janela deslizante."""

    __slots__ = ("W", "L", "val", "sv", "svv", "prod", "sp", "recent")

    def __init__(self, window: int, max_lag: int):
        self.W = window
        self.L = max_lag
        self.val = RingBuffer(window)
        self.sv = 0.0
        self.svv = 0.0
        self.prod = [RingBuffer(window) for _ in range(max_lag)]
        self.sp = [0.0] * max_lag
        self.recent: deque = deque(maxlen=max_lag)  # v_{t-1}, ..., v_{t-L}

    def update(self, v: float) -> None:
        for k in range(1, self.L + 1):
            if len(self.recent) >= k:
                p = v * self.recent[-k]  # recent[-1]=v_{t-1}, recent[-k]=v_{t-k}
                ev = self.prod[k - 1].push(p)
                self.sp[k - 1] += p - _d(ev)
        ev = self.val.push(v)
        self.sv += v - _d(ev)
        self.svv += v * v - _d(ev) ** 2
        self.recent.append(v)

    def _mean_var(self):
        n = len(self.val)
        if n < 2:
            return None
        mean = self.sv / n
        var = self.svv / n - mean * mean
        return mean, var, n

    def rho(self, k: int) -> float:
        mv = self._mean_var()
        if mv is None:
            return _NAN
        mean, var, _ = mv
        nk = len(self.prod[k - 1])
        if nk < 1 or var <= 1e-12:
            return 0.0
        autocov = self.sp[k - 1] / nk - mean * mean
        return autocov / var

    def mass(self) -> float:
        mv = self._mean_var()
        if mv is None:
            return _NAN
        s = 0.0
        for k in range(1, self.L + 1):
            r = self.rho(k)
            s += r * r
        return s


class DependenceBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        dc = cfg.dependence
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(dc.windows)
        self.mass_window = dc.mass_window
        self.abs_ac = {w: _RollingAutocorr(w, 1) for w in self.windows}
        self.sq_ac = {w: _RollingAutocorr(w, 1) for w in self.windows}
        self.mass_abs = _RollingAutocorr(dc.mass_window, dc.mass_max_lag)
        self.mass_evol = _RollingAutocorr(dc.mass_window, dc.mass_max_lag)
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        ae = abs(e)
        se = e * e
        for w in self.windows:
            self.abs_ac[w].update(ae)
            self.sq_ac[w].update(se)
        self.mass_abs.update(ae)
        self.mass_evol.update(e_vol)

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            out[f"dep_absrho1_w{w:03d}"] = self.abs_ac[w].rho(1) if warm else _NAN
            out[f"dep_sqrho1_w{w:03d}"] = self.sq_ac[w].rho(1) if warm else _NAN
        mw = self.mass_window
        out[f"dep_mass_abs_w{mw:03d}"] = self.mass_abs.mass() if warm else _NAN
        out[f"dep_mass_evol_w{mw:03d}"] = self.mass_evol.mass() if warm else _NAN
        return out


def dependence_history_null_series(e_hist, cfg) -> dict:
    """Roda o PRÓPRIO DependenceBlock sobre o histórico (H0 por definição) e devolve a série de cada
    feature e-based, para a calibração de nulo por série (F1, state/calibration.py). Rodar o bloco
    real — em vez de uma reimplementação vetorizada — garante por construção que o nulo é medido
    exatamente com a mesma estatística do online (elimina o risco de desalinhamento que exigiu testes
    dedicados no MMD/Haar). `e_vol` é aproximado por `e` no histórico (o ajuste de volatilidade
    converge a ~1 sobre o histórico estacionário); por isso `dep_mass_evol_*` NÃO é calibrado."""
    blk = DependenceBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name.startswith("dep_mass_evol"):
                continue
            acc.setdefault(name, []).append(val)
    return acc


# @crunch/keep:on
# ============================== sbrt/state/varloc.py ==============================
"""VarLocBlock — variância localizada no changepoint (docs/INVESTIGACAO_FALHAS_V3.md P3).

## Por que existe

A maior folga de EXTRAÇÃO do modelo (INVESTIGACAO §2–3): um detector ótimo de variância que conhece
tau atinge AUC≈0,856; o V3 fica em 0,604, e a folga é máxima em t alto. A hipótese diagnosticada (§3):
**toda janela fixa (accum_window_var_ln_w010..w250) dilui o sinal porque mistura pontos pré e
pós-quebra** — uma janela de 250 com tau no meio estima uma variância *atenuada*. O oracle usa só os
pontos pós-tau; o modelo não sabe onde é tau.

Este bloco não estima tau explicitamente — ele **seleciona a escala** que melhor revela a elevação de
variância, o que atinge o mesmo efeito de forma auto-contida:

- `varloc_max_z` = max sobre escalas d∈{10..250} do z padronizado de ln(E[e²]) na janela-d recente.
  Uma quebra de idade ~a é melhor vista com d≈a; ao maximizar sobre d, a feature *localiza a escala*
  automaticamente (uma quebra recente acende as escalas curtas; uma antiga, as longas), em vez de
  diluir num comprimento fixo. z_d = (ln(mean e²_d) + 1/n) / sqrt(2/n), n=min(t,d) — a padronização
  teórica i.i.d.-gaussiana; a inflação por curtose da série é corrigida pela calibração F1.
- `varloc_min_z` = min sobre d (elevação negativa: melhor escala para uma *queda* de variância).
- `varloc_argmax_lnscale` = ln(d*) da escala que maximiza z — um localizador barato (quão recente é
  a elevação mais forte).
- `varloc_recent_vs_lagged` = ln(E[e²] dos últimos R) − ln(E[e²] da janela [R, R+L) atrás): contraste
  direto "regime recente vs. regime anterior", sensível justamente quando a variância mudou de patamar.

Consome `e` (escala congelada) — família de variância, trava CE2 (plano §3.4). A versão calibrada
(F1) usa o nulo de max_z/min_z da própria série, o que remove a inflação por curtose (D-10) que
tornaria a padronização teórica incomparável entre séries.
"""

import math
from typing import TYPE_CHECKING


_NAN = math.nan


def _d0(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RollingSum:
    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, v: float) -> None:
        ev = self.ring.push(v)
        self.total += v - _d0(ev)

    def mean(self) -> float:
        n = len(self.ring)
        return self.total / n if n > 0 else _NAN

    def __len__(self) -> int:
        return len(self.ring)


class VarLocBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        vc = cfg.varloc
        self.warmup = cfg.features.warmup_min_n
        self.scales = list(vc.scales)
        self.recent = vc.recent
        self.lagged = vc.lagged
        self.t = 0
        needed = sorted(set(self.scales) | {self.recent, self.recent + self.lagged})
        self.sums = {w: _RollingSum(w) for w in needed}

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        se = e * e
        for s in self.sums.values():
            s.push(se)

    def _z(self, d: int) -> float:
        n = min(self.t, d)
        if n < 2:
            return _NAN
        mean_e2 = self.sums[d].mean()
        ln_v = math.log(max(mean_e2, 1e-12))
        return (ln_v + 1.0 / n) / math.sqrt(2.0 / n)

    def features(self) -> dict[str, float]:
        if self.t < self.warmup:
            return {
                "varloc_max_z": _NAN, "varloc_min_z": _NAN,
                "varloc_argmax_lnscale": _NAN, "varloc_recent_vs_lagged": _NAN,
            }
        zs = [(self._z(d), d) for d in self.scales]
        zs = [(z, d) for z, d in zs if not math.isnan(z)]
        if zs:
            zmax, dmax = max(zs, key=lambda p: p[0])
            zmin, _ = min(zs, key=lambda p: p[0])
            argmax_ln = math.log(dmax)
        else:
            zmax = zmin = argmax_ln = _NAN

        rl = _NAN
        s_full = self.sums[self.recent + self.lagged]
        s_rec = self.sums[self.recent]
        if len(s_full) >= self.recent + self.lagged:
            mean_rec = s_rec.mean()
            mean_lag = (s_full.total - s_rec.total) / self.lagged
            rl = math.log(max(mean_rec, 1e-12)) - math.log(max(mean_lag, 1e-12))

        return {
            "varloc_max_z": zmax,
            "varloc_min_z": zmin,
            "varloc_argmax_lnscale": argmax_ln,
            "varloc_recent_vs_lagged": rl,
        }


def varloc_history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio VarLocBlock sobre o histórico (H0), para a calibração F1. `argmax_lnscale` NÃO
    é calibrado (é um índice de escala, não uma magnitude)."""
    blk = VarLocBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name == "varloc_argmax_lnscale":
                continue
            acc.setdefault(name, []).append(val)
    return acc


# @crunch/keep:on
# ============================== sbrt/state/jumps.py ==============================
"""JumpBlock — decomposição salto/contínuo e assimetria de leverage
(docs/INVESTIGACAO_FALHAS_V3.md P4; Barndorff-Nielsen & Shephard 2004).

## Por que existe

Papel de PRECISÃO, não de recall (INVESTIGACAO §4.3): separa duas coisas que o banco confunde e que
são os falsos-positivos nomeados do projeto — cluster GARCH (T6) e outlier isolado (T9) — de uma
quebra genuína de variância.

**Bipower variation.** A variância realizada RV = Σe² capta variância contínua + saltos; a bipower
variation BV = (π/2)·Σ|eₜ||eₜ₋₁| é *robusta a saltos* (o produto de dois vizinhos é grande só se
AMBOS forem grandes, o que um salto isolado não garante). Logo:

- razão de salto (RV−BV)/RV isola a componente descontínua: **alta** num outlier/salto isolado (T9),
  **baixa** num cluster GARCH (volatilidade contínua) ou numa quebra de patamar de variância;
- BV dá um estimador de variância *robusto a saltos* — uma quebra de variância limpa move BV; um
  outlier isolado não. O contraste ln(RV) − ln(BV) é o discriminador.

**Semivariância / leverage.** RS⁺ = Σe²·1{e>0}, RS⁻ = Σe²·1{e<0}: a assimetria (RS⁺−RS⁻)/(RS⁺+RS⁻)
distingue uma quebra que empurra a cauda para um lado. E a correlação sinal-magnitude
corr(sign(eₜ₋₁), |eₜ|) capta o *efeito leverage* (volatilidade responde de forma assimétrica ao
sinal) — um eixo de dependência que cruza sinal×magnitude, ausente do banco (que tem CUSUMs de sinal
e de dependência separados, nunca cruzados).

Consome `e` (escala congelada) — família de variância/cauda, trava CE2 (plano §3.4). Calibrado por
F1: um GARCH tem razão de salto e leverage característicos no seu *próprio histórico*, então a versão
`_cal` só acende no *excesso* pós-quebra — é isso que o torna um discriminador de T6, não mais um
detector que dispara junto com o GARCH.
"""

import math
from typing import TYPE_CHECKING


_NAN = math.nan
_MU1 = math.pi / 2.0  # fator de escala da bipower (E[|Z|]² para Z~N(0,1) é 2/pi -> inverso)


def _d0(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RSum:
    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, v: float) -> None:
        ev = self.ring.push(v)
        self.total += v - _d0(ev)

    def __len__(self) -> int:
        return len(self.ring)


class JumpBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        jc = cfg.jumps
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(jc.windows)
        self.t = 0
        self.prev_abs: float | None = None
        self.prev_sign: float | None = None
        # por janela: RV=Σe², BV=Σ|e||e_prev|, RS+=Σe²1{e>0}, RS-, LEV=Σ sign(e_prev)|e|
        self.rv = {w: _RSum(w) for w in self.windows}
        self.bv = {w: _RSum(w) for w in self.windows}
        self.rsp = {w: _RSum(w) for w in self.windows}
        self.rsm = {w: _RSum(w) for w in self.windows}
        self.lev = {w: _RSum(w) for w in self.windows}
        self.abs_sum = {w: _RSum(w) for w in self.windows}  # Σ|e| para a média usada no leverage

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        ae = abs(e)
        se = e * e
        sgn = 1.0 if e > 0 else (-1.0 if e < 0 else 0.0)
        bp = ae * self.prev_abs if self.prev_abs is not None else 0.0
        lv = self.prev_sign * ae if self.prev_sign is not None else 0.0
        for w in self.windows:
            self.rv[w].push(se)
            self.bv[w].push(bp)
            self.rsp[w].push(se if e > 0 else 0.0)
            self.rsm[w].push(se if e < 0 else 0.0)
            self.lev[w].push(lv)
            self.abs_sum[w].push(ae)
        self.prev_abs = ae
        self.prev_sign = sgn

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            if not warm or len(self.rv[w]) < 3:
                out[f"jump_ratio_w{w:03d}"] = _NAN
                out[f"jump_rvbv_ln_w{w:03d}"] = _NAN
                out[f"jump_semivar_asym_w{w:03d}"] = _NAN
                out[f"jump_leverage_w{w:03d}"] = _NAN
                continue
            n = len(self.rv[w])
            rv = self.rv[w].total / n
            n_bv = len(self.bv[w])
            bv = _MU1 * self.bv[w].total / max(n_bv, 1)
            out[f"jump_ratio_w{w:03d}"] = max(rv - bv, 0.0) / (rv + 1e-12)
            out[f"jump_rvbv_ln_w{w:03d}"] = math.log(max(rv, 1e-12)) - math.log(max(bv, 1e-12))
            rsp = self.rsp[w].total
            rsm = self.rsm[w].total
            out[f"jump_semivar_asym_w{w:03d}"] = (rsp - rsm) / (rsp + rsm + 1e-12)
            out[f"jump_leverage_w{w:03d}"] = self.lev[w].total / max(n_bv, 1)
        return out


def jumps_history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio JumpBlock sobre o histórico (H0), para a calibração F1."""
    blk = JumpBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            acc.setdefault(name, []).append(val)
    return acc


# @crunch/keep:on
# ============================== sbrt/state/bocpd.py ==============================
"""BOCPDBlock — detecção bayesiana online de changepoint com posterior de run-length
(docs/RESULTADOS_P1_P4.md §próximo passo; Adams & MacKay 2007).

## Por que existe

A família mais valiosa que a sessão adicionou foi `varloc` (P3, 4,81% de SHAP): variância
*localizada* no changepoint via um max heurístico sobre escalas fixas. Isto é a versão PRINCIPIADA da
mesma ideia. Em vez de escolher a melhor de um punhado de janelas, o BOCPD mantém o **posterior
completo sobre o run-length** r_t = número de passos desde o último changepoint, por passo, via a
recursão de mensagem de Adams-MacKay — O(R_max) por passo, sem armazenar a série.

Modelo por regime: eₜ ~ N(0, σ²), σ² ~ Inverse-Gamma(α₀, β₀) (conjugado; preditiva Student-t, robusta
a cauda pesada — apropriado dado o censo). Hazard constante H=1/λ (prior geométrico no run-length).
Recursão: para cada eₜ, cresce cada hipótese de run-length pela verossimilhança preditiva e pelo
(1−H), e acumula massa em r=0 pela hazard. As estatísticas suficientes (contagem, Σe²) por
run-length crescem por mensagem.

Features:
- `bocpd_regime_var_ln` = ln(Σ_r p(r)·E[σ²|r]) — a variância do regime ATUAL, ponderada pelo
  posterior de run-length. É o estimador de variância limpo e localizado em tau que o oracle usa
  (INVESTIGACAO §2, AUC 0,856) e que as janelas fixas do banco diluem por não saber onde é tau.
- `bocpd_cp_prob` = Σ_{r<k} p(r) — probabilidade de um changepoint MUITO recente (alarme localizado).
- `bocpd_rl_entropy` = −Σ p(r) ln p(r) — incerteza da localização (baixa = changepoint nítido).
- `bocpd_map_runlen` = ln(1+argmax_r p(r)) — idade MAP do changepoint (localizador).

Consome `e` (escala congelada) — família de variância, trava CE2 (plano §3.4). A versão calibrada
(F1) usa o nulo da própria série (uma série naturalmente ruidosa acumula mais changepoints espúrios
no histórico), tornando `regime_var_ln`/`cp_prob`/`rl_entropy` comparáveis entre séries.
"""

import math
from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

_NAN = math.nan


class BOCPDBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        bc = cfg.bocpd
        self.warmup = cfg.features.warmup_min_n
        self.R = bc.r_max
        self.H = 1.0 / bc.hazard_lambda
        self.a0 = bc.alpha0
        self.b0 = bc.beta0
        self.recent_k = bc.recent_k
        self.t = 0

        r = np.arange(self.R + 1, dtype=np.float64)
        self.alpha = self.a0 + r / 2.0                 # α_r determinístico em r -> precomputa lgammas
        self._lg_ah = gammaln(self.alpha + 0.5)
        self._lg_a = gammaln(self.alpha)
        self._alpha_m1 = np.maximum(self.alpha - 1.0, 0.5)  # guarda para E[σ²|r]=β_r/(α_r−1)

        self.prob = np.zeros(self.R + 1, dtype=np.float64)
        self.prob[0] = 1.0
        self.sum_e2 = np.zeros(self.R + 1, dtype=np.float64)

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        se = e * e
        beta = self.b0 + self.sum_e2 / 2.0
        # log-preditiva Student-t de cada run-length (lgammas precomputados)
        logpred = (
            self._lg_ah - self._lg_a
            - 0.5 * np.log(2.0 * math.pi * beta)
            - (self.alpha + 0.5) * np.log1p(se / (2.0 * beta))
        )
        pred = np.exp(logpred - logpred.max())
        w = self.prob * pred
        cp = self.H * w.sum()          # massa de changepoint -> r=0
        growth = (1.0 - self.H) * w    # crescimento -> r+1

        new = np.empty(self.R + 1, dtype=np.float64)
        new[0] = cp
        new[1:] = growth[:-1]
        new[self.R] += growth[self.R]  # dobra o overflow no bin "≥R_max" (truncagem)
        s = new.sum()
        self.prob = new / s if s > 0 else new

        ns = np.empty(self.R + 1, dtype=np.float64)
        ns[0] = 0.0
        ns[1:] = self.sum_e2[:-1] + se
        ns[self.R] = self.sum_e2[self.R] + se  # o bin truncado continua acumulando
        self.sum_e2 = ns

    def features(self) -> dict[str, float]:
        if self.t < self.warmup:
            return {
                "bocpd_regime_var_ln": _NAN, "bocpd_cp_prob": _NAN,
                "bocpd_rl_entropy": _NAN, "bocpd_map_runlen": _NAN,
            }
        beta = self.b0 + self.sum_e2 / 2.0
        e_var = beta / self._alpha_m1                       # E[σ²|r]
        regime_var = float(np.dot(self.prob, e_var))
        p = self.prob
        pp = p[p > 1e-12]  # evita log(0) (numpy avalia ambos os ramos de np.where)
        entropy = float(-np.sum(pp * np.log(pp)))
        cp_prob = float(p[: self.recent_k].sum())
        map_rl = int(np.argmax(p))
        return {
            "bocpd_regime_var_ln": math.log(max(regime_var, 1e-12)),
            "bocpd_cp_prob": cp_prob,
            "bocpd_rl_entropy": entropy,
            "bocpd_map_runlen": math.log1p(map_rl),
        }


def bocpd_history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio BOCPDBlock sobre o histórico (H0), para a calibração F1. `map_runlen` não é
    calibrado (é um localizador de idade, não uma magnitude)."""
    blk = BOCPDBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            if name == "bocpd_map_runlen":
                continue
            acc.setdefault(name, []).append(val)
    return acc


# @crunch/keep:on
# ============================== sbrt/state/fingerprint.py ==============================
"""Impressão digital estendida do regime H0 (docs/PROPOSTA_FEATURES_V2.md F2).

Motivação empírica. A decomposição de SHAP do modelo mostrou que as features `meta_h0_*` —
constantes dentro de uma série — são a **maior família do modelo (34,3% do |SHAP|)**, apesar de o
CE6 provar que elas não carregam efeito principal (classificador só-histórico: AUC 0,5067). Ou seja,
um terço da capacidade explicativa do modelo é gasto **calibrando**: decidindo o quanto um dado
desvio é surpreendente *para uma série com esta cara*. O modelo está faminto por contexto de
calibração — e dar-lhe um retrato melhor da série custa **zero latência por passo** (tudo é
calculado uma vez em `fit_h0`).

Todos os descritores são funções apenas do histórico (H0 por definição) e portanto não podem vazar
rótulo; a checagem CE6 (`scripts/ce6_history_classifier.py`) deve continuar ≈0,5 depois desta adição
— se subir, é sinal de que o gerador correlaciona propriedades do histórico com a existência de
quebra, e isso mudaria a leitura de várias decisões do projeto.
"""

import math

import numpy as np


def _safe(value: float, default: float = 0.0) -> float:
    return float(value) if np.isfinite(value) else default


def _acf_fp(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 0:
        return 0.0
    return float(np.dot(xc[:-lag], xc[lag:]) / den)


def _hurst_aggvar(e: np.ndarray, scales: list) -> float:
    """Hurst pelo método da variância agregada: Var(média de blocos de tamanho m) ∝ m^(2H−2).
    H≈0,5 para ruído branco; H>0,5 indica memória longa (persistência)."""
    xs, ys = [], []
    n = len(e)
    for m in scales:
        if m < 1 or n // m < 8:
            continue
        k = n // m
        blocks = e[: k * m].reshape(k, m).mean(axis=1)
        v = float(blocks.var(ddof=1)) if k > 1 else 0.0
        if v > 0:
            xs.append(math.log(m))
            ys.append(math.log(v))
    if len(xs) < 3:
        return 0.5
    slope = float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
    return _safe(1.0 + slope / 2.0, 0.5)


def _hill_xi(e: np.ndarray, frac: float) -> float:
    """Estimador de Hill do índice de cauda sobre |e|. Retorna xi = 1/alpha (xi maior = cauda mais
    pesada); 0 para cauda fina. Complementa `nu_hat` (que vem da curtose e é sensível a outliers de
    forma diferente)."""
    a = np.sort(np.abs(np.asarray(e, dtype=np.float64)))[::-1]
    n = len(a)
    k = max(10, int(frac * n))
    if n < 20 or k >= n:
        return 0.0
    thresh = a[k]
    if thresh <= 1e-12:
        return 0.0
    top = a[:k]
    top = top[top > 0]
    if len(top) < 2:
        return 0.0
    return _safe(float(np.mean(np.log(top / thresh))), 0.0)


def _spectral_slope(e: np.ndarray) -> float:
    """Inclinação da log-periodograma vs. log-frequência. ≈0 para ruído branco; negativa indica
    dominância de baixa frequência (drift/memória longa); positiva, alta frequência."""
    n = len(e)
    if n < 64:
        return 0.0
    x = e - e.mean()
    psd = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    mask = (freqs > 0) & (psd > 0)
    if mask.sum() < 16:
        return 0.0
    lf, lp = np.log(freqs[mask]), np.log(psd[mask])
    return _safe(float(np.polyfit(lf, lp, 1)[0]), 0.0)


def _ljung_box(x: np.ndarray, max_lag: int) -> float:
    """Q de Ljung-Box normalizada por graus de liberdade (Q/L) — massa de dependência linear.
    Aplicada a |e| mede clustering de volatilidade; ≈1 sob independência."""
    n = len(x)
    if n < max_lag + 10:
        return 1.0
    q = 0.0
    for lag in range(1, max_lag + 1):
        r = _acf_fp(x, lag)
        q += r * r / max(n - lag, 1)
    return _safe(n * (n + 2) * q / max_lag, 1.0)


def compute_fingerprint(e_hist: np.ndarray, hist: np.ndarray, q: dict, cfg) -> dict:
    """Descritores escalares do regime H0. Chamado uma vez por série em `fit_h0`; custo O(n_h log n_h)
    dominado pela FFT/ordenação, desprezível frente ao que `fit_h0` já faz."""
    fp_cfg = cfg.h0_fingerprint
    e = np.asarray(e_hist, dtype=np.float64)
    abs_e = np.abs(e)
    n = len(e)

    lags = list(range(1, fp_cfg.acf_max_lag + 1))
    acf_abs = [abs(_acf_fp(abs_e, l)) for l in lags]
    acf_mass = float(np.mean(acf_abs)) if acf_abs else 0.0

    # decaimento: inclinação de log|acf| vs log(lag) (mais negativa = dependência morre mais rápido)
    xs, ys = [], []
    for l, a in zip(lags, acf_abs):
        if a > 1e-6:
            xs.append(math.log(l))
            ys.append(math.log(a))
    acf_decay = _safe(float(np.polyfit(np.array(xs), np.array(ys), 1)[0]), 0.0) if len(xs) >= 3 else 0.0

    w = fp_cfg.volvol_window
    if n >= 4 * w:
        k = n // w
        block_var = e[: k * w].reshape(k, w).var(axis=1, ddof=1)
        mean_bv = float(block_var.mean())
        volvol = _safe(float(block_var.std(ddof=1)) / mean_bv, 0.0) if mean_bv > 1e-12 else 0.0
    else:
        volvol = 0.0

    q01, q25, q75, q99 = q["0.01"], q["0.25"], q["0.75"], q["0.99"]
    tail_span = q99 - q01
    iqr_tail_ratio = _safe((q75 - q25) / tail_span, 0.0) if tail_span > 1e-12 else 0.0

    return {
        "hurst": _hurst_aggvar(e, list(fp_cfg.hurst_scales)),
        "hill_xi": _hill_xi(e, fp_cfg.hill_frac),
        "acf_e2_l1": _safe(_acf_fp(e * e, 1), 0.0),
        "acf_abs_mass": acf_mass,
        "acf_decay": acf_decay,
        "spectral_slope": _spectral_slope(e),
        "ljungbox_abs": _ljung_box(abs_e, fp_cfg.acf_max_lag),
        "volvol": volvol,
        "iqr_tail_ratio": iqr_tail_ratio,
    }


# @crunch/keep:on
# ============================== sbrt/state/calibration.py ==============================
"""Calibração de nulo por série (docs/PROPOSTA_FEATURES_V2.md F1) — o item de maior leverage da
proposta V2, e o único motivado por uma medição direta do modelo atual em vez de analogia externa.

## O problema que isto resolve

A TS-AUC ordena **séries diferentes no mesmo passo**. Uma estatística cujo nível sob H0 depende das
idiossincrasias da série (curtose, dependência, clustering de volatilidade) está intrinsecamente
mal-ordenada nessa seção transversal: um `ranktwo_wilcoxon_z` de 2,5 significa coisas diferentes numa
série i.i.d. e numa série com forte clustering, porque o z é normalizado por √(12·w) — uma fórmula
que **assume independência**. Com p-values consecutivos correlacionados, a variância verdadeira da
média de janela é muito maior, e o z fica sistematicamente inflado justamente nas séries mais
dependentes.

Hoje o modelo corrige isso sozinho, aprendendo interações `meta_h0_* × estatística` — e é por isso
que as `meta_h0_*` consomem **34,3% do |SHAP|** apesar de o CE6 mostrar que não carregam efeito
principal (AUC 0,5067). Um terço do orçamento do modelo é gasto reconstruindo uma calibração que
podemos simplesmente calcular.

## A ideia

O histórico **é H0 por definição** (livre de quebra, plano §3). Então basta deslizar a MESMA
estatística sobre o histórico da própria série para obter a distribuição nula dela *naquela série*,
e emitir, além do valor cru, o desvio padronizado contra esse nulo:

    S_cal(t) = (S(t) − μ_nulo) / σ_nulo

Custo: O(n_h) uma vez por série dentro do `fit_h0` (que já é O(n_h log n_h)); **zero µs por passo**
além de uma subtração e uma divisão.

## Decisões de implementação (e por quê)

- **Só estatísticas baseadas em `e`** (variância/cauda/rank). As de média usam `e_vol`, cuja
  reprodução exigiria replicar a EWMA de volatilidade sobre o histórico; e o censo A1 mostra que o
  canal de média é quase morto (6,8% das séries com |Δmean_e|>0,3) — não vale a complexidade.
- **`_cal` só quando a janela está cheia** (`t >= min_t`). Para t < w o estatístico online usa
  n_eff = t, cuja distribuição nula é outra (σ de `ln E[e²]` escala com √(2/n_eff)); calibrar com o
  nulo de janela cheia daria um número errado. NaN é tratado nativamente pelo LightGBM e é a resposta
  honesta: ainda não há janela suficiente.
- **Encolhimento para o nulo teórico** onde ele é conhecido. Com janela w e histórico n_h há apenas
  ~n_h/w janelas *independentes* (4 a 20 para w=250), então σ empírico é ruidoso. Encolhemos para o
  σ teórico i.i.d. com peso n_eff/(n_eff+pseudo). Onde não há teoria (MMD, Haar), usa-se o empírico
  puro — lá o número de amostras efetivas é alto porque a escala de tempo do estatístico (1/λ) é
  muito menor que n_h.
- **Vetorização.** O cálculo sobre o histórico é vetorizado, enquanto o online é recursivo. Isto NÃO
  é a armadilha §13.2 do plano ("backtest vetorizado ≠ execução causal"): não produz features de
  treino nem scores, produz uma *constante por série* a partir de dados que já são H0. Ainda assim a
  equivalência é verificada por testes dedicados (`tests/unit/test_calibration.py`,
  `test_mmd.py`, `test_multiscale.py`), porque um desalinhamento aqui envenenaria silenciosamente
  todas as features calibradas.
"""

import math
from typing import NamedTuple

import numpy as np


# P(|Z| > 2) para Z ~ N(0,1) — taxa nominal de excedência usada por `accum_window_exceed2_frac_*`.
_P0_EXCEED2 = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(2.0 / math.sqrt(2.0))))

# Piso de amostras para aplicar a extrapolação de escala: abaixo disto as aproximações assintóticas
# do nulo teórico (ex.: dp de ln(chi²_n/n) ≈ sqrt(2/n)) são ruins demais para valer a pena.
_MIN_N_FOR_SCALING = 10


class NullSpec(NamedTuple):
    """Nulo de uma estatística, medido na JANELA CHEIA, mais o necessário para transportá-lo a
    janelas parciais (t < w).

    `kind` diz como o nulo escala com o número efetivo de amostras n = min(t, window):
    - `"z"`      : já normalizado por sqrt(n) na origem (ranktwo) -> nulo não depende de n;
    - `"var_ln"` : ln(média de e²) -> mu_teo(n) = −1/n, dp_teo(n) = sqrt(2/n);
    - `"frac"`   : fração de excedência -> mu_teo(n) = p0, dp_teo(n) = sqrt(p0(1−p0)/n);
    - `"rho"`    : autocorrelação -> mu ~ const, dp_teo(n) ∝ 1/sqrt(n) (P1, dependência);
    - `"none"`   : sem lei de escala conhecida (MMD, Haar, massa multi-lag) -> só vale na janela cheia.

    A ideia do transporte: o que a série tem de idiossincrático é o *fator de inflação* em relação ao
    nulo i.i.d. (k = dp_medido / dp_teórico), não o nível absoluto. Esse fator é aproximadamente
    constante em n para uma série estacionária, então podemos aplicá-lo ao dp teórico de qualquer n.
    Isso libera a versão calibrada muito antes de a janela encher — exatamente no regime de t pequeno
    onde o modelo é mais fraco e onde antes essas colunas eram 100% NaN."""

    mu: float
    sd: float
    min_t: int
    kind: str = "none"
    window: int = 0
    aux: float = 0.0  # p0, para kind="frac"


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Média móvel de janela cheia (comprimento len(x)-w+1). Equivale ao que o bloco online emite
    quando o ring está cheio."""
    if len(x) < w or w < 1:
        return np.empty(0, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    return (c[w:] - c[:-w]) / w


def _upper_tail_p_vec(sorted_arr: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Versão vetorizada de `conformal._upper_tail_p` (mid-rank, cauda superior)."""
    lo = np.searchsorted(sorted_arr, x, side="left")
    hi = np.searchsorted(sorted_arr, x, side="right")
    mid_rank = (lo + hi) / 2.0
    return (n - mid_rank + 0.5) / (n + 1.0)


def _add(
    out: dict,
    name: str,
    arr: np.ndarray,
    min_t: int,
    n_eff: float,
    theory: tuple | None,
    pseudo: float,
    kind: str = "none",
    window: int = 0,
    aux: float = 0.0,
) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        return
    mu_emp = float(arr.mean())
    sd_emp = float(arr.std(ddof=1))

    if theory is not None and pseudo > 0.0:
        mu_th, sd_th = theory
        wgt = n_eff / (n_eff + pseudo)
        mu = wgt * mu_emp + (1.0 - wgt) * mu_th
        sd = math.sqrt(max(wgt * sd_emp ** 2 + (1.0 - wgt) * sd_th ** 2, 1e-12))
    else:
        mu, sd = mu_emp, sd_emp

    if not (np.isfinite(mu) and np.isfinite(sd)):
        return
    # Com lei de escala conhecida a calibração vale desde cedo (ver NullSpec); sem ela, só na
    # janela cheia.
    effective_min_t = min(int(min_t), _MIN_N_FOR_SCALING) if kind != "none" else int(min_t)
    out[name] = NullSpec(mu, max(sd, 1e-6), effective_min_t, kind, int(window), float(aux))


def compute_null_stats(
    e_hist: np.ndarray,
    sorted_e_hist: np.ndarray,
    sorted_abs_e_hist: np.ndarray,
    rff_href: np.ndarray,
    rff_href_joint: np.ndarray,
    cfg,
) -> dict:
    """{nome_da_feature: (mu_nulo, sd_nulo, min_t)}. Chamado uma vez por série em `fit_h0`."""
    cal_cfg = cfg.calibration
    if not cal_cfg.enabled:
        return {}

    e = np.asarray(e_hist, dtype=np.float64)
    n_h = len(e)
    if n_h < 64:
        return {}

    out: dict = {}
    pseudo = cal_cfg.shrink_pseudo
    e2 = e * e
    exceed2 = (np.abs(e) > 2.0).astype(np.float64)

    # --- accum: variância de janela (ln) e fração de excedência ---
    for w in cfg.state.window_sizes:
        arr = np.log(np.maximum(_rolling_mean(e2, w), 1e-12))
        # teoria i.i.d. gaussiana: w·E[e²] ~ chi²_w  =>  ln(E[e²]) tem média ≈ -1/w e dp ≈ sqrt(2/w)
        _add(out, f"accum_window_var_ln_w{w:03d}", arr, min_t=w, n_eff=n_h / w,
             theory=(-1.0 / w, math.sqrt(2.0 / w)), pseudo=pseudo, kind="var_ln", window=w)

    for w in cfg.state.exceedance_windows:
        arr = _rolling_mean(exceed2, w)
        _add(out, f"accum_window_exceed2_frac_w{w:03d}", arr, min_t=w, n_eff=n_h / w,
             theory=(_P0_EXCEED2, math.sqrt(_P0_EXCEED2 * (1.0 - _P0_EXCEED2) / w)), pseudo=pseudo,
             kind="frac", window=w, aux=_P0_EXCEED2)

    # --- ranktwo (R4): z de Wilcoxon e de dispersão ---
    p_right = _upper_tail_p_vec(sorted_e_hist, e, n_h)
    p_abs = _upper_tail_p_vec(sorted_abs_e_hist, np.abs(e), n_h)
    for w in cfg.rank_twosample.windows:
        scale = math.sqrt(12.0 * w)
        _add(out, f"ranktwo_wilcoxon_z_w{w:03d}", _rolling_mean(p_right - 0.5, w) * scale,
             min_t=w, n_eff=n_h / w, theory=(0.0, 1.0), pseudo=pseudo, kind="z", window=w)
        _add(out, f"ranktwo_dispersion_z_w{w:03d}", _rolling_mean(p_abs - 0.5, w) * scale,
             min_t=w, n_eff=n_h / w, theory=(0.0, 1.0), pseudo=pseudo, kind="z", window=w)

    # --- MMD (F3): sem teoria fechada -> nulo empírico puro, só na janela cheia ---
    mmd_series = mmd_history_series(e, rff_href, rff_href_joint, cfg)
    taus = {
        "_vfast": int(1.0 / max(cfg.mmd.lambda_vfast, 1e-9)),
        "_fast": int(1.0 / max(cfg.mmd.lambda_fast, 1e-9)),
        "_slow": int(1.0 / max(cfg.mmd.lambda_slow, 1e-9)),
    }
    tau_slow = taus["_slow"]
    for name, arr in mmd_series.items():
        min_t = next((v for suf, v in taus.items() if name.endswith(suf)), tau_slow)
        # descarta o transiente inicial da EWMA antes de medir o nulo
        _add(out, name, arr[min_t:], min_t=min_t, n_eff=len(arr), theory=None, pseudo=0.0)

    # --- dependência (P1): roda o próprio DependenceBlock sobre o histórico (garante equivalência
    # online/nulo por construção). ρ₁ de |e|/e² tem kind="rho" (escala 1/sqrt(n), disponível cedo);
    # a massa multi-lag não tem lei fechada -> kind="none" (janela cheia). ---
    dep_series = dependence_history_null_series(e, cfg)
    for name, series in dep_series.items():
        w = int(name.rsplit("_w", 1)[1])
        kind = "rho" if "rho1" in name else "none"
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=w, n_eff=n_h / w,
             theory=None, pseudo=0.0, kind=kind, window=w)

    # --- BOCPD (localização principiada): nulo empírico da própria série -- uma série ruidosa
    # acumula changepoints espúrios no histórico, então a versão calibrada só acende no excesso. ---
    bocpd_series = bocpd_history_null_series(e, cfg)
    for name, series in bocpd_series.items():
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=cfg.features.warmup_min_n,
             n_eff=len(series), theory=None, pseudo=0.0)

    # --- variância localizada (P3): max/min_z já são z-scores; nulo empírico corrige a inflação por
    # curtose da série (D-10). recent_vs_lagged só existe com a janela cheia. ---
    varloc_series = varloc_history_null_series(e, cfg)
    rl_min_t = cfg.varloc.recent + cfg.varloc.lagged
    for name, series in varloc_series.items():
        min_t = rl_min_t if name.endswith("recent_vs_lagged") else cfg.features.warmup_min_n
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=min_t, n_eff=len(series),
             theory=None, pseudo=0.0)

    # --- bipower/saltos/leverage (P4): nulo empírico da própria série (a razão de salto e o leverage
    # de um GARCH são altos no seu histórico -> a versão calibrada só acende no excesso pós-quebra) ---
    jump_series = jumps_history_null_series(e, cfg)
    for name, series in jump_series.items():
        w = int(name.rsplit("_w", 1)[1])
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=w, n_eff=n_h / w,
             theory=None, pseudo=0.0)

    # --- Haar multi-escala (F4): idem ---
    haar_series = multiscale_history_series(e, cfg)
    n_scales, min_coeffs = cfg.multiscale.n_scales, cfg.multiscale.warmup_min_coeffs
    for name, arr in haar_series.items():
        if name.startswith("haar_energy_ln_s"):
            j = int(name.rsplit("s", 1)[1])
        elif name == "haar_contrast_fine_mid":
            # o contraste fino-vs-médio depende da escala 2, NÃO da mais grossa: usar n_scales-1
            # aqui mantinha a feature em NaN até t=96 sem necessidade nenhuma.
            j = min(2, n_scales - 1)
        else:
            j = n_scales - 1
        _add(out, name, arr, min_t=(2 ** (j + 1)) * min_coeffs, n_eff=len(arr), theory=None, pseudo=0.0)

    return out


def _null_at(spec: NullSpec, t: int) -> tuple[float, float]:
    """(mu, sd) do nulo no número efetivo de amostras n = min(t, window), transportando o nulo
    medido na janela cheia pela lei de escala de `spec.kind` (ver NullSpec)."""
    if spec.kind == "none" or spec.kind == "z" or spec.window <= 0:
        return spec.mu, spec.sd

    n = max(min(t, spec.window), 1)
    if n >= spec.window:
        return spec.mu, spec.sd

    if spec.kind == "rho":
        # autocorrelação: média ~ const, dp ∝ 1/sqrt(n) -> transporta a dp da janela cheia por
        # sqrt(W/n), preservando o fator de inflação idiossincrático da série.
        return spec.mu, max(spec.sd * math.sqrt(spec.window / n), 1e-9)

    if spec.kind == "var_ln":
        mu_th_w, sd_th_w = -1.0 / spec.window, math.sqrt(2.0 / spec.window)
        mu_th_n, sd_th_n = -1.0 / n, math.sqrt(2.0 / n)
    elif spec.kind == "frac":
        p0 = spec.aux
        var_w = max(p0 * (1.0 - p0) / spec.window, 1e-18)
        var_n = max(p0 * (1.0 - p0) / n, 1e-18)
        mu_th_w, sd_th_w = p0, math.sqrt(var_w)
        mu_th_n, sd_th_n = p0, math.sqrt(var_n)
    else:
        return spec.mu, spec.sd

    # o que é idiossincrático da série é o fator de inflação sobre o nulo i.i.d., não o nível
    inflation = spec.sd / max(sd_th_w, 1e-12)
    mu = mu_th_n + (spec.mu - mu_th_w)
    sd = max(inflation * sd_th_n, 1e-9)
    return mu, sd


def apply_calibration(feats: dict, null_stats: dict, t: int) -> None:
    """Acrescenta `<nome>_cal` a `feats`, in-place. NaN quando ainda não há amostras suficientes
    (t < min_t) ou quando o valor cru é NaN — nunca inventa um número."""
    for name, spec in null_stats.items():
        raw = feats.get(name)
        if raw is None or t < spec.min_t or not math.isfinite(raw):
            feats[f"{name}_cal"] = math.nan
            continue
        mu, sd = _null_at(spec, t)
        feats[f"{name}_cal"] = (raw - mu) / sd


# @crunch/keep:on
# ============================== sbrt/state/h0.py ==============================
"""Fase-histórico: caracterização do regime H0 e whitening causal (plano §3).

Executado uma vez por série, sobre o histórico completo (livre de quebra por definição). Tudo aqui é
determinístico e O(n_h*p + n_h log n_h). `H0Params` é imutável — não existe `.refit()`: torna
estruturalmente impossível reestimar o H0 no meio do online (bloqueio B2 do plano técnico, §2.2-B2).
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np


@dataclass(frozen=True)
class H0Params:
    phi: np.ndarray
    c: float
    mu0: float
    sigma0: float
    sigma_e: float
    sigma_e_rob: float
    nu_hat: float
    q: dict
    sorted_e_hist: np.ndarray
    sorted_abs_e_hist: np.ndarray
    sigma_u: float
    rho1_e: float
    rho1_abs_e: float
    seasonal_lag: int | None
    seasonal_coef: float
    ar_r2: float
    n_h: int
    last_hist_e: float
    lag_seed: np.ndarray
    # --- proposta V2 (docs/PROPOSTA_FEATURES_V2.md), tudo calculado uma vez aqui e nunca no online ---
    fingerprint: dict          # F2: descritores estendidos do regime H0 (state/fingerprint.py)
    rff_href: np.ndarray       # F3: média de z(e) sobre o histórico, referência congelada do MMD
    rff_href_joint: np.ndarray # F3: idem para o par (e_t, e_{t-1})
    null_stats: dict           # F1: {feature: (mu, sd, min_t)} do nulo da própria série

    @property
    def lag_capacity(self) -> int:
        base = len(self.phi)
        return max(base, self.seasonal_lag or 0)


def _design_matrix(hist: np.ndarray, lags: list) -> tuple[np.ndarray, np.ndarray]:
    """Monta [1, x_{t-l1}, x_{t-l2}, ...] -> x_t para t = max(lags)+1 .. n_h (1-based)."""
    max_lag = max(lags)
    n_h = len(hist)
    rows = n_h - max_lag
    X = np.empty((rows, len(lags) + 1), dtype=np.float64)
    X[:, 0] = 1.0
    for i, lag in enumerate(lags):
        X[:, i + 1] = hist[max_lag - lag: max_lag - lag + rows]
    y = hist[max_lag: max_lag + rows]
    return X, y


def _acf_h0(x: np.ndarray, lag: int) -> float:
    x = x - x.mean()
    n = len(x)
    if lag <= 0 or lag >= n:
        return 0.0
    num = float(np.dot(x[: n - lag], x[lag:]))
    den = float(np.dot(x, x))
    return num / den if den > 0 else 0.0


def fit_h0(hist: np.ndarray, cfg: "Config") -> H0Params:
    """plano §3.1. Puro e determinístico. ValueError se n_h < mínimo configurado."""
    hist = np.asarray(hist, dtype=np.float64)
    n_h = len(hist)
    if n_h < cfg.h0.min_hist_len:
        raise ValueError(f"histórico com {n_h} pontos, mínimo exigido {cfg.h0.min_hist_len}")

    mu0 = float(hist.mean())
    sigma0 = float(hist.std(ddof=1)) if n_h > 1 else 1.0
    sigma0 = max(sigma0, 1e-8)

    p = cfg.h0.ar_order
    base_lags = list(range(1, p + 1))

    # AR(p) via mínimos quadrados (equivalente a Yule-Walker/Levinson-Durbin para este propósito,
    # e permite adicionar o lag sazonal ao mesmo design matrix sem recursão separada, §3.1 item 3).
    var_x = float(hist.var(ddof=1)) if n_h > 1 else 1.0
    var_x = max(var_x, 1e-12)

    if n_h > p + 5:
        X, y = _design_matrix(hist, base_lags)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        var_resid = float(resid.var(ddof=1)) if len(resid) > 1 else var_x
        ar_r2 = 1.0 - var_resid / var_x
    else:
        coef = np.zeros(p + 1)
        resid = hist - mu0
        ar_r2 = 0.0

    accept_ar = ar_r2 >= cfg.h0.ar_r2_min_reduction
    if accept_ar:
        c = float(coef[0])
        phi = coef[1:].copy()
    else:
        c = mu0
        phi = np.zeros(p)
        resid = hist[p:] - mu0

    seasonal_lag: int | None = None
    seasonal_coef = 0.0
    lo, hi = cfg.h0.seasonal_lag_range
    if len(resid) > hi + 10:
        best_lag, best_abs_rho = None, cfg.h0.seasonal_acf_threshold
        for lag in range(lo, hi + 1):
            rho = _acf_h0(resid, lag)
            if abs(rho) > best_abs_rho:
                best_abs_rho = abs(rho)
                best_lag = lag
        if best_lag is not None:
            seasonal_lags = base_lags + [best_lag]
            X2, y2 = _design_matrix(hist, seasonal_lags)
            coef2, *_ = np.linalg.lstsq(X2, y2, rcond=None)
            resid2 = y2 - X2 @ coef2
            var_resid2 = float(resid2.var(ddof=1)) if len(resid2) > 1 else var_x
            ar_r2 = 1.0 - var_resid2 / var_x
            c = float(coef2[0])
            phi = coef2[1: 1 + p].copy()
            seasonal_coef = float(coef2[-1])
            seasonal_lag = best_lag
            resid = resid2

    sigma_e = float(resid.std(ddof=1)) if len(resid) > 1 else 1.0
    sigma_e = max(sigma_e, 1e-8)
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    sigma_e_rob = max(1.4826 * mad, 1e-8)

    e_hist = resid / sigma_e

    m2 = float(np.mean(e_hist ** 2))
    m4 = float(np.mean(e_hist ** 4))
    kappa_ex = m4 / (m2 ** 2) - 3.0 if m2 > 0 else 0.0
    nu_lo, nu_hi = cfg.h0.nu_clip
    nu_hat = float(np.clip(4.0 + 6.0 / kappa_ex, nu_lo, nu_hi)) if kappa_ex > 0 else nu_hi

    q = {f"{level:.2f}": float(np.quantile(e_hist, level)) for level in cfg.h0.quantile_levels}

    rho1_e = _acf_h0(e_hist, 1)
    rho1_abs_e = _acf_h0(np.abs(e_hist), 1)

    if len(e_hist) > 2:
        u = e_hist[1:] * e_hist[:-1]
        sigma_u = float(u.std(ddof=1)) if len(u) > 1 else 1.0
    else:
        sigma_u = 1.0
    sigma_u = max(sigma_u, 1e-8)

    sorted_e_hist = np.sort(e_hist)
    sorted_abs_e_hist = np.sort(np.abs(e_hist))
    last_hist_e = float(e_hist[-1])

    lag_capacity = max(p, seasonal_lag or 0)
    lag_seed = hist[-lag_capacity:].copy()

    # --- proposta V2: tudo abaixo é função APENAS do histórico (H0 por definição), calculado uma
    # vez por série. Nada disso adiciona custo ao laço online. ---
    fingerprint = compute_fingerprint(e_hist, hist, q, cfg)
    rff_href, rff_href_joint = mmd_history_reference(e_hist, cfg)
    null_stats = compute_null_stats(
        e_hist, sorted_e_hist, sorted_abs_e_hist, rff_href, rff_href_joint, cfg
    )

    return H0Params(
        phi=phi,
        c=c,
        mu0=mu0,
        sigma0=sigma0,
        sigma_e=sigma_e,
        sigma_e_rob=sigma_e_rob,
        nu_hat=nu_hat,
        q=q,
        sorted_e_hist=sorted_e_hist,
        sorted_abs_e_hist=sorted_abs_e_hist,
        sigma_u=sigma_u,
        rho1_e=rho1_e,
        rho1_abs_e=rho1_abs_e,
        seasonal_lag=seasonal_lag,
        seasonal_coef=seasonal_coef,
        ar_r2=ar_r2,
        n_h=n_h,
        last_hist_e=last_hist_e,
        lag_seed=lag_seed,
        fingerprint=fingerprint,
        rff_href=rff_href,
        rff_href_joint=rff_href_joint,
        null_stats=null_stats,
    )


def seed_lag_buffer(params: H0Params) -> RingBuffer:
    """Semeia o ring de lags com a cauda do histórico (`params.lag_seed`, guardada por `fit_h0`) —
    garante continuidade exata na fronteira histórico->online (plano §3.1 item 8, armadilha §13.3):
    e_1 do online usa os mesmos lags que teriam sido usados se o histórico continuasse."""
    buf = RingBuffer(params.lag_capacity)
    for x in params.lag_seed:
        buf.push(float(x))
    return buf


def whiten_step(x: float, lags: RingBuffer, params: H0Params, cfg: "Config") -> tuple[float, float]:
    """plano §3.2. Retorna (e_clipado, e_raw); empurra x em `lags`. `params` é imutável — nunca
    reestimado no online (bloqueio B2)."""
    x_hat = params.c
    for j, phi_j in enumerate(params.phi):
        x_hat += phi_j * lags.peek(j)
    if params.seasonal_lag is not None:
        x_hat += params.seasonal_coef * lags.peek(params.seasonal_lag - 1)

    e_raw = (x - x_hat) / params.sigma_e
    lo, hi = cfg.h0.clip_e
    e_clip = float(np.clip(e_raw, lo, hi))

    lags.push(float(x))
    return e_clip, float(e_raw)


# @crunch/keep:on
# ============================== sbrt/postprocess/monotonicity.py ==============================
"""Decisão sobre monotonicidade do score (plano §7). Default = V-livre (identidade): o posterior
P(tau<=t|dados) é não-monótono por natureza (evidência transitória deve decair) e o max-hold trava
alarmes falsos nas séries sem quebra (contraexemplo CE1, plano §12.5). Variantes com retenção só
seriam adotadas mediante confirmação por submissão oficial — nunca por métrica local (§9)."""

from typing import TYPE_CHECKING, Literal

Mode = Literal["free", "hold", "soft", "ema"]


def apply_monotonicity(p: float, prev: float | None, mode: Mode, cfg: "Config") -> float:
    """mode='free' (default) = identidade. 'hold'/'soft'/'ema' só habilitados em
    configs/default.yaml se o gate G-mono (plano §9) tiver sido confirmado por submissão oficial."""
    if prev is None or mode == "free":
        return p
    if mode == "hold":
        return max(prev, p)
    if mode == "soft":
        return max(p, prev - cfg.postprocess.soft_decay)
    if mode == "ema":
        alpha = cfg.postprocess.ema_alpha
        return alpha * p + (1.0 - alpha) * prev
    raise ValueError(f"modo de monotonicidade desconhecido: {mode!r}")


# @crunch/keep:on
# ============================== sbrt/model/fallback.py ==============================
"""Fallback puro-estatístico (plano §8.5) — caminho de emergência determinístico, sem ML. Também
serve de baseline (ii) do gate G-0 e de score por padrão até a camada supervisionada (Frente H) ser
treinada e congelada.

score = sigma(w_lo * LO_{1/400} + w_cusum * sqrt(2 * max(banco_CUSUM)) + w_conformal * logM_abs_reset - bias)

O banco de CUSUM acumula log-likelihood-ratios truncadas em 0 (recursão max); sob H0, 2*LLR se
comporta como qui-quadrado, então sqrt(2*LLR) é uma transformação monótona para uma escala ~z,
usada aqui só para combinar grandezas heterogêneas num único logit (não entra no LightGBM, que
recebe as features cruas — plano §5).
"""

import math
from typing import TYPE_CHECKING

_CUSUM_BANK_KEYS = (
    "cusum_mean_pos_d025", "cusum_mean_pos_d050", "cusum_mean_pos_d100",
    "cusum_mean_neg_d025", "cusum_mean_neg_d050", "cusum_mean_neg_d100",
    "cusum_var_up_r150", "cusum_var_up_r250", "cusum_var_down_r050",
    "cusum_exceed_q95", "cusum_exceed_q99",
    "cusum_sign_pos", "cusum_sign_neg",
    "cusum_dep_pos", "cusum_dep_neg",
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def fallback_score(feats: dict, cfg: "Config") -> float:
    lo = feats.get("bayes_lo_h0025", 0.0)

    max_cusum = 0.0
    for k in _CUSUM_BANK_KEYS:
        v = feats.get(k)
        if v is not None and not math.isnan(v) and v > max_cusum:
            max_cusum = v
    z_cusum = math.sqrt(2.0 * max_cusum)

    logm = feats.get("conformal_logm_abs_reset", 0.0)

    fb = cfg.fallback
    logit = fb.w_lo * lo + fb.w_cusum * z_cusum + fb.w_conformal * logm - fb.bias
    return _sigmoid(logit)


# @crunch/keep:on
# ============================== sbrt/state/scorer.py ==============================
"""StreamScorer — motor único (plano §15.1, §8.1): o mesmo laço gera as features de treino
(`model/dataset.py`) e roda na inferência real. Nenhuma implementação vetorizada paralela existe —
isso elimina por construção a classe de bug "backtest vetorizado != execução causal" (armadilha
§13.2, docs/PLANO_REPOSITORIO.md §1).

Features #26 (hedge bruto, precisa de x cru), #27 (meta-t) e #28 (meta H0) e #25 (concordância de
localizadores, cruza bayes+cusum) não cabem no contrato `StateBlock` — são calculadas aqui.
"""

import math
from typing import TYPE_CHECKING


def default_blocks() -> list:
    return [
        AccumulatorBlock(),
        CusumBlock(),
        BayesFilterBlock(),
        ConformalBlock(),
        RankTwoSampleBlock(),
        MMDBlock(),          # F3 (proposta V2): MMD de kernel via RFF, marginal e conjunto
        MultiScaleBlock(),   # F4 (proposta V2): energia por escala (Haar diádico causal)
        DependenceBlock(),   # P1 (INVESTIGACAO §4.1): dependência não-linear/multi-lag
        VarLocBlock(),       # P3 (INVESTIGACAO §3): variância localizada no changepoint
        JumpBlock(),         # P4 (INVESTIGACAO §4.3): bipower/saltos + leverage (precisão T6/T9)
        BOCPDBlock(),        # (RESULTADOS_P1_P4): posterior de run-length de variância (localização
                             # principiada -- a versão correta do varloc, que foi a família mais valiosa)
    ]
    # P2 (LMomentBlock) foi PODADA aqui: medida em 0,51% de SHAP transversal por ~65 µs/passo (o mais
    # caro do banco) -- ROI claramente negativo (docs/RESULTADOS_P1_P4.md + SHAP do V4). O bloco e o
    # teste continuam em state/lmoments.py, reabríveis, mas fora do pipeline.


class StreamScorer:
    def __init__(self, h0: H0Params, blocks: list, ensemble, cfg: "Config"):
        self.h0 = h0
        self.blocks = blocks
        self.ensemble = ensemble
        self.cfg = cfg

        self.lags = seed_lag_buffer(h0)
        self.t = 0
        self._prev_score: float | None = None

        self.use_vol_adjust = h0.rho1_abs_e > cfg.state.vol_adjust["threshold_rho1_abs"]
        self.lambda_v = cfg.state.vol_adjust["lambda_v"]
        self.v = 1.0

        self._hedge_ewma = 0.0
        self._hedge_ring = RingBuffer(cfg.state.hedge_window)
        self._hedge_sum = 0.0
        self._hedge_sumsq = 0.0

        for b in self.blocks:
            b.reset(h0, cfg)

    def update_features(self, x: float) -> dict[str, float]:
        """Um passo: whiten_step -> update() de cada block -> merge + meta-features (t, n_h, nu_hat,
        rho1, ...). MOTOR ÚNICO: usado tanto por update() quanto por model/dataset.py."""
        self.t += 1
        t = self.t
        e, e_raw = whiten_step(x, self.lags, self.h0, self.cfg)

        if self.use_vol_adjust:
            self.v = ewma_update(self.v, e * e, self.lambda_v)
            e_vol = e / math.sqrt(max(self.v, 1e-12))
        else:
            e_vol = e

        feats: dict[str, float] = {}
        for b in self.blocks:
            b.update(e, e_raw, e_vol, t)
            feats.update(b.features())

        self._update_hedge(x)
        wmin = self.cfg.features.warmup_min_n
        n_eff_hedge = min(t, self.cfg.state.hedge_window)
        feats["hedge_ewma_z"] = self._hedge_ewma / max(self.h0.sigma0, 1e-8) if t >= wmin else math.nan
        if t >= wmin and n_eff_hedge > 1:
            mean_w = self._hedge_sum / n_eff_hedge
            var_w = max(self._hedge_sumsq / n_eff_hedge - mean_w * mean_w, 1e-12)
            feats["hedge_window_var_ln"] = math.log(var_w)
        else:
            feats["hedge_window_var_ln"] = math.nan

        feats["meta_t"] = float(t)
        feats["meta_ln1p_t"] = math.log1p(t)

        h0 = self.h0
        feats["meta_h0_n_h"] = float(h0.n_h)
        feats["meta_h0_nu_hat"] = h0.nu_hat
        feats["meta_h0_rho1_e"] = h0.rho1_e
        feats["meta_h0_rho1_abs_e"] = h0.rho1_abs_e
        feats["meta_h0_ar_r2"] = h0.ar_r2
        feats["meta_h0_seasonal_flag"] = 1.0 if h0.seasonal_lag is not None else 0.0
        feats["meta_h0_q99"] = h0.q["0.99"]
        feats["meta_h0_scale_ratio"] = h0.sigma_e_rob / h0.sigma_e

        # F2 (proposta V2): impressão digital estendida do regime H0 — constantes por série,
        # calculadas em fit_h0, custo zero por passo. São condicionadores (CE6 nulo como efeito
        # principal), e a família meta_h0 já é a mais usada do modelo (34,3% do |SHAP|).
        for key, value in h0.fingerprint.items():
            feats[f"meta_h0_{key}"] = value

        age_map = feats.get("bayes_age_map_h0100")
        age_cusum = feats.get("cusum_age_mean_pos_d050")
        if age_map is not None and age_cusum is not None and not (math.isnan(age_map) or math.isnan(age_cusum)):
            feats["meta_locator_diff"] = abs(age_map - age_cusum)
            feats["meta_locator_min"] = min(age_map, age_cusum)
        else:
            feats["meta_locator_diff"] = math.nan
            feats["meta_locator_min"] = math.nan

        # F1 (proposta V2): versões `_cal` padronizadas contra o nulo da PRÓPRIA série, medido sobre
        # o histórico em fit_h0. Aplicado por último, depois que todos os blocos já emitiram seus
        # valores crus. Ver a docstring de state/calibration.py para o porquê.
        apply_calibration(feats, self.h0.null_stats, t)

        return feats

    def _update_hedge(self, x: float) -> None:
        self._hedge_ewma = ewma_update(self._hedge_ewma, x - self.h0.mu0, self.cfg.state.hedge_ewma_lambda)
        evicted = self._hedge_ring.push(x)
        if evicted is None:
            self._hedge_sum += x
            self._hedge_sumsq += x * x
        else:
            self._hedge_sum += x - evicted
            self._hedge_sumsq += x * x - evicted * evicted

    def update(self, x: float) -> float:
        """UMA observação -> UM score em [0,1]."""

        feats = self.update_features(x)
        p = self.ensemble.predict_one(feats) if self.ensemble is not None else fallback_score(feats, self.cfg)
        score = apply_monotonicity(p, self._prev_score, self.cfg.postprocess.mode, self.cfg)
        self._prev_score = score
        return score


# @crunch/keep:on
# ============================== sbrt/model/base_rate.py ==============================
"""Curva de taxa-base empírica p_hat(t) (plano_acao_v1_para_v2.md §4, ação A2).

y_t = 1{tau<=t} tem uma taxa-base que cresce fortemente com t (~7.6% em t<=50 até ~39.7% em t>400,
plano_acao_v1_para_v2.md §1.1) -- uma amplitude de ~2 em log-odds, previsível a partir de t sozinho.
Por invariância C1 (plano_structural_break_realtime.md §1.2), um componente de score que depende só
de t desloca todas as séries vivas igualmente no mesmo passo e é NEUTRO para a TS-AUC -- mas domina
a logloss binária/AUC de linha usadas para treinar e parar o LightGBM. Esta curva vira `init_score`
(plano §8.3 revisado) para que o modelo aprenda só o resíduo: a discriminação transversal que a
métrica de fato mede.

Ajustada UMA VEZ sobre o dataset de treino completo (não por fold) -- simplificação documentada: a
curva agrega milhares de séries sem usar identidade de série nenhuma, então o vazamento marginal de
incluir ~20% de linhas de validação no ajuste é desprezível frente ao que ela corrige.
"""

import numpy as np


def fit_base_rate_curve(t: np.ndarray, y: np.ndarray, bin_width: int = 20, pseudo_count: float = 10.0) -> dict:
    """p_hat(t) por bins de largura `bin_width`, com suavização aditiva (pseudo_count em direção a
    0.5) para bins esparsos em t alto não colapsarem para 0/1. Retorna centros e taxas para
    interpolação linear em `predict_base_rate_logit`."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    max_t = int(np.ceil(t.max())) if len(t) else 1
    edges = np.arange(0, max_t + bin_width, bin_width, dtype=np.float64)
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, len(edges) - 2)

    centers, rates = [], []
    for b in range(len(edges) - 1):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        pos = float(y[mask].sum())
        rate = (pos + pseudo_count * 0.5) / (n + pseudo_count)
        centers.append(float((edges[b] + edges[b + 1]) / 2.0))
        rates.append(rate)

    return {"centers": centers, "rates": rates}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def predict_base_rate_logit(t: np.ndarray, curve: dict) -> np.ndarray:
    """logit(p_hat(t)) via interpolação linear entre os centros ajustados; constante fora do
    intervalo (mantém comportamento definido nas bordas)."""
    t = np.asarray(t, dtype=np.float64)
    centers = np.asarray(curve["centers"], dtype=np.float64)
    rates = np.asarray(curve["rates"], dtype=np.float64)
    rate_at_t = np.interp(t, centers, rates)
    return _logit(rate_at_t)


# @crunch/keep:on
# ============================== sbrt/model/predict.py ==============================
"""ModelEnsemble.predict_one (plano §8.4)."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np



@dataclass
class ModelEnsemble:
    boosters: list
    feature_order: tuple
    predict_num_threads: int = 1
    fold_evals: list = field(default_factory=list)  # diagnóstico (plano §9.1) — persistido em fold_evals.json
    base_rate_curve: dict | None = None  # plano_acao_v1_para_v2.md A2 — metadado de treino, NÃO usado
    # em predict_one (ver docstring): somar de volta é neutro para TS-AUC por invariância C1, mas
    # infla o score em cenários sintéticos fora da distribuição real. Fica salvo para diagnóstico
    # (ex.: reconstruir o resíduo de treino) e para quem quiser reabilitar a calibração absoluta.

    @classmethod
    def load(cls, path: str | Path) -> "ModelEnsemble":
        path = Path(path)
        boosters = joblib.load(path / "boosters.joblib")
        feature_order = tuple(json.loads((path / "feature_schema.json").read_text(encoding="utf-8")))
        meta = json.loads((path / "ensemble_meta.json").read_text(encoding="utf-8"))
        fold_evals_path = path / "fold_evals.json"
        fold_evals = json.loads(fold_evals_path.read_text(encoding="utf-8")) if fold_evals_path.exists() else []
        base_rate_path = path / "base_rate_curve.json"
        base_rate_curve = json.loads(base_rate_path.read_text(encoding="utf-8")) if base_rate_path.exists() else None
        return cls(
            boosters=boosters,
            feature_order=feature_order,
            predict_num_threads=meta["predict_num_threads"],
            fold_evals=fold_evals,
            base_rate_curve=base_rate_curve,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.boosters, path / "boosters.joblib")
        (path / "feature_schema.json").write_text(json.dumps(list(self.feature_order), indent=2), encoding="utf-8")
        (path / "ensemble_meta.json").write_text(
            json.dumps({"predict_num_threads": self.predict_num_threads}), encoding="utf-8"
        )
        (path / "fold_evals.json").write_text(json.dumps(self.fold_evals, indent=2), encoding="utf-8")
        if self.base_rate_curve is not None:
            (path / "base_rate_curve.json").write_text(json.dumps(self.base_rate_curve), encoding="utf-8")

    def predict_one(self, feats: dict) -> float:
        """Média dos folds; num_threads=1; ordem de colunas fixada pelo schema salvo. SEM tqdm —
        é caminho de inferência real (plano §8, regra tqdm).

        plano_acao_v1_para_v2.md A2/A5: os boosters são treinados com `init_score = logit(p_hat(t))`
        (model/train.py), então `predict()` (sem `raw_score`) já devolve `sigmoid(raw)` -- o resíduo
        transversal, SEM o offset de taxa-base (LightGBM nunca readiciona `init_score` para dados
        novos). Deliberadamente NÃO somamos o offset de volta aqui: por invariância C1 (plano técnico
        §1.2) isso é neutro para a TS-AUC oficial (desloca todas as séries vivas igualmente em cada
        t), mas somá-lo de volta infla o score em cenários fora da distribuição de treino (medido:
        piorou a suíte de robustez em T6/T9/T10/T12/T12b — decisão tomada com o usuário após ver o
        efeito). O score aqui é o resíduo puro; não é uma probabilidade calibrada absoluta."""
        x = to_array(feats, self.feature_order).reshape(1, -1)
        preds = [b.predict(x, num_threads=self.predict_num_threads)[0] for b in self.boosters]
        return float(np.mean(preds))


@dataclass
class RankModelEnsemble:
    """R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): ensemble treinado com objetivo de ranking por
    grupo t (lambdarank/rank_xendcg, model/train.py:train_rank) -- membro PARALELO do ensemble
    binário, não um substituto (nenhum precedente interno ainda, parecer §6-R3). `booster.predict()`
    para um objetivo de ranking devolve um score de relevância CRU (sem semântica de probabilidade,
    escala arbitrária, pode ser negativo) -- diferente do `ModelEnsemble` binário. Aplicamos uma
    sigmoide FIXA (não recalibrada) só para mapear em (0,1) e manter compatibilidade com o resto do
    pipeline (postprocess, formato de submissão); como TS-AUC/gates dependem só de ORDEM relativa
    (parecer §3.1), qualquer mapeamento monótono fixo preserva o desempenho de ranking exatamente."""

    boosters: list
    feature_order: tuple
    predict_num_threads: int = 1
    fold_evals: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "RankModelEnsemble":
        path = Path(path)
        boosters = joblib.load(path / "boosters.joblib")
        feature_order = tuple(json.loads((path / "feature_schema.json").read_text(encoding="utf-8")))
        meta = json.loads((path / "ensemble_meta.json").read_text(encoding="utf-8"))
        fold_evals_path = path / "fold_evals.json"
        fold_evals = json.loads(fold_evals_path.read_text(encoding="utf-8")) if fold_evals_path.exists() else []
        return cls(
            boosters=boosters,
            feature_order=feature_order,
            predict_num_threads=meta["predict_num_threads"],
            fold_evals=fold_evals,
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.boosters, path / "boosters.joblib")
        (path / "feature_schema.json").write_text(json.dumps(list(self.feature_order), indent=2), encoding="utf-8")
        (path / "ensemble_meta.json").write_text(
            json.dumps({"predict_num_threads": self.predict_num_threads}), encoding="utf-8"
        )
        (path / "fold_evals.json").write_text(json.dumps(self.fold_evals, indent=2), encoding="utf-8")

    def predict_one(self, feats: dict) -> float:
        x = to_array(feats, self.feature_order).reshape(1, -1)
        raw = [b.predict(x, num_threads=self.predict_num_threads)[0] for b in self.boosters]
        sigm = [1.0 / (1.0 + np.exp(-r)) for r in raw]
        return float(np.mean(sigm))


@dataclass
class CombinedModelEnsemble:
    """Combinador implantável dos dois braços do ensemble (binário-R1 + rank, R3): média simples
    dos dois `predict_one` em (0,1) -- a única combinação que um scorer causal em tempo real pode
    computar por passo, sem acesso à seção transversal de outras séries no mesmo t (diferente do
    "rank-average" via percentil OOF usado só para comparação offline, scripts/combine_oof.py)."""

    binary: ModelEnsemble
    rank: RankModelEnsemble

    def predict_one(self, feats: dict) -> float:
        return 0.5 * (self.binary.predict_one(feats) + self.rank.predict_one(feats))


# @crunch/keep:on
# ============================== sbrt/evaluation/splits.py ==============================
"""Divisão agrupada e estratificada por série (plano §9.4). GroupKFold por `id` é obrigatório:
linhas da mesma série são fortemente autocorrelacionadas — um split não agrupado infla o CV de forma
catastrófica (armadilha §13.6)."""

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def build_series_meta(rows: pd.DataFrame) -> pd.DataFrame:
    """Um registro por id: rótulo da série (teve quebra?) e terço de tau (bucket 0/1/2 do primeiro
    t com y=1 relativo ao T observado da série, ou -1 se não houver quebra)."""
    g = rows.groupby("id")
    has_break = g["y"].max().astype(int)

    def _tau_bucket(sub: pd.DataFrame) -> int:
        pos = sub.loc[sub["y"] == 1, "t"]
        if pos.empty:
            return -1
        tau = pos.min()
        t_max = sub["t"].max()
        frac = tau / max(t_max, 1)
        return min(int(frac * 3), 2)

    tau_bucket = g.apply(_tau_bucket, include_groups=False)
    meta = pd.DataFrame({"has_break": has_break, "tau_bucket": tau_bucket})
    return meta


def grouped_stratified_kfold(meta: pd.DataFrame, k: int, seed: int) -> Iterator[tuple]:
    """`meta` = linhas por passo (uma linha por (id,t)), como produzido por model/dataset.py.
    Agrupado por id; estratificado por (rótulo da série, terço de tau). Retorna, por fold, posições
    0-based (não ids) em `meta` para treino/validação — prontas para indexar a matriz X."""
    series = build_series_meta(meta)
    strata = series["has_break"].astype(str) + "_" + series["tau_bucket"].astype(str)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    series_ids = series.index.to_numpy()

    id_to_positions = meta.groupby("id").indices

    for train_pos, valid_pos in skf.split(series_ids, strata):
        train_ids = series_ids[train_pos]
        valid_ids = series_ids[valid_pos]
        train_rows = np.concatenate([id_to_positions[i] for i in train_ids])
        valid_rows = np.concatenate([id_to_positions[i] for i in valid_ids])
        yield train_rows, valid_rows


# @crunch/keep:on
# ============================== sbrt/evaluation/ts_auc.py ==============================
"""TS-AUC ponderada por passo (docs/PLANO_TECNICO.md §1: AUC_t agregada com peso n_pos(t)*n_neg(t))
— implementação vetorizada via rank médio intra-grupo (estatística de Mann-Whitney), equivalente a
`roc_auc_score` por grupo em laço Python (scripts/oof_ts_auc_by_bucket.py, scripts/local_ts_auc.py)
mas ordens de magnitude mais rápida — necessária onde isto roda centenas/milhares de vezes: o `feval`
de treino por rodada de boosting (model/train.py, R2) e o bootstrap pareado (scripts/compare_oof.py,
R0). Nunca usada como estimador de leaderboard (plano §9.0) — é critério interno de fold/diagnóstico
relativo."""

import numpy as np
import pandas as pd


def weighted_ts_auc(t: np.ndarray, y: np.ndarray, score: np.ndarray) -> float:
    if len(t) == 0:
        return float("nan")
    df = pd.DataFrame({"t": t, "y": y, "s": score})
    df["rank"] = df.groupby("t")["s"].rank(method="average")
    g = df.groupby("t")
    n = g["y"].size()
    n_pos = g["y"].sum()
    n_neg = n - n_pos
    r_pos = df.loc[df["y"] == 1].groupby("t")["rank"].sum().reindex(n.index, fill_value=0.0)

    valid = (n_pos > 0) & (n_neg > 0)
    if not valid.any():
        return float("nan")
    auc_t = (r_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / (n_pos[valid] * n_neg[valid])
    w = (n_pos[valid] * n_neg[valid]).astype(np.float64)
    tot = w.sum()
    return float((auc_t * w).sum() / tot) if tot > 0 else float("nan")


# @crunch/keep:on
# ============================== sbrt/model/weights.py ==============================
"""Pesos de linha pareado-consistentes com a TS-AUC (parecer de auditoria §3.10/§4.4, roadmap R1).

A TS-AUC se reescreve como a fração de pares (positivo, negativo) do MESMO passo t corretamente
ordenados, agregada sobre todos os t (parecer §3.1). Nesse pool de pares, cada linha positiva de t
participa de n_neg(t) pares e cada linha negativa de n_pos(t) pares — logo o surrogate pontual
pareado-consistente dá peso ∝ n_neg(t) aos positivos e ∝ n_pos(t) aos negativos, e não o mesmo peso
às duas classes como antes (w(t) = n_pos(t)*n_neg(t)/n_alive(t) para TODA linha de t, sem distinguir
classe). O esquema antigo acerta a massa agregada por passo (∝ n_pos*n_neg) mas erra a partição
intra-passo (1:1 em vez de n_neg:n_pos) — em t<=50 isso dilui o gradiente dos positivos por ~12x,
exatamente no bucket onde a AUC medida é mais fraca (parecer §3.10).

Contagens suavizadas por pseudo-contagem (t com poucos positivos não vira peso quase-infinito) e a
razão w_pos(t)/w_neg(t) capada em `max_ratio` (t muito pequeno tem n_pos raro, n_neg~5000 — sem cap
isso troca viés por variância de gradiente, parecer §4.4). Multiplicado pelo fator de thinning e
normalizado para média 1, como antes."""

import numpy as np
import pandas as pd


def compute_row_weights(
    rows: pd.DataFrame, cfg, pseudo_count: float = 5.0, max_ratio: float = 50.0
) -> np.ndarray:
    counts = rows.groupby("t")["y"].agg(n_pos="sum", n_alive="count")
    counts["n_neg"] = counts["n_alive"] - counts["n_pos"]
    n_pos_s = counts["n_pos"] + pseudo_count
    n_neg_s = counts["n_neg"] + pseudo_count

    # w_pos(t) ~ n_neg_s (número de pares que cada positivo participa), w_neg(t) ~ n_pos_s,
    # capando a razão entre as duas em max_ratio (equivalente a clipar n_neg_s/n_pos_s em
    # [1/max_ratio, max_ratio] e manter a proporcionalidade exata dentro do cap).
    counts["w_t_pos"] = np.minimum(n_neg_s, max_ratio * n_pos_s)
    counts["w_t_neg"] = np.minimum(n_pos_s, max_ratio * n_neg_s)

    w_t_pos_map = counts["w_t_pos"].to_dict()
    w_t_neg_map = counts["w_t_neg"].to_dict()
    t_pos = rows["t"].map(w_t_pos_map).to_numpy(dtype=np.float64)
    t_neg = rows["t"].map(w_t_neg_map).to_numpy(dtype=np.float64)
    is_pos = rows["y"].to_numpy(dtype=bool)
    base_w = np.where(is_pos, t_pos, t_neg)

    w = base_w * rows["thin_weight"].to_numpy(dtype=np.float64)

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w


# @crunch/keep:on
# ============================== sbrt/model/train.py ==============================
"""GroupKFold(5, groups=id) + 1 LightGBM por fold (plano §8.3). Predição final = média das
probabilidades dos 5 modelos (model/predict.py).

plano_acao_v1_para_v2.md A2: `y_t = 1{tau<=t}` tem uma taxa-base fortemente crescente com t (~7.6%
a ~39.7%), neutra para TS-AUC por invariância C1 mas dominante para logloss/AUC de linha. Por isso:
(1) a curva de taxa-base vira `init_score`, deixando o LightGBM aprender só o resíduo transversal;
(2) o early stopping usa `binary_logloss` (agora medindo só o resíduo, já que init_score desloca a
métrica), não mais `auc` (que saturava cedo dominada pela taxa-base — 89-110 árvores medidas contra
400-800 esperadas, plano §8.3).

R2 (docs/PARECER_AUDITORIA_ONYX.md §6-R2): mesmo com init_score corrigindo o offset, a parada e a
seleção de hiperparâmetros continuavam julgadas por `binary_logloss` pontual -- uma régua diferente
da métrica oficial (TS-AUC = fração de pares concordantes por passo, parecer §3.1). O juiz agora é
um `feval` custom: AUC ponderada por passo t sobre o PRÓPRIO fold de validação (`ts_auc_by_t`,
`first_metric_only=True` na parada); `binary_logloss` continua computado à mão dentro do mesmo feval
só para diagnóstico/plot (training_curves), sem influenciar a parada. Isto é critério interno de
fold, não estimador de leaderboard -- compatível com a §9.0 (nunca substitui a submissão oficial nem
o comparador pareado de scripts/compare_oof.py)."""

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb


_NON_FEATURE_COLS = {"id", "t", "y", "thin_weight"}


def _make_fold_feval(
    t_valid: np.ndarray, max_rows: int | None, seed: int, raw_to_prob: bool = False, stopping_metric: str = "logloss"
):
    """Fecha sobre os valores de `t` do fold de validação (posicionalmente alinhados com as linhas
    do `lgb.Dataset` de validação, que preserva a ordem original -- LightGBM não embaralha dados
    internamente). `max_rows`: subamostra FIXA (sorteada uma vez, não a cada rodada -- resortear a
    cada rodada injetaria ruído extra no critério de parada) para manter o custo por rodada baixo em
    folds grandes (parecer §6-R2). `ts_auc_by_t` (rank-based, invariante a qualquer transformação
    monótona de `preds`) funciona idêntico para objetivo binário ou de ranking (R3, train_rank);
    `raw_to_prob=True` só afeta o diagnóstico `binary_logloss_diag`, aplicando a MESMA sigmoide fixa
    usada por `RankModelEnsemble.predict_one` antes de computar a logloss -- sem isso, `preds` de um
    objetivo de ranking (escala arbitrária, pode ser negativo) tornaria essa logloss sem sentido.

    `stopping_metric` controla qual das duas entra PRIMEIRO na lista retornada -- é essa ordem que
    `first_metric_only=True` usa para decidir a parada (cfg.lightgbm.early_stopping_metric). AMBAS
    são sempre computadas e registradas (fold_evals, training_curves); só a ordem muda. Ver a nota
    empírica em config.py:LightGBMConfig.early_stopping_metric -- "ts_auc_by_t" sozinho regrediu a
    TS-AUC OOF real por ruído de seleção (n efetivo ~10^4 séries, não o número de linhas)."""
    n = len(t_valid)
    if max_rows is not None and n > max_rows:
        rng = np.random.default_rng(seed)
        sub_idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    else:
        sub_idx = None

    def _feval(preds: np.ndarray, dataset: "lgb.Dataset"):
        y_full = dataset.get_label()
        w_full = dataset.get_weight()
        if sub_idx is not None:
            preds_s, y_s, t_s = preds[sub_idx], y_full[sub_idx], t_valid[sub_idx]
            w_s = w_full[sub_idx] if w_full is not None else None
        else:
            preds_s, y_s, t_s = preds, y_full, t_valid
            w_s = w_full

        auc = weighted_ts_auc(t_s, y_s, preds_s)
        if not np.isfinite(auc):
            auc = 0.5  # nunca deixar a parada ver NaN (subamostra sem par completo em algum t)

        p_for_diag = 1.0 / (1.0 + np.exp(-preds_s)) if raw_to_prob else preds_s
        p = np.clip(p_for_diag, 1e-7, 1.0 - 1e-7)
        terms = y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p)
        logloss = float(-np.average(terms, weights=w_s))

        entries = [
            ("ts_auc_by_t", float(auc), True),
            ("binary_logloss_diag", logloss, False),
        ]
        return entries if stopping_metric == "ts_auc_by_t" else list(reversed(entries))

    return _feval


def train_ensemble(rows: pd.DataFrame, weights: np.ndarray, cfg, progress: bool = True) -> tuple:
    """tqdm sobre os 5 folds; dentro de cada fold, LightGBM usa seu próprio log verbose (não
    duplicar barra de progresso, plano §8 regra tqdm). Retorna (ModelEnsemble, oof_pred) — oof_pred
    é a probabilidade calibrada out-of-fold por linha, alinhada a `rows` (diagnóstico A4, não faz
    parte do artefato salvo)."""
    feature_cols = sorted(c for c in rows.columns if c not in _NON_FEATURE_COLS)
    X = rows[feature_cols].to_numpy(dtype=np.float32)  # plano §8.1: float32 no dataset de treino
    y = rows["y"].to_numpy(dtype=np.int32)
    t_values = rows["t"].to_numpy(dtype=np.float64)

    base_rate_curve = fit_base_rate_curve(t_values, y.astype(np.float64))
    init_score_full = predict_base_rate_logit(t_values, base_rate_curve)

    lgb_cfg = cfg.lightgbm
    params = dict(
        objective="binary",
        metric="None",  # R2: métricas internas desligadas -- o feval custom cobre AUC-por-t
        # (parada, first_metric_only) e binary_logloss (diagnóstico), ambos vistos por valid_sets.
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        max_depth=lgb_cfg.max_depth,
        min_data_in_leaf=lgb_cfg.min_data_in_leaf,
        feature_fraction=lgb_cfg.feature_fraction,
        bagging_fraction=lgb_cfg.bagging_fraction,
        bagging_freq=lgb_cfg.bagging_freq,
        lambda_l2=lgb_cfg.lambda_l2,
        max_bin=lgb_cfg.max_bin,
        deterministic=lgb_cfg.deterministic,
        force_row_wise=lgb_cfg.force_row_wise,
        num_threads=lgb_cfg.train_num_threads,
        seed=cfg.seed,
        verbose=-1,
    )

    boosters = []
    fold_evals = []
    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    folds = list(grouped_stratified_kfold(rows, lgb_cfg.n_folds, cfg.seed))
    fold_iter = tqdm(folds, desc="treinando folds") if progress else folds

    for train_idx, valid_idx in fold_iter:
        dtrain = lgb.Dataset(
            X[train_idx], label=y[train_idx], weight=weights[train_idx], init_score=init_score_full[train_idx]
        )
        dvalid = lgb.Dataset(
            X[valid_idx],
            label=y[valid_idx],
            weight=weights[valid_idx],
            init_score=init_score_full[valid_idx],
            reference=dtrain,
        )
        feval = _make_fold_feval(
            t_values[valid_idx], lgb_cfg.feval_max_valid_rows, cfg.seed,
            stopping_metric=lgb_cfg.early_stopping_metric,
        )
        evals_result: dict = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=lgb_cfg.n_estimators_cap,
            valid_sets=[dvalid],
            feval=feval,
            callbacks=[
                lgb.early_stopping(lgb_cfg.early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        boosters.append(booster)
        fold_evals.append(evals_result)

        # raw_score=True nunca inclui init_score (é um construto do Dataset, não do modelo salvo) —
        # por isso somamos o mesmo offset usado no treino antes do sigmoid (plano A2, model/predict.py).
        raw_valid = booster.predict(X[valid_idx], raw_score=True)
        full_logit_valid = raw_valid + init_score_full[valid_idx]
        oof_pred[valid_idx] = 1.0 / (1.0 + np.exp(-full_logit_valid))

    ensemble = ModelEnsemble(
        boosters=boosters,
        feature_order=tuple(feature_cols),
        predict_num_threads=lgb_cfg.predict_num_threads,
        fold_evals=fold_evals,
        base_rate_curve=base_rate_curve,
    )
    return ensemble, oof_pred


def train_rank(rows: pd.DataFrame, cfg, progress: bool = True) -> tuple:
    """R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): mesmo split por série (`grouped_stratified_kfold`)
    do modo binário (`train`), mas objetivo de RANKING por grupo t -- otimiza diretamente a
    concordância de pares intra-t, a forma fechada exata da TS-AUC (parecer §3.1). Membro PARALELO
    do ensemble binário, não substituto (nenhum precedente interno, parecer §6-R3) -- retorna um
    `RankModelEnsemble` (model/predict.py), combinável com o binário via `CombinedModelEnsemble` ou
    comparável via `scripts/combine_oof.py` (rank-average offline).

    SEM `init_score`: dentro de um grupo (linhas do mesmo t), um deslocamento constante não muda a
    ordem relativa dos itens -- matematicamente neutro para uma perda de ranking, ao contrário do
    modo binário (onde a taxa-base domina a logloss pontual, plano_acao A2). Peso de linha = só
    `thin_weight` normalizado: o desbalanceamento de classe intra-t já é tratado estruturalmente pela
    perda pareada (cada par (pos,neg) do grupo contribui um termo de gradiente) -- aplicar também os
    pesos classe-balanceados de R1 (model/weights.py) duplicaria esse efeito."""
    feature_cols = sorted(c for c in rows.columns if c not in _NON_FEATURE_COLS)
    X_full = rows[feature_cols].to_numpy(dtype=np.float32)
    y_full = rows["y"].to_numpy(dtype=np.int32)
    t_full = rows["t"].to_numpy(dtype=np.int64)

    thin_w = rows["thin_weight"].to_numpy(dtype=np.float64)
    thin_w = thin_w / thin_w.mean()

    lgb_cfg = cfg.lightgbm
    rank_cfg = cfg.rank
    base_params = dict(
        objective=rank_cfg.objective,
        label_gain=list(rank_cfg.label_gain),
        metric="None",
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        max_depth=lgb_cfg.max_depth,
        min_data_in_leaf=lgb_cfg.min_data_in_leaf,
        feature_fraction=lgb_cfg.feature_fraction,
        bagging_fraction=lgb_cfg.bagging_fraction,
        bagging_freq=lgb_cfg.bagging_freq,
        lambda_l2=lgb_cfg.lambda_l2,
        max_bin=lgb_cfg.max_bin,
        deterministic=lgb_cfg.deterministic,
        force_row_wise=lgb_cfg.force_row_wise,
        num_threads=lgb_cfg.train_num_threads,
        seed=cfg.seed,
        verbose=-1,
    )

    boosters = []
    fold_evals = []
    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    folds = list(grouped_stratified_kfold(rows, lgb_cfg.n_folds, cfg.seed))
    fold_iter = tqdm(folds, desc="treinando folds (rank)") if progress else folds

    for train_idx, valid_idx in fold_iter:
        # contiguidade por grupo: lambdarank exige que as linhas do mesmo t estejam adjacentes,
        # com `group` = contagens por t NESSA ordem (armadilha do objetivo de ranking do LightGBM).
        train_sorted = train_idx[np.argsort(t_full[train_idx], kind="stable")]
        valid_sorted = valid_idx[np.argsort(t_full[valid_idx], kind="stable")]

        _, train_group = np.unique(t_full[train_sorted], return_counts=True)
        _, valid_group = np.unique(t_full[valid_sorted], return_counts=True)

        # truncation_level idealmente cobriria o maior grupo por inteiro (o default do LightGBM, 30,
        # daria gradiente só ao topo de cada grupo, péssimo para uma AUC que depende de TODOS os
        # pares, parecer §6-R3) -- mas t<=100 mantém ~10000 séries vivas (thinning só começa depois,
        # configs/default.yaml:thinning), então o maior grupo de um fold chega a ~8000 linhas: sem
        # cap, o custo por grupo (~group_size*truncation_level) trava o treino (medido: >4h sem
        # terminar). `truncation_level_cap` (rank.truncation_level_cap) limita isso a um valor
        # tratável -- grupos maiores que o cap ficam com gradiente pleno só no topo, risco aceito
        # por tratabilidade computacional.
        truncation_level = min(int(train_group.max()), rank_cfg.truncation_level_cap)
        params = dict(base_params, lambdarank_truncation_level=truncation_level)

        dtrain = lgb.Dataset(
            X_full[train_sorted], label=y_full[train_sorted], weight=thin_w[train_sorted], group=train_group
        )
        dvalid = lgb.Dataset(
            X_full[valid_sorted],
            label=y_full[valid_sorted],
            weight=thin_w[valid_sorted],
            group=valid_group,
            reference=dtrain,
        )
        feval = _make_fold_feval(
            t_full[valid_sorted], lgb_cfg.feval_max_valid_rows, cfg.seed, raw_to_prob=True,
            stopping_metric=lgb_cfg.early_stopping_metric,
        )
        evals_result: dict = {}
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=lgb_cfg.n_estimators_cap,
            valid_sets=[dvalid],
            feval=feval,
            callbacks=[
                lgb.early_stopping(lgb_cfg.early_stopping_rounds, first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        boosters.append(booster)
        fold_evals.append(evals_result)

        raw_valid = booster.predict(X_full[valid_sorted], raw_score=True)
        oof_pred[valid_sorted] = 1.0 / (1.0 + np.exp(-raw_valid))

    ensemble = RankModelEnsemble(
        boosters=boosters,
        feature_order=tuple(feature_cols),
        predict_num_threads=lgb_cfg.predict_num_threads,
        fold_evals=fold_evals,
    )
    return ensemble, oof_pred


# @crunch/keep:on
# ============================== sbrt/model/dataset.py ==============================
"""Motor único -> linhas de treino (plano §8.1). Para cada série: fit_h0 + o MESMO `StreamScorer`
da submissão (ensemble=None), passo a passo, coletando (features, y_t=1{tau<=t}, peso). NUNCA
vetorizar o laço *dentro* de uma série (substituir o loop incremental por uma reconstrução em lote é
exatamente a armadilha §13.2 — "backtest vetorizado != execução causal", motor único,
docs/PLANO_REPOSITORIO.md §1).

`n_jobs` paraleliza o laço *externo*, entre séries — cada série é 100% independente (nenhum estado
compartilhado) e continua rodando pelo idêntico laço serial passo a passo dentro do seu próprio
processo; não é a vetorização proibida acima, é só a mesma computação rodando em paralelo em vez de
em sequência. Existe porque o motor de estado é Python puro (~1ms/passo medido) e o dataset real tem
~5M passos — sem isso, construir o dataset de treino sozinho leva ~85 min (plano §11 assumia 25us/
passo vetorizado/numba; isto ainda não foi feito, ver §11.4)."""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm



@dataclass
class SeriesRecord:
    dataset_id: int
    x_hist: np.ndarray
    x_online: np.ndarray
    tau_index: Optional[int]  # 0-based índice em x_online onde a quebra ocorre, ou None


def _thinning_keep_and_weight(t: int, cfg) -> tuple:
    """plano §8.1: mantém todos os passos t<=100; 101-400 a cada 2 (peso x2); >400 a cada 4 (peso x4)."""
    th = cfg.thinning
    if t <= th.full_until:
        return True, 1.0
    if t <= 400:
        return (t % th.step_101_400 == 0), float(th.step_101_400)
    return (t % th.step_401_plus == 0), float(th.step_401_plus)


def _build_rows_for_series(rec: SeriesRecord, cfg) -> pd.DataFrame:
    """Uma série inteira, do jeito serial/causal de sempre — é isto que roda em cada worker quando
    `n_jobs != 1`. Definida em nível de módulo (não uma closure) para ser picklable pelo joblib.

    Devolve um DataFrame pequeno (uma série só, no máximo ~250 linhas após o thinning) já em
    float32, não uma lista de dicts crua — ver nota em `build_training_rows` sobre por quê."""
    h0 = fit_h0(rec.x_hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)

    rows = []
    order = None
    for i, x in enumerate(rec.x_online):
        t = i + 1
        feats = scorer.update_features(float(x))
        if order is None:
            order = build_feature_order(feats)

        keep, thin_w = _thinning_keep_and_weight(t, cfg)
        if not keep:
            continue

        y = 1 if (rec.tau_index is not None and i >= rec.tau_index) else 0
        row = {k: feats.get(k, np.nan) for k in order}
        row["id"] = rec.dataset_id
        row["t"] = t
        row["y"] = y
        row["thin_weight"] = thin_w
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        feature_cols = list(order)  # `order` é tuple — df[tuple] é lida como 1 chave multi-index, não uma lista de colunas
        df[feature_cols] = df[feature_cols].astype(np.float32)
        df["thin_weight"] = df["thin_weight"].astype(np.float32)
        df["id"] = df["id"].astype(np.int32)
        df["t"] = df["t"].astype(np.int32)
        df["y"] = df["y"].astype(np.int8)
    return df


def build_training_rows(
    train_series: Iterable[SeriesRecord], cfg, progress: bool = True, n_jobs: int = 1
) -> pd.DataFrame:
    """Laço externo (por série) envolvido em tqdm quando progress=True — é laço de desenvolvimento
    local, não caminho de submissão (plano §8, regra tqdm). `n_jobs=1` (default) = serial, idêntico
    ao comportamento original; `n_jobs=-1`/N>1 = paralelo entre séries via joblib (ver docstring do
    módulo) — resultado é o mesmo DataFrame, só mais rápido de construir.

    IMPORTANTE (bug real encontrado com o dataset completo, ~2.5M linhas): construir uma lista
    Python de milhões de dicts e só then chamar `pd.DataFrame(lista_gigante)` uma única vez no
    final é um anti-padrão conhecido do pandas — cada dict tem overhead de objeto Python real, e a
    lista intermediária de milhões de dicts consome ordens de magnitude mais memória que o
    DataFrame final (float32, ~78 features x 2.5M linhas ~= 750MB, plano §8.1). Isso causou um
    `numpy._core._exceptions._ArrayMemoryError` com >8GB consumidos na conversão final. Correção:
    cada série vira seu próprio DataFrame pequeno (já em float32) dentro de `_build_rows_for_series`
    logo depois de coletar sua própria lista curta de dicts (no máximo ~250 linhas após thinning);
    aqui só concatenamos os ~10.000 DataFrames pequenos, uma vez, no final."""
    if n_jobs == 1:
        iterator = tqdm(train_series, desc="construindo dataset de treino") if progress else train_series
        dfs = [_build_rows_for_series(rec, cfg) for rec in iterator]
    else:
        train_series = list(train_series)
        jobs = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_build_rows_for_series)(rec, cfg) for rec in train_series
        )
        iterator = (
            tqdm(jobs, total=len(train_series), desc=f"construindo dataset de treino (n_jobs={n_jobs})")
            if progress
            else jobs
        )
        dfs = list(iterator)

    dfs = [df for df in dfs if not df.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# @crunch/keep:on
# ============================== sbrt/adapter/platform.py ==============================
"""Shim para o callback oficial da plataforma (plano §15.1 P0).

Contrato CONFIRMADO via `quickstarter_notebook.ipynb` (célula `def train`/`def infer`) — não é mais
best-guess: `train(datasets, model_directory_path)` recebe uma lista de
`(dataset_id, x_historical, x_online, tau_index)`; `infer(datasets, model_directory_path)` é um
generator que primeiro dá um `yield` vazio (sinaliza prontidão ao runner) e depois, para cada
`(x_historical, x_online)`, itera `x_online` emitindo exatamente um `float` por ponto. Casa
exatamente com `crunch.container.GeneratorWrapper` (`ERROR_FIRST_YIELD_MUST_BE_NONE`).

`train()` em modo `fallback` (padrão em configs/default.yaml) não precisa de dado de treino — o
score é 100% determinístico a partir do histórico de cada série (plano §8.5) — mas ainda grava um
artefato placeholder para que `infer()` tenha algo a carregar, no mesmo espírito do baseline oficial.
Em modo `supervised`, treina o pipeline completo (Frente H) e persiste o `ModelEnsemble`.
"""

import os
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np


_MODEL_FILE = "model.joblib"


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    cfg = load_config()
    os.makedirs(model_directory_path, exist_ok=True)

    if cfg.model.mode == "supervised":

        records = [
            SeriesRecord(
                dataset_id=dataset_id,
                x_hist=np.asarray(x_hist, dtype=np.float64),
                x_online=np.asarray(x_online, dtype=np.float64),
                tau_index=tau_index,
            )
            for dataset_id, x_hist, x_online, tau_index in datasets
        ]
        rows = build_training_rows(records, cfg, n_jobs=cfg.model.dataset_n_jobs)
        weights = compute_row_weights(rows, cfg)
        ensemble, _oof_pred = train_ensemble(rows, weights, cfg)
        ensemble.save(model_directory_path)
        joblib.dump({"mode": "supervised"}, os.path.join(model_directory_path, _MODEL_FILE))
    else:
        joblib.dump({"mode": "fallback"}, os.path.join(model_directory_path, _MODEL_FILE))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    cfg = load_config()
    model_path = os.path.join(model_directory_path, _MODEL_FILE)
    payload = joblib.load(model_path) if os.path.exists(model_path) else {"mode": "fallback"}

    ensemble = None
    if payload.get("mode") == "supervised":

        ensemble = ModelEnsemble.load(model_directory_path)

    yield  # sinaliza prontidão ao runner (GeneratorWrapper.ERROR_FIRST_YIELD_MUST_BE_NONE)

    for x_historical, x_online in datasets:
        hist = np.asarray(x_historical, dtype=np.float64)
        h0 = fit_h0(hist, cfg)
        scorer = StreamScorer(h0, default_blocks(), ensemble, cfg)
        for point in x_online:
            yield float(scorer.update(float(point)))


# @crunch/keep:on
# infer() roda em N processos (fork); defina 1 se o ambiente usar spawn e algo falhar.
INFER_PARALLELISM = 4
# INFER_PARALLELISM = 1


#crunch_tools.test(
#    # force_first_train=False,
#    # no_determinism_check=True,
#)


import pandas as pd
from sklearn.metrics import roc_auc_score

#y_test = pd.read_parquet("data/y_test.reduced.parquet")
#prediction = pd.read_parquet("prediction/prediction.parquet")
#merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)
#merged["time_online"] = merged.groupby("id").cumcount()
#wsum = tot = 0.0
#for _, g in merged.groupby("time_online"):
#    lab = g["target"].values; n_pos = int(lab.sum()); n_neg = int((1 - lab).sum())
#    if n_pos == 0 or n_neg == 0:
#        continue
#    wsum += n_pos * n_neg * roc_auc_score(lab, g["prediction"].values); tot += n_pos * n_neg
#print(f"Local TS-AUC: {wsum / tot if tot else 0.5:.4f}")
