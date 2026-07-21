"""Typed loader for configs/*.yaml — the single source of truth for every hyperparameter (plan §4)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


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
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

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
