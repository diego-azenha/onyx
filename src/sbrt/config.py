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
    features: FeaturesConfig
    lightgbm: LightGBMConfig
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
        features=FeaturesConfig(**raw["features"]),
        lightgbm=LightGBMConfig(**raw["lightgbm"]),
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
