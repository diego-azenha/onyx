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
    null_clip_match: bool = False  # A6: aplicar `clip_e` também ao histórico usado pelo replay de
    # calibração (state/calibration.py), para que o nulo de cada coluna `_cal` seja estimado na MESMA
    # escala em que a produção calcula a estatística. Ver a nota medida em configs/default.yaml.


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
    use_raw: bool = False  # A6: ranquear `e_raw` (não o `e` clipado) contra o histórico, que também
    # não é clipado. Ver a nota medida em configs/default.yaml e no topo de state/conformal.py.


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
    """F2: descritores estendidos do regime H0 (state/fingerprint.py).

    Os campos `precursor_*` servem só a `compute_precursors`, que é o gate F0.d e ainda NÃO alimenta
    features de produção — ver a docstring daquela função."""
    hill_frac: float
    acf_max_lag: int
    hurst_scales: tuple
    volvol_window: int
    precursor_tail_frac: float
    precursor_window: int


@dataclass(frozen=True)
class TrajectoryConfig:
    """F4+F9 (docs/BACKLOG_TSAUC.md): trajetória do estatístico (state/trajectory.py).

    `track` mapeia feature rastreada -> alias curto usado no nome da saída. Guardado como tupla
    ordenada de pares para que a iteração seja determinística no caminho de inferência
    (docs/NOTAS_AGENTES.md §1)."""
    ewma_lambda: float
    threshold: float
    track: tuple

    def __post_init__(self):
        raw = self.track
        items = sorted(raw.items()) if isinstance(raw, dict) else sorted(tuple(p) for p in raw)
        aliases = [a for _, a in items]
        if len(set(aliases)) != len(aliases):
            raise ValueError(f"trajectory.track com alias repetido: {aliases}")
        object.__setattr__(self, "track", tuple(items))


@dataclass(frozen=True)
class MismatchConfig:
    """F2 (docs/BACKLOG_TSAUC.md): brancura multi-lag do filtro congelado (state/mismatch.py)."""
    windows: tuple
    max_lag: int
    arch_windows: tuple
    arch_max_lag: int
    cusum_delta: float


@dataclass(frozen=True)
class SpectralConfig:
    """Eixo novo (docs/BACKLOG_TSAUC.md): forma do espectro de `e` (state/spectral.py)."""
    n_bins: int
    decay: float        # fator de esquecimento da DFT de tempo curto (janela ~1/(1-decay))
    alpha_fast: float   # taxa da EWMA de potência (média de Welch) — convenção do projeto:
    alpha_slow: float   # alpha é a TAXA, janela efetiva ~1/alpha (igual a multiscale.ewma_lambda)
    low_bins: int


@dataclass(frozen=True)
class OrdinalConfig:
    """Eixo novo (docs/BACKLOG_TSAUC.md): padrões ordinais de Bandt-Pompe (state/ordinal.py)."""
    m3_windows: tuple
    m4_windows: tuple
    min_counts_m3: int
    min_counts_m4: int


@dataclass(frozen=True)
class MultiRepConfig:
    """Eixo novo (docs/BACKLOG_TSAUC.md): ponte tipo-integral sobre três representações
    (state/multirep.py)."""
    windows: tuple
    min_n: int


@dataclass(frozen=True)
class CalibrationConfig:
    """F1: calibração de nulo por série (state/calibration.py). `shrink_pseudo` é a pseudo-contagem
    de encolhimento do desvio empírico para o teórico i.i.d. — necessária porque uma janela w sobre
    um histórico n_h só tem ~n_h/w janelas independentes.

    `recursive_features` (F1.a/F1.b) mapeia estatística RECURSIVA -> como o nulo dela escala em t.
    Essas têm o nulo medido por réplicas com reinício sobre o histórico, não por passada contínua.
    Manter isto em YAML é o que permite abrir a cobertura de F1.b como diff de configuração, um
    sub-braço por vez — e é o que impede o erro do V5 (empacotar mudanças e não conseguir atribuir o
    efeito a nenhuma delas).

    Leis aceitas: `none` (recursão refletida -> nulo estacionário: CUSUMs, martingale com reset) e
    `cumsum` (acumulador sem reset -> mu ∝ t, dp ∝ sqrt(t): os log-martingales conformais)."""
    enabled: bool
    shrink_pseudo: float
    transient_restart_every: int
    transient_smooth_w: int
    transient_max_reps: int
    recursive_features: tuple  # ((nome, kind), ...) ordenado — tupla, não dict, para iteração determinística

    def __post_init__(self):
        raw = self.recursive_features
        items = sorted(raw.items()) if isinstance(raw, dict) else sorted(tuple(p) for p in raw)
        bad = [n for n, k in items if k not in ("none", "cumsum")]
        if bad:
            raise ValueError(f"calibration.recursive_features com lei de escala desconhecida: {bad}")
        object.__setattr__(self, "recursive_features", tuple(items))


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
    bag_seeds: tuple = ()  # sementes a treinar e FUNDIR em `adapter/platform.py:train()`. Vazio =
    # uma so. MEDIDO 2026-07-22: bagging de 4 sementes vale +0,0040 de TS-AUC e satura em ~+0,0048
    # (K=7); a fusao dos boosters (model/fuse.py) mantem o custo de inferencia praticamente igual.
    # Cada semente multiplica o TEMPO DE TREINO na nuvem -- o dataset e construido uma vez so.
    boost_seed: int | None = None  # semente do sorteio interno do LightGBM (bagging/feature_fraction).
    # None = usa `seed` global. Só existe para calibrar o NULO da regra de decisão de R0: trocá-la
    # perturba o booster sem mexer nos folds, o que dá a variância de retreino que o bootstrap
    # pareado por série não enxerga. Ver src/sbrt/model/train.py.
    feval_max_valid_rows: int | None = None  # R2 (parecer §6-R2): subamostra determinística do fold
    # de validação usada pelo feval de AUC-por-passo a cada rodada de boosting; None = fold inteiro.
    pairwise_alpha: float = 0.0  # B4 (BRAINSTORM_RUPTURA_V2.md §1.5): peso do termo pairwise intra-t no
    # objetivo custom (grad=(1−α)·logloss+α·pairwise). 0 = objetivo binário normal.
    pairwise_pairs: int = 1      # B4: nº de negativos amostrados por positivo (t-casados).
    monotone_llr: bool = False  # B5 (BRAINSTORM_RUPTURA_V2.md §2.5): monotone_constraints=+1 nos
    # acumuladores de evidência (cusum_*_pos, cusum_var_up, conformal_logm_abs_reset, bayes_lo*) --
    # P(quebra≤t) deve ser não-decrescente em cada LLR. Regularização estrutural contra interações
    # espúrias com efeitos-fixos quando n_eff~10^4.
    feature_contri_meta: float = 1.0  # B6 (§2.6): multiplicador de ganho de split das colunas meta_h0_*
    # (0,5-0,7 empurra a árvore a usá-las como condicionador, não intercepto por série). 1.0 = desligado.
    # C1(ii) (CAMPANHA_POLIMENTO.md frente C): o B6 varreu o VALOR do contri, nunca o CONJUNTO de
    # colunas penalizadas. Estes dois campos estendem a penalização a outros rastreadores de `t` que o
    # xs-SHAP expôs (`conformal_logm_abs`, `mmd_joint_slow_cal`) com multiplicador PRÓPRIO -- eles não
    # são metadados de série como as `meta_h0_*`, então compartilhar o valor do B6 seria um chute.
    #
    # Casam por NOME EXATO, não por prefixo -- deliberadamente. O prefixo `conformal_logm_abs`
    # capturaria também `conformal_logm_abs_reset`, que está na lista `_MONO` de restrições
    # monotônicas (model/train.py), ou seja: o repo o trata como ACUMULADOR DE EVIDÊNCIA, não como
    # rastreador de `t`. Penalizá-lo junto conflataria o braço com uma mudança de sinal.
    # Default = tupla vazia + 1.0 => no-op exato.
    feature_contri_extra_cols: tuple[str, ...] = ()
    feature_contri_extra: float = 1.0
    early_stopping_metric: str = "logloss"  # "logloss" ou "ts_auc_by_t" -- qual das duas métricas do
    # feval (model/train.py:_make_fold_feval) governa a parada via first_metric_only. MEDIDO
    # (retreino real, 2026-07-20): "ts_auc_by_t" sozinho treina 100-236 rodadas (vs. 61-89 com
    # logloss) perseguindo o argmax de uma métrica rank-based cujo ruído entre rodadas é dominado
    # pelo n efetivo de ~10^4 séries (não pelo número de linhas) -- isso produziu uma regressão real
    # e estatisticamente significativa na TS-AUC OOF completa (Delta -0.0099, IC exclui 0) mesmo
    # usando o fold de validação inteiro no feval (sem subamostra). "logloss" (default) reproduz o
    # comportamento original, validado; "ts_auc_by_t" fica disponível para experimentação futura com
    # estabilização adicional (ex.: min_delta, suavização), não para uso direto.
    base_rate_weighted: bool = False  # A3b (CAMPANHA_POLIMENTO.md, frente A): ajustar a curva de
    # taxa-base (o `init_score`) sob os MESMOS pesos de linha usados no treino.
    #
    # A curva existe para tirar do modelo o componente puramente-f(t) do alvo (model/base_rate.py).
    # Ela é ajustada em `p(t)` NÃO ponderado -- mas desde o R1 (model/weights.py) o treino usa pesos
    # pareado-consistentes `w_pos(t) ∝ n_neg(t)`, `w_neg(t) ∝ n_pos(t)`, que EQUALIZAM as classes
    # dentro de cada passo. MEDIDO (2026-07-25, sobre train_rows_bocpd): a taxa de positivos
    # PONDERADA é 0,4947 em t=10 e 0,4996 em t=400 -- praticamente 0,5 em todo t>=10 -- enquanto o
    # `init_score` injetado vale logit(0,0226) = -3,83 em t=10 e -0,79 em t=400. O descasamento chega
    # a +3,81 em log-odds (média +1,22).
    #
    # Ou seja: o R1 invalidou a premissa do A2 e a curva deixou de remover um offset para passar a
    # INJETAR um. É metricamente neutro no limite (invariância C1: é função só de t), mas o modelo
    # gasta árvores desfazendo-o, e a logloss que governa a parada antecipada passa a ser dominada
    # por essa correção de f(t) em vez do resíduo transversal que a métrica cobra -- o que é
    # consistente com a parada disparar em 79-117 árvores.
    #
    # True = ajusta a curva com `weights`, o que a leva a ~logit(0,5)=0 e faz o `init_score` sumir.
    linear_tree: bool = False  # D1 (CAMPANHA_POLIMENTO.md): modelo LINEAR por folha em vez de
    # constante. O E0c mediu que o problema é essencialmente aditivo e suave (um toco `max_depth=1`
    # recupera 99,4% do incumbente na grade de treino); folhas constantes aproximam curvas suaves com
    # escadinhas, folhas lineares ajustam a curva. Custo: o LightGBM ignora `linear_tree` em `dart` e
    # exige `num_leaves` moderado para não sobreajustar a regressão por folha.
    enable_bundle: bool = True  # D5 (CAMPANHA_POLIMENTO.md): EFB do LightGBM. Com dezenas de
    # colunas NaN-de-warmup o agrupamento exclusivo pode juntar features que nao sao mutuamente
    # exclusivas de fato -- `false` desliga e testa isso. Default `true` = comportamento LightGBM.
    dart: bool = False  # D3: `boosting=dart` (dropout entre árvores). Célula do sweep C4-mini.


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
    ema_up_alpha: float = 0.7    # A1: EWMA assimétrico -- α da subida (rápido)
    ema_down_alpha: float = 0.1  # A1: α da descida (lento). ema_up >= ema_down. Só ativo se mode='ema_asym'.
    min_t: float | None = None
    """B3 (2026-07-25): gate de regime -- o pós-processo só se aplica em `t > min_t`; abaixo disso o
    score passa intacto. `None` = sem gate (aplica em todo t).

    Existe porque o A1 é um braço de t ALTO: ganha em `150<t<=400` e `t>400` e PERDE em `t<=50`. O
    gate evita pagar esse pedágio. A recursão RECOMEÇA na primeira observação acima do corte, o que
    reproduz exatamente a semântica offline de `scripts/apply_vema_oof.py --min-t` (que foi como o
    +0,0009 foi medido nas duas partições). Sem o reinício, a produção herdaria o `prev` do regime
    frio e divergiria da medição."""


@dataclass(frozen=True)
class WeightsConfig:
    # X1 (BRAINSTORM_RUPTURA_TSAUC.md §3): pondera os POSITIVOS por detectabilidade (o professor tira
    # peso de positivos que não carregam sinal no passo em que pesam). MEDIDO 2026-07-24, K=4 vs. V4:
    # soft +0,0040 [+0,0005, +0,0073] (IC exclui 0), hard +0,0036 [+0,0004, +0,0071]. Ganho concentrado
    # em 50<t≤150 e 150<t≤400 (os buckets de maior peso). detectabilidade = norma L2 dos eixos do censo
    # A1 (delta_logvar_e/rho1/kurt/exceed) × √m_bucket, computada INLINE no treino (model/detectability.py).
    detectability_mode: str = "none"     # none|hard|soft|ramp -- 'soft' é a variante adotada
    detect_floor: float = 0.3            # piso do multiplicador suave: w_pos ×= clip(d/q95, floor, 1)
    wt_align: bool = False               # B1/F3: alinhar massa de peso por t com w_t=n_pos·n_neg (braço)
    # C2(ii): inclui `delta_mean` (deslocamento de nível do resíduo, em desvios pré-τ) na norma L2 que
    # forma `d_i`. A vizinhança do X1 foi varrida só no eixo do `floor` (0,15 e 0,50, ambos fechados);
    # este é o segundo braço que a campanha listou e nunca rodou. `false` = X1 adotado, bit-a-bit.
    detect_include_delta_mean: bool = False


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
    trajectory: TrajectoryConfig
    mismatch: MismatchConfig
    spectral: SpectralConfig
    ordinal: OrdinalConfig
    multirep: MultiRepConfig
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
    weights: WeightsConfig = WeightsConfig()


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
        trajectory=TrajectoryConfig(**raw["trajectory"]),
        mismatch=MismatchConfig(**raw["mismatch"]),
        spectral=SpectralConfig(**raw["spectral"]),
        ordinal=OrdinalConfig(**raw["ordinal"]),
        multirep=MultiRepConfig(**raw["multirep"]),
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
        weights=WeightsConfig(**raw.get("weights", {})),
    )
