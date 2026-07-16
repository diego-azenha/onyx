# Plano de Implementação — Repositório Python
## ADIA Lab Structural Break Challenge: Real-Time Edition

**Escopo deste documento:** engenharia de software apenas — estrutura de pastas, contratos de interface, gestão de configuração, testes e ordem de trabalho para uma equipe de agentes implementar o plano técnico já produzido.
**Fonte da verdade matemática/metodológica:** o documento `plano_structural_break_realtime.md` (16 seções, doravante "**o plano técnico**"). Este documento **não redefine** fórmulas, hiperparâmetros ou gates — apenas referencia as seções onde vivem (`§N`) e diz **onde no código cada um entra**. Recomenda-se copiá-lo para `docs/PLANO_TECNICO.md` dentro do repositório (ver árvore, §2) para que as referências `§N` nos docstrings resolvam para um arquivo real.

---

## 0. Como usar este documento

Leitura recomendada para qualquer agente entrando no projeto: (1) esta seção; (2) §1 (decisão arquitetural — explica por que a árvore tem o formato que tem); (3) a tabela de frentes de trabalho (§7) para achar sua tarefa; (4) o contrato do(s) arquivo(s) que vai tocar (§3); (5) só então o plano técnico, na(s) seção(ões) referenciada(s) pelo contrato — não é preciso ler o plano técnico inteiro para começar a trabalhar em uma frente isolada.

---

## 1. Decisão arquitetural central: blocos de estado uniformes

O plano técnico especifica cinco famílias de estatísticas sequenciais (acumuladores/EWMA/janelas, CUSUM, filtro bayesiano, martingales conformais — §4) que são matematicamente distintas mas **operacionalmente idênticas**: todas recebem uma inovação por passo e mantêm estado interno. Para manter o repositório simples apesar da riqueza matemática, todo bloco implementa o mesmo contrato mínimo:

```python
class StateBlock(Protocol):
    def reset(self, h0: H0Params, cfg: Config) -> None: ...   # uma vez por série
    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None: ...  # uma vez por passo
    def features(self) -> dict[str, float]: ...               # feature nomeadas, a qualquer momento
```

`StreamScorer` (§3, arquivo `state/scorer.py`) não conhece CUSUM, Bayes ou conformal — ele só possui uma `list[StateBlock]` e faz `for b in blocks: b.update(...)`. Consequências práticas desta escolha:

- **Cada família é um arquivo pequeno e testável isoladamente**, sem precisar do resto do sistema funcionando (um teste unitário injeta `e` sintético direto, sem passar pelo whitening real).
- **Paralelismo real entre agentes**: quem implementa `cusum.py` não espera quem implementa `bayes_filter.py`, porque nenhum depende do outro — só do contrato `StateBlock`, que é congelado no dia 1 (Fase 0, §7).
- **Adicionar uma família nova é um novo arquivo + uma linha no `scorer.py`**, nunca uma mudança espalhada.
- Recursões que se repetem entre famílias (Welford, log-sum-exp) vivem em `utils/`, nunca duplicadas — evita a classe de bug "duas implementações de Welford que divergem depois de 500 passos por causa de um `if` diferente".

Nota de nomenclatura: o esqueleto do plano técnico (§15.1) descreve `H0Model.fit(hist)`. Aqui isso é implementado como `H0Params` (dataclass **imutável**) + função pura `fit_h0(hist, cfg) -> H0Params`, funcionalmente equivalente. A imutabilidade é deliberada: torna estruturalmente impossível reestimar o H0 no meio do online (bloqueio B2 do plano técnico) — não existe método `.refit()` para chamar por engano.

---

## 2. Estrutura de diretórios

Convenção de tamanho: arquivos em `state/` e `model/` devem ficar, como orientação, entre 60–200 linhas. Passar disso é sinal de que o arquivo está fazendo mais de uma coisa.

```
structural-break-rt/
├── README.md                    # quickstart + link para docs/PLANO_TECNICO.md
├── pyproject.toml
├── Makefile
├── .gitignore                   # cobre data/ e artifacts/ desde o 1º commit
├── configs/
│   └── default.yaml             # ÚNICA fonte de números/hiperparâmetros/gates (§4)
├── docs/
│   ├── PLANO_TECNICO.md         # cópia do plano técnico — fonte da verdade
│   └── CONTRACTS.md             # interfaces congeladas na Fase 0 (§7)
├── src/sbrt/                    # sbrt = Structural Break Real-Time
│   ├── __init__.py
│   ├── config.py                # loader tipado de configs/default.yaml
│   ├── utils/
│   │   ├── numerics.py          # welford_update, logsumexp, lgamma_cached, ewma_update
│   │   └── ring_buffer.py       # RingBuffer genérico O(1) (usado por janelas e lags)
│   ├── state/
│   │   ├── base.py              # Protocol StateBlock (§1 acima)
│   │   ├── h0.py                 # H0Params, fit_h0, whiten_step         — plano §3
│   │   ├── accumulators.py       # Welford global, EWMA, 5 janelas       — plano §4.2
│   │   ├── cusum.py              # banco de 15 CUSUMs + idades           — plano §4.2
│   │   ├── bayes_filter.py       # filtro de troca única, 2 hazards      — plano §4.3
│   │   ├── conformal.py          # martingales conformais                — plano §4.2
│   │   └── scorer.py             # StreamScorer: orquestra os blocks     — plano §15.1
│   ├── features/
│   │   └── assembly.py          # ordem canônica das features + schema  — plano §5
│   ├── postprocess/
│   │   └── monotonicity.py      # V-livre/hold/soft/ema                 — plano §7
│   ├── model/
│   │   ├── dataset.py            # motor único → linhas de treino        — plano §8.1
│   │   ├── weights.py            # pesos de linha w_row                  — plano §8.2
│   │   ├── train.py              # GroupKFold + LightGBM                 — plano §8.3
│   │   ├── predict.py            # ModelEnsemble.predict_one             — plano §8.4
│   │   └── fallback.py           # score puro-estatístico (sem ML)       — plano §8.5
│   ├── evaluation/
│   │   ├── harness.py            # replay causal + checagem de prefixo  — plano §9.2, §12.1
│   │   ├── splits.py             # GroupKFold estratificado             — plano §9.4
│   │   └── diagnostics.py        # curvas de treino, importância, distribuições — plano §9.1
│   ├── robustness/
│   │   ├── generators.py         # cenários T1–T13 (+T5b,T12b)          — plano §10
│   │   └── gates.py              # gates de mediana (comportamentais, não AUC) — plano §10
│   ├── adversarial/
│   │   ├── leaky_canary.py       # variante que espia o futuro          — plano §12.1
│   │   └── determinism.py        # re-execução 30% bit-a-bit            — plano §12.4
│   └── adapter/
│       └── platform.py          # shim para o callback oficial          — plano §15.1 (P0)
├── scripts/                     # CLIs finos — nenhuma lógica nova aqui; todas com barra de progresso (tqdm)
│   ├── build_dataset.py
│   ├── train.py
│   ├── diagnose.py               # gera relatório local (§9.1) — NÃO estima TS-AUC
│   ├── run_robustness_suite.py
│   ├── benchmark_latency.py
│   ├── check_determinism.py
│   └── submission_smoke_test.py
├── tests/
│   ├── unit/                    # um arquivo por módulo de src/sbrt
│   ├── causality/                # teste de prefixo + canário — plano §12.1
│   ├── determinism/               # rerun bit-exato — plano §12.4
│   └── robustness/               # T1–T13 com gates comportamentais — plano §10
├── data/                          # git-ignored
│   ├── raw/                     # dados baixados da plataforma
│   └── processed/               # cache do dataset de treino (§8.1, ~750MB)
├── artifacts/                    # git-ignored
│   ├── models/                  # ensembles LightGBM + feature_schema.json, versionados vN/
│   └── reports/                 # saída json/md dos scripts (diagnose, robustness, latency)
│       └── submission_log.md    # registro manual de cada sonda oficial: hipótese + resultado (§9.3)
└── notebooks/                    # exploratório apenas — nunca caminho crítico
```

---

## 3. Contratos por módulo

Os blocos abaixo são o conteúdo mínimo de `docs/CONTRACTS.md` (Fase 0, §7): assinaturas + docstring de uma linha, sem implementação. Todo agente implementa **contra** isto, não redefine assinaturas por conta própria — mudança de contrato exige atualizar este arquivo e avisar as frentes afetadas.

```python
# src/sbrt/state/base.py
from typing import Protocol

class StateBlock(Protocol):
    """Contrato comum a todo bloco de state/*. Um StreamScorer é, na prática,
    uma lista de StateBlocks (§1 deste documento)."""

    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        """Uma vez por série, logo após fit_h0."""

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        """Uma vez por passo. e = inovação whitened+clipada (escala congelada,
        usar para variância/cauda); e_vol = inovação whitened+vol-ajustada
        (plano §3.4, usar para média/dependência/forma); t = índice do passo, 1-based."""

    def features(self) -> dict[str, float]:
        """Features atuais do bloco, nomes estáveis (documentar em docs/CONTRACTS.md
        a convenção de nomes adotada, ex.: `cusum_mean_pos_d050` — os nomes exatos
        são decisão de implementação, desde que estáveis e persistidos no schema)."""
```

```python
# src/sbrt/utils/numerics.py — primitivas reaproveitadas por múltiplos blocks
def welford_update(n: int, mean: float, m2: float, x: float) -> tuple[int, float, float]:
    """Recursão de Welford (1962). Usada por accumulators.py E por bayes_filter.py
    (stats por candidato) — NUNCA reimplementar Welford localmente em outro arquivo."""

def logsumexp(values: "Sequence[float]") -> float: ...
def lgamma_cached(x: float) -> float: ...          # cache por ν inteiro-deslocado, plano §4.3
def ewma_update(prev: float, x: float, lam: float) -> float: ...

# src/sbrt/utils/ring_buffer.py
class RingBuffer:
    """Buffer circular O(1); usado por accumulators.py (janelas) e h0.py (lags,
    atravessa a fronteira histórico→online — plano §3.1 item 8, armadilha §13.3)."""
    def push(self, x: float) -> float | None: ...   # retorna o elemento expulso, ou None
```

```python
# src/sbrt/state/h0.py — plano §3
@dataclass(frozen=True)
class H0Params:
    phi: np.ndarray; c: float; sigma_e: float; sigma_e_rob: float
    nu_hat: float; q: dict[str, float]
    sorted_e_hist: np.ndarray; sorted_abs_e_hist: np.ndarray
    sigma_u: float; rho1_e: float; rho1_abs_e: float
    seasonal_lag: int | None; seasonal_coef: float; ar_r2: float

def fit_h0(hist: np.ndarray, cfg: "H0Config") -> H0Params:
    """§3.1. Puro e determinístico. ValueError se n_h < mínimo configurado."""

def whiten_step(x: float, lags: RingBuffer, params: H0Params, cfg: "H0Config") -> tuple[float, float]:
    """§3.2. Retorna (e_clipado, e_raw); empurra x em `lags`. `params` é imutável —
    nunca reestimado no online (bloqueio B2 do plano técnico)."""
```

```python
# src/sbrt/state/{accumulators,cusum,bayes_filter,conformal}.py — todos implementam StateBlock
#
# AccumulatorBlock  — Welford global + EWMA (média/var/sinal/exced.) + 5 janelas   — plano §4.2
# CusumBlock        — 15 acumuladores + idades (média/var/sinal/exced./dependência) — plano §4.2
# BayesFilterBlock  — 2 filtros (hazards h∈{1/100,1/400}), K=48, protege 8 recentes — plano §4.3
# ConformalBlock    — 4 famílias de log-martingale (abs/direita/sinal, com reset)   — plano §4.2
#
# Cada um expõe exclusivamente via features() -> dict[str, float]; nenhum outro
# módulo lê seus atributos internos diretamente.
```

```python
# src/sbrt/state/scorer.py — plano §15.1
class StreamScorer:
    def __init__(self, h0: H0Params, blocks: list["StateBlock"],
                 ensemble: "ModelEnsemble | None", cfg: "Config"): ...

    def update_features(self, x: float) -> dict[str, float]:
        """Um passo: whiten_step -> update() de cada block -> merge + meta-features
        (t, n_h, ν̂, ρ̂₁, ...). MOTOR ÚNICO: usado tanto por update() quanto por
        model/dataset.py — nenhum outro lugar reimplementa este laço (plano §8.1)."""

    def update(self, x: float) -> float:
        """UMA observação → UM score em [0,1]."""
        feats = self.update_features(x)
        p = (self.ensemble.predict_one(feats) if self.ensemble
             else fallback_score(feats, self.cfg))
        return apply_monotonicity(p, self._prev_score, self.cfg.postprocess.mode, self.cfg)
```

```python
# src/sbrt/features/assembly.py — plano §5
FEATURE_ORDER: tuple[str, ...]   # união estável das features() de todos os blocks + meta

def to_array(feats: dict[str, float]) -> np.ndarray:
    """Materializa na ordem canônica; chave ausente (warm-up) -> NaN. LightGBM
    trata NaN nativamente — nunca usar sentinela numérica."""

def save_schema(path: Path) -> None: ...   # grava junto com o modelo (artifacts/models/vN/)
def load_schema(path: Path) -> tuple[str, ...]: ...
```

```python
# src/sbrt/postprocess/monotonicity.py — plano §7
Mode = Literal["free", "hold", "soft", "ema"]

def apply(p: float, prev: float | None, mode: Mode, cfg: "PostprocessConfig") -> float:
    """mode='free' (default) = identidade. 'hold'/'soft'/'ema' só habilitados em
    configs/default.yaml se o gate G-mono (plano §9.4) tiver sido confirmado."""
```

```python
# src/sbrt/model/dataset.py — plano §8.1
def build_training_rows(train_series: Iterable["SeriesRecord"], cfg: "Config",
                         progress: bool = True) -> pd.DataFrame:
    """Para cada série: fit_h0 + StreamScorer(ensemble=None).update_features() passo
    a passo + thinning + y_t=1{tau<=t}. Proibido vetorizar o segmento online aqui.
    Laço externo (por série) envolvido em tqdm quando progress=True — é laço de
    desenvolvimento local, não caminho de submissão (§8 regra tqdm)."""

# src/sbrt/model/weights.py — plano §8.2
def compute_row_weights(rows: pd.DataFrame, cfg: "Config") -> np.ndarray:
    """w(t) = n_pos(t)·n_neg(t) empírico / n_alive(t), normalizado, × multiplicador
    de thinning."""

# src/sbrt/model/train.py — plano §8.3
def train(rows: pd.DataFrame, weights: np.ndarray, cfg: "Config",
          progress: bool = True) -> "ModelEnsemble":
    """GroupKFold(5, groups=rows.id) + 1 LightGBM por fold; salva em artifacts/models/vN/.
    tqdm sobre os 5 folds; dentro de cada fold, LightGBM usa seu próprio log verbose
    (não duplicar barra de progresso)."""

# src/sbrt/model/predict.py — plano §8.4
class ModelEnsemble:
    @classmethod
    def load(cls, path: Path) -> "ModelEnsemble": ...
    def predict_one(self, feats: dict[str, float]) -> float:
        """Média dos folds; num_threads=1; ordem de colunas fixada pelo schema salvo.
        SEM tqdm — é caminho de inferência real (§8 regra tqdm)."""

# src/sbrt/model/fallback.py — plano §8.5
def fallback_score(feats: dict[str, float], cfg: "Config") -> float:
    """score = σ(0,9·LO_h400 + 0,4·max(CUSUM_z) + 0,3·logM_abs_reset − b)."""
```

```python
# src/sbrt/evaluation/harness.py — plano §9.2, §12.1
def replay(hist: np.ndarray, online: np.ndarray, scorer: "StreamScorer",
           progress: bool = False) -> list[float]:
    """Alimenta o online um ponto por vez. NUNCA vetoriza. progress=True só em uso
    manual/exploratório — desligado por padrão para não poluir laços aninhados
    (ex.: quando chamado de dentro de build_training_rows, que já tem tqdm por série)."""

def check_prefix_equivalence(hist, online, scorer_factory, cut_points: list[int]) -> bool:
    """Para cada k em cut_points: score(replay completo)[:k] == score(replay
    truncado em k), bit a bit. Usado por tests/causality/ e pelos scripts de CI.
    NÃO calcula nenhuma métrica de desempenho — só corretude de código (§9.0)."""

# src/sbrt/evaluation/splits.py — plano §9.4
def grouped_stratified_kfold(meta: pd.DataFrame, k: int, seed: int
                              ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Agrupado por id; estratificado por (rótulo da série, bucket de T, terço de τ)."""

# src/sbrt/evaluation/diagnostics.py — plano §9.1
def training_curves(booster_evals: dict, out_path: Path) -> None:
    """Plota logloss/AUC de linha por fold e rodada de boosting — métrica de treino,
    não estimativa de TS-AUC oficial (rótulo explícito no eixo do gráfico)."""

def feature_importance_report(ensemble: "ModelEnsemble", out_path: Path) -> pd.DataFrame:
    """Gain + split count por feature, agregados nos 5 folds."""

def score_distribution_report(rows: pd.DataFrame, scores: np.ndarray, out_path: Path) -> None:
    """Histogramas de score por fatia de t, y=0 vs y=1 — inspeção visual, sem reduzir
    a um número de desempenho."""
```

```python
# src/sbrt/robustness/generators.py — plano §10
def generate(scenario_id: str, seed: int, cfg: "RobustnessConfig"
             ) -> tuple[np.ndarray, np.ndarray, int | None]:
    """T1..T13 (+T5b,T12b). Retorna (hist, online, tau_or_None). RNG livre aqui —
    é geração de dado sintético, não caminho de inferência."""

# src/sbrt/robustness/gates.py — plano §10
def evaluate(scenario_id: str, scores: list[float], tau: int | None,
             cfg: "RobustnessConfig") -> "GateResult":
    """Aplica o(s) gate(s) do cenário (tabela §10, revisada) — comparação de MEDIANA
    entre cenário e controle, deliberadamente não uma AUC (§9.0)."""
```

```python
# src/sbrt/adversarial/leaky_canary.py — plano §12.1
class LeakyStreamScorer(StreamScorer):
    """Variante que deliberadamente espia x_{t+1}. DEVE ser reprovada por
    check_prefix_equivalence — é a prova de que o detector de vazamento funciona."""

# src/sbrt/adversarial/determinism.py — plano §12.4
def rerun_bitexact(series_sample, scorer_factory, fraction: float, seed: int,
                    progress: bool = True) -> bool:
    """Re-executa `fraction` das séries e compara bit a bit com a 1ª execução
    (mais estrito que a tolerância 1e-8 da plataforma, plano F9). tqdm sobre a
    amostra reexecutada — pode levar minutos em 3·10³ séries."""
```

```python
# src/sbrt/adapter/platform.py — plano §15.1 (tarefa de investigação da Fase P0)
# ATENÇÃO: assinatura abaixo é best-guess — ver §9 (Riscos) deste documento.
# Confirmar contra o notebook oficial antes de finalizar.
def train(X_train, y_train, model_directory_path: str) -> None: ...
def infer(...) -> Iterator[float]:
    """Formato provável: recebe o histórico, depois itera o stream fazendo
    `yield score` a cada observação (padrão de streaming mencionado na
    documentação da família de competições)."""
```

---

## 4. Configuração centralizada

Regra: **nenhum número que aparece no plano técnico (delta, hazard, λ, learning_rate, threshold de gate...) é escrito diretamente em código-fonte.** Tudo vem de `configs/default.yaml`, carregado uma vez por `config.py` num objeto tipado (dataclasses aninhadas ou pydantic — qualquer um serve, priorizar o que a equipe já conhece). Isso permite que a Fase P3 de ablações (plano §15.3) seja feita trocando YAML, não código.

```yaml
# configs/default.yaml (trecho ilustrativo — lista completa segue as tabelas do plano técnico)
seed: 42

h0:                                    # plano §3.1
  ar_order: 10
  seasonal_acf_threshold: 0.25
  seasonal_lag_range: [6, 128]
  min_hist_len: 50

state:                                 # plano §4.2
  ewma_lambdas: [0.05, 0.10, 0.30]
  window_sizes: [10, 25, 50, 100, 250]
  cusum_mean_deltas: [0.25, 0.5, 1.0]
  cusum_var_ratios_up: [1.5, 2.5]
  cusum_var_ratio_down: 0.5
  vol_adjust: {threshold_rho1_abs: 0.15, lambda_v: 0.06}

bayes:                                 # plano §4.3
  hazards: [0.01, 0.0025]
  max_candidates: 48
  protect_recent: 8
  prior: {mu0: 0.0, kappa0: 0.5, nu0: 2, sigma0_sq: 1.5}

conformal:
  epsilons: [0.05, 0.1, 0.2, 0.4]     # plano §4.2

lightgbm:                              # plano §8.3
  learning_rate: 0.05
  num_leaves: 63
  min_data_in_leaf: 200
  lambda_l2: 5.0
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 1
  max_bin: 255
  n_estimators_cap: 1500
  early_stopping_rounds: 100
  deterministic: true
  force_row_wise: true
  train_num_threads: 8
  predict_num_threads: 1

thinning:                              # plano §8.1
  full_until: 100
  step_101_400: 2
  step_401_plus: 4

gates:                                  # plano §10 (tabela revisada), §11.1 — thresholds
                                         # COMPORTAMENTAIS (gap de mediana), nunca TS-AUC local (§9.0)
  robustness_median_gap:
    t1: 0.40
    t3: 0.15
    t4: 0.55
    t5: 0.35
    t5b: 0.20
    t7: 0.20
    t8: 0.15
  drift_slope_abs_max: 1.0e-4
  latency_budget_us_per_step: 300

submission:                             # plano §9.3 — G-0/G-mono/G-peso são decididos por
  log_path: "artifacts/reports/submission_log.md"   # submissão oficial, registrada aqui,
                                                      # nunca por um número calculado localmente

postprocess:
  mode: free                           # free | hold | soft | ema — plano §7
```

---

## 5. Pipeline de execução

```
scripts/build_dataset.py         → data/processed/train_rows.parquet          [§8.1]
scripts/train.py                 → artifacts/models/vN/ (+feature_schema.json) [§8.3]
scripts/diagnose.py              → artifacts/reports/diagnose_vN.json + gráficos [§9.1]
scripts/run_robustness_suite.py  → artifacts/reports/robustness_vN.json        [§10]
scripts/benchmark_latency.py     → artifacts/reports/latency_vN.json           [§11]
scripts/check_determinism.py     → pass/fail no stdout                        [§12.4]
scripts/submission_smoke_test.py → roda adapter/platform.py fim a fim         [§15.1]
```

`scripts/*.py` são CLIs finas (parse de argumentos + chamada a `src/sbrt/...` + gravação do relatório) — nenhuma lógica nova deve ser escrita dentro de `scripts/`. Todos os scripts que iteram sobre séries (`build_dataset.py`, `diagnose.py`, `run_robustness_suite.py`, `benchmark_latency.py`, `check_determinism.py`) usam `tqdm` no laço externo, com descrição do que está rodando (`tqdm(series, desc="construindo dataset de treino")`) — ver regra em §8. `scripts/diagnose.py` **não produz** nenhuma estimativa de TS-AUC; gera as curvas/relatórios de §9.1.

```makefile
# Makefile (alvos esperados)
dataset:      python scripts/build_dataset.py --config configs/default.yaml
train:        python scripts/train.py --config configs/default.yaml
diagnose:     python scripts/diagnose.py --config configs/default.yaml
robustness:   python scripts/run_robustness_suite.py --config configs/default.yaml
benchmark:    python scripts/benchmark_latency.py
determinism:  python scripts/check_determinism.py
smoke:        python scripts/submission_smoke_test.py
test:         pytest tests/
ci:           pytest tests/unit tests/causality tests/determinism   # subconjunto rápido
```

---

## 6. Estratégia de testes

| Pasta | Verifica | Referência no plano técnico |
|---|---|---|
| `tests/unit/` | corretude isolada por módulo (Welford bate a mão; CUSUM zera corretamente; `grouped_stratified_kfold` particiona sem vazar `id` entre folds) | pré-requisito de tudo |
| `tests/causality/` | teste de prefixo aprova o `StreamScorer` real e reprova o `LeakyStreamScorer` | §12.1 |
| `tests/determinism/` | re-execução de 30% das séries, bit a bit | §12.4, checklist §15.2 |
| `tests/robustness/` | T1–T13 (+T5b,T12b) cumprem os gates de mediana da tabela §10 (comportamentais, não TS-AUC — §9.0) | §10 |

Nenhuma pasta de teste calcula ou reporta uma estimativa de TS-AUC (§9.0) — `tests/unit/` inclui testes de `diagnostics.py` (as curvas/relatórios são gerados sem erro, com dados sintéticos pequenos), não testes de "o modelo aprendeu bem o suficiente".

`tests/unit/test_scorer.py` roda `StreamScorer` sobre as 13+2 séries de `robustness/generators.py` só para garantir 1000 passos sem exceção/NaN fora do warm-up — não checa qualidade de detecção (isso é `tests/robustness/`).

---

## 7. Plano de execução para a equipe de agentes

### 7.1 Fase 0 — Contratos (pré-requisito de tudo, dia 1)

Uma única sessão: criar toda a árvore de `src/sbrt/` com os contratos da §3 como stubs (`raise NotImplementedError`), preencher `configs/default.yaml` com os valores das tabelas do plano técnico, implementar `config.py` (carrega e valida o YAML), copiar o plano técnico para `docs/PLANO_TECNICO.md`, e congelar `docs/CONTRACTS.md`. **Critério de saída:** `pytest tests/` roda (falhando com `NotImplementedError`, não com `ImportError`) — isso prova que o esqueleto é válido e dá a todas as frentes um alvo estável para trabalhar em paralelo.

### 7.2 Frentes de trabalho

| Frente | Arquivos | Depende de | Pode iniciar | Critério de pronto (DoD) |
|---|---|---|---|---|
| **A** — H0 + whitening | `state/h0.py` | Fase 0 | imediatamente | `fit_h0` recupera φ conhecidos (±0,05) de um AR(10) sintético; `whiten_step` produz e ~N(0,1) (KS-test p>0,05) sobre séries H0 puras; buffers atravessam a fronteira sem descontinuidade |
| **B** — Acumuladores + CUSUM | `state/accumulators.py`, `state/cusum.py` | `state/base.py` | imediatamente, com `e` sintético | recursões batem a 1e-9 contra ≥20 passos calculados à mão; idades resetam corretamente |
| **C** — Bayes + conformal | `state/bayes_filter.py`, `state/conformal.py` | `state/base.py` | imediatamente, idem B | LO<0 na maioria dos passos em série sem quebra; τ̂_MAP acerta ±5 passos em quebra abrupta clara; zero NaN/Inf em 1000 passos |
| **D** — Harness e diagnóstico | `evaluation/harness.py`, `splits.py`, `diagnostics.py` | Fase 0 | imediatamente, com scores sintéticos | `check_prefix_equivalence` funcional contra scorer-dummy; `diagnostics.py` gera curva/relatório de exemplo sem erro sobre dados sintéticos pequenos. **Não** produz nenhuma estimativa de score — isso é fora de escopo por decisão de projeto (§9.0) |
| **E** — Geradores de robustez | `robustness/generators.py`, `gates.py` | Fase 0 | imediatamente | os 13+2 cenários produzem `(hist, online, tau)` com as propriedades estatísticas esperadas (ex.: T6 com ρ̂₁(\|e\|) alto de fato); `gates.py` aplica os gates de mediana da tabela §10 revisada, não AUC |
| **F** — Orquestração | `state/scorer.py`, `features/assembly.py` | A, B, C (interface) | após contrato `StateBlock` congelado; pode usar blocks-stub no início | `update_features` retorna as chaves esperadas; `update()` roda 1000 passos sem exceção sobre as séries de E |
| **G** — Adversarial | `adversarial/leaky_canary.py`, `determinism.py` | D + F | versão inicial contra scorer-dummy em paralelo com D; validação final após F | teste de prefixo reprova o canário e aprova o scorer real; rerun 30% bit-exato passa |
| **H** — Camada supervisionada | `model/*.py` | F (scorer completo) + D | `fallback.py` pode começar cedo (só depende do contrato de features); o resto só após F | `scripts/diagnose.py` roda fim a fim e produz curvas de treino + importância de features (§9.1); gate G-0 fica pendente de submissão oficial (§9.3) — o DoD desta frente é o pipeline de treino funcionar, não um número local aprovado |
| **I** — Monotonicidade | `postprocess/monotonicity.py` | convenção de `prev_score` acordada com F | cedo, é isolado | as 4 variantes implementadas; teste prova que V-hold trava no contraexemplo CE1 e V-livre não |
| **J** — Adapter de plataforma | `adapter/platform.py` | assinatura oficial confirmada (ver §9) | investigação desde o dia 1; implementação final bloqueada até confirmação | `scripts/submission_smoke_test.py` roda contra o adapter (real ou mock fiel documentado) |

### 7.3 Checkpoints de integração (mapeiam às fases P0–P4 do plano técnico §15.3)

| Checkpoint | Frentes envolvidas | Corresponde a | Critério |
|---|---|---|---|
| 1 | Fase 0 | pré-requisito de P0 | `pytest` roda; config carrega |
| 2 | D, E, H(fallback) | P0 | fallback + harness + robustez rodam ponta a ponta; 1ª sonda de submissão usa este caminho (para validar pipeline, não para comparar arquiteturas) |
| 3 | A, B, C, F, G | P1 | `StreamScorer` real substitui o fallback; checklist de determinismo (plano §15.2) 100% verde |
| 4 | H (completo) | P2 | modelo treinado; `scripts/diagnose.py` é a referência local de qualidade (aprendizado, não desempenho); gate G-0 decidido por submissão oficial dedicada |
| 5 | I, J + ablações | P3–P4 | ablações via config; `adapter/platform.py` finalizado e testado |

---

## 8. Checklist de PR ("regras de ouro")

- [ ] Nenhum `import random` / `np.random` / `default_rng` em `state/`, `features/`, `model/predict.py`, `model/fallback.py`, `postprocess/`, `adapter/` — permitido **apenas** em `robustness/generators.py` e em `tests/`.
- [ ] Nenhuma função em `state/`, `features/`, `model/predict.py`, `postprocess/`, `adapter/` recebe `T` (tamanho total da série) como argumento.
- [ ] Todo número que seria "mágico" vem de `cfg`, nunca hardcoded no corpo da função.
- [ ] Toda feature nova é adicionada ao `features()` de algum `StateBlock` e à `FEATURE_ORDER` de `assembly.py` — nunca calculada solta dentro de `scorer.py`.
- [ ] `model/dataset.py` chama `StreamScorer.update_features` — não reimplementa nenhuma fórmula de `state/`.
- [ ] Qualquer mudança em `state/` ou `model/predict.py` roda `tests/determinism/` e `tests/causality/` localmente antes do PR.
- [ ] Checklist de determinismo do plano técnico (§15.2) revisado antes de qualquer sonda de submissão.
- [ ] **Nenhum código novo calcula ou reporta uma estimativa de TS-AUC** (nenhuma função chamada `ts_auc`/`estimate_score`/similar reaparece no repositório) — decisão de projeto §9.0. Se alguém achar necessário reintroduzir isso, é uma decisão a discutir explicitamente, não um PR silencioso.
- [ ] Todo laço em `scripts/*.py` e em `model/dataset.py`/`model/train.py`/`adversarial/determinism.py` que itera sobre uma coleção de séries (≥10) usa `tqdm` com `desc` descritivo — nunca dentro de `state/scorer.py` (`update()`) nem em `adapter/platform.py`.

---

## 9. Riscos e pendências de engenharia

- **Validação local enganosa (risco confirmado por experiência prévia, não hipotético):** tentativas anteriores de replicar a métrica oficial localmente produziram números sistematicamente otimistas. Mitigação já incorporada em todo este documento (§9.0): nenhum harness local estima TS-AUC; toda decisão de "isto é melhor que aquilo" passa pela engine oficial. Se, no futuro, a equipe considerar reintroduzir uma estimativa local, tratar como uma decisão de projeto explícita (não um utilitário "conveniente" adicionado de passagem) e validar a própria ferramenta de validação contra pelo menos uma submissão oficial antes de confiar nela.
- **Assinatura oficial do callback (`adapter/platform.py`):** não confirmada nesta sessão — a página do baseline no Kaggle é renderizada via JS e não pôde ser lida por fetch automatizado. Ação da frente J: obter o notebook oficial (download manual ou ferramenta oficial de setup da CrunchDAO) antes de fechar P4. Até lá, o contrato best-guess em `docs/CONTRACTS.md` desbloqueia as demais frentes.
- **Download de dados de treino:** confirmar e usar a via oficial de setup/download da CrunchDAO (a competição tipicamente distribui uma ferramenta própria para isso) em vez de scraping manual do hub — decisão da frente J na Fase 0.
- **Versionamento:** fixar a versão exata de `lightgbm` e `numpy` em `pyproject.toml`/lockfile — `deterministic=True` é garantia relativa à versão da biblioteca, não absoluta entre versões diferentes. Adicionar `tqdm` às dependências de execução (não é só ferramenta de dev — os scripts de produção local dependem dela).
- **Volume de dados:** `data/processed/` (~750MB, plano §8.1) e `artifacts/models/` nunca vão para o git — `.gitignore` deve cobri-los desde o primeiro commit; ambos são regeneráveis via `make dataset` / `make train`.
- **Memória do array ordenado do histórico** (plano §11.2): se o teto de memória com ~10⁴ séries simultâneas apertar, subamostrar `sorted_e_hist`/`sorted_abs_e_hist` para 1024 pontos (degradação negligível do p-value conformal) — mitigação já prevista, não é bloqueio.
- **Decisão A1 (interpretação do rótulo por passo, plano §9.4/§14):** deve ser resolvida (documentação oficial ou fórum) antes do Checkpoint 4 — afeta diretamente `model/dataset.py` e `model/weights.py`.

— Fim do plano de repositório —
