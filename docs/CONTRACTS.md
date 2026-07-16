# Contratos congelados (Fase 0)

Fonte: `docs/PLANO_REPOSITORIO.md` §3. Assinaturas abaixo são o contrato — mudar uma assinatura exige
atualizar este arquivo e avisar as frentes afetadas (checklist de PR, `docs/PLANO_REPOSITORIO.md` §8).

## `src/sbrt/state/base.py`

```python
class StateBlock(Protocol):
    def reset(self, h0: H0Params, cfg: Config) -> None: ...   # uma vez por série
    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None: ...  # uma vez por passo
    def features(self) -> dict[str, float]: ...               # feature nomeadas, a qualquer momento
```

`e` = inovação whitened+clipada, escala congelada do histórico (usar para variância/cauda).
`e_vol` = inovação whitened+vol-ajustada (plano §3.4, usar para média/dependência/forma).
`t` = índice do passo, 1-based.

Nomes de feature seguem a convenção `<bloco>_<estatistica>_<parametro>`, ex.: `cusum_mean_pos_d050`,
`ewma_mean_z_l010`, `bayes_lo_h0100`. Estáveis e persistidos em `features/assembly.py` (ordem canônica
= `sorted(feats.keys())`, congelada em `artifacts/models/vN/feature_schema.json`).

## `src/sbrt/state/h0.py` (plano §3)

```python
@dataclass(frozen=True)
class H0Params: ...

def fit_h0(hist: np.ndarray, cfg: Config) -> H0Params: ...   # puro, determinístico, ValueError se n_h < mínimo
def whiten_step(x: float, lags: RingBuffer, params: H0Params, cfg: Config) -> tuple[float, float]: ...
```

## `src/sbrt/state/scorer.py` (plano §15.1)

```python
class StreamScorer:
    def __init__(self, h0, blocks, ensemble, cfg): ...
    def update_features(self, x: float) -> dict[str, float]: ...   # MOTOR ÚNICO — usado por update() e model/dataset.py
    def update(self, x: float) -> float: ...                        # uma observação -> um score em [0,1]
```

## `src/sbrt/postprocess/monotonicity.py` (plano §7)

```python
Mode = Literal["free", "hold", "soft", "ema"]
def apply(p: float, prev: float | None, mode: Mode, cfg: Config) -> float: ...
```

## `src/sbrt/adapter/platform.py` — confirmado via `quickstarter_notebook.ipynb`

```python
def train(datasets: list[tuple[int, list[float], list[float], int | None]], model_directory_path: str) -> None: ...
def infer(datasets: Iterable[tuple[list[float], Iterable[float]]], model_directory_path: str): ...  # generator
```

`train`: cada elemento é `(dataset_id, x_historical, x_online, tau_index)`; `tau_index` é o índice
0-based **dentro de `x_online`** onde a quebra ocorre, ou `None`. `infer`: generator que primeiro dá um
`yield` vazio (sinaliza prontidão ao runner), depois para cada `(x_historical, x_online)` itera
`x_online` emitindo exatamente um `float` por ponto, em ordem. `x_online` só pode ser iterado uma vez em
produção.

## `src/sbrt/robustness/{generators,gates}.py` (plano §10) — ajuste ao contrato original

```python
def generate(scenario_id: str, seed: int, cfg=None) -> tuple[np.ndarray, np.ndarray, int | None]: ...

def evaluate(scenario_id: str, trajectories: list, control_trajectories: list | None,
             tau: int | None, cfg) -> GateResult: ...
```

O contrato original do plano de repositório tinha `evaluate(scenario_id, scores, tau, cfg)` — sem
como comparar cenário-vs-controle sem receber os dois. Correção: `evaluate` recebe explicitamente
`trajectories` (lista de trajetórias de score, uma por seed) e `control_trajectories` (idem, ou
`None` para cenários sem par, ex. T6/T9/T10/T12/T12b). `scenario_id` com sufixo `_ctrl` em
`generate` produz o gêmeo de controle com a mesma seed.

## `src/sbrt/evaluation/harness.py` — ajuste ao contrato original

```python
def check_prefix_equivalence(hist, online, scorer_factory: Callable[[np.ndarray, np.ndarray], object],
                              cut_points: list[int]) -> bool: ...
```

O plano original tinha `scorer_factory(hist)` (só o histórico). Correção necessária: `scorer_factory`
recebe também o segmento online que SERÁ replay-ado (completo ou truncado). Sem isso, um canário que
espia o futuro via estado interno (em vez de via `update(x)`) nunca é pego — o teste de prefixo
comparava execuções que "sabiam" o futuro completo dos dois lados por igual. Um `scorer_factory`
honesto simplesmente ignora o 2º argumento.

## `src/sbrt/model/dataset.py` — adição ao contrato original

```python
def build_training_rows(train_series, cfg, progress: bool = True, n_jobs: int = 1) -> pd.DataFrame: ...
```

`n_jobs` (novo, default 1 = comportamento original) paraleliza o laço **entre séries** via joblib —
cada série continua rodando pelo mesmo laço serial passo a passo dentro do seu processo; não é a
vetorização proibida por armadilha §13.2 (essa seria vetorizar o laço *dentro* de uma série). Existe
porque o motor de estado é Python puro (~1ms/passo medido) e o dataset real tem ~5M passos — sem
paralelismo, construir o dataset sozinho leva ~85 min. `configs/default.yaml: model.dataset_n_jobs`
controla o valor usado por `adapter/platform.py` (`-1` = todos os núcleos).

## `src/sbrt/model/train.py` — adição ao contrato original (plano_acao_v1_para_v2.md A2)

```python
def train(rows, weights, cfg, progress: bool = True) -> tuple["ModelEnsemble", np.ndarray]: ...
```

Passou a devolver `(ensemble, oof_pred)` em vez de só `ensemble` — `oof_pred` é a probabilidade
calibrada out-of-fold por linha, alinhada a `rows` (usada por A4, não persistida no artefato salvo).
`adapter/platform.py` descarta o segundo elemento (`ensemble, _ = train(...)`); `scripts/train.py`
salva as duas coisas.

Além disso, o treino agora usa `init_score = logit(p_hat(t))` (`model/base_rate.py`) e
`metric="binary_logloss"` em vez de `"auc"` — a taxa-base de `y_t` cresce fortemente com `t` (neutra
para TS-AUC por invariância C1, plano técnico §1.2) e dominava a métrica de early stopping,
parando o treino cedo demais (89-110 árvores medidas vs. 400-800 esperadas, plano §8.3). A curva de
taxa-base é persistida em `ModelEnsemble.base_rate_curve` / `base_rate_curve.json`; `predict_one`
soma o mesmo offset ao raw_score antes do sigmoid (LightGBM não readiciona `init_score` sozinho para
dados novos). Modelos salvos sem essa curva (v1) continuam funcionando sem offset (compat).

## Regras de ouro (não repetir aqui — ver `docs/PLANO_REPOSITORIO.md` §8)

Nenhum RNG fora de `robustness/generators.py` e `tests/`; nenhuma função em `state/`, `features/`,
`model/predict.py`, `postprocess/`, `adapter/` recebe `T`; nenhum número mágico fora de `cfg`; nenhuma
estimativa de TS-AUC em código de produção (§9.0).
