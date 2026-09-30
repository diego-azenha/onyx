# Notas operacionais para agentes

**Público:** agentes de IA (ou pessoas) que vão modificar este repositório. Nada aqui é necessário
para *entender* o modelo — isso está em [`MODELO.md`](../modelo/MODELO.md) — nem para saber o que já foi
tentado — isso está em [`HISTORICO.md`](../historico/HISTORICO.md) — nem para saber o que vem a seguir — isso está
em [`BACKLOG_TSAUC.md`](../historico/BACKLOG_TSAUC.md). Aqui ficam invariantes, contratos, comandos, inventário de
artefatos, pegadinhas medidas e pendências abertas.

**Ordem de leitura recomendada ao entrar no projeto:** §1 (invariantes) → §2 (contratos) → o arquivo
que você vai tocar → a seção `§N` de `MODELO.md` referenciada no docstring dele. Não é preciso ler
`MODELO.md` inteiro para trabalhar numa frente isolada.

---

## 1. Invariantes — o checklist de PR ("regras de ouro")

Violar qualquer um destes quebra causalidade, determinismo ou o motor único. Não são preferências de
estilo.

- [ ] **Nenhum RNG** (`import random`, `np.random`, `default_rng`) em `state/`, `features/`,
      `model/predict.py`, `model/fallback.py`, `postprocess/`, `adapter/`. Permitido **apenas** em
      `robustness/generators.py` e `tests/`.
- [ ] **Nenhuma função** em `state/`, `features/`, `model/predict.py`, `postprocess/`, `adapter/`
      recebe `T` (tamanho total da série) como argumento. `T` é futuro (`MODELO.md` §12.1, CE5).
- [ ] **Todo número mágico vem de `cfg`** (`configs/default.yaml`), nunca hardcoded no corpo da
      função. Isso é o que permite ablações por YAML em vez de por código.
- [ ] **Toda feature nova** é adicionada ao `features()` de algum `StateBlock` e entra na ordem
      canônica de `features/assembly.py` — nunca calculada solta dentro de `scorer.py`.
- [ ] **`model/dataset.py` chama `StreamScorer.update_features`** — não reimplementa nenhuma fórmula
      de `state/`. Vetorizar o laço *dentro* de uma série é proibido (paralelizar *entre* séries via
      `n_jobs` é permitido e usado).
- [ ] **Qualquer mudança em `state/` ou `model/predict.py`** roda `tests/determinism/` e
      `tests/causality/` **antes** do PR.
- [ ] **Nenhum código de produção calcula ou reporta uma estimativa de TS-AUC** como substituto do
      score oficial (`MODELO.md` §9.0). Ferramentas de *diagnóstico* em `scripts/` e
      `evaluation/ts_auc.py` são a exceção explícita e existem para o protocolo R0 — não devem
      migrar para o caminho de inferência.
- [ ] **Nenhuma iteração sobre `dict`/`set`** no caminho de inferência (ordem não determinística).
- [ ] **Score emitido em float64 sem arredondamento** — empates artificiais alteram a AUC.
- [ ] **`tqdm` em todo laço sobre ≥10 séries** em `scripts/*.py`, `model/dataset.py`, `model/train.py`
      e `adversarial/determinism.py` — **nunca** dentro de `scorer.update()` nem em
      `adapter/platform.py`.
- [ ] **Arquivos em `state/` e `model/`**: 60–200 linhas como orientação. Passar disso costuma
      significar que o arquivo faz mais de uma coisa.

---

## 2. Contratos congelados

Mudar uma assinatura aqui exige atualizar esta seção e avisar as frentes afetadas.

### 2.1 O bloco de estado (a abstração central)

```python
# src/sbrt/state/base.py
class StateBlock(Protocol):
    def reset(self, h0: H0Params, cfg: Config) -> None: ...        # uma vez por série
    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None: ...  # uma vez por passo
    def features(self) -> dict[str, float]: ...                     # nomes estáveis
```

- `e` = inovação whitened+clipada, **escala congelada** do histórico → usar para **variância/cauda**.
- `e_vol` = inovação whitened+**vol-ajustada** → usar para **média/dependência/forma**.
  Essa separação é a trava anti-absorção (`MODELO.md` §3.4 / CE2). Trocar o fluxo de uma família de
  variância para `e_vol` cega quebras reais de variância em ~17 passos.
- `t` = índice do passo, 1-based.

`StreamScorer` não conhece CUSUM, Bayes ou conformal — ele possui uma `list[StateBlock]` e faz
`for b in blocks: b.update(...)`. **Adicionar uma família nova = um arquivo novo + uma linha em
`scorer.py:default_blocks`**, nunca uma mudança espalhada.

**Segundo contrato: blocos de meta-estatística** (F4, `state/trajectory.py`). Uma família que mede a
trajetória *dos estatísticos já calculados* não cabe em `update(e, e_raw, e_vol, t)` — ela precisa do
dict de features, não da inovação. O contrato é:

```python
def update_from_feats(self, feats: dict[str, float], t: int) -> None: ...
```

Chamado por `StreamScorer.update_features` **depois** de `apply_calibration`, para que a trajetória
seja medida sobre estatísticos já calibrados contra o nulo da própria série (a inclinação de um
z-score é comparável entre séries; a de um valor cru não é). Features vindas daqui **não** recebem
`_cal` — já nascem em unidades calibradas.

Convenção de nomes: `<bloco>_<estatistica>_<parametro>` (`cusum_mean_pos_d050`, `ewma_mean_z_l010`,
`bayes_lo_h0100`). Sufixo `_cal` = versão calibrada contra o nulo da própria série (F1). Ordem
canônica = `sorted(feats.keys())`, congelada em `artifacts/models/vN/feature_schema.json`.

### 2.2 Núcleo

```python
# state/h0.py
@dataclass(frozen=True)
class H0Params: ...                                   # imutável de propósito: não existe .refit()
def fit_h0(hist: np.ndarray, cfg: Config) -> H0Params: ...
def whiten_step(x, lags: RingBuffer, params, cfg) -> tuple[float, float]: ...  # (e_clip, e_raw)

# state/scorer.py
class StreamScorer:
    def update_features(self, x: float) -> dict[str, float]: ...   # MOTOR ÚNICO
    def update(self, x: float) -> float: ...                        # uma observação → um score

# utils/numerics.py — primitivas compartilhadas; NUNCA reimplementar Welford localmente
welford_update · logsumexp · lgamma_cached · ewma_update
# utils/ring_buffer.py — RingBuffer.push(x) -> elemento expulso | None
```

### 2.3 Adaptador da plataforma (confirmado contra o `quickstarter_notebook.ipynb`)

```python
def train(datasets: list[tuple[int, list[float], list[float], int | None]],
          model_directory_path: str) -> None: ...
def infer(datasets: Iterable[tuple[list[float], Iterable[float]]],
          model_directory_path: str): ...  # generator
```

- `train`: cada elemento é `(dataset_id, x_historical, x_online, tau_index)`; `tau_index` é 0-based
  **dentro de `x_online`**, ou `None`.
- `infer`: generator que **primeiro dá um `yield` vazio** (sinaliza prontidão ao runner —
  `crunch.container.GeneratorWrapper`, `ERROR_FIRST_YIELD_MUST_BE_NONE`), depois, para cada
  `(x_historical, x_online)`, itera `x_online` emitindo **exatamente um `float` por ponto**, em
  ordem. **`x_online` só pode ser iterado uma vez em produção.**

### 2.4 Correções feitas sobre o contrato original (não regredir)

| Assinatura | Correção | Por quê |
|---|---|---|
| `check_prefix_equivalence(hist, online, scorer_factory, cut_points)` | `scorer_factory` recebe **também** o segmento online que será replay-ado | com `scorer_factory(hist)` só, um canário que espia o futuro via estado interno nunca é pego — os dois lados "sabiam" o futuro por igual. Um factory honesto ignora o 2º argumento |
| `robustness.gates.evaluate(scenario_id, trajectories, control_trajectories, tau, cfg, reference_trajectories=None)` | recebe cenário **e** controle explicitamente; `reference_trajectories` é o painel i.i.d. de R5 | não dá para comparar cenário-vs-controle sem receber os dois. `reference_trajectories=None` reproduz o gate absoluto original bit-a-bit (usado só pelo modo `fallback`) |
| `build_training_rows(..., n_jobs=1)` | paraleliza **entre séries** via joblib | motor de estado é Python puro; sem isso o dataset leva ~85 min. Não é a vetorização proibida (essa seria dentro de uma série) |
| `train(...) -> tuple[ModelEnsemble, np.ndarray]` | devolve `(ensemble, oof_pred)` | `oof_pred` alimenta o protocolo R0. `adapter/platform.py` descarta o segundo elemento |

---

## 3. Estrutura do repositório

```
src/sbrt/
  config.py              loader tipado de configs/default.yaml
  utils/                 numerics.py · ring_buffer.py
  state/                 base.py (Protocol) · h0.py · calibration.py · scorer.py
                         + um arquivo por família: accumulators · cusum · bayes_filter · conformal
                           · rank_twosample · mmd · multiscale · fingerprint · dependence
                           · lmoments · varloc · jumps · bocpd (estacionado)
  features/assembly.py   ordem canônica + schema persistido
  postprocess/           monotonicity.py (free | hold | soft | ema)
  model/                 dataset · weights · base_rate · train · predict · fallback
  evaluation/            harness (replay causal + teste de prefixo) · splits · diagnostics · ts_auc
  robustness/            generators (T1–T13 + controles + painel de referência) · gates
  adversarial/           leaky_canary.py · determinism.py
  adapter/platform.py    shim do callback oficial
scripts/                 CLIs finas — NENHUMA lógica nova aqui
tests/                   unit/ · causality/ · determinism/ · robustness/
configs/default.yaml     ÚNICA fonte de números
data/ artifacts/         git-ignored, regeneráveis
resources/               artefato empacotado na submissão (modelo + schema + curva de taxa-base)
```

**`configs/default.yaml` é a única fonte de verdade numérica.** Comentários no YAML apontam para
`plano §N` (= `MODELO.md` §N) e para as rodadas que justificaram cada valor. Mudanças de
comportamento devem ser expressáveis como diff de YAML sempre que possível.

---

## 4. Comandos

```bash
make ci            # unit + causality + determinism (rápido — rode sempre)
make test          # tudo, incluindo robustez
make dataset       # build_dataset.py  → data/processed/train_rows.parquet   (~9 min paralelo)
make train         # train.py          → artifacts/models/vN/ + oof_vN.parquet (~10–15 min)
make diagnose      # curvas de treino, importância, distribuições
make robustness    # suíte T1–T13 (200 seeds ≈ 1 h; CI usa 40)
make benchmark     # latência por passo (gate 1500 µs)
make determinism   # re-execução de 30% bit a bit
make smoke         # adapter/platform.py fim a fim
make run_all       # encadeia dataset → train → diagnose → robustness → OOF por bucket
```

Scripts de análise fora do Makefile: `compare_oof.py` (**R0 — o juiz**), `shap_report.py`
(XS-SHAP), `crunch_local_ts_auc.py` (held-out no molde crunch), `oof_ts_auc_by_bucket.py`,
`break_type_census.py`, `oof_step_response.py`, `power_envelope_check.py`,
`ce6_history_classifier.py`, `sweep_hyperparams.py`, `train_rank.py`, `combine_oof.py`,
`build_submission_notebook.py` + `verify_submission_notebook.py`.

**Dependências:** numpy, pandas, scikit-learn, lightgbm, pyyaml, tqdm, joblib, scipy, matplotlib.
`deterministic=true` do LightGBM é garantia **relativa à versão** — pinar `lightgbm` e `numpy` antes
de congelar uma submissão.

---

## 5. Protocolo de medição (obrigatório para qualquer mudança)

Ordem que evita gastar horas em nada:

1. **Registre a hipótese por escrito antes de medir** (anti garden-of-forking-paths), incluindo o
   bucket-alvo esperado. Sem isso, o IC do R0 perde sentido.
2. `make ci` — causalidade e determinismo primeiro.
3. `make dataset && make train` para a variante e para o baseline **nos mesmos dados**.
4. **R0:** `python scripts/compare_oof.py <oof_baseline.parquet> <oof_novo.parquet>` — Δ TS-AUC geral
   e por bucket com IC 95% por bootstrap pareado por série (300 réplicas). **Regra de decisão: adotar
   se o IC excluir 0 no agregado ou no bucket-alvo declarado a priori.**
5. **CE6** (`ce6_history_classifier.py`) — deve permanecer ≈0,50 (taxa-base 0,4967). Se subir, há
   vazamento pelo histórico.
6. **Suíte de robustez com `--model`** — usa gates **relativos** (o supervisionado não é calibrado em
   [0,1]). Serve para pegar regressões comportamentais, não para medir desempenho.
7. **Latência** (`make benchmark`) — gate 1500 µs/passo.
8. **XS-SHAP** (`shap_report.py`, coluna `xs_shap`) para entender *onde* o ganho veio. **Nunca use
   `mean|SHAP|` para priorizar** (`MODELO.md` §9.5).

**Escalas de ruído que importam:** σ(TS-AUC) ≈ **0,054** com 100 séries; ≈0,005–0,008 no nível com o
OOF de 10.000; muito menor ainda na **diferença pareada**. Um Δ de 0,004 é irresolvível no held-out
de 100 séries e resolvível no OOF pareado — não confunda os dois instrumentos.

**9. Replicar na OUTRA PARTIÇÃO DE FOLDS antes de acreditar em qualquer efeito de ~0,003**
(`HISTORICO.md` §14.2). Trocar `cfg.seed` 42→43, com tudo o mais igual, move a TS-AUC em **~0,007** —
7× a variância de semente de boosting e 5× a barra. A partição 42 é a **sortuda** (X1-soft: 0,6102 em
42, 0,6029 em 43). O bootstrap pareado do R0 **não** vê essa fonte: ele reamostra séries dentro de uma
partição fixa, então um IC que exclui 0 continua condicionado à partição em que foi medido. Rodar
`scripts/train.py --fold-seed 43` para o braço e comparar contra o baseline da mesma partição custa
~40 min e já derrubou um braço inteiro que parecia ter IC limpo (a costura C5, §14.3): o perfil por
bucket que a motivava **trocou de sinal** entre partições. Vale sobretudo quando a decisão se apoia em
UM bucket.

**A régua local está verificada contra o código oficial** (`scoring.py` da CrunchDAO, lido em 2026-09-30):
por passo, `roc_auc_score` sobre as séries vivas, peso `n_pos·n_neg`, passos com uma classe só pulados,
e `time_online = merged.groupby("id").cumcount()`, ou seja, o **passo online** e não o tempo absoluto.
Sem arredondamento nem float32 no cálculo. É exatamente a nossa `weighted_ts_auc` na grade cheia.

**Onde a métrica realmente paga.** ⚠️ **A tabela de 2026-07-21 media a GRADE ERRADA. Corrigida em
2026-07-25 (A6/A2 de `CAMPANHA_POLIMENTO.md`, `scripts/a2_full_grid_ts_auc.py`).**

O OOF vive na grade com thinning (`configs/default.yaml:thinning`: todos os `t` até 100, passo 2 em
101–400, passo 4 em 401+) — 399 valores de `t`. O board avalia **todos** os ~999 passos. `AUC_t` em
cada passo retido é exato (o thinning descarta passos, nunca séries — verificado: as contagens
`n_pos`/`n_neg` por passo batem com `y_train.parquet` casa a casa), **mas o agregado não é**: cada `t`
retido entra com o próprio `w_t` em vez da massa do bloco que representa, o que subamostra 401+ por
~2× contra 151–400.

| bucket | peso na grade de treino (o que se media) | **peso no BOARD (correto)** | TS-AUC |
|---|---|---|---|
| 1–50 | 8,1% | **3,9%** | 0,5359 |
| 51–150 | 26,6% | **17,5%** | 0,5818 |
| **151–400** | 48,7% | **46,7%** | 0,6312 |
| **401+** | 16,5% | **31,9%** | 0,6592 |

**Consequências, em ordem de gravidade:**

1. **O "câmbio OOF→board de +0,013" nunca existiu — era a grade.** Reponderado, o OOF prevê o placar
   com erro de ~0,002 (V4 `5e42ff5`: 0,6227 vs board 0,6201; incumbente B6+C6: 0,6278 vs 0,6267). O
   nível local passou a ser comparável com o placar, o que a §9.0 dizia ser impossível.
2. **Todo braço cujo ganho vive em `t>400` foi subcreditado por ~2×; todo braço que ganha em `t≤150`,
   supercreditado por ~1,5–2×.** O caso extremo é o E0c (o experimento que fundamentou o veredito de
   "beco sem saída"): o Δ das interações passa de **+0,0039, IC [−0,0015; +0,0100] (inclui 0)** para
   **+0,0104, IC [+0,0056; +0,0157] (exclui 0)**. A conclusão "`g` está saturado" **não sobrevive à
   correção** — capacidade voltou a ser o braço de melhor mecanismo.
3. `t≤50` é ainda menos alavancado do que se pensava (3,9%, não 8,1%): +0,05 ali rende +0,002 no
   agregado. **Declarar `t≤50` como bucket-alvo continua tornando o sucesso indetectável.** O bucket
   de maior peso agregado passa a ser `151–400` + `401+` = **78,6%**.

**`scripts/compare_oof.py` agora usa `--grid full` por default** (`w_mult` de
`evaluation/ts_auc.py:board_grid_multipliers`); `--grid thin` existe só para reauditar números
antigos. `scripts/a2_reaudit.py` reroda os pares que decidiram o projeto nas duas ponderações.

### 5.1 Rastreios baratos — rodar ANTES de gastar um ciclo de build+treino (~50 min)

Três erros custaram fronts inteiras neste projeto; cada um tem um teste de minutos que o pega:

1. **Redundância transversal** (`scripts/xs_redundancy.py`). A TS-AUC só vê ordenação dentro do passo
   (C1): coluna nova com correlação ~1 contra uma existente *dentro do passo* não pode mover nada.
   Previu corretamente o Δ ~0 do braço F1.
2. **Teto de oráculo tem de passar por teste honesto.** "Oráculo" = qualquer estimativa que use o
   rótulo, inclusive indiretamente ("selecionar as linhas pré-quebra"). Padrão: estimar num
   subconjunto de linhas, avaliar em **outro**. A frente de centragem por série mostrou +0,267 no
   oráculo e **−0,0014** no teste honesto — o teto era contabilidade, não sinal.
3. **Contraste pré/pós dentro de uma série tem de controlar `t`.** A quebra divide a série no tempo
   (em `t≤50`: `t` médio 12,4 antes, 35,3 depois) e o score cresce com `t` pela taxa-base. Sem
   controlar, mede-se a tendência: um gap "real" de 0,0462 virou **0,0010** ao ser medido contra a
   seção transversal do mesmo passo.

**E não somar coluna que duplique uma existente.** Duas ampliações independentes regrediram com a
mesma assinatura (V5 −0,0042; F1 −0,0069, ambas piores em `50<t≤150`): largura sem ordenação nova
dilui o sorteio de `feature_fraction=0,8`. Se a versão nova é melhor, ela **substitui**.

**Como rastrear um bloco que ainda não está ligado.** `xs_redundancy.py --extra-blocks <nome>`
instancia blocos fora de `default_blocks()` só para o rastreio — a ordem correta é rastrear, medir,
ligar por último. Nomes disponíveis no dicionário `EXTRA_BLOCKS` do próprio script.

4. **Feature que é razão de estimativas ruidosas não melhora com t.** O `SpectralBlock` na primeira
   versão normalizava o periodograma cru `|z_k|²`, que é exponencial (desvio = média) *em qualquer
   comprimento de série*: a entropia de H0 saía 0,81 ± 0,10 entre séries i.i.d., ruído maior que o
   efeito de um AR(1) com φ=0,6 — e o teste de sinal reprovou. Só promediação explícita (média de
   Welch) resolve; esperar mais dados não. Vale para qualquer feature do tipo `a/(a+b)` com `a`, `b`
   estimados: verificar a **dispersão de H0 entre séries**, não só o valor médio.

**Dois gates não testam o caminho implantado — descoberto em 2026-07-22, ambos abertos.**
1. `scripts/benchmark_latency.py` instancia `StreamScorer(h0, default_blocks(), None, cfg)` — com
   `ensemble=None`, ou seja, cai no `fallback_score` e **nunca mede a predição**. Reporta 1046 µs
   quando o custo real com modelo é ~1405 µs (a predição de 5 folds custa 358 µs). ~30% otimista.
2. `scripts/run_robustness_suite.py` tem `--model default=None`, então `make robustness` roda em
   fallback. Os limiares por cenário (ex.: `t1.control_median_max: 0.25`) foram calibrados nessa
   escala e **não se aplicam a um modelo treinado**: `predict_one` devolve `sigmoid(raw)` sem o
   offset de taxa-base — "resíduo puro, não probabilidade calibrada" (docstring de `predict_one`).
   Medido: o próprio artefato V4 de produção dá controle 0,637 contra limiar 0,25. Rodar a suíte com
   `--model` produz FAIL em tudo e **não é sinal de regressão**.

Consequência prática: nenhum dos dois é um gate do artefato. O que de fato exercita o caminho real é
`scripts/submission_smoke_test.py`. Corrigir os dois é trabalho pendente; até lá, não interpretar
`make benchmark`/`make robustness` como validação do modelo empacotado.

**Um parquet não carrega sua proveniência — confira as COLUNAS antes de usá-lo como baseline.**
`data/processed/train_rows.parquet` é o dataset do **V5** (178 features, com BOCPD, sem L-momentos),
não o do V4: a reversão do V5 restaurou o modelo e o `resources/`, não o dataset. Assumir pelo nome
custou duas horas e uma conclusão errada sobre ruído de semente (BACKLOG_TSAUC.md, ERRATA de
2026-07-22). Uma linha — comparar o conjunto de colunas com o que o modelo alega ter — teria pego.

**MEDIDO: a régua tem dp de 0,0041 que o bootstrap pareado não vê.** Quatro V4 legítimos (mesmas 183
features, mesmos dados, mesmos folds, só `boost_seed`) dão 0,6100 · 0,6006 · 0,6020 · 0,6030 — média
**0,6039**, não os 0,6100 do artefato histórico, que é o maior dos quatro. O bootstrap reamostra
séries e trata as predições como fixas; é cego à variância do booster. EP da diferença entre dois
modelos de UMA semente = 0,0058, o que torna **inconclusivas** as três rejeições do projeto
(F1 −0,0069 = 1,2 EP; V5 −0,0042 = 0,7 EP; F2 −0,0024 = 0,4 EP).
**Nenhum braço se decide contra um sorteio único**: K sementes por lado (`train.py --boost-seed`,
que muda o sorteio SEM tocar nos folds), comparar as OOF médias (`scripts/avg_oof.py`), mesmo K dos
dois lados, e `scripts/seed_spread.py` para ver as duas distribuições. Com K=4 a barra de vitória a
2 EP é Δ ≥ 0,0058. Causa mecânica provável: o nº de árvores por fold varia 51→103 entre sementes
(early stopping sobre logloss). Ver docs/BACKLOG_TSAUC.md, "O nulo da regra de decisão".

**OOF e placar não estão na mesma escala — primeiro ponto de calibração (2026-07-22).** A submissão
`5e42ff5` (V4, 183 features, semente única) marcou **0,6201 no leaderboard** contra **0,6100 de
TS-AUC OOF** local: o placar é +0,0101 MAIOR. Provável causa: são bases diferentes e o modelo
submetido treina em 100% dos dados, enquanto cada modelo do OOF vê 80% (GroupKFold). Consequências:
(a) não comparar NÍVEIS entre OOF e placar; (b) o mapeamento de DELTAS entre os dois é desconhecido —
com um ponto só não dá para estimá-lo. Registrar o par (OOF, placar) a cada submissão: dois pontos já
permitem uma primeira leitura de se o OOF sub ou superestima melhorias.

**A TS-AUC é transversal — subamostrar séries muda a métrica.** O mesmo `oof_v4` dá 0,6100 nas 10.000
séries e 0,8098 num subconjunto de 500. Não existe piloto barato por amostragem de séries.

---

## 6. Pendências e discrepâncias abertas

1. ~~`resources/` empacota o V5, que regride contra o V4.~~ **RESOLVIDO:** V5 revertido, `resources/`
   reempacotado com o V4 e a reversão verificada bit-a-bit contra `oof_v4.parquet`
   (`HISTORICO.md` §9). O que restou é um **experimento**, não uma pendência: o V5 juntou BOCPD com a
   poda de L-momentos/`dep_w050`, então o efeito isolado do BOCPD segue não medido — rodar
   V4 + BOCPD (~190 features) responderia.
2. ~~**Não existe log de submissões oficiais**~~ **CRIADO (2026-07-24):**
   `artifacts/reports/submission_log.md` com a âncora `5e42ff5` (board 0,6201 / OOF 0,6100). Registrar o
   par (OOF, placar) a cada submissão; o mapa OOF↔placar segue com um ponto só.

   **X1 ligado (2026-07-24, `HISTORICO.md` §13).** `configs/default.yaml:weights.detectability_mode: soft`
   pondera os positivos por detectabilidade (+0,0040 OOF, IC exclui 0). O **caminho de submissão**
   (`adapter/platform.py` → notebook) computa o mapa `d_i` **inline** (`model/detectability.py`,
   bit-idêntico ao `detectability.csv`: max|Δ|=3,6e-15) — não depende de artefato local. O **offline**
   `scripts/train.py` mantém `--detectability-mode none` por default (para não exigir `detectability.csv`
   no `make train`/`run_all`); passe `--detectability-mode soft` para reproduzir X1 offline (lê o CSV do
   F0.b). Desligar em produção: `weights.detectability_mode: none`. Só afeta o TREINO; inferência e
   latência intocadas. Notebook regenerado e verificado bit-a-bit.
3. **`drift_slope_abs_max = 1e-4`** reprova T2/T6/T10/T12/T12b por slopes da ordem de −0,0003 a
   −0,0008. Provavelmente limiar apertado demais, não falha real — mas recalibrar é decisão de tuning
   com validação própria.
4. **As 42 features de P1–P4 nunca tiveram XS-SHAP individual medido** — é a medição mais barata
   disponível para decidir o que podar (o V5 já podou `lmom_*` e `dep_*_w050` sem esse número
   publicado).
5. **`_nb_check_main.py`** (~176 kB na raiz) é um artefato de verificação do notebook de submissão,
   não código de produção.

---

## 7. Pegadinhas medidas (custaram tempo real)

- **`| tail -N` em comando de background quebra o streaming de progresso.** `tail` sem `-f` não emite
  nada até o pipe fechar — mesmo com `python -u`/`flush=True`, um job de dezenas de minutos parece
  travado até terminar. Rode sem `| tail` e leia o arquivo de saída incrementalmente.
- **`lambdarank_truncation_level` sem cap trava o treino.** Com `t ≤ 100` todas as ~10.000 séries
  ficam vivas ⇒ maior grupo ~8.000 linhas ⇒ >4 h sem terminar. Use `rank.truncation_level_cap`
  (default 300).
- **Família nova com NaN em t pequeno piora o bucket t≤50** mesmo sendo boa em t alto: com
  `feature_fraction=0,8`, colunas 100%-NaN diluem o sorteio. Toda família nova precisa de variante de
  janela curta ou de transporte de escala do nulo (`state/calibration.py:_null_at`).
- **`ts_auc_by_t` como critério de early stopping regride** (winner's curse com n efetivo ~10⁴). O
  default correto é `logloss`; ambas as métricas são computadas e registradas, só a que decide muda.
- **Objetivo CUSTOM muda o que o `feval` recebe.** Com o objetivo embutido `binary` o LightGBM entrega
  ao `feval` a **probabilidade**; com `params["objective"] = <callable>` ele entrega o **score bruto**.
  Quem escrever um braço de objetivo custom tem de passar `raw_to_prob=True` a `_make_fold_feval`, ou
  o `binary_logloss_diag` trata score bruto como probabilidade (medido: **5,20** contra 0,66 do
  incumbente) — e, como a parada antecipada lê justamente essa métrica, **ela passa a rodar num sinal
  sem sentido**. A `ts_auc` é imune (a sigmoide é monótona, a AUC não muda), o que é útil: dá um
  veredicto independente da regra de parada mesmo com a logloss quebrada (`HISTORICO.md` §14.6).
- **Neste ambiente, wrappers de job em background são mortos no meio** (~5–35 min, sem padrão claro;
  não é suspensão da máquina — foi verificado que não houve sleep). O trabalho real sobrevive se for
  destacado: `nohup <cmd> > log 2>&1 &` e depois acompanhar o log/artefatos. Scripts de campanha devem
  ser **resumíveis** (pular a semente cujo diretório já existe), para que um kill custe no máximo uma
  semente — ver `scripts/run_partb_b6.sh` e `scripts/run_c6_bocpd.sh`.
- **Ablação de fold único não prevê o ciclo completo.** Já custou uma rodada inteira.
- Treino do braço rank: ~16 min (5 folds). Suíte de robustez com 200 seeds: perto de uma hora.
- **UM job pesado por vez — treino concorrente com bootstrap TRAVA (2026-07-25).** Um
  `scripts/train.py` rodando junto com dois `compare_oof.py` (`--n-jobs -1` ⇒ ~13 workers cada, numa
  máquina de 12 CPUs lógicas) caiu para **1,2 núcleo efetivo** e depois **deadlocou**: 51 threads em
  estado *Wait*, 0% de CPU, parado por 1h30 no fold 0. Um fold normal leva **~85 s**. Diagnóstico:
  `Get-CimInstance Win32_Process` + amostrar `UserModeTime` em dois instantes — se o delta é 0, é
  deadlock, não lentidão. Mitigação adotada em `scripts/run_polimento.sh`: `OMP_NUM_THREADS=6`,
  `OMP_WAIT_POLICY=PASSIVE`, saída por arquivo (sem `| tee` no caminho do stderr do `tqdm`), e nunca
  dois jobs pesados simultâneos. Ao medir durante um treino, usar `--n-jobs 3`.
- **O wrapper de background é morto, o processo Python NÃO.** Confirmado nos dois sentidos nesta
  sessão: um `nohup` que se julgou morto continuava vivo (e duplicou a carga, causando a contenção
  acima), e um `train.py` cujo script pai foi morto **terminou o fold e salvou o artefato**. Antes de
  relançar, conferir o que de fato está rodando por `Win32_Process`, não por `ps`, que no Git Bash
  não enxerga os processos Windows.
- **Não editar um `.sh` enquanto ele executa.** O bash relê o arquivo por offset de bytes durante a
  execução; editar no meio faz ele saltar para o lugar errado.
- **`run_arm <nome> <sementes>` nomeia o bag por `wc -w` das sementes** — chamar o mesmo braço duas
  vezes com 2 sementes cada grava `bag2` duas vezes e a segunda SOBRESCREVE a primeira. Aconteceu com
  o `p2_brw` (777+101 sobrescrito por 202+303). Os OOFs por semente sobrevivem, então dá para
  reconstruir; a medição já registrada não é invalidada, mas o arquivo deixa de reproduzi-la.

- **`--set` do `train.py` é `nargs="*"`: dois `--set` na mesma linha, e o segundo SOBRESCREVE o
  primeiro** (2026-09-30). `--set extra_trees=true ... --set feature_fraction=0.5` treina SEM extra-trees.
  Passe todas as chaves num `--set` só (`--set extra_trees=true feature_fraction=0.5`) e confira no log
  as linhas `--set k=v (era ...)`: uma por chave. `scripts/run_braco.sh` já funde os dois.
- **Limite de RAM: ~2,5M linhas com `linear_tree` é o teto em 16 GB** (2026-09-30). Com 4,7M linhas o
  treino chegou a 18 GB de memória privada, trocou para disco (0,5 s de CPU a cada 10 s) e levaria ~3 h.
  Diagnóstico: `PrivateMemorySize64` do processo contra `FreePhysicalMemory`. Para aumento de dados,
  meça em metade das séries.
- **Não encadeie com `;` depois de uma checagem que pode falhar** (2026-09-30): uma asserção falhou e
  o `;` relançou a fila com o bug. Use `&&`.
- **Matar processo por padrão de linha de comando mata o próprio comando** (2026-09-30):
  `Where-Object { $_.CommandLine -match '_fila_v3' } | Stop-Process` casou com o PowerShell e o bash
  que executavam a instrução, e o resto do comando não rodou. Filtre por nome (`bash.exe`) e exclua o
  próprio PID (`$PID`), ou mate por PID explícito.
- **Parquets derivados precisam de `thin_weight`** além de `id`/`t`/`y`: `compute_row_weights` exige a
  coluna, e o V4 (poda) falhou por isso.
- **Monitores de espera em background são encerrados pelo Claude Code sob pressão de memória**, e o
  treino em `nohup` sobrevive. Filas longas devem ser scripts `nohup` resumíveis, conferidos com
  checagens curtas.

- **O Git Bash deste Windows não tem `pgrep`/`pkill`** (2026-09-30): uma fila que esperava com
  `while ... && pgrep -f X; do sleep` saiu na hora (comando inexistente = falso) e subiu um segundo job
  pesado junto com o que estava rodando. Espere por um **arquivo marcador** que o job anterior escreve
  ao terminar (`until grep -q COMPLETA fila.log; do sleep 30; done`).

---

## 8. Custos de referência (nesta máquina)

| Operação | Custo |
|---|---|
| `fit_h0` | **~490 ms/série** em `n_h`=3000 (~180 ms em 1.000, ~780 ms em 5.000). O número antigo desta linha (32–68 ms) é anterior às famílias F3/F4/P1–P4, cujos `history_null_series` dominam o custo — remedido em 2026-07-21. `fit_h0` é hoje ~50% do custo por série, não uma nota de rodapé: uma série típica tem ~500 passos online × ~1 ms |
| `fit_h0` + calibração recursiva (`calibration.recursive_features` não-vazio) | +~155 ms/série, aproximadamente constante em `n_h` (teto de réplicas, `calibration.transient_max_reps`) |
| Inferência completa | 973,8 µs/passo (V4, o empacotado) na medição original; **1062 µs medidos em 2026-07-21 com o conjunto de features IDÊNTICO ao V4** — a diferença é da máquina/sessão, não do código. Gate 1500, folga confortável. Comparar µs/passo entre sessões diferentes não é confiável; para julgar o custo de uma mudança, medir A/B na mesma sessão |
| Blocos caros | L-momentos ~65 µs · MMD ~9 µs · Haar ~4 µs · BOCPD ~30 µs (**fora do pipeline**) |
| Build do dataset | ~9 min (paralelo, `n_jobs`), 2.541.134 linhas |
| Treino (5 folds) | ~10–15 min |
| Comparação R0 | segundos por réplica; 300 réplicas em minutos |
| Suíte de robustez | ~1 h com 200 seeds; CI usa 40 |

---

## 9. Inventário de artefatos

- `artifacts/models/vN/` — ensembles + `feature_schema.json` + `base_rate_curve.json`.
  `oof_vN.parquet` ao lado, alinhado a `train_rows.parquet` (entrada do R0).
  `v1_preV2`, `v1_rank`, `models_baseline_preaudit` = baselines preservados para comparação pareada.
- `artifacts/reports/` — `compare_*.json` (saídas de R0), `shap_*.csv` (com e sem a coluna XS),
  `break_type_census.csv` (o mapa do gerador), `oof_step_response.csv`, `robustness_*.json`,
  `latency_*.json`, `ce6_history_classifier.csv`, logs de build/treino por versão.
- `resources/` — o que a submissão empacota: `boosters.joblib`, `model.joblib`,
  `feature_schema.json`, `base_rate_curve.json`, `fold_evals.json`, `ensemble_meta.json`.
- `data/processed/train_rows.parquet` — dataset de treino (git-ignored, ~650–750 MB, regenerável).

`data/` e `artifacts/` **nunca** vão para o git; são regeneráveis por `make dataset` / `make train`.

**Armadilha de baseline — quatro números quase iguais, um contaminado** (`HISTORICO.md` §14.8):

| número | artefato | o que é |
|---|---|---|
| 0,6018 | — | média das TS-AUC **individuais** das 4 sementes limpas |
| **0,6062** | **`oof_x0_a_bag4`** | **bag das 4 sementes limpas — o baseline CORRETO** (o bagging vale +0,0044) |
| 0,6084 | `oof_b183_bag4` | média de **(42, 101, 202, 777)** — **contém a semente 42**, lê +0,0022 alto |
| 0,6100 | `oof_b183_s42` | semente 42 sozinha — o "0,6100 histórico", contaminado por seleção |

**Não use `oof_b183_bag4` como baseline** apesar do nome sugerir "o V4 de 183 features": o +0,0040 do
X1 está medido contra `oof_x0_a_bag4`; contra o b183 o mesmo braço mediria +0,0018. Incumbente atual
para braços novos: **`oof_x1_soft_bag4` = 0,6102** (partição 42) e **`oof_c3_partB_bag4` = 0,6029**
(partição 43).

---

## 10. Referências a documentos antigos (os 10 arquivos condensados)

Docstrings, comentários e `configs/default.yaml` referenciam os nomes antigos em ~60 arquivos. A
**numeração `§N` foi preservada**, então `plano §3.4`, `§9.0`, `§13.2` continuam resolvendo — só o
nome do arquivo mudou:

| Referência antiga | Onde está agora |
|---|---|
| `docs/PLANO_TECNICO.md §N`, "plano §N", "plano técnico" | `docs/MODELO.md §N` (mesma numeração) |
| `docs/PLANO_REPOSITORIO.md` | `docs/MODELO.md` §15 (esqueleto) + este arquivo (§1–§4) |
| `docs/CONTRACTS.md` | este arquivo, §2 |
| `docs/DIAGNOSTICO_TS_AUC.md` | `docs/HISTORICO.md` §2 (protocolo 3.8 → `HISTORICO.md` §2, "protocolo padrão") |
| `docs/PARECER_AUDITORIA_ONYX.md` (D1–D4, R0–R6) | `docs/HISTORICO.md` §3 |
| `docs/RESULTADOS_ROADMAP_R0_R6.md` | `docs/HISTORICO.md` §4 |
| `docs/PROPOSTA_FEATURES_V2.md` (F1–F6) | `docs/HISTORICO.md` §5 |
| `docs/RESULTADOS_FEATURES_V2.md` (V2/V3) | `docs/HISTORICO.md` §5–§6 |
| `docs/INVESTIGACAO_FALHAS_V3.md` (P1–P4) | `docs/HISTORICO.md` §7 |
| `docs/RESULTADOS_P1_P4.md` (V4) | `docs/HISTORICO.md` §8 |

Os originais permanecem recuperáveis no git (commit `9bc0395` e anteriores). Ao editar um arquivo por
outro motivo, atualize a referência de passagem; não vale um PR mecânico só para isso.

---

## 11. Ao adicionar uma família de features nova

1. Novo arquivo em `state/`, implementando `StateBlock`; docstring apontando para a seção de
   `MODELO.md` e para a rodada de `HISTORICO.md` que a motivou.
2. Escolher o fluxo certo (`e` vs `e_vol`) segundo §2.1 — errar aqui cega ou infla uma família
   inteira.
3. Registrar os parâmetros em `configs/default.yaml`, com comentário do custo medido.
4. Registrar o nulo por série em `state/calibration.py` se a estatística tiver lei de escala conhecida
   (emite a versão `_cal`, custo zero por passo, e é o que dá comparabilidade transversal).
5. Uma linha em `scorer.py:default_blocks`.
6. Teste unitário + **teste de equivalência online × vetorizado** (o caminho vetorizado alimenta o
   nulo de calibração; divergência entre os dois é o risco real dessa arquitetura).
7. Verificar disponibilidade em t pequeno (NaN — ver §7) e medir a latência antes de retreinar.
8. Rodar o protocolo §5 inteiro, com a hipótese registrada antes.
