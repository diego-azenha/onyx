# Notas operacionais para agentes

**Público:** agentes de IA (ou pessoas) que vão modificar este repositório. Nada aqui é necessário
para *entender* o modelo — isso está em [`MODELO.md`](MODELO.md) — nem para saber o que já foi
tentado — isso está em [`HISTORICO.md`](HISTORICO.md). Aqui ficam invariantes, contratos, comandos,
inventário de artefatos, pegadinhas medidas e pendências abertas.

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

---

## 6. Pendências e discrepâncias abertas

1. ~~`resources/` empacota o V5, que regride contra o V4.~~ **RESOLVIDO:** V5 revertido, `resources/`
   reempacotado com o V4 e a reversão verificada bit-a-bit contra `oof_v4.parquet`
   (`HISTORICO.md` §9). O que restou é um **experimento**, não uma pendência: o V5 juntou BOCPD com a
   poda de L-momentos/`dep_w050`, então o efeito isolado do BOCPD segue não medido — rodar
   V4 + BOCPD (~190 features) responderia.
2. **Não existe log de submissões oficiais** (`artifacts/reports/submission_log.md` está previsto no
   plano e ausente). A âncora oficial (cláusula 3 de §9.0) nunca foi exercida de forma registrada.
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
- **Ablação de fold único não prevê o ciclo completo.** Já custou uma rodada inteira.
- Treino do braço rank: ~16 min (5 folds). Suíte de robustez com 200 seeds: perto de uma hora.

---

## 8. Custos de referência (nesta máquina)

| Operação | Custo |
|---|---|
| `fit_h0` | 32–68 ms/série (uma vez) |
| Inferência completa | 973,8 µs/passo (V4, o empacotado); gate 1500 |
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
