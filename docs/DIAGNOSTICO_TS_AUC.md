# Diagnóstico da TS-AUC local no structural-break-rt: hipóteses, método, resultados e uma
# intervenção revertida

## Resumo

Uma medição local de TS-AUC de 0,5244 (fórmula do quickstarter oficial, aplicada ao conjunto de
teste reduzido de 100 séries) disparou esta investigação. Usando reamostragem, decomposição por
tempo, um classificador de controle (CE6), atribuição via SHAP nativo, ablação direta e uma
suíte de cenários sintéticos com verdade conhecida (T1-T13), estabelecemos que: (i) o alarme
inicial era majoritariamente ruído amostral — a habilidade real do modelo, medida corretamente,
é ~0,60 de TS-AUC out-of-fold; (ii) essa habilidade é modesta e concentrada em t>150; (iii) o
calibrador LightGBM aloca 64% do seu gain a features de contexto por série sem retorno preditivo
líquido, às custas de detectores esparsos (bayes, conformal); (iv) o motor sequencial determinístico
tem uma lacuna comportamental concreta e localizada — falso positivo por clustering de
volatilidade tipo GARCH; e (v) o calibrador supervisionado tem uma incompatibilidade de escala
pré-existente e estrutural com os limiares absolutos da suíte de robustez, que antecede qualquer
mudança feita nesta investigação. A partir de (iii)-(iv), desenhamos e implementamos três
intervenções pontuais e bem fundamentadas. Um experimento de validação controlado — comparando,
ponta a ponta, um modelo treinado com as três intervenções contra um baseline treinado nos
mesmos dados sem elas — mostrou que **nenhuma das três entregou ganho líquido na métrica que
importa** (TS-AUC out-of-fold caiu de 0,5996 para 0,5961) e que uma delas introduziu uma
regressão real e explicável num gate de robustez. As três foram revertidas. A lição central: os
mecanismos identificados são reais, mas atacá-los via ajustes pontuais de feature/hiperparâmetro
não foi suficiente — o gargalo mais estrutural (o descasamento de escala do calibrador) permanece
como o candidato de maior potencial para as próximas iterações.

**Escopo e uma ressalva permanente (plano §9.0):** nada neste documento propõe ou reporta uma
TS-AUC local como substituto do score oficial da competição para decisões de gate — isso é uma
decisão de design deliberada e documentada do projeto (`README.md`, `docs/PLANO_TECNICO.md` §9.0).
Todo número aqui é diagnóstico: aponta onde o pipeline está fraco e por quê, nunca "quanto o
modelo vai tirar no leaderboard".

---

## 1. Introdução e contexto

O `structural-break-rt` implementa um detector causal de quebra estrutural em tempo real para o
ADIA Lab Structural Break Challenge (CrunchDAO). A arquitetura é: um motor sequencial de
estatísticas (whitening AR, banco de CUSUM, filtro bayesiano de changepoint único, martingales
conformais, janelas/EWMAs — `src/sbrt/state/*`) que alimenta um calibrador LightGBM (ensemble de
5 folds, `src/sbrt/model/train.py`) treinado no rótulo por passo `y_t = 1{tau<=t}`. A métrica
oficial da competição, TS-AUC, é a AUC transversal calculada em cada passo de tempo online `t`
(agrupando por `id`), combinada como média ponderada por `n_pos*n_neg` em cada `t`.

O gatilho desta investigação foi um número: rodando a mesma fórmula de TS-AUC do
`quickstarter_notebook.ipynb` (célula "Computing TS-AUC locally") via `scripts/local_ts_auc.py`
sobre `prediction/prediction.parquet` (saída de `crunch test`) contra o rótulo local reduzido
`data/y_test.reduced.parquet`, o resultado foi **0,5244** — perigosamente perto de 0,5 (aleatório).
Isso motivou a pergunta que estrutura este documento: **por que a TS-AUC local está tão baixa, e
em que direção seguir para melhorá-la?**

---

## 2. Hipóteses iniciais

Antes de qualquer medição adicional, formulamos cinco hipóteses candidatas para explicar o número
baixo, não mutuamente exclusivas:

- **H1 (ruído amostral):** o conjunto de teste reduzido tem só 100 séries; a TS-AUC pondera AUCs
  por passo com poucos positivos/negativos, então o estimador tem variância alta nessa escala. O
  número observado pode não refletir a habilidade real do modelo.
- **H2 (habilidade real modesta):** mesmo livre de ruído, o modelo pode simplesmente ter pouca
  habilidade discriminativa — e essa fraqueza pode estar concentrada em alguma faixa de tempo
  online, já que a métrica pondera diferente t's de forma desigual.
- **H3 (feature degenerada por construção):** alguma família de features pode estar quebrada —
  por exemplo, o whitening AR poderia estar absorvendo o sinal de deslocamento de média antes de
  ele chegar aos CUSUMs, tornando essas features inúteis por design, não por fraqueza real do
  sinal.
- **H4 (desperdício de capacidade do calibrador):** o LightGBM pode estar alocando gain a
  features que não geram poder preditivo líquido, às custas de features que sim geram — um
  problema de alocação, não de sinal disponível.
- **H5 (diferença de composição):** o conjunto de teste reduzido pode ter uma distribuição de
  tipos de quebra ou taxa de quebra diferente do treino, explicando parte do gap.

Uma sexta linha de investigação — se o **motor sequencial em si** (não só o calibrador) responde
corretamente a cenários canônicos de quebra — não fazia parte do conjunto inicial de hipóteses,
mas emergiu naturalmente como extensão de H3/H4 (seção 4.5) e acabou sendo a que revelou os
achados mais acionáveis.

---

## 3. Metodologia

Cada hipótese foi testada com uma ferramenta específica, nesta ordem:

**3.1 Recomputação controlada da TS-AUC local.** `scripts/local_ts_auc.py` replica exatamente a
célula do quickstarter: agrupa por `id`, deriva o passo de tempo online via `cumcount()`, calcula
`roc_auc_score` por passo e pondera por `n_pos*n_neg`.

**3.2 Estimação da distribuição amostral do estimador (bootstrap).** Para testar H1, calculamos a
mesma métrica sobre as predições *out-of-fold* (OOF) do treino — 10.000 séries, sem vazamento,
via `GroupKFold` — e então reamostramos repetidamente subconjuntos de 100 `id`s desse OOF,
recalculando a métrica em cada subconjunto para estimar empiricamente a variância do estimador no
tamanho do conjunto de teste reduzido.

**3.3 Decomposição por bucket de tempo online.** `scripts/oof_ts_auc_by_bucket.py` (criado nesta
investigação) recorta a TS-AUC OOF em quatro faixas de `t` (≤50, 50-150, 150-400, >400) — os
mesmos cortes usados em `src/sbrt/evaluation/diagnostics.py::score_distribution_report` — e
reporta a AUC ponderada e a fração do peso total da métrica que cada faixa carrega, para testar H2.

**3.4 CE6 — classificador só-histórico.** `scripts/ce6_history_classifier.py` (já existente no
repositório, nunca executado antes desta investigação) treina uma regressão logística usando
*apenas* as 19 estatísticas do ajuste H0 (nenhum ponto online) para prever se a série terá
quebra em algum momento. Se essa AUC bate a taxa-base, o histórico vaza informação sobre quebra
futura e `meta_h0_*` merece ser efeito principal; se não, qualquer uso de `meta_h0_*` como efeito
principal é presumivelmente ruído. Testa a componente de vazamento de H4.

**3.5 Atribuição via SHAP nativo do LightGBM.** Usamos `booster.predict(X, pred_contrib=True)`
(Tree SHAP exato, sem a dependência `shap`) sobre uma amostra estratificada por bucket de `t`
(60.000 linhas), agregando `|SHAP|` médio por família de features. Isso complementa a importância
por *gain* (que mede o quanto uma feature reduz a perda quando usada num split, mas pode ser
inflada por poucos splits de alto impacto) com uma medida da contribuição real média por predição
individual — testa se H4 é sobre *dominância de gain* ou *contribuição real*.

**3.6 Ablação direta.** Para isolar causalidade (não só correlação), retreinamos o LightGBM no
mesmo fold, mesmos dados, pesos e seed, com e sem as features candidatas (`meta_h0_*`), e
comparamos logloss de validação e redistribuição de gain entre famílias.

**3.7 Suíte de robustez sintética T1-T13.** `scripts/run_robustness_suite.py` (já existente,
nunca executado com o modelo treinado antes desta investigação) gera 13 cenários sintéticos
(mais duas variantes T5b/T12b) com verdade conhecida — quebra bem cedo, bem tarde, sutil,
abrupta, de variância, de forma/cauda, sem quebra com clustering GARCH, outliers isolados sem
quebra, série mínima, estabilidade numérica, excursão transitória — cada um com um gate
comportamental (mediana de score, gap contra controle, decaimento) definido em
`configs/default.yaml`. Rodamos primeiro só com o motor determinístico congelado
(`fallback_score`, sem `--model`) e depois — pela primeira vez nesta linha de investigação — com
o calibrador supervisionado (`--model artifacts/models/v1`).

**3.8 Experimento de intervenção controlado.** Depois de implementar três alterações motivadas
pelos achados de 3.4-3.7, testamos rigorosamente seu efeito líquido: treinamos um modelo
"baseline" nos *mesmos dados* (`data/processed/train_rows.parquet`), excluindo exatamente as três
alterações, e comparamos ponta a ponta contra o modelo com as três alterações — mesma suíte de
robustez (`--model`), mesma decomposição de TS-AUC OOF por bucket.

---

## 4. Resultados

### 4.1 H1 confirmada — o alarme inicial é majoritariamente ruído amostral

A TS-AUC medida corretamente (OOF, 10.000 séries, sem vazamento) é **0,601** — bem acima do 0,5244
observado no teste reduzido. Reamostrando 30 subconjuntos de 100 `id`s do OOF: média 0,600,
desvio-padrão **0,054**, percentil 5 = **0,5254**, mínimo 0,5119. O valor observado no teste
(0,5244) coincide quase exatamente com o percentil ~5 da distribuição amostral do próprio modelo
medida em 100 séries. Com ~8-15 positivos e ~20-92 negativos por passo no teste reduzido, e os
passos fortemente correlacionados entre si (mesmas séries vivas ao longo do tempo), o `n` efetivo
da amostra é da ordem de ~100, não das dezenas de milhares de observações-passo nominais.

**Conclusão prática:** qualquer leitura do número local sem uma barra de erro de ±0,05-0,1 é
ruído — reforça, com evidência quantitativa, a decisão de design do projeto (§9.0) de nunca tratar
TS-AUC local como substituto do score oficial.

### 4.2 H2 confirmada — a habilidade real é modesta e concentrada em t≤150

Decomposição da TS-AUC OOF por bucket de tempo online:

| bucket de t | AUC ponderada | peso na métrica |
|---|---|---|
| t ≤ 50 | 0,530 | 8,1% |
| 50 < t ≤ 150 | 0,568 | 26,6% |
| 150 < t ≤ 400 | 0,617 | 48,7% |
| t > 400 | 0,641 | 16,5% |

Aproximadamente 35% do peso da métrica cai em t≤150, faixa em que o modelo mal supera o acaso
(0,53-0,57). Alinhando pelo `tau_index` verdadeiro no teste reduzido: nos 50 `id`s com quebra e
≥5 passos pré/pós, a média do score sobe apenas **+0,045** em média pós-quebra (mediana +0,004;
só 64% dos `id`s sobem) — direção certa, magnitude pequena, consistente com uma AUC ~0,60, não
com um pipeline quebrado.

### 4.3 H3 refutada — as features "mortas" não são degeneradas por construção

Das 80 features originais, 17 tinham gain zero no LightGBM — inicialmente suspeitas de estarem
quebradas pelo whitening. A investigação directa das distribuições (por `y` e bucket de `t`)
mostrou o contrário:

- **CUSUMs de média com threshold alto (`cusum_mean_*_d050/d100`)** não são degenerados — têm
  variância real (máximos de 66 a 285, só 32-55% de zeros exatos). O gain zero vem de
  **redundância informacional**: `cusum_mean_*_d025` (threshold mais sensível) já domina (gain
  730.643/154.000) porque a soma acumulada mais sensível subsume a informação dos thresholds
  maiores.
- **Toda a família `accum_*mean_z`** (qualquer janela, curta ou longa, incluindo `welford_mean_z`
  que tem gain altíssimo) tem correlação **linear** com `y` próxima de zero — esperado, já que
  quebras têm direção +/- que se cancela na correlação simples. Usando `|mean_z|` (magnitude,
  direção-agnóstica) a correlação sobe, mas continua **estruturalmente fraca** (0,003 a 0,037 em
  qualquer janela), sempre bem abaixo das features de variância (`accum_window_var_ln_*`, `bayes_lo_*`,
  correlação 0,06-0,09). **Conclusão: o sinal de deslocamento de média é mais fraco que o de
  variância neste problema, em qualquer janela testada — propriedade do problema, não bug de
  construção.**
- `warmup_min_n=5` gera só 8,14% de NaN em t≤50 — não é o gargalo.

### 4.4 H4 confirmada, com uma correção de diagnóstico via SHAP

**CE6 (3.4):** treinado sobre 10.000 séries, 5-fold CV, usando só as 19 estatísticas do histórico
(H0) — AUC out-of-fold = **0,5067** (taxa-base 0,4967). O histórico sozinho não prevê se a série
vai quebrar; não há vazamento do gerador. Qualquer uso de `meta_h0_*` como efeito principal
(não interação) é, a priori, ruído estatístico.

**Gain vs. SHAP (3.5):** a importância por *gain* atribuía 64,3% do total a `meta_h0_*` (8
features constantes por série); o `|SHAP|` médio real, numa amostra estratificada por bucket de
`t`, atribuía só **34,3%**:

| família | gain (%) | \|SHAP\| médio (%) |
|---|---|---|
| meta_h0 | 64,3 | 34,3 |
| accum | 20,1 | 33,0 |
| cusum | 11,3 | 21,6 |
| hedge | ~1 | 5,0 |
| bayes | 2,1 | 3,4 |
| conformal | 1,1 | 2,2 |

`cusum` sobe de 17,5% (t≤50) para 23,9% (t>400) por bucket — consistente com acumulação de
evidência ao longo do tempo; `bayes`/`conformal` ficam baixos e estáveis (3-4%) em todos os
buckets. **O gain sozinho superestimava a dominância de `meta_h0_*`** — o calibrador usa mais
`accum`/`cusum` na prática do que a métrica de gain sugeria.

**Ablação direta (3.6, fold único):** removendo as 8 features `meta_h0_*` do treino no mesmo
fold/dados/seed, o logloss de validação **melhora** (0,52765 vs. 0,52817 com elas) e o treino usa
mais árvores antes de parar (74 vs. 69) — a presença de `meta_h0_*` **rouba gain** de bayes
(+83%: 65.558→120.123) e conformal (+168%: 29.307→78.448) sem ganho líquido de logloss. Este
resultado de fold único motivou a Intervenção 2 (seção 5), mas **não se confirmou no experimento
de validação completo** (seção 6) — ver a ressalva metodológica na Discussão.

### 4.5 Achado emergente — o motor sequencial tem lacunas comportamentais concretas

Rodando a suíte de robustez (3.7) sobre o motor determinístico congelado (sem calibrador), **4 de
15 gates falham**: T3, T6, T8, T9.

| gate | cenário | limiar | medido |
|---|---|---|---|
| T3 | shift sutil de média (+0,15σ, τ=200) | gap mediana ≥ 0,15 em τ+200 | 0,117 |
| **T6** | GARCH(1,1) sem quebra (falso positivo por *vol-clustering*) | média final ≤ 0,40 | **0,807** (2× o limite) |
| T8 | mudança pura de forma/cauda (τ=200) | gap mediana ≥ 0,15 em τ+200 | 0,107 |
| T9 | outliers isolados sem quebra | média final ≤ 0,35 | 0,461 |

T4/T5/T5b (quebras abruptas de média/variância) passam com folga; T1/T2/T10-T13 também passam.

**T6 é o mais grave e o único com mecanismo totalmente identificado.** Investigação do código
(`src/sbrt/state/scorer.py:59-66`, `accumulators.py`, `cusum.py`) mostra que a trava
`vol_adjust` (§3.4/CE2 do plano) — que existe para não deixar o próprio ajuste de volatilidade
mascarar quebras reais de variância — protege os fluxos de **média/sinal/dependência**, mas
**nunca chega à família de variância** (`cusum_var_up_r150/r250`, `accum_window_var_ln_*`), que
sempre consome o fluxo `e` congelado (escala global fixa de H0). Durante um cluster GARCH, mesmo
sem quebra real, `e²` fica persistentemente elevado nessa escala fixa, o CUSUM de variância
acumula (ratchet, nunca decresce sozinho) e o score sobe — no fallback, isso alimenta diretamente
`max(banco_CUSUM)` (`src/sbrt/model/fallback.py:41-46`), levando o score a 0,807.

### 4.6 As três intervenções desenhadas e implementadas

A partir de 4.4 e 4.5, desenhamos três alterações — uma por causa raiz identificada:

1. **[Motor] CUSUM de variância vol-ajustado como canal adicional** (`src/sbrt/state/cusum.py`)
   — `cusum_var_up_vol_r150/r250`, calculado sobre `e_vol` em paralelo ao `cusum_var_up_r150/r250`
   existente (mantido intacto, para não reabrir T5/T5b). Alvo: T6.
2. **[Calibrador] Exclusão de `meta_h0_*` do treino como efeito principal**
   (`src/sbrt/model/train.py`) — respaldada pela ablação de fold único (4.4).
3. **[Motor] CUSUM de média mais sensível, δ=0,15** (`configs/default.yaml`) — o único threshold
   de média com gain real era o mais sensível já configurado (δ=0,25); alvo: T3.

Todas as três foram implementadas com sucesso, testes unitários ajustados e passando (54/54), e
validadas sintaticamente antes de qualquer execução pesada.

### 4.7 Validação end-to-end: nenhuma das três entrega ganho líquido, e uma regride

A suíte completa (`build_dataset` → `train` → `diagnose` → `robustness --model`) foi executada
com as três alterações aplicadas. Resultado da suíte de robustez com o **calibrador
supervisionado** — pela primeira vez testado nesta linha de investigação: **10 de 15 gates
falham**, muito mais que os 4 observados testando só o motor fallback (4.5).

Para não confundir "regressão causada pelas três alterações" com "fragilidade pré-existente do
calibrador supervisionado nunca antes testada", treinamos um modelo **baseline** nos mesmos dados
(`data/processed/train_rows.parquet`, já com as colunas novas geradas), revertendo exatamente as
três alterações antes do treino (sem rebuild de dataset), e rodamos a mesma suíte `--model` nele.

**Resultado 1 — a maior parte da quebra de gates é pré-existente.** O baseline (sem nenhuma das
três alterações) **também falha 9 de 15 gates** (T1, T2, T3, T4, T6, T7, T8, T9, T10) — o mesmo
padrão, com magnitudes próximas, observado com as alterações aplicadas:

| gate | baseline | com as 3 alterações |
|---|---|---|
| T1-T12b (14 gates) | idêntico (mesmo PASS/FAIL, magnitudes próximas) | idêntico |
| **T13** | **PASS** (decay 0,270) | **FAIL** (decay 0,116) |

A causa é estrutural e documentada em `src/sbrt/model/predict.py:59-66`: `predict_one` devolve
deliberadamente só o **resíduo** (sem readicionar o offset de taxa-base) — uma decisão já tomada
antes desta investigação especificamente porque readicionar "piorou a suíte de robustez em
T6/T9/T10/T12/T12b". O score do calibrador supervisionado nunca se aproxima de 0 ou 1 como os
gates fixos (calibrados pensando numa escala tipo a do fallback) esperam. Prova direta: em T4
(quebra abrupta), inspecionado via SHAP nativo sem retreinar, o modelo reage **corretamente e
rápido** (score sobe de ~0,46 para ~0,67 poucos passos após a quebra; controle fica estável em
~0,45) — a discriminação relativa existe, mas o piso do score nunca cai perto de 0, então o gate
absoluto (`median_min: 0,75`, `control_max: 0,20`) falha mesmo com o modelo "funcionando" no
sentido relativo.

**Resultado 2 — só a Intervenção 3 causou uma regressão real, com mecanismo limpo.** O gate T13
(`src/sbrt/robustness/generators.py:188-193`) testa uma excursão *transitória* de média (+1,0σ
por 60 passos, sem quebra permanente) — o gate exige que o score **decaia** depois (mediana em
t=260 menos mediana em t=600 ≥ 0,15). O CUSUM de média mais sensível (δ=0,15) acumula mais
evidência durante a excursão e, pelo mecanismo de *ratchet* (`max(0, ...)`), demora mais para
zerar — o score fica "grudado" além do necessário.

**Resultado 3 — na métrica que importa, as três alterações juntas pioraram, não melhoraram:**

| | baseline (sem as 3 alterações) | com as 3 alterações |
|---|---|---|
| TS-AUC OOF geral | 0,5996 | 0,5961 |
| t≤50 | 0,5287 | 0,5217 |
| 50<t≤150 | 0,5663 | 0,5669 |
| 150<t≤400 | 0,6151 | 0,6132 |
| t>400 | 0,6425 | 0,6292 |

Queda em quase todo bucket, mais visível em t>400 (0,6425→0,6292) — exatamente onde a
Intervenção 1 deveria ajudar (T6 também não melhorou de forma líquida: 0,521 baseline vs. 0,549
com as alterações). A ablação de fold único (4.4), que motivou a Intervenção 2, **não se
confirmou no retreino completo de 5 folds** combinado com as outras duas alterações.

**Decisão tomada:** as três alterações foram revertidas por completo (`configs/default.yaml`,
`src/sbrt/state/cusum.py`, `src/sbrt/model/train.py` e os testes associados voltaram ao estado
anterior a esta rodada de intervenção).

---

## 5. Discussão

**O que os dados sustentam com confiança:** o alarme original (H1) e a habilidade real modesta
(H2) estão bem estabelecidos por evidência direta e quantificável (reamostragem, decomposição por
bucket). A refutação de H3 (features de média não são degeneradas, são estruturalmente mais
fracas que as de variância) também é sólida. A dominância de `meta_h0_*` no *gain* é real, mas
menos severa na contribuição real (SHAP) do que parecia — e mais importante: **o efeito de
removê-la não generalizou de um fold único para o retreino completo.**

**Lição metodológica principal desta investigação:** um resultado de ablação em fold único, por
mais limpo e bem explicado que pareça mecanisticamente, **não é evidência suficiente** para
prever o efeito de uma mudança no ciclo completo (5 folds, combinado com outras mudanças,
avaliado contra a suíte de robustez e a TS-AUC OOF completa). As três intervenções desta rodada
eram, individualmente, bem fundamentadas — cada uma atacava uma causa raiz real e identificada
com evidência direta — mas nenhuma sobreviveu à validação rigorosa ponta a ponta.

**O achado mais estruturalmente importante não estava nas hipóteses originais:** o calibrador
supervisionado tem uma incompatibilidade de escala com os gates de robustez que **antecede
qualquer coisa feita nesta investigação** — 9 de 15 gates já falhavam antes de qualquer alteração.
Isso não invalida o pipeline (a discriminação relativa existe, como mostrado em T4), mas indica
que o "resíduo puro" que `predict_one` devolve não é diretamente comparável, em escala absoluta,
ao que os gates (ou intuições sobre "score alto = quebra") esperam. Esse descasamento é
provavelmente uma explicação parcial de por que a TS-AUC OOF (que **é** invariante à escala por
construção — invariância C1, plano técnico §1.2 — já que a métrica é uma AUC transversal, não um
limiar absoluto) fica em ~0,60 mesmo quando o motor parece reagir corretamente a eventos como T4:
o modelo pode estar discriminando corretamente em *ranking* mas de forma fraca em *magnitude*,
e/ou a suíte de robustez (limiares absolutos) simplesmente não é a ferramenta certa para avaliar
esse tipo de calibrador — duas explicações não mutuamente exclusivas que esta investigação não
teve como separar completamente.

---

## 6. Próximos passos recomendados

Em ordem de prioridade, dado tudo o que foi aprendido:

1. **Investigar a incompatibilidade de escala do `predict_one`.** É o achado de maior potencial
   e o único ainda não endereçado por nenhuma tentativa desta investigação. Dois caminhos não
   mutuamente exclusivos: (a) recalibração pós-hoc do resíduo (ex.: Platt scaling ou isotonic
   sobre o OOF, mapeando o resíduo para uma escala mais próxima de [0,1] sem reintroduzir o
   problema que motivou não readicionar o offset); (b) redesenhar os gates de robustez para serem
   **relativos** (separação break-vs-controle, como já fazem T1/T3/T4/T5/T7/T8) em vez de
   absolutos (T2/T6/T9/T10/T13), já que o design atual do calibrador não parece compatível com
   limiares fixos por construção.
2. **Testar a regularização (`min_data_in_leaf` 200→50, `lambda_l2` 5→1) via ciclo completo desta
   vez**, não ablação de fold único — dado que a ablação de fold único para `meta_h0_*` não
   generalizou, qualquer hipótese sobre realocação de capacidade do calibrador precisa ser
   validada com o protocolo completo da seção 3.8 antes de ser adotada.
3. **Adotar o protocolo de validação da seção 3.8 (baseline controlado + suíte de robustez
   `--model` + TS-AUC OOF por bucket) como prática padrão** para qualquer mudança futura de
   feature ou hiperparâmetro — é o que permitiu distinguir, nesta rodada, regressão real (T13) de
   fragilidade pré-existente (os outros 9 gates), e teria evitado a falsa esperança gerada pela
   ablação de fold único.
4. **Rodar CE6 e a suíte de robustez em cada iteração futura**, não só quando um problema já é
   suspeito — ambos existiam no repositório sem nunca terem sido executados antes desta
   investigação, e a suíte de robustez com `--model` revelou o achado estruturalmente mais
   importante de toda a rodada.

---

## 7. Conclusão

O número que motivou esta investigação (TS-AUC local 0,5244) era, em grande parte, ruído de
amostra pequena — a habilidade real do modelo, medida corretamente, é ~0,60, modesta e
concentrada nos passos iniciais de cada série. Identificamos, com evidência direta, três
mecanismos plausíveis de melhoria (variância cega a *vol-clustering*, calibrador dominado por
contexto por série, sinal de média subexplorado) e desenhamos uma intervenção para cada um. Nenhuma
sobreviveu à validação rigorosa: o ganho na TS-AUC OOF foi nulo a levemente negativo, e uma das
três intervenções introduziu uma regressão real. As três foram revertidas. O achado mais valioso
desta rodada não foi nenhuma das três intervenções, mas a descoberta de que **o calibrador
supervisionado tem uma incompatibilidade estrutural e pré-existente de escala com a suíte de
robustez** — um problema mais fundamental do que qualquer ajuste pontual de feature, e o candidato
mais promissor para a próxima iteração.

---

## Apêndice — artefatos produzidos nesta investigação

| artefato | conteúdo |
|---|---|
| `scripts/local_ts_auc.py` | recomputação da TS-AUC local (fórmula do quickstarter) |
| `scripts/oof_ts_auc_by_bucket.py` | decomposição da TS-AUC OOF por bucket de `t` |
| `scripts/run_all.py` | encadeia build_dataset → train → diagnose → robustness → decomposição OOF |
| `artifacts/reports/ce6_history_classifier.csv` | saída do CE6 (classificador só-histórico) |
| `artifacts/reports/shap_feature_importance.csv` | \|SHAP\| médio por feature, amostra estratificada por bucket de `t` |
| `artifacts/reports/robustness.json` | resultado da suíte T1-T13 (motor fallback) |
| `artifacts/models/v1`, `artifacts/reports/*` | modelo e diagnósticos no estado atual (pós-reversão das 3 alterações) |

Todas as três intervenções desta rodada (CUSUM de variância vol-ajustado, exclusão de
`meta_h0_*`, CUSUM de média δ=0,15) foram implementadas, validadas e **revertidas** — o código em
`configs/default.yaml`, `src/sbrt/state/cusum.py` e `src/sbrt/model/train.py` está no estado
anterior a esta investigação, exceto pelo hazard bayesiano adicional (1/50, `bayes.hazards`),
que é anterior a este documento e não fez parte do experimento de intervenção controlado da
seção 4.7 (teve seu próprio teste, também com resultado nulo na TS-AUC OOF, registrado
separadamente na história deste repositório).
