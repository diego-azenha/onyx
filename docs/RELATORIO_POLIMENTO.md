# Relatório da campanha de polimento — a régua estava errada

**Data:** 2026-07-25 · **Repo:** `onyx` · **Insumos:** `CAMPANHA_POLIMENTO.md` (as seis frentes),
`RELATORIO_EXP0.md`, `RELATORIO_CAMPANHA_B_C.md`, `NOTAS_AGENTES.md` §5.
**Posição de partida declarada:** incumbente X1-soft + B6 + C6, board **0,6267**, OOF **0,6139**.

---

## 0. Sumário executivo

A frente A (auditorias de custo ~0, executadas primeiro por desenho) achou **três defeitos reais**,
e o primeiro deles muda o significado de quase todo número já medido neste projeto.

| # | achado | custo de achar | efeito |
|---|---|---|---|
| **A2** | **a TS-AUC OOF era agregada na grade errada** | 1 h | o "câmbio OOF→board de +0,013" não existe; `t>400` vale 31,9% do peso e não 16,5%; **o veredito do E0c se inverte** |
| **A6.1** | o bloco conformal (feature nº 1) compara `e` **clipado** contra um histórico **não clipado** | 1 h | assimetria de cauda em 7,5% das séries, **2,6× mais frequente nas positivas** |
| **A6.2** | o nulo por série das colunas `_cal` é estimado em escala diferente da produção | 30 min | mesmo mecanismo, atinge `mmd_joint_slow_cal` e `accum_window_var_ln_w100_cal` (nº 4 e nº 10 do xs-SHAP) |
| **A3** | o conjunto **implantado** (193 features) ≠ o conjunto **medido** (187) | 1 h | B6 e C6 foram medidos numa base sem `mrep_` e aplicados a uma produção com ele |
| A1 | — (verificado, **está correto**) | 10 min | rótulos são pós-W23; nada a corrigir |
| A5 | anti-predição em `t≤8` é real | 20 min | **vale +0,00004** — não compensa um braço |
| B2 | média em logit vs probabilidade | 15 min | **empate exato** (Δ +0,00001); o OOF representa a produção fielmente |
| **bônus** | `train_rank` **quebrado desde 24/jul** | (achado por `pytest` completo) | `feature_contri_meta: 0.6` no YAML fazia o braço de ranking levantar `UnboundLocalError` em toda chamada |

**E o que os braços renderam, até agora.** Três medidos na partição 42, um único sobrevivente
provisório:

| braço | partição 42 (K=4) | partição 43 (K=4) | estado |
|---|---|---|---|
| **`p3_lintree` (D1)** | **+0,0027 [+0,0002; +0,0049] · exclui 0** | **+0,0005 [−0,0020; +0,0029]** | **positivo nas duas, significativo em uma** |
| `p2_brw` (A3b) | +0,0010 | −0,0018 (K=2) | fechado, abaixo da barra |
| **`p8_a6` (A6)** | **+0,0015 [−0,0008; +0,0036]** (K=2) | — | **promissor, inconclusivo — exige K=4** |
| `p4_cap` (D3) | +0,0012, zero em `t>400` | — | fraco, mas delimita a releitura do E0c |
| `p5_maxbin1023` (D2) | −0,0004; `t≤50` −0,0080 exclui 0 | — | **fechado negativo** |
| `p6_contri04` (C1) | +0,0017 (K=2) | — | gradiente monótono (ver abaixo) |
| `p7_contri08` (C1) | −0,0013 (K=2) | — | idem |
| `p13_floor015` (C2) | −0,0014 (K=2) | — | **fechado negativo** |
| `p13_floor050` (C2) | +0,0009 (K=2) | — | fechado, abaixo da barra |
| `p14_mdl100` (D3) | +0,0008 (K=2) | — | fechado, abaixo da barra |
| `p15_l2_20` (D3) | +0,0003 (K=2) | — | fechado, nulo |

### O veredito do D1, com a precisão que ele merece

**Não é "aprovado" nem "refutado".** É positivo nas duas partições (+0,0027 e +0,0005), significativo
em uma. Média ≈ **+0,0016**, logo acima da barra de 0,0014.

E a trajetória da medição é ela própria a lição:

| medição | Δ | o que eu concluí na hora |
|---|---|---|
| partição 42, K=2 | +0,0025 | "passa pelo bucket-alvo" |
| partição 42, K=4 | +0,0027 | "sobrevive ao K=4, é o resultado da campanha" |
| partição 43, **K=2** | −0,0019 | **"refutado" — e adotei/retratei em cima disso** |
| partição 43, **K=4** | **+0,0005** | o sinal volta a positivo |

O −0,0019 era ruído de semente que o bootstrap pareado não vê (~0,004 na diferença de dois bags K=2).
**Comparar K=4 contra K=2 e reverter pela medição mais fraca foi um erro de padrão probatório**, e o
K=4 na partição 43 mostrou isso com número. Ver `HISTORICO.md` §15.16.

### O par D1+A3b é pior que o D1 sozinho

| bucket | Δ (par − D1) | IC 95% | |
|---|---|---|---|
| geral | −0,0018 | [−0,0063; +0,0019] | |
| **`t>400`** | **−0,0069** | **[−0,0121; −0,0021]** | **exclui 0** |

Acrescentar o A3b piora o bucket de **maior peso** (31,9%). **Quarta não-aditividade do repo**
(V5+mrep, B6+C6 parcial, `mrep_` de +0,0042 → +0,0008, e esta). Se o D1 entrar, entra **sozinho**.

### C1 fecha com um gradiente, que vale mais que um braço

| `feature_contri_meta` | Δ (K=2, partição 42) |
|---|---|
| **0,2** | **+0,0001** · `t≤50` **−0,0117** (IC exclui 0) |
| 0,4 | **+0,0017** |
| 0,6 (atual) | referência |
| 0,8 | −0,0013 |

**O 0,2 fechou a curva (medido em 2026-07-25 11:53).** O gradiente não é monótono até o fim: ele
sobe de 0,8 para 0,4 e depois **volta a zero em 0,2**, com um dano grande e significativo nos passos
frios (−0,0117 em `t≤50`). Isso é o que um ótimo interior parece — e localiza-o **em torno de 0,4**,
não "quanto mais penalizar, melhor". A leitura mecânica bate: penalizar `meta_h0_*` demais tira do
modelo o único condicionador de "que tipo de série é esta" justamente quando ainda não há histórico
para mais nada, que é exatamente o regime `t≤50`.

Monótono na direção "penalizar `meta_h0` **mais**" — exatamente o que o mecanismo do B6 prevê (essas
colunas descrevem a série inteira e viravam quase-intercepto por série, inútil para uma AUC calculada
*dentro* do passo). **Três pontos alinhados com a teoria são evidência melhor que um ponto isolado
com o mesmo Δ**, porque um gradiente monótono é muito mais difícil de produzir por ruído. `contri=0,2`
foi medido para achar onde a curva vira.

### B3 — o menor efeito da campanha, e o mais bem sustentado

| | Δ | IC 95% | |
|---|---|---|---|
| partição 42 | **+0,0008** | **[+0,0002; +0,0013]** | **exclui 0** |
| `150–400` | +0,0009 | [+0,0002; +0,0016] | exclui 0 |
| **partição 43 (bootstrap pareado, 2026-07-25 12:01)** | **+0,00094** | **[+0,00039; +0,00143]** | **exclui 0** |
| `150–400` (partição 43) | +0,0010 | [+0,0004; +0,0017] | exclui 0 |

**O B3 é o primeiro item desta campanha a passar nas DUAS partições com IC excluindo zero** — que é,
literalmente, a condição que o protocolo exige para adoção. Dois detalhes reforçam a leitura:

- O bootstrap pareado na 43 (+0,00094) caiu **em cima** do que a varredura tinha previsto (+0,00094),
  e em cima da partição 42 (+0,0008). Três estimativas independentes, mesma magnitude.
- `t≤50` dá exatamente **0,0000** nas duas partições — o gate tem `min-t 50`, então não toca nada
  abaixo disso. É a assinatura de que o efeito medido é o do gate e não deriva de outra fonte.

O efeito é pequeno (+0,0009, abaixo da barra de 0,0014), mas é **o mais bem sustentado da campanha**:
pós-processo puro, sem retreino, sem feature nova, sem custo de inferência.

Config vencedora pelo **pior das duas partições**: `a_up=1,0 · a_dn=0,2 · gate em t>50`. É uma
**catraca** — o score sobe livre e desce devagar, que é o comportamento certo para um detector de
quebra: evidência acumulada não deve evaporar. E o corte ótimo é **`t>50`, não o `t>150` atual** — o
gate foi calibrado sob os pesos que o A2 corrigiu, e o ótimo se moveu como previsto.

**Por que o IC dele é 4× mais estreito que o de qualquer braço de treino** (largura 0,0011 contra
~0,005): o B3 **não tem ruído de semente**. É uma transformação determinística do mesmo OOF, então o
bootstrap pareado só enfrenta reamostragem de séries — não há retreino nem sorteio de booster. É
exatamente a fonte de variância que derrubou o A3b e quase derrubou o D1, e que aqui simplesmente não
existe.

Isso reordena a leitura da campanha: **o efeito é pequeno, mas é o único medido sem a principal fonte
de erro do projeto**, positivo nas duas partições, e passou por seleção/confirmação em folds
separados.

### Um padrão que emergiu dos dados, não do plano

**Três braços independentes** (`max_bin`, `contri=0,4`, `linear_tree`) **perdem em `t≤50` e ganham em
`t` alto.** Não é coincidência: `t≤50` tem 68% das colunas em NaN de warmup, e qualquer coisa que
aumente a flexibilidade do aprendiz paga variância ali. Como `t≤50` vale 3,9% do board, o trade é
favorável — mas sugere que aplicar essas mudanças **com gate por `t`** pode render mais que qualquer
uma isolada. Hipótese nova, barata de testar, não estava no plano original.

**Dois braços empatados em ~+0,0025 com K=2; um cai pela metade e o outro se mantém.** É a medida
empírica de quanto a triagem K=2 infla neste projeto — e a razão de a regra ser: em K=2, Δ **ordena
braços**, não estima efeitos.

---

## 1. A2 — a métrica local media a grade errada

### O mecanismo

O OOF vive na grade com thinning (`configs/default.yaml:thinning`: todos os `t` até 100, passo 2 em
101–400, passo 4 em 401+) — **399** valores de `t`. O board avalia **todos** os ~999 passos.

A TS-AUC é `Σ_t w_t·AUC_t / Σ_t w_t` com `w_t = n_pos(t)·n_neg(t)`. Duas coisas são diferentes e
foram confundidas:

- **`AUC_t` em cada passo retido é exato.** O thinning descarta *passos*, nunca *séries*, então a
  população viva em cada `t` retido está inteira. Verificado casa a casa contra `y_train.parquet`:
  `max|Δn_pos| = 0`, `max|Δn_neg| = 0`.
- **O agregado não é.** Cada `t` retido entra com o próprio `w_t` em vez da massa do bloco que
  representa. Uma região subamostrada 4× entra com ¼ da massa que o board lhe dá.

### O tamanho

| bucket | peso na grade de treino | **peso no board** | razão |
|---|---|---|---|
| 1–50 | 8,1% | **3,9%** | 0,48 |
| 51–150 | 26,6% | **17,5%** | 0,66 |
| 151–400 | 48,7% | **46,7%** | 0,96 |
| **401+** | 16,5% | **31,9%** | **1,93** |

Medido em três modelos independentes, a TS-AUC agregada sobe **+0,0127 a +0,0139** ao trocar de
grade. Duas verificações de que não é artefato do método:

1. **Reponderação por bloco** (cada `t` retido recebe a massa do bloco, sem interpolar nada) e
   **interpolação** de `AUC_t` na grade cheia concordam em **0,0001** (0,6277 vs 0,6278).
2. O erro de interpolação de `AUC_t`, medido por holdout nos próprios `t` retidos, é 0,0015 por
   passo — e o agregado média ~600 pontos interpolados, então o resíduo no agregado é muito menor.

### O que isso conserta

O log de submissões registrava um "câmbio OOF→placar" de +0,0101 tratado como desconhecido e
possivelmente estrutural (80% dos dados na CV vs 100% na submissão). Ele era **a grade**:

| submissão | OOF (grade de treino) | **OOF (grade do board)** | board | erro |
|---|---|---|---|---|
| `5e42ff5` (V4, semente 42) | 0,6100 | **0,6227** | 0,6201 | +0,0026 |
| incumbente (B6+C6, bag4) | 0,6139 | **0,6278** | 0,6267 | +0,0011 |

**O OOF reponderado prevê o placar com erro de ~0,002 em dois pontos independentes.** A regra da
§9.0 ("nunca comparar níveis entre OOF e placar") era uma consequência do bug, não uma lei.

### O que isso derruba

O `RELATORIO_EXP0.md` concluiu que a construção é um beco sem saída, apoiado em três pilares. O
primeiro — e o único com medição própria — é o E0c: treinar o incumbente com `max_depth=1`
(modelo aditivo puro, zero interações) e medir o que as interações valem. Refeito no par limpo
183-vs-183 (`oof_e0c_stump_s777` vs `oof_b6_contri06_s777`), 300 réplicas:

| grade | Δ atribuível a **todas** as interações | IC 95% | exclui 0 |
|---|---|---|---|
| thinning (como o EXP0 mediu) | +0,0039 | [−0,0015; +0,0100] | **não** |
| **board (correta)** | **+0,0059** | **[+0,0006; +0,0117]** | **sim** |

Os buckets batem casa a casa com o EXP0 (`t≤50` +0,0029, `150–400` +0,0050, **`t>400` +0,0144, IC
exclui 0**) — como tinha de ser: dentro desses três buckets o multiplicador é constante, então só o
agregado se move. É o bucket `t>400`, com peso dobrado, que empurra o agregado para fora do zero.

**A frase decisiva do `RELATORIO_EXP0.md` §2.1** — *"a estrutura de interação de um GBM de 63 folhas e
~1500 árvores é estatisticamente indistinguível de zero no agregado"* — **não sobrevive à correção**.
+0,0059 é 4× a barra de decisão (~0,0014).

**Mas é preciso ler o que isso diz e o que não diz** — e o §3 desta campanha já mediu a diferença.
O E0c compara o incumbente (COM interações) contra o toco (SEM). Um Δ de +0,0059 que exclui 0 diz
que **as interações contribuem +0,0059 para o incumbente**; **não** diz que existe outro +0,0059
esperando em mais capacidade. A consequência correta é que `g` deixa de ser *formalmente fechado* —
o EXP0 o listava como tal —, não que capacidade seja uma mina. O braço `p4_cap`, medido nesta
campanha, é a evidência direta disso: 2,4× mais rodadas renderam +0,0012, com **zero em `t>400`**.

O ponto estrutural é mais amplo que o E0c: **um braço cujo ganho se concentra em `t>400` foi
subcreditado por ~2×, e um que ganha em `t≤150`, supercreditado por ~1,5–2×.** Todas as decisões de
adoção/rejeição do projeto foram tomadas com essa distorção.

### O erro estava no juiz, não no treino — e isso é uma boa notícia

`model/dataset.py:_thinning_keep_and_weight` devolve `thin_weight` = 1 / 2 / 4 nas três regiões, e
`model/weights.py` faz `w = base_w · thin_weight`. **A supervisão sempre teve a massa por `t`
correta**: o modelo foi treinado desde sempre apontando para a ponderação do board.

Só a **avaliação** errava. Consequências práticas:

- **Nenhum modelo precisa ser retreinado por causa do A2.** Os artefatos estão certos; o que está
  errado é o ranking entre eles.
- Mas **toda decisão de adoção/rejeição** foi tomada com o juiz torto, então o *conjunto* de
  escolhas que trouxe o incumbente até aqui é que fica sob suspeita — não os pesos dele.
- Reauditar é barato: só bootstrap sobre OOFs que já existem (`scripts/a2_reaudit.py`).

### O que foi feito

- `evaluation/ts_auc.py:weighted_ts_auc` ganhou `w_mult`, e `board_grid_multipliers` deriva os
  multiplicadores da grade do board direto de `y_train.parquet` (sem depender de feature nenhuma).
  É um multiplicador, não um peso absoluto, para compor corretamente com a reamostragem do bootstrap.
- **`scripts/compare_oof.py` agora usa `--grid full` por default**; `--grid thin` só para reauditar.
- `scripts/a2_full_grid_ts_auc.py` (a medição) e `scripts/a2_reaudit.py` (rerodar os pares que
  decidiram o projeto nas duas ponderações).
- `NOTAS_AGENTES.md` §5 corrigido — era a tabela mais consultada do repo.

---

## 2. A6 — o clip do resíduo, e o nulo que não bate com ele

### A6.1 — o conformal compara escalas diferentes

`whiten_step` devolve `(e_clip, e_raw)` com `e_clip = clip(e_raw, -8, +8)`. De 19 blocos, **só
`lmoments.py` consome `e_raw`**; `accumulators.py` o usa nas duas estatísticas de cauda
(`max_abs_eraw`, `count_exceed99`). Todo o resto vê `e` clipado.

O problema não é o clip em si — é que **a referência do bloco conformal não é clipada**:
`h0.py:151` constrói `e_hist = resid / sigma_e` cru, e `sorted_abs_e_hist` sai daí. Então toda
observação online com `|e_raw| > 8` é ranqueada como se valesse exatamente 8,0 contra um histórico
que preservou os próprios extremos: o p-value satura na massa de cauda do histórico acima de 8, em
vez de cair para o piso `1/(n_h+1)`.

Isso importa porque `conformal_logm_abs` é a **feature nº 1** do modelo (xs-SHAP 0,061;
`conv_share` 0,147 — a maior das duas listas).

**Quanto morde** (medido em `train_rows_bocpd`, 2,54 M linhas):

| | linhas | séries |
|---|---|---|
| `max|e_raw|` já > 8 | 3,51% | **7,50%** |
| > 10 | 1,89% | 4,19% |
| > 15 | 0,28% | 0,95% |

E, o que decide: **por classe, 6,99% das linhas positivas contra 2,64% das negativas — 2,6×.** O
clip está comprimindo justamente o material que separa as classes. Um p-value de rank é robusto por
construção; o clip não lhe compra estabilidade nenhuma.

### A6.2 — o nulo das colunas `_cal` mora em outra escala

Mesmo mecanismo, alcance maior. `state/calibration.py:compute_null_stats` faz o replay dos blocos
reais sobre o histórico para estimar o nulo por série de cada coluna `_cal` (F1) — e recebe
`e_hist` **não clipado**, enquanto a produção calcula a mesma estatística sobre `e` **clipado**.

A estatística observada é comprimida em [−8, 8]; o nulo dela não é. O z-score resultante fica
enviesado exatamente nas séries de cauda pesada — as mesmas 7,5%. Entre as colunas afetadas estão
`mmd_joint_slow_cal` (nº 4 do xs-SHAP) e `accum_window_var_ln_w100_cal` (nº 10).

### A correção, e por que ela é direcional

Não existe um "unclip tudo": o maior `|e_raw|` da base é **15.774**, e um único ponto desses destrói
`accum_welford_var_ln`. A correção coerente é pôr cada estatística na mesma escala do **seu próprio**
nulo:

- **conformal → `e_raw`** (`conformal.use_raw`): é rank-based, a referência é o histórico cru, e o
  clip só custa resolução;
- **replay do nulo → clipado** (`h0.null_clip_match`): as colunas `_cal` são calculadas em produção
  sobre `e` clipado, então o nulo delas tem de ser também.

Ambas entram como flags, `false` por default (defaults byte-a-byte inalterados, verificado: 193
features emitidas, mesma lista).

### MEDIDO em 2026-07-25 (K=2, partição 42, grade do board)

Rebuild feito com `configs/a6.yaml` (idêntico ao default **exceto** as duas flags, verificado por
diff: 4 linhas). Dataset resultante: 2.541.134 linhas × 197 colunas — **mesma forma do baseline**,
então a comparação é pareada linha a linha, sem o problema de grade que o E1 vai ter.

**As duas flags mordem, e mordem onde o mecanismo previa:**

| coluna | difere em | |
|---|---|---|
| `conformal_logm_abs` (**feature nº 1**) | 2,16% das linhas | `use_raw` |
| as 4 colunas `conformal_*` | 0,24% – 2,16% | `use_raw` |
| `accum_window_var_ln_*_cal` | **15,44%** | `null_clip_match` |
| `accum_window_exceed2_frac_*_cal` | **0,00%** | frações limitadas em [0,1] — clipar o histórico não as move |

**Resultado.** TS-AUC 0,62640 → **0,62785**.

| bucket | Δ | IC 95% | exclui 0 | peso |
|---|---|---|---|---|
| **overall** | **+0,0015** | [−0,0008; +0,0036] | **não** | — |
| t≤50 | −0,0022 | [−0,0098; +0,0060] | não | 3,9% |
| 50<t≤150 | +0,0035 | [−0,0019; +0,0084] | não | ~17% |
| 150<t≤400 | +0,0013 | [−0,0014; +0,0038] | não | 46,7% |
| t>400 | +0,0010 | [−0,0014; +0,0037] | não | 31,9% |

**Veredito: promissor e inconclusivo — exige K=4.** O ponto está acima da barra (0,0014) e o IC
inclui zero; as duas coisas são verdade. É exatamente o estado em que o D1 estava antes do K=4, e
esta campanha já se queimou duas vezes lendo K=2 como veredito (A3b caiu de +0,0024 para +0,0010; o
D1 foi adotado e retratado em 24 h). **Não promover por este número.**

Três razões para não arquivar também:

1. **Positivo em 3 dos 4 buckets, cobrindo 95,6% do peso.** O único negativo é `t≤50` (3,9%) — e é
   onde o mecanismo prevê perda, já que o clip existia para estabilizar o regime sem histórico.
2. **É correção de defeito, não busca de hiperparâmetro.** O mecanismo foi verificado
   independentemente do Δ (o clip morde em 7,50% das séries, 2,6× mais nas positivas). A pergunta
   "isso é ruído?" tem uma âncora que um `max_bin` não tem.
3. **No ranking de K=2 da campanha**, +0,0015 fica atrás só de `contri=0,4` (+0,0017) e
   `linear_tree` (+0,0025), à frente de tudo que foi descartado.

**Próximo passo:** K=4 na partição 42 (+2 sementes, ~17 min) e réplica na 43 (~35 min). Não estão na
fila atual.

---

## 3. A3 — o que é medido não é o que é implantado

A submissão **não empacota modelo**: o notebook embute o pipeline e a nuvem chama `train()`
(`adapter/platform.py`), então o que define o conjunto de features implantado é `default_blocks()`.

**(i) Ensemble — correto, sem ganho na mesa.** `platform.train` treina as 4 sementes de
`lightgbm.bag_seeds`, cada uma com 5 folds, e funde os 20 boosters num só (`model/fuse.py`) com
verificação numérica que levanta erro se a fusão não reproduzir a média dos raws. Confere com o
artefato: 1.612 árvores ≈ 20 × 80.

**(ii) `init_score` — descasado desde o R1.** Ver §4.

**(iii) Score em float64 sem arredondamento** — `predict_one` devolve `float(np.mean(preds))`, e o
adaptador emite `float(...)`. Nenhum clip nem `round` no caminho. Correto.

**(iv) `zero_as_missing`** — não é setado, e o default do LightGBM é `false`. Correto: LLR em zero é
informação, não ausência.

**(v) O descasamento.** `default_blocks()` emite **193 features** = V4 (183) + `MultiRep` (6) +
`BOCPD` (4). Mas o incumbente **medido** — `oof_b6c6_joint_bag4`, o 0,6278 — tem **187**: o braço foi
treinado com `--drop-prefix spec_ ord_ mrep_`. Ou seja **o B6 (+0,0036) e o C6 (+0,0014) foram
medidos sobre uma base sem `mrep_` e aplicados a uma produção que o contém.**

Isso não é hipotético neste repo: o precedente documentado em `scorer.py` é literal — *"V5+mrep 181:
os ganhos NÃO SOMAM: juntar piora −0,0024 vs V5, nas 3 sementes. Cada componente ajuda sozinho, o
conjunto não."*

**Medido** (`p1_deploy193`, K=2, grade do board): implantado (193) **0,62718** contra medido (187)
0,62640 — **Δ +0,0008 [−0,0015; +0,0029]**, indistinguível de zero.

- **Tranquilizador:** os números do incumbente **descrevem** o pipeline implantado, dentro do ruído.
  O risco de que os Δ do B6 e do C6 falassem de uma base que a produção não usa **não se
  materializou**.
- **Acionável:** o `mrep_` foi **adotado valendo +0,0042** (IC excluindo 0) sobre a base V4. Sobre a
  base atual vale **+0,0008**. O ganho evaporou — a não-aditividade de novo.

**Isto NÃO é um argumento para podar `mrep_`, e a primeira versão desta seção estava errada ao
sugerir que era.** Três razões, em ordem de peso:

1. **Valor marginal contra o pacote final subestima o componente.** Cada mudança deste projeto foi
   medida contra a base que existia quando entrou. Reavaliar cada uma contra tudo o que veio depois e
   podar as que "não pagam mais" leva a zero por indução — a certa altura a contribuição marginal de
   qualquer peça contra todas as outras é pequena. Isso é **redundância entre contribuições
   correlacionadas**, não inutilidade.
2. **A barra de adoção não é a barra de manutenção.** Acrescentar custa complexidade, latência e
   risco, e por isso precisa se justificar. Manter algo já construído, testado e empacotado não custa
   nada de novo; **remover** custa retreino, reverificação e o risco de a estimativa marginal estar
   errada. Usar o mesmo limiar para as duas operações enviesa tudo para podar.
3. **É um número de K=2.** Esta mesma campanha mediu que Δ em K=2 ordena braços mas não estima
   efeitos — o A3b caiu de +0,0024 para +0,0010 ao ganhar sementes. Usar +0,0008 de K=2 como veredito
   sobre um componente adotado com IC excluindo 0 é aplicar a régua que eu mesmo acabara de desmontar.

O que a medição de fato estabelece é só a leitura (1) acima: **os números do incumbente descrevem o
pipeline implantado**. O `p1_deploy193` era uma auditoria de fidelidade e passou. Nada aqui é
recomendação de mexer no conjunto de features.

---

## 4. A3b — o `init_score` deixou de remover um offset e passou a injetar um

`model/base_rate.py` ajusta `p̂(t)` e usa `logit(p̂(t))` como `init_score`, para o LightGBM aprender
só o resíduo transversal. A curva é ajustada na **contagem crua**.

Só que desde o R1 (`model/weights.py`) o treino usa pesos pareado-consistentes — `w_pos(t) ∝ n_neg(t)`,
`w_neg(t) ∝ n_pos(t)` — que **equalizam as classes dentro de cada passo**. Medido:

| t | taxa de positivos crua | **taxa ponderada** | `init_score` injetado | descasamento |
|---|---|---|---|---|
| 10 | 0,0226 | **0,4947** | −3,833 | **+3,812** |
| 50 | 0,0836 | **0,4986** | −2,426 | +2,421 |
| 150 | 0,1742 | **0,4993** | −1,558 | +1,555 |
| 400 | 0,3123 | **0,4996** | −0,790 | +0,788 |
| 900 | 0,4690 | **0,4997** | −0,117 | +0,115 |

A taxa-base do problema que o LightGBM de fato otimiza é **0,5 em todo `t ≥ 10`**, então o
`init_score` correto seria ≈ 0. O que é injetado vai a −3,83. **Descasamento médio de +1,22 em
log-odds, máximo +3,81.**

O R1 invalidou a premissa do A2 (a curva) e ninguém rechecou. É metricamente neutro no limite
(invariância C1: é função só de `t`), mas o modelo gasta árvores desfazendo o offset, e **a logloss
que governa a parada antecipada passa a ser dominada por essa correção de `f(t)`** em vez do resíduo
transversal que a métrica cobra — consistente com a parada disparar em 79–117 árvores enquanto o
toco do E0c treinou 1.529–3.186.

`lightgbm.base_rate_weighted` ajusta a curva sob os mesmos pesos. Braço `p2_brw`.

---

## 4b. A triagem K=2 na partição 42 — o que os braços mediram

Baseline pareado `oof_b6c6_joint_bag2` = **0,62640** (grade do board). Barra ~0,0014.

| braço | o que muda | nível | Δ | IC 95% | veredito |
|---|---|---|---|---|---|
| **`p3_lintree`** | D1: folhas lineares | 0,62887 | **+0,0025** | [−0,0007; +0,0053] | **passa por `150–400`** |
| **`p2_brw`** | A3b: `init_score` sob os pesos | 0,62877 | **+0,0024** | [−0,0012; +0,0057] | inconclusivo (1,7× a barra) |
| `p4_cap` | `lr` 0,05→0,03, cap 3000, ES 300 | 0,62756 | +0,0012 | [−0,0011; +0,0036] | inconclusivo (0,8×) |

O `NOTAS_AGENTES.md` §5 fixa a barra de vitória a 2 EP em Δ ≥ 0,0058 com K=4, então nenhum decide no
agregado em K=2. Mas a regra do R0 admite o **bucket-alvo declarado a priori**, e o default do projeto
é `150–400`:

**`p3_lintree` (D1) passa.** `150–400` **+0,0043 [+0,0001; +0,0080], IC exclui 0** — e `150–400` +
`401+` são 78,6% do peso do board. O perfil por bucket é o que o mecanismo prevê, o que importa mais
que o número: folhas lineares ganham onde há pontos por folha para ajustar uma reta (`150–400`
+0,0043, `t>400` +0,0036) e **perdem no warmup** (`t≤50` −0,0119, onde 68% das colunas são NaN em
`t≤4` e uma regressão por folha é instável). Era exatamente a previsão do E0c — o problema é aditivo
e suave, e árvores de degraus aproximam curvas suaves com escadinhas.

**`p2_brw` (A3b)** era o segundo: sinal positivo e consistente em `50–150` (+0,0023), `150–400`
(+0,0026) e `t>400` (+0,0028), negativo só em `t≤50` (−0,0030).

### O K=4 derruba o A3b — e é a lição da rodada

| | baseline | candidato | Δ geral | IC 95% |
|---|---|---|---|---|
| `p2_brw` K=2 | 0,62640 | 0,62877 | +0,0024 | [−0,0012; +0,0057] |
| **`p2_brw` K=4** | 0,62772 | 0,62869 | **+0,0010** | **[−0,0018; +0,0033]** |

**De 1,7× a barra para 0,7×.** E a decomposição diz de onde veio: o nível do candidato mal se mexeu
(0,62877 → 0,62869); quem subiu foi o **baseline** (0,62640 → 0,62772), porque o bag de 4 sementes é
melhor que o de 2. Boa parte do "ganho" era o candidato ter tido sorte de sorteio contra um baseline
mais ruidoso — o mecanismo que o `NOTAS_AGENTES.md` §5 documenta ("a régua tem dp de 0,0041 que o
bootstrap pareado não vê"), acontecendo ao vivo.

**Veredito do A3b: abaixo da barra, não promover.** O descasamento de +3,81 em log-odds no
`init_score` continua sendo um defeito real e vale corrigir por higiene — mas não paga em TS-AUC, e a
hipótese mecânica dele já tinha sido refutada pelo número de rodadas.

**E isso recalibra a leitura do D1.** O `p3_lintree` está em +0,0025 com K=2 — o mesmo patamar de onde
o A3b caiu pela metade. Que o IC dele em `150–400` exclua 0 e que o perfil por bucket tenha mecanismo
são sinais melhores; **nada disso o dispensa do K=4 e da partição 43.**

**Regra a herdar desta rodada:** em K=2, tratar Δ como **ordenação entre braços**, nunca como
estimativa de efeito. Só K=4 — e depois a réplica de partição — produz número.

E se o D1 sobreviver, **não se pode supor que some com nada** — o precedente literal deste repo
(`scorer.py`: *"V5+mrep: os ganhos NÃO SOMAM"*) e a aditividade só parcial do B6+C6 (§14.9) dizem o
contrário. O par tem de ser medido **junto** (`p10_combo` na fase 4).

### O diagnóstico barato que falsificou o mecanismo do A3b

Antes de gastar K=4, a lição do §14.10 do `HISTORICO.md` ("olhar o número de rodadas e a logloss, não
só o Δ"):

| modelo | rodadas por fold | melhor logloss |
|---|---|---|
| incumbente | 173–220 | 0,6625 |
| `p2_brw` | **160–213** | 0,6607 |
| `p4_cap` | 420–491 | 0,6622 |
| toco E0c | 1829–3486 | 0,6662 |

**A hipótese mecânica do A3b está errada.** Eu previ que remover o offset de `f(t)` faria a parada
antecipada disparar mais tarde (por deixar de ser dominada pela correção de `f(t)`). Não aconteceu:
a parada dispara no mesmo lugar e a logloss é praticamente a mesma. A razão é simples e estava à
vista — **`meta_t` e `meta_ln1p_t` são features**, então o modelo representa qualquer `f(t)` com duas
divisões, não com cem árvores. O descasamento de +3,81 em log-odds é real, mas **barato de desfazer**.

O Δ de +0,0024 pode ser um efeito de segunda ordem real (menos capacidade gasta, gradientes melhor
escalados) ou ruído — o IC não separa. **Vai para K=4 + partição 43 pelo Δ, com a hipótese mecânica
registrada como refutada**, que é diferente de ir por ela.

E `p4_cap` responde a pergunta que o E0c revisado levantou: 2,4× mais rodadas com `lr` menor rendem
+0,0012 e **exatamente 0,0000 em `t>400`**. A capacidade que falta ao toco não é a que se compra com
mais rodadas.

### O D1 está sendo subestimado pela própria regra de parada

O mesmo diagnóstico aplicado ao vencedor mostra que o número dele é um **piso**:

| modelo | rodadas | árvores/fold | melhor logloss por fold |
|---|---|---|---|
| incumbente | 173–220 | 73–120 | 0,6625 · 0,6472 · 0,6593 · 0,6515 · 0,6615 |
| `p3_lintree` | 146–177 | **46–77** | 0,6606 · 0,6504 · 0,6581 · **0,6565** · **0,6657** |

O braço entrega **+0,0025 de TS-AUC com ~35% menos árvores** e logloss **igual ou pior em três dos
cinco folds**. Isso tem duas explicações possíveis: (a) a parada antecipada corta o braço cedo, ou
(b) o modelo é simplesmente melhor por árvore. **Medi, e é (b).**

`p9_lintree_pac` — mesmo braço com `--early-stopping-rounds 300` (contra 100) e cap 3000:

| | rodadas por fold | árvores/fold | melhor logloss |
|---|---|---|---|
| `p3_lintree` | 173, 176, 166, 177, 146 | **73, 76, 66, 77, 46** | 0,6606 · 0,6504 · 0,6581 · 0,6565 · 0,6657 |
| `p9_lintree_pac` | 373, 376, 366, 377, 346 | **73, 76, 66, 77, 46** | **idênticos** |

As 200 rodadas a mais são exatamente a janela de paciência estendida — a **melhor iteração não muda**,
e como o LightGBM guarda o melhor modelo, o artefato é **bit-idêntico**. **Hipótese refutada:** o
ótimo de logloss do D1 ocorre de fato em 46–77 árvores, e o número medido não é um piso por esse
motivo.

O que **não** fica descartado é que a *régua* esteja errada — a TS-AUC pode seguir melhorando além do
ótimo de logloss. Mas isso é pergunta sobre o **critério**, não sobre a paciência, e testá-la exige
**rodadas fixas** (o desenho do X0), não uma janela maior; `early_stopping_metric: ts_auc_by_t` já foi
medido e regrediu (−0,0099).

`scripts/run_polimento_fase4.sh` segue com o que importa: promoção do D1 a K=4 + partição 43 e o
**par** `D1+A3b` medido junto.

---

## 4c. Reauditoria das decisões passadas sob a ponderação do board

`scripts/a2_reaudit.py` reroda os pares que decidiram o projeto nas **duas** ponderações. A coluna
"thinning" é o controle: se ela não reproduzir o número histórico, o instrumento está errado e nada
mais na tabela vale. Reproduziu em todos os pares — B6/42 +0,0023, B6/43 +0,0049, C6 +0,0014, todos
batendo com `HISTORICO.md` §14.

| decisão | thinning (como foi julgada) | **board (correta)** | muda? |
|---|---|---|---|
| **B6, partição 42** | +0,0023 [−0,0010; +0,0057] · inclui 0 | **+0,0033 [+0,0004; +0,0063] · exclui 0** | **sim, a favor** |
| **B6, partição 43** | +0,0049 [+0,0016; +0,0080] · exclui 0 | +0,0054 [+0,0027; +0,0087] · exclui 0 | não (reforça) |
| C6 (BOCPD sobre o B6) | +0,0014 [−0,0006; +0,0036] · inclui 0 | +0,0014 [−0,0003; +0,0031] · inclui 0 | **não** |
| **X1-soft** | +0,0040 [+0,0001; +0,0074] · exclui 0 | **+0,0031 [−0,0002; +0,0055] · inclui 0** | **sim, CONTRA** |

**B6 — a adoção que sustenta o incumbente fica mais bem sustentada.** O `HISTORICO.md` registra que na
partição 42 "o agregado inclui 0" e que a adoção se apoiava na réplica da 43. Sob a ponderação
correta, **a 42 passa sozinha**. Média das duas ≈ +0,0044, IC excluindo 0 em ambas.

**C6 — a correção não o move, e minha previsão mecânica estava errada.** Eu esperava que ele piorasse,
porque o ganho vivia só em `150–400` com `t>400` em exatamente +0,0000. O que houve foi cancelamento:
`150–400` mal mudou de peso (48,7% → 46,7%), dobrar um zero continua zero, e o peso retirado de
`t≤150` (onde o C6 é plano) compensou. **O C6 fica exatamente onde estava** — na barra, IC incluindo
0, ganho num único bucket, sem réplica na partição 43, custando 4 features e ~30 µs/passo. A
reauditoria não o resgata nem o condena; só confirma que o número não depende da grade.

**X1-soft — o veredito cai, e é a descoberta mais desconfortável da reauditoria.** Uma mudança **em
produção** deixa de passar a regra do R0: de +0,0040 com IC excluindo 0 para **+0,0031 com IC
incluindo 0**. O mecanismo estava escrito no próprio `configs/default.yaml:weights`, que registra o
ganho do X1 em `50<t≤150` (+0,008) e `150<t≤400` (+0,004) — e a reauditoria mostra `t>400` em
**−0,0003**. Ele ganha exatamente nos buckets cujo peso a grade antiga **inflava** (26,6% → 17,5%) e é
levemente negativo naquele que ela subpesava (16,5% → 31,9%).

**E aqui é preciso não superinterpretar, porque a tentação é grande e estaria errada.** O ponto
continua **+0,0031, acima da barra**, com o IC apenas encostando em zero (−0,0002). "IC inclui 0"
**não é evidência de efeito nulo** — é ausência de evidência de efeito, que é outra coisa. Isto diz
que a base para adotar o X1 era mais fraca do que o registro afirma; **não** diz que ele deve sair.

E remover tem barra própria, mais alta: o X1 está treinado, empacotado, verificado bit-a-bit e em
produção. A barra de **adoção** existe para conter complexidade e risco novos; a de **manutenção**
tem de considerar que remover custa retreino, reverificação e o risco de a estimativa marginal estar
errada. Usar o mesmo limiar para as duas operações enviesa tudo para desfazer o que já funciona.

**O que fica é o achado de método, não uma ação sobre o X1:** o B6 foi na direção **oposta** na mesma
reauditoria. **A correção reordena os braços; não desloca todos igualmente** — que é precisamente por
que ela importa para decisões **futuras**.

**Consequência para o cardápio:** o C2 (vizinhança do X1) deixa de partir de "este ganho está
estabelecido, vamos afiná-lo" e passa a partir de "a evidência é mais fina do que se pensava, então
vale medir a vizinhança com atenção". Continua sendo um braço de afinação, não uma auditoria de
remoção.

---

## 4d. A rodada de 13:25–15:00 — C2 fecha, D3 fica sem célula viva

Quatro braços do `run_sprint_d3.sh` mediram entre 13:25 e 15:00, todos K=2 na partição 42, contra o
mesmo baseline `oof_b6c6_joint_bag2` (TS-AUC 0,6264, grade do board). **Nenhum passa.**

| braço | frente | Δ geral | IC 95% | TS-AUC | leitura |
|---|---|---|---|---|---|
| `p13_floor015` (`detect_floor=0,15`) | C2 | **−0,0014** | [−0,0037; +0,0011] | 0,6250 | **negativo**; −0,0066 em `t≤50` |
| `p13_floor050` (`detect_floor=0,50`) | C2 | +0,0009 | [−0,0020; +0,0029] | 0,6273 | abaixo da barra, IC inclui 0 |
| `p14_mdl100` (`min_data_in_leaf=100`) | D3 | +0,0008 | [−0,0015; +0,0031] | 0,6272 | abaixo da barra, IC inclui 0 |
| `p15_l2_20` (`lambda_l2=20`) | D3 | +0,0003 | [−0,0031; +0,0031] | 0,6267 | nulo |

**C2 está fechada, e o sinal dela é coerente.** Os dois lados do `detect_floor=0,3` foram varridos:
baixar para 0,15 **piora** (−0,0014, e o dano se concentra em `t≤50`, onde o peso da detectabilidade
mais morde), subir para 0,50 dá um ponto positivo pequeno que o IC não sustenta. A previsão
pré-registrada do C2 era "0 a +0,002" — o medido cabe na faixa, na metade inferior dela. O `floor`
atual não é um chute infeliz: é um ótimo raso, e a vizinhança não paga um braço.

**D3 ainda não tem célula que sobreviva.** Três células medidas (`p4_cap` +0,0012, `p14_mdl100`
+0,0008, `p15_l2_20` +0,0003) desenham um padrão sem exceção: **toda mexida de regularização rende
entre 0 e +0,0012, sempre com IC incluindo 0**. Os HPs atuais foram escolhidos na era da semente 42 e
eram "suspeitos por construção" — a suspeita não se confirmou. A previsão do D3 era "0 a +0,004"; o
que se vê até agora é o piso dessa faixa.

**O `p16_ff06` foi pulado por decisão, não por falta de tempo.** `feature_fraction=0,6` aponta para o
lado errado do mecanismo que motivava o braço: o defeito documentado (§5, "A5 bis") é que 15 colunas
100% NaN em `t=50` **diluem** o sorteio de `feature_fraction=0,8`. Baixar para 0,6 sorteia *menos*
features, então as colunas NaN ocupam fração *maior* do sorteio — piora exatamente o que se queria
testar. O braço informativo é `ff=0,9` **ou** derrubar as 15 colunas NaN. Registrado como pendência.

**Padrão da rodada, sem maquiagem.** Nove braços medidos na partição 42 até aqui; um único com IC que
exclui zero e réplica na 43 (o B3, +0,0009). A distribuição empírica dos Δ desta campanha está
concentrada em **[−0,0015; +0,0015]** — o que é a leitura honesta de uma construção madura, e
sustenta o pré-registro do `CAMPANHA_POLIMENTO.md` em vez de revisá-lo para cima.

---

## 4e. O gate do pacote — o pacote é aditivo, e a fusão estava quebrada

### O número

Candidato = **D1 (`linear_tree`, K=4) + B3 (v-EMA assimétrico com gate `min_t=50`)**, aplicado sobre
o mesmo OOF; baseline = `oof_b6c6_joint_bag4`, o incumbente K=4 da partição 42. Grade do board.

| bucket | Δ | IC 95% | exclui 0 | peso |
|---|---|---|---|---|
| **geral** | **+0,0035** | **[+0,0008; +0,0057]** | **sim** | 100% |
| `t≤50` | −0,0041 | [−0,0141; +0,0070] | não | 3,9% |
| `50<t≤150` | +0,0006 | [−0,0051; +0,0051] | não | 17,5% |
| **`150<t≤400`** | **+0,0052** | **[+0,0021; +0,0078]** | **sim** | 46,7% |
| **`t>400`** | **+0,0033** | **[+0,0003; +0,0067]** | **sim** | 31,9% |

TS-AUC **0,6277 → 0,6312**. Regra de decisão: **adotar**.

**É o melhor resultado da campanha**, por três razões independentes: o ponto é 2,5× a barra; o IC
exclui zero **e** exclui zero nos dois buckets de maior peso (78,6% somados), então não é ganho
concentrado num canto barato da métrica; e **o pacote é aditivo** — D1 (+0,0027) + B3 (+0,0008) =
+0,0035, exatamente o medido.

Essa terceira leitura era a razão de o gate existir. Depois de quatro não-aditividades documentadas
neste repo — uma delas (`p10_combo`) transformando duas melhorias num −0,0069 no bucket de maior peso
— aditividade não podia ser assumida. **Aqui ela foi medida, não presumida.**

O `t≤50` em −0,0041 é o pedágio conhecido do `linear_tree` no warmup (68% das colunas em NaN até
`t≤4`, e uma regressão por folha é instável sem dados). Vale 3,9% do peso, o IC inclui zero, e o gate
`min_t: 50` do B3 já existe para não agravá-lo.

**Ressalvas honestas:** (a) o artefato do `linear_tree` é **K=4**, e o `default.yaml` embute **K=7** —
o B1 entra por cima pelo argumento de redução de variância, então +0,0035 é um **piso** medido, não a
estimativa central do que será submetido; (b) **partição 42 apenas** — a réplica na 43 continua
devendo.

### O defeito que o gate expôs: `linear_tree` quebrava a fusão de boosters

Ao regenerar o notebook de submissão, o `verify_submission_notebook.py` falhou — **não no notebook,
no pacote real**:

```
RuntimeError: fuse_boosters: fusão INVÁLIDA
(max |raw_fundido - media_raws| = 1.521e+02 > 1e-09)
```

**Mecanismo.** `fuse_boosters` funde K boosters concatenando as árvores e dividindo as folhas por K —
exato enquanto a folha é uma **constante**, porque o raw do LightGBM é uma soma sobre árvores. Com
`linear_tree=true` a folha passa a valer `leaf_const + Σ leaf_coeff_i · x_i`, e é por **esses** campos
que o LightGBM prediz quando `is_linear=1`. O `_scale_leaves` escalava só `leaf_value`, deixando
`leaf_const` e **35.318 coeficientes por booster** intactos.

**Por que ninguém viu.** O OOF nunca passa por aqui: `scripts/train.py` faz média das **predições**
por semente (`avg_oof.py`). A fusão só existe no caminho de **produção**
(`adapter/platform.py:train`, que roda na nuvem e treina do zero). Ou seja, **todo o +0,0027 do D1 foi
medido num caminho que a produção não usa** — é literalmente a classe do achado A3 ("o que é medido
não é o que é implantado"), desta vez com consequência fatal: o `train()` na nuvem levantaria
`RuntimeError` e a submissão falharia inteira.

**Foi a guarda numérica do próprio `fuse_boosters` que pegou.** O docstring dela diz "falhar alto é
infinitamente melhor que submeter um modelo corrompido" — e foi exatamente o que aconteceu. Sem essa
guarda, a fusão teria devolvido em silêncio um modelo que não representa a média dos originais.

**Correção** (`model/fuse.py`): escalar `leaf_const` e `leaf_coeff` junto com `leaf_value`. É exata
pelo mesmo argumento de linearidade que já justificava a fusão. `leaf_features` fica de fora (são
**índices** de coluna) e `leaf_count`/`leaf_weight` também (são contagens de amostra) — dividi-los
corromperia o modelo em silêncio.

| caminho | antes | depois |
|---|---|---|
| folhas lineares (5 folds reais do `p3_lintree`) | **1,1e+02** | **1,1e−13** |
| folhas constantes (regressão, `b6c6_joint`) | 4,0e−15 | **4,0e−15** (intocado) |

Coberto por `tests/unit/test_fuse_linear_tree.py`, que testa os dois modos de folha **e** verifica que
as árvores lineares do teste têm coeficientes de verdade — sem essa segunda asserção, folhas
degeneradas fariam o teste passar sem exercitar o caminho do D1.

---

## 5. Resultados negativos (registrados para não serem refeitos)

- **A1 — rótulos.** O snapshot local é **pós-correção do W23**: exatamente **29 séries** com
  `tau_index = 0`, a assinatura do changelog, e as 2.868 linhas delas no parquet estão todas com
  `y = 1`. Nada a corrigir; nenhum refresh necessário.
- **A5 — passos frios.** O modelo é **anti-preditivo** em `t ∈ [1,8]` (AUC 0,4374 em `t=3`; só cruza
  0,50 em `t=9`), com 67,9% das features em NaN de warmup em `t≤4`. Real — e **irrelevante**: `t≤8`
  carrega **0,16%** do peso do board, então forçar score constante ali vale **+0,00004**, 1/35 da
  barra. *Dimensionar antes de construir economizou um dia.*
- **B2 — espaço de média do bag.** Média de probabilidade 0,62772 · média de **logit** 0,62773
  (+0,00001) · rank-average 0,62756 (−0,00016); Spearman 0,99999. Empate. Vale como auditoria: a
  produção funde em logit e o OOF media probabilidade, e essa diferença é **imaterial** — o OOF
  representa o objeto submetido.
- **A5 bis — 15 colunas 100% NaN ainda em `t=50`** (`haar_*`, `mmd_*_cal`, `jump_*_w100_cal`,
  `varloc_recent_vs_lagged*`). É a pegadinha do `NOTAS_AGENTES.md` §7 (colunas NaN diluem o sorteio
  de `feature_fraction=0,8`) acontecendo em produção. Não medido ainda; candidato barato.

---

## 6. Estado de cada item da campanha, sem maquiagem

| item | estado | resultado |
|---|---|---|
| A1 rótulos | **fechado** | correto, nada a fazer |
| A2 grade da métrica | **fechado** | defeito achado e corrigido; régua reescrita |
| A3 o que é implantado | **fechado** | (i)(iii)(iv) corretos; (v) descasamento 193 vs 187 achado |
| A3b `init_score` | **fechado — não promover** | K=2 +0,0024 → **K=4 +0,0010** [−0,0018; +0,0033], abaixo da barra; mecanismo refutado |
| A5 passos frios | **fechado** | real, vale +0,00004 |
| A6 clip do resíduo | **MEDIDO (K=2, part. 42)** | **+0,0015 [−0,0008; +0,0036]** · ponto acima da barra, IC inclui 0 · positivo em 95,6% do peso · **exige K=4**, não promover ainda |
| B1 bag 4→7 | **medido (partição 42)** | **+0,0011 [+0,0002; +0,0018]** · IC exclui 0 · réplica na 43 pendente |
| B2 espaço da média | **fechado** | empate (+0,00001) |
| B3 bundle de pós-processo | **PASSA NAS DUAS PARTIÇÕES** | 42: +0,0008 [+0,0002; +0,0013] · 43: **+0,00094 [+0,00039; +0,00143]** · ambos excluem 0 — **único item da campanha em condição de adoção pelo protocolo** |
| C1 vizinhança do B6 | **fechado — gradiente** | 0,4 **+0,0017** · 0,6 ref. · 0,8 −0,0013 · **0,2 +0,0001 e −0,0117 em `t≤50`** (rejeitado). A curva vira entre 0,2 e 0,4; `contri=0,4` é o candidato |
| C2 vizinhança do X1 | **FECHADA — nenhum lado paga** | `floor=0,15` **−0,0014** [−0,0037; +0,0011] (−0,0066 em `t≤50`) · `floor=0,50` +0,0009 [−0,0020; +0,0029] · o 0,3 é um ótimo raso (§4d) |
| D1 `linear_tree` | **decisão pendente, não veredito** | `false` no YAML por uma retratação que a §15.16 classificou como inconclusiva; saldo +0,0027 / +0,0005 nas duas partições |
| D2 `max_bin` | **fechado negativo** | `p5_maxbin1023`: −0,0004; `t≤50` −0,0080, IC exclui 0 |
| D3 sweep de HPs | **três células, nenhuma viva** | `p4_cap` +0,0012 · `p14_mdl100` +0,0008 [−0,0015; +0,0031] · `p15_l2_20` +0,0003 [−0,0031; +0,0031] · todos com IC incluindo 0 (§4d). `p16_ff06` **pulado por decisão** — mecanismo invertido; o braço certo é `ff=0,9` ou derrubar as 15 colunas NaN |
| D4 `refit` em 100% | não iniciado | o OOF não vê o ganho por construção; exige quota do board |
| D5 micro-higienes | não iniciado | |
| E1 des-thinning | não iniciado | **o A2 muda o alvo**: a região mais subamostrada é `401+` (passo 4, 31,9% do peso), não `151–400` |
| F1/F2 knobs de φ | não iniciados | exigem rebuild |
| reauditoria em lote | **script pronto**, interrompida | `scripts/a2_reaudit.py`, 15 pares |

O driver atual é **`scripts/run_resto.sh`** (v2), que substituiu o `run_polimento.sh` depois que a v1
sobre-assinou a máquina (ver `NOTAS_AGENTES.md` §7). Ele é **resumível** — cada etapa pula se o
artefato dela já existe — e **estritamente serial**, com `OMP_NUM_THREADS=4`, build com `--n-jobs 4`
e bootstrap com `--n-jobs 2`. A fila que resta, em ordem:

~~`p11_contri02`~~ → ~~`b3_p43`~~ → ~~**A6** (rebuild + `p8_a6`)~~ → ~~`p13_floor015`~~ →
~~`p13_floor050`~~ → ~~`p14_mdl100`~~ → ~~`p15_l2_20`~~ → ~~`p16_ff06`~~ (**pulado por decisão**) →
**`p8_a6` K=4** ← *aqui* → `p17_nobundle` → **E1** (rebuild + `p12_e1`) → notebook.

**Retomada de 2026-07-25 11:45** após queda de energia às ~11:26 (a máquina caiu durante o bootstrap
do `p11_contri02`; o `b1_bag7` já havia terminado). O lockfile órfão foi removido e a fila relançada
do ponto em que parou.

**Segunda queda: ~15:03.** A máquina caiu ~3 min depois de o `run_a6_k4.sh` assumir o lock do sprint
(que ele mesmo encerrou às 15:00, após o `p15_l2_20` fechar). Perdido: o treino da semente **202** do
`p8_a6`, que tinha 3 min de vida e não deixou diretório parcial. **Nada mais se perdeu** — os cinco
JSONs da rodada (13:25–15:00) estavam todos gravados, e o sprint já havia chegado ao seu fim
deliberado.

**Retomada de 2026-07-25 15:08:** lockfile órfão removido, `run_a6_k4.sh` relançado do zero (ele é
idempotente: a espera pelo `p15` passa direto porque o JSON existe, e o `skip` por diretório impede
retreinar as sementes 777/101). Em curso: sementes **202** e **303** → média das quatro →
`compare_oof` contra `oof_b6c6_joint_bag4` — **K=4 contra K=4**, que é o padrão probatório que a
retratação do D1 tornou obrigatório. ETA ~25 min.

**Para retomar o resto:** `bash scripts/run_resto.sh` (resumível; pula tudo que tem JSON). Custos
medidos, para planejar: treino **8,5 min/semente**, medição **~8 min**, rebuild **~45–60 min**
(`build_v5` levou 26 min com `n_jobs=-1`; agora é `--n-jobs 4` sobre 6 núcleos físicos).

### Pendências pontuais

1. **Remedir o E0c no par limpo** — **feito**: a grade antiga reproduz o `RELATORIO_EXP0.md` casa a
   casa (toco 0,6028, B6 0,6067, Δ +0,0039), e a grade do board dá +0,0059 com IC excluindo 0. O
   primeiro par que usei (`b6c6_joint`, 187) conflacionava interações com o BOCPD.
2. **A6 exige rebuild** do dataset com `conformal.use_raw` + `h0.null_clip_match`. É a medição de
   maior valor esperado ainda não feita: dois defeitos com mecanismo, sobre a feature nº 1.
   *Nota de diagnóstico:* na v1 do driver, A6 e E1 aparecem como "falharam" nos logs de 11:06/11:08,
   mas **não foi RAM nem oversubscription** — foi um defeito de quebra de linha no próprio script,
   que soltou o `--out` do comando (`s_a6.parquet: command not found`) e fez o build não escrever
   nada; os treinos seguintes então morreram com `FileNotFoundError` no parquet inexistente. Os dois
   braços continuam **não medidos**, não refutados.
3. **O gate do A1 (v-EMA em `t>150`) foi escolhido sob os pesos errados** — a justificativa registrada
   em `scripts/apply_vema_oof.py` pesa `t>400` por 16,5% e `t≤50` por 8,1%. Reotimizar (B3).
4. **`scripts/a2_reaudit.py` precisa rodar sozinho** — foi ele que, competindo com o treino, causou o
   deadlock de OpenMP documentado no `NOTAS_AGENTES.md` §7.
5. **O braço de `feature_fraction` continua por medir na direção certa.** O `p16_ff06` (0,6) foi
   pulado porque inverte o mecanismo (§4d). O teste que responde à pergunta é **`ff=0,9`** ou, melhor,
   **derrubar as 15 colunas 100% NaN em `t=50`** e manter `ff=0,8` — este último não é sweep de HP, é
   higiene de conjunto, e é barato (sem rebuild).

---

## 5b. O que foi adotado, retratado, e o que está em `default.yaml` AGORA

> **ERRATA (2026-07-25).** A versão original desta seção dizia que `linear_tree` e
> `base_rate_weighted` haviam **entrado** em `configs/default.yaml`. **Não estão lá.** As duas foram
> adotadas e **retratadas no mesmo dia**, e o YAML tem ambas em `false`. O texto abaixo foi corrigido
> para bater com o arquivo; a justificativa original fica registrada porque o D1 continua vivo.

**Estado real de `configs/default.yaml` neste momento:**

| flag | valor | história |
|---|---|---|
| `lightgbm.feature_contri_meta` | **0,6** | B6, adotado em 2026-07-24 nas duas partições (+0,0023 e +0,0049). **A única adoção que está de pé.** |
| `lightgbm.linear_tree` | **`false`** | D1 — adotado e retratado; a retratação depois virou ERRATA (ver abaixo) |
| `lightgbm.base_rate_weighted` | **`false`** | A3b — adotado e retratado; permanece fechado |
| `conformal.use_raw` / `h0.null_clip_match` | **`false`** | A6 — flags criadas, defeito documentado, **nunca medidas** |

### O D1 está num limbo que o YAML ainda não reflete

A sequência foi: 42/K=2 +0,0025 → 42/K=4 +0,0027 (adotado) → **43/K=2 −0,0019 (retratado)** →
43/K=4 **+0,0005** (a retratação era inconclusiva, §15.16).

O comentário dentro do `default.yaml` justifica a retratação com o −0,0019 do K=2 e fecha dizendo
*"reabrir exige K=4 nas duas partições, não K=4 numa só"*. **Essa condição já foi satisfeita** — o
K=4 na partição 43 existe e deu +0,0005. O saldo atual do D1 é **positivo nas duas partições
(+0,0027 e +0,0005), média ≈ +0,0016, acima da barra**, significativo em uma.

Ou seja: o D1 não está `false` porque foi refutado, está `false` porque foi revertido por uma
medição que o próprio repo depois classificou como inconclusiva, e ninguém reabriu. **É uma decisão
pendente, não um veredito.** Se reabrir, entra **sozinho** — o par D1+A3b é pior que o D1 puro
(−0,0069 em `t>400`, IC exclui 0).

### Por que o A3b entrou abaixo da barra

Porque **a barra de adoção existe para conter complexidade e risco novos**, e essa mudança não traz
nenhum dos dois: é um flag, não toca inferência, não acrescenta coluna, não muda latência. O que ela
faz é alinhar o `init_score` com a distribuição que o LightGBM de fato otimiza — hoje a curva é
ajustada na contagem crua enquanto o treino usa pesos que equalizam as classes (descasamento medido
de até +3,81 em log-odds).

Aplicar o mesmo limiar a uma correção de coerência de custo zero e a uma família de features nova é
um erro de categoria. O sinal é positivo e o mecanismo do defeito é verificável independentemente do
Δ; o Δ pequeno só diz que o modelo já contornava o problema com duas divisões em `meta_t`.

**O que NÃO entrou, e por quê factual:**

- **A6** (`conformal.use_raw`, `h0.null_clip_match`) — **não medidas**. Exigem rebuild. São defeitos
  reais e a aposta é favorável, mas "não medido" é categoria diferente de "medido e pequeno".
- **D2** (`max_bin`) e **C1** (`contri` 0,4/0,8) — treinadas, em medição.
- **B1** (bag 4→7) — medido e positivo, mas **só na partição 42**; ver abaixo.

### B1 (bag 4→7): medido, positivo, ainda não adotado

Primeiro resultado da retomada após a queda de energia. Na grade do board, partição 42,
`oof_b6c6_joint_bag7` contra `oof_b6c6_joint_bag4`:

| bucket | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| **overall** | **+0,0011** | [+0,0002; +0,0018] | **sim** |
| t≤50 | +0,0021 | [−0,0008; +0,0051] | não |
| 50<t≤150 | +0,0011 | [−0,0008; +0,0026] | não |
| 150<t≤400 | +0,0014 | [+0,0002; +0,0023] | **sim** |
| t>400 | +0,0006 | [−0,0004; +0,0016] | não |

TS-AUC 0,6277 → 0,6288. Três observações, sem maquiagem:

1. **O ponto (+0,0011) fica abaixo da barra declarada de 0,0014**, ainda que o IC exclua 0. Vale aqui
   o mesmo argumento de categoria do A3b: a barra existe para conter complexidade e risco novos, e
   três sementes a mais não trazem nenhum dos dois — é compute puro, sem coluna nova, sem mudança de
   inferência. A diferença em relação ao A3b é que este tem **custo recorrente** (bag 75% maior em
   treino e em latência de inferência), e é esse custo, não o Δ, que deve decidir.
2. **O valor caiu exatamente onde a campanha previu.** O `CAMPANHA_POLIMENTO.md` pré-registrou
   "+0,0008 a +0,0010, só compute" a partir da curva da campanha X. Medido: +0,0011. É o único item
   da campanha até agora cuja previsão acertou a faixa — o que reforça a leitura de que K=7 está
   perto da saturação e que **não há mais nada a extrair aumentando o bag**.
3. **Falta a réplica na partição 43**, que o protocolo torna obrigatória antes de qualquer adoção.
   Até ela existir, B1 fica fora do `default.yaml`.

**Pendência de empacotamento:** o notebook de submissão embute o YAML e precisa ser regenerado e
reverificado bit-a-bit (`scripts/build_submission_notebook.py` + `verify_submission_notebook.py`).

---

## 6b. O que o A2 faz com a aritmética da própria campanha

O `CAMPANHA_POLIMENTO.md` fecha com um pré-registro: *"+0,005 a +0,010 de OOF, board ~0,632–0,637 …
isso disputa a linha do rank 10 se ela estiver ≲0,638"*. Essa conta foi feita com **duas incógnitas
que o A2 elimina**.

**Primeira: o câmbio.** A campanha somava um "+0,013 de câmbio OOF→board" a cada ganho de OOF. Esse
câmbio era a grade. Com as duas séries na mesma ponderação, **um Δ de OOF é um Δ de board**, a menos
do ruído entre bases (~0,002 nos dois pontos que temos). A previsão passa de "0,6125 + ganho + 0,013"
para **"0,6278 + ganho"** — que dá o mesmo número por outro caminho, mas agora sem um termo estimado
em n=1 que ninguém sabia se era offset ou artefato.

**Segunda: a barra de detecção.** Com `t>400` valendo 31,9%, um braço que ganha `+0,01` naquele bucket
vale `+0,0032` no agregado, e não `+0,0017`. Isso não cria ganho — mas muda **onde procurar**, e o
`p3_lintree` é o primeiro caso: ele é positivo em `150–400` e `t>400` (78,6% do peso somados) e
negativo em `t≤50` (3,9%), um perfil que a ponderação antiga penalizava por quase o dobro.

**O que a rodada de hoje diz sobre o teto, honestamente.** Três braços medidos, um único sobrevivente
provisório (+0,0025, ainda sem K=4) e um que encolheu pela metade ao ganhar sementes. **Nada aqui
sustenta revisar o pré-registro para cima.** O que mudou não foi o teto — foi a régua, e com ela a
capacidade de *ver* qual braço funciona. A conta continua a mesma: se a linha do rank 10 estiver
acima de ~0,64, esta campanha sozinha não paga prêmio, e a leitura do leaderboard segue sendo o
primeiro entregável que falta.

---

## 7. A leitura que reorganiza a campanha

O `CAMPANHA_POLIMENTO.md` foi escrito sob a premissa do `RELATORIO_EXP0.md`: `g` saturado,
ensembling saturado, calibração saturada, teto de ~+0,01 em sete semanas. Duas dessas três
conclusões foram medidas **na grade errada**, e a terceira (ensembling, +0,0021 sobre 16 membros)
também — o câmbio empírico daquela tabela reponderaria.

Isso não prova que a construção tem muito mais a dar. Prova que **o instrumento que declarou o beco
sem saída estava descalibrado no eixo exato em que a conclusão foi tirada**, e que a fonte de erro
tinha o mesmo tamanho (+0,013) do gap que se tentava explicar. A ordem correta era esta: auditar a
régua antes de acreditar no que ela mede — que é, literalmente, o que a frente A existia para fazer.
