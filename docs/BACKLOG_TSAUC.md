# Backlog de implementação — plano de ação TS-AUC consolidado

**O que é este arquivo.** A ponte entre `plano_acao_tsauc_consolidado.md` (raiz do repo, síntese dos
relatórios `compass_artifact` + `informacao_nao_capturada`) e o código. O plano consolidado prioriza
frentes F0→F10 mas foi escrito **sem acesso linha-a-linha a `src/sbrt/`** — caveat explícito do
próprio compass. Aqui cada frente vira item executável: arquivo alvo, hipótese registrável a priori,
bucket declarado, DoD e comando de medição.

**Leia antes:** [`NOTAS_AGENTES.md`](NOTAS_AGENTES.md) §1 (invariantes), §2 (contratos), §5
(protocolo de medição), §11 (como adicionar família nova). Nada aqui substitui aquele protocolo.

## ERRATA (2026-07-22, 02:40) — a primeira "calibração do nulo" estava contaminada

**`data/processed/train_rows.parquet` é o dataset do V5, não o do V4.** 178 features, com BOCPD e
sem L-momentos/`dep_w050` — exatamente a poda do V5. A reversão do V5 (HISTORICO §9) restaurou o
modelo e o `resources/`, **não o dataset**; a proveniência foi assumida pelo nome do arquivo.

Consequência: os retreinos "V4 com semente diferente" mudaram a semente **e** o conjunto de features.
Os −0,0037 reportados como "trocar só a semente" são V5-vs-V4 remedido (coerente com os −0,0042 já
documentados). **A tese de que a régua tem ~0,004 de ruído invisível fica SEM SUPORTE** até ser
remedida com um V4 legítimo. O mesmo vale para a média 0,6081/dp 0,0018 atribuída ao V4 e para o
veredito do F2 com K=4, que comparou F2 contra uma mistura de 1 V4 + 3 V5.

Não afetados: as 4 sementes do F2 (dataset correto), o código dos três blocos novos, o rastreio de
redundância, e a build de 202 features.

Recuperação: **`train_rows_3eixos.parquet` menos as 19 colunas novas é exatamente o conjunto de 183
features do V4** (verificado por diferença de conjuntos + igualdade bit-a-bit nas colunas comuns).
`scripts/train.py --drop-prefix` deixa de ser otimização e passa a ser a única fonte de um V4
legítimo em disco. Todas as comparações abaixo são remedidas a partir dessa build única.

**Lição de processo, mais importante que o erro:** um artefato de dados não carrega sua proveniência.
`oof_v4.parquet` existir não implica que `train_rows.parquet` seja o dataset do V4. Antes de usar um
parquet como baseline, **checar o conjunto de colunas** contra o que o modelo alega ter — é uma
linha de código e teria evitado duas horas de conclusão errada.

---

## O nulo da regra de decisão — REMEDIDO com o V4 legítimo (2026-07-22, 03:20)

**Leia isto antes de interpretar qualquer Δ deste documento.** A tese abaixo estava certa; a primeira
medição dela estava contaminada (ver ERRATA). Com o V4 reconstruído de `train_rows_3eixos.parquet`
via `--drop-prefix spec_ ord_ mrep_` — 183 features, e a semente 42 reproduz **exatamente** os 0,6100
do `oof_v4` histórico, o que valida a reconstrução:

| semente | árvores por fold | total | TS-AUC |
|---|---|---|---|
| 42 (o artefato histórico) | 80, 98, 51, 79, 96 | 404 | **0,6100** |
| 777 | 75, 82, 63, 97, 67 | 384 | 0,6006 |
| 101 | 54, 103, 60, 85, 57 | 359 | 0,6020 |
| 202 | 64, 91, 73, 66, 73 | 367 | 0,6030 |
| **média** | | | **0,6039** (dp **0,0041**, n=4) |

**O V4 real é 0,6039, não 0,6100.** O número que este projeto usa como âncora é o maior de quatro
sorteios do mesmo modelo — maldição do vencedor no nível do artefato.

### O que isso faz com as três rejeições

Com dp 0,0041 entre sementes, o EP da diferença entre dois modelos de **uma semente cada** é 0,0058:

| braço | Δ medido | em EP | veredito dado | veredito correto |
|---|---|---|---|---|
| F1 (+9 col `_cal`) | −0,0069 | 1,2 EP | "regrediu, IC exclui 0" | **inconclusivo** |
| V5 (BOCPD + poda) | −0,0042 | 0,7 EP | "regrediu, revertido" | **inconclusivo** |
| F2 (mismatch) | −0,0024 | 0,4 EP | "não adotar" | **inconclusivo** |

**Nenhuma das três rejeições era distinguível de ruído de semente.** O IC do F1 excluía zero porque
media só a reamostragem de *séries*; a componente de *sorteio do modelo* estava ausente da conta.

### A semente 42 está contaminada por seleção — e isso reinterpreta o histórico inteiro

Com 6 sementes do V4 legítimo: 0,6006 · 0,6016 · 0,6020 · 0,6022 · 0,6030 · **0,6100**. As cinco que
não são a 42 têm média 0,6019 e dp 0,0009 — a 42 está a **9 desvios**. Não é sorte.

O teste que dá o mecanismo: comparar cada braço na semente 42 contra a média das suas outras três.

| braço | semente 42 | outras 3 | excesso da 42 |
|---|---|---|---|
| V4 (183) | 0,6100 | 0,6019 | **+0,0081** |
| SPEC (191) | 0,6035 | 0,6039 | −0,0004 |
| ORD (188) | 0,6030 | 0,6031 | −0,0001 |
| MREP (189) | 0,6048 | 0,6080 | −0,0032 |

**A semente 42 é excepcional exclusivamente para o conjunto de features do V4.** Para toda
configuração modificada ela é neutra ou pior. Mecanismo: todas as decisões históricas do projeto —
quais features manter, quais podar, quais hiperparâmetros, qual critério de parada — foram avaliadas
comparando corridas na semente 42. Ao longo de dezenas dessas decisões a configuração deriva para uma
que explora as particularidades daquele sorteio. É sobreajuste à semente, acumulado pela seleção, e
explica as duas coisas a explicar: o tamanho do efeito e o fato de ser **unilateral**.

Consequências:

- **Os 0,6100 do V4 não são reprodutíveis.** O mesmo modelo com sorteio novo dá 0,6019. O artefato em
  `resources/` continua válido como modelo, mas o número associado a ele está inflado por seleção.
- **Todo veredito histórico favorecia estruturalmente o incumbente**, porque o incumbente era o único
  cuja configuração fora selecionada naquela semente. Isso soma-se ao ruído de sorteio e explica por
  que três ampliações independentes "regrediram" com a mesma assinatura.
- Comparações futuras excluem a semente 42, ou usam K grande o bastante para diluí-la.

### Bagging de sementes vale +0,0046 — e não é feature nenhuma

| | TS-AUC |
|---|---|
| média das 5 sementes individuais do V4 (sem a 42) | 0,6019 |
| **modelo com as 5 sementes empacotadas** | **0,6065** |

É honesto: todas as sementes usam os mesmos folds, então a média das predições continua sendo OOF
legítimo. É bagging clássico reduzindo variância do preditor. O ganho é **maior que qualquer efeito
de feature já medido neste projeto**, é ortogonal à escolha de features (todos os braços ganham
~+0,004 ao serem empacotados) e está disponível hoje, sem build nova.

### A causa mecânica do ruído, e a frente que ela abre

O número de árvores por fold varia de 51 a 103 entre sementes — 2× no mesmo fold. O
`early_stopping_rounds=100` sobre **logloss** escolhe pontos de parada erráticos, e o modelo com mais
árvores (404) é o de melhor TS-AUC. Fixar o número de rodadas eliminaria de uma vez um ruído maior
que qualquer efeito de feature já medido neste projeto — **é provavelmente a maior alavanca
disponível hoje, e não é uma feature.**

Corolário incômodo: o comentário de `config.py:LightGBMConfig.early_stopping_metric` que descarta
`ts_auc_by_t` ("Δ −0,0099, IC exclui 0") foi medido com um sorteio por lado = 1,7 EP. **O critério de
parada volta a ser questão aberta.**

### A regra corrigida

1. **Nenhum braço se decide contra um sorteio único.** K sementes por lado, MESMO K, comparar as OOF
   médias (`scripts/avg_oof.py`). Com K=4 o EP da diferença cai para 0,0029, então a barra para
   declarar vitória a 2 EP é Δ ≥ 0,0058.
2. `scripts/train.py --boost-seed N` perturba o booster sem tocar nos folds.
3. `scripts/seed_spread.py` mostra as duas famílias como distribuições, que é a apresentação honesta.
4. O caro é a build (~30 min); cada semente é ~10 min sobre o parquet que já existe.

### O que reabre (todos, não só um)

- **F1, V5, F2**: as três rejeições ficam sem suporte. Reabrir exige K sementes.
- **Critério de parada** (`logloss` vs `ts_auc_by_t` vs rodadas fixas): questão aberta, e
  provavelmente com efeito maior que qualquer família de features.
- Qualquer Δ deste documento abaixo de ~0,006 medido com uma semente por lado é **inconclusivo**,
  não negativo. Inclui a frente de centragem por série (+0,0018).

---

## Registro original da tese (medição contaminada, preservado para rastreabilidade)

O F2 fechou em Δ −0,0024 [−0,0078, +0,0027] e seria o terceiro braço descartado com a MESMA
assinatura (pior em `50<t≤150`). Três mudanças sem nada em comum — uma poda com BOCPD, uma
calibração, um bloco de brancura multi-lag — caindo no mesmo lugar não é coincidência plausível. E a
explicação corrente ("largura dilui `feature_fraction`") **não cobre o V5, que era mais estreito que
o V4**. Ou seja: a hipótese estava errada.

Teste do nulo: retreinar o V4 **contra ele mesmo** — mesmo parquet, mesmas 183 features, MESMOS folds
(`cfg.seed=42` intocado), só `lightgbm.boost_seed` diferente. Resultado
(`artifacts/reports/compare_null_boostseed.json`):

| braço | Δ geral | IC 95% | veredito que foi dado |
|---|---|---|---|
| **NULO — V4 vs V4, só a semente** | **−0,0037** | [−0,0088, +0,0012] | — |
| F2 (mismatch, +7 col) | −0,0024 | [−0,0078, +0,0027] | "não adotar" |
| V5 (BOCPD + poda, −5 col) | −0,0042 | [−0,0095, +0,0006] | "regrediu, revertido" |
| F1 (+9 col `_cal`) | −0,0069 | [−0,0125, −0,0018] | "regrediu, IC exclui 0" |

**O nulo é maior em módulo que o F2 e igual ao V5.** Consequências, sem atenuação:

- **F2 é melhor que o nulo.** O −0,0024 não é regressão; é um sorteio acima da média.
- **A reversão do V5 foi decidida sobre um número indistinguível de trocar semente.** O que o V5
  media não era a poda nem o BOCPD.
- **Só o F1 tem excesso real sobre o nulo**: −0,0069 contra −0,0037 = excesso de −0,0032, menos da
  metade do que foi reportado. A autópsia de duplicata ruidosa continua de pé como mecanismo, mas o
  tamanho do dano foi superestimado.
- O nulo é negativo em 4 dos 5 buckets, e em `150<t≤400` dá −0,0050 [−0,0109, +0,0006] — **quase
  significativo por nossa própria regra, só por trocar semente.**

**Por que o nulo é negativo, e não centrado em zero.** `oof_v4` é o incumbente porque foi o sorteio
que ficou. Qualquer sorteio novo regride à média — maldição do vencedor no nível do *artefato*. O
mesmo mecanismo já está documentado em `config.py:LightGBMConfig.early_stopping_metric` para a
escolha da rodada de boosting; ninguém o aplicou ao modelo inteiro.

**Por que o bootstrap pareado não pega isso.** Ele reamostra `id`s e trata as predições de cada modelo
como **fixas**. Cancela a variância comum entre séries — que é o que ele promete — mas é cego à
variância de *retreino*, que com `feature_fraction=0,8` e `bagging_fraction=0,8` não é pequena: mudar
o conjunto de colunas embaralha todos os sorteios, então o candidato nunca é "o baseline mais uma
coluna", é um modelo novo da mesma distribuição.

### A regra corrigida

1. **Nenhum braço se decide contra um único sorteio do baseline.** Cada lado da comparação usa K
   sementes com os MESMOS folds, e o R0 compara as OOF **médias** (`scripts/avg_oof.py`). O ruído de
   semente cai ~1/√K; com K=4 vai de 0,004 para 0,002, que é o que permite resolver efeitos da ordem
   de 0,005.
2. **Mesmo K dos dois lados**, senão a diferença de suavização vira efeito.
3. Custo real: o caro é a build (~28 min); cada semente extra é ~9 min de treino sobre o parquet que
   já existe. K=4 custa +27 min por braço — barato perto de descartar uma frente boa.
4. `scripts/train.py --boost-seed N` faz a perturbação sem tocar nos folds.

### F2 remedido com K=4 — REPROVADO, agora de forma defensável (2026-07-22)

| | sementes (42, 777, 101, 202) | média | dp entre sementes |
|---|---|---|---|
| V4 (183) | 0,6100 · 0,6062 · 0,6093 · 0,6069 | **0,6081** | 0,0018 |
| F2 (190) | 0,6075 · 0,6001 · 0,6036 · 0,6033 | **0,6036** | 0,0030 |

R0 sobre as OOF médias (`compare_f2k4_vs_v4k4.json`), bucket-alvo `150<t≤400` declarado a priori:

| bucket | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| geral | **−0,0050** | [−0,0078, −0,0021] | **sim** |
| `t≤50` | −0,0023 | [−0,0115, +0,0066] | não |
| `50<t≤150` | −0,0078 | [−0,0133, −0,0027] | **sim** |
| `150<t≤400` | **−0,0056** | [−0,0085, −0,0027] | **sim** |
| `t>400` | +0,0000 | [−0,0024, +0,0027] | não |

**As quatro sementes do F2 ficam abaixo da média do V4**, e a melhor delas perde para três das
quatro do V4. `MismatchBlock` sai de `default_blocks()`.

O ponto metodológico é que o protocolo funcionou **nos dois sentidos**: com uma semente por lado o
F2 media −0,0024 com IC [−0,0078, +0,0027] — indistinguível do nulo de trocar semente (−0,0037) e
portanto sem conteúdo. Com K=4 o IC apertou para [−0,0078, −0,0021] e o efeito apareceu. A média de
sementes não é só "mais cuidado": ela **resolve** efeitos de 0,005 que o sorteio único não resolvia.

Nota secundária, mas coerente: o dp entre sementes do F2 (0,0030) é maior que o do V4 (0,0018).
Mais colunas => mais variância de sorteio. Isso é um custo por si só, independente da média.

### O que reabre

- **F2**: precisa de veredito honesto com K=4. Em medição.
- **V5 / F0.c**: o experimento "V4 + BOCPD sem a poda" deixa de ser opcional — a razão para achar que
  o BOCPD não serve nunca foi medida.
- **Todos os Δ deste documento abaixo de ~0,005 medidos com uma semente por lado** viram
  inconclusivos, não negativos. Isso inclui a frente de centragem por série (+0,0018).

---

## Resultados (2026-07-21)

| item | veredito |
|---|---|
| **F1.a + F1.b-1** | **REVERTIDOS.** R0 contra o V4: Δ geral **−0,0069** [−0,0125, −0,0018], IC exclui 0 **contra**; `50<t≤150` em −0,0136 [−0,0222, −0,0053]. `calibration.recursive_features` está vazio |
| **F5** | **CANCELADO** por F0.d — precursores não preveem quebra precoce (AUC 0,4798, abaixo do acaso) |
| **F0.b** | causa (2) **quantificada**: 91% das séries com gap>0, mas razão gap/espalhamento = 0,81 |
| F1.0, F2, F4 | código e testes prontos; **nenhum ligado** em produção |

O conjunto de features está de volta ao do V4 (183, verificado idêntico), 167 testes verdes.

### A autópsia encontrou uma frente maior que o fracasso

Duas camadas explicam o R0 negativo.

**Rasa — duplicata ruidosa.** As colunas `_cal` correlacionam 0,92–0,98 com as cruas dentro do passo:
não acrescentam ordenação. Pior, `_cal` é `(x−μ̂ᵢ)/σ̂ᵢ` com μ̂ᵢ, σ̂ᵢ estimados por série a partir de ~20
réplicas, e esse erro de estimação é **independente entre séries** — ruído transversal puro, injetado
num alvo que só pontua ordenação transversal. Uma árvore que sorteie a versão calibrada em vez da
crua faz um corte estritamente pior. Isso explica o dano exceder o que 5% de largura sugeririam.

**Profunda — o F1 mirou um nível que não é o gargalo.** Calibrar a escala de cada *feature* não podia
funcionar porque a ordenação transversal já estava preservada (corr 0,92–0,98). Mas a explicação
alternativa que persegui em seguida — "o gargalo é o NÍVEL de score de cada série" — também está
errada, e por um artefato que vale registrar em detalhe porque é fácil de repetir.

> **ARMADILHA: comparação dentro-da-série sem controlar `t`.** Reportei que o gap dentro da série era
> 0,0462 contra um efeito transversal de 0,0064 — "7,3×", "91% das séries com gap>0", "86% do sinal
> destruído". **Tudo artefato.** Dentro de uma série, os passos pré-quebra têm `t` médio 12,4 e os
> pós-quebra 35,3: a quebra *divide* a série no tempo, então pós-quebra é sempre `t` maior. E o score
> cresce com `t` pela curva de taxa-base, haja quebra ou não. Eu estava medindo a tendência temporal.
>
> Controlando por `t` (desvio da média transversal do mesmo passo), o gap interno é **0,0010** —
> *menor* que o efeito transversal de 0,0064. Não há sinal escondido, não há 86% destruído, e a
> centragem por série estava condenada antes de eu escrever a primeira linha dela.
>
> Regra: **qualquer contraste pré/pós dentro de uma série tem de ser medido contra a seção
> transversal do mesmo passo.** Sem isso mede-se a taxa-base.

### A frente de centragem por série — construída, medida e MORTA (2026-07-21)

O 7,3× acima sugeria que o culpado era o *nível de score de cada série*, e que centrar cada série
pelo próprio nível H0 resolveria. Um oráculo (subtraindo a média pré-quebra real) parecia confirmar
com folga: `t≤50` de 0,5357 para **0,8026**, geral de 0,6100 para **0,6753**.

**Era artefato.** O oráculo centra pelas MESMAS linhas em que avalia, o que colapsa os negativos para
zero por construção. O teste honesto — centrar pelos primeiros K passos pré-quebra e avaliar só nos
posteriores — mata a hipótese:

| K passos usados para centrar | linhas avaliadas | Δ TS-AUC |
|---|---|---|
| 10 | 2.417.149 | **−0,0014** |
| 25 | 2.225.254 | **−0,0018** |
| 50 | 1.937.990 | **−0,0014** |

Nem com 50 passos pré-quebra da própria série — informação que nenhum detector causal teria — a
centragem ajuda. A implementação causal completa (baseline H0 medido sobre 200 passos da cauda do
histórico, `scripts/score_centering_experiment.py`, 10.000 séries) confirma:

| bucket | base | centrado | β=0,25 | β=0,50 | β=0,75 |
|---|---|---|---|---|---|
| `t≤50` | 0,5357 | −0,0067 | +0,0010 | +0,0001 | −0,0024 |
| geral | 0,6100 | −0,0149 | +0,0018 | −0,0001 | −0,0061 |

O melhor é +0,0018 — máximo sobre 7 variantes, dentro de 1 desvio do bootstrap (≈0,0028). Ruído.
Variantes por média-da-série e suavizada também foram testadas e são piores.

**Três medições minhas estavam viciadas, e vale registrar quais:**

1. **O oráculo era circular** — eu anotei "parcialmente circular" no plano e mesmo assim usei a
   magnitude para dimensionar a frente. O teste honesto custava minutos e devia ter vindo ANTES.
2. **A confiabilidade split-half de 0,624** foi medida sobre `oof_pred` cru, que carrega o offset de
   taxa-base crescente em `t`. Séries de comprimentos diferentes têm médias de `t` diferentes em cada
   metade, o que infla a correlação. Em logit sem offset, o nível é bem menos "traço" do que parecia.
3. **A compressão do baseline não vinha de viés in-sample.** Ajustar o H0 em `hist[:-N]` PIOROU a
   correlação (0,322 → 0,191): o H0 com menos dados perde mais do que o viés custava.

O que sobrevive: a correlação entre nível-do-histórico e nível-online é real (0,653 no estrato onde o
critério é bem medido — o 0,322 agregado é atenuação por ruído do critério). **Mas ela não é
acionável**, porque o nível por série não é o que separa quebra de não-quebra no corte transversal.

**Armadilha metodológica registrada:** a TS-AUC é transversal, então **subamostrar séries muda a
própria métrica** — o subconjunto de 500 séries tem TS-AUC geral 0,8098 contra 0,6100 do conjunto
completo. Deltas medidos em subamostra de séries não estimam os deltas reais. Diferente de uma
métrica por linha, aqui não existe atalho por amostragem.

### A lição que sobrevive ao resultado negativo

Duas ampliações de largura independentes agora regrediram com a **mesma assinatura**: o V5 (+7 BOCPD,
−12 podadas) com Δ −0,0042 e pior em `50<t≤150` (−0,0114); e F1.a+F1.b-1 (+9 colunas `_cal`) com
Δ −0,0069 e pior em `50<t≤150` (−0,0136). Duas causas completamente diferentes, o mesmo bucket. Isso
promove a pegadinha de `feature_fraction=0,8` (`NOTAS_AGENTES.md` §7) de anedota a **mecanismo
esperado**: largura sem ordenação nova dilui o sorteio, e `50<t≤150` é onde dói.

O corolário prático é o rastreio de §1.5: **medir redundância transversal antes de gastar um ciclo de
R0.** Ele previu este resultado (corr 0,92–0,98 = "Δ esperado ~0") por minutos de cômputo, contra os
~50 min do ciclo. Passa a ser passo obrigatório antes de qualquer braço de calibração.

### A regra que a frente de centragem acrescenta

**Todo teto estimado com oráculo tem de passar por um teste honesto antes de justificar trabalho.**
"Oráculo" aqui significa qualquer estimativa que use o rótulo — inclusive indiretamente, como
"selecionar as linhas pré-quebra". O padrão de teste é o mesmo em qualquer frente: *estimar num
subconjunto de linhas, avaliar em outro*. Se o ganho evapora, o teto era contabilidade, não sinal.

Custa minutos. Neste caso teria poupado a construção inteira da frente — e o número que ela produziu
(+0,267 em `t≤50`) era grande o bastante para justificar quase qualquer investimento, que é
exatamente o que torna esse tipo de erro caro.

---

**Status.** Nenhuma mudança de produção adotada. O que ficou: infraestrutura testada (replay de
`e_vol`, réplicas com reinício, tabela de transiente, `kind="cumsum"`), dois blocos prontos e
desligados (`mismatch`, `trajectory`), três ferramentas de diagnóstico novas
(`xs_redundancy`, `detectability_report`, CE6 estendido) e uma frente cancelada por medição.

---

## 1. Onde o plano consolidado desalinha do código

Três desalinhamentos materiais. Todos **reduzem** o trabalho, nenhum aumenta.

| Frente do plano | Estado real | Consequência |
|---|---|---|
| **F1** "null personalizado" — tratado como construção de P0 | **Já implementado** desde o V2 em [`state/calibration.py`](../src/sbrt/state/calibration.py). `fit_h0` → `H0Params.null_stats` → `scorer.py:apply_calibration`. Cobre **46 de 183** features | F1 vira *extensão de cobertura*. Custo ~0 µs/passo |
| **F6.1** "calibrar `cusum_dep`, `accum_window_rho1_fz` (mortas)" — P2 | Mesmas features, mesma mecânica de F1 | **F6.1 ≡ F1.** Item único, medido primeiro |
| **F2** "congelar o filtro AR(10)" — P0 | **Já congelado.** `H0Params` é `frozen=True` e não tem `.refit()` (bloqueio B2). `h0.py:whiten_step` aplica φ do histórico no online: **`e` É o erro de predição um-passo do filtro congelado** | Variância residual (`accum_window_var_ln_w*`, que é log da razão online/histórico pois `e` já vem dividido por `sigma_e`) e ρ₁ residual (`accum_*_rho1_fz`, `dep_absrho1/sqrho1`) já são emitidas. **Falta só** brancura multi-lag sobre `e` e score-CUSUM |
| **F4** "trajetória do estatístico" | Ausente, e **não cabe no contrato**: `StateBlock.update(e, e_raw, e_vol, t)` não vê as saídas dos outros blocos | Exige extensão de contrato — ver F4 |
| **F5** "precursores no passo 1" | [`state/fingerprint.py`](../src/sbrt/state/fingerprint.py) já emite 9 descritores só-histórico, 0 µs/passo | Extensão barata, **mas gated por CE6** — ver F0.d |
| **F9** "detectabilidade estimada" | `artifacts/reports/break_type_census.csv` já tem `delta_rho1`, `delta_logvar_e`, `delta_kurt`, `delta_exceed`, `n_post` por série | Insumo pronto para F0.b |

### As 91 features online sem `_cal` — o alvo real de F1

```
cusum      21   incl. cusum_dep_pos/neg + 6 idades         <- o item do compass Estágio 1
accum      24   incl. accum_global_rho1_fz, accum_window_rho1_fz_w100,
                     ewma_var_ln_l{050,100,300}, window_mean_z_w{010..250},
                     sign_z, skew_z, qcross_{mid,low}, volvol_cv, ewma_exceed2_z, welford_*
bayes      15   log-odds e idades do filtro bayesiano
conformal   4   logm_{abs,abs_reset,right,sign}
ranktwo     2   shape_chi2_w{025,100}
dep         1   dep_mass_evol_w100
varloc      1   argmax_lnscale
hedge       2   opera sobre x cru — fora do escopo de nulo em e
meta       21   constantes por série — corretamente NÃO calibráveis
```

### O bloqueio único de F1, e ele é pequeno

`calibration.py` documenta por que parou em 46: *"Só estatísticas baseadas em `e`. As de média usam
`e_vol`, cuja reprodução exigiria replicar a EWMA de volatilidade sobre o histórico"*. Quase toda a
lista acima é `e_vol`-based.

Essa EWMA é determinística e trivialmente reproduzível — `scorer.py:91-95`:

```python
use_vol_adjust = h0.rho1_abs_e > cfg.state.vol_adjust["threshold_rho1_abs"]
v = ewma_update(v, e * e, cfg.state.vol_adjust["lambda_v"])   # v0 = 1.0
e_vol = e / math.sqrt(max(v, 1e-12))
```

Replayar isso sobre `e_hist` desbloqueia **toda** a lista de uma vez. **É o item habilitador (F1.0) e
deve ser o primeiro commit.**

### Três achados que mudam decisões

1. **CE6 já mede 0,5067** ([`scripts/ce6_history_classifier.py`](../scripts/ce6_history_classifier.py),
   28 features só-histórico, taxa-base 0,4967): o histórico sozinho **não prevê se a série quebra**.
   É evidência medida contra a premissa de F5. Não é refutação completa — CE6 mede *existência* de
   quebra, F5 quer *timing precoce* — mas justifica gatear F5 antes de gastar código.
2. **`conformal_logm_{abs,right,sign}` são acumuladores sem reset.** Sob H0 o incremento tem
   esperança negativa (`log ε − ε + 1 < 0` para ε≠1, já que `E[log p] = −1` com p~U(0,1)), então
   derivam **linearmente em t**. Isto motivou o `kind="cumsum"` de `NullSpec` (μ ∝ t, σ ∝ √t).

   > **CORREÇÃO MEDIDA (2026-07-21).** A leitura original aqui era que a inclinação *varia por série*,
   > e que o nível seria portanto "escala idiossincrática pura" — o argumento de F1 na sua forma mais
   > nítida. **Errado na magnitude.** Medido sobre painel heterogêneo (54 séries H0, 6 dinâmicas):
   > a deriva é −0,347/passo com desvio ENTRE séries de **0,004** — CV de 1,2%. Ela é praticamente
   > *universal*, e a teoria diz por quê: `log ε − ε + 1` depende só da grade de ε, não da série.
   > No passo `t`, a dispersão transversal causada pela deriva é **2–6%** do espalhamento total, e por
   > ser quase comum a todas as séries é quase exatamente **neutra por C1** — o mesmo motivo pelo qual
   > recalibração pós-hoc global não funciona. O mecanismo era real; a alavanca, não.
   > **Previsão revista para F1.b-1: Δ ≈ 0.** Registrada antes do R0, não depois.
3. **O V5 regrediu porque empacotou duas mudanças** (BOCPD + poda), deixando o efeito de cada uma
   não-identificado ([`HISTORICO.md`](HISTORICO.md) §9, pendência #1 de `NOTAS_AGENTES.md` §6).
   F1 tem 91 colunas candidatas — empacotar repete o erro. **F1 vai em braços separados.**

---

### 1.5 O rastreio que deveria vir antes de qualquer braço de calibração

A correção acima produziu a ferramenta que faltava no protocolo:
[`scripts/xs_redundancy.py`](../scripts/xs_redundancy.py).

A TS-AUC só enxerga a **ordenação dentro de cada passo** (C1). Disso seguem dois testes que dispensam
treinar: (a) se a coluna calibrada tem correlação ~1 com a crua **dentro do passo**, ela não pode
reordenar nada — Δ esperado ~0, por melhor que seja a teoria por trás; (b) qualquer componente
**comum a todas as séries** naquele passo é exatamente neutro. Custo: minutos, contra ~50 min do
ciclo build+treino+R0 que ele tria.

Medido para os 9 candidatos de F1 (painel heterogêneo de 54 séries H0, passos 30/60/120):

| coluna | corr_xs | leitura |
|---|---|---|
| `conformal_logm_abs_reset` | **0,695** | reordena de verdade — o único claramente promissor |
| `conformal_logm_abs` | 0,920 | marginal |
| `cusum_dep_neg` | 0,939 | marginal |
| `dep_mass_evol_w100`, `cusum_dep_pos`, `accum_*_rho1_fz` | 0,956–0,967 | redundante |
| `conformal_logm_{right,sign}` | 0,977–0,982 | redundante |

**Rastreio, não veredito** — 0,95 ainda deixa 5% de variação residual, e uma árvore pode usá-la se o
resíduo cair na região certa. Mas a expectativa honesta para o R0 de F1 passa a ser **pequena ou
neutra**, e fica registrada *antes* do resultado.

O painel precisa ser **heterogêneo**: a primeira medição usou 40 séries i.i.d. e deu 0,88–0,98, um
número sem informação — com séries homogêneas o nulo é praticamente o mesmo para todas, calibrar vira
transformação afim com as MESMAS constantes, e a ordenação não muda por construção. Rodar o rastreio
sobre painel homogêneo produziria a conclusão certa pelo motivo errado.

## 2. Stack reordenado

Numeração do plano consolidado preservada. Latência medida: **948,7 µs/passo**
(`artifacts/reports/latency_v5.json`), gate 1500 → **~550 µs de folga**.

| # | Item | Arquivos | µs/passo | Fase | Status |
|---|---|---|---|---|---|
| F0.a | XS-SHAP do V4 (P1–P4) | `shap_report.py` | offline | **P0** | aguarda dataset |
| F0.b | Detectabilidade por série | `detectability_report.py` (novo) | offline | **P0** | **feito** (script pronto) |
| F0.c | V4 + BOCPD sem a poda | `scorer.py` | ~30 | **P0** | [ ] |
| F0.d | CE6 estendido — gate de F5 | `ce6_history_classifier.py`, `fingerprint.py` | offline | **P0** | **FEITO — F5 morto** |
| F1.0 | Replay de `e_vol` sobre o histórico *(habilitador)* | `calibration.py`, `numerics.py` | 0 | **P0** | **feito** (infra) |
| F1.a | Calibrar **só dependência** (braço isolado) | `calibration.py`, `cusum.py`, `accumulators.py` | 0 | **P0** | **medido → REVERTIDO** |
| F1.b-1 | Calibrar os martingales conformais | `conformal.py`, `calibration.py` | 0 | **P0** | **medido → REVERTIDO** |
| F1.b-2..4 | Calibrar o restante (accum/cusum/bayes/ranktwo/varloc) | YAML | 0 | ~~P0~~ | **desaconselhado** — rastreio §1.5 |
| F2 | Bloco de mismatch *(absorve F6.2/F6.3)* | `state/mismatch.py` (novo) | O(L) | **P1** | **código+testes**, não ligado |
| F4 | Trajetória + integrador *(absorve F9)* | `state/trajectory.py` (novo) | O(k) | **P1** | **código+testes**, não ligado |
| ~~F5~~ | ~~Precursores terminais~~ | — | — | — | **CANCELADO por F0.d** |
| F3 | Row-weighting por `w_t` | `model/weights.py` | offline | **P2** | [ ] |
| F7 | Forma distribucional (um representante) | `state/` novo | O(w) | **P2** | [ ] |
| F8 | Multi-representação ×3 | `scorer.py` | ×3 núcleo | **P3** | [ ] |
| F10 | NEWMA como feature | `mmd.py` | O(1) | **P4** | [ ] |

**"Não ligado" é deliberado.** `MismatchBlock` e `TrajectoryBlock` existem, com testes de premissa
passando, mas **não** estão em `scorer.py:default_blocks`. Entram como braços próprios depois que o
R0 de F1 fechar — ligar agora empacotaria três mudanças num retreino só e reproduziria exatamente a
falha do V5 (efeito não-atribuível). Ligar cada um custa uma linha.

**Regra de dependência:** F1.0 antes de qualquer coisa em F1. F1 completo antes de F2/F4/F5/F7 —
toda feature nova nasce com seu `_cal`, senão reintroduz o problema de escala que F1 acabou de
corrigir (`NOTAS_AGENTES.md` §11 item 4 já exige isso).

---

## 3. Itens

Cada item segue `NOTAS_AGENTES.md` §5: **hipótese e bucket-alvo registrados por escrito ANTES de
medir** (anti garden-of-forking-paths — sem isso o IC do R0 perde sentido).

### F0 — Gate diagnóstico *(P0, sem código de produção)*

#### F0.a — XS-SHAP do V4
Pendência aberta #4: as 42 features de P1–P4 nunca tiveram XS-SHAP individual medido
(`shap_v2.csv` tem 137 linhas = V2; o V4 tem 183 features). O V5 já podou `lmom_*` e `dep_*_w050`
sem esse número publicado. `shap_report.py` já emite a coluna `xs_shap` correta — desvio da
contribuição *dentro* de cada passo, ponderado por `w_t`, a única medida válida sob C1.

```bash
python scripts/shap_report.py --model artifacts/models/v4 --out artifacts/reports/shap_v4.csv
```

**DoD:** lista de features P1–P4 com `xs_shap` abaixo do limiar, candidatas a poda, registrada
**antes** de qualquer adição de largura.

#### F0.b — Detectabilidade por série
Script novo `scripts/detectability_report.py`: join de `break_type_census.csv` (divergência por eixo
+ `n_post` por série, já pronto) com `artifacts/models/oof_v4.parquet` por `id`. Estimar
divergência × `n_post` e cruzar com o erro OOF em `t≤50`.

**DoD:** repartição estimada — causa (1) informação / (2) objetivo / (3) indetectável — do peso de
erro em `t≤50`. **Se (3) domina, F6/F7 perdem teto e o plano deve parar em P1.**

##### Resultado (2026-07-21) — CORRIGIDO: a causa (3) domina em `t≤50`, e o bucket quase não importa

> **A primeira versão desta seção estava contaminada** pela armadilha do `t` (ver §1 acima): usava o
> gap não-controlado, e por isso concluía "causa (2), 91% das séries com sinal". Refeita com o efeito
> medido contra a seção transversal do mesmo passo, a conclusão **inverte**.

Efeito controlado por `t`, por quintil de detectabilidade (712 séries com quebra e ambos os rótulos
em `t≤50`; dp transversal do score = 0,0484):

| quintil | divergência | `m` no bucket | efeito | frac ≤ 0 |
|---|---|---|---|---|
| 0 | 0,71 | 6 | −0,0018 | 52% |
| 1 | 1,11 | 14 | −0,0030 | 55% |
| 2 | 1,39 | 20 | −0,0013 | 53% |
| 3 | 2,01 | 26 | +0,0047 | 41% |
| 4 | 3,33 | 31 | +0,0101 | 36% |

**Em `t≤50` o modelo não detecta nada em 60% das séries quebradas** — 52–55% com efeito ≤ 0 é
cara-ou-coroa. O gradiente é monótono na detectabilidade, o que valida o estimador e sustenta a
leitura de **piso informacional** (causa 3) nos três quintis de baixo: com `m` de 6 a 20 observações
pós-quebra e divergência pequena, nenhum algoritmo separa.

##### E o achado que reorganiza a prioridade do projeto inteiro

| bucket | **peso da métrica** | TS-AUC | quanto rende um ganho de +0,05 ali |
|---|---|---|---|
| 1–50 | **8,1%** | 0,5357 | +0,0041 |
| 51–150 | 26,6% | 0,5799 | +0,0133 |
| **151–400** | **48,7%** | 0,6242 | **+0,0243** |
| 401+ | 16,5% | 0,6529 | +0,0083 |

**Metade do peso está em 151–400.** O projeto vem otimizando `t≤50` — os dois relatórios, o plano
consolidado e o bucket-alvo declarado em cada R0 — e esse bucket carrega 8,1%. Um ganho de +0,05 ali
move o agregado em +0,004, **abaixo do ruído do próprio instrumento** (2 desvios do bootstrap
pareado ≈ 0,006). Declarar `t≤50` como bucket-alvo a priori é auto-derrotante: mesmo um sucesso
grande é indetectável no agregado.

E há folga onde o peso está. Efeito controlado por `t` em 151–400 (1.597 séries, dp 0,1120):

| tercil de divergência | efeito | frac > 0 |
|---|---|---|
| 0,48 | −0,00 dp | 50% |
| 0,93 | +0,02 dp | 51% |
| 2,09 | +0,31 dp | 68% |

Dois terços das séries quebradas estão no acaso **apesar de haver 150+ observações pós-quebra**.
Diferente de `t≤50`, aqui não é piso — é eixo de informação faltando. É onde F2/F7 devem ser medidos.

##### Resultado original (contaminado, preservado para rastreabilidade)

712 séries com quebra e ambos os rótulos em `t≤50`; TS-AUC do bucket = 0,5357.

| quintil de detectabilidade | divergência | `m` no bucket | gap mediano | frac. gap ≤ 0 |
|---|---|---|---|---|
| 0 (menos detectável) | 0,71 | 6 | 0,0278 | 15% |
| 4 (mais detectável) | 3,33 | 31 | 0,0621 | 2% |

O gradiente é monótono e limpo — o estimador ordena o que deveria ordenar. **Mas a causa (3) não
domina:** mesmo no quintil mais detectável o gap mediano é só 0,062. E o número que reorganiza a
leitura:

```
gap mediano DENTRO da série (t<=50):     0,0408
dp do score ENTRE séries no mesmo passo: 0,0503   (ponderado por n_pos*n_neg)
razão gap/espalhamento:                  0,81
fração de séries com gap > 0:            91%
```

**O modelo acerta a direção em 91% das séries — o sinal está nas features.** O que o impede de virar
ranking é que o deslocamento de base entre séries, no mesmo passo, é da mesma ordem de grandeza que o
efeito da quebra (razão 0,81). É a causa (2) do filtro da Seção 0, medida em vez de suposta, e explica
como a TS-AUC do bucket fica em 0,5357 com o gap interno quase sempre positivo.

**Consequência para o desenho de F1** (ver §1.5): a alavanca não está em calibrar features
individuais — o rastreio mostra que as calibradas reproduzem a ordenação das cruas. Está em remover o
**deslocamento de base por série do próprio score**, que é uma intervenção diferente e ainda não
tentada. Ela escapa da invariância C1 justamente por ser *por série* (um transform comum seria
neutro), mas só é legítima se a escala vier **exclusivamente do histórico** — nada de estimar o
offset com dados online.

**Cuidado de leitura (`m` no bucket).** A primeira versão deste relatório usou `n_post` da série
inteira como tamanho amostral. Isso infla a detectabilidade das quebras precoces — que são *todas* as
deste recorte, já que ter ambos os rótulos em `t≤50` exige `tau < 50` — e o gradiente sumia
(0,037 → 0,056, sem ordem). Corrigido para `m = min(t_max − tau, n_post)`, o que o detector de fato
viu dentro do bucket.

#### F0.c — V4 + BOCPD sem a poda
Resolve o "BOCPD não ajuda" que ficou não-demonstrado (pendência #1). `state/bocpd.py` e
`configs/default.yaml:bocpd` estão preservados; basta uma linha em `scorer.py:default_blocks`.
Custo: rebuild de dataset (~9 min) + treino (~15 min) + R0. ~30 µs/passo medidos.

**Bucket-alvo declarado a priori: `t≤50`.** (O TCPD dá razão empírica para BOCPD ser competitivo, e
o argumento do V5 era reação rápida.)
**DoD:** R0 contra `oof_v4.parquet` com IC 95%; adotar só se o IC excluir 0 no agregado ou em `t≤50`.

#### F0.d — CE6 estendido *(gate de F5)*
Acrescentar os descritores de precursor a `fingerprint.py` e às `FEATURE_NAMES` de
`ce6_history_classifier.py`; rerodar. **Estender a formulação**: hoje CE6 pergunta "a série quebra?";
F5 pergunta "a quebra é precoce?". Acrescentar um segundo alvo — prever `tau_index ≤ k` **entre as
séries que quebram**.

**DoD:** se CE6 permanece ≈0,50 nas duas formulações, **F5 é cancelado** antes de escrever código
online. Se sobe, é achado grande (o gerador correlaciona histórico com quebra), reabre decisões de
projeto e a política de §12.2 — escalar antes de prosseguir.

---

### F1.0 — Replay de `e_vol` sobre o histórico *(P0, habilitador)*

Em `calibration.py`, `_history_evol(e_hist, h0, cfg) -> np.ndarray` reproduzindo exatamente
`scorer.py:91-95` (ver §1 acima). Passar `e_vol_hist` a todos os `history_null_series(...)` que
precisarem.

**Teste obrigatório** em `tests/unit/test_calibration.py`: equivalência bit-a-bit entre este replay e
a EWMA do `StreamScorer` alimentado com a mesma sequência. Um desalinhamento aqui envenenaria
silenciosamente **todas** as features calibradas — é exatamente o risco que motivou os testes
dedicados de MMD/Haar.

**Nota:** `dependence.py:history_null_series` hoje aproxima `e_vol ≈ e` no histórico e por isso
exclui `dep_mass_evol_*` da calibração. Com F1.0 essa aproximação pode ser removida — mas isso muda
o nulo das features `dep_*` já calibradas, então **é uma mudança de comportamento e precisa de R0
própria**, não pode pegar carona.

---

### F1.a — Calibrar só a dependência *(P0, braço isolado)*

**Hipótese a priori:** *quebras de dependência pura pontuam 0,492 — abaixo do acaso — porque as
features de dependência linear não são comparáveis transversalmente. Calibrá-las contra o nulo da
própria série tira o eixo do sub-acaso.*
**Bucket-alvo: agregado, estratificado pelo eixo `delta_rho1` do censo A1.**

Escopo deliberadamente estreito — **5 colunas novas**:
`cusum_dep_pos`, `cusum_dep_neg`, `accum_global_rho1_fz`, `accum_window_rho1_fz_w100`,
`dep_mass_evol_w100`.

Mecânica: seguir o padrão já estabelecido por `dependence.py:history_null_series` — rodar o **bloco
real** sobre o histórico, não uma reimplementação vetorizada (garante equivalência online/nulo por
construção).

- `cusum.py`: nova `history_null_series(e_hist, e_vol_hist, h0, cfg)`. `CusumBlock.reset` precisa de
  `h0` (usa `sigma_u` e quantis) — disponível em `fit_h0`.
  **Ponto de design:** o CUSUM é um passeio aleatório refletido com deriva negativa sob H0 → nulo
  estacionário, `kind="none"`; mas exige **burn-in**: descartar o transiente inicial antes de medir
  μ/σ, como `calibration.py:207` já faz com as EWMAs do MMD.
- `accumulators.py`: `history_null_series` para os dois `rho1_fz`. Ambos são Fisher-z escalados por
  √n → `kind="rho"`, disponível desde t≈10 via transporte de escala (`_null_at`).

**DoD:** R0 contra `oof_v4.parquet` com IC 95% pareado; `detect` do eixo dependência sai de 0,492
estratificado pelo censo A1; determinismo bit-a-bit PASS; latência inalterada (0 µs/passo esperado).

---

### F1.b — Calibrar o restante *(P0)*

**Como abrir cada sub-braço:** acrescentar linhas em `calibration.recursive_features` no YAML
(`nome: none` para recursão refletida, `nome: cumsum` para acumulador sem reset) e reconstruir. Não
há código a escrever — a máquina de réplicas cobre qualquer bloco que exponha `history_null_series`
com a assinatura de réplicas (`cusum`, `accumulators`, `conformal` já expõem; `bayes_filter` ainda
não).

Só depois de F1.a medir. Ordem por valor esperado:

1. ~~**`conformal_*` (4)**~~ — **FEITO (F1.b-1).** `kind="cumsum"` implementado em `_null_at`
   (μ(t) = t·deriva, σ(t) = √t·σ₁, ambos medidos por réplicas). `logm_abs_reset` é refletido em 0 e
   portanto estacionário → entrou como `none`.
   **Caveat resolvido por medição, não por complexidade:** rodar o bloco sobre o próprio histórico
   compara cada ponto contra uma ECDF que o contém, enquanto o online compara pontos novos. O viés no
   p-value de mid-rank é O(1/n_h) — com n_h típico de 3.000, ~0,03%, desprezível frente ao dp do
   nulo. Split/leave-one-out seria complexidade sem retorno; registrado na docstring de
   `conformal.history_null_series`.
2. **`accum_*` de variância/forma (24)** — `ewma_var_ln_*` (`kind="var_ln"`, teoria conhecida),
   `window_mean_z` e `sign_z` (`kind="z"`), `skew_z`, `qcross_*`, `volvol_cv`, `ewma_exceed2_z`.
3. **`cusum_*` restantes (19)** e **`bayes_*` (15)** — as idades são inteiros com massa em 0;
   `(x−μ)/σ` continua monotônico por série, ok, mas medir em braço separado.
4. **`ranktwo_shape_chi2` (2)**, **`varloc_argmax_lnscale` (1)**.

**Risco a vigiar:** cobertura total leva o modelo de 183 → ~274 colunas. Com `feature_fraction=0,8`,
largura extra sem sinal dilui o sorteio — a pegadinha medida de `NOTAS_AGENTES.md` §7. Por isso os
sub-braços vão em ordem, e `xs_shap ≈ 0` num sub-braço é motivo para **não adicionar aquele grupo**,
mesmo que o Δ agregado seja neutro.

---

### F2 — HIPÓTESE REGISTRADA A PRIORI (2026-07-22, antes de qualquer medição)

> **Hipótese.** O filtro AR(10) do histórico é congelado no online, mas o banco nunca testou a
> brancura dele de forma multi-lag sobre o nível `e` puro — só sobre `|e|` (`dep_mass_abs`) e sobre
> `e_vol` (`dep_mass_evol`). Uma quebra que muda a estrutura de dependência em lags > 1 deixa o
> resíduo do filtro congelado correlacionado, e nada no banco vê isso. Acrescentar a massa
> Σρ_k² sobre `e`, o portmanteau de McLeod-Li sobre `e²` e um CUSUM de escore multi-lag deve
> aumentar a separação transversal onde há janela suficiente para as estatísticas se estabilizarem.
>
> **Bucket-alvo declarado: `151–400`** (48,7% do peso da métrica). Escolhido porque é onde está a
> alavancagem e onde a medição de 2026-07-21 mostra folga real: dois terços das séries quebradas
> estão em acaso ali *apesar de 150+ observações pós-quebra* — não é piso informacional, é eixo
> faltando. **Não** declarar `t≤50`: 8,1% do peso, e um ganho de +0,05 lá rende +0,004, abaixo de
> 2 desvios do bootstrap pareado.
>
> **Critério de adoção.** IC 95% do Δ pareado excluindo 0 a favor em `151–400` ou no agregado.
>
> **Escopo.** 7 colunas, **sem versões `_cal`** — a evidência desta sessão é que a calibração produz
> duplicata ruidosa (F1: Δ −0,0069). 183 → 190 features.
>
> **Rastreio prévio (portão, passou).** `xs_redundancy.py --vs-all mismatch_`: correlação transversal
> máxima contra QUALQUER coluna existente entre 0,569 e 0,869 — todas abaixo de 0,90, ou seja, eixo
> novo e não duplicata. A mais próxima é `white_e_w050` contra `dep_mass_evol_w100` (0,869), o que é
> coerente: ambas são massa multi-lag, uma sobre `e`, outra sobre `e_vol`.
>
> **Latência.** 1105,8 µs/passo contra gate de 1500 (V4 media 1062 na mesma sessão) — +44 µs.

### F2 — Bloco de mismatch *(P1, absorve F6.2/F6.3)*

Novo `state/mismatch.py`, `StateBlock` padrão. **Não reimplementar o filtro congelado — ele já
existe.** O que falta é medir a brancura dele de forma multi-lag e no fluxo certo:

1. **Portmanteau sobre `e`** (escala congelada): Σ_{k=1}^{L} ρ_k² em janelas `w ∈ {10, 25, 50}`.
   `dep_mass_abs_w100` já faz isso para `|e|` e `dep_mass_evol_w100` para `e_vol` — **falta o nível
   `e` puro**, que é o teste direto de "o AR(10) do histórico ainda branqueia". Reusar
   `dependence.py:_RollingAutocorr` (já incremental, O(L)/passo).
2. **Efeito ARCH:** ARCH-LM de Engle / portmanteau de McLeod-Li em janela curta. Calibração
   sequencial vinda do **escore**, não de `e²` cru (Berkes et al. 2004; e `dep_sqrho1_w*` já ocupa o
   `e²` cru).
3. **Score-CUSUM de AR** (Na-Lee-Lee / Gombay-Serban), O(1) para p fixo, janelas `w ∈ {10, 25, 50}`,
   nulo transportado por 1/√n para liberar a feature desde t≈10.

Registrar parâmetros em `configs/default.yaml:mismatch` com custo medido. Emitir `_cal` para todos.
**Bucket-alvo: `25 < m` pós-quebra, não `m` minúsculo** — o teto causal em `t≤50` (mediana 14 obs
pós-quebra) é real e nenhuma feature o elimina.
**DoD:** ganho com IC 95%; sentinelas T7/T8/T11 sem regressão; latência dentro dos ~550 µs de folga.

---

### Os três eixos NÃO COBERTOS — implementados 2026-07-22, desligados até o R0 do F2 fechar

Depois do F2, o banco cobre bem: nível de variância, cauda, dependência linear em janelas, energia
por escala, evidência bayesiana/conformal e agora brancura multi-lag. O xs-SHAP (F0.a) diz onde está
a massa: `meta_h0` 34,9%, acumuladores 14,7%, MMD 12,1%. **Tudo isso é a mesma família: nível.**

Três direções ficaram fora de qualquer bloco existente, e cada uma tem uma propriedade que a torna
*estruturalmente* incapaz de duplicar a família de nível:

| bloco | arquivo | colunas | propriedade que garante ortogonalidade |
|---|---|---|---|
| `SpectralBlock` | `state/spectral.py` | 8 | razões de potência: **invariante a escala** |
| `OrdinalBlock` | `state/ordinal.py` | 5 | só lê ordem: **invariante a qualquer transformação monótona** |
| `MultiRepBlock` | `state/multirep.py` | 6 | ponte auto-normalizada: **invariante a shift e escala** |

Isto não é retórica: uma quebra pura de variância multiplica o fluxo por uma constante e **não move
uma única dessas 19 colunas**. Elas só podem contribuir onde o modelo hoje é cego — e o ponto cego
está medido: quebras puras de dependência têm detectabilidade 0,492, *abaixo do acaso*
(docs/INVESTIGACAO_FALHAS_V3.md §1).

**O que cada uma acrescenta que não existia**

1. **Espectral** — o banco mede dependência só no domínio do tempo. `dep_mass` e `mismatch_white`
   somam ρ_k²: são cegos à **direção** da dependência. O centroide espectral tem sinal — persistência
   positiva o derruba, alternância o levanta. Medido (40 séries de 2000 pontos): H0 = 0,492 ± 0,039;
   AR(1) φ=+0,6 → 0,297; φ=−0,6 → 0,718. Cinco desvios para cada lado.
2. **Ordinal (Bandt-Pompe)** — a entropia de permutação é o análogo não-paramétrico da massa
   multi-lag, sem escolher lag e sem supor linearidade. E a **irreversibilidade temporal**
   (½·Σ|p(π) − p(π^R)|) é zero em esperança sob *todo* processo linear gaussiano; nenhuma feature do
   banco tem essa propriedade, porque variância, |e|, e², ρ_k e Haar são simétricas no tempo por
   construção.
3. **Multi-representação** — todo detector sequencial do banco é do tipo **supremo** (CUSUM, MAP,
   max de z, martingale). A teoria clássica separa sup-type (forte contra uma quebra brusca) de
   integral-type/Cramér-von Mises (forte contra deriva difusa, quebras múltiplas pequenas, e quebra
   **perto da borda da janela**, onde o pico ainda não somou evidência mas a ponte inteira já se
   deslocou). O banco não tinha *nenhuma* estatística integral. Aplicada a três representações:
   `e` (média), `e²` (variância), PIT contra o H0 (distribuição, livre de cauda).

**A representação diferenciada foi descartada por álgebra, não por medição:** com `d = e_t − e_{t-1}`
a soma parcial telescopa e a ponte colapsa em `n·var(janela)` — é `accum_window_var_ln_*` com outro
nome. Ela estava na proposta original; teria custado 30 min de build para o rastreio reprovar.

**Rastreio prévio (portão, passou nas 19).** `xs_redundancy.py --extra-blocks spectral ordinal
multirep --vs-all spec_ ord_ mrep_ --seeds 8 --t-max 300 --at-steps 60 200`, painel de 48 séries H0
de 6 dinâmicas: max|corr| transversal contra QUALQUER coluna existente entre **0,525 e 0,882** —
todas abaixo de 0,90 (para comparação, o F2 deu 0,569–0,869). As mais próximas são `mrep_kpss_rank_w050`
contra `mismatch_score_cusum_pos` (0,882) e `spec_centroid_slow` contra `accum_global_rho1_fz`
(0,865), ambas coerentes com a teoria e ainda com >20% de variação residual.

**Um erro de desenho pego pelo teste, registrado porque é reutilizável.** A primeira versão do
`SpectralBlock` usava `|z_k|²` cru. O periodograma num ponto do tempo é exponencialmente
distribuído — desvio igual à média, *por mais longa que seja a série*. Com K=6 as proporções viram
Dirichlet(1,…,1), a entropia de H0 cai para 0,81 **± 0,10 entre séries i.i.d.**, e o teste de sinal
reprovou: a série AR(1) ficou ACIMA da branca. A correção (média de Welch: segunda EWMA sobre a
potência, taxa bem mais lenta que a janela da DFT) levou H0 a 0,982 ± 0,012. Lição geral: quando a
feature é uma **razão de estimativas ruidosas**, o ruído da razão não some com t — só com promediação
explícita.

**RESULTADO FINAL (2026-07-22): MREP ADOTADO, SPEC e ORD não.**

Médias de **7 sementes limpas** por lado (777/101/202/303/404/505/606 — a 42 fica fora, ver "A
semente 42 está contaminada" abaixo), todos da mesma build, braços separados por `--drop-prefix`:

| modelo | features | média de 7 | dp | Δ vs V4 |
|---|---|---|---|---|
| V4 | 183 | 0,6020 | 0,0009 | — |
| SPEC | 191 | 0,6039 (n=3) | — | +0,0021 |
| ORD | 188 | 0,6031 (n=3) | — | +0,0015 |
| **MREP** | **189** | **0,6056** | 0,0024 | **+0,0036** (3,6 EP) |

Empacotados: V4 **0,6068** → MREP **0,6110**. IC pareado (`compare_mrep_bag7_FINAL.json`), bucket-alvo
`150<t≤400` declarado a priori antes de qualquer medição:

| bucket | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| geral | **+0,0042** | [+0,0022, +0,0063] | **sim, a favor** |
| `t≤50` | **+0,0115** | [+0,0030, +0,0193] | **sim** |
| `50<t≤150` | +0,0056 | [+0,0013, +0,0097] | **sim** |
| `150<t≤400` (alvo) | +0,0034 | [+0,0012, +0,0058] | **sim** |
| `t>400` | +0,0006 | [−0,0015, +0,0024] | não |

Latência: **1046,8 µs/passo contra gate de 1500 → PASS** (V4 puro mede 926,9; o bloco custa ~+120 µs).
Suíte: 182 testes passam.

**Quanto a estimativa andou com o número de sementes** — vale registrar, porque é o argumento contra
decidir com poucas: +0,0067 (n=3) → +0,0050 (n=4) → +0,0046 (n=5) → +0,0041 (n=6) → **+0,0036** (n=7).
As três primeiras sementes calharam de ser as altas. **Com n=3 eu teria reportado um efeito 86% maior
que o real.**

**Robusto à decisão de excluir a semente 42**, tomada depois de ver os dados: incluindo-a nos dois
lados o Δ fica em +0,0036 também. A exclusão muda a magnitude intermediária, não a conclusão.

**Custo próprio do bloco:** o MREP tem dp entre sementes de 0,0024 contra 0,0009 do V4 — 2,7× mais
variância de sorteio, e duas das sete sementes caem dentro da faixa do baseline. O ganho é na média,
não em cada entrega. Reforça que a submissão deveria ser um modelo **empacotado**, não um sorteio.

O maior ganho relativo está em `t≤50` (+0,0161), e isso é **coerente com a teoria do bloco, não
contra ela**: a estatística tipo-integral é forte quando a quebra está perto da borda da janela — o
CUSUM ainda não somou evidência, mas a ponte inteira já se deslocou. É exatamente o regime de t
pequeno, onde o projeto vinha atribuindo o teto a limite informacional.

SPEC e ORD ficam abaixo da barra de 0,0058 e **não são adotados**. Ficam escritos, testados e
desligados.

**A lição de método:** passar no rastreio de redundância prova que a direção é **nova**, não que ela
carrega **sinal** — SPEC e ORD são genuinamente ortogonais ao banco e também ao alvo. O que separou
o MREP dos outros dois não foi a novidade da direção (os três passaram no rastreio com folga
parecida), foi ter uma hipótese sobre *em que regime* o banco falha: sup-type contra integral-type,
com efeito previsto perto da borda da janela.

---

**Medições intermediárias (K=4 incluindo a semente 42), preservadas para rastreabilidade.**
Baseline = V4 de 183 features reconstruído da mesma build (a semente 42 reproduz exatamente os
0,6100 históricos — verificado bit-a-bit contra `oof_v4.parquet`, o que valida a reconstrução).

| braço | features | sementes 42/777/101/202 | média | dp | Δ vs V4 |
|---|---|---|---|---|---|
| V4 | 183 | 0,6100 · 0,6006 · 0,6020 · 0,6030 | 0,6039 | 0,0041 | — |
| SPEC | 191 | 0,6035 · 0,6045 · 0,6036 · 0,6036 | 0,6038 | **0,0005** | −0,0001 |
| ORD | 188 | 0,6030 · 0,6011 · 0,6059 · 0,6024 | 0,6031 | 0,0020 | −0,0008 |

**As três médias cabem dentro de 0,0008.** Nenhuma das duas famílias move a TS-AUC, e a barra
declarada a priori era Δ ≥ 0,0058 (2 EP com K=4).

**A lição de método, que vale mais que o resultado:** passar no rastreio de redundância prova que a
direção é **nova**, não que ela carrega **sinal**. As colunas espectrais e ordinais são de fato
ortogonais ao banco — e ortogonais ao alvo também. O rastreio continua útil (evita gastar ciclo em
duplicata), mas é um filtro de *necessidade*, nunca de suficiência.

**Achado lateral, real mas não conclusivo:** o braço espectral tem dp entre sementes de 0,0005 contra
0,0041 do baseline — as quatro sementes caem num intervalo de 0,0010. Razão de variâncias F(3,3)=67
contra crítico 29,5 a 1%. O mecanismo NÃO é a parada antecipada (o nº de árvores continua variando
igual: 378–419 no espectral, 359–404 no baseline), então vem das features: colunas invariantes a
escala e de baixo ruído dariam um "esqueleto" que as árvores encontram independentemente do sorteio
de `feature_fraction`. Isso não aumenta a TS-AUC esperada, mas removeria a dependência de sorte se a
submissão for de sorteio único. **Ressalva: dp com n=4 é fraco, e o dp do baseline é quase todo
produzido por um único ponto (a semente 42, em 0,6100, contra 0,6006/0,6020/0,6030 das outras três —
dp 0,0012 sem ela).** Baseline estendido para 8 sementes em medição para separar sorte de estrutura.

**Estado.** Escritos, testados (18 testes unitários novos, incluindo os de invariância que são a
justificativa inteira), ligados em `default_blocks()` **só para a build de medição**. `scripts/xs_redundancy.py --extra-blocks`
existe justamente para rastrear sem ligar. Entram um por vez, cada um com R0 própria e bucket-alvo
`151–400` declarado a priori, depois que o F2 fechar. Não empacotar — foi assim que o V5 ficou
inatribuível.

---

### F4 — Trajetória + integrador *(P1, absorve F9 — exige extensão de contrato)*

**Problema arquitetural.** F4 quer inclinação/curvatura/persistência **dos estatísticos já
existentes**, mas `StateBlock.update(e, e_raw, e_vol, t)` não recebe as saídas dos outros blocos.

**Solução recomendada** — espelhar o padrão que `apply_calibration` já estabeleceu (consumir o dict
`feats` depois do laço de blocos):

```python
# state/trajectory.py
class TrajectoryBlock:
    def reset(self, h0, cfg) -> None: ...
    def update_from_feats(self, feats: dict[str, float], t: int) -> None: ...   # protocolo novo
    def features(self) -> dict[str, float]: ...
```

Chamado em `scorer.py:update_features` **depois** do `for b in self.blocks` e **antes** de
`apply_calibration`, para que a trajetória também ganhe `_cal`. Respeita a regra "toda feature nova
vive no `features()` de um bloco, nunca solta em `scorer.py`" (`NOTAS_AGENTES.md` §1) e **exige
atualizar §2.1 daquele documento** com o contrato novo.

Emitir, sobre uma **whitelist curta vinda do YAML** (não sobre tudo — multiplica largura):
inclinação, curvatura, monotonicidade, tempo-desde-a-primeira-excedência, persistência da
excedência, área sob o estatístico. Tudo recursivo O(1). Junto, o integrador de F9: Page-Hinkley /
CUSUM lado a lado com o instantâneo (Shewhart) sobre média, variância e resíduo.

**Bônus:** isto exporta a *idade da quebra* que o compass queria para ponderação condicional —
**sem** segundo modelo (F10).
**DoD:** ganho no bucket-alvo **e** queda de falso-positivo nas sentinelas T1/T7 — é a promessa
específica desta frente (separar spike transitório de quebra sustentada).

---

### F5 — Precursores terminais *(P1) — **CANCELADO por medição** (2026-07-21)*

O gate F0.d foi rodado nas duas formulações, e **F5 não sobrevive a ele**.

| Alvo (só-histórico, 10.000 séries, 5-fold) | 28 descritores | +4 precursores |
|---|---|---|
| A série quebra? (taxa-base 0,4967) | 0,5030 | 0,5027 |
| A quebra é precoce, `tau≤50`? (taxa-base 0,2046) | 0,4878 | **0,4798** |

Os precursores testados são exatamente os que a teoria de *critical slowing down* prescreve, medidos
onde ela diz que aparecem — inclinação de AC(1), da variância e da assimetria em janelas rolantes
sobre a **cauda** do histórico, mais o delta bruto de AC(1) entre o início e o fim dessa cauda
(`state/fingerprint.py:compute_precursors`). Nenhum move o número; o alvo de precocidade fica abaixo
do acaso nas duas versões, com 1.016 quebras precoces entre 4.967 — amostra que resolve
folgadamente um efeito real.

Isto fecha a divergência que o plano consolidado deixou aberta. O relatório
`informacao_nao_capturada.md` §3 já registrava a ressalva ("muitas quebras deste desafio são
provavelmente exógenas/abruptas; para essas *não existe* precursor"); a medição diz que, neste
gerador, **elas dominam a ponto de o eixo inteiro não render sinal algum**.

`compute_precursors` fica no repositório, deliberadamente **não** ligado a `compute_fingerprint`:
serve ao gate e documenta o que foi testado, sem custar largura ao modelo. Reabrir F5 exige primeiro
um descritor novo que mova o alvo 2 — não vale construir a frente online antes disso.

---

### F3, F7, F8, F10 — esboço

- **F3** (`model/weights.py`): ponderar linhas pelo conteúdo informacional do passo (∝ `w_t =
  n_pos·n_neg`). Teto baixo — tratar como confirmação, não aposta. **Não** mexer na loss (lambdarank
  regrediu em R3) nem em early stopping por `ts_auc_by_t` (R2, winner's curse — documentado em
  `configs/default.yaml:122-135`). Se o IC 95% não excluir 0, reverter.
- **F7:** escolher **um** representante por eixo (energy distance OU Wasserstein-1 OU KS — a família
  completa já regrediu como redundante) e medir isolado. Checar redundância com F3/F4 via `xs_shap`.
  Bucket-alvo `t>150`.
- **F8:** ×3 representações (bruta / diferenciada / rank) é o maior risco de latência — ~2850 µs
  estimado contra gate 1500. Liberar **uma representação por vez**, só as que passam no bootstrap.
- **F10:** saídas NEWMA como *features* do LightGBM existente — a infraestrutura RFF já está em
  `state/mmd.py` (D=64, seed fixa compartilhada). Segundo modelo + stacking condicional só se o
  caminho feature-only estabilizar. Bloqueadores: risco de originalidade (>95% de correlação com o
  top-10 desqualifica), latência, complexidade.

---

## 4. Verificação

Protocolo de `NOTAS_AGENTES.md` §5, na ordem:

```bash
make ci                    # unit + causality + determinism — sempre primeiro
make dataset               # ~9 min paralelo
make train                 # ~10-15 min → artifacts/models/vN + oof_vN.parquet

python scripts/compare_oof.py \
    --baseline  artifacts/models/oof_v4.parquet \
    --candidate artifacts/models/oof_vN.parquet \
    --target-bucket "t<=50" --n-boot 300 \
    --out artifacts/reports/compare_vN_vs_v4.json

python scripts/ce6_history_classifier.py                     # deve permanecer ≈0,50
make robustness                                              # gates relativos T1-T13 (~1 h, 200 seeds)
make benchmark                                               # gate 1500 µs/passo
python scripts/shap_report.py --model artifacts/models/vN    # coluna xs_shap, NUNCA mean_abs_shap
```

**Critério de adoção:** o IC 95% do Δ pareado exclui 0 no agregado **ou** no bucket declarado a
priori. `compare_oof.py --target-bucket` já implementa exatamente essa regra.

**Escalas de ruído — não confundir os instrumentos.** σ(TS-AUC) ≈ 0,054 com 100 séries; 0,005–0,008
no nível com o OOF de 10.000; muito menor na diferença pareada. Um Δ de 0,004 é irresolvível no
held-out de 100 séries e resolvível no OOF pareado.

**Pendência a resolver de passagem:** `artifacts/reports/submission_log.md` está previsto em
`configs/default.yaml:184` e **não existe** (pendência #2). A âncora oficial nunca foi exercida de
forma registrada. Criá-lo antes da primeira submissão desta série — o plano consolidado o declara
única fonte de verdade para comparação de performance.

---

## 5. O que NÃO fazer

Consolidado dos dois relatórios **mais** o que este repositório já mediu:

- **Empacotar mudanças.** Causa raiz da pendência #1: o V5 juntou BOCPD com a poda e o efeito de cada
  um segue não-identificado. F1 vai em braços por isso.
- Qualquer intervenção no **canal de média** (β multivariado −0,005, encerrado por medição).
- **Recalibração pós-hoc global** (Platt / isotônica / offset) — matematicamente neutra para a
  TS-AUC pela invariância C1.
- **Pesos pareado-consistentes** (R1, Δ=−0,0014), **early stopping por `ts_auc_by_t`** (R2),
  **lambdarank** (R3), **mais funcionais do mesmo contraste de CDF** (JS/Hellinger/W1/KS).
- **Força bruta de milhares de features** — máquina de overfit com n efetivo ~10⁴ séries.
- **Priorizar por `mean_abs_shap`** — mistura variação entre passos com variação entre séries, e sob
  C1 a primeira é exatamente neutra. Só `xs_shap`.
- **Confiar em AUC local absoluta** — só gap-vs-controle pareado e submissão oficial.
- **Injetar precursores cegamente** (pioram quebras abruptas — sempre condicionar ao tipo de
  dinâmica) e **perseguir casos intrinsecamente indetectáveis** (variância sem sinal machuca a
  calibração transversal).
- **Importar features vencedoras de 2025 verbatim** — >95% de correlação com o top-10 desqualifica,
  e resolve só onde o onyx já vai bem.
