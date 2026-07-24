# Histórico: o que foi mudado e o que cada mudança rendeu

**Escopo:** todas as rodadas de intervenção do projeto, em ordem cronológica, com hipótese, mecanismo,
resultado medido e decisão. Substitui `DIAGNOSTICO_TS_AUC.md`, `PARECER_AUDITORIA_ONYX.md`,
`RESULTADOS_ROADMAP_R0_R6.md`, `PROPOSTA_FEATURES_V2.md`, `RESULTADOS_FEATURES_V2.md`,
`INVESTIGACAO_FALHAS_V3.md` e `RESULTADOS_P1_P4.md`.
**Fundamentos e definições:** [`MODELO.md`](MODELO.md). **Operação:** [`NOTAS_AGENTES.md`](NOTAS_AGENTES.md).

---

## 1. Placar consolidado

TS-AUC **out-of-fold** (GroupKFold, 10.000 séries, 2.541.134 linhas) — o juiz relativo do projeto
(§9.0 revisada). Todos os Δ com IC 95% por bootstrap pareado por série (300 réplicas,
`scripts/compare_oof.py`).

| Versão | O que entrou | Features | Geral | t≤50 | 50–150 | 150–400 | >400 |
|---|---|---|---|---|---|---|---|
| pré-auditoria | banco original + A2 (init_score) | 80 | 0,5996 | 0,529 | 0,566 | 0,615 | 0,643 |
| V1 | R1 (pesos pareados) + R4 (rank two-sample) | 91 | 0,5982 | 0,5220 | 0,5670 | 0,6139 | 0,6362 |
| V2 | F1 calibração · F2 digital H0 · F3 MMD/RFF · F4 Haar | 137 | 0,5997 | 0,5125 | 0,5658 | 0,6155 | **0,6504** |
| V3 | transporte de escala do nulo (corrige diluição por NaN) | 141 | 0,6039 | 0,5299 | 0,5736 | 0,6169 | 0,6510 |
| **V4 (empacotado)** | P1 dependência · P2 L-momentos · P3 varloc · P4 saltos | 183 | **0,6100** | **0,5357** | **0,5799** | **0,6242** | **0,6529** |
| V5 (revertido) | BOCPD; poda de L-momentos e `dep_*_w050` | 178 | ~0,606 | — | — | — | — |

**Deltas que passam no critério de R0 (IC 95% exclui 0):**

| Comparação | Δ geral | IC 95% | Buckets significativos |
|---|---|---|---|
| V2 vs V1 | +0,0014 | [−0,0046, 0,0092] | **t>400: +0,0105** [0,0030, 0,0187] |
| V3 vs V2 | +0,0043 | [−0,0009, 0,0094] | nenhum (t≤50 +0,0174 isolado, recupera a perda de V2) |
| V3 vs pré-aud. | +0,0043 | [−0,0033, 0,0122] | **t>400: +0,0085** [0,0004, 0,0185] |
| **V4 vs V3** | **+0,0060** | **[0,0007, 0,0117]** | **150–400: +0,0074** |
| **V4 vs pré-aud.** | **+0,0104** | **[0,0032, 0,0189]** | **50–150, 150–400, t>400** |
| V5 vs V4 | −0,0042 | [−0,0095, 0,0006] | **50–150: −0,0114** (regressão significativa) |

**Held-out no molde crunch** (100 séries, caminho real de submissão, fórmula oficial —
`scripts/crunch_local_ts_auc.py`): baseline 0,5073 · V3 **0,5470** · V4 **0,5416**. O delta
baseline→V4 (+0,0344) concorda em sinal e ordem de magnitude com o OOF, validando o instrumento.
**Esse conjunto não resolve V3 vs V4** — com 100 séries o erro-padrão é ≈0,054, uma ordem de
magnitude acima da diferença.

> **Errata (2026-07-22) e X1 (2026-07-24), ver §13.** O 0,6100 do V4 é a semente 42 (contaminada); o V4
> real é **0,6018** (média de 4 sementes limpas) / **0,6065** (bag). Sobre esse baseline, **X1
> (supervisão ponderada por detectabilidade, `soft`) rende +0,0040 [+0,0005, +0,0073], IC exclui 0** — a
> única aposta da campanha X0–X4 adotada, hoje ligada por `configs/default.yaml:weights`. A régua de
> decisão correta (semente 42 fora) é **~0,0014**, não 0,0058.

---

## 2. Rodada 1 — Diagnóstico da TS-AUC baixa (três intervenções, todas revertidas)

**Gatilho.** Uma TS-AUC local de **0,5244** medida com a fórmula do quickstarter sobre o conjunto de
teste reduzido (100 séries) — perigosamente perto do acaso.

**Hipóteses testadas e veredictos:**

| # | Hipótese | Veredicto |
|---|---|---|
| H1 | ruído amostral | **confirmada** — TS-AUC OOF real = 0,601; reamostrando 100 séries do OOF: média 0,600, **σ = 0,054**, p5 = 0,5254. O 0,5244 observado é o percentil ~5 da distribuição amostral do próprio modelo |
| H2 | habilidade real modesta, concentrada em t alto | **confirmada** — por bucket: 0,530 (t≤50, 8,1% do peso), 0,568 (50–150, 26,6%), 0,617 (150–400, 48,7%), 0,641 (>400, 16,5%). ~35% do peso cai onde o modelo mal supera o acaso |
| H3 | features de média degeneradas pelo whitening | **refutada** — os CUSUMs de média com δ alto não são degenerados, são **redundantes** com δ=0,25; toda a família `mean_z` tem correlação com y de 0,003–0,037 em qualquer janela vs. 0,06–0,09 para variância. **O sinal de média é estruturalmente mais fraco, não bugado** |
| H4 | calibrador desperdiça capacidade em `meta_h0_*` | confirmada com correção: `gain` atribuía 64,3% a `meta_h0_*`, o `mean\|SHAP\|` só 34,3%. CE6 = **0,5067** (taxa-base 0,4967): o histórico não prevê quebra futura ⇒ uso como efeito principal é ruído a priori |
| H5 | diferença de composição treino/teste | não isolada; subsumida por H1 |

**Achado emergente (o mais valioso):** rodando a suíte de robustez pela primeira vez, o motor
determinístico falha 4/15 gates (T3, T6, T8, T9), com **T6 (GARCH sem quebra) medindo 0,807 contra
limite 0,40** — mecanismo totalmente identificado: a trava `vol_adjust` protege média/dependência/
forma, mas a família de variância consome sempre `e` congelado (defesa CE2), então um cluster GARCH
faz o CUSUM de variância acumular (ratchet) sem quebra real.

**As três intervenções desenhadas — e o resultado:**

1. CUSUM de variância vol-ajustado como canal adicional (alvo: T6);
2. exclusão de `meta_h0_*` do treino como efeito principal (respaldada por ablação de fold único:
   logloss melhorava 0,52765 vs 0,52817, e bayes/conformal ganhavam +83%/+168% de gain);
3. CUSUM de média mais sensível, δ=0,15 (alvo: T3).

**Resultado:** TS-AUC OOF **caiu** de 0,5996 → 0,5961, com queda em quase todo bucket; a intervenção 3
causou uma **regressão real e explicável** em T13 (o ratchet do CUSUM mais sensível demora mais a
zerar depois de uma excursão transitória: decay 0,270 → 0,116). **Todas revertidas.**

**Achado colateral decisivo:** um baseline treinado nos mesmos dados sem nenhuma das três alterações
**também falha 9/15 gates** com o calibrador supervisionado. A causa é estrutural: `predict_one`
devolve o resíduo sem readicionar o offset de taxa-base, logo o score nunca se aproxima de 0 ou 1 —
os gates absolutos medem a **régua**, não o detector. Prova direta: em T4 o modelo reage
corretamente (score sobe de ~0,46 para ~0,67 poucos passos após a quebra, controle estável em ~0,45).

**Lições registradas:**
- **Ablação de fold único não é evidência.** Um resultado limpo e mecanisticamente explicado em um
  fold não previu o efeito no ciclo completo de 5 folds.
- **Protocolo padrão** (usado desde então): baseline controlado + suíte de robustez `--model` +
  TS-AUC OOF por bucket, para toda mudança de feature ou hiperparâmetro.
- Rodar CE6 e a suíte de robustez **a cada iteração**, não só quando há suspeita.

---

## 3. Rodada 2 — Auditoria externa: quatro veredictos e o roadmap R0–R6

Revisão independente de todo o material e do código (53/53 testes passando na época).

**Concordâncias registradas:** rótulo por passo + análise C1–C3; motor único; teste de prefixo com
canário; determinismo bit-a-bit; GroupKFold por série; score livre (CE1); trava CE2; `init_score`;
continuidade na fronteira; o protocolo de validação da rodada 1; a análise H1 de ruído amostral.

**Quatro discordâncias:**

- **D1 — recalibração pós-hoc do resíduo (Platt/isotônica/readição de offset) é impossível como
  alavanca de desempenho.** É transformação monótona uniforme entre séries ⇒ por C1, `AUC_t` idêntica
  em todo t ⇒ TS-AUC idêntica. É higiene para a suíte de robustez e para leitura humana, nada mais.
- **D2 — logo, "o descasamento de escala é o candidato de maior potencial" (conclusão da rodada 1)
  está errado.** Magnitude é invisível à métrica. O 0,60 é fraqueza de **ranking**.
- **D3 — a tese "objetivo ≠ métrica era o gargalo" foi parcialmente falsificada pelos próprios
  critérios.** O plano de ação previa que, com `init_score` + logloss, as árvores subiriam de ~90 para
  300–800; os `fold_evals` mostram best-iters de **69, 89, 61, 84, 85**. Leitura refinada: a família
  de objetivos *pontuais* satura rápido quando o **n efetivo é ~10⁴ séries**; o que resta desalinhado
  não é o *offset*, é a **estrutura** do objetivo.
- **D4 — a regra §9.0 na forma absoluta custa mais do que protege.** O projeto já usava o OOF como
  juiz de facto (0,5996 vs 0,5961); o problema não foi usá-lo, foi usá-lo **sem barra de erro**.
  Proposta adotada: OOF pareado com IC como juiz relativo, submissão oficial como âncora absoluta
  (ver `MODELO.md` §9.0).

**Desalinhamento estrutural apontado (R1):** a TS-AUC é a fração de pares (positivo, negativo) do
mesmo passo corretamente ordenados. O surrogate pontual consistente com isso dá `w_pos(t) ∝ n_neg(t)`
e `w_neg(t) ∝ n_pos(t)`. O esquema então vigente dava o mesmo peso às duas classes — em t≤50 a massa
de gradiente dos positivos era ~8% da dos negativos, exatamente no bucket com AUC 0,53.

**A bifurcação que o roadmap deveria resolver:** **H-extração** (o sinal existe além de 0,60 mas a
forma do objetivo não o extrai) vs. **H-informação** (para estas features, o gerador não dá mais que
~0,60–0,65).

---

## 4. Rodada 3 — R0–R6 implementados: alinhamento de objetivo dá **zero**

| Item | Entregável | Resultado |
|---|---|---|
| **R0** | `scripts/compare_oof.py` — comparador OOF pareado com IC 95% | **o item mais valioso da rodada**: converte cada retreino de ~10 min num experimento com veredicto estatístico |
| **R1** | pesos pareado-consistentes em `model/weights.py` | Δ = **−0,0014** [−0,0069, 0,0045] — indistinguível de zero |
| **R2** | `feval` de AUC-por-passo + `scripts/sweep_hyperparams.py` | **regressão real**: usar `ts_auc_by_t` como critério de parada dá Δ = −0,0119 (subamostra) / −0,0099 (fold inteiro), ambos com IC excluindo 0 |
| **R3** | `train_rank` (lambdarank), `RankModelEnsemble`, `CombinedModelEnsemble` | rank sozinho = 0,5852 (pior); combinado = 0,5993, Δ vs. binário +0,0010 [−0,0035, 0,0058] |
| **R4** | `state/rank_twosample.py` (6 features) | agregado plano (ver §5 para a leitura correta) |
| **R5** | gates relativos (`RELATIVE_GATE_SCENARIOS`) + painel de referência | mecanismo validado |
| **R6** | censo A1, resposta ao degrau OOF, envelope de potência | números reais abaixo |

**Achado R2 (contraintuitivo e importante).** A "régua certa em teoria" **regride na prática**: com
`ts_auc_by_t` na parada, o nº de árvores sobe de ~61–89 para 100–236, mas o ruído entre rodadas é
dominado pelo n efetivo de ~10⁴ séries (não pelo nº de linhas, mesmo com o fold inteiro no feval).
Selecionar o argmax ao longo de 100+ rodadas nessa métrica ruidosa produz **winner's curse** que não
generaliza. Correção: `lightgbm.early_stopping_metric` default `"logloss"`; `ts_auc_by_t` continua
disponível e é sempre computada e registrada, só não decide a parada.

**Achado R3 (armadilha de performance).** `lambdarank_truncation_level ≥ maior grupo` (recomendação
literal) é inviável aqui: com `t ≤ 100` todas as ~10.000 séries ficam vivas, então o maior grupo tem
~8.000 linhas e o treino rodou **>4 h sem terminar**. Correção: `rank.truncation_level_cap` (default
300).

**R6 — o mapa do gerador (censo A1, 4.552 séries com quebra):**
- mediana |Δmean_e| = **0,0033**; só **6,8%** das séries têm |Δmean_e| > 0,3 → **o canal de média está
  morto**;
- mediana |Δlogvar_e| = 0,0798, mas **41,8%** têm |Δlogvar_e| > 0,3 → **variância/cauda é o sinal
  dominante**;
- a AUC observada **excede** o envelope de potência de um detector de shift-de-média em todos os
  buckets — o modelo já extrai sinal que a média sozinha não explica.

**Conclusão da rodada:** as três alavancas de *como o modelo consome* (peso, parada, objetivo) são
estatisticamente nulas. A hipótese de trabalho migra para **o gargalo está no que o modelo consome**.

---

## 5. Rodada 4 — Proposta V2: comparabilidade e famílias novas (F1–F6)

**Correção metodológica que reordenou tudo (agora em `MODELO.md` §9.5):** `mean|SHAP|` é a medida
errada; a medida certa é a dispersão **dentro do passo** ponderada por `w_t` (XS-SHAP). Sob a medida
correta, `meta_h0_*` é a **maior família do modelo (30,5%)** — e CE6 provou que ela não carrega efeito
principal. Ou seja: **quase um terço da capacidade de ordenação era gasta aprendendo a calibrar**
("dada uma série com esta cara, um CUSUM de variância deste tamanho é ou não surpreendente"). E a
carga era maior exatamente onde o modelo é pior: 40,1% do XS-SHAP em t≤50 (AUC 0,522) vs. 27,0% em
t>400 (AUC 0,640).

**Diagnóstico pré-registrado sobre R4** (por que a família nova rendeu zero): das 6 features,
`ranktwo_dispersion_z_w100` entrou direto no **top-10** do modelo — e mesmo assim a TS-AUC ficou
plana. Ela **substituiu** capacidade existente em vez de somar. As duas de *localização*
(`wilcoxon`) são as **piores features do modelo inteiro** — confirmação independente do canal de
média morto. Consequência: F6 (painel JS/Hellinger/W1/KS) **morre** (são funcionais do mesmo eixo
inerte); F5 (padrões ordinais) desce; F1 sobe a prioridade máxima.

**Evidência direta da premissa de F1** (medida, não inferida do SHAP): a razão entre a escala nula de
uma série GARCH e de uma i.i.d. é **1,95–2,35× nas estatísticas cruas e 0,86–1,14× nas calibradas**
(`tests/unit/test_calibration.py::test_calibration_equalizes_null_scale_across_series`).

**Assimetria batch × tempo-real (por que não copiar a lista de 2025):** os vencedores calculavam
divergências sobre o segmento pós-quebra inteiro. Aqui a mediana de pontos pós-quebra é 14 (t≤50),
45, 133 e 303 por bucket — divergências multi-bin são estruturalmente inadequadas para t pequeno e
ajudariam só onde já vamos bem.

**Resultado de V2 (F1+F2+F3+F4, 137 features):**

- **Primeiro ganho estatisticamente significativo do projeto, mas localizado:** `t>400` **+0,0105**
  [0,0030, 0,0187]. Previsão "ganho concentrado nos buckets ≥150" **confirmada**.
- Agregado plano (+0,0014) porque `t≤50` **perdeu −0,0095**.
- **F1 fez exatamente o que prometeu no mecanismo:** as 8 `meta_h0_*` originais caem de **30,5% →
  14,5%** de XS-SHAP; as versões `_cal` vencem as cruas em **15 de 24 pares**
  (`accum_window_var_ln_w250`: 0,58% → **6,80%**, a 1ª feature do modelo). As famílias novas são
  usadas de verdade: `mmd` 10,9%, `haar` 5,9%; `mmd_joint_slow_cal` (detector não-paramétrico de
  quebra de *dependência*) é a 7ª feature.
- **Mas o mecanismo não virou ganho agregado.** Leitura defensável: o condicionamento implícito que o
  modelo já fazia era aproximadamente tão bom quanto o explícito — os 30% não eram capacidade
  desperdiçada, eram trabalho necessário que ficou mais barato sem mover o teto.

**A causa da perda em t≤50 — diluição por NaN, não perda de informação:**

| Bucket | NaN médio (features novas) | features novas 100% NaN |
|---|---|---|
| t≤50 | **64,8%** | **14 de 37** |
| 50–150 | 29,9% | 6 |
| 150–400 | 4,7% | 0 |
| >400 | 0,0% | 0 |

Com `feature_fraction=0,8`, ter ~24 colunas de puro nada reduz a probabilidade de uma árvore ver as
features que informam naquele bucket.

---

## 6. Rodada 5 — V3: corrigir a diluição (a previsão mais limpa do projeto)

**O que mudou:**
1. **Transporte de escala na calibração** (o principal): o bloqueio `t < min_t` das features `_cal`
   era conservador demais. Para estatísticas com lei de escala conhecida, o que é idiossincrático da
   série é o **fator de inflação** sobre o nulo i.i.d. (k = dp_medido/dp_teórico), ~constante em n —
   não o nível absoluto. O nulo passa a ser transportado para `n = min(t, w)`, liberando a versão
   calibrada desde t≈10 em vez de t=w (`state/calibration.py:_null_at`).
2. **Bug corrigido:** `haar_contrast_fine_mid` calculava `min_t` com a escala errada (2⁵·3=96 em vez
   de 2³·3=24), mantendo a feature em NaN sem necessidade.
3. **λ muito rápido no MMD** (`lambda_vfast = 0,08`, janela efetiva ~12), cobrindo o regime t≤50.

**Resultado estrutural:** colunas 100%-NaN em t≤50 caem de **14 para 8**; NaN médio das famílias novas
nesse bucket cai de **64,8% para 42,4%**.

**Resultado na métrica:** **os quatro buckets sobem simultaneamente**. Isolando (V3 vs V2): `t≤50`
**+0,0174** — mais do que recupera a perda — com `t>400` intacto (+0,0006). V3 vira o melhor modelo
até então (0,6039).

**O que NÃO se confirmou:** a segunda metade da previsão ("o agregado passa a excluir 0"). Δ geral
+0,0043 [−0,0033, 0,0122] — melhor estimativa pontual do projeto, direção certa em todos os buckets,
**ainda indistinguível de zero** pelo critério pré-registrado.

---

## 7. Rodada 6 — Investigação de onde o V3 falha (nada implementado, só direção)

Cruzamento do censo A1 com o OOF do V3 + limites de Neyman-Pearson simulados contra as magnitudes
reais. Métrica de trabalho: **percentil do score da série dentro da seção transversal de cada passo**
(exatamente o que a TS-AUC agrega), médio 20–120 passos após τ.

**1. O gargalo NÃO é informação, é extração.** Um detector ótimo de variância que *conhece* τ e o tipo
atinge **AUC ≈ 0,856** contra a mistura real do gerador; o V3 estava em 0,604. Por faixa de pontos
pós-quebra: m≤25 → 0,775; 25–75 → 0,798; 75–200 → 0,839; **m>200 → 0,890**. A afirmação anterior de
que "t≤50 está perto do teto de informação" era **pessimista demais**.

**2. O modelo é cego a dois eixos que existem no gerador.** Regressão OLS padronizada
`detect ~ |Δlogvar| + |Δmean| + |Δρ₁| + |Δkurt|`:

| Eixo | β independente | detect (tercil alto) | Leitura |
|---|---|---|---|
| Variância | **+0,312** | 0,667 | único eixo forte |
| Cauda/forma | +0,052 | 0,619 | fraco mas **independente** (corr 0,0 com variância) |
| Dependência | +0,043 | 0,580 | fraco mas **independente** (corr 0,14) |
| Média | −0,005 | 0,584 | **morto** (o marginal +0,089 era confundido) |

Isolando: quebras de **dependência pura** (437 séries) têm detect = **0,492 — abaixo do acaso**;
**cauda pura** (357 séries) = 0,553. E a teoria diz que uma quebra de dependência lag-1 de magnitude
moderada é **altamente detectável** (0,81–0,99 com janela média/longa). Não é limite de informação, é
**feature ausente**. As 1.809 quebras mal detectadas têm o **mesmo** número de pontos pós-quebra
(~230) das bem detectadas — não falta dado, falta feature no eixo certo.

**3. A hipótese que explica a folga: diluição por τ desconhecido.** Toda janela fixa mistura pontos
pré e pós-quebra e estima uma variância *atenuada*; o oracle usa só os pontos pós-τ. O mecanismo que
deveria resolver isso (`bayes_map_var_ln`) rende pouco, provavelmente porque o modelo de
troca-única-gaussiana é mal-especificado para um gerador com cauda pesada e dependência.

**Achado de manutenção:** os features de dependência mortos (`cusum_dep`, `accum_window_rho1_fz`)
**não recebem calibração F1** — parte da sua morte pode ser miscalibração transversal.

---

## 8. Rodada 7 — P1–P4 → V4: o maior ganho do projeto

Quatro famílias, cada uma atacando um ponto cego medido, todas calibradas via F1. 183 features.

| | Bloco | Alvo |
|---|---|---|
| **P1** | `state/dependence.py` | dependência não-linear/multi-lag (ρ₁ de \|e\| e e², massa multi-lag) |
| **P2** | `state/lmoments.py` | forma de cauda dinâmica (L-skewness/L-kurtosis) |
| **P3** | `state/varloc.py` | variância localizada no changepoint (max sobre escalas) |
| **P4** | `state/jumps.py` | bipower/saltos + leverage (precisão em T6/T9) |

**Resultado — o primeiro ganho agregado estatisticamente significativo do projeto:**
- V4 vs V3 (isola P1–P4): **+0,0060** [0,0007, 0,0117], com 150–400 **+0,0074** significativo;
- V4 vs baseline pré-auditoria (a sessão inteira): **+0,0104** [0,0032, 0,0189], com **3 de 4 buckets**
  individualmente significativos;
- os quatro buckets subiram de V3 para V4.

O ganho concentrou-se em **50<t≤400** — exatamente o regime que a investigação previu ter a maior
folga e onde há janela suficiente para as estatísticas novas se estabilizarem. `t≤50` subiu (+0,007)
mas não significativamente, coerente com o teto causal apertado (mediana de 14 pontos pós-quebra).

**Conformidade:** 133 testes passando (era 104), incluindo causalidade e determinismo bit-a-bit;
latência 980 µs/passo (gate 1500); CE6 inalterado (as 4 famílias são online, estruturalmente ausentes
do classificador só-histórico).

**Estado da hipótese "o gargalo são as features": sustentada, com significância.** A sessão testou
três teses em ordem: objetivo/peso/parada (nulo) → comparabilidade/calibração (mecanismo válido,
agregado nulo) → **informação nova nos eixos cegos medidos** (moveu o agregado, duas vezes). O
gargalo era o que o modelo consome, mas de forma específica: não "mais features", e sim *informação
nos eixos que o censo mostra existirem e que o banco não cobria*.

---

## 9. Rodada 8 — V5 (BOCPD + poda): regressão medida e **revertida**

Não houve documento de rodada para esta mudança; o que segue foi reconstruído dos artefatos e do
código no commit `9bc0395 "Submission v1"`.

**O que mudou:** entrou `state/bocpd.py` (posterior completo sobre run-length, Adams–MacKay,
R_max=256 — a versão principiada de `varloc`, 4 features + calibradas) e saíram do pipeline `lmom_*`
(L-momentos, P2: 0,51% de XS-SHAP por ~65 µs/passo — o bloco mais caro) e `dep_*_w050` (mortas no
SHAP do V4). Total: 183 → 178 features. **Duas mudanças empacotadas juntas**, o que torna o
resultado não-atribuível.

**Resultado (`artifacts/reports/compare_v5_vs_v4.json`):** Δ geral **−0,0042** [−0,0095, +0,0006] —
IC não exclui 0, mas o ponto é negativo e o bucket **50<t≤150 regride significativamente
(−0,0114, IC [−0,0195, −0,0020])**. Nenhum bucket melhorou. O único argumento a favor era latência
(949 vs 980 µs/passo), que não é restritiva — ambos com folga sobre o gate de 1500.

**Decisão: revertido.** Pela regra de R0 (adotar só se o IC excluir 0 **a favor**), o V5 não passa, e
não havia justificativa registrada para tê-lo empacotado. O pipeline voltou ao conjunto de blocos do
V4 e `resources/` foi reempacotado com o artefato do V4.

**Como a reversão foi verificada** (o código do V4 não estava no git — o repositório tem um único
commit com tudo, então o pipeline dele teve de ser *reconstruído*):

1. **Schema:** o pipeline reconstruído gera exatamente as 183 features de
   `artifacts/models/v4/feature_schema.json` — nenhuma faltando, nenhuma sobrando.
2. **Valores, bit-a-bit:** para 23 séries de treino cobrindo os 5 folds, a trajetória de scores
   gerada por `StreamScorer` + o booster do fold de validação da série reproduz
   `artifacts/models/oof_v4.parquet` com `maxdiff = 0.000e+00`
   (comparando `sigmoid(raw_score + logit(p̂(t)))`, que é como o OOF é salvo em `train.py`).
   Esse teste também **encontrou** o único detalhe que a reconstrução tinha errado: a calibração dos
   L-momentos usa `kind="rho"` (nulo transportado por 1/√n, disponível a partir de t=10), não
   `kind="none"` — com o valor errado a divergência aparecia exatamente a partir de t=10.
3. **Fim a fim:** `resources/` + código atual pontua **0,5416** no held-out de 100 séries (molde
   crunch), idêntico bucket a bucket ao registro do V4 em `artifacts/reports/crunch_local.log`.
4. **Conformidade:** 139 testes passam (unit + causality + determinism); notebook de submissão
   regenerado e aprovado nos três estágios do verificador (células, script convertido, subprocesso
   paralelo) com `max|Δscore| = 0`; latência **973,8 µs/passo** (gate 1500) → PASS.

**O que fica em aberto (experimento, não pendência):** o V5 empacotou duas mudanças, então "BOCPD
não ajuda" **não** está demonstrado — o que está medido é que "BOCPD + poda de L-momentos e
`dep_w050`" piora. O bloco, sua config e seus testes foram preservados; o experimento que separa as
duas metades (V4 + BOCPD, **sem** a poda, ~190 features) nunca foi rodado.

---

## 10. Lições metodológicas acumuladas

1. **Mecanismo limpo não garante ganho.** Já aconteceu cinco vezes: as três intervenções da rodada 1,
   R1 (pesos pareados, derivação fechada), R3 (ranking por grupo), e F1 (calibração, com previsão
   confirmada no mecanismo e nula no agregado). Intervir **menos vezes, com hipóteses maiores**.
2. **Ablação de fold único não é evidência.** Custou uma rodada inteira.
3. **A régua certa em teoria pode ser a errada na prática** (R2): quando o n efetivo é ~10⁴, otimizar
   diretamente a métrica ruidosa gera winner's curse.
4. **Sempre com barra de erro.** Δ de 0,003–0,004 sem IC não decide nada — nem para adotar nem para
   reverter.
5. **Medir a coisa certa:** `mean|SHAP|` enviesava a priorização para features que só acompanham o
   relógio; a medida transversal (XS) mudou a leitura de famílias inteiras.
6. **Toda família nova precisa de disponibilidade em t pequeno**, ou dilui o `feature_fraction` e
   piora o bucket mais fraco (V2 → V3).
7. **Falsificações são resultado.** A previsão de árvores (D3), a de R1 e a de F1 falharam — e cada
   falha reposicionou a busca. As duas que renderam (F3/F4 e P1–P4) vieram de **injetar informação em
   eixos medidos como cegos**, não de refinar o que já existia.

---

## 11. O que NÃO fazer (reafirmado por medição, não por opinião)

- **Qualquer coisa no canal de média** — β = −0,005 multivariado; censo mostra 6,8% de séries com
  |Δmean|>0,3; as features de localização rank são as piores do modelo. Encerrado.
- **Recalibração pós-hoc do score como alavanca de desempenho** — C1-neutra por impossibilidade
  matemática (D1).
- **Mais funcionais do mesmo contraste de CDF** (JS/Hellinger/W1/KS, mais χ²-de-forma) — o eixo está
  saturado; `ranktwo_shape_chi2` rende 0,00–0,02%.
- **Força bruta de milhares de features** (abordagem do 2º lugar de 2025) — n efetivo ≈ 10⁴; é uma
  máquina de overfitting de seleção, e cada feature custa latência real no motor causal.
- **Reabrir grades de CUSUM, hazards ou `meta_h0`** — já julgadas nulas duas vezes.
- **Features transversais** (rank da feature entre séries vivas no mesmo t) — a API entrega uma série
  por vez; é estruturalmente impossível e seria vazamento.
- **Wavelet denoising como pré-processamento** — não é causal na forma padrão.

**Estacionados, com critério de reabertura:** t-likelihood no filtro bayesiano (se o censo mostrar
fatia grande de ν̂<8); GRU em numpy (só em platô, com folga de cronograma); L-momentos (bloco e testes
preservados; reabrir se houver orçamento de latência e evidência de eixo de forma subexplorado);
V-ema no pós-processamento (quando sobrar uma sonda).

---

## 12. Próximos passos sugeridos (não executados)

1. **Sonda oficial do V4** (o artefato agora empacotado) — o OOF já não é o gargalo de decisão para ganhos da ordem
   de +0,01; a resolução da âncora oficial é. Registrar hipótese por escrito antes de submeter.
2. **Separar as duas metades do V5:** V4 + BOCPD (~190 features), **sem** a poda de L-momentos/
   `dep_w050`, julgado por R0 contra o V4. É o experimento que o pacote do V5 impediu de atribuir.
3. **XS-SHAP do V4/V5** — as 42 features de P1–P4 nunca tiveram atribuição individual medida; é a
   medição mais barata para decidir o que podar.
4. **Estender a calibração F1 às features de dependência** (`cusum_dep`, `accum_window_rho1_fz`), hoje
   não calibradas e mortas — hipótese: parte da morte é miscalibração transversal.
5. **Sweep de hiperparâmetros** (`scripts/sweep_hyperparams.py`, pronto) julgado por `logloss` (o
   critério validado), não por `ts_auc_by_t`.
6. **Recalibrar `drift_slope_abs_max`** (1e-4 reprova T2/T6/T10/T12 por slopes da ordem de −0,0005;
   provavelmente limiar apertado demais, não falha real) — decisão de tuning com validação própria.

---

## 13. Rodada 9 — Campanha de ruptura (X0–X4): **X1 adotado**, régua recalibrada

**Gatilho.** `BRAINSTORM_RUPTURA_TSAUC.md`: cinco apostas ortogonais (X0–X4), cada uma com mecanismo,
gate barato, previsão pré-registrada e critério de morte. Baseline: V4 legítimo (`train_rows_3eixos
--drop-prefix spec_ ord_ mrep_` = 183 feat), OOF K=4 (sementes 777/101/202/303) = **0,6018** — o número
real, não os 0,6100 históricos (semente 42, contaminada por seleção; ver a errata do BACKLOG).

**Resultado: 1 vitória, 4 mortes, e uma recalibração da régua mais valiosa que qualquer braço.**

| braço | eixo | resultado | decisão |
|---|---|---|---|
| **X1** | supervisão ponderada por detectabilidade | **soft +0,0040 [+0,0005, +0,0073]**, hard +0,0036 [+0,0004, +0,0071] — IC exclui 0 no agregado E em 50–150/150–400 | **ADOTADO (soft)** |
| X0 | rodadas fixas por curva média | σ_seed não caiu (0,0010→0,0011); Δ logloss −0,0019 (exclui 0 *contra*), tsauc −0,0028 | morto |
| X2 | canal de média (2 defeitos) | gate D1: AUC da família média por tercil de `ar_r2` = baixo 0,636 / médio 0,577 / **alto 0,638** — persistência NÃO cega (Defeito 1 refutado) | morto no gate (build evitado) |
| X3 | studentização por CDF nula | reordena (corr_xs 0,33 < 0,95) mas −0,0294; mesmo destino da centragem aditiva | morto |
| X4 | injeção sintética | probe de artefato obrigatório: discriminador real-vs-sintético sob H0 = **AUC 0,9998** (>>0,6) → gerador vaza (portador com `meta_h0` de histórico truncado) | bloqueado (build evitado) |

**X1 — o que foi adotado.** O peso de cada linha POSITIVA vira `w ×= clip(d_i/q95, 0,3, 1)`, com `d_i` a
detectabilidade do censo A1 (norma L2 dos eixos `delta_logvar_e/rho1/kurt/exceed` padronizados × √m_bucket).
Tira gradiente dos positivos que não carregam sinal no passo em que pesam (magnitude baixa; linhas logo
após τ). É peso do PROFESSOR — usa τ só no treino, como o próprio alvo `y=1{τ≤t}`; a inferência é
intocada. Só afeta séries de quebra precoce (0<τ<50, ~712 séries); as demais ficam com peso 1. Ganho
concentrado em 50–150 (+0,008) e 150–400 (+0,004), os buckets de maior peso da métrica.

**Implementação (produção).** `configs/default.yaml:weights.detectability_mode: soft`
(+ `WeightsConfig` em `config.py`). A nuvem treina do zero e não tem o `detectability.csv` local, então
`adapter/platform.py` computa o mapa `d_i` **inline** dos registros de treino via
`model/detectability.py:compute_detectability_map` (mesma matemática de `break_type_census` +
`detectability_report`), e passa a `model/weights.py:compute_row_weights`. Notebook de submissão
regenerado e **verificado bit-a-bit** (`scripts/verify_submission_notebook.py`: max|Δscore| = 0). Custo:
só treino (um passe extra de fit_h0/whiten por série); latência de inferência inalterada.

**A recalibração da régua (subproduto do X0, mais importante que o X0).** A tese σ_seed ≈ 0,0041 / barra
0,0058 que fundamentava o BACKLOG era **artefato de incluir a semente 42**. Entre as 4 sementes limpas, o
σ single-seed é **0,0010**; o EP da diferença de duas médias K=4 é **0,0007** ⇒ **barra 2-EP ≈ 0,0014**,
~4× mais fina. Sob 0,0058 a vitória do X1 (+0,0040) teria sido "inconclusiva"; contra 0,0014 ela é ~2,5×
a barra. **Régua nova para todo braço futuro: ~0,0014 (K=4, semente 42 fora).**

**Disciplina que se pagou.** Os gates baratos (D1 em minutos; probe de artefato antes do braço)
mataram X2 e X4 **antes** de qualquer build/K=4 — exatamente o que os gates existem para fazer.

**Pendências.** Submissão oficial do V4+X1-soft (âncora em `artifacts/reports/submission_log.md`, criado
nesta rodada) para confirmar contra o placar — o mapa OOF↔placar continua desconhecido (um ponto só).
Reabrir X4 exige redesenhar a construção do portador (~2 dias) para o H0 sintético casar com o real.

---

## 14. Rodada 10 — Campanha B/C: **B6 adotado**, e a costura C5 nasceu e morreu no mesmo dia

**Gatilho.** `BRAINSTORM_RUPTURA_V2.md` (o item de dados de 2025 foi descartado: não é permitido).
Baseline/incumbente de toda a rodada: **V4 + X1-soft, K=4 sementes limpas = `oof_x1_soft_bag4` = 0,6102**.
Régua: **~0,0014** (a barra recalibrada da §13). Duas sessões: a primeira caiu no meio do B4 e foi
reconstruída do transcript; os veredictos abaixo cobrem as duas.

### 14.1 Placar da rodada

| braço | o que era | Δ vs incumbente | decisão |
|---|---|---|---|
| **B6** | `feature_contri=0,6` nas colunas `meta_h0_*` | **+0,0023** [−0,0010, +0,0053] geral; **150–400 +0,0049** [0,0012, 0,0085] e **t>400 +0,0048** [0,0009, 0,0084] (IC exclui 0) | **ADOTADO** (ver 14.3) |
| A1 | EWMA assimétrico (α_sobe 0,8 / α_desce 0,2) no pós-processo | +0,0005 [0,0000, 0,0010]; 150–400 +0,0009 | passa sozinho, **fora do pacote** (14.4) |
| T0.1 | diagnóstico de pares difíceis / negativos quentes | perda **DIFUSA**: 0,5% piores séries = 2,0% da massa de pares invertidos; 5% = 16%; 10% = 29%. 60 controles quentes marcados como rótulo-suspeito = 0,60% das séries | sem alvo concentrado |
| T0.2 | rank-average das 4 sementes X1-soft | −0,0001 geral, ~0 em todos os buckets | eixo fechado |
| A2 | stacking heterogêneo (binário+rank+fallback) | membro rank (lambdarank) rodou >15 min sem saída neste ambiente; membro historicamente fraco (~0,585 vs 0,610) | bloqueado pelo ambiente |
| A3 | emulador do nulo REAL (LGBM pinball q10/q50/q90) + studentização | 0,6102 → **0,5645**, Δ **−0,0457** | morto; **fecha o eixo** (14.5) |
| B1 | pesos de linha por `w_t = n_pos·n_neg` | −0,0030 [−0,0058, −0,0002] (exclui 0 CONTRA); 50–150 −0,0065 | morto |
| B2 | variante `ramp` do X1 | −0,0038 [−0,0073, +0,0001]; 50–150 −0,0068 e 150–400 −0,0035 excluem 0 contra | morto (confirma `soft` como o sabor certo) |
| B3 | rebaixar negativos das 60 séries marcadas em T0.1 | −0,0009 [−0,0036, +0,0015] | morto (~0, como T0.1 previu) |
| B4 | objetivo custom `(1−α)·logloss + α·pairwise` intra-t | α=0,3 **−0,0035**; α=0,1 +0,0007 (metade da barra) | morto / inconclusivo (14.6) |
| B5 | `monotone_constraints=+1` em 9 acumuladores de LLR | −0,0005 [−0,0034, +0,0022] | neutro, morto |
| C2 | conserto do suporte do portador "as-of" + re-probe | probe real-vs-sintético **0,9993** mesmo com o conserto | bloqueado (14.5) |
| C3 | bagging de partição (2 partições × 4 sementes) | partB 0,6029 vs partA 0,6102; bag das duas −0,0014 [−0,0039, +0,0013] | morto, **mas achou o principal** (14.2) |
| C5 | modelo por regime de t | ver 14.3 — **nasceu vencedor e foi retratado no mesmo dia** | costura descartada |
| C4 | varredura de hiperparâmetros | **NÃO RODADO** (decisão do usuário, 14.7) | pulado |

### 14.2 O achado que reorganiza a rodada: variância de PARTIÇÃO

O C3 mediu o que ninguém tinha medido: trocar a **partição de folds** (`cfg.seed` 42→43), mantendo
tudo o mais, move a TS-AUC em **~0,007** — **7× a variância de semente de boosting** (0,0010) e **5× a
barra** (0,0014). E a partição 42 é a **sortuda**: o mesmo X1-soft mede 0,6102 em A e **0,6029** em B.
É o mesmo padrão da semente 42 da §13, um nível acima. **Consequência prática: qualquer efeito da ordem
de 0,003 medido numa partição só é indistinguível de sorte de partição.** Foi exatamente isto que
derrubou a costura C5 (14.3).

### 14.3 C5 — a costura por regime: adotada de manhã, retratada à tarde

O C5 original pedia dois modelos treinados (t≤64 / t>64). Não foi preciso: o perfil por bucket do B6
(−0,0029 em 50–150, +0,0049/+0,0048 acima de 150) **é** o conflito de regimes que o C5 postulava, e a
**C1** (`MODELO.md` §1.2) autoriza trocar de modelo por t — a `AUC_t` só enxerga um modelo dentro de
cada passo, então costurar dois OOFs numa borda de bucket é legítimo pela métrica, de graça.

Na partição A a costura mediu **+0,0032** [+0,0012, +0,0051] (exclui 0; 2,3× a barra), com dois sinais
de que não era otimismo de seleção: os folds de **confirmação** deram Δ maior que os de seleção
(+0,0044 vs +0,0025), e o ganho é um **platô** no ponto de corte (+0,0028 a +0,0032 para qualquer corte
entre 100 e 200), não uma quina.

**A replicação na partição B derrubou o desenho.** O ganho da costura sobre o baseline replica
(+0,0032 em A, +0,0040 em B) — o efeito é real. Mas em B o **B6 sozinho (+0,0049) BATE a costura
(+0,0040)**. O porquê está no bucket que motivava a costura:

| bucket | B6 − X1-soft, partição A | partição B |
|---|---|---|
| t≤50 | −0,0011 | −0,0016 |
| **50–150** | **−0,0029** | **+0,0039** ← troca de sinal |
| 150–400 | +0,0049 | +0,0065 |
| t>400 | +0,0048 | +0,0050 |

O conflito de regimes era **ruído de partição**, não estrutura. O que replica é o ganho do B6 em t alto
e um −0,001 pequeno e consistente em t≤50 (bucket de peso desprezível na métrica).

**Decisão: adota-se o B6 SOZINHO.** Um só modelo, sem dobrar o treino na nuvem, melhor na partição que
não foi usada para desenhá-lo, e implantável por **um knob de config** (`lightgbm.feature_contri_meta:
0.6`) — sem tocar em `adapter/platform.py`, sem custo extra de inferência.

**A evidência formal do B6, nas duas partições** (`compare_b6_contri06.json`, `compare_b6_partB.json`):

| | partição 42 (a sortuda) | **partição 43 (independente)** |
|---|---|---|
| geral | +0,0023 [−0,0010, +0,0053] (inclui 0) | **+0,0049 [+0,0018, +0,0079] EXCLUI 0** |
| 150–400 | +0,0049 [0,0012, 0,0085] exclui 0 | **+0,0065 [0,0030, 0,0102] exclui 0** |
| t>400 | +0,0048 [0,0009, 0,0084] exclui 0 | **+0,0050 [0,0017, 0,0084] exclui 0** |

Na partição 42 o agregado era inconclusivo e só os buckets altos passavam; **na partição que não foi
usada para desenhar nada, o IC do agregado exclui 0**. Média das duas ~**+0,0036** = 2,6× a barra. É a
evidência mais forte da campanha, e veio justamente da réplica que derrubou a costura.

### 14.4 A1 fica fora do pacote

O A1 passa sozinho (+0,0005), mas **em cima do B6** rende só +0,0004 global e +0,0004 com gate em
t>150 — ambos abaixo da barra. Motivo: A1 também é um braço de t ALTO (+0,0009/+0,0012 nos dois
buckets altos) e **se sobrepõe ao B6**. Somar os dois não soma os ganhos. Código preservado
(`postproc_vema_probe.py`, `apply_vema_oof.py`, `monotonicity.ema_asym_step`).

### 14.5 Dois eixos FECHADOS por medição

1. **Comparabilidade auto-referencial (studentizar o score pelo nulo da própria série).** Três mortes,
   cada uma com um nulo melhor que a anterior: centragem aditiva −0,002, X3 (CDF nula do histórico)
   −0,029, **A3 (emulador do nulo REAL, pinball q10/q50/q90 out-of-fold) −0,046**. A cláusula
   pré-registrada do A3 era justamente "se falhar, fecha o eixo". **Fechado.**
2. **Fábrica de nulos "as-of" / injeção sintética.** X4 probe 0,9998, C2 (com conserto de suporte)
   0,9993. A causa não é suporte de comprimento: é um **fosso de domínio** — o online real é do
   período 2 e o `meta_h0`/whitening é ajustado no histórico inteiro, enquanto o sintético é uma
   fatia do próprio histórico com H0 ajustado no histórico anterior (limpo demais, dentro do regime).
   Reabrir exige redesenhar o portador (~2 dias).

### 14.6 B4 — dois defeitos de código achados antes do veredicto

O braço estava cabeado desde a sessão que caiu. Antes de gastar ~3 h em 2×K=4, dois diagnósticos de
~12 min (cap 200, semente 777) resolveram — e acharam dois defeitos reais:

1. **Desempenho.** O pré-sorteio de pares varria as ~506k linhas positivas em laço Python, com uma
   varredura de 2,5 M linhas por valor de t. Reescrito como um `groupby` sobre ~399 valores de t:
   **60 s → 0,2 s por fold**.
2. **Bug.** Com objetivo **custom** o LightGBM entrega ao `feval` o **score bruto** (com o objetivo
   embutido `binary`, entrega probabilidade). O braço binário chamava `_make_fold_feval` sem
   `raw_to_prob`, então o `binary_logloss_diag` tratava score bruto como probabilidade — **5,20 contra
   0,66 do incumbente** — e, como `early_stopping_metric=logloss`, **a parada antecipada rodava nesse
   sinal sem sentido**. Corrigido (`raw_to_prob=use_pairwise`, `model/train.py`).

Como a `ts_auc` é invariante à sigmoide (monótona), ela **nunca** foi afetada, o que deu um teste
independente da regra de parada: melhor `ts_auc` por fold alcançável (limite superior, favorável ao
B4). x1_soft 0,6102 / b6 0,6114 / **α=0,3 0,6067** (não ganha sob nenhuma regra de parada) / **α=0,1
0,6109** (+0,0007, metade da barra, dentro de 1σ de uma semente). Nenhum K=4 foi gasto.

### 14.7 O que NÃO foi rodado, e por quê

**C4 (varredura de HP)** — pulado por decisão do usuário. Caça efeitos da ordem de 0,003, exatamente a
escala que o 14.2 mostrou ser indistinguível de sorte de partição; a triagem selecionaria na partição
42 e qualquer vencedor precisaria de confirmação na 43 antes de valer. O harness ficou **pronto**:
`sweep_hyperparams.py` ganhou `--drop-prefix` e `--detectability-mode` — sem eles cada célula diferia
do incumbente em **duas** coisas além dos HPs (o incumbente treina com pesos X1-soft e 183 features),
e a varredura inteira teria sido confundida em silêncio.

**B4 α=0,1 em K=4** — pulado: +0,0007 é metade da barra e cabe no ruído de uma semente.

### 14.8 Armadilha de baseline (para o próximo agente)

Quatro números quase iguais convivem no repositório e **um deles é contaminado**:

| número | o que é |
|---|---|
| 0,6018 | média das TS-AUC **individuais** das 4 sementes limpas (é o que a §13 cita) |
| **0,6062** | **bag das 4 sementes limpas** (`oof_x0_a_bag4`) — o bagging vale +0,0044. **Baseline correto** |
| 0,6084 | `oof_b183_bag4` — é a média das sementes **(42, 101, 202, 777)**, ou seja, **contém a semente 42**; lê +0,0022 alto |
| 0,6100 | semente 42 **sozinha** — o "0,6100 histórico" que a §13 já marca como contaminado |

**Não use `oof_b183_bag4` como baseline.** O +0,0040 do X1 está medido contra `oof_x0_a_bag4` (limpo),
que é o certo; contra o b183 contaminado o mesmo X1 mediria só +0,0018.

### 14.9 Lições desta rodada

- **Variância de partição > variância de semente > barra.** Antes de acreditar em qualquer efeito de
  ~0,003, replicar na outra partição. Custou ~1 h e evitou empacotar um modelo mais caro e pior.
- **Um perfil por bucket numa partição só não é estrutura.** O bucket 50–150 trocou de sinal.
- **Diagnóstico barato antes de K=4.** O B4 fechou com dois runs de 12 min em vez de 3 h — e só porque
  se olhou para o número de rodadas e para a logloss, não só para o Δ final.
- **Métrica invariante = teste de graça.** A invariância da `ts_auc` à sigmoide isolou o veredicto do
  B4 do bug de parada que existia ao mesmo tempo.
