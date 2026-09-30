# Histórico: o que foi mudado e o que cada mudança rendeu

**Escopo:** todas as rodadas de intervenção do projeto, em ordem cronológica, com hipótese, mecanismo,
resultado medido e decisão. Substitui `DIAGNOSTICO_TS_AUC.md`, `PARECER_AUDITORIA_ONYX.md`,
`RESULTADOS_ROADMAP_R0_R6.md`, `PROPOSTA_FEATURES_V2.md`, `RESULTADOS_FEATURES_V2.md`,
`INVESTIGACAO_FALHAS_V3.md` e `RESULTADOS_P1_P4.md`.
**Fundamentos e definições:** [`MODELO.md`](../modelo/MODELO.md). **Operação:** [`NOTAS_AGENTES.md`](../operacao/NOTAS_AGENTES.md).

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
| **C6** | BOCPD isolado sobre o V4 (187 feat) | geral **+0,0020** [−0,0010, +0,0048] (inclui 0); **150–400 +0,0041** [0,0007, 0,0072] **exclui 0** (bucket-alvo) | **passa a regra, NÃO adotado** (14.10) |
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

### 14.9 C6 — BOCPD isolado: o experimento que faltava desde o V5

**O que era.** O BOCPD entrou no projeto **dentro** do V5, junto com a poda de L-momentos e de
`dep_*_w050`. O V5 regrediu como pacote e foi revertido (§9), mas os dois componentes tinham medido
positivo **separados** (poda +0,0027, BOCPD +0,0029) — sempre na régua velha. O experimento que separa
as duas mudanças (**V4 + BOCPD, SEM a poda**) estava registrado como "nunca rodado" desde 22/07.
Rodou agora: `SBRT_ENABLE_BOCPD=1` acrescenta o `BOCPDBlock` ao `default_blocks()` (opt-in; o default
de produção fica byte-a-byte igual, verificado), 4 colunas `bocpd_*` ⇒ **187 features**.

**Integridade da build.** A build nova reproduz as 183 colunas compartilhadas **bit-a-bit** contra
`train_rows_3eixos.parquet` (max|diff| = 0 em amostra de colunas, padrão de NaN idêntico, linhas
alinhadas) — o ganho é do BOCPD, não de deriva de build.

**Resultado.** 0,6102 → **0,6121**. Geral **+0,0020** [−0,0010, +0,0048] (inclui 0); **150–400
+0,0041** [0,0007, 0,0072] **exclui 0**; t>400 +0,0015; 50–150 −0,0004; t≤50 −0,0019.

**Veredicto: passa a regra formal, mas NÃO é adotado agora.** Pela regra do R0 (IC exclui 0 no
agregado **ou** no bucket-alvo declarado a priori) o C6 passa pelo bucket-alvo. Três ressalvas honestas:

1. O bucket-alvo `150–400` é o **default recomendado** do `NOTAS_AGENTES.md` §5 (49% do peso), não uma
   previsão específica do C6 — foi declarado antes de ver o perfil por bucket, mas não é uma hipótese
   mecanicista sobre *este* braço.
2. **É partição 42 (a sortuda) apenas.** É exatamente a situação do B6 antes da réplica — e o B6
   sobreviveu, a costura C5 não. Um +0,0020 com IC agregado incluindo 0 precisa do `--fold-seed 43`
   antes de entrar em qualquer pacote (§14.2).
3. **Implantação é mais cara que a do B6.** O B6 é um knob de config; o C6 muda o **conjunto de
   features** — exige `BOCPDBlock` no `default_blocks()` de produção, novo `feature_schema`, e ~30
   µs/passo (folga confortável no gate de 1500, mas não é zero).

**E não se pode assumir que soma com o B6** — então foi medido. **B6+C6 conjuntos = 0,6139**
(`compare_b6c6_vs_b6.json`), contra B6 sozinho 0,6125:

| bucket | conjunto − B6 sozinho | IC 95% | exclui 0 |
|---|---|---|---|
| geral | **+0,0014** | [−0,0008, +0,0033] | **não** |
| 150–400 | +0,0027 | [+0,0004, +0,0051] | sim |
| t>400 | **+0,0000** | [−0,0023, +0,0024] | não |
| 50–150 | +0,0005 | [−0,0033, +0,0044] | não |
| t≤50 | −0,0012 | [−0,0084, +0,0051] | não |

**Aditividade PARCIAL, não nula e não total.** A soma ingênua previa 0,6102+0,0023+0,0020 = 0,6145; o
medido é 0,6139 — sobrevivem ~70% do efeito isolado do C6. É melhor que o precedente do V5 (que
regrediu de vez), mas o Δ marginal cai exatamente **em cima da barra**, com IC agregado incluindo 0.

A linha que decide é **t>400: +0,0000**. O ganho isolado do C6 naquele bucket (+0,0015) **desaparece
por completo** em cima do B6 — o B6 já captura o que o BOCPD tinha lá. O valor marginal do C6 encolhe
para **um único bucket** (150–400).

**Decisão: C6 NÃO entra.** Custa mudança de conjunto de features + `feature_schema` + ~30 µs/passo, e
entrega um efeito que raspa a barra, some no agregado e vive num bucket só — medido na partição
sortuda. Se alguém quiser reabrir, o preço de entrada é a réplica em `--fold-seed 43` (~40 min), o
mesmo padrão que matou a costura C5.

### 14.10 Lições desta rodada

- **Variância de partição > variância de semente > barra.** Antes de acreditar em qualquer efeito de
  ~0,003, replicar na outra partição. Custou ~1 h e evitou empacotar um modelo mais caro e pior.
- **Um perfil por bucket numa partição só não é estrutura.** O bucket 50–150 trocou de sinal.
- **Diagnóstico barato antes de K=4.** O B4 fechou com dois runs de 12 min em vez de 3 h — e só porque
  se olhou para o número de rodadas e para a logloss, não só para o Δ final.
- **Métrica invariante = teste de graça.** A invariância da `ts_auc` à sigmoide isolou o veredicto do
  B4 do bug de parada que existia ao mesmo tempo.
- **"Passa a regra" ≠ "adotar".** O C6 passa pelo bucket-alvo e mesmo assim fica de fora até a réplica
  de partição: a regra do R0 foi escrita antes de o projeto saber que a variância de partição é 5× a
  barra. A regra continua necessária; deixou de ser suficiente.

---

## 15. Campanha de polimento (2026-07-25) — a frente A achou um defeito na régua

Ver `docs/RELATORIO_POLIMENTO.md` para o relatório completo e `CAMPANHA_POLIMENTO.md` para o plano
que a originou. Aqui ficam só os vereditos e o que mudou no código.

### 15.1 A2 — a TS-AUC OOF era agregada na GRADE ERRADA (o achado central)

O OOF vive na grade com thinning (399 valores de `t`); o board avalia todos os ~999 passos. `AUC_t`
em cada passo retido é **exato** (o thinning descarta passos, nunca séries — verificado casa a casa
contra `y_train.parquet`: `max|Δn_pos| = max|Δn_neg| = 0`), mas o **agregado** não é: cada `t` retido
entra com o próprio `w_t` em vez da massa do bloco que representa.

**Pesos por bucket, corrigidos:**

| bucket | media-se | **board** | razão |
|---|---|---|---|
| 1–50 | 8,1% | **3,9%** | 0,48 |
| 51–150 | 26,6% | **17,5%** | 0,66 |
| 151–400 | 48,7% | **46,7%** | 0,96 |
| **401+** | 16,5% | **31,9%** | **1,93** |

Duas verificações independentes: reponderar por bloco (sem interpolar nada) e interpolar `AUC_t` na
grade cheia dão **0,6277 e 0,6278** — concordam em 0,0001.

**Consequência 1 — o "câmbio OOF→placar" era a grade.** Reponderado, o OOF prevê o placar com erro
de ~0,002 em dois pontos independentes: `5e42ff5` 0,6227 vs board 0,6201; incumbente 0,6278 vs
0,6267. O offset de +0,0101 registrado como desconhecido em `NOTAS_AGENTES.md` §5 não existia.

**Consequência 2 — todo braço que ganha em `t>400` foi subcreditado por ~2×**, e todo braço que ganha
em `t≤150`, supercreditado por ~1,5–2×. O caso mais grave é o E0c, o experimento que fundamentou o
veredito de "beco sem saída" do `RELATORIO_EXP0.md`: o ganho das interações vive em `t>400`
(+0,0175, IC exclui 0), exatamente o bucket cujo peso dobra.

**Mudanças:** `evaluation/ts_auc.py:weighted_ts_auc(..., w_mult)` + `board_grid_multipliers`;
`scripts/compare_oof.py --grid full` **por default**; `scripts/a2_full_grid_ts_auc.py` e
`scripts/a2_reaudit.py`; `NOTAS_AGENTES.md` §5 corrigido.

### 15.2 A6 — o clip do resíduo contra nulos que não são clipados (dois defeitos)

`whiten_step` clipa `e` em `cfg.h0.clip_e` = [−8, 8]. De 19 blocos, só `lmoments.py` usa `e_raw`
(`accumulators.py` o usa nas duas estatísticas de cauda). O problema é que **as referências não são
clipadas**:

1. **`state/conformal.py`** — a feature nº 1 do modelo (`conformal_logm_abs`: xs-SHAP 0,061,
   `conv_share` 0,147) ranqueia `|e|` clipado contra `h0.sorted_abs_e_hist`, construído de
   `e_hist = resid/sigma_e` **cru**. Toda observação com `|e_raw| > 8` é ranqueada como se valesse
   8,0 → o p-value satura na massa de cauda do histórico em vez do piso `1/(n_h+1)`.
2. **`state/calibration.py:compute_null_stats`** — o replay que estima o nulo por série de cada
   coluna `_cal` recebe `e_hist` **não clipado**, enquanto a produção calcula a mesma estatística
   sobre `e` clipado. Atinge `mmd_joint_slow_cal` (nº 4) e `accum_window_var_ln_w100_cal` (nº 10).

**Quanto morde** (`train_rows_bocpd`, 2,54 M linhas): `max|e_raw|` excede 8 em **7,50% das séries**;
por classe, **6,99% das linhas positivas contra 2,64% das negativas (2,6×)**. O clip comprime
justamente o material que separa as classes.

**Correção direcional** (não existe "unclip tudo": o maior `|e_raw|` da base é **15.774**, e um ponto
desses destrói `accum_welford_var_ln`): pôr cada estatística na escala do **seu próprio** nulo.
`conformal.use_raw` (rank-based, referência crua → usar `e_raw`) e `h0.null_clip_match` (colunas
`_cal` são calculadas sobre `e` clipado → clipar o replay). Ambas `false` por default; defaults
inalterados, verificado (193 features, mesma lista). **Medição pendente — exige rebuild.**

### 15.3 A3 — o conjunto implantado ≠ o conjunto medido

A submissão não empacota modelo: o notebook embute o pipeline e a nuvem chama `train()`, então quem
define o conjunto implantado é `default_blocks()` — hoje **193 features** (V4 183 + MultiRep 6 +
BOCPD 4). Mas o incumbente **medido** (`oof_b6c6_joint_bag4`) tem **187**: foi treinado com
`--drop-prefix spec_ ord_ mrep_`. **O B6 (+0,0036) e o C6 (+0,0014) foram medidos numa base sem
`mrep_` e aplicados a uma produção que o contém** — e o precedente literal em `scorer.py` é que esses
ganhos não somam (*"V5+mrep: os ganhos NÃO SOMAM, −0,0024"*). Braço `p1_deploy193`.

Verificado e **correto**: (i) `platform.train` treina as 4 sementes × 5 folds e funde os 20 boosters
com verificação numérica; (ii) score em float64 sem clip/arredondamento; (iii) `zero_as_missing`
fica no default `false`.

### 15.4 A3b — o `init_score` deixou de remover um offset e passou a injetar um

`model/base_rate.py` ajusta a curva de taxa-base na **contagem crua**, mas desde o R1
(`model/weights.py`) o treino usa pesos pareado-consistentes que **equalizam as classes dentro de
cada passo**. Medido: a taxa de positivos **ponderada** é 0,4947 em `t=10` e 0,4996 em `t=400` —
praticamente 0,5 em todo `t≥10` — enquanto o `init_score` injetado vale −3,833 e −0,790.
**Descasamento médio +1,22 em log-odds, máximo +3,81.**

O R1 invalidou a premissa do A2 e ninguém rechecou. É neutro para a métrica no limite (invariância
C1), mas o modelo gasta árvores desfazendo o offset e **a logloss que governa a parada antecipada
passa a ser dominada por essa correção de `f(t)`** — consistente com a parada disparar em 79–117
árvores enquanto o toco do E0c treinou 1.529–3.186. `lightgbm.base_rate_weighted`, braço `p2_brw`.

### 15.5 Vereditos negativos desta rodada (não refazer)

- **A1 — rótulos: está CORRETO.** O snapshot local é pós-W23: exatamente **29 séries** com
  `tau_index = 0` (a assinatura do changelog) e as 2.868 linhas delas com `y = 1` no parquet.
- **A5 — passos frios: real e irrelevante.** O modelo é anti-preditivo em `t ∈ [1,8]` (AUC 0,4374 em
  `t=3`, cruza 0,50 só em `t=9`), com 67,9% das features em NaN de warmup em `t≤4`. Mas `t≤8` carrega
  **0,16%** do peso do board: forçar score constante ali vale **+0,00004**, 1/35 da barra.
- **B2 — espaço de média do bag: empate exato.** Probabilidade 0,62772 · logit 0,62773 · rank-average
  0,62756; Spearman 0,99999. Vale como auditoria: a produção funde em logit e o OOF media
  probabilidade, e a diferença é imaterial.
- **Registrado, não medido:** 15 colunas seguem 100% NaN em `t=50` (`haar_*`, `mmd_*_cal`,
  `jump_*_w100_cal`, `varloc_recent_vs_lagged*`) — a pegadinha do §7 (NaN dilui `feature_fraction`)
  acontecendo em produção.

### 15.6 Dois bugs pré-existentes que a suíte completa expôs

Rodar `pytest tests/unit tests/causality tests/determinism` (que a campanha fez para validar as
mudanças do A2/A6) reprovou 5 testes — **nenhum deles causado pelas mudanças da campanha**.

**1. `train_rank` está quebrado desde 2026-07-24** (`model/train.py`). Os blocos do B5/B6 escrevem em
`params`, mas em `train_rank` essa variável só passa a existir DENTRO do laço de folds (onde recebe
`lambdarank_truncation_level`); antes disso só existe `base_params`. Resultado: `UnboundLocalError`.

Ficou **latente** enquanto `monotone_llr` e `feature_contri_meta` eram off por default, e passou a
disparar **sempre** quando o B6 adotou `feature_contri_meta: 0.6` no YAML. Ou seja: **o braço de
ranking inteiro (R3) não roda desde a adoção do B6**, e ninguém percebeu porque `make ci` não estava
sendo rodado inteiro. Corrigido escrevendo em `base_params`.

**2. `test_scorer_feature_order_stable_and_no_leakage_of_T` guardava a premissa errada.** O teste
exigia 189 features e "tem de casar com `resources/feature_schema.json` — o scorer e o modelo
empacotado são um par". Duas coisas mudaram e o comentário não:

- `default_blocks()` emite **193** desde que o C6 ligou o BOCPD em produção;
- **a submissão não empacota modelo** (§15.3), então `resources/` não tem consumidor no caminho de
  submissão e a premissa do "par" não descreve mais o sistema.

O número foi atualizado para 193 e o comentário reescrito para guardar o que de fato importa: que
ninguém mude `default_blocks()` sem perceber que o conjunto **implantado** deixa de casar com aquele
em que os braços foram **medidos**.

**Lição operacional:** `make ci` existe e não estava sendo rodado inteiro. Um dos dois bugs deixou uma
frente inteira do projeto morta por um dia sem sinal nenhum.

### 15.7 Triagem K=2 na partição 42 (grade do board)

Baseline pareado `oof_b6c6_joint_bag2` = **0,62640**. Barra ~0,0014. Bucket-alvo declarado a priori:
`150–400` (o default do `NOTAS_AGENTES.md` §5, agora 46,7% do peso).

| braço | o que muda | nível | Δ geral | IC 95% | `150–400` |
|---|---|---|---|---|---|
| **`p3_lintree`** (D1) | `linear_tree=true` | 0,62887 | +0,0025 | [−0,0007; +0,0053] | **+0,0043 [+0,0001; +0,0080] EXCLUI 0** |
| **`p2_brw`** (A3b) | `init_score` sob os pesos | 0,62877 | +0,0024 | [−0,0012; +0,0057] | +0,0026 |
| `p4_cap` | `lr` 0,03, cap 3000, ES 300 | 0,62756 | +0,0012 | [−0,0011; +0,0036] | +0,0000 |

**`p3_lintree` passa a regra do R0 pelo bucket-alvo.** E o perfil por bucket é o que o mecanismo
prevê, o que vale mais que o número: ganha onde há pontos por folha para ajustar uma reta (`150–400`
+0,0043, `t>400` +0,0036) e **perde no warmup** (`t≤50` −0,0119, onde 68% das colunas são NaN em
`t≤4`). É a previsão literal do E0c — problema aditivo e suave, degraus aproximando curvas.

**`p4_cap` responde à pergunta que o E0c revisado levantou.** 2,4× mais rodadas (420–491 contra
173–220) com `lr` menor rendem +0,0012 e **exatamente 0,0000 em `t>400`**. A capacidade que falta ao
toco não é a que se compra com mais rodadas — e isso delimita a releitura do §15.1: o E0c diz que as
interações **contribuem** +0,0059 para o incumbente, não que exista outro +0,0059 disponível.

**Diagnóstico barato que refutou o mecanismo do A3b** (a lição do §14.10, antes de gastar K=4):

| modelo | rodadas por fold | melhor logloss |
|---|---|---|
| incumbente | 173–220 | 0,6625 |
| `p2_brw` | 160–213 | 0,6607 |
| `p4_cap` | 420–491 | 0,6622 |
| toco E0c | 1829–3486 | 0,6662 |

A previsão era que remover o offset de `f(t)` faria a parada disparar mais tarde. **Não aconteceu** —
`meta_t` e `meta_ln1p_t` são features, então o modelo desfaz qualquer `f(t)` com duas divisões. O
descasamento de +3,81 em log-odds é real e **barato de desfazer**. O braço segue para K=4 pelo Δ,
com a hipótese mecânica registrada como refutada.

**Os dois líderes não podem ser supostos aditivos** (`scorer.py`: *"V5+mrep: os ganhos NÃO SOMAM"*;
B6+C6 só parcialmente aditivo, §14.9). Se ambos sobreviverem à réplica na partição 43, o par tem de
ser medido junto.

### 15.8 O D1 está sendo subestimado pela regra de parada

Diagnóstico estrutural do `linear_tree`, na linha do §14.10 ("olhar as rodadas e a logloss, não só o Δ"):

| modelo | rodadas | árvores/fold | melhor logloss por fold |
|---|---|---|---|
| incumbente | 173–220 | 73–120 | 0,6625 · 0,6472 · 0,6593 · 0,6515 · 0,6615 |
| `p3_lintree` | 146–177 | **46–77** | 0,6606 · 0,6504 · 0,6581 · **0,6565** · **0,6657** |

O braço entrega **+0,0025 de TS-AUC com ~35% menos árvores** e uma logloss **igual ou pior em três
dos cinco folds**. As duas coisas juntas dizem que o ganho é de **qualidade de ordenação por árvore**,
não de ajuste pontual — e que a parada antecipada, que lê `binary_logloss`, corta o braço antes do
ponto em que ele ainda ganharia na métrica que conta.

É a mesma divergência de réguas já documentada em `config.py:LightGBMConfig.early_stopping_metric`,
agora com um caso em que ela custa. **Consequência prática:** antes de julgar o D1 pelo Δ de K=2,
vale um braço `linear_tree` + paciência maior (`--early-stopping-rounds 300`) — o custo é um treino e
o teto é maior que o medido. Registrado como o próximo passo natural do D1, não como conclusão.

### 15.9 O K=4 derruba o A3b — e calibra a leitura do D1

Promovido de K=2 para K=4 na partição 42 (grade do board):

| | baseline | candidato | Δ geral | IC 95% |
|---|---|---|---|---|
| `p2_brw` K=2 | 0,62640 | 0,62877 | +0,0024 | [−0,0012; +0,0057] |
| **`p2_brw` K=4** | 0,62772 | 0,62869 | **+0,0010** | **[−0,0018; +0,0033]** |

**De 1,7× a barra para 0,7×.** O nível do candidato mal se mexeu (0,62877 → 0,62869); quem subiu foi
o **baseline** (0,62640 → 0,62772), porque o bag de 4 sementes é melhor que o de 2. Ou seja: boa parte
do "ganho" de K=2 era o candidato ter tido sorte de sorteio contra um baseline mais ruidoso.

É o mecanismo já documentado em `NOTAS_AGENTES.md` §5 ("a régua tem dp de 0,0041 que o bootstrap
pareado não vê") acontecendo ao vivo. **Veredito do A3b: abaixo da barra, não promover.** O
descasamento de +3,81 em log-odds no `init_score` continua sendo um defeito real e vale corrigir por
higiene — mas não paga em TS-AUC, e a hipótese mecânica dele já havia sido refutada pelo número de
rodadas (§15.7).

**A consequência para o D1 é a lição desta rodada.** O `p3_lintree` está em +0,0025 com K=2 — o mesmo
patamar de onde o A3b caiu pela metade. O IC dele em `150–400` excluir 0 é um sinal melhor, e o perfil
por bucket tem mecanismo, mas **nada disso o dispensa do K=4**. Dois braços a ~+0,0025 em K=2, um dos
quais encolhe para +0,0010 em K=4, é a medida empírica de quanto a triagem K=2 infla neste projeto.

**Regra a herdar:** em K=2, tratar Δ como **ordenação entre braços**, nunca como estimativa de
efeito. Só o K=4 (e depois a partição 43) produz número.

### 15.10 Hipótese do §15.8 REFUTADA: a parada não estava cortando o D1

O §15.8 observou que o `linear_tree` entrega +0,0025 com ~35% **menos** árvores e logloss igual ou
pior, e propôs que a parada antecipada (que lê `binary_logloss`) o cortava antes do ponto útil — logo
o número medido seria um **piso**. Braço `p9_lintree_pac`: mesmo D1, `--early-stopping-rounds 300`
(contra 100) e `--n-estimators-cap 3000`.

| | rodadas por fold | árvores/fold | melhor logloss |
|---|---|---|---|
| `p3_lintree` | 173, 176, 166, 177, 146 | **73, 76, 66, 77, 46** | 0,6606 · 0,6504 · 0,6581 · 0,6565 · 0,6657 |
| `p9_lintree_pac` | 373, 376, 366, 377, 346 | **73, 76, 66, 77, 46** | **idênticos** |

As 200 rodadas a mais são exatamente a janela de paciência estendida: a **melhor iteração não muda**.
Como o LightGBM guarda o melhor modelo, `p9_lintree_pac` é **bit-idêntico** ao `p3_lintree` — 30 min
de treino por semente para reproduzir o mesmo artefato. Braço removido do
`scripts/run_polimento_fase4.sh`.

**O que fica refutado:** que o ótimo de logloss do D1 estivesse além do alcance da paciência atual.
Ele ocorre de fato em 46–77 árvores.

**O que NÃO fica descartado:** que a *régua* esteja errada — a TS-AUC pode seguir melhorando além do
ótimo de logloss. Mas isso é uma pergunta sobre o **critério**, não sobre a paciência, e testá-la
exige **rodadas fixas** (o desenho do X0: ES desligado, cap fixo, argmax da curva média entre
sementes), não uma janela maior. `early_stopping_metric: ts_auc_by_t` já foi medido e regrediu
(−0,0099, `config.py`), então o caminho é rodadas fixas, não trocar a métrica de parada.

**Lição de método:** "menos árvores + logloss pior + TS-AUC melhor" tem duas explicações — parada
prematura, ou o modelo simplesmente ser melhor por árvore. O diagnóstico das rodadas separava as
duas e custou um braço descobrir qual era. Rodar `p9` primeiro em UMA semente teria custado 30 min em
vez de 60.

### 15.11 Reauditoria das decisões passadas sob a ponderação do board

`scripts/a2_reaudit.py`, 150 réplicas, nas duas ponderações. A coluna "thinning" é o **controle**: se
não reproduzir o número histórico, o instrumento está errado. Reproduziu em todos os pares.

| decisão | thinning (como foi julgada) | **board (correta)** | muda? |
|---|---|---|---|
| **B6, partição 42** | +0,0023 [−0,0010; +0,0057] · inclui 0 | **+0,0033 [+0,0004; +0,0063] · exclui 0** | **sim** |
| **B6, partição 43** | +0,0049 [+0,0016; +0,0080] · exclui 0 | +0,0054 [+0,0027; +0,0087] · exclui 0 | não (reforça) |
| C6 (BOCPD sobre o B6) | +0,0014 [−0,0006; +0,0036] · inclui 0 | +0,0014 [−0,0003; +0,0031] · inclui 0 | **não** |

**B6:** o §14 registra que na partição 42 "o agregado inclui 0" e que a adoção se apoiava na réplica
da 43. Sob a ponderação correta a **42 passa sozinha**; média das duas ≈ +0,0044. A adoção estava
certa e está mais bem sustentada do que o próprio registro dizia.

**C6:** a correção **não o move**. A previsão mecânica (piorar, porque o ganho vivia só em `150–400`
com `t>400` em +0,0000) estava errada — houve cancelamento: `150–400` mal mudou de peso (48,7% →
46,7%), dobrar zero continua zero, e o peso retirado de `t≤150`, onde o C6 é plano, compensou. Fica
onde estava: na barra, IC incluindo 0, um único bucket, sem réplica de partição.

**Generalização que a reauditoria permite:** a correção de grade só move um braço quando o perfil
por bucket é **desigual entre `t≤150` e `t>400`**. Braços planos (C6) não se movem; braços com ganho
em `t` alto (B6, E0c) sobem. Isso dá uma triagem barata de quais números históricos vale reauditar —
basta olhar o perfil por bucket já registrado, sem rodar bootstrap nenhum.

**X1-soft — o veredito cai (2026-07-25).**

| grade | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| thinning (como foi adotado) | +0,0040 | [+0,0001; +0,0074] | **sim** |
| **board (correta)** | **+0,0031** | **[−0,0002; +0,0055]** | **não** |

**Uma mudança em produção deixa de passar a regra do R0.** Mecanismo previsto pelo próprio registro:
`configs/default.yaml:weights` diz que o ganho do X1 vive em `50<t≤150` (+0,008) e `150<t≤400`
(+0,004), e a reauditoria mostra `t>400` em **−0,0003**. Ele ganha exatamente nos buckets cujo peso a
grade antiga inflava (26,6% → 17,5%) e é levemente negativo naquele que ela subpesava (16,5% → 31,9%).

**Não superinterpretar:** o ponto ainda é +0,0031, acima da barra — isto não diz que o X1 machuca,
diz que a base para adotá-lo era mais fraca do que o registro afirma. E o B6 foi na direção oposta na
mesma reauditoria: **a correção reordena os braços, não desloca todos igualmente.**

**Consequência para o cardápio:** o C2 (vizinhança do X1 — `detect_floor`, `d_i` com `delta_mean`)
deixa de ser "afinar um ganho estabelecido" e passa a ser "verificar se há ganho". E vale medir o
X1 desligado (`weights.detectability_mode: none`) sobre o pacote atual, que nunca foi feito na
presença do B6 — as duas adoções nunca foram testadas uma contra a outra sob a grade certa.

**A4 — a suspeita do `v4_k4` era artefato de comparação (2026-07-25).**

O `CAMPANHA_POLIMENTO.md` §A4 registrava: *"a higiene do `v4_k4 = 0,6133 > incumbente 0,6125` — se
esse número for real e não artefato, o incumbente está mal escolhido e o polimento inteiro parte da
referência errada"*.

| grade | Δ (v4_k4 − incumbente) | IC 95% |
|---|---|---|
| thinning | −0,0006 | [−0,0040; +0,0031] |
| **board** | **−0,0019** | [−0,0049; +0,0015] |

**Era artefato de comparação, não de grade.** O `0,6125` citado é o **B6 sozinho**; o incumbente é o
B6+C6, que mede **0,6139** na mesma grade. Contra o incumbente de verdade o `v4_k4` é pior nas duas
ponderações, e mais claramente pior na do board (0,6259 contra 0,6277). **O incumbente não está mal
escolhido.**

Lição: dois números "quase iguais" citados de contextos diferentes já custaram tempo neste projeto
(a armadilha de baseline do `NOTAS_AGENTES.md` §9, com quatro V4 quase idênticos e um contaminado).
Aqui o mesmo padrão apareceu de novo — a defesa é sempre a mesma, comparar **pareado na mesma grade**
em vez de confrontar níveis anotados.

### 15.12 Saldo da reauditoria: 2 de 5 vereditos mudam

| decisão | thinning | **board** | |
|---|---|---|---|
| B6, partição 42 | +0,0023 · inclui 0 | **+0,0033 · exclui 0** | **muda A FAVOR** |
| B6, partição 43 | +0,0049 · exclui 0 | +0,0054 · exclui 0 | reforça |
| **X1-soft** | +0,0040 · exclui 0 | **+0,0031 · inclui 0** | **muda CONTRA** |
| C6 | +0,0014 · inclui 0 | +0,0014 · inclui 0 | imóvel |
| A4 (`v4_k4`) | −0,0006 | −0,0019 | fecha a suspeita |

**O padrão é o que importa: a correção REORDENA.** Se ela deslocasse todos os braços igualmente,
seria cosmética e nenhuma decisão mudaria. O que ela faz é redistribuir peso de `t≤150` para `t>400`,
então **premia braços com ganho em `t` alto (B6, E0c) e pune os que ganham em `t` baixo (X1-soft)**.
Isso dá o rastreio barato: para saber se um número histórico merece reauditoria, basta olhar o perfil
por bucket já registrado — nenhum bootstrap necessário.

### 15.13 A3(v) — o conjunto implantado mede igual ao medido, e o `mrep_` deixou de pagar

`p1_deploy193`: o conjunto que `default_blocks()` de fato emite (**193** = V4 183 + mrep 6 + bocpd 4)
contra o conjunto em que o incumbente foi **medido** (**187**, `--drop-prefix mrep_`). K=2, partição
42, grade do board:

| | nível |
|---|---|
| medido (187) | 0,62640 |
| implantado (193) | 0,62718 |
| **Δ** | **+0,0008 [−0,0015; +0,0029]** |

**O descasamento é real e imaterial.** Duas leituras:

1. **Tranquilizadora:** os números medidos do incumbente descrevem o pipeline implantado dentro do
   ruído. O risco levantado no §15.3 — de que os Δ do B6 e do C6 descrevessem uma base que a produção
   não usa — não se materializou.
2. **Acionável:** o `mrep_` foi **adotado valendo +0,0042** (IC excluindo 0) sobre a base V4
   (2026-07-22). Sobre a base atual, com B6 e C6, vale **+0,0008**. O ganho evaporou — o mesmo padrão
   de não-aditividade que o próprio `scorer.py` documenta ("V5+mrep: os ganhos NÃO SOMAM").

**Somando os dois passageiros:** `mrep_` (+0,0008, 6 colunas, ~120 µs/passo) e `bocpd_` (+0,0014 na
barra, 4 colunas, ~30 µs/passo). A produção carrega **10 features e ~150 µs/passo** por um ganho
conjunto que raspa o ruído. Nenhum dos dois é prejudicial; os dois são **peso morto provável**.

Isso não é uma recomendação de podar às cegas — a poda tem de ser medida como braço próprio, e o
histórico do V5 mostra que podar dois componentes juntos pode regredir. Mas é a primeira vez que os
dois aparecem com o custo e o benefício na mesma tabela, sob a grade certa.

### 15.14 D1 (`linear_tree`) sobrevive ao K=4 — o único braço vivo da campanha

Partição 42, grade do board, baseline `oof_b6c6_joint_bag4` = 0,62772:

| | Δ geral | IC 95% | `150–400` | `t>400` | `t≤50` |
|---|---|---|---|---|---|
| K=2 | +0,0025 | [−0,0007; +0,0053] | +0,0043 · exclui 0 | +0,0036 | −0,0119 |
| **K=4** | **+0,0027** | **[+0,0002; +0,0049] · EXCLUI 0** | **+0,0044 · exclui 0** | +0,0027 | −0,0041 |

Nível 0,62772 → **0,63046**. Três coisas separam este número do A3b, que caiu pela metade na mesma
transição:

1. **Não encolheu com sementes** — +0,0025 → +0,0027, e o IC agregado passou a excluir 0.
2. **O bucket-alvo é estável** (+0,0043 → +0,0044) e exclui 0 nas duas medições. Não é um bucket que
   apareceu depois de olhar o resultado: `150–400` é o default declarado a priori do
   `NOTAS_AGENTES.md` §5.
3. **O perfil tem mecanismo.** Ganha em `150–400` e `t>400` (78,6% do peso somados), perde em `t≤50`
   (3,9%) — exatamente onde 68% das colunas são NaN de warmup e uma regressão linear por folha é
   instável. É a previsão literal do E0c: o problema é aditivo e suave, e árvores de degraus
   aproximam curvas suaves com escadinhas.

**O que falta antes de virar decisão: a réplica na partição 43.** A variância de partição é ~0,007,
2,6× o efeito medido, e este projeto já adotou e retratou um braço em 24 h (a costura C5, §14.3) por
pular exatamente esse passo. Em treino em `scripts/run_polimento_fase4.sh`.

**E se passar, ainda não entra sozinho:** o par `D1 + A3b` (`p10_combo`) precisa ser medido junto,
porque a não-aditividade já apareceu três vezes neste repo (V5+mrep, B6+C6 parcial, e agora o `mrep_`
evaporando de +0,0042 para +0,0008 no §15.13).

### 15.15 ERRATA aos §15.13 e §15.11 — barra de adoção ≠ barra de manutenção

Os §15.13 e §15.11 foram escritos sugerindo que o `mrep_` seria "peso morto provável" e que o
X1-soft "deixa de passar a regra do R0". **As duas formulações estão erradas como base para agir**, e
ficam retificadas aqui. Os números permanecem; a leitura muda.

**1. Valor marginal contra o pacote final subestima o componente.** Toda mudança deste projeto foi
medida contra a base que existia quando entrou. Reavaliar cada uma contra tudo o que veio depois e
podar as que "não pagam mais" leva a **zero features por indução** — a certa altura a contribuição
marginal de qualquer peça contra todas as outras é pequena. Isso é redundância entre contribuições
correlacionadas, não inutilidade. O `mrep_` valer +0,0042 sobre o V4 e +0,0008 sobre o pacote atual é
o comportamento **esperado** de um componente correlacionado, não sintoma de que ele não serve.

**2. A barra de adoção não é a barra de manutenção.** Acrescentar custa complexidade, latência e
risco novos, e por isso precisa se justificar com IC excluindo 0. Manter algo já construído, testado,
empacotado e verificado bit-a-bit **não custa nada de novo**; **remover** custa retreino,
reverificação, e o risco de a estimativa marginal estar errada. Aplicar o mesmo limiar às duas
operações enviesa sistematicamente para desfazer o que funciona.

**3. "IC inclui 0" não é evidência de efeito nulo.** É ausência de evidência de efeito. O X1-soft
segue com ponto **+0,0031, acima da barra**, e o IC apenas encosta em zero (−0,0002).

**4. Coerência interna.** O §15.9 desta mesma campanha estabeleceu que Δ em K=2 **ordena braços mas
não estima efeitos** (o A3b caiu de +0,0024 para +0,0010 ao ganhar sementes). O +0,0008 do `mrep_` é
um número de **K=2** — usá-lo como veredito sobre um componente adotado com IC excluindo 0 é aplicar
a régua que a própria campanha acabara de desmontar.

**O que os dois resultados de fato estabelecem, e só isso:**
- §15.13 — auditoria de fidelidade: **os números do incumbente descrevem o pipeline implantado**
  (Δ +0,0008, indistinguível de 0). O `p1_deploy193` passou. Nenhuma implicação sobre o conjunto de
  features.
- §15.11 — achado de **método**: a correção de grade **reordena** os braços em vez de deslocá-los
  igualmente, o que importa para decisões **futuras**. Nenhuma implicação sobre desfazer adoções.

**Regra a herdar:** a reauditoria de grade serve para **calibrar como medir daqui em diante**, não
para reabrir o que já está em produção. Reabrir uma adoção exige um braço próprio, com a mesma
disciplina de qualquer outro (K=4, duas partições) — e com a barra invertida, porque quem propõe a
remoção é quem tem de mostrar que ela não custa.

### 15.16 ERRATA ao §15.14 — a "refutação" do D1 na partição 43 era INCONCLUSIVA

O §15.14 e a retratação que o seguiu trataram um número da partição 43 como refutação do D1. **Foi um
erro de método**, e o registro fica corrigido aqui.

**O que foi comparado:**

| | precisão | Δ | IC |
|---|---|---|---|
| partição 42 | **K=4** | +0,0027 | [+0,0002; +0,0049] |
| partição 43 | **K=2** | −0,0019 | **nenhum** |

**Por que isso não decide nada.** O ruído de semente que o bootstrap pareado **não enxerga** é
dp ≈ 0,0041 por modelo de semente única (§5 do `NOTAS_AGENTES.md`); num bag K=2 fica ~0,0029 por
lado, e a diferença pareada de dois bags K=2 carrega ~0,004 de ruído extra. Logo **−0,0019 ± ~0,004 é
compatível com o efeito verdadeiro ser +0,002.** Aquele número estabelece *inconclusivo*, não
refutação — e a própria campanha já tinha medido esse efeito: o A3b se moveu 0,0014 só ao ir de K=2
para K=4.

**O erro de fundo é uma assimetria de padrão probatório:** exigir K=4 com IC excluindo 0 para
**adotar**, e aceitar um ponto de K=2 sem IC para **rejeitar**. É o mesmo erro do §15.15 (barra de
adoção usada como barra de manutenção) em outra roupa. Evidência mais fraca não derruba evidência
mais forte só por ser mais nova e negativa.

**Estado correto do D1: INCONCLUSIVO.** Fica `false` em produção — não porque foi refutado, mas
porque não há evidência suficiente para mexer, e o default seguro quando a evidência é genuinamente
mista é não mexer. `scripts/run_d1_part43_k4.sh` leva a partição 43 a K=4 dos dois lados, que é a
única comparação que decide.

**Nota sobre o A2, para calibrar futuras leituras deste relatório.** A correção de grade é sustentada
por validação externa (prevê dois placares que não foram usados para construí-la, com erro caindo de
~0,011 para ~0,002). Mas o enquadramento de que ela "reordena o significado de quase todo número do
projeto" foi **inflado**. O que a reauditoria mostrou é majoritariamente **confirmatório**: B6
melhorou, C6 não mudou, X1-soft enfraqueceu mas seguiu com ponto acima da barra, e o E0c inverteu —
uma conclusão *pessimista* que caiu. **Nenhuma melhoria anterior foi desprovada.**

### 15.17 O par D1+A3b é PIOR que o D1 sozinho — a quarta não-aditividade

`p10_combo` (linear_tree + base_rate_weighted) contra o D1 sozinho, K=2, partição 42, grade do board:

| bucket | Δ (par − D1 sozinho) | IC 95% | |
|---|---|---|---|
| geral | −0,0018 | [−0,0063; +0,0019] | inconclusivo |
| `t≤50` | +0,0152 | [−0,0047; +0,0305] | 3,9% do peso |
| `150–400` | −0,0021 | [−0,0073; +0,0024] | |
| **`t>400`** | **−0,0069** | **[−0,0121; −0,0021]** | **EXCLUI 0** |

Acrescentar o A3b ao D1 **piora significativamente o bucket de maior peso** (31,9%). O perfil explica:
o A3b mexe no tratamento de `f(t)`, então redistribui desempenho ao longo de `t` — ganha em `t≤50` e
perde em `t>400`. Sob os pesos do board é um trade ruim.

**Quarta ocorrência de não-aditividade neste repo**, e já dá para tratar como regra em vez de
surpresa: V5+mrep (−0,0024), B6+C6 (só ~70% aditivo), `mrep_` caindo de +0,0042 para +0,0008 sobre a
base atual, e agora este. **Duas mudanças positivas isoladas não somam aqui.** Combinar tem de ser
medido, nunca assumido — e o pacote final da campanha precisa ser medido INTEIRO, como o
`CAMPANHA_POLIMENTO.md` já previa na semana 7.

**Consequência:** se o D1 sobreviver à réplica na partição 43, entra **sozinho**.

### 15.18 O gate do pacote: D1+B3 são aditivos (+0,0035, IC exclui 0)

O §15.17 fechou pedindo que o pacote fosse medido inteiro. Foi, na partição 42, grade do board:
candidato = `linear_tree` (K=4) + v-EMA assimétrico com gate `min_t=50`, baseline =
`oof_b6c6_joint_bag4`.

| bucket | Δ | IC 95% | exclui 0 | peso |
|---|---|---|---|---|
| **geral** | **+0,0035** | **[+0,0008; +0,0057]** | **sim** | 100% |
| `t≤50` | −0,0041 | [−0,0141; +0,0070] | não | 3,9% |
| `50<t≤150` | +0,0006 | [−0,0051; +0,0051] | não | 17,5% |
| **`150<t≤400`** | **+0,0052** | **[+0,0021; +0,0078]** | **sim** | 46,7% |
| **`t>400`** | **+0,0033** | **[+0,0003; +0,0067]** | **sim** | 31,9% |

TS-AUC **0,6277 → 0,6312**. D1 (+0,0027) + B3 (+0,0008) = +0,0035 — **exatamente o medido**.

Depois de quatro não-aditividades seguidas, esta é a primeira combinação do projeto que soma. O que
mudou não foi a sorte: as quatro anteriores combinavam mudanças que disputavam o **mesmo** mecanismo
(features redundantes, dois tratamentos de `f(t)`), enquanto aqui uma muda a **forma da folha** e a
outra é **pós-processo sobre a sequência de scores** — operam em estágios diferentes do pipeline. Não
é regra geral, mas é uma heurística melhor que "combinar é imprevisível".

Ressalvas: o artefato do `linear_tree` é K=4 e o YAML embute K=7 (o +0,0035 é **piso**, não estimativa
central), e a réplica na partição 43 continua devendo.

### 15.19 `linear_tree` quebrava a fusão de boosters — o defeito que só a produção via

Ao regenerar o notebook de submissão com o pacote aprovado, o `verify_submission_notebook.py` falhou
**no pacote real**, não no notebook:

```
RuntimeError: fuse_boosters: fusão INVÁLIDA (max |raw_fundido - media_raws| = 1.521e+02 > 1e-09)
```

`fuse_boosters` concatena as árvores de K boosters e divide as folhas por K — exato **enquanto a folha
é uma constante**, porque o raw do LightGBM é uma soma sobre árvores. Com `linear_tree=true` a folha
vale `leaf_const + Σ leaf_coeff_i · x_i`, e é por esses campos que o LightGBM prediz quando
`is_linear=1`. O `_scale_leaves` escalava só `leaf_value`, deixando `leaf_const` e **35.318
coeficientes por booster** intactos.

**Por que passou despercebido por um dia inteiro de medições.** O OOF nunca funde: `scripts/train.py`
faz média das **predições** por semente (`avg_oof.py`). A fusão existe só no caminho de **produção**
(`adapter/platform.py:train`, que treina do zero na nuvem). **Todo o +0,0027 do D1 foi medido num
caminho que a produção não usa.**

É a **terceira ocorrência da classe A3** ("o que é medido ≠ o que é implantado") e a primeira com
consequência fatal: o `train()` na nuvem teria levantado `RuntimeError` e derrubado a submissão
inteira. As duas anteriores (193 vs 187 features; `init_score` descasado) eram vieses de estimativa;
esta era uma falha dura.

**A guarda numérica fez o seu trabalho.** O docstring dela — *"falhar alto é infinitamente melhor que
submeter um modelo corrompido"* — descreve exatamente o que aconteceu. Sem ela, a fusão devolveria em
silêncio um modelo que não representa a média dos originais.

**Correção:** escalar `leaf_const` e `leaf_coeff` junto com `leaf_value` (exata pelo mesmo argumento
de linearidade que já justificava a fusão). `leaf_features` fica de fora — são **índices** de coluna —
e `leaf_count`/`leaf_weight` também, que são contagens.

| caminho | antes | depois |
|---|---|---|
| folhas lineares (5 folds do `p3_lintree`) | **1,1e+02** | **1,1e−13** |
| folhas constantes (regressão) | 4,0e−15 | 4,0e−15 (intocado) |

Coberto por `tests/unit/test_fuse_linear_tree.py` (os dois modos de folha, mais uma asserção de que as
árvores lineares do teste têm coeficientes de verdade — sem ela, folhas degeneradas fariam o teste
passar sem exercitar o caminho do D1).

**Lição de processo, e a mais cara desta campanha:** a verificação bit-a-bit era tratada como portão
de **empacotamento**, rodada no fim. Ela é, de fato, o único ponto do projeto que exercita o caminho
de produção — logo é portão de **adoção**. Um flag novo deveria passar por ela **antes** de entrar no
`default.yaml`, não depois. Ver `PROXIMA_SEMANA.md`, item 1-bis.

---

## 16. Rodada 11 (2026-09-29/30) — Repensar do zero: o teto é de amostra de eventos, e o extra-trees paga

Registro detalhado em [`knowledge/frentes/`](../frentes/). Aqui fica o resumo e as decisões.

### 16.1 A premissa da surpresa acumulada (frente [surpresa-acumulada](../frentes/surpresa-acumulada/README.md))

Detector novo (`src/surpresa/`): modelo por série → fluxo universal N(0,1) → razões de verossimilhança
direcionais acumuladas. Sozinho dá **0,575** (no nível das melhores colunas do Onyx). Somado ao OOF,
**+0,0001**, nulo. Como features (S5), +0,0044 com IC excluindo 0, mas uma semente em quatro negativa e
a checagem de mecanismo falhou: não adotado. **A surpresa ingênua (Σ −log p) fica abaixo do acaso em
quebras de dependência e cauda.**

### 16.2 O diagnóstico do teto (frente [teto-offline](../frentes/teto-offline/README.md))

| Pergunta | Resposta medida |
|---|---|
| Com τ conhecido, quanto 118 features de duas amostras separam? (T1) | 0,609 / 0,666 / 0,689 (L = 100 / 200 / 400), **igual ao Onyx** no mesmo ponto |
| Representações aprendidas acrescentam? (T2, T3) | ROCKET 0,600, Chronos-Bolt 0,53, nenhuma soma |
| O Onyx é limitado por amostra? (C1) | **sim: ×2 séries = +0,0144** |
| Por positivos ou negativos? (C2) | **só positivos**: negativos a mais = −0,0003 |
| Há fonte de eventos extras? (A1, A2, D25) | re-corte −0,012 (duplica eventos), transplante não transfere, 2025 é **permitido** mas é outra distribuição (AUC de domínio 0,962) |
| Há canal fora do sinal? | não: comprimento, padronização, repetição exata ou afim, teste×treino, cardápio discreto e deriva, todos fechados |

### 16.3 A alavanca: reduzir a variância do aprendiz

**E1, `lightgbm.extra_trees: true`, ADOTADO.** Partição 42, K=4: **+0,0071 [+0,0038; +0,0102]**, 4/4
sementes (0,6278 → **0,6349**). Partição 43, K=2: **+0,0115 [+0,0080; +0,0159]**, IC exclui 0 em todos
os buckets. É o maior ganho isolado do projeto. Smoke test e notebook verificados bit a bit.

Empilhamentos sobre o E1 ([V1–V8](../frentes/teto-offline/V-empilhamento.md)): `feature_fraction` 0,5
nulo; metade dos negativos −0,0088; receita histórica nula; os demais na fila na hora deste registro.
Refit com 100% das séries ([R1](../frentes/teto-offline/R1-refit-completo.md)): inconclusivo,
implementado e desligado.

### 16.4 Lições

1. **Medir o teto antes de construir** (a F4 do DIAGNOSTICO_ESTRUTURAL, feita finalmente): o oráculo
   com τ conhecido mostrou que o problema não era a arquitetura sequencial.
2. **A curva de aprendizado é o instrumento mais barato e mais decisivo**: ×2 dados = +0,0144 disse,
   em 15 minutos, que a alavanca é variância e não features.
3. **Duplicar eventos não é aumentar dados**: o re-corte piorou −0,012.
4. **Premissa escrita não é premissa verificada**: "dados de 2025 não são permitidos" estava errado. A
   doc oficial permite.
