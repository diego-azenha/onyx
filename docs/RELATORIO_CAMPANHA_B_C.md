# Relatório — Campanha B/C (T0, A, B, C) pós-X1

**Data:** 2026-07-24 · **Repo:** `onyx` (ADIA Lab Structural Break Challenge — Real-Time) · **Fonte:** `BRAINSTORM_RUPTURA_V2.md`

Documento autocontido para handoff a um agente externo. Continuação de `RELATORIO_CAMPANHA_X0_X4.md`. Rodou em duas sessões — a primeira caiu no meio do B4 e os veredictos foram reconstruídos do transcript, então **este relatório cobre as duas**.

**Resultado: 1 adoção (B6), 9 mortes, 2 eixos formalmente fechados, 1 braço que foi adotado e retratado no mesmo dia, e um achado metodológico (variância de partição) que reorganiza como o projeto mede daqui pra frente.**

---

## 1. Sumário executivo

**Incumbente de entrada:** V4 + X1-soft, K=4 sementes limpas = `oof_x1_soft_bag4` = **0,6102** (partição de folds 42). **Barra:** ~0,0014.

| braço | eixo | veredito | Δ TS-AUC (OOF pareado) | decisão |
|---|---|---|---|---|
| **B6** | `feature_contri=0,6` em `meta_h0_*` | **✓ ADOTADO** | part. 42 **+0,0023** [−0,0010; +0,0053] · part. 43 **+0,0049** [+0,0018; +0,0079] **exclui 0** | **adotar** |
| C5 | modelo por regime de t (costura de OOFs) | ⚠ **adotado e RETRATADO** | +0,0032 [+0,0012; +0,0051] na part. 42, mas **perde para o B6 sozinho** na 43 | descartar |
| A1 | EWMA assimétrico no pós-processo | ~ passa sozinho, **fora do pacote** | +0,0005 [0,0000; +0,0010]; só **+0,0004** em cima do B6 | não empacotar |
| A3 | emulador do nulo REAL + studentização | ✗ morto | **−0,0457** | **fecha o eixo** |
| B1 | pesos de linha por `w_t = n_pos·n_neg` | ✗ morto | −0,0030 [−0,0058; −0,0002] (exclui 0 *contra*) | descartar |
| B2 | variante `ramp` do X1 | ✗ morto | −0,0038 | descartar (confirma `soft`) |
| B3 | rebaixar negativos de rótulo-suspeito | ✗ morto | −0,0009 [−0,0036; +0,0015] | descartar |
| B4 | objetivo custom logloss+pairwise intra-t | ✗ morto / inconclusivo | α=0,3 **−0,0035** · α=0,1 +0,0007 | descartar α=0,3; α=0,1 não vale K=4 |
| B5 | `monotone_constraints` em acumuladores de LLR | ✗ neutro | −0,0005 [−0,0034; +0,0022] | descartar |
| C2 | conserto do portador "as-of" + re-probe | ✗ bloqueado | probe 0,9993 (alvo ≤0,55) | **fecha o eixo** |
| C3 | bagging de partição | ✗ morto | −0,0014 [−0,0039; +0,0013] | descartar, **mas ver §4** |
| T0.1 | diagnóstico de pares difíceis | — | perda **difusa** (5% piores = 16% da massa) | sem alvo concentrado |
| T0.2 | rank-average de sementes homogêneas | ✗ ~0 | −0,0001 | eixo fechado |
| A2 | stacking heterogêneo | ✗ bloqueado pelo ambiente | lambdarank >15 min sem saída | reabrir só com mais compute |
| C6 | BOCPD isolado sobre o V4 | *(preenchido em §3.C6)* | — | — |
| C4 | varredura de hiperparâmetros | **não rodado** (decisão do usuário) | — | ver §7 |

---

## 2. Metodologia — o que mudou em relação à campanha X

Os trilhos do relatório anterior continuam (K=4 sementes `[777,101,202,303]` com a 42 excluída, `compare_oof.py` com IC 95% por bootstrap pareado, adotar sse o IC exclui 0 no agregado ou no bucket-alvo a priori). **Uma exigência nova entrou, e é a lição central da campanha:**

> **Replicar na outra PARTIÇÃO DE FOLDS antes de acreditar em qualquer efeito de ~0,003.**

O bootstrap pareado do R0 reamostra séries **dentro de uma partição fixa** — ele não enxerga a variância de trocar a partição, que esta campanha mediu em **~0,007** (§4). Um IC que exclui 0 continua condicionado à partição em que foi medido. Custo da réplica: ~40 min (`--fold-seed 43`).

---

## 3. Braço a braço

### B6 — `feature_contri` penalizando `meta_h0_*` · **ADOTADO**

**Mecanismo.** `feature_contri=0,6` nas colunas `meta_h0_*` encolhe o ganho que a árvore pode extrair delas. A hipótese: essas colunas descrevem o **ajuste do H0 da própria série** (um condicionador — "que tipo de série é esta"), e o LightGBM as estava usando como quase-intercepto por série, o que não ajuda uma AUC calculada **dentro** de cada passo. Penalizá-las força o resto do banco a carregar a decisão.

**Resultado (as duas partições):**

| | partição 42 (a sortuda) | **partição 43 (independente)** |
|---|---|---|
| geral | +0,0023 [−0,0010; +0,0053] (inclui 0) | **+0,0049 [+0,0018; +0,0079] exclui 0** |
| 150–400 | +0,0049 [0,0012; 0,0085] exclui 0 | **+0,0065 [0,0030; 0,0102] exclui 0** |
| t>400 | +0,0048 [0,0009; 0,0084] exclui 0 | **+0,0050 [0,0017; 0,0084] exclui 0** |
| 50–150 | −0,0029 | +0,0039 |
| t≤50 | −0,0011 | −0,0016 |

Na partição 42 o agregado era inconclusivo e só os buckets altos passavam. **Na partição que não foi usada para desenhar nada, o IC do agregado exclui 0.** Média ~+0,0036 = 2,6× a barra. O ganho vive em 150–400 e t>400 = **65% do peso da métrica**.

**Implantação:** um knob — `configs/default.yaml: lightgbm.feature_contri_meta: 0.6`. `model/train.py` já monta o vetor `feature_contri` alinhado a `feature_cols`. **Sem** mudança em `adapter/platform.py`, **sem** custo extra de treino ou de inferência.

### C5 — Modelo por regime de t · **ADOTADO DE MANHÃ, RETRATADO À TARDE**

Vale registrar inteiro porque o erro é instrutivo.

O C5 original pedia dois modelos treinados (t≤64 / t>64). Não foi preciso treinar nada: o perfil por bucket do B6 (negativo em 50–150, positivo acima de 150) **é** o conflito de regimes postulado, e a **C1** (`MODELO.md` §1.2) autoriza trocar de modelo por t — a `AUC_t` só compara scores **dentro** de um passo, então costurar dois OOFs numa borda de bucket é legítimo pela métrica e sai de graça. Não há o problema de comparabilidade que matou X3/A3.

Na partição 42 a costura (X1-soft em t≤150, B6 acima) mediu **+0,0032 [+0,0012; +0,0051]**, com dois sinais de solidez: os folds de **confirmação** deram Δ *maior* que os de seleção (+0,0044 vs +0,0025), e o ganho é um **platô** no corte (+0,0028 a +0,0032 para qualquer corte entre 100 e 200), não uma quina.

**A réplica na partição 43 derrubou o desenho.** O ganho da costura sobre o baseline replica (+0,0032 → +0,0040), mas **o B6 sozinho (+0,0049) bate a costura (+0,0040)**, porque o bucket que motivava a costura — 50–150 — **troca de sinal** entre partições (−0,0029 → +0,0039). O "conflito de regimes" era ruído de partição.

**Lição:** IC limpo, gate de confirmação passado e robustez ao hiperparâmetro do corte **não** bastaram. Só a réplica de partição pegou. Custo evitado: um modelo com 2× o treino na nuvem que era **pior** que o simples.

### A1 — EWMA assimétrico no pós-processo · **passa sozinho, fora do pacote**

+0,0005 [0,0000; +0,0010] sozinho, com 150–400 +0,0009. Mas **em cima do B6** rende só +0,0004 (global) / +0,0004 (com gate em t>150), ambos abaixo da barra — A1 também é braço de t alto e **se sobrepõe** ao B6. Ganhos não somam. Código preservado.

### A3 e C2 — os dois eixos que FECHARAM

**A3 (comparabilidade auto-referencial).** Studentizar o score pelo nulo da própria série já tinha morrido duas vezes (centragem aditiva −0,002; X3 com CDF nula do histórico −0,029). O A3 fez a versão certa — emulador do nulo **real** (LGBM pinball q10/q50/q90 sobre linhas online reais sob H0, out-of-fold) — e deu **−0,0457**, a pior das três. A cláusula pré-registrada era "se falhar com o nulo correto, fecha o eixo". **Fechado.**

**C2 (fábrica de nulos sintéticos).** O X4 tinha sido bloqueado por um probe de artefato (discriminador real-vs-sintético AUC 0,9998; alvo ≤0,6). O C2 consertou a hipótese óbvia (suporte de comprimento) e o probe ficou em **0,9993**. A causa não é suporte: é **fosso de domínio** — o online real é do período 2 com `meta_h0`/whitening ajustados no histórico inteiro, enquanto o sintético é uma fatia do próprio histórico com H0 do histórico anterior (limpo demais, dentro do regime). Reabrir exige redesenhar o portador (~2 dias).

### B4 — Objetivo custom logloss+pairwise · **morto, e dois bugs no caminho**

Antes de gastar ~3 h em 2×K=4, dois diagnósticos de ~12 min resolveram — e acharam dois defeitos reais:

1. **Desempenho:** o pré-sorteio de pares varria as ~506k linhas positivas em laço Python com uma varredura de 2,5 M linhas por valor de t → reescrito como um `groupby` sobre ~399 valores de t: **60 s → 0,2 s por fold**.
2. **Bug:** com objetivo **custom** o LightGBM entrega ao `feval` o **score bruto** (com o `binary` embutido, entrega probabilidade). Faltava `raw_to_prob`, então `binary_logloss_diag` lia **5,20** contra 0,66 do incumbente — e, como `early_stopping_metric=logloss`, **a parada antecipada rodava nesse sinal sem sentido**.

Como a `ts_auc` é invariante à sigmoide, ela nunca foi afetada — o que deu um veredicto **independente da regra de parada**: melhor `ts_auc` por fold alcançável (limite superior, favorável ao B4) = x1_soft 0,6102 · b6 0,6114 · **α=0,3 0,6067** · **α=0,1 0,6109** (+0,0007, metade da barra, dentro de 1σ de uma semente). Nenhum K=4 gasto.

### B1, B2, B3, B5, T0.1, T0.2, A2 — mortes rápidas

- **B1** (massa de peso ∝ `n_pos·n_neg`): −0,0030, IC exclui 0 **contra**. Superpondera t alto às custas dos buckets médios onde o sinal está.
- **B2** (variante `ramp` do X1): −0,0038; confirma que `soft` era o sabor certo (hard/soft ganharam, ramp perde).
- **B3** (rebaixar negativos rótulo-suspeito): −0,0009, inconclusivo — exatamente o que o T0.1 previu (só 0,60% das séries marcadas).
- **B5** (`monotone_constraints=+1` em 9 acumuladores de LLR): −0,0005, neutro. Regularização estrutural não adiciona.
- **T0.1**: a perda é **difusa** — as 0,5% piores séries carregam 2,0% da massa de pares invertidos; 5% → 16%; 10% → 29%. **Não existe alvo pequeno e dominante.** Foi este diagnóstico que previu corretamente a morte do B3.
- **T0.2**: rank-average de 4 sementes **homogêneas** dá −0,0001 (Jensen sobre membros quase idênticos ≈ 0). Combinador reutilizável em `scripts/rank_avg_oof.py`.
- **A2**: bloqueado pelo ambiente (lambdarank >15 min sem saída; membro historicamente fraco ~0,585 vs 0,610).

### C6 — BOCPD isolado sobre o V4

*(Preenchido quando o braço terminar — build do dataset com `SBRT_ENABLE_BOCPD=1` + K=4. Ver §3.C6 na versão final.)*

---

## 4. Achado transversal — variância de PARTIÇÃO (o mais importante da campanha)

O C3 morreu como braço (bagging de partição −0,0014) mas mediu o que ninguém tinha medido: trocar a **partição de folds** (`cfg.seed` 42→43), tudo o mais igual, move a TS-AUC em **~0,007**:

- **7×** a variância de semente de boosting (σ ≈ 0,0010, a régua da campanha anterior);
- **5×** a barra de decisão (0,0014);
- e a **partição 42 é a sortuda**: o mesmo X1-soft mede **0,6102** em 42 e **0,6029** em 43.

É o mesmo padrão da semente 42 da campanha anterior, um nível acima na hierarquia. **Qualquer efeito da ordem de 0,003 medido numa partição só é indistinguível de sorte de partição** — foi o que aconteceu com a costura C5. O R0 continua válido como juiz *pareado*, mas o resultado é condicional à partição.

**Recomendação para o próximo agente:** todo braço que sobreviver ao R0 na partição 42 deve ser replicado com `--fold-seed 43` antes de entrar em qualquer pacote. Sobretudo se a decisão se apoiar em **um** bucket.

---

## 5. Implementação recomendada

Uma linha em `configs/default.yaml`:

```yaml
lightgbm:
  feature_contri_meta: 0.6   # B6 — penaliza meta_h0_* (condicionador, não intercepto por série)
```

`model/train.py` já monta `params["feature_contri"]` alinhado a `feature_cols` quando o valor ≠ 1,0. **Não requer** mudança em `adapter/platform.py`, nem custo extra de treino/inferência.

**Ainda NÃO aplicado** ao `configs/default.yaml` nem ao notebook de submissão — é um passo deliberado à parte, que exige regenerar o notebook e re-verificar bit-a-bit (`scripts/verify_submission_notebook.py`).

---

## 6. Artefatos e código produzidos

| arquivo | o que é |
|---|---|
| `scripts/c5_regime_splice.py` | costura de dois OOFs por faixa de t + teste honesto (seleção em 3 folds, confirmação em 2). Mantido: a **técnica** é válida pela C1, o que falhou foi a hipótese de conflito de regimes |
| `scripts/apply_vema_oof.py` | aplica o v-ema do A1 sobre um OOF qualquer (com `--min-t` para gate por regime), para empilhar sem retreinar |
| `scripts/run_partb_b6.sh` | réplica na partição 43, **resumível** |
| `scripts/run_c6_bocpd.sh` | C6 com `SBRT_ENABLE_BOCPD=1`, **resumível** |
| `scripts/rank_avg_oof.py`, `scripts/hard_pair_analysis.py`, `scripts/null_emulator_studentize.py`, `scripts/postproc_vema_probe.py`, `scripts/stack_by_bucket.py`, `scripts/fallback_oof.py` | da primeira sessão (T0.2, T0.1, A3, A1) |
| `src/sbrt/model/train.py` | fix de desempenho do pré-sorteio de pares; fix `raw_to_prob` no `feval` sob objetivo custom |
| `src/sbrt/state/scorer.py` | BOCPD opt-in por `SBRT_ENABLE_BOCPD=1` (default de produção inalterado, verificado) |
| `scripts/sweep_hyperparams.py` | ganhou `--drop-prefix` e `--detectability-mode` (sem eles a varredura seria confundida em silêncio — ver §7) |
| `artifacts/reports/compare_*.json` | todos os R0 da campanha |

---

## 7. Nota operacional (armadilhas que custaram tempo)

- **Armadilha de baseline.** Quatro números quase iguais convivem no repo e **um é contaminado**: 0,6018 = média das TS-AUC *individuais* das 4 sementes limpas · **0,6062 = `oof_x0_a_bag4`, o bag limpo, é o baseline correto** · 0,6084 = `oof_b183_bag4`, que apesar do nome é a média de **(42, 101, 202, 777)** e portanto **contém a semente 42** (lê +0,0022 alto) · 0,6100 = semente 42 sozinha. **Não usar `oof_b183_bag4` como baseline.**
- **`sweep_hyperparams.py` estava mudando três coisas de uma vez.** Chamava `compute_row_weights(rows, cfg)` sem modo de detectabilidade e não tinha `--drop-prefix` — cada célula diferia do incumbente em **duas** coisas além dos HPs. Corrigido antes de qualquer uso.
- **Neste ambiente, wrappers de job em background são mortos no meio** (~5–35 min; verificado que **não** é suspensão da máquina). O trabalho sobrevive com `nohup <cmd> > log 2>&1 &`. Scripts de campanha devem ser **resumíveis**.
- **`| tail -N` em job de background esconde o progresso** (já estava em `NOTAS_AGENTES.md` §7 — e ainda assim custou 36 min de treino sem visibilidade).

---

## 8. Pendências e recomendações

1. **Aplicar o B6** em `configs/default.yaml` + regenerar e verificar o notebook de submissão.
2. **Submissão oficial** do V4+X1-soft+B6 — o mapa OOF↔placar continua desconhecido (um ponto só: `5e42ff5` marcou 0,6201 no placar contra 0,6100 de OOF).
3. **Replicar na partição 43** qualquer braço futuro antes de empacotar (§4).
4. **Não reabrir:** studentização auto-referencial (§3.A3) e injeção sintética sem redesenho do portador (§3.C2).
5. **Reabrível com mais compute:** A2 (stacking heterogêneo) e C4 (varredura de HP, harness já corrigido) — ambos de prioridade baixa: A2 depende de um membro fraco, e C4 caça efeitos na escala que o §4 mostrou ser ruído de partição.
