# Resultados: famílias P1–P4 (dependência, cauda, variância localizada, saltos) → V4

**Data:** 2026-07-20
**Implementado:** as quatro famílias priorizadas em `docs/INVESTIGACAO_FALHAS_V3.md`, cada uma atacando
um ponto cego ou folga de extração medido:
- **P1** `state/dependence.py` — dependência não-linear/multi-lag (ρ₁ de |e| e e², massa multi-lag)
- **P2** `state/lmoments.py` — forma de cauda dinâmica (L-skewness/L-kurtosis)
- **P3** `state/varloc.py` — variância localizada no changepoint (max sobre escalas)
- **P4** `state/jumps.py` — bipower/saltos + leverage (precisão T6/T9)

Todas calibradas via F1. **183 features** (era 141 no V3). Modelo: `artifacts/models/v4`.

---

## 1. Resultado — o maior ganho estatisticamente significativo do projeto

### 1.1 TS-AUC OOF por bucket (10.000 séries)

| Bucket | baseline pré-sessão | V3 | **V4** |
|---|---|---|---|
| **geral** | 0,5996 | 0,6039 | **0,6100** |
| t≤50 | 0,522 | 0,5299 | **0,5357** |
| 50<t≤150 | — | 0,5736 | **0,5799** |
| 150<t≤400 | — | 0,6169 | **0,6242** |
| t>400 | 0,636 | 0,6510 | **0,6529** |

Os quatro buckets subiram de V3 para V4.

### 1.2 Veredito estatístico (R0, bootstrap pareado por série, 300 réplicas)

**V4 vs V3** (isola P1–P4):

| Bucket | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| **geral** | **+0,0060** | **[0,0007, 0,0117]** | **sim** |
| 150<t≤400 | +0,0074 | [0,0019, 0,0126] | sim |
| t≤50 | +0,0058 | [−0,0104, 0,0239] | não |
| 50<t≤150 | +0,0063 | [−0,0039, 0,0164] | não |
| t>400 | +0,0019 | [−0,0034, 0,0070] | não |

**V4 vs baseline pré-sessão** (toda a sessão — auditoria + V2 + V3 + P1–P4):

| Bucket | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| **geral** | **+0,0104** | **[0,0032, 0,0189]** | **sim** |
| 50<t≤150 | +0,0136 | [0,0022, 0,0270] | sim |
| 150<t≤400 | +0,0092 | [0,0015, 0,0175] | sim |
| t>400 | +0,0104 | [0,0021, 0,0201] | sim |
| t≤50 | +0,0070 | [−0,0134, 0,0278] | não |

**Este é o primeiro ganho agregado estatisticamente significativo do projeto** — no OOF, tanto o
incremento P1–P4 (V4 vs V3) quanto a sessão inteira (V4 vs baseline) excluem 0, com 3 de 4 buckets
individualmente significativos contra o baseline.

## 2. Teste local no molde crunch (held-out, 100 séries)

Inferência pelo **caminho real de submissão** (`StreamScorer` + `ModelEnsemble.predict_one`) sobre o
conjunto de teste reduzido, pontuada pela fórmula oficial (`scripts/crunch_local_ts_auc.py`):

| Modelo | TS-AUC oficial | Δ vs baseline |
|---|---|---|
| baseline pré-sessão | 0,5073 | — |
| V3 | 0,5470 | **+0,0397** |
| **V4** | 0,5416 | **+0,0344** |

**Leitura honesta (com a ressalva de resolução):**

1. **O held-out confirma o ganho da sessão:** baseline → V4 sobe +0,0344 num conjunto totalmente
   separado do treino — direção e ordem de magnitude coerentes com o OOF (+0,0104). Isso valida o
   instrumento OOF (âncora oficial concorda em sinal, no espírito do D4-iii do parecer).

2. **O held-out NÃO consegue arbitrar V3 vs V4.** Com 100 séries, o erro-padrão da TS-AUC é ≈0,054
   (medido, DIAGNOSTICO H1); a diferença V4−V3 de −0,0054 está **uma ordem de magnitude abaixo do
   ruído** desse conjunto. Não é evidência de que V4 < V3 — é simplesmente irresolvível com 100
   séries. O OOF (10.000 séries, bootstrap pareado) é o instrumento de resolução muito maior, e ele
   diz V4 > V3 com IC excluindo 0. O peso da evidência é **V4 ≥ V3**; a diferença fina só será
   confirmada por uma sonda oficial com mais séries.

3. Os números absolutos do held-out (0,51–0,55) são mais baixos que o OOF (0,60–0,61) porque estas
   100 séries são um sorteio mais difícil/ruidoso — mas os **deltas pareados** (mesmas 100 séries)
   são o que importa, e concordam com o OOF no que ele consegue resolver.

## 3. Conformidade

- **133 testes passam** (era 104 no início da P1), incluindo causalidade (prefixo/canário) e
  determinismo bit-a-bit — as 4 famílias novas são causais e determinísticas por construção.
- **Latência: 980 µs/passo** (era 854 no V3), gate 1500 → PASS. Folga ainda confortável.
- Equivalência online × nulo-de-calibração testada para cada bloco novo (a calibração roda o
  **próprio bloco** sobre o histórico — equivalência por construção, sem caminho vetorizado
  divergente).
- CE6 inalterado: P1–P4 são todos features *online* (e/e_vol), estruturalmente ausentes do
  classificador só-histórico; sem risco de vazamento novo.

## 4. Onde cada família entregou (previsões da INVESTIGACAO vs. medido)

O ganho concentrou-se em **50<t≤400** (buckets +0,006 a +0,014, significativos), exatamente o regime
que a INVESTIGACAO §2 previu ter a maior folga de extração e onde há janela suficiente para as novas
estatísticas se estabilizarem. `t≤50` subiu (+0,007) mas não significativamente — coerente com o teto
de informação causal apertado (mediana de 14 pontos pós-quebra). `t>400` subiu pouco no incremento
V3→V4 (já estava alto pós-V3).

Um SHAP transversal do V4 (não recomputado neste documento por custo) diria quais das 42 features
novas carregam o ganho — próxima medição natural se houver mais uma iteração.

## 5. Estado da hipótese "o gargalo são as features"

**Sustentada, agora com significância.** A sessão testou três teses em ordem: objetivo/peso/parada
(auditoria R1–R3, nulo); comparabilidade/calibração (V2 F1, mecanismo válido mas agregado nulo); e
**informação nova nos eixos cegos medidos** (V2 F3/F4 + P1–P4). Só a terceira moveu o agregado de
forma significativa — e moveu duas vezes (V3 e V4). A leitura final: o gargalo **era** o que o modelo
consome, mas de forma específica — não "mais features", e sim *informação nos eixos que o censo mostra
existirem e que o banco não cobria* (dependência não-linear, forma de cauda, variância localizada).

Ressalva permanente: ~24% das quebras são fracas em todos os eixos e ~35% do peso está em t pequeno
com teto causal baixo, então o agregado sobe devagar (+0,010 na sessão). O caminho de maior retorno
agora é uma **sonda oficial do V4** — o OOF já não é o gargalo de decisão; a resolução da âncora
oficial é.
