# S6 — Especialistas por tipo de quebra (o tipo como supervisão extra)

**Frente:** [teto offline](README.md) · **Status:** concluído: **nulo/negativo**, com uma lição estrutural · **Hipótese:** 2026-09-30 14:40

## Por quê

O Onyx é limitado por eventos ([C1](C1-curva-aprendizado-onyx.md)). No treino, **sabemos o tipo** de cada
quebra, e isso é informação extra por evento que o GBM único não usa: ele aprende tudo misturado, e as
quedas de variância (~17% dos positivos) ficam diluídas. Caso motivador: nas quedas, o Onyx separa 0,564,
e um sinal especialista separa 0,774 ([ST1](ST1-segundo-estagio.md)). Tipos por série
(`artifacts/roteiro/tipo_quebra.parquet`): 5.033 negativos, 3.092 fixa, 908 desce, 759 sobe e 208 curtas
(que vão para fixa).

## Desenho

- `scripts/s6_especialistas.py`: três especialistas (sobe, desce, fixa), cada um com a receita do E1.
  Os positivos dos outros tipos saem do treino; a validação fica com todas as linhas. Folds da partição
  42, sementes 777 e 101.
- Combinação: logística cross-fit por paridade de id sobre (logit E1, logits dos 3 especialistas, log t),
  avaliada contra o E1. Também um blend simples, E1 + máximo dos especialistas.

## Hipótese (escrita ANTES de medir)

- Combinação contra o E1: **+0,000 a +0,008**. O ganho deveria aparecer nas quedas (`desce`).
- Adotar a direção (K=4 e réplica) só com ≥ +0,004 na combinação cross-fit.
- Risco: cada especialista vê menos positivos (759–3.300), e isso pode anular a supervisão extra.

## Resultado

Especialistas com 2 sementes (777, 101), ~5 min cada.

**Dentro do próprio tipo, os especialistas vencem o E1 com folga** (TS-AUC com positivos do tipo contra
todos os negativos):

| Tipo | E1 | Especialista do tipo |
|---|---|---|
| queda (`desce`) | 0,575 | **0,799** |
| alta (`sobe`) | 0,834 | **0,860** |
| `fixa` | 0,608 | **0,626** |

**Mas nenhuma combinação global aproveita isso:**

| Combinação (cross-fit por paridade) | TS-AUC geral |
|---|---|
| E1 | **0,6349** |
| logística linear [E1, 3 especialistas, log t] | 0,6345 |
| E1 + w·máx(especialistas) | 0,6342 |
| log-sum-exp dos especialistas (a combinação bayesiana de uma mistura) | 0,578 (puro) / 0,582 (com c_k ajustado) |
| E1 + log-sum-exp | 0,6344 |
| GBM pequeno [E1, 3 especialistas, log t] | 0,6285 |
| GBM só com os especialistas | 0,6051 |

## Decisão e lição

**Descartado.** A lição vale para o projeto inteiro, e é a terceira vez que aparece (ST1, P4, S6):
**AUC por tipo de quebra não vira AUC global.** No teste o tipo é desconhecido. Cada especialista separa
bem o seu tipo dos negativos, mas se comporta como ruído nos outros tipos. Qualquer agregação (máximo,
soma de chances, stacker) dá aos negativos o "melhor de três" desses ruídos, e com isso eles sobem mais
que os positivos. O modelo único já aprendeu o que é transferível. **Não reabrir abordagens "por tipo"
sem uma forma de identificar o tipo a partir do trecho observado.**

## Por que nem o stacker aproveita as quedas, e o número que orienta o resto

Por tipo, o stacker GBM [E1, especialistas, t] **não melhora nem as quedas**: desce 0,575 → 0,565,
sobe 0,834 → 0,820, fixa 0,608 → 0,604. A ordenação ótima no agregado sacrifica os tipos minoritários,
porque **o peso da métrica está concentrado nas quebras `fixa`**. Fração dos pares positivo × negativo
(ponderados pela grade do board):

| Tipo | Peso na TS-AUC | E1 no subconjunto |
|---|---|---|
| **fixa** (\|Δlogvar\| ≤ 0,3) | **70,7%** | 0,608 |
| sobe | 15,6% | 0,834 |
| desce | 13,6% | 0,575 |
| curta (<10 pontos pós) | 0,1% | 0,532 |

**Consequência:** ir de 0,635 a 0,66 no geral exige quase +0,035 nas `fixa`, justamente onde a sonda
(0,578), o censo espectral (0,579) e o especialista do tipo (0,626) apontam um teto de ~0,60–0,63 com
estas features. As quedas (13,6%) não salvam, nem com o especialista perfeito. **O ganho do pelotão,
seja qual for, tem de estar principalmente nas quebras que mudam pouco a variância.**

### Dentro das `fixa`: o que o E1 usa (m = 100 e 200, 700–780 quebras com ≥ 200 pontos pós)

Spearman entre o percentil do E1 e os eixos do censo: |Δlogvar| +0,13/+0,18 · |ΔACF₁| +0,13/+0,14 ·
|Δcurtose| +0,17/+0,08 · |ΔACF de |x|| +0,19/+0,16 · |Δmédia| 0,00/+0,02. Por quartil de |Δlogvar|
(m=200): ≤0,06 → 0,542 · 0,06–0,11 → 0,546 · 0,11–0,19 → 0,645 · 0,19–0,30 → 0,655.

**Metade das `fixa` (|Δlogvar| < 0,11) fica praticamente no acaso para o E1.** Com τ e a alternativa
conhecidos e H0 i.i.d., uma mudança de 0,1 em log-variância com 200 pontos dá KL ≈ 0,5 nat, ou AUC ~0,76.
O que derruba isso na prática é a heteroscedasticidade natural das séries (o nulo de Δlogvar entre
negativos é largo, [premissa 4](../../premissas/04-gerador-das-quebras.md)). É onde um nulo condicional
melhor (volatilidade prevista pelo próprio histórico) poderia render, e é o candidato mais concreto
para o próximo ciclo.

### A troca, medida: E1 + a · especialista de quedas

`s = z_t(E1) + a · z_t(especialista desce)`, com z padronizado dentro de cada passo t (sem cross-fit;
basta para ver a direção):

| a | geral | fixa | sobe | desce |
|---|---|---|---|---|
| 0,0 | **0,6349** | 0,6084 | 0,8345 | 0,5746 |
| 0,1 | 0,6346 | 0,6027 | 0,8316 | 0,6099 |
| 0,2 | 0,6332 | 0,5964 | 0,8275 | 0,6416 |
| 0,3 | 0,6311 | 0,5901 | 0,8220 | 0,6689 |
| 0,5 | 0,6255 | 0,5782 | 0,8079 | 0,7107 |
| 0,8 | 0,6159 | 0,5637 | 0,7826 | 0,7492 |

O ganho em `desce` (+0,17) vem inteiro às custas de `fixa` e `sobe`, e o geral cai de forma monotônica.
O especialista de quedas sobe também os negativos que "parecem quedas", e esses competem com as `fixa`.

### S7: só a evidência extrema do especialista (limiar)

`scripts/s7_limiar.py`: `s = z(E1) + a_d · relu(z(desce) − c_d) + a_s · relu(z(sobe) − c_s)`, com (a, c)
escolhidos por cross-fit de paridade de id numa grade (a ∈ {0; 0,5; 1; 2}, c ∈ {1; …; 3}). A ideia era
deixar os negativos comuns e as `fixa` intocados.

| | geral | fixa | sobe | desce |
|---|---|---|---|---|
| E1 | 0,6349 | 0,6084 | 0,8345 | 0,5746 |
| S7 (cross-fit) | 0,6347 | 0,6080 | 0,8339 | 0,5763 |

No lado ímpar, a grade escolheu a = 0: nenhum limiar ajuda. No par, o ganho no próprio ajuste foi de
+0,0001. **Fechado:** nem a cauda do especialista traz informação que o E1 já não tenha.
