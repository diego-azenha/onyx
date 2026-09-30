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
