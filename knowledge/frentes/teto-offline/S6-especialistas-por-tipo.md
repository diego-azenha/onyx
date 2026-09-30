# S6 — Especialistas por tipo de quebra (o tipo como supervisão extra)

**Frente:** [teto offline](README.md) · **Status:** hipótese registrada · **Hipótese:** 2026-09-30 14:40

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

(pendente)
