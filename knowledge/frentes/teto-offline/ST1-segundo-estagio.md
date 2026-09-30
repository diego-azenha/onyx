# ST1 — Segundo estágio pequeno (Onyx + famílias de evidência)

**Frente:** [teto offline](README.md) · **Status:** concluído: **negativo** · **Data:** 2026-09-30 01:45

## Motivação: um sinal forte que o GBM não usa

TS-AUC com positivos de **um tipo** contra todos os negativos (grade do board):

| Tipo de quebra | Onyx (B0) | S5a (Onyx + famílias como feature) | `fam_escala_desce` sozinha | `fam_escala_sobe` sozinha |
|---|---|---|---|---|
| queda de variância (`desce`) | 0,564 | 0,574 | **0,774** | 0,321 |
| alta de variância (`sobe`) | 0,834 | 0,831 | 0,246 | **0,857** |
| `var_fixa` | 0,602 | 0,607 | 0,441 | 0,578 |
| curta (<30 pontos pós) | 0,542 | 0,554 | 0,463 | 0,543 |

Uma única coluna separa as quedas a 0,77, e o Onyx, mesmo recebendo a coluna, fica em 0,57. A
hipótese foi que ~700 quedas diluídas em 190 colunas são pouca amostra para o GBM aprender a regra,
e que um estágio com ~10 colunas aprenderia.

## Desenho

`scripts/st1_stacker.py`: GBM raso (15 folhas, `min_data_in_leaf=2000`, λ₂=10) sobre logit(OOF do
Onyx) + 8 colunas de evidência + t. Mesma partição de folds do Onyx, pesos pareados por passo e o
v-EMA do B3 por cima.

## Resultado

| | TS-AUC geral | `desce` | `sobe` | `var_fixa` | curta |
|---|---|---|---|---|---|
| B0 (com v-EMA) | **0,6278** | 0,564 | 0,834 | 0,602 | 0,542 |
| ST1 (com v-EMA) | 0,6267 | 0,579 | 0,822 | 0,599 | 0,536 |

## Leitura, e por que é estrutural

O segundo estágio também não aproveita as quedas. **Evidência forte de queda é comum entre os
negativos**: trechos calmos de séries heteroscedásticas deixam `escala_desce` no percentil 0,51 dos
negativos. Subir o score das quedas sobe junto esses negativos, que passam à frente das quebras
`var_fixa` (67% dos positivos). A posterior ótima já pondera isso, e o ganho líquido some. A AUC de
0,77 **por subconjunto** não se converte em ordenação **global**.
