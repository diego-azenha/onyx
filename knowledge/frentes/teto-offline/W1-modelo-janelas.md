# W1 — Segundo modelo, de construção diferente, para ensemble com o E1

**Frente:** [teto offline](README.md) · **Status:** concluído: **nulo** · **Hipótese:** 2026-09-30 13:20

## Por quê

Nada empilha sobre o E1 como coluna ou regularização. O Exp0 ("forma B") dizia que um ensemble só paga
com um membro **forte** (~0,62+) e **de construção diferente**. O T1 mostrou que as 118 features de duas
amostras empatam com o Onyx ponto a ponto; construídas de outro jeito, podem errar em lugares diferentes.

## Desenho (`scripts/w1_janelas.py`)

- Features de `src/offline/features.py` calculadas **causalmente** em t ∈ {10, 20, 30, 45, 65, 90, 120,
  160, 210, 280, 360, 460, 580, 720, 880, 999} sobre três janelas do online (prefixo inteiro, últimas 50,
  últimas 200), contra o histórico. Forward-fill para as linhas do `train_rows`, como o tempo real
  recalculando a cada k passos.
- GBM (extra-trees, pesos pareados por t), folds do Onyx (partição 42), OOF por linha.
- Blend cross-fit com o E1 (logits, peso de uma grade 0–0,5 ajustado nos ids pares e aplicado nos ímpares).

## Hipótese (escrita ANTES de medir)

- W1 sozinho: **0,60–0,62**. Blend: **+0,000 a +0,005** contra o E1.
- Adotar a direção (construir o caminho de produção) só se o blend der ≥ +0,003 e o R0 excluir 0.

## Resultado

W1 sozinho: **0,5921** (abaixo da faixa prevista de 0,60–0,62). Blend cross-fit com o E1 (pesos 0,1 e
0,2): **0,6349 → 0,6353 (+0,0003)**.

## Decisão

**Nulo.** As features de duas amostras em janelas causais ficam ~0,04 abaixo do Onyx como modelo, e o
que elas têm de diferente não soma. Junto com a GRU (0,575) e o bag entre partições (+0,0007 com 6
modelos), isso fecha a linha de "ensemble com membro de construção diferente".
