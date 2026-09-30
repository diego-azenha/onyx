# C1 — O Onyx é limitado por amostra?

**Frente:** [teto offline](README.md) · **Status:** concluído · **Hipótese:** 2026-09-30 00:35 · **Resultado:** 00:45

## Desenho

A receita do [B0](../surpresa-acumulada/B0-incumbente.md) (sementes 777 e 101), treinada só nas
**5.000 séries de id par** (`scripts/run_c1_curva_onyx.sh`), com a mesma CV de 5 folds por série. É
comparada **nas mesmas 5.000 séries** com o bag das mesmas sementes do B0, treinado nas 10.000. Cada
fold vê 4.000 séries contra 8.000: é um ponto da curva de aprendizado (×2 de dados).

## Hipótese (escrita ANTES de medir)

- Se **Δ(10k − 5k) ≥ +0,010** na grade do board, o Onyx está limitado por amostra, e
  gerar dados realistas (×4–×10) é o caminho mais provável para o salto: a curva extrapolada daria
  +0,02 a +0,04.
- Se **Δ ≤ +0,004**, o Onyx está perto do platô de amostra, e mais dados não explicam 0,68.
- Previsão: +0,006 a +0,012.

## Resultado

Mesmas 5.000 séries (id par), bag das sementes 777 e 101, grade do board:

| Treino | TS-AUC |
|---|---|
| 5.000 séries (4.000 por fold) | 0,6129 |
| 10.000 séries (8.000 por fold) | **0,6273** |
| **Δ (×2 de dados)** | **+0,0144** (semente 777: +0,0133 · 101: +0,0171) |

## Decisão

**O Onyx é fortemente limitado por amostra.** Pela regra registrada (≥ +0,010), gerar mais dados de
treino realistas passa a ser o caminho principal. Extrapolando log-linear: ×4 dá ~+0,025 e ×8 dá
~+0,036, o que leva de ~0,63 para 0,66–0,67. É também uma explicação coerente para um salto tardio e
não documentado: quem descobre como gerar dados bons dá um salto sem mudar de modelo.

Este é o achado mais importante da frente, e ele reorganiza a pergunta: não é "que feature falta",
é "**quantos eventos de quebra o modelo vê**". O próximo experimento é o
[A1, re-corte de séries reais](A1-recorte.md).
