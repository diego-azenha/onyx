# Frente: rede sequencial de ponta a ponta (GRU)

**Aberta em:** 2026-09-30 13:00 · **Por quê:** é a única classe de modelo listada pelos organizadores que
o projeto nunca mediu ("deep learning treinado em pares série-passo"). Mesmo sem bater o E1, uma classe
diferente é a forma forte de ensemble que o Exp0 dizia faltar.

## Desenho (`scripts/r2_gru.py`, venv isolado com PyTorch CPU)

- Uma GRU (64) codifica os últimos 512 pontos do histórico (x, |x|, x²) e inicializa, via camada linear,
  uma GRU (96) que lê o online passo a passo (x, |x|, x², resíduo AR(5) do histórico, log t/7).
  Uma cabeça MLP dá o logit de P(τ ≤ t) em cada passo.
- BCE com pesos pareados por passo (como o R1 do Onyx), Adam + OneCycle, lotes de 32 séries agrupadas
  por comprimento. Época escolhida por validação interna de 10% das séries de treino.
- Folds da partição 42 (`artifacts/rnn/dados.npz`).

## Hipótese (escrita ANTES de medir)

GRU sozinha **0,57–0,61**. Só vale rodar os 5 folds se o fold 0 passar de **0,60** com 25 épocas.
Nesse caso, o blend com o E1 viria em seguida.

## Resultado

Teste curto (fold 0, 3 épocas, ~90 s por época): TS-AUC interna 0,540 → 0,554; no fold 0, **0,533**.
Rodada de 25 épocas (fold 0, ~150 s por época dividindo a CPU): TS-AUC interna 0,540 (ép. 0) →
0,579 (ép. 8) → pico de **0,587** (ép. 14) → 0,582 (ép. 24). Melhor estado guardado.

| Fold 0 (grade do board) | TS-AUC |
|---|---|
| E1 | 0,6302 |
| **GRU sozinha** | **0,5747** |
| blend E1 + GRU (peso cruzado por paridade: 0 / 0,1) | 0,6286 |

## Decisão

**Frente fechada.** A GRU fica abaixo da faixa prevista (0,57–0,61, e a barra para os 5 folds era
0,60) e **piora o blend**. Com 8 mil séries e CPU, a rede ponta a ponta não aprende o que as features à
mão já codificam, e o que ela aprende de diferente não ajuda a ordenação. Coerente com o C1: o
problema é limitado por amostra de eventos, e um modelo com mais parâmetros sofre mais com isso.
Código mantido (`scripts/r2_gru.py`); o venv com PyTorch fica no scratchpad.
