# T2 — Uma representação aprendida acha informação fora dos eixos desenhados à mão?

**Frente:** [teto offline](README.md) · **Status:** concluído · **Hipótese:** 2026-09-30 00:45 · **Resultado:** 2026-09-30

## Hipótese (escrita ANTES de medir)

- **Representação:** 1.008 features de convolução aleatória no estilo MiniRocket
  (`src/offline/rocket.py`): 84 kernels × 4 dilatações × 3 vieses nos quantis do histórico. Cada
  feature é o desvio da fração de valores positivos no trecho contra o histórico, então não
  depende de eixo escolhido. Sem deep learning (não há PyTorch instalado).
- **Mesmas amostras e mesma CV do oráculo do [T1](T1-teto-offline.md) com L=200** (0,666 com as
  features à mão).
- **Decisão:** se ROCKET sozinho ou ROCKET + T1 passar de **0,70**, há informação fora dos eixos à mão,
  e o próximo passo é levar essa representação para o tempo real. Se ficar em ~0,67, a informação
  disponível nos dados é, na prática, o que o Onyx já usa.
- Previsão: 0,66–0,69.

## Resultado

`python -u scripts/t2_rocket_oraculo.py --n-jobs 4`, oráculo com L=200, 6.532 séries, CV de 5 folds:

| Features | AUC |
|---|---|
| 118 features à mão (T1) | **0,666** |
| 1.008 ROCKET | 0,600 |
| T1 + ROCKET | 0,654 |

**Não há informação nova nessa representação.** Somar o ROCKET piora o resultado, porque 1.008
colunas ruidosas diluem 6,5k exemplos.

### Complemento 1: curva de aprendizado do oráculo (mesmas features do T1)

Teste fixo (1 de 5 folds), treino com uma fração do restante:

| Fração do treino | 12,5% | 25% | 50% | 100% |
|---|---|---|---|---|
| L=100 | 0,582 | 0,587 | 0,601 | 0,610 |
| L=200 | 0,614 | 0,628 | 0,651 | 0,663 |

**Sem platô: +0,012 a +0,023 a cada vez que os dados dobram.** O teto de ~0,67 do T1 é, pelo menos
em parte, **limite de amostra**, e o teste com o ROCKET não o distingue de limite de informação.

### Complemento 2: complementaridade com o Onyx (L=200, prefixo contaminado, realista)

Onyx sozinho no mesmo ponto: 0,665 · features contaminadas + score do Onyx como feature: 0,654.
Nas mesmas séries, as features offline não somam ao Onyx.

## Decisão

A pergunta passa a ser se **o próprio Onyx é limitado por amostra** ([C1](C1-curva-aprendizado-onyx.md)).
Se for, o caminho para o salto é **mais dados de treino realistas**: aumento por bootstrap de
eventos reais de quebra, ou reconstrução do gerador ([premissa 4](../../premissas/04-gerador-das-quebras.md)).
