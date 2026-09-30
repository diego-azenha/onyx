# T3 — Embeddings de um modelo de fundação (Chronos-Bolt-tiny) no oráculo

**Frente:** [teto offline](README.md) · **Status:** concluído: **negativo** · **Data:** 2026-09-30 02:20

## Por quê

O Onyx é limitado por amostra ([C1](C1-curva-aprendizado-onyx.md)), e nenhuma fonte de eventos extras
serviu. Um prior forte vindo de fora, como um modelo de fundação pré-treinado em corpora enormes, é a
forma clássica de ganhar eficiência de amostra. É a proposta P2 do
[DIAGNOSTICO_ESTRUTURAL](../../modelo/DIAGNOSTICO_ESTRUTURAL.md), nunca testada. A 2ª colocação de 2025
usou TabPFN, que na edição de 2026 é **proibido** por licença. O Chronos é Apache 2.0.

## Desenho

- Ambiente **isolado** (virtualenv no scratchpad, torch 2.14 CPU + chronos-forecasting), para não
  tocar o numpy e o lightgbm que o Onyx fixa por determinismo.
- `amazon/chronos-bolt-tiny`, `pipeline.embed`: 64 janelas em 23 ms na CPU.
- Mesma amostra do oráculo com L=200 do T1. Janelas de 128 pontos: 7–8 do fim do histórico e a
  janela final do trecho pós-quebra. Features: distância padronizada pelo desvio entre as janelas
  do histórico, cosseno, distância do token final e 16 PCs da diferença (PCA ajustado dentro de
  cada fold).

## Resultado (AUC, CV de 5 folds, GBM com extra-trees)

| Features | AUC |
|---|---|
| T1 (118 à mão) | **0,6718** |
| só distâncias de embedding | 0,5293 |
| distâncias + 16 PCs | 0,5163 |
| T1 + distâncias | 0,6711 |
| T1 + distâncias + PCs | 0,6675 |

## Decisão

**Descartado.** O Chronos normaliza cada janela (média e escala), então mede só forma e dinâmica, e
nesse espaço as quebras destes dados são quase invisíveis (0,53). Isso fecha com o censo e com o
ROCKET: o que separa as quebras é escala e cauda, e isso as features à mão já cobrem.
