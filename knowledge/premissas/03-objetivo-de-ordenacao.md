# Premissa 3 — "O modelo aprende probabilidades, mas o placar premia a ordem"

**Status:** testada uma vez numa régua antiga; candidata a reauditoria

## A afirmação questionada

No instante 50, com 3 séries quebradas entre 100, o modelo reduz bem o erro de treino chutando 3%
para todas, e isso não vale nada no placar. O que vale é colocar as 3 certas no topo. Treinar para
**ordenar séries dentro de cada instante** deveria alinhar o objetivo com a métrica.

## O que o repositório já sabe

- **Foi testado, na rodada 3 (R3)** ([HISTORICO §4](../historico/HISTORICO.md)): `train_rank` com
  lambdarank agrupado por passo. O modelo de rank sozinho ficou em **0,5852**, pior que o binário.
  Combinado, ficou em 0,5993, com Δ **+0,0010 [−0,0035; +0,0058]**, indistinguível de zero.
  Conclusão da rodada: *"as três alavancas de como o modelo consome (peso, parada, objetivo) são
  estatisticamente nulas"*.
- **Por que o resultado não é definitivo:**
  - foi medido **antes** do X1, do B6 e do D1, e na **grade thin**, que subpondera `t>400` por ~2×
    ([HISTORICO §15.1](../historico/HISTORICO.md));
  - usou uma semente só, e a régua tem dp de 0,0041 que o bootstrap não enxerga
    ([NOTAS §5](../operacao/NOTAS_AGENTES.md));
  - o `truncation_level` teve de ser limitado (cap 300) para o treino terminar, e isso muda o
    objetivo;
  - o pairwise de pesos (R1) e o `--pairwise-alpha` são aproximações do mesmo objetivo, também nulas.
- **O código está quebrado desde 2026-07-24** ([HISTORICO §15.6](../historico/HISTORICO.md)):
  `train_rank` levanta `UnboundLocalError` desde a adoção do B6. O stacking A2 com membro rank não
  rodou.
- A invariância C1 e o `init_score` com a taxa-base já removem a parte "chutar a taxa-base" do
  problema: o modelo aprende o **resíduo** sobre `f(t)`
  ([HISTORICO §15.4](../historico/HISTORICO.md)).

## Próximo passo, se reaberta

Consertar `train_rank`, medir com K=4 na grade do board, nas duas partições, como membro de stacking
sobre o incumbente.

## Experimentos

Nenhum nesta fase.
