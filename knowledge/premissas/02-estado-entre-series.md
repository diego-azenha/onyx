# Premissa 2 — "Cada série é julgada sozinha"

**Status:** fechada (2026-09-30): sem ganho simulado e desaconselhado pelo organizador · **Prioridade:** 2ª, ou 1ª se a frente de surpresa for descartada

## A afirmação questionada

O placar compara séries entre si a cada instante. Um score de 0,3 numa série calma e um de 0,3 numa
série muito ruidosa querem dizer coisas diferentes, e um modelo que vê uma série de cada vez não
tem como saber disso. Um **estado compartilhado entre séries**, que os organizadores teriam
admitido ser possível com processamento sequencial, ataca isso diretamente.

## O que o repositório já sabe, e a contradição

- [HISTORICO §11](../historico/HISTORICO.md), "O que NÃO fazer": *"Features transversais (rank da
  feature entre séries vivas no mesmo t) — a API entrega uma série por vez; é estruturalmente
  impossível e seria vazamento."*
- **Fonte (verificada em 2026-09-29):** [fórum, "may infer use prior completed-series state?"](https://forum.crunchdao.com/t/structural-break-real-time-may-infer-use-prior-completed-series-state/1186).
  Pergunta: `infer()` pode manter um resumo das séries já processadas e usá-lo nas seguintes? Resposta
  do organizador: *"Yes, that's totally doable"*, **com a ressalva** de que isso provavelmente torna o
  código não determinístico. A checagem de determinismo re-executa 10% dos dados começando em pontos
  diferentes (paralelismo), com tolerância de 1e-8, e código não determinístico **perde a elegibilidade
  a prêmios**. Na prática, é permitido, mas inviável para quem quer prêmio.
- A versão legal da ideia não precisa de estado em tempo real: a comparabilidade entre séries pode ser
  aprendida **no treino**, sobre a mesma distribuição. É o que a calibração F1 e o próprio GBM já
  fazem.
- A mesma página de docs diz que o treino **mistura séries sintéticas e reais** e que cada série é
  padronizada só com o próprio histórico.
- O contrato do adaptador ([NOTAS §2.3](../operacao/NOTAS_AGENTES.md)) mostra que `infer` é um
  generator que recebe as séries **em sequência**, no mesmo processo. O estado do processo persiste
  entre séries, mas uma série só é vista depois de a anterior ter sido **inteiramente** pontuada. O
  estado compartilhado pode então transportar informação *entre séries passadas* (por exemplo, a
  distribuição de scores já emitidos em cada `t`), mas não permite ranquear séries vivas
  simultaneamente.
- A calibração por série (F1, `state/calibration.py`) já ataca parte do problema pelo outro lado:
  torna o score de cada série comparável sem ver as outras.

## CORREÇÃO (2026-09-30): com `INFER_PARALLELISM=1`, o estado entre séries é determinístico

A conclusão acima ("permitido, mas inviável para quem quer prêmio") **estava errada** para o caso de
um só processo. O código oficial do avaliador
([`runner.py`](https://raw.githubusercontent.com/crunchdao/competitions/master/competitions/structural-break-real-time/scoring/runner.py),
lido em 30/09) mostra:

- a checagem de determinismo re-executa **as primeiras N séries**, na mesma ordem:
  `determinism_slice = slice(None, int(len(datasets) * determinism_check))`;
- cada worker percorre a sua fatia em ordem (`for dataset in datasets[start_index:end_index]`);
- a comparação é `numpy.allclose(..., atol=tolerance)`.

Com `INFER_PARALLELISM=1`, as duas execuções percorrem as mesmas séries na mesma ordem, o estado
acumulado é idêntico e as previsões batem. A explicação do organizador ("começar em pontos
diferentes") descreve o caso com **vários** processos.

**Ressalvas que continuam valendo:**
1. o organizador pediu que o código funcione igual com ordem fixa ou aleatória;
2. a avaliação fora da amostra roda **uma vez**, sobre dados novos; o placar público não é o final;
3. usar posição na fila, ids ou mudar o comportamento depois dos primeiros 10% seria trapaça e está
   fora de questão.

O uso legítimo, "aprender com o teste" (rotular em retrospecto as séries já concluídas e reajustar uma
correção pequena), é simulado no [passo 5 do roteiro de 30/09](../frentes/roteiro-30-09/README.md).

## ATUALIZAÇÃO (2026-09-30, tarde): a palavra final do organizador é contra

Relido o fio do fórum até o fim: depois da pergunta sobre a re-execução de 10%, o organizador conclui que
*"trying to persist state across time series will likely result in your code not being deterministic
based on when it starts"*. O `INFER_PARALLELISM` da nuvem não está sob nosso controle. Como o passo 5
(simulação de aprender com o teste) também não rendeu nada, **estado entre séries fica fora do projeto**.

## Perguntas a responder antes de construir

1. Qual é a fonte exata da afirmação dos organizadores, e o que ela permite?
2. Na ordem em que o runner entrega as séries, que informação de séries anteriores pode ser usada
   sem violar a causalidade *dentro* de cada série?
3. A normalização por "scores já vistos no passo t" muda a ordenação dentro do passo? Se for uma
   transformação monótona comum a todas as séries, pela invariância C1 ela é **neutra**. Só ajuda
   se depender da série (por exemplo, um quantil condicional ao perfil do histórico).

## Experimentos

Nenhum ainda.
