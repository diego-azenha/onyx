# Premissa 4 — "Os dados são séries financeiras genéricas"

**Status:** parcialmente investigada · **Papel:** apoio à premissa 1 (diz o que a surpresa deve vigiar)

## A afirmação questionada

Nunca se investigou *como* as quebras foram fabricadas. A correção das 29 séries mal rotuladas
sugere que os rótulos vêm de um processo controlado pelos organizadores. Se as quebras saem de um
cardápio limitado de tipos e tamanhos, conhecer esse cardápio diz exatamente o que vigiar e permite
gerar dados de treino realistas.

## O que o repositório já sabe

- **Censo A1** (`scripts/break_type_census.py`, [HISTORICO §4, R6](../historico/HISTORICO.md)),
  sobre 4.552 séries com quebra: a mediana de |Δmean_e| é 0,0033 e só 6,8% têm |Δmean_e| > 0,3,
  então **o canal de média é quase morto**. Já 41,8% têm |Δlogvar_e| > 0,3, o que faz de
  **variância e cauda o sinal dominante**.
- **Eixos independentes que existem no gerador** ([HISTORICO §7](../historico/HISTORICO.md)):
  variância (β +0,312), cauda/forma (+0,052), dependência (+0,043). Há 437 quebras de dependência
  pura e 357 de cauda pura.
- **Rótulos:** o snapshot local é pós-W23. As 29 séries corrigidas têm `tau_index = 0`
  ([HISTORICO §15.5](../historico/HISTORICO.md)). Em 2026-09-29, a cópia de `Downloads/data` (de
  2026-06-23) e a de `Documents/CrunchDAO/SBC/data` (de 2026-09-28) deram **idênticas** em
  `X_train` e `y_train_index`: 4.967 séries com quebra em 10.000.
- O censo mede **magnitudes**, mas nunca procurou uma **estrutura discreta**: níveis de magnitude
  repetidos, famílias paramétricas, relação entre τ e o comprimento da série.

## Medido em 2026-09-29

**Censo sem modelo** (`scripts/premissa4_censo_eixos.py`). Compara o trecho pós-quebra (até 300
pontos, mínimo 100) com o histórico em 10 eixos. O nulo são 3.318 negativos com um pseudo-τ sorteado
da distribuição real de τ.

| Fração fora da faixa de 95% do nulo | `var_fixa` (\|Δlogvar\| ≤ 0,3, n=2.354) | `var_muda` (n=1.057) |
|---|---|---|
| Δ log variância | 0,000 (por definição) | 0,272 |
| Δ log variância das diferenças | 0,006 | 0,289 |
| Δ média / dp | 0,042 | 0,093 |
| Δ ACF lag 1 · 2 · 5 · 10 | 0,071 · 0,093 · 0,085 · 0,086 | 0,131 · 0,132 · 0,071 · 0,079 |
| Δ ACF de \|x\| lag 1 | 0,051 | 0,114 |
| Δ curtose · assimetria | 0,035 · 0,032 | 0,156 · 0,140 |

- **69% das quebras `var_fixa` não saem do nulo em nenhum eixo.** Com eixos independentes, o
  acaso daria ~60%, então elas parecem tão nulas quanto os negativos. O único excesso fica na
  dependência de curto prazo (ACF lag 1–2: 7–9% contra 5% esperado).
- **O nulo de Δlogvar é largo**: mesmo entre as quebras com |Δlogvar| > 0,3, só 27% saem da faixa de
  95% dos negativos. Os históricos variam muito de variância por conta própria (não
  estacionariedade, cauda pesada), e isso limita o que qualquer detector de variância consegue.
- Com o [S1](../frentes/surpresa-acumulada/S1-detector-sozinho.md), a leitura é: **cerca de ⅔ das
  quebras são indetectáveis nesses eixos com até 300 pontos**. Se 0,68 existe, ou o gerador muda algo
  fora desses 10 eixos (dependência em lags maiores de forma não linear, sazonalidade, estrutura
  espectral fina), ou o ganho vem de ordenar melhor o terço detectável e os passos em que ele
  acontece.

## Checagens estruturais (2026-09-29): nenhum atalho

| Pergunta | Resultado |
|---|---|
| O comprimento do histórico, ou o início, carrega o rótulo? | AUC 0,495 · 0,500 |
| A padronização vaza? | o histórico tem média 0 e dp 1 exatos (padronizado só nele mesmo, como diz a doc) |
| τ tem estrutura? | uniforme dentro do online (τ/n_on: mediana 0,48) |
| Trechos se repetem entre séries (janelas de 12 valores exatos)? | **0** no treino; **0** das 100 séries do teste reduzido aparecem no treino |
| Valores discretos (tick de preço real)? | não: todas as séries têm >97% de valores únicos |
| O [changelog W23](https://forum.crunchdao.com/t/2026-w23-structural-break-fixes/1156) fechou um vazamento de série completa na nuvem | em 8/jun, antes do salto de 0,652 → 0,68 (que foi nas últimas 3 semanas), então esse vazamento não explica o salto |

**Heterogeneidade entre séries.** TS-AUC do Onyx (B0) dentro de cada grupo: séries muito
persistentes (ACF lag 1 > 0,7, n=322) dão **0,950**; ACF < −0,3 dá 0,661; o miolo (|ACF| < 0,3) fica
em 0,614–0,618; curtose 0,3–2 dá **0,593**. A dificuldade está concentrada nas séries de ruído
quase branco com cauda moderada, que são a maioria.

## Cardápio e dados reais compartilhados (2026-09-30)

- **Não há cardápio discreto.** Em séries com ≥ 400 pontos pós-quebra (1.207 quebras, 3.036 negativos
  com online ≥ 400), os histogramas finos de meia log-razão de variância, Δ ACF lag 1 e Δ média são
  contínuos e unimodais, só um pouco mais largos que os dos negativos. A maior parte das quebras
  fica abaixo do ruído.
- **As séries reais não compartilham dados.** Janelas de 16 valores normalizadas pela própria média e
  dp (invariante a padronização) não coincidem entre séries em nenhum deslocamento: 10,3M janelas de
  3.000 séries indexadas e as outras 7.000 consultadas com passo 4, zero coincidências. Não existe
  um pool finito de ativos cortado em janelas sobrepostas.
- **Dados de 2025:** permitidos pela doc (o HISTORICO dizia o contrário), mas o cardápio mudou
  entre as edições ([D25](../frentes/teto-offline/D25-dados-2025.md)).

## Perguntas em aberto

1. As magnitudes medidas (Δlogvar, Δρ₁, Δkurt) se agrupam em poucos valores? Um histograma fino
   responde.
2. A posição de τ tem distribuição reconhecível (uniforme no online? mínimo de pontos antes e
   depois)?
3. O histórico parece gerado (AR/GARCH de baixa ordem) ou real (retornos financeiros)?

## Experimentos

Nenhum ainda. Candidato barato: rodar `break_type_census.py` neste clone (precisa só de `fit_h0`) e
histogramar os eixos.
