# Frente: estratégias novas para subir o score (01/10, véspera do prazo)

**Aberta em:** 2026-09-30 21:00 · **Pedido do usuário:** parar de medir limites e atacar o score com
estratégias novas, só com modelos nossos (nada pré-treinado de terceiros), produção em CPU de 16
núcleos e 64 GB. Cada ideia vira direto um braço sobre o E1 (OOF 0,6349, partição 42, grade do board).

## Brainstorm

| # | Ideia | Status |
|---|---|---|
| I1 | 2º estágio sobre a trajetória do score do E1 | **nulo** (abaixo) |
| I2 | Modelo de risco (hazard): rótulo "quebra nos últimos K passos" | **nulo** (abaixo) |
| I3 | Nulo condicional aprendido nos ~3M pontos de histórico (GBM nosso, sem rótulo) → surpresa do online | **+0,0024 numa semente**: a pista para depois do prazo |
| I4 | Espelhamento x → −x para dobrar os eventos | reserva |
| I5 | Pacote final por rank-average por passo | não rodado (nada além do I3 ganhou) |
| I8 | Um modelo por faixa de t (≤50, 51–150, 151–400, >400) | **negativo** (−0,0082) |
| I6 | Features padronizadas dentro de cada passo t (mapa congelado por faixa de t) | **negativo** (−0,0021) |

## I1: trajetória do score (`scripts/i1_trajetoria.py`)

Features causais da trajetória do logit do E1 (bag4, sem v-EMA): EWMAs em λ ∈ {0,01; 0,03; 0,1; 0,3},
máximo e mínimo acumulados, média acumulada, fração do tempo acima de limiares, desvios do próprio
passado. 2º estágio cross-fit por paridade de id.

| Combinação | TS-AUC |
|---|---|
| E1 bag4 cru · com v-EMA (incumbente) | 0,6346 · 0,6349 |
| GBM sobre todas as features da trajetória | 0,6161 |
| GBM só com (z, log t) | **0,6305** (perde 0,004 para o próprio z!) |
| Logística com todas, padronizadas por t | 0,6308 |
| z + a·feature, peso ajustado direto na TS-AUC (melhor: máximo acumulado) | 0,6349 |

**Nulo.** A trajetória não traz informação além do score atual: o v-EMA já pega o pouco que há.

**Achado lateral, que motivou o I6:** um GBM com perda agregada **perde ordenação dentro do passo** até
com uma única feature monotônica mais log t, porque gasta cortes nas interações com t. A métrica só
olha a ordem dentro de cada passo.

## I2: modelo de risco, K = 50 (`scripts/i2_hazard.py`)

LightGBM com a receita do E1 no rótulo h_t = 1{τ ≤ t < τ+50}, linhas tardias fora do treino, folds do
Onyx (5 a 70 árvores por fold). Sozinho, a hazard e as agregações dela dão 0,575–0,594. Somada ao E1,
com o peso ajustado direto na TS-AUC por cross-fit de paridade, o melhor caso é 0,6348, contra 0,6349
do E1. **Nulo:** o início da quebra é visto pelas mesmas features, e o E1 já carrega essa informação.

## I3: nulo condicional aprendido (`scripts/i3_nulo_aprendido.py`)

- **O modelo:** um GBM nosso, treinado em 3M posições dos históricos (sem rótulo; 19 entradas:
  defasagens de x e |x|, EWMAs de x², perfil fixo da série), prevê log(x²) do próximo ponto.
  Cross-fit por fold.
- **As features:** a surpresa do online contra esse previsor vira 12 colunas `nz_*` (variância,
  dependência e cauda, nas janelas 25/100/prefixo, mais CUSUMs), padronizadas pela cauda do histórico.
- **O custo:** a construção leva 4 min.

Braço com a receita do E1, semente 777, pareado com o E1 da mesma semente:

| | geral | fixa | sobe | desce |
|---|---|---|---|---|
| E1 s777 | 0,6303 | 0,6034 | 0,8241 | 0,5793 |
| I3 s777 | **0,6327 (+0,0024)** | 0,6055 | 0,8213 | 0,5884 |

**Promissor, mas não confirmado:** uma semente e +0,0024, abaixo da barra de +0,003. Ganha nos tipos
difíceis (`fixa` e `desce`). Não entra na submissão desta noite: precisa de um bloco de streaming novo e
do modelo do nulo dentro do `train()`, com verificação bit a bit. Próximo passo: K=4 e R0.

## I6: features padronizadas por passo (`scripts/i6_padroniza_por_t.py`)

Semente 777, pareado: 0,6282 contra 0,6303 do E1 (**−0,0021**); `fixa` 0,6029, `sobe` 0,8180, `desce`
0,5754. **Descartado.** O GBM com a curva de taxa-base já lida com a escala por t. A perda do I1 vinha
do 2º estágio, não da escala das features do E1.

## I8: um modelo por faixa de t (`scripts/i8_faixas.py`)

Semente 777, as 4 faixas concatenadas: 0,6221 contra 0,6303 do E1 (**−0,0082**). **Descartado.** Dividir
os dados custa mais do que separar as escalas ganha. É coerente com o C1: o aprendiz é limitado por
amostra, e cada faixa vê menos eventos.

## Decisão para a submissão de 30/09 → 01/10

**Submeter o E1** (`submission_notebook.ipynb`, verificado bit a bit). Das cinco estratégias novas
testadas hoje, só o I3 (nulo aprendido) foi positivo (+0,0024 numa semente). Ele não cabe no prazo: precisa
de um bloco de streaming novo e de verificação. É a frente para depois do prazo: K=4, R0 e integração.
