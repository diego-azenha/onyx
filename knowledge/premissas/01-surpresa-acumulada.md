# Premissa 1 — "A quebra aparece em resumos da série"

**Status:** testada em 2026-09-29, **nula para o placar** · **Frente:** [surpresa acumulada](../frentes/surpresa-acumulada/README.md)

## A afirmação questionada

As 183+ features do Onyx comparam **resumos**, como média, variância ou forma numa janela recente
contra o histórico. A alternativa: usar os 1.000 a 5.000 pontos do histórico para aprender em
detalhe como *aquela* série se comporta, perguntar a cada ponto novo "quão surpreendente é este
ponto para esta série?" e **acumular** a evidência.

Em quebras pequenas a diferença é grande. Com σ indo de 1,0 para 1,2, o desvio medido numa janela de
20 pontos erra cerca de 16% (1/√38), então a janela quase não distingue a quebra de ruído. Já a
razão de verossimilhança acumulada contra a alternativa certa rende KL ≈ 0,038 nat por ponto, e em
100 pontos chega a ≈ 3,8 nats ≈ **44:1**.

**Ressalva da conta:** o 44:1 vale para a razão de verossimilhança **direcional** (contra
"σ subiu"). Somar só a surpresa (−log p₀) dá z ≈ 2,2 em 100 pontos, porque a surpresa ingênua não
sabe para que lado olhar. Por isso o detector da frente acumula razões direcionais e usa a surpresa
ingênua como controle.

## O que o repositório já sabe

- **O gargalo é extração, não informação** ([HISTORICO §7](../historico/HISTORICO.md)): um detector
  ótimo de variância que conhece τ chega a AUC ≈ 0,856 contra a mistura real do gerador (0,890 com
  mais de 200 pontos pós-quebra); o V3 estava em 0,604. A causa apontada é a **diluição por τ
  desconhecido**: janelas fixas misturam pontos pré e pós-quebra. Um CUSUM/GLR de verossimilhança
  usa, por construção, só os pontos depois do início mais provável da mudança.
- **O Onyx já tem acumuladores de surpresa**, mas por eixo e comprimidos por árvore:
  martingales conformais (`state/conformal.py`; `conformal_logm_abs` é a **feature nº 1**), filtro
  bayesiano de troca única gaussiana (`state/bayes_filter.py`, "mal especificado para cauda pesada e
  dependência", §7) e banco de CUSUM (`state/cusum.py`).
- **Onde o banco é cego** (§7): quebras de dependência pura (detect 0,492) e de cauda pura (0,553).
  P1–P4 do V4 atacaram isso por janelas.
- O vencedor de 2025 chegou a 0,90 comparando o histórico com todo o trecho pós-quebra. A diferença
  é detectável com dados suficientes; a fraqueza é a **velocidade**.

## Hipótese testável (a versão estreita)

Um modelo condicional por série (AR + volatilidade + forma empírica da inovação) que leva cada ponto
a um fluxo **universal** `g_t ~ N(0,1)` sob H0, com acumulação direcional de razão de
verossimilhança e agregado num único score calibrado, carrega ordenação transversal que o banco
comprimido por árvore não carrega.

## Veredito

- **Verdadeiro no sintético:** a acumulação detecta σ×1,2 em 38 passos, contra 66 da janela de 20
  pontos ([S0](../frentes/surpresa-acumulada/S0-sanidade-sintetica.md)).
- **Sem efeito no placar:** sozinha dá 0,575, no nível das melhores colunas do Onyx. Somada ao OOF,
  +0,0001 [−0,0003; +0,0004] ([S3](../frentes/surpresa-acumulada/S3-combinacao.md)).
- **O motivo mais importante:** o gargalo não é *como* se lê a série (resumo ou acumulação), é
  *o que* muda nela. ~⅔ das quebras não mudam variância, média, dependência de curto prazo nem
  cauda de forma distinguível dos negativos ([premissa 4](04-gerador-das-quebras.md)).

## Experimentos

[S0](../frentes/surpresa-acumulada/S0-sanidade-sintetica.md) ·
[S1](../frentes/surpresa-acumulada/S1-detector-sozinho.md) · S2 · S3 · S4 (ver o
[README da frente](../frentes/surpresa-acumulada/README.md)).
