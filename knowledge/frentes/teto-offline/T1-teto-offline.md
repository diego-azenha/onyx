# T1 — Teto offline: com a fronteira conhecida, quanto estes dados deixam separar?

**Frente:** [teto offline](README.md) · **Status:** concluído · **Hipótese:** 2026-09-30 00:05 · **Resultado:** 2026-09-30 00:50

## Por quê

57% do peso da TS-AUC está em pares cujo positivo já tem ≥100 pontos pós-quebra, e ali o Onyx
separa só 0,65–0,70 ([diário 2026-09-29](../../diario/2026-09-29.md), item 16). Se estes dados
permitem ~0,80+ nesse regime, a arquitetura sequencial O(1) é o gargalo. Se não, 0,68 vem de outro
lugar.

## Desenho (versão mínima)

- **Uma linha por série.** Positivo: o trecho `online[τ : τ+L]`. Negativo: `online[p : p+L]`, com
  `p` sorteado da distribuição real de τ (semente fixa). L ∈ {100, 200, 400}; entram só as séries com
  pontos suficientes.
- **Variante oráculo:** o trecho começa exatamente em τ. **Variante contaminada:** o prefixo inteiro
  `online[: τ+L]`, que é o que o tempo real vê no passo t = τ+L.
- **Classificador:** LightGBM sobre ~100 features de duas amostras (histórico vs trecho), calculadas
  em várias representações: valor, |x|, x², diferença, resíduo de AR(5) ajustado no histórico. As
  features cobrem momentos, quantis, KS, Wasserstein, ACF nos lags 1–10, bandas espectrais e contexto
  do histórico. Validação cruzada de 5 folds; métrica AUC.
- **Referência Onyx nas mesmas séries:** o OOF do B0 no passo τ+L (positivos) e p+L (negativos), no
  passo retido mais próximo.

## Hipótese (escrita ANTES de medir)

- Onyx nessa régua: ~0,65 / 0,67 / 0,70 para L = 100 / 200 / 400.
- **Decisão:** se o **oráculo com L=200 der ≥ 0,80**, a informação existe e a arquitetura está
  errada: reconstruir como "classificador offline sobre o prefixo, com busca do ponto de quebra". Se
  der **≤ 0,72**, os dados são difíceis mesmo, e o 0,68 veio de outro lugar.
- Previsão pessoal: oráculo L=200 entre 0,72 e 0,80. O censo de 10 eixos achou ⅔ das quebras
  invisíveis, mas o oráculo de variância do HISTORICO §7 chegava a 0,856 nas quebras de variância.
- A diferença entre oráculo e contaminada mede o custo de não saber τ.

## Como rodar

```bash
python -u scripts/t1_teto_offline.py --n-jobs 4      # ~8 min; artifacts/offline/t1.json + parquets de features
```

## Resultado

### Rodada 1: descartada, amostragem viciada

Os pseudo-τ dos negativos vinham da distribuição **agregada** de τ, que favorece τ pequeno (séries
curtas só contribuem τ pequenos). Os positivos têm τ uniforme dado o comprimento. Com isso o
comprimento do prefixo e o t diferiam entre as classes, e tanto a feature `n` quanto a curva de
taxa-base do Onyx vazavam. O sintoma foi o prefixo contaminado sair **acima** do oráculo (0,673
contra 0,614), o que não pode acontecer sem vazamento. Log em `artifacts/offline/t1_v1.log`.

### Rodada 2: negativos com pseudo-τ ~ U[0, len−L] (a distribuição condicional real)

| L | n séries | Oráculo (τ conhecido, só o trecho pós) | Prefixo contaminado | **Onyx no mesmo ponto** |
|---|---|---|---|---|
| 100 | 7.965 | 0,609 | 0,590 | **0,630** |
| 200 | 6.532 | 0,666 | 0,632 | **0,665** |
| 400 | 4.243 | 0,689 | 0,668 | **0,694** |

**Trecho online pré-quebra contra o histórico** (τ ≥ 50; negativos com pseudo-τ equivalente):
AUC **0,505**. Não há informação antes da quebra.

## Decisão

- **Teto baixo. Pela regra registrada (oráculo L=200 ≤ 0,72), a hipótese "a arquitetura sequencial
  é o gargalo" cai.** Um classificador offline com 118 features de duas amostras, sabendo
  **exatamente** onde a quebra está, empata com o Onyx rodando em tempo real sem saber τ. O Onyx já
  extrai o que essa família de features sabe.
- O custo de não saber τ (oráculo − contaminado) é pequeno: 0,02–0,03.
- A previsão pessoal (0,72–0,80) errou para cima.
- **O que fica em aberto:** o teto *dessa família de features*. Só uma representação **aprendida**
  pode mostrar se há informação fora dos eixos desenhados à mão ([T2](T2-representacao-aprendida.md)).
