# Passo 5 — Simulação de "aprender com o teste" (estado entre séries, 1 processo)

**Frente:** [roteiro 30/09](README.md) · **Status:** concluído: **descartado (até com rótulos perfeitos piora)** · **Hipótese:** 2026-09-30

## Desenho (`scripts/p5_simula_online.py`, só offline)

- **Fluxo de teste:** fold 0 da partição 42 (2.000 séries), com os scores do E1 (OOF com v-EMA),
  processado na ordem dos ids e com 3 embaralhamentos (o organizador exige funcionar com qualquer
  ordem).
- **Rotulador retrospectivo sem τ:** GBM sobre as 118 features de duas amostras do **online inteiro**
  contra o histórico, treinado nos folds 1–4. Entre as séries já concluídas, as 10% de nota mais alta
  viram "quebra" e as 10% mais baixas viram "normais". Nas "quebras", as linhas com t > τ̂ são
  positivas, com τ̂ da varredura de mudança de variância.
- **Correção:** a cada 500 séries concluídas, uma logística determinística sobre logit do aluno,
  log t, 4 colunas do Onyx (`conformal_logm_abs`, `mmd_joint_slow_cal`, `cusum_var_up_r150`,
  `bayes_lo_h0100`) e 2 famílias de surpresa. Com só (logit, t), a correção seria monotônica dentro de
  cada passo e **neutra** pela invariância C1.
- **Métrica:** TS-AUC da **2ª metade** do fluxo, com e sem a correção, grade do board.

## Hipótese (escrita ANTES de medir)

- Δ médio entre **0 e +0,004**. Risco: os pseudo-rótulos (10% extremos, τ̂ estimado) são enviesados
  para quebras fáceis, e a correção pode reaprender o que o aluno já sabe.
- **Regra:** Δ > +0,003 nas 4 ordens → vale implementar em produção (com `INFER_PARALLELISM=1`
  declarado). Δ ≤ +0,001 → descartar. Nada vai para a produção hoje.

## Resultado

`python -u scripts/p5_simula_online.py` (e `--oraculo`). O rotulador retrospectivo sem τ separa as
séries do fluxo com AUC **0,585**.

| Ordem do fluxo | Base (E1) | Corrigido com pseudo-rótulos | Δ | Corrigido com rótulos **verdadeiros** (teto) | Δ teto |
|---|---|---|---|---|---|
| ids | 0,6143 | 0,6090 | −0,0053 | 0,6097 | −0,0046 |
| embaralhado 1 | 0,6081 | 0,5993 | −0,0088 | 0,6015 | −0,0066 |
| embaralhado 2 | 0,6245 | 0,6252 | +0,0007 | 0,6119 | −0,0126 |
| embaralhado 3 | 0,6133 | 0,6163 | +0,0029 | 0,6124 | −0,0010 |
| **média** | | | **−0,0026** | | **−0,0062** |

## Decisão

**Descartado, e pelo teto, não só pelo rotulador fraco.** Mesmo com os rótulos verdadeiros de todas as
séries concluídas, uma camada linear reajustada com ~1.000 séries piora o aluno, que é um GBM treinado
com 8.000. Os eventos do próprio teste somam ~12% à amostra, pouco demais para compensar a perda de
trocar o aluno por uma recombinação linear. O ganho real do estado entre séries teria de vir de
**re-treinar o aluno** com os eventos do teste, e o C1 dá o teto disso: ~+0,002 para +12% de dados, com
rótulos perfeitos, antes do custo dos rótulos retrospectivos, que são ruins (0,585). Não compensa.
