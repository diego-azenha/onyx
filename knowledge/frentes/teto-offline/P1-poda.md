# P1 — Poda para as 65 features de maior ganho (redução de variância)

**Frente:** [teto offline](README.md) · **Status:** hipótese registrada, na fila · **Hipótese:** 2026-09-30 01:50

## Por quê

O Onyx é limitado por amostra ([C1](C1-curva-aprendizado-onyx.md)). Nos 20 modelos de fold do B0
(`artifacts/reports/importancia_b0_gain.csv`), a importância por ganho é concentrada: **49 features
explicam 90% do ganho, 65 explicam 95% e 104 explicam 99%**, de 193. As ~130 restantes pouco
contribuem e somam variância no sorteio de `feature_fraction`, efeito já medido duas vezes
([NOTAS §5.1](../../operacao/NOTAS_AGENTES.md)). A 1ª é `conformal_logm_abs`, a 2ª é `mmd_joint_slow_cal`.

## Hipótese (escrita ANTES de medir)

- Δ entre **−0,002 e +0,004** contra o B0.
- **Viés de seleção a declarar:** a lista foi escolhida com modelos que viram todos os folds, então
  qualquer ganho é levemente otimista. Se o P1 ganhar, a réplica na partição 43 é obrigatória.
- Regra: adotar só com IC excluindo 0 e ≥ 3 de 4 sementes positivas no pareado.

## Resultado

(pendente)
