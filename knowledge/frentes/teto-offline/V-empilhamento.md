# V1–V8 — Empilhar redução de variância sobre o E1

**Frente:** [teto offline](README.md) · **Status:** concluído: **nenhum braço empilha sobre o E1** · **Início:** 2026-09-30 02:50 · **Fim:** 10:30

## Hipótese (escrita ANTES de medir)

O [E1](E1-extra-trees.md) mostrou que reduzir a variância do aprendiz paga (+0,0071). Cada braço
adiciona uma alavanca sobre o E1 (receita do B0 + `extra_trees`), com K=4, v-EMA e R0 contra o
**E1** (0,6349) e contra o B0 (0,6278). `scripts/run_braco.sh`, filas `_fila_v*.sh`.

| Braço | Alavanca | Previsão contra o E1 |
|---|---|---|
| V1 | `feature_fraction` 0,8 → 0,5 | 0 a +0,002 |
| V2 | N1: metade dos negativos fora do treino | 0 a +0,003 |
| V3 | receita histórica: sem `mrep_`, `feature_contri_meta` 0,6 | 0 a +0,003 |
| V4 | poda para as top-65 features por ganho do B0 | −0,002 a +0,003 |
| V5 | `bagging_fraction` 0,8 → 0,5 | 0 a +0,002 |
| V6 | sem `linear_tree` (folhas constantes) | −0,003 a +0,002 |
| V7 | learning rate 0,025 com cap de 3000 rodadas | 0 a +0,002 |
| V8 | `min_data_in_leaf` 200 → 800 | 0 a +0,002 |

Regra: adotar se o IC contra o E1 excluir 0 **e** ≥3/4 sementes positivas no pareado. Pegadinha
evitada: um segundo `--set` sobrescreve o primeiro (NOTAS §7). A primeira largada do V1 rodou sem
extra-trees, foi descartada e refeita.

## Resultados

| Braço | TS-AUC | Δ contra E1 [IC 95%] | Δ contra B0 [IC 95%] | Sementes (pareado contra E1) | Veredito |
|---|---|---|---|---|---|
| V1 ff0,5 | 0,6339 | −0,0010 [−0,0027; +0,0009] | +0,0061 [+0,0030; +0,0098] | 777 +0,0001 · 101 −0,0012 · 202 −0,0000 | nulo |
| V2 N1 (metade dos negativos) | 0,6262 | **−0,0088 [−0,0121; −0,0055]** | — | — | **piora**: o `meio_neg` do C2 não vira técnica. Tirar negativos do treino, com os pesos pareados do R1 calculados sobre todas as linhas, desequilibra o modelo |
| V3 receita histórica (sem `mrep_`, contri 0,6) | 0,6336 | −0,0013 [−0,0032; +0,0006] | +0,0058 [+0,0027; +0,0090] | 777 +0,0005 · 101 −0,0023 · 202 −0,0007 | nulo. A diferença entre o B0 (0,6278) e os 0,6312 históricos **não vem da receita** |
| V5 `bagging_fraction` 0,5 | 0,6337 | −0,0013 [−0,0033; +0,0007] | +0,0059 [+0,0025; +0,0094] | 777 −0,0021 · 101 −0,0009 | nulo |
| V6 sem `linear_tree` | 0,6333 | −0,0016 [−0,0037; +0,0003] | +0,0055 [+0,0026; +0,0090] | 777 −0,0004 · 101 +0,0005 | nulo, levemente pior. As folhas lineares ainda somam um pouco no bag, e o treino não ficou mais rápido (~9,8 min por semente). Mantém o `linear_tree` |
| V7 lr 0,025, cap 3000 | 0,6348 | −0,0002 [−0,0019; +0,0016] | +0,0070 [+0,0041; +0,0101] | 777 −0,0001 · 101 **+0,0035** · 202 **+0,0023** | nulo **no bag**: ganha por semente e some na média de 4. Mesmo padrão do E1 (os efeitos de redução de variância se sobrepõem). Custa 2× o tempo de treino e de inferência |
| V8 `min_data_in_leaf` 800 | 0,6343 | −0,0006 [−0,0025; +0,0013] | +0,0065 [+0,0035; +0,0097] | 777 −0,0007 · 101 +0,0006 | nulo |
| V4 top-65 (poda) | 0,6330 | −0,0019 [−0,0041; +0,0003] | +0,0053 [+0,0023; +0,0082] | — | nulo, levemente pior (a 1ª largada falhou por falta de `thin_weight` e foi refeita) |

## V9: E1 + S5a (famílias de evidência `escala_desce`/`escala_sobe` como features)

**Hipótese (antes de medir, 13:15):** o S5a deu +0,0044 [+0,0016; +0,0072] sobre o B0 (sem extra-trees).
Colunas novas e regularização são mecanismos diferentes, então podem somar sobre o E1. Previsão:
+0,000 a +0,004 contra o E1. Adotar se o IC excluir 0 e ≥ 3/4 sementes forem positivas.

**Resultado V9 (14:18):** 0,6356 · contra o E1 **+0,0007 [−0,0015; +0,0027]** · contra o B0 +0,0078 [+0,0047; +0,0116] · sementes 777 −0,0032 · 101 +0,0009. **Nulo:** as famílias de evidência não somam sobre o extra-trees.

## V10: E1 + |x| das features de log-variância/energia (hipótese antes de medir, 14:50)

**Mecanismo:** as quedas de variância aparecem como valores **negativos** nas features de log-variância
com sinal. No objetivo global, "variância baixa" costuma andar com negativos, então o GBM precisa de uma
relação em **U** aprendida com poucos eventos de queda (Onyx 0,564 nas quedas; um sinal especialista,
0,774). O |x| torna a relação monotônica e mais barata em amostra. Só a família de variância (~25
colunas), para não diluir o `feature_fraction`. Previsão: **+0,000 a +0,004** contra o E1, com o ganho
concentrado nas quedas. Adotar com IC excluindo 0 e ≥3/4 sementes positivas.
