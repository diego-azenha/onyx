# E1 — Extra-trees no Onyx (redução de variância)

**Frente:** [teto offline](README.md) · **Status:** **ADOTADO** (partições 42 e 43; `default.yaml`; notebook verificado) · **Hipótese:** 2026-09-30 01:40 · **Resultado:** 02:50

## Por quê

O [C1](C1-curva-aprendizado-onyx.md) mediu o Onyx limitado por amostra. Sem fonte de dados extras
([A2](A2-transplante.md) e [D25](D25-dados-2025.md) descartados), a alternativa é reduzir a variância
do aprendiz. No oráculo offline, o extra-trees foi o único regularizador com ganho visível: 0,6665 →
0,6718. Ensembles diversos não passaram disso ([D25](D25-dados-2025.md), nota lateral; teste de
ensemble no diário).

## Mudança

`LightGBMConfig.extra_trees` (default `False`, no-op exato) repassado ao LightGBM em
`model/train.py:train_ensemble`. Braço: `--set extra_trees=true` sobre a receita do B0.
`scripts/run_e1_extra_trees.sh`.

## Hipótese (escrita ANTES de medir)

- Δ entre **+0,002 e +0,006** contra `oof_b0_vema_bag4`, com K=4 e grade do board.
- **Regra:** IC excluindo 0 e ≥ 3 de 4 sementes positivas no pareado → adotar e combinar com o
  S5/A1, se passarem.
- Risco conhecido: com `linear_tree`, splits aleatórios podem gerar folhas com regressões mal
  condicionadas.

## Resultado

R0 do bag de 4 com v-EMA contra `oof_b0_vema_bag4`, grade do board, partição 42:
**0,6278 → 0,6349, Δ +0,0071 [+0,0038; +0,0102]**.

| Bucket | Δ | IC 95% |
|---|---|---|
| **geral** | **+0,0071** | **[+0,0038; +0,0102]** |
| t≤50 | +0,0058 | [−0,0065; +0,0165] |
| **50<t≤150** | **+0,0116** | **[+0,0053; +0,0173]** |
| **150<t≤400** | **+0,0073** | **[+0,0032; +0,0113]** |
| **t>400** | **+0,0045** | **[+0,0002; +0,0088]** |

Por semente, pareado com a mesma semente do B0: 777 **+0,0111** · 101 **+0,0079** · 202 **+0,0082** ·
303 **+0,0025**. As 4 são positivas, e os modelos individuais vão de ~0,620 para ~0,630.

## Leitura

- **É o maior ganho isolado do projeto** (o V4 tinha dado +0,0060), e a previsão (+0,002 a +0,006)
  errou para baixo. **Confirma a tese do [C1](C1-curva-aprendizado-onyx.md)**: o Onyx é limitado por
  amostra, e randomizar os limiares de split reduz a variância do aprendiz.
- O ganho por semente (~+0,008) é maior que no bag (+0,0071), porque o bag também reduz variância e os
  dois efeitos se sobrepõem em parte.
- O risco previsto (extra-trees com `linear_tree` gerando folhas mal condicionadas) não apareceu.

## Réplica na partição 43 (05:43)

`scripts/run_rep43.sh`: B0 e E1 com `--fold-seed 43`, sementes 777 e 101, v-EMA, R0 na mesma partição:
**0,6193 → 0,6308, Δ +0,0115 [+0,0080; +0,0159]**, com IC excluindo 0 em **todos** os buckets
(t≤50 +0,0231 · 50–150 +0,0101 · 150–400 +0,0105 · >400 +0,0123). Semente 777 sozinha: +0,0136.

## Adoção (05:50)

- `configs/default.yaml`: `lightgbm.extra_trees: true`, com o registro das duas partições no comentário.
- `scripts/submission_smoke_test.py`: `train()` OK (K=7, a guarda numérica da fusão de boosters passou)
  e `infer()` OK.
- `submission_notebook.ipynb` regenerado; `verify_submission_notebook.py`: **equivalência bit a bit**
  (células, convertido e subprocesso paralelo), max |Δ| = 0.
- **Estimativa de placar:** o OOF prevê o placar com erro de ~0,002 (NOTAS §5), então ~0,635.

## Próximos passos (registrados antes da adoção)

1. Réplica na partição 43 (`--fold-seed 43`), com baseline e candidato na mesma partição.
2. Empilhar mais redução de variância sobre o E1: fila `scripts/_fila_v.sh` (V1 `feature_fraction=0,5`,
   V2 + N1, V3 receita histórica, V4 poda top-65), R0 contra o E1.
3. Antes de ligar em produção: `verify_submission_notebook.py` / `submission_smoke_test.py`, porque o
   extra-trees passa pela fusão de boosters (a lição do HISTORICO §15.19).
