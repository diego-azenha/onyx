# S2 — Redundância transversal contra o Onyx

**Frente:** [surpresa acumulada](README.md) · **Premissa:** [1](../../premissas/01-surpresa-acumulada.md) ·
**Status:** concluído · **Hipótese:** 2026-09-29 (depois do S1, antes do B0 terminar) · **Resultado:** 2026-09-29

## Hipótese (escrita ANTES de medir)

- **Métrica:** correlação de Spearman **dentro do passo**, média sobre uma grade de passos
  (10–100 de 10 em 10, 150–950 de 50 em 50). A TS-AUC só vê a ordenação dentro do passo (C1), então
  uma correlação ~1 com algo que o Onyx já tem significa que a coluna não pode mover nada
  ([NOTAS §5.1](../../operacao/NOTAS_AGENTES.md)).
- **Previsão:** `sr_eta` vs OOF do Onyx (B0) entre **0,45 e 0,70**. O S1 mostrou que o sinal do
  detector vive quase todo no eixo de variância, que o Onyx cobre com dezenas de colunas.
  `sr_eta` vs `conformal_logm_abs`: entre 0,6 e 0,8, porque são acumuladores de surpresa sobre o
  mesmo fluxo congelado.
- **Leitura:** acima de 0,9 contra o OOF, o S3 dificilmente passa. Abaixo de 0,7, existe espaço de
  ordenação, e o S3 decide.

## Como rodar

```bash
python -u scripts/surpresa_s2_s3.py     # S2 e S3 juntos → artifacts/surpresa/s2_s3.json
```

## Resultado

Spearman médio dentro do passo (28 passos entre t=10 e t=950), sobre o OOF do [B0](B0-incumbente.md):

| Score | vs OOF Onyx | vs `conformal_logm_abs` | vs `cusum_var_up_r150` | vs `accum_window_var_ln_w100_cal` | vs `bayes_lo_h0100` |
|---|---|---|---|---|---|
| **`sr_eta`** | **0,283** | −0,144 | 0,072 | −0,086 | 0,679 |
| `sr_cal` | 0,324 | 0,052 | 0,188 | 0,066 | 0,503 |
| `fam_escala_sobe` | 0,492 | 0,745 | 0,826 | 0,803 | 0,146 |

## Leitura

- **A previsão errou para baixo:** a correlação com o OOF (0,28) ficou bem abaixo da faixa prevista
  (0,45–0,70), e com `conformal_logm_abs` ela é **negativa**. O score agregado ordena as séries de um
  jeito muito diferente do Onyx.
- A família `escala_sobe` sozinha é, como esperado, muito parecida com as colunas de variância do
  Onyx (0,75–0,83). O agregado `sr_eta` se afasta delas porque o logsumexp mistura as 23
  alternativas, e no corte transversal quem domina é a direção de maior evidência em cada série. Na
  prática ele é mais parecido com o filtro bayesiano de troca única (0,68).
- **Espaço de ordenação existe.** Pela regra registrada (abaixo de 0,7 o S3 decide), a redundância
  não bloqueia nada. O [S3](S3-combinacao.md) mostra que esse espaço **não carrega sinal**: a parte
  ortogonal ao Onyx é, em grande parte, a ordenação dos ~⅔ de quebras invisíveis e dos negativos
  com evidência espúria ([S1](S1-detector-sozinho.md)).
