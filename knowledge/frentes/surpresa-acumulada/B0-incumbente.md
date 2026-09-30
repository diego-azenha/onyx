# B0 — Reconstruir o incumbente neste clone

**Frente:** [surpresa acumulada](README.md) · **Status:** concluído · **Início:** 2026-09-29 21:02 · **Fim:** 22:15

## Por quê

Este clone não tem `data/processed/` nem `artifacts/`, e não existe OOF do Onyx em nenhum lugar da
máquina. S2, S3 e S4 precisam de um OOF do incumbente sobre **os mesmos dados**.

## Hipótese (antes de rodar)

- O OOF reconstruído, depois do v-EMA, fica em **0,631 ± 0,005** de TS-AUC na grade do board
  (partição 42), o que reproduz os 0,6312 do [HISTORICO §15.18](../../historico/HISTORICO.md).
- Se ficar fora dessa faixa, a receita diverge do pacote medido. Nesse caso ele ainda serve de
  baseline **relativo** para S2 e S3 (as comparações são pareadas contra ele), mas o nível não pode
  ser citado como "o incumbente".

## Receita

`scripts/run_b0_incumbente.sh` (resumível):
1. `build_dataset.py --n-jobs 4` com a config atual (`default_blocks`: V4 + mrep + BOCPD) →
   `data/processed/train_rows.parquet`.
2. `detectability.csv` via `model/detectability.py:compute_detectability_map`, o mesmo cálculo
   inline da produção, bit-idêntico ao CSV do F0.b segundo a
   [NOTAS §6](../../operacao/NOTAS_AGENTES.md).
3. `train.py --detectability-mode soft --linear-tree`, sementes 777, 101, 202 e 303 → `oof_b0_s*`.
4. `avg_oof.py` → `oof_b0_bag4`; `apply_vema_oof.py --a-up 1.0 --a-dn 0.2 --min-t 50` →
   `oof_b0_vema_bag4`.

Dados: `data/` copiado de `Documents/CrunchDAO/SBC/data` (2026-09-28), idêntico à cópia de junho
([premissa 4](../../premissas/04-gerador-das-quebras.md)).

## Resultado

Custos neste clone: dataset em **35 min** (`--n-jobs 4`, 2.541.134 linhas, 194 features, a mesma
contagem de linhas do histórico); `detectability.csv` em 7 min (712 séries, o mesmo número do
experimento X1); **7,4 min por semente** de treino com `linear_tree`.

| OOF | TS-AUC, grade do board |
|---|---|
| `oof_b0_s777` · `s101` · `s202` · `s303` | 0,6193 · 0,6212 · 0,6213 · 0,6270 |
| `oof_b0_bag4` | 0,6271 |
| **`oof_b0_vema_bag4`** (baseline do S2/S3) | **0,6278** |

- **Dentro da faixa registrada (0,631 ± 0,005), mas 0,0034 abaixo dos 0,6312 do HISTORICO
  §15.18.** O B3 (v-EMA) rendeu +0,0007, o esperado. A diferença está no D1: aqui, o `linear_tree`
  com K=4 dá o mesmo nível do baseline pré-D1 (`oof_b6c6_joint_bag4` = 0,6277). Diferenças de
  receita que podem explicar isso: o dataset atual tem 194 features (`default_blocks` com mrep e
  BOCPD, **sem** o `--drop-prefix spec_ ord_ mrep_` da campanha); o `feature_contri_meta` do YAML é
  0,7 (C1, adotado depois) contra os 0,6 da medição do D1; e as sementes são outras. **Não
  investigado**: o B0 serve como baseline **relativo** das comparações pareadas, e o nível não deve
  ser citado como "o incumbente".
- Pendência para quem retomar o Onyx: o D1 foi adotado com K=4 contra um baseline de 187 features.
  Vale medir o `linear_tree` isolado nesta receita de 194 features antes de confiar nele no pacote.
