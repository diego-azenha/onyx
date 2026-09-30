# S5 — Famílias de evidência direcional como features do Onyx

**Frente:** [surpresa acumulada](README.md) (reaberta pelo achado pós-fechamento) ·
**Premissa:** [1](../../premissas/01-surpresa-acumulada.md) · **Status:** hipótese registrada ·
**Hipótese:** 2026-09-29, antes de qualquer treino com as colunas novas

## Motivação

O [achado pós-fechamento](README.md#achado-pós-fechamento-2026-09-29-o-onyx-é-quase-cego-a-quedas-de-variância)
mostrou que o Onyx é quase cego a **quedas** de variância: m=100 passos depois de τ, essas quebras
ficam no percentil 0,56, contra 0,47 dos negativos. A família `escala_desce` do detector as põe em
0,80. O S3 não capturou isso porque a combinação era aditiva, e a `escala_desce` derruba as
quebras de alta. Uma árvore pode condicionar.

## Hipótese (escrita ANTES de medir)

- **Braço principal S5a:** `train_rows` + 2 colunas, `fam_escala_desce` e `fam_escala_sobe`.
- **Braço secundário S5b (exploratório):** `train_rows` + as 8 colunas `fam_*`. Largura sem
  ordenação nova já regrediu duas vezes ([NOTAS §5.1](../../operacao/NOTAS_AGENTES.md)); este braço
  mede se as outras 6 famílias somam ou diluem.
- **Receita idêntica ao [B0](B0-incumbente.md):** mesmas sementes (777, 101, 202, 303), `--detectability-mode
  soft --linear-tree`, mesma partição de folds (`cfg.seed` 42), v-EMA do B3 por cima.
  **Baseline:** `oof_b0_vema_bag4` (0,6278).
- **Como as colunas entram:** o merge por (`id`, `t`) de `artifacts/surpresa/scores.parquet` equivale
  a adicionar features, porque o score em t só usa histórico + online até t (teste de prefixo em
  `tests/surpresa_det`). Não há `StateBlock` ainda: isto mede o **valor**. Custo de latência e
  implantação ficam para depois, se passar.
- **Previsão (S5a):** Δ geral entre **+0,002 e +0,008**, IC excluindo 0.
- **Buckets-alvo declarados:** `150<t≤400` e `t>400`, onde a separação das quedas é maior (0,80
  contra 0,56).
- **Checagem de mecanismo:** o percentil médio do OOF nas quebras `desce` em m=100 sobe de 0,560
  para **≥ 0,65**, sem as `sobe` caírem mais que 0,02. Se o Δ agregado vier positivo sem esse
  movimento, o ganho não é o que se pensava.
- **Regra de decisão:** adotar a direção (e então construir o `StateBlock`) se o IC excluir 0 no
  agregado ou num bucket-alvo. Com uma partição só, a réplica na 43 é obrigatória antes de
  empacotar ([NOTAS §5, item 9](../../operacao/NOTAS_AGENTES.md)).

## Como rodar

```bash
nohup bash scripts/run_s5_familias.sh > artifacts/surpresa/s5.log 2>&1 &
```

## Resultado — S5a (o S5b, com 8 colunas, ainda estava rodando quando isto foi escrito)

R0 do bag de 4 com v-EMA contra `oof_b0_vema_bag4`: **0,6278 → 0,6322, Δ +0,0044 [+0,0016; +0,0072]**,
IC excluindo 0 no agregado e em `50–150` (+0,0062), `150–400` (+0,0032) e `>400` (+0,0051).

**Mas a leitura honesta é mais fraca que o IC:**

| Checagem | Resultado |
|---|---|
| Δ por semente (pareado com a mesma semente do B0) | 777 **+0,0075** · 101 **+0,0027** · 202 **+0,0040** · 303 **−0,0015**; média +0,0032, EP ≈ 0,002. O bootstrap não vê essa variância ([NOTAS §5](../../operacao/NOTAS_AGENTES.md)) |
| **Mecanismo registrado** (quebras `desce` em m=100 de 0,560 para ≥ 0,65) | **falhou**: 0,560 → 0,572 |
| TS-AUC com positivos de um tipo só contra todos os negativos | `desce` +0,0106 · quebras curtas (<30 pontos pós) +0,0118 · `var_fixa` +0,0047 · `sobe` **−0,0027** |

## Decisão

- **Não adotar ainda.** O ganho tem a direção certa (quedas e quebras curtas), mas é pequeno,
  depende de semente e o movimento previsto não aconteceu. Na melhor leitura é +0,003, polimento,
  não salto.
- Se for retomado: K=7 por lado e réplica na partição 43 antes de construir o `StateBlock`.
