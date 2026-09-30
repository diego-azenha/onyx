# S3 — Combinação sem retreino: Onyx + surpresa

**Frente:** [surpresa acumulada](README.md) · **Premissa:** [1](../../premissas/01-surpresa-acumulada.md) ·
**Status:** concluído: **nulo** · **Hipótese:** 2026-09-29 (depois do S1, antes do B0 terminar) · **Resultado:** 2026-09-29

## Hipótese (escrita ANTES de medir)

- **Candidato:** `logit(OOF Onyx) + w · sr_eta/dp(sr_eta)`, com `w` escolhido numa grade de 9
  valores. A escolha é cruzada: `w` ajustado nas séries de id par é aplicado nas de id ímpar, e
  vice-versa. Nenhuma série é avaliada com um `w` que a viu.
- **Baseline:** `oof_b0_vema_bag4` (o incumbente reconstruído, [B0](B0-incumbente.md)).
- **Régua:** `compare_oof.py --grid full --n-boot 300`, com IC 95% pareado por série.
- **Declarado a priori:** a coluna de decisão é **`sr_eta`**, e os buckets-alvo são **`50<t≤150`
  e `150<t≤400`**, onde a acumulação deveria ganhar em velocidade das janelas. `sr_cal` e
  `fam_escala_sobe` também são medidos, mas como **exploratórios** (foram escolhidos depois de ver o
  S1) e não decidem nada sozinhos.
- **Previsão:** Δ geral entre **+0,000 e +0,004**. A combinação só pode somar o que não for
  redundante (S2), e o sinal do detector é de variância, que é justamente o eixo mais forte do Onyx.
- **Regra de decisão** ([NOTAS §5](../../operacao/NOTAS_AGENTES.md)): o S4 só acontece se o IC
  excluir 0 no agregado ou num bucket-alvo. Como o OOF do Onyx é um bag de K=4 e o detector é
  determinístico, a variância de semente do booster já está no baseline e no candidato por igual.
- **Regra de descarte (texto original):** o S1 ficou em 0,575 (> 0,55), então a regra "≤ 0,55 e S3
  sem ganho" não se aplica. Se o S3 der nulo, a frente fecha pelo S3.

## Como rodar

```bash
python -u scripts/surpresa_s2_s3.py
python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
  --candidate artifacts/surpresa/oof_b0_mais_sr_eta.parquet --n-boot 300 --n-jobs 4 \
  --target-bucket "150<t<=400" --out artifacts/surpresa/s3_r0_sr_eta.json
```

## Resultado

Pesos escolhidos pelo ajuste cruzado: `sr_eta` w=0,05 nas duas metades (ganho de +0,0001 a +0,00015
**dentro** da própria metade de ajuste); `sr_cal` w=0,1 / 0,3; `fam_escala_sobe` w=0,2 / 0,1.

R0 contra `oof_b0_vema_bag4` (0,6278), grade do board, 300 réplicas:

| Candidato | Geral | t≤50 | 50–150 | 150–400 | >400 |
|---|---|---|---|---|---|
| **`sr_eta`** (decisão) | **+0,0001** [−0,0003; +0,0004] | −0,0000 | −0,0000 | +0,0001 [−0,0003; +0,0005] | +0,0002 |
| `sr_cal` (exploratório) | +0,0001 [−0,0013; +0,0014] | −0,0005 | +0,0002 | +0,0001 | −0,0000 |
| `fam_escala_sobe` (exploratório) | +0,0002 [−0,0004; +0,0009] | +0,0009 [−0,0005; +0,0026] | +0,0006 [−0,0002; +0,0013] | +0,0001 | +0,0001 |

Nenhum IC exclui 0, nem no agregado nem nos buckets-alvo. JSONs em `artifacts/surpresa/s3_r0_*.json`.

## Decisão

- **Nulo. O S4 não acontece**, pela regra registrada. A previsão (+0,000 a +0,004) acertou na
  borda de baixo.
- **Por quê, juntando S1, S2 e S3:** a surpresa acumulada é uma boa coluna de variância, mais rápida
  que as janelas. Mas o Onyx já extrai o eixo de variância com dezenas de colunas, e o que o score
  tem de ortogonal ao Onyx (ρ = 0,28) é ruído: a ordenação dos ~⅔ de quebras que não aparecem em
  eixo nenhum.
- **Sobre a premissa 1:** "a quebra aparece em resumos" não era o gargalo. Resumo ou surpresa
  acumulada, os dois leem os mesmos eixos; a acumulação só lê mais rápido, e a velocidade que ela
  ganha já não vale nada no agregado (os buckets 50–150 e 150–400 ficaram em ~0).
- A dica mais forte que sobra é exploratória: `fam_escala_sobe` mostra os maiores pontos em `t≤150`
  (+0,0009 e +0,0006), exatamente onde a velocidade deveria pagar, mas com IC incluindo 0 e peso de
  21% no board. Não justifica o S4 sozinho.
