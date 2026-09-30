# Passo 4 — Especialistas somados ao score com poucos pesos

**Frente:** [roteiro 30/09](README.md) · **Status:** concluído: **nulo** · **Hipótese:** 2026-09-30 11:45

## Contexto

Já medido: o S3 (aditivo com um peso, sobre o B0) deu +0,0001, e o ST1 (GBM raso de segundo estágio)
deu −0,0011. Hoje, uma única variante sobre o **E1**: logit(E1 com v-EMA) + w₁·`fam_escala_desce`/dp +
w₂·`fam_escala_sobe`/dp. Os pesos saem de uma grade 6×6 ajustada em ids pares e aplicada nos ímpares, e
vice-versa. `scripts/p4_especialistas.py`, R0 contra o E1.

## Hipótese (escrita ANTES de medir)

Δ de **0 a +0,003**. Adotar só com IC excluindo 0; senão, o passo 4 fecha junto com o S3 e o ST1.

## Resultado

| Metade de ajuste | w_desce | w_sobe | Ganho na própria metade |
|---|---|---|---|
| pares → aplicado nos ímpares | **0,0** | 0,1 | +0,00012 |
| ímpares → aplicado nos pares | **0,0** | 0,1 | +0,00009 |

TS-AUC com cross-fit: **0,63492 → 0,63502 (+0,0001)**. O R0 não foi rodado: com Δ=+0,0001, ele não
tem como excluir 0.

## Decisão

**Nulo. O passo 4 fecha** junto com o S3 (+0,0001) e o ST1 (−0,0011). As duas metades escolhem
**peso zero para a família de quedas**, mesmo ela separando as quedas a 0,77 no subconjunto: no
ranking global, o que ela sobe entre os negativos cancela o que ganha nas quebras. O E1 já absorve o
pouco que a família de altas acrescentava.
