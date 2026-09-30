# X1 — Calibração das famílias pelo nulo de excursões, e o teste de deriva

**Frente:** [teto offline](README.md) · **Status:** concluído: **negativo (os dois)** · **Data:** 2026-09-30 01:55

## X1: excursões

**Hipótese.** O [ST1](ST1-segundo-estagio.md) mostrou evidência falsa de queda nos negativos (trechos
calmos). O CUSUM de cada família roda no ⅓ final do histórico (modelo dos ⅔ iniciais), e o valor
online vira quantil dessa distribuição (`src/surpresa/excursao.py`, `scripts/x1_excursoes.py`). Com
isso, uma série que naturalmente produz excursões recebe evidência calibrada menor.

**Resultado** (TS-AUC só com positivos do tipo, contra todos os negativos; `exd` = calibrado com
desempate pelo cru):

| Coluna | desce | sobe | var_fixa | curta | global |
|---|---|---|---|---|---|
| `cus_escala_desce` (cru) | **0,744** | 0,288 | 0,452 | 0,456 | 0,464 |
| `exd_escala_desce` (calibrado) | 0,673 | 0,329 | 0,466 | 0,484 | 0,473 |
| `cus_escala_sobe` (cru) | 0,339 | **0,843** | 0,569 | 0,544 | **0,579** |
| `exd_escala_sobe` (calibrado) | 0,420 | 0,794 | 0,555 | 0,516 | 0,571 |

**A calibração tira poder sem ganhar ordenação global.** O ⅓ do histórico (333–1.666 pontos) é
curto para estimar a cauda das excursões, e o ruído dessa estimativa entra no score.

## Deriva: o H0 é estacionário?

Se o online de um negativo se afastasse do histórico com t (deriva de séries reais), o nulo teria de
depender de t. Mediana de |Δ log var| entre a janela de 100 pontos que termina em t e os últimos 500
do histórico, nos negativos, contra o mesmo contraste feito **dentro** do histórico na mesma
distância:

| t | 100 | 200 | 400 | 600 | 800 |
|---|---|---|---|---|---|
| online | 0,173 | 0,179 | 0,178 | 0,188 | 0,184 |
| dentro do histórico | 0,170 | 0,176 | 0,173 | 0,176 | 0,171 |

A deriva é pequena (~0,01 em t ≥ 600). O H0 é praticamente estacionário e não explica os falsos
positivos, que vêm do ruído amostral de janela: o |Δ log var| de 0,17 é o próprio erro de uma
variância estimada com 100 pontos em cauda pesada.
