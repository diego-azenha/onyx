# C2 — O limite de amostra está nos positivos ou nos negativos?

**Frente:** [teto offline](README.md) · **Status:** concluído · **Hipótese:** 2026-09-30 01:25 · **Resultado:** 02:10

## Por quê

O [C1](C1-curva-aprendizado-onyx.md) mostrou que dobrar as séries rende +0,0144. Dobrar o quê? Se o que
falta são **negativos** (variedade de H0 entre tipos de série, ou seja, aprender o nulo de cada
perfil), a solução é barata: H0 é praticamente ilimitado, porque qualquer trecho sem quebra, inclusive
o histórico das séries com quebra, gera negativos reais por re-corte. Se faltam **positivos**
(eventos), só há saída com eventos novos, e as duas fontes testadas falharam
([A2](A2-transplante.md), [D25](D25-dados-2025.md)).

## Desenho

`scripts/run_c2_pos_neg.sh`, receita do B0, sementes 777 e 101:
- `meio_neg`: todos os positivos + metade (id par) dos negativos;
- `meio_pos`: todos os negativos + metade (id par) dos positivos.

Avaliação no mesmo conjunto do C1 (séries de id par), onde já se conhecem: metade de tudo **0,6129** e
tudo **0,6273**.

## Hipótese (escrita ANTES de medir)

- Se `meio_pos` ficar perto de 0,6273 e `meio_neg` perto de 0,6129, **o limite é de negativos** →
  aumentar negativos por re-corte (A3).
- No caso inverso, **o limite é de positivos** → só eventos novos resolvem.
- Previsão: um pouco de cada, com peso maior nos positivos (~60/40).

## Resultado

Mesmas 5.000 séries (id par), bag das sementes 777 e 101, grade do board:

| Treino | TS-AUC | Δ contra metade |
|---|---|---|
| metade de tudo (C1) | 0,6129 | — |
| **todos os positivos + metade dos negativos** (`meio_neg`) | **0,6299** | **+0,0170** |
| metade dos positivos + todos os negativos (`meio_pos`) | 0,6126 | −0,0003 |
| tudo (B0) | 0,6273 | +0,0144 |

## Decisão

- **O limite de amostra é inteiramente de positivos** (eventos de quebra). Negativos a mais não
  rendem nada. Isso fecha a porta do aumento de negativos por re-corte e explica por que o
  [A1](A1-recorte.md), que duplicava eventos, piorou.
- **Achado lateral:** `meio_neg` (0,6299) ficou **acima** de tudo (0,6273). Tirar metade dos
  negativos do treino ajudou. Os negativos em excesso parecem atrapalhar o aprendizado dos
  positivos, apesar dos pesos pareados por passo do R1. Braço direto: **[N1](N1-subamostra-negativos.md)**,
  subamostrar negativos no treino e prever OOF para todas as séries.
