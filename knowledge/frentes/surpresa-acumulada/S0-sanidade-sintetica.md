# S0 — Sanidade sintética do detector

**Frente:** [surpresa acumulada](README.md) · **Premissa:** [1](../../premissas/01-surpresa-acumulada.md) ·
**Status:** concluído (2 rodadas) · **Hipótese:** 2026-09-29 · **Resultado:** 2026-09-29

## Hipótese (escrita ANTES de medir)

1. **Nulo universal:** em séries sem quebra (iid N(0,1), AR(1) φ=0,6, GARCH(1,1) e t de Student com
   ν=4), `g_A` e `g_B` online têm média em |·| < 0,05 e dp entre 0,95 e 1,05, medidos em 200
   séries por tipo.
2. **A deriva de `sr` sob H0 é comum:** a mediana entre séries de `sr` em t=500 varia menos de 1,0
   nat entre os quatro tipos de série. Esse é o teste de comparabilidade transversal.
3. **Poder, conferindo a conta do texto:** com σ indo de 1,0 para 1,2 a partir de τ=1, a família
   `escala_sobe` passa o quantil 95% do controle em mediana **antes de t=150**, e a janela de 20
   pontos (log da razão de variâncias) não passa em t=150.
4. **Cobertura dos cenários do Onyx:** nos cenários com controle (T1, T3, T4, T5, T5b, T7, T8), a
   AUC `sr`-cenário × `sr`-controle, medida 100 passos depois de τ, é > 0,8 em todos, menos o T3
   (média +0,15, pequeno de propósito).
5. **Causalidade e caminho único:** o score em lote sobre `online[:k]` é igual às k primeiras linhas
   do lote sobre `online` inteiro, e `SurpresaStream` bate com o lote (|Δ| < 1e-9).

**Regra:** se 1 ou 2 falhar, o modelo por série está mal especificado e é preciso corrigi-lo antes
do S1. Se 3 ou 4 falharem, a evidência direcional não funciona como a teoria diz.

## Como rodar

```bash
python -m pytest tests/surpresa_det -q                 # item 5 + versões rápidas de 1 e 3
python -u scripts/surpresa_s0.py --n 200 --out artifacts/surpresa/s0_rodada2.json
```

## Resultado

200 séries por tipo, histórico de 2000 pontos, online de 500. JSONs em `artifacts/surpresa/s0_rodada{1,2}.json`.

### Rodada 1: desenho original (`sr`, e `sr_cal` = incrementos menos o viés cru)

| Item | Resultado | Veredito |
|---|---|---|
| 1. Nulo universal | média de `g` entre −0,004 e +0,003; dp entre 1,002 e 1,006 nos 4 tipos | **passa** |
| 2. Deriva comum de `sr` em t=500 | mediana 4,97 (iid) · 4,91 (AR1) · **6,32 (GARCH)** · 5,19 (t4), faixa **1,41**; q95 GARCH **20,9** contra ~7 | **falha** |
| 2'. `sr_cal` | pior: mediana 8,2 e q95 46 no GARCH, e infla até no iid (5,5) | **falha** |
| 3. Poder, σ 1,0→1,2 | a surpresa passa o q95 do controle em **t=33**; a janela de 20 pontos, em t=66, e oscila (não passa em t=150) | **passa** |
| 4. Cenários (AUC em τ+100) | T1 1,00 · T4 1,00 · T5 1,00 · T7 0,98 · T8 0,93 · **T5b 0,74** · T3 0,58 | falha só no T5b |
| 5. Stream = lote, prefixo causal | Δ máximo = 0,0 | **passa** |

**Diagnóstico.** No fluxo A, a escala congelada é obrigatória para não absorver quebras de
variância, mas numa série GARCH os períodos de alta volatilidade viram "evidência de escala". O
nulo dos **incrementos** continua certo em média; é a **variância** da soma que explode. A correção
pelo viés cru não ajuda: ela injeta ruído por direção, e o logsumexp, que é um máximo suave, o
transforma em inflação. O T5b é uma rampa de 200 passos (σ médio de ~1,12 nos 100 primeiros), um
caso lento de verdade.

### Mudança entre as rodadas (feita antes do S1, portanto não é seleção sobre dados reais)

`calibracao_por_direcao` (`src/surpresa/stream.py`) mede no ⅓ final do histórico, com o modelo
ajustado nos ⅔ iniciais:
- **têmpera** η_k = Var_LP teórica / Var_LP medida (batch means, lote de 25), cortada em [0,1; 1].
  A razão de verossimilhança é multiplicada por η_k, como numa verossimilhança temperada para um
  modelo mal especificado. A Var_LP teórica sai de Gauss-Hermite 3D (`_var_lp_nula`);
- **viés** encolhido por James-Stein contra o próprio erro-padrão.

Saídas: `sr` (cru), **`sr_eta`** (só têmpera), `sr_cal` (têmpera + viés encolhido).

### Rodada 2

| Score | Mediana em t=500: iid · AR1 · GARCH · t4 | Faixa | q95 GARCH / iid |
|---|---|---|---|
| `sr` | 4,97 · 4,91 · 6,32 · 5,19 | 1,41 | 20,9 / 6,8 |
| **`sr_eta`** | **4,87 · 4,86 · 5,30 · 5,05** | **0,44** | **10,7 / 6,8** |
| `sr_cal` | 5,06 · 5,22 · 5,72 · 5,39 | 0,66 | 14,6 / 8,0 |

| Cenário (AUC em τ+100) | `sr` | `sr_eta` | `sr_cal` | `ingenua_z` |
|---|---|---|---|---|
| T1 média +0,8 em t=3 | 1,000 | 1,000 | 1,000 | 0,992 |
| T3 média +0,15 | 0,582 | 0,582 | 0,585 | 0,525 |
| T4 média +1,5 | 1,000 | 1,000 | 1,000 | 1,000 |
| T5 σ×1,5 | 1,000 | 1,000 | 0,998 | 0,999 |
| T5b rampa de σ | 0,739 | 0,731 | 0,682 | 0,767 |
| T7 AR φ 0,2→0,6 (variância fixa) | 0,983 | 0,979 | 0,968 | **0,307** |
| T8 cauda t4 (variância fixa) | 0,929 | 0,928 | 0,915 | **0,395** |

Poder σ 1,0→1,2: primeiro t = 38, contra 66 da janela de 20 pontos.

## Decisão

- **`sr_eta` vira o score primário do S1.** É o único que passa o item 2 (faixa de 0,44). O
  q95 GARCH ainda está 1,6× o do iid: resta uma cauda de falsos alarmes em séries com clustering
  forte.
- `sr_cal` fica como variante; o viés, mesmo encolhido, piora as caudas.
- **A surpresa ingênua do texto original fica abaixo do acaso em quebras de dependência (T7 0,31)
  e de cauda (T8 0,40).** Somar só −log p não basta: quebras que reduzem a variância condicional ou
  mudam a forma diminuem a surpresa média. É a razão de verossimilhança direcional que faz a ideia
  funcionar.
- A falha do T5b (rampa lenta) está registrada; não vale mexer no desenho por ela.
- **Config congelada** em `configs/surpresa.yaml` para o S1.
