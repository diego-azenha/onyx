# S1 — O detector sozinho nas 10.000 séries de treino

**Frente:** [surpresa acumulada](README.md) · **Premissa:** [1](../../premissas/01-surpresa-acumulada.md) ·
**Status:** concluído · **Hipótese:** 2026-09-29 · **Resultado:** 2026-09-29

## Hipótese (escrita ANTES de medir)

- **Régua:** TS-AUC ponderada por `n_pos·n_neg` em **todos** os passos online de `y_train.parquet`
  (5.036.517 linhas, que é a grade do board), sobre as 10k séries de treino. O detector não usa
  rótulo nenhum, então isso é uma medida honesta, desde que a config fique fixa. A config de
  `configs/surpresa.yaml` está **congelada antes da primeira medição**; qualquer ajuste depois dela
  é reportado como seleção, com a comparação entre as metades par e ímpar de ids.
- **Gates do texto original, mantidos:** `sr` ≥ **0,62** significa fonte de informação forte e
  diferente. `sr` ≤ **0,55** significa descartar, se o S3 também não mostrar ganho. Entre os dois,
  o S2 e o S3 decidem.
- **Previsão pontual:** `sr` ≈ 0,58–0,61. O Onyx completo faz 0,6312 com ~190 features e
  supervisão, então um único score não supervisionado acima de 0,62 seria surpreendente.
- **Score primário (atualizado depois do [S0](S0-sanidade-sintetica.md), antes de qualquer medição
  real): `sr_eta`**, o único que passou o teste de deriva comum. Os gates acima valem para ele.
- **Previsão de ordenação:** `sr_eta` ≥ `sr` > `ingenua_z`; `sr_cal` ≤ `sr_eta`. A surpresa ingênua é
  unidirecional e fica abaixo do acaso em dependência e cauda (S0).
- **Bucket onde a surpresa deveria ganhar**, declarado a priori: `150<t≤400` e `t>400`, onde a
  acumulação tem pontos pós-quebra suficientes e a diluição por janela mais pesa
  ([HISTORICO §7](../../historico/HISTORICO.md)).
- **Referência:** `conformal_logm_abs` (feature nº 1 do Onyx), usada sozinha como score na grade
  thin com multiplicadores do board. Só fica disponível depois do dataset do B0.
- **Controle de seleção:** as TS-AUC nas metades par e ímpar de ids devem concordar dentro do IC.

## Como rodar

```bash
python -u scripts/surpresa_avaliar.py --n-jobs 4            # ~40 s de pontuação + ~20 min de TS-AUC/bootstrap
# depois do B0, com a referência do Onyx:
python -u scripts/surpresa_avaliar.py --reusar --n-boot 0 --onyx-rows data/processed/train_rows.parquet
```

Saídas: `artifacts/surpresa/scores.parquet` (5.036.517 linhas × 16 colunas) e `artifacts/surpresa/s1.json`.
A pontuação das 10k séries leva **35 s** com 4 processos.

## Resultado

TS-AUC na grade do board, 10.000 séries:

| Score | Geral | t≤50 | 50–150 | 150–400 | >400 | metade par / ímpar |
|---|---|---|---|---|---|---|
| **`sr_eta`** (primário) | **0,5750** IC [0,5673; 0,5830] | 0,5240 | 0,5437 | 0,5742 | 0,5995 | 0,5769 / 0,5730 |
| `sr` | 0,5753 | 0,5219 | 0,5432 | 0,5757 | 0,5988 | 0,5782 / 0,5723 |
| `sr_cal` | 0,5837 | 0,5298 | 0,5501 | 0,5820 | 0,6113 | 0,5830 / 0,5844 |
| `cusum` | 0,5767 | 0,5211 | 0,5464 | 0,5762 | 0,6007 | 0,5810 / 0,5723 |
| `ingenua_z` | 0,5674 | 0,5322 | 0,5560 | 0,5712 | 0,5724 | 0,5707 / 0,5642 |
| `fam_escala_sobe` | 0,5851 | 0,5298 | 0,5654 | 0,5886 | 0,5974 | 0,5897 / 0,5803 |
| `fam_escala_desce` | **0,4533** | 0,4753 | 0,4534 | 0,4495 | 0,4561 | 0,4500 / 0,4565 |
| `fam_forma4` | 0,5407 | 0,5219 | 0,5258 | 0,5375 | 0,5559 | |
| `fam_cauda` | 0,5169 | | | | | |
| `fam_correlacao` | 0,5168 | | | | | |
| `fam_arch` | 0,5103 | | | | | |
| `fam_media` | 0,5093 | | | | | |
| `fam_forma3` | 0,4959 | | | | | |

### Referência: features do Onyx usadas sozinhas como score

Grade thin do dataset do B0 com os multiplicadores do board (`board_grid_multipliers`). **Checagem do
instrumento:** nessa régua o `sr_eta` dá 0,5750, idêntico à grade cheia, então as duas réguas
concordam. JSON em `artifacts/surpresa/s1_referencia_onyx.json`.

| Coluna | Geral | t≤50 | 50–150 | 150–400 | >400 |
|---|---|---|---|---|---|
| `cusum_var_up_r150` (Onyx) | 0,5797 | 0,5284 | 0,5607 | 0,5851 | 0,5888 |
| `accum_window_var_ln_w100_cal` (Onyx) | 0,5661 | 0,5251 | 0,5541 | 0,5719 | 0,5692 |
| `bayes_lo_h0100` (Onyx) | 0,5648 | 0,5229 | 0,5480 | 0,5634 | 0,5814 |
| `conformal_logm_abs` (Onyx, nº 1 em XS-SHAP) | 0,5633 | 0,5307 | 0,5541 | 0,5664 | 0,5678 |
| `bocpd_cp_prob` (Onyx) | 0,4910 | | | | |
| **`sr_eta`** | 0,5750 | 0,5240 | 0,5439 | 0,5743 | 0,5996 |
| `sr_cal` | **0,5837** | 0,5298 | 0,5502 | 0,5821 | **0,6113** |
| `fam_escala_sobe` | **0,5851** | 0,5298 | **0,5656** | **0,5887** | 0,5975 |

**Leitura:** o score de surpresa, sozinho, fica no nível das **melhores colunas individuais do Onyx**.
O `sr_cal` e o `fam_escala_sobe` ficam um pouco acima de todas as colunas medidas, e o ganho maior
aparece em `t>400`, onde a acumulação tem mais pontos. É uma coluna **boa**, não uma fonte de
informação **diferente**. Os 0,63 do Onyx vêm de combinar ~190 colunas desse nível.

### Diagnóstico por tipo de quebra (com rótulos, só para descrição)

Tipo pela variância do trecho pós-quebra contra o histórico: `sobe` Δlogvar > 0,3, `desce` < −0,3,
`var_fixa` no meio. Números = percentil médio da série no corte transversal do passo, m passos
depois de τ (a métrica de trabalho do [HISTORICO §7](../../historico/HISTORICO.md)):

| m | tipo (n) | `sr_eta` | `fam_escala_sobe` | `fam_escala_desce` | `fam_cauda` | `fam_correlacao` |
|---|---|---|---|---|---|---|
| 100 | desce (477) | 0,728 | 0,284 | **0,800** | 0,723 | 0,628 |
| 100 | sobe (537) | 0,703 | **0,833** | 0,261 | 0,607 | 0,466 |
| 100 | **var_fixa (2397)** | **0,470** | 0,546 | 0,449 | 0,458 | 0,498 |
| 200 | desce (309) | 0,771 | 0,281 | 0,826 | 0,752 | 0,652 |
| 200 | sobe (400) | 0,777 | 0,873 | 0,238 | 0,634 | 0,457 |
| 200 | **var_fixa (1770)** | **0,489** | 0,558 | 0,452 | 0,447 | 0,502 |
| — | negativos, todos os passos | 0,475 | 0,475 | 0,513 | 0,496 | 0,494 |

Censo sem modelo, sobre 4.364 séries com quebra e ≥30 pontos pós-quebra: as quebras de variância
são **simétricas** (15,9% com Δlogvar > +0,3 e 16,8% com < −0,3; mediana −0,018). ⅔ das quebras
não mexem na variância nesse limiar.

## Leitura

1. **Veredito pelos gates registrados:** `sr_eta` = 0,575, entre 0,55 e 0,62. Não é descarte nem
   fonte forte; **S2 e S3 decidem**.
2. **A previsão pontual (0,58–0,61) errou por pouco para baixo.** A de ordenação errou na parte do
   `sr_cal`: fora do sintético, a correção de viés por série **ajuda** (+0,009, e as duas metades
   concordam). O S0 penalizou o `sr_cal` pelas caudas em GARCH puro; os dados reais têm mais
   não-estacionariedade do que o GARCH sintético, e o viés medido fora da amostra a captura.
3. **Onde o detector funciona, funciona bem.** Nas quebras de variância, nos dois sentidos, a série
   quebrada fica no percentil 0,70–0,78 do corte transversal, e cada família de escala acerta o seu
   sentido (0,83–0,87).
4. **Os ~⅔ restantes das quebras são invisíveis para ele, e ficam abaixo dos próprios negativos**
   (0,47–0,49). Não há quebra forte de variância aí, e as outras famílias (cauda, correlação,
   forma) não compensam. É o mesmo ponto cego do Onyx em dependência pura ([HISTORICO §7](../../historico/HISTORICO.md), 0,492).
5. **Por que `fam_escala_desce` < 0,5 no agregado, com quebras de queda tão comuns quanto as de
   alta:** ela acerta as de queda (0,80), mas as de alta ficam em 0,26 e as `var_fixa` em 0,45,
   abaixo dos negativos (0,51). A mistura de sinais cancela. **O score agregado sofre do mesmo
   efeito**: somar evidência de todas as direções não ajuda quando ⅔ das quebras não aparecem em
   nenhuma delas.
6. **Implicação para a premissa 1:** a surpresa acumulada **acelera** a detecção nos eixos que já
   eram visíveis (S0: σ×1,2 em 38 passos contra 66 da janela), mas não enxerga um eixo novo nos
   dados reais. Se o Onyx já extrai bem o eixo de variância, o S2 vai mostrar redundância alta. A
   chance de ganho no S3 está na **velocidade** (buckets 50–150 e 150–400).
7. **Pergunta nova (premissa 4):** o que muda nas ~2.900 quebras `var_fixa`? Nem variância, nem
   cauda, nem correlação lag-1, nem média, com os limiares do detector. Candidatos: dependência em
   lags maiores, sazonalidade, ou mudanças pequenas demais. O censo A1 do Onyx
   (`break_type_census.py`) responde isso em minutos.
