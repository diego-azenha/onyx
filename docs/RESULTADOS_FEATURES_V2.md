# Resultados da implementação da proposta V2 (F1–F4)

**Data:** 2026-07-20
**Implementado:** F1 (calibração de nulo por série), F2 (impressão digital de H0 estendida),
F3 (MMD de kernel via RFF, marginal e conjunto), F4 (energia multi-escala Haar).
**Não implementado, por decisão do próprio diagnóstico:** F6 (painel JS/Hellinger/W1/KS) — morto pelo
§5; F5 (padrões ordinais) — despriorizado pelo mesmo motivo.
**Modelo:** `artifacts/models/v2` · 137 features (era 91) · dataset `train_rows.parquet` reconstruído.

---

## 1. Resumo

**O primeiro ganho estatisticamente significativo do projeto**, mas localizado: `t>400` melhora
+0,0105 (IC 95% [0,0030, 0,0187], exclui 0) contra o modelo anterior. O agregado continua plano
(+0,0014, IC inclui 0) porque o bucket `t≤50` perdeu −0,0095 — e a causa dessa perda foi identificada
e é **estrutural e corrigível**, não informacional.

Além disso, o mecanismo de F1 foi validado de forma direta e forte, e a metodologia de diagnóstico do
projeto foi corrigida (§4), o que muda a leitura de várias decisões anteriores.

## 2. TS-AUC OOF (R0, bootstrap pareado por série, 300 réplicas)

| Bucket | V1 (pré-V2) | **V2** | Δ vs V1 | IC 95% | exclui 0 |
|---|---|---|---|---|---|
| geral | 0,5982 | **0,5997** | +0,0014 | [−0,0046, 0,0092] | não |
| t≤50 | 0,5220 | 0,5125 | −0,0095 | [−0,0291, 0,0126] | não |
| 50<t≤150 | 0,5670 | 0,5658 | −0,0012 | [−0,0107, 0,0121] | não |
| 150<t≤400 | 0,6139 | 0,6155 | +0,0016 | [−0,0063, 0,0105] | não |
| **t>400** | 0,6362 | **0,6504** | **+0,0105** | **[0,0030, 0,0187]** | **sim** |

Contra a linha de base *pré-auditoria*: geral +0,0001; t>400 +0,0079 (IC [−0,0008, 0,0176], não
exclui 0 por pouco).

**Previsão (b) da proposta — "ganho concentrado nos buckets ≥150" — confirmada.**

## 3. A perda em t≤50 é diluição por NaN, não perda de informação

As famílias novas são estruturalmente indisponíveis com poucos passos: janela não cheia, EWMA não
aquecida, escalas grossas de Haar precisam de 2^(j+1)·3 amostras.

| Bucket | NaN médio (features novas) | NaN médio (antigas) | features novas 100% NaN |
|---|---|---|---|
| t≤50 | **64,8%** | 3,1% | **14 de 37** |
| 50<t≤150 | 29,9% | 0,0% | 6 de 37 |
| 150<t≤400 | 4,7% | 0,0% | 0 |
| t>400 | 0,0% | 0,0% | 0 |

Em `t≤50` o modelo passou a ter 137 colunas das quais ~24 são puro nada, e `feature_fraction=0,8`
amostra 80% das colunas por árvore — a probabilidade de uma árvore ver as features que de fato
informam nesse bucket caiu. É o mecanismo mais simples e mais provável para o −0,0095, e é
**corrigível** (§6).

## 4. Correção metodológica: `mean|SHAP|` é a medida errada para esta competição

A proposta V2 §2 foi escrita citando "34,3% de |SHAP| em `meta_h0_*`" a partir de um CSV pré-existente
sem script gerador, referente ao modelo pré-auditoria. Ao escrever `scripts/shap_report.py` (TreeSHAP
exato, versionado) descobriu-se um problema mais fundo:

`mean|SHAP|` mistura variação **entre passos** com variação **entre séries**. A TS-AUC compara séries
*dentro do mesmo passo* — pela invariância C1, o que é constante dentro de um passo é **exatamente
neutro**. Logo a medida certa é a dispersão da contribuição dentro do passo, agregada com o peso
oficial w_t = n_pos·n_neg (coluna `xs_shap`).

A discordância entre as duas medidas é dramática e sistemática (modelo pré-V2):

| Família | XS% (correta) | conv% |
|---|---|---|
| `meta_h0_*` | **30,5%** | 17,8% |
| `accum` | 21,3% | 11,7% |
| `conformal` | 15,0% | **36,6%** |
| `cusum` | 14,2% | 6,6% |
| `meta_tempo`/locator | 4,6% | **16,9%** |

`meta_ln1p_t` cai de 11,9% para 2,7% e `meta_t` de 4,5% para 1,1% — exatamente como C1 prevê (o
resíduo é a parte de interação). `conformal` cai pela metade: os log-martingales crescem com t, e
essa variação temporal inflava a medida convencional. **Qualquer priorização feita com a coluna
convencional estava enviesada para features que apenas acompanham o relógio** — inclusive a leitura
do parecer de auditoria de que o bloco conformal era marginal (2,2%): pela medida transversal ele
vale 15,0%, e `conformal_logm_abs` é a 1ª feature do modelo pré-V2.

## 5. F1: mecanismo validado, previsão (a) confirmada

A previsão pré-registrada era: *"o share de SHAP de `meta_h0_*` cai substancialmente; se NÃO cair, a
hipótese de §2 está errada"*. O número agregado subiu (30,5% → 35,4%) — mas isso é confundido, porque
F2 **adicionou 9 features novas à própria família `meta_h0`**. Desconfundindo:

| | pré-V2 | V2 |
|---|---|---|
| `meta_h0_*` **originais (8)** | 30,5% | **14,5%** |
| `meta_h0_*` novas de F2 (9) | — | 20,9% |

**As features de condicionamento originais perderam mais da metade do seu peso transversal.** O
modelo deixou de reconstruir a calibração a partir de parâmetros crus de H0, porque as estatísticas
já chegam calibradas. Previsão (a): confirmada.

Evidência direta no nível da feature — **as versões `_cal` vencem as cruas em 15 de 24 pares**, e o
deslocamento é enorme onde importa:

| estatística | XS% crua | XS% calibrada |
|---|---|---|
| `accum_window_var_ln_w250` | 0,58% | **6,80%** (1ª do modelo) |
| `accum_window_var_ln_w100` | 0,22% | 3,92% |
| `mmd_joint_slow` | 2,20% | 3,33% |
| `ranktwo_dispersion_z_w100` | 0,78% | 1,71% |

E a premissa medida de forma independente dos modelos: a razão entre a escala nula de uma série GARCH
e de uma i.i.d. é **1,95–2,35× nas cruas e 0,86–1,14× nas calibradas**
(`tests/unit/test_calibration.py::test_calibration_equalizes_null_scale_across_series`).

**As famílias novas são usadas de verdade** (XS% no V2): `mmd` 10,9% · `haar` 5,9%. `mmd_joint_slow_cal`
— o detector não-paramétrico de quebra de *dependência*, que o banco não tinha — é a 7ª feature do
modelo.

## 6. Leitura honesta e próximo passo óbvio

F1 fez **exatamente** o que prometeu no nível do mecanismo: reorganizou um terço da atribuição do
modelo, as calibradas dominaram as cruas, e o condicionamento explícito substituiu o implícito. Só
que isso **não se converteu em ganho agregado de TS-AUC**. A leitura mais defensável: o
condicionamento que o modelo fazia implicitamente já era aproximadamente tão bom quanto o que agora
fornecemos de graça — os 30% não eram capacidade *desperdiçada*, eram capacidade fazendo trabalho
necessário, que agora ficou mais barato sem que o teto se mexesse.

O que **de fato** produziu ganho novo foi a injeção de informação nova (F3/F4) onde há dados para
estimá-la: `t>400`, +0,0105, significativo.

**Próximo passo, bem definido e barato:** atacar a diluição de §3, não adicionar mais famílias.
Concretamente: variantes de janela curta das novas famílias para o regime de t pequeno (MMD com λ
mais rápido; Haar só nas escalas finas; calibração para janelas menores), de modo a eliminar as 14
colunas 100%-NaN em `t≤50`. Previsão falsificável: recupera-se a perda de −0,0095 mantendo o ganho de
+0,0105, e o agregado passa a excluir 0. Se isso falhar, o peso da evidência desloca-se
definitivamente para H-informação (parecer §4.5) no regime de t pequeno.

---

# ADENDO — V3: correção da diluição (executada)

## A1. O que mudou

1. **Transporte de escala na calibração** (o principal). O bloqueio `t < min_t` das features `_cal`
   era conservador demais: para estatísticas com lei de escala conhecida, o que é idiossincrático da
   série é o *fator de inflação* sobre o nulo i.i.d. (k = dp_medido/dp_teórico), não o nível
   absoluto — e esse fator é ~constante em n. Agora o nulo é transportado para n = min(t, w) via
   `NullSpec.kind` (`z` | `var_ln` | `frac`), liberando a versão calibrada desde t≈10 em vez de
   t = w. Ver `state/calibration.py:_null_at`.
2. **Bug corrigido:** `haar_contrast_fine_mid` calculava `min_t` com a escala mais grossa
   (2⁵·3 = 96) em vez da escala 2 que ele de fato usa (2³·3 = 24) — mantinha a feature em NaN sem
   necessidade.
3. **λ muito rápido no MMD** (`lambda_vfast = 0,08`, janela efetiva ~12): a família passa a informar
   no regime onde `fast`/`slow` ainda não aqueceram. +2 features cruas, +2 calibradas.

Resultado estrutural: **colunas 100%-NaN em `t≤50` caem de 14 para 8**; NaN médio das famílias novas
nesse bucket cai de **64,8% para 42,4%**. As 8 remanescentes são limite de informação genuíno (Haar
grosso precisa de 96 amostras) ou o `_cal` do MMD lento — cujas versões *cruas* já estão disponíveis.

## A2. Resultado (R0, 300 réplicas)

| Bucket | pré-aud. | V1 | V2 | **V3** |
|---|---|---|---|---|
| **geral** | 0,5996 | 0,5982 | 0,5997 | **0,6039** |
| t≤50 | — | 0,5220 | 0,5125 | **0,5299** |
| 50<t≤150 | — | 0,5670 | 0,5658 | **0,5736** |
| 150<t≤400 | — | 0,6139 | 0,6155 | **0,6169** |
| t>400 | — | 0,6362 | 0,6504 | **0,6510** |

**A previsão se confirmou.** Isolando a correção (V3 vs V2): `t≤50` **+0,0174** — mais do que
recuperou a perda de −0,0095 — enquanto `t>400` ficou intacto (+0,0006). Todos os quatro buckets
melhoraram simultaneamente.

Contra a linha de base pré-auditoria, V3 é o melhor modelo já medido no projeto:

| Comparação | Δ geral | IC 95% | t>400 | IC 95% |
|---|---|---|---|---|
| V3 vs pré-auditoria | +0,0043 | [−0,0033, 0,0122] | **+0,0085** | **[0,0004, 0,0185]** ✓ |
| V3 vs V1 | +0,0057 | [−0,0013, 0,0132] | **+0,0112** | **[0,0033, 0,0194]** ✓ |
| V3 vs V2 | +0,0043 | [−0,0009, 0,0094] | +0,0006 | [−0,0043, 0,0055] |

## A3. Leitura honesta

**O que se confirmou:** a perda em `t≤50` era mesmo diluição estrutural, não limite de informação —
foi revertida por engenharia de disponibilidade de feature, sem tocar em nenhum detector. O ganho em
`t>400` é robusto e significativo em todas as comparações (+0,0085 a +0,0112, IC exclui 0).

**O que NÃO se confirmou:** a segunda metade da previsão — *"o agregado passa a excluir 0"*. O Δ geral
é +0,0043 com IC [−0,0033, 0,0122]: a melhor estimativa pontual do projeto, na direção certa, em
todos os buckets simultaneamente, mas **ainda indistinguível de zero pelo critério pré-registrado de
R0**. Pela regra de decisão do próprio parecer (§6-R0), isto **não autoriza** declarar vitória no
agregado; autoriza declarar o ganho em `t>400`, que é o único que passa no critério.

**Estado da hipótese "o gargalo são as features":** parcialmente sustentada. Injetar informação nova
(MMD/Haar) rendeu o único ganho significativo do projeto, mas concentrado onde há dados para
estimá-la. Em `t≤50` o ganho de V3 veio de *disponibilizar* features, não de informação nova, e o
bucket segue em 0,53 — consistente com um teto de informação real com 14 pontos pós-quebra na
mediana.

## A4. Conformidade

107 testes passam (era 104) · latência **854 µs/passo** (gate 1500) · CE6 **0,5030** com as features
de F2 incluídas (taxa-base 0,4967) → sem vazamento · equivalência online × vetorizado testada para
MMD (agora com 4 escalas de tempo) e Haar.

## 7. Conformidade com o protocolo

- 104 testes passam (era 80), incluindo causalidade (prefixo/canário) e determinismo bit-a-bit.
- Equivalência online × vetorizado testada explicitamente para MMD e Haar — era o risco real da
  arquitetura de calibração (o caminho vetorizado alimenta o nulo de F1).
- Latência: **802 µs/passo** (era 682), gate 1500 µs/passo → PASS.
- `fit_h0`: 32–68 ms/série (uma vez por série, ~40 s no total do build paralelo).
- **CE6 estendido para incluir as 9 features de F2** (de nada adiantaria continuar ≈0,5 apenas sobre
  as 19 antigas): **AUC 0,5030** sobre 10.000 séries, taxa-base 0,4967 — sem vazamento. Os novos
  descritores (Hurst, Hill, inclinação espectral, clustering de vol) **não** preveem a existência de
  quebra a partir do histórico, confirmando que continuam sendo condicionadores legítimos e não um
  atalho para o rótulo.

### Artefatos preservados para reprodução

- `artifacts/models/v1_preV2/` + `artifacts/models/oof_v1_preV2.parquet` — modelo anterior.
- `artifacts/reports/shap_preV2.csv` / `shap_v2.csv` — decomposições transversais comparáveis.
- `data/processed/train_rows_preV2.parquet` (~647 MB) — dataset de 91 features, guardado só para
  reanálise do modelo antigo (que não roda sobre o dataset novo, de 137 colunas). **Descartável**: o
  código que o gera está no git.
