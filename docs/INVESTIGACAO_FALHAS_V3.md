# Investigação rigorosa: onde o V3 falha e o que mais entregar ao modelo

**Data:** 2026-07-20
**Método:** cruzamento do censo A1 (magnitude de cada eixo de quebra por série,
`artifacts/reports/break_type_census.csv`) com o OOF do V3 (`artifacts/models/oof_v3.parquet`);
limites de Neyman-Pearson simulados contra as magnitudes reais; SHAP transversal por família; pesquisa
teórica externa. Nada implementado — este documento decide a direção antes de gastar ciclos.

---

## 0. Resposta em uma página

Três descobertas mudam a estratégia:

1. **O gargalo NÃO é informação, é extração.** Um detector ótimo de variância *que conhece tau e o
   tipo de quebra* atinge **AUC ≈ 0,856** contra a mistura real de quebras do gerador; o V3 está em
   **0,604**. Mesmo com um desconto pesado pelo custo causal (tau desconhecido, sinal parcial), a
   folga é grande — e concentrada em **t alto** (oracle 0,89 para m>200), exatamente onde você
   suspeitou que dava para melhorar. A minha afirmação anterior de que "t≤50 está perto do teto de
   informação" era **pessimista demais**: mesmo com m≤25 pontos o oracle é 0,775.

2. **O modelo é cego a dois eixos de sinal que existem no gerador.** Desconfundido por regressão
   multivariada, só a variância é um preditor forte de detectabilidade (β=+0,31). **Dependência**
   (β=+0,04, mas quebras puras de dependência dão detect=0,492 — *abaixo do acaso*) e **cauda/forma**
   (β=+0,05, detect=0,553) carregam sinal independente e estão praticamente inexplorados. Média está
   **definitivamente morta** (β=−0,005) — pode ser encerrada para sempre.

3. **A estrutura-no-tempo da variância (bipower/saltos) sozinha não basta** — você estava certo. Ela
   melhora *precisão* no eixo que já dominamos (separar GARCH de quebra real), não *recall* nos eixos
   cegos. Para mover o agregado é preciso atacar os três: localização da variância (o maior naco de
   folga), dependência não-linear/multi-lag, e forma de cauda robusta (L-momentos).

---

## 1. Onde o modelo falha — evidência empírica

Métrica: para cada série de quebra, o **percentil do seu score dentro da seção transversal de cada
passo** (exatamente o que a TS-AUC agrega), médio na janela pós-quebra madura (20–120 passos após
tau). 0,5 = indistinguível; →1 = bem ranqueado. 4.344 séries.

### 1.1 Detectabilidade por eixo de quebra (desconfundida)

Regressão OLS padronizada `detect ~ |Δlogvar| + |Δmean| + |Δρ₁| + |Δkurt|` (R²=0,110):

| Eixo | β independente | detect (tercil alto) | Leitura |
|---|---|---|---|
| **Variância** `|Δlogvar|` | **+0,312** | 0,667 | único eixo forte |
| Cauda/forma `|Δkurt|` | +0,052 | 0,619 | fraco, **independente** (corr 0,0 com variância) |
| Dependência `|Δρ₁|` | +0,043 | 0,580 | fraco, **independente** (corr 0,14) |
| Média `|Δmean|` | −0,005 | 0,584 | **morto** (o marginal +0,089 era confundido: corr 0,27 com variância) |

### 1.2 Os pontos cegos, isolados

Quebras onde a **variância não muda** mas outro eixo sim:

| Tipo de quebra | n | detect |
|---|---|---|
| Dependência pura (`|Δρ₁|` alto, `|Δlogvar|` baixo) | 437 | **0,492** (abaixo do acaso) |
| Cauda/forma pura (`|Δkurt|` alto, `|Δlogvar|` baixo) | 357 | 0,553 |

Perfil das 1.809 quebras mal detectadas (detect<0,5) vs. 1.757 bem detectadas (>0,65): a única
diferença material é `|Δlogvar|` (mediana 0,17 vs 0,35) e `|Δkurt|` (0,47 vs 0,71). O número de
pontos pós-quebra é **idêntico** (~230) — não é falta de dados, é falta de *feature* no eixo certo.

---

## 2. O teto de informação — a folga é real e está em t alto

Limite de Neyman-Pearson (estatística ótima, tau e tipo conhecidos), simulado contra as magnitudes
reais do censo.

### 2.1 AUC ótima por magnitude e nº de pontos pós-quebra (m)

**Shift de variância** (o eixo forte):

| Δlogvar | m=14 (t≤50) | m=45 | m=130 | m=300 |
|---|---|---|---|---|
| 0,08 (mediana) | 0,559 | 0,601 | 0,668 | 0,756 |
| 0,25 (p75) | 0,662 | 0,798 | **0,919** | **0,983** |
| 0,50 (p90) | 0,816 | 0,953 | 0,997 | 1,000 |

**Shift de dependência lag-1** (ponto cego):

| Δρ₁ | m=14 | m=45 | m=130 | m=300 |
|---|---|---|---|---|
| 0,20 | 0,663 | 0,814 | 0,944 | 0,993 |
| 0,30 | 0,746 | 0,913 | 0,991 | 1,000 |

Ou seja: uma quebra de dependência lag-1 de magnitude moderada é **altamente detectável** em teoria
(0,81–0,99 com janela média/longa), e o modelo entrega **0,492**. Não é limite de informação — é
feature ausente.

### 2.2 Teto agregado real

Detector ótimo de variância contra a **mistura real** de quebras (assumindo cada quebra como shift de
variância puro da sua magnitude observada, usando m = n_post e conhecendo tau):

| Faixa de m | n | AUC ótima |
|---|---|---|
| m≤25 | 283 | 0,775 |
| 25–75 | 668 | 0,798 |
| 75–200 | 1.200 | 0,839 |
| m>200 | 2.401 | **0,890** |
| **agregado** | 4.552 | **0,856** |

**Observado V3: 0,604.** O oracle é otimista (conhece tau e tipo, usa a janela pós-quebra inteira) —
o detector causal real paga um custo de localização e de não saber o tipo. Mas mesmo que metade da
folga de 0,25 seja irredutível, sobram ~0,10 extraíveis, **concentrados em t alto** (m>200 → 0,89).
24% das quebras têm `|Δlogvar|<0,1` (quase indetectáveis por variância — mas potencialmente
detectáveis nos eixos de dependência/cauda que o modelo ignora); 42% têm `|Δlogvar|>0,3` (fortes,
onde a folga de extração é máxima).

---

## 3. A hipótese que explica a folga de extração: diluição por tau desconhecido

Por que o modelo extrai 0,60 quando o oracle tira 0,86, se `accum_window_var_ln_w250_cal` é uma
estatística quase-suficiente para a variância e é a feature #1 (6,8%)? **Porque toda janela fixa
mistura pontos pré e pós-quebra.** Uma janela de 250 terminando em t, com tau no meio, estima uma
variância *atenuada* — média entre o regime antigo e o novo. O oracle usa só os pontos pós-tau.

O mecanismo que deveria resolver isso — estimar a variância *do regime mais recente*, localizado após
o changepoint mais provável — existe no filtro bayesiano (`bayes_map_var_ln`) mas rende só **3,4% de
SHAP**. Hipótese: o modelo de troca-única-gaussiana do filtro é mal-especificado para o gerador (que
tem cauda pesada e dependência), então sua estimativa de variância pós-quebra é ruidosa.

**Consequência estratégica:** o maior naco de folga não é um eixo novo — é **localização**. Uma
família de "variância do regime recente" (janelas ancoradas no changepoint estimado, ou um contraste
janela-curta-recente vs. janela-longa-defasada) ataca diretamente a diluição, no regime (t alto) onde
a folga é máxima. Isto é *expandir uma família existente* (variância) com uma variedade nova
(localizada), não um eixo novo.

---

## 4. Os eixos que valem entregar — além da estrutura-no-tempo da variância

### 4.1 Dependência não-linear e multi-lag (o maior ponto cego)

O banco só mede dependência **linear lag-1**: `accum_*_rho1_fz` (Fisher-z de ρ₁), `cusum_dep_*`
(produto defasado), `mmd_joint` (distribuição conjunta de (eₜ, eₜ₋₁)). SHAP transversal:

- `mmd_joint_*` (todas as variantes): ~10% — **carrega toda a dependência que é capturada**
- `accum_window_rho1_fz_w100`: **0,12%** · `cusum_dep_pos`: **0,16%** · `accum_global_rho1_fz`: 0,60%
  — os features lineares clássicos estão **mortos**

Mesmo o MMD joint (lag-1, não-paramétrico) não crava as quebras de dependência (detect ainda 0,492).
O que falta:
- **Dependência em |e| e e²** (clustering de volatilidade como *estrutura*, não nível) — ρ₁ de |e|,
  ρ de e² em múltiplos lags. Uma quebra pode mudar a *persistência* da volatilidade sem mudar seu
  nível médio, e nada no banco vê isso online (só o `meta_h0_acf_e2_l1` estático, da F2).
- **Multi-lag**: ρ₂, ρ₃, ou a soma de |ρ_k| até lag L (a "massa de dependência") numa janela.
- **Não-linearidade**: MMD conjunto sobre pares mais distantes (eₜ, eₜ₋₂), ou uma estatística tipo
  BDS de baixo custo.

Referência externa: GLR/focus estendido para AR com custo O(log n)
([Ward et al. 2026](https://arxiv.org/html/2607.16106)); testes de dependência serial não-linear para
erros não-gaussianos.

### 4.2 Forma de cauda robusta via L-momentos (o segundo ponto cego)

O banco mede cauda por *contagem de excedência* (`accum_*_exceed*`, `cusum_exceed_*`) e pelo conforme
`conformal_logm_abs` (4,53%, forte). Mas a *forma* da cauda — assimetria e peso — com variância
inalterada é fracamente captada (detect 0,553). Momentos clássicos (curtose) são péssimos em amostra
pequena e dominados por outliers.

**L-momentos** (combinações lineares de estatísticas de ordem) caracterizam forma/cauda com robustez
e baixa variância amostral, exatamente onde importa (t pequeno, cauda pesada). L-skewness (τ₃) e
L-kurtosis (τ₄) numa janela, contra os L-momentos do histórico, dão um eixo de forma que:
- é robusto a outliers (ataca de lado o falso-positivo T9),
- funciona em amostra pequena (ao contrário da curtose),
- é *ortogonal* ao nível de variância (mede forma, não escala).

Custo: O(w log w) por causa da ordenação — mais caro que o resto, mas viável com janelas ≤100 ou
reuso dos arrays já ordenados do bloco conformal. Referência:
[L-moments para caudas pesadas](https://arxiv.org/pdf/2306.09548) (change-point heavy-tailed).
Nota de precedente interno: o `meta_h0_hill_xi` (índice de cauda de Hill) que adicionei na F2 já
rende **2,13% de SHAP** — evidência de que o eixo de cauda *estático* informa; falta a versão
*dinâmica* (mudança de cauda online).

### 4.3 Estrutura-no-tempo da variância / saltos (a família que você já gostou — precisão, não recall)

Bipower variation (RV vs BV) e semivariância/leverage, como discutido. Registro honesto do seu papel
após esta análise: **melhora precisão** (separa GARCH de quebra real, ataca T6/T9), não recall nos
eixos cegos. É complementar, não substituto, das §4.1–4.2. Mantém-se na proposta, mas re-priorizado
abaixo de dependência.

---

## 5. Expansão das famílias existentes — oportunidades concretas

| Família | Estado (SHAP XS) | Oportunidade de expansão |
|---|---|---|
| **Dependência** (rho1/dep) | lineares mortos (0,1–0,6%); só MMD-joint vive (~10%) | **Alta.** Multi-lag, dependência em \|e\|/e², não-linear. §4.1 |
| **Cauda/excedência** | conformal_abs forte (4,5%); contagens fracas | **Alta.** L-momentos dinâmicos; forma robusta. §4.2 |
| **Variância** | forte, mas diluída por tau | **Alta.** Variância localizada no changepoint. §3 |
| **Calibração F1** | 26 twins, dominam as cruas | **Média.** Estender a dependência/exceedance (hoje fora): `cusum_dep`, `rho1_fz` NÃO são calibrados — e são os mortos |
| **Bayes** | 3,4%, subutilizado | **Média.** t-likelihood (parked; censo mostra ν̂ baixo em fração relevante → critério de reabertura atendido?) — melhoraria a localização §3 |
| **Conformal** | p-values ótimos, agregador fraco (D-8) | **Média-baixa.** Martingale de potência em vez do Vovk-mixture; risco de "substituir" como no R4 |
| **MMD** | joint carrega dependência | **Baixa.** Multi-bandwidth (multi-kernel) ou pares mais distantes — barato, testável |
| **Haar / multiescala** | 5,9%, usado | **Baixa.** Já cobre estrutura de escala; expandir wavelet dá pouco |
| **CUSUM / média** | grade de média = unidades de inovação | **Nenhuma.** Média morta (§1); grades já julgadas |

**Achado de manutenção importante:** os features de dependência mortos (`cusum_dep`,
`accum_window_rho1_fz`) **não recebem calibração F1** — só variância/rank/mmd/haar recebem. Parte da
sua morte pode ser miscalibração transversal (um ρ₁ de 0,2 significa coisas diferentes em séries com
persistências diferentes). Estender F1 a eles é barato e testável isoladamente.

---

## 6. Recomendação priorizada

Ordenada por (folga de extração medida × independência do eixo) / custo. Cada item com previsão
falsificável, julgado por R0, CE6 + robustez a cada iteração.

**P1 — Dependência não-linear/multi-lag + calibração (maior ponto cego, β independente, detect<0,5).**
Novo bloco: ρ₁ de |e| e e² em janelas, massa multi-lag Σ|ρ_k| (k≤5) sobre e e |e|, MMD conjunto
lag-2. Estender F1 a todos. *Previsão:* ganho concentrado nas 437 séries de dependência pura e no
bucket 150–400; se detect dessas séries não subir acima de 0,55, o eixo é informacionalmente mais
fraco do que a teoria §2.1 sugere.

**P2 — Forma de cauda dinâmica via L-momentos (segundo ponto cego, ortogonal à variância).**
L-skewness/L-kurtosis de janela vs. histórico, reusando arrays ordenados do conformal; calibrados.
*Previsão:* ganho nas 357 séries de cauda pura e sinergia com o `conformal_logm_abs` já forte.

**P3 — Variância localizada no changepoint (a maior folga de extração, §3, t alto).**
Contraste variância-recente-curta vs. variância-defasada-longa; ou variância ancorada na idade do
CUSUM/Bayes MAP. *Previsão:* ganho em t>400 (onde o oracle mostra 0,89 vs 0,65 observado).

**P4 — Bipower/saltos + leverage (precisão, ataca T6/T9).** Como já especificado. Re-priorizado abaixo
dos eixos de recall.

**Meta-observação sobre o agregado.** Mesmo executando P1–P4, o agregado pode continuar dentro do
ruído do R0, porque ~24% das quebras são fracas em todos os eixos e ~35% do peso está em t pequeno com
teto causal baixo. Os ganhos aparecerão **por bucket e por tipo de quebra** (t>400, dependência pura,
cauda pura) antes de aparecerem no número único. O árbitro correto para um ganho agregado pequeno mas
real passa a ser a **sonda oficial** (âncora D4-iii), não mais o OOF — que não resolve um Δ de 0,004.

---

## 7. O que NÃO fazer (reafirmado pela evidência)

- **Qualquer coisa no canal de média** — β=−0,005 multivariado, encerrado por medição direta, não só
  por teoria.
- **Mais funcionais do mesmo contraste de CDF** (JS/Hellinger/W1 = F6; mais shape_chi2) — os
  `ranktwo_shape_chi2` já rendem 0,00–0,02%; o eixo de forma-por-CDF está saturado.
- **Força bruta de features** (abordagem 2º lugar 2025) — n_eff≈10⁴; máquina de overfitting de
  seleção.
- **Reabrir grades de CUSUM/hazards/meta_h0** — já julgadas nulas duas vezes.

---

## Fontes

- [Efficient LR test for online changepoint under autocorrelation (2026)](https://arxiv.org/html/2607.16106)
- [Online heavy-tailed change-point detection (L-moments / robust)](https://arxiv.org/abs/2306.09548)
- [Power & bipower variation with stochastic volatility and jumps — Barndorff-Nielsen & Shephard](https://academic.oup.com/jfec/article-abstract/2/1/1/960705)
- [CHASM: online changepoint in temporal & cross-variable dependence (2026)](https://arxiv.org/pdf/2605.07852)
- [NEWMA: model-free online change-point detection](https://arxiv.org/abs/1805.08061)
