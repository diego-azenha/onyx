# O modelo: fundamentos, ideia original e decisões de projeto

**Projeto:** `structural-break-rt` (`onyx`) — detector causal de quebra estrutural em tempo real para o
ADIA Lab Structural Break Challenge: Real-Time Edition (CrunchDAO), 2026.
**Papel deste documento:** fonte da verdade sobre *o que* o sistema é e *por que* foi desenhado assim
(formulação, fórmulas, hiperparâmetros, gates, invariantes). Substitui `PLANO_TECNICO.md` +
`PLANO_REPOSITORIO.md` (parte conceitual).
**Numeração:** as seções `§N` abaixo preservam a numeração do plano técnico original, porque
docstrings, comentários de código e `configs/default.yaml` referenciam `plano §N` em dezenas de
lugares. `§3.4`, `§9.0`, `§13.2` etc. continuam resolvendo aqui.

Companheiros: [`HISTORICO.md`](HISTORICO.md) (o que mudou e o que rendeu) ·
[`NOTAS_AGENTES.md`](NOTAS_AGENTES.md) (layout, contratos, comandos, armadilhas operacionais).

---

## §0. Sumário executivo

**O problema.** Por série: um histórico `H` de 1000–5000 pontos, **livre de quebra por definição** e
z-scorado, seguido de um segmento online `x_1..x_T` (T ∈ [10, 1000]) revelado **um ponto por vez**.
Com probabilidade ½ existe um único `τ ∈ {1..T}` a partir do qual o DGP muda permanentemente
(abrupta ou suavemente); com probabilidade ½ não há quebra. Após cada `x_t` o algoritmo emite
`s_t ∈ [0,1]`, sob causalidade estrita.

**A abordagem.** Uma **máquina de estados causal por série** (O(1) a O(K) por passo) transforma cada
observação em um vetor de estatísticas sequenciais suficientes, todas atualizadas incrementalmente,
e um **calibrador LightGBM** treinado sobre o rótulo por passo `y_t = 1{τ ≤ t}` funde tudo. Quatro
mecanismos clássicos complementares formam o núcleo:

1. **Whitening causal pelo modelo H0** (AR(p) + escala robusta + termo sazonal opcional), ajustado
   uma única vez no histórico: converte "detectar qualquer mudança de DGP" em "detectar mudança na
   distribuição das inovações padronizadas" e neutraliza o falso positivo clássico por
   autocorrelação.
2. **Banco de CUSUMs** (média ×3 magnitudes ×2 lados, variância ↑/↓, sinal, excedência de cauda,
   dependência lag-1) — acumuladores O(1) minimax-ótimos, aproximando o GLR por grade.
3. **Filtro bayesiano de troca única** (Shiryaev/BOCPD restrito a ≤1 mudança, K=48 candidatos,
   hazards múltiplos) — produz diretamente `P(τ ≤ t | dados)` e exporta de graça estatísticas de
   duas amostras "desde o τ̂ mais provável".
4. **Martingales conformais** sobre p-values causais das inovações contra o histórico — evidência
   livre de distribuição, robusta a caudas pesadas, O(log n_h)/passo.

Janelas rodantes (10–250) e EWMAs cobrem quebras suaves e recentes. O LightGBM aprende, dos ~10.000 τ
rotulados do treino, (i) a combinação ótima entre famílias, (ii) a recalibração dos priors do gerador
e (iii) o condicionamento por *nuisances* da série que garante **comparabilidade transversal do score
entre séries no mesmo passo** — exatamente o que a TS-AUC mede.

> Famílias adicionadas depois da concepção original (MMD/RFF, Haar multiescala, calibração por nulo
> de série, dependência não-linear, variância localizada, saltos, BOCPD) estão em §5.2 e sua
> justificativa empírica em `HISTORICO.md`.

**Como resolve os três desafios centrais:**

- **Causalidade estrita** — por construção: não existe caminho de código que veja `t' > t`. Um único
  motor sequencial (`StreamScorer.update(x_t) → score`) gera tanto a inferência quanto as features de
  treino (**princípio do motor único**), e o teste de prefixo com canário prova que o detector de
  vazamento morde (§12.1).
- **Posição de quebra desconhecida** — tratada três vezes: recursão max do CUSUM (varre todos os τ
  candidatos a O(1)), soma bayesiana explícita sobre candidatos podados, e o supervisionado que
  aprende a distribuição real de τ do gerador.
- **Saída compatível com TS-AUC** — alvo de treino é o rótulo *por passo*, com pesos de linha
  alinhados ao perfil `w_t` da métrica; performance é decidida por submissão oficial, não por métrica
  local (§9).

**Orçamento.** ~974 µs/passo medido (gate 1500), com folga confortável sobre o limite semanal da
plataforma. Determinismo garantido por ausência total de aleatoriedade na inferência, ordem fixa de
operações e `num_threads=1` no predict (§15.2).

---

## §1. Formulação probabilística e leitura estrutural da TS-AUC

### §1.1 Definições

**Rótulo por passo:** `y_{i,t} = 1{τ_i ≤ t}`. Uma série com quebra *futura* (τ_i > t) é **negativa**
no passo t; vira positiva a partir de τ_i. Séries sem quebra são negativas em todos os passos.

**Métrica:** `TS-AUC = Σ_t w_t·AUC_t / Σ_t w_t`, onde `AUC_t` é a AUC transversal no passo t sobre as
séries vivas (T_i ≥ t) e `w_t = n_pos(t)·n_neg(t)`. Passos de classe única têm peso zero.

**Premissas registradas:** **A1** séries com quebra ainda não ocorrida contam como negativas;
**A2** séries são alinhadas pelo índice do passo online; **A3** `T_i` é desconhecido durante a
inferência — **nenhuma feature pode usar T**.

### §1.2 Três consequências estruturais (usadas o tempo todo)

**(C1) Invariância por passo.** `AUC_t` é invariante a qualquer transformação estritamente monótona
aplicada uniformemente a todos os scores do passo t. Corolários: (a) um componente de score que
depende **só de t** é **neutro**; (b) o perigo real não é a subida uniforme, é a **subida heterogênea
entre séries** (travamento de ruído, drift dependente de artefato), que reordena séries dentro do
passo. C1 mata, por impossibilidade matemática, qualquer "ganho" via recalibração pós-hoc do score
(Platt/isotônica/readição de offset) — ver `HISTORICO.md`, D1.

**(C2) Alvo ótimo = posterior.** Por passo, o ranqueador que maximiza `AUC_t` é qualquer
transformação monótona da razão de verossimilhança entre `{τ ≤ t}` e `{τ > t ∨ sem quebra}` — ou
seja, o posterior `P(τ ≤ t | x_{1:t})`. Logo (i) o alvo de treino correto é o rótulo por passo, não o
da série; (ii) o posterior é **não-monótono em t por natureza** (evidência transitória deve decair) —
fundamento de §7.

**(C3) Perfil de pesos.** `w_t` é pequeno nos primeiros passos (quase não há positivos), pequeno nos
últimos (poucas séries vivas), e máximo no horizonte intermediário. Detecção "instantânea" em 1–3
passos vale pouco peso; o regime que paga é **10–200 pontos pós-quebra**. Os pesos de linha do treino
replicam `w_t` medido **empiricamente** no treino, não a fórmula idealizada (§8.2).

**Reformulação operacional (da auditoria, §3.1 do parecer):** como `AUC_t = concordâncias/(n_pos·n_neg)`
e `w_t = n_pos·n_neg`, os pesos cancelam:

> **TS-AUC = fração de pares (positivo, negativo) do mesmo passo corretamente ordenados, agregada
> sobre todos os passos.**

É a lente correta para decidir pesos de linha, objetivo de treino e medida de importância de feature.

### §1.3 Decomposição em três blocos

1. **Fase-histórico** (1× por série, antes do 1º passo): ajustar H0, pré-computar constantes,
   calibrar nulos por série. O(n_h·p + n_h log n_h). Fora do caminho crítico.
2. **Fase-online** (1× por observação): atualizar estado, montar features, prever.
3. **Fase-treino** (offline, 1×): repassar as 10.000 séries pelo MESMO motor, coletar
   (features, y_t, peso) por passo, treinar LightGBM com GroupKFold por série, congelar o ensemble.

---

## §2. Alternativas exploradas, veredictos e bloqueios

| Linha | Mecanismo | Veredicto |
|---|---|---|
| L1 | Bayes de troca única (Shiryaev; BOCPD-1; Fearnhead–Liu) | **Núcleo** como família de features, não como score final |
| L2 | CUSUM / Page-Hinkley / GLR window-limited (Lai 1998) | **Núcleo** (banco de 15 CUSUMs + janelas) |
| L3 | Cartas de controle / janelas rodantes / EWMA | **Suporte** |
| L4 | Supervisionado sequencial (LightGBM sobre features incrementais) | **Núcleo como camada de fusão** |
| L5 | Martingales conformais (Vovk; Volkhonskiy) | **Suporte** (p-values valem mais que o agregador — §5.2) |
| L6 | Foundation models / deep forecasting por passo | **Bloqueada** (custo + evidência pública negativa) |

**Bloqueios com números:**

- **B1 — GLR exato sobre todos os candidatos:** O(t)/passo ⇒ ~5·10⁹ atualizações por conjunto de
  teste ⇒ horas só neste componente. Substituído por janelas rodantes (= GLR window-limited) + filtro
  bayesiano podado.
- **B2 — Refit adaptativo do baseline no online:** o baseline adaptativo **absorve a própria quebra**
  (vol-EWMA λ=0,06 converge à nova variância em ~17 passos — CE2). Bloqueado como política geral;
  whitening adaptativo permitido **apenas** para média/dependência/forma (§3.4).
- **B3 — Transformers/foundation models por passo:** ~10¹⁴ FLOPs, risco de não-determinismo, e
  evidência pública de ≈0,5 AUC no domínio.
- **P1 (estacionada) — GRU causal em numpy:** cabe no orçamento (~20 µs/passo), mas custo de
  desenvolvimento alto e evidência pública desfavorável. Reabrir só em platô com folga de cronograma.
- **P2 — p-values por permutação/bootstrap:** multiplica o custo por ≥200×. Substituído pelos
  martingales conformais (p-values causais exatos sem reamostragem).

**Evidência externa que sustenta o paradigma** ("estatísticas suficientes + árvores supervisionadas"):
o vencedor da edição batch 2025 usou stacking de árvores sobre blocos de features estatísticas
(~0,90 AUC privado); redes puras ficaram em ≈0,5 (§16).

---

## §3. Fase-histórico: caracterização do H0 e whitening causal

Executada uma vez por série sobre o histórico completo (permitido: o histórico é dado de uma vez e é,
por definição, livre de quebra — é referência, nunca objeto de teste). Tudo determinístico.

### §3.1 Estimativas do H0

1. **Momentos:** μ̂₀, ŝ₀, ŝ_rob = 1,4826·MAD. Nunca assumir μ=0/σ=1 nominais — sempre estimar.
2. **AR(p), p=10**, aceito se `var(resíduo)/var(x) ≤ 0,98`; caso contrário φ := 0.
3. **Checagem sazonal parcimoniosa:** ACF dos resíduos até lag 128; se |ρ(L)| > 0,25 com L ∈ [6,128],
   reajustar com lags {1..10, L}. No máximo um termo sazonal.
4. **Escala de inovação:** σ̂_e e σ̂_e,rob.
5. **Caudas:** ν̂ = clip(4 + 6/κ̂_ex, 5, 50); quantis de e em 8 níveis.
6. **Dependência residual:** ρ̂₁(e), ρ̂₁(|e|), σ̂_u (escala do detector de dependência).
7. **Arrays ordenados** das inovações do histórico → p-values conformais por busca binária.
8. **Buffers iniciais** (`lag_seed`): garantem **continuidade exata** na fronteira histórico→online
   (armadilha §13.3).
9. **Impressão digital estendida** (F2, adicionada depois): Hurst, índice de cauda de Hill,
   persistência tipo GARCH, perfil de decaimento da ACF, inclinação espectral, vol-of-vol. Custo zero
   por passo.
10. **Nulos por série** (F1, adicionada depois): deslizar as MESMAS estatísticas online sobre o
    próprio histórico e guardar sua distribuição nula por série (média/desvio, com encolhimento para
    o nulo teórico i.i.d.). Online, além de `S_t`, emite-se o **z de S_t contra o nulo da própria
    série** (`*_cal`). É o que torna as estatísticas comparáveis **entre séries** — a propriedade que
    a TS-AUC premia. Ver `state/calibration.py`.

### §3.2 Whitening causal no online

A cada passo: `x̂_t = ĉ + Σ φ̂_j·x_{t−j}` (lags do ring, que atravessa a fronteira);
`e_raw = (x_t − x̂_t)/σ̂_e`; `e = clip(e_raw, −8, +8)`. Os **indicadores de excedência crus**
(`|e_raw| > q₉₅, q₉₉, 6`) preservam o sinal de cauda antes do clip. **Parâmetros do H0 nunca são
reestimados no online** (bloqueio B2) — `H0Params` é `frozen` justamente para tornar isso
estruturalmente impossível.

### §3.3 Por que whitening é obrigatório

Sob H0 verdadeiro, `e_t ≈ i.i.d.(0,1)`-ish. Sem whitening, um AR(1) com φ=0,9 infla a variância de
médias amostrais por (1+φ)/(1−φ) = 19×, ou seja, z-scores ~4,4× maiores que o nominal — e todo
detector de média dispara em série *sem* quebra (CE4). O whitening **preserva todas as famílias de
quebra**: mudança de média/variância/dependência/forma de `x` aparece como mudança correspondente em
`e`.

> **Nota verificada (auditoria §4.2):** para AR(1) com marginal unitária, o z por δ√m é
> √((1−φ)/(1+φ)) **idêntico** nas inovações e na média crua. A atenuação do canal de média é limite
> de informação do problema, **não culpa do whitening**. Não há bug de whitening a caçar no canal de
> média — essa linha está encerrada.

### §3.4 Segundo fluxo de inovações — com trava anti-absorção

Se ρ̂₁(|e|) > 0,15 (série tipo GARCH), mantém-se `ẽ_t = e_t/√v_t` com `v ← (1−λ_v)v + λ_v e²`,
λ_v = 0,06. **Regra rígida:** `ẽ` (= `e_vol`) alimenta apenas **média/dependência/forma**; as famílias
**variância e cauda usam sempre `e` com escala congelada do histórico** — senão o EWMA-vol absorve a
quebra de variância em ~17 passos e a cega (CE2).

> **Trade-off consciente (auditoria §4.3):** essa trava é a causa raiz do falso positivo GARCH (T6).
> É o lado certo do trade-off, porque quebras de variância são o sinal dominante do gerador (censo
> A1: 41,8% das séries com |Δlogvar|>0,3 vs. 6,8% com |Δmean|>0,3). O resíduo se trata **no
> calibrador** (discriminadores de "burst vs. patamar": `volvol`, contrastes multiescala Haar,
> bipower/saltos), nunca adaptando o baseline.

---

## §4. Modelo de estado e atualização incremental

Estado por série, todo float64, nenhuma atualização recomputa sobre o prefixo — tudo é recursivo.

### §4.1 Campos fixos (pós fase-histórico)
`phi[10], c, sigma_e, sigma_e_rob, nu_hat, q[8], sorted_e_hist, sorted_abs_e_hist, sigma_u, rho1_e,
rho1_abs_e, ar_r2, seasonal_lag, seasonal_coef`, impressão digital estendida e os nulos de calibração.

### §4.2 Blocos dinâmicos (por passo)

| Bloco | Estado | Atualização |
|---|---|---|
| Lags / contador | ring `xlag[10]`, `last_e`, `t` | push O(1) |
| Welford global | `n, mean_e, M2` | recursão de Welford (estável em 10³ passos) |
| EWMAs | média/variância/sinal/excedência, λ ∈ {0,05; 0,10; 0,30} | `m ← (1−λ)m + λ·x`; z = m/√(λ/(2−λ)) |
| Janelas | ring de 256; w ∈ {10, 25, 50, 100, 250} | somas incrementais O(1); w efetivo = min(w,t) no warm-up |
| Banco CUSUM (15 + 15 idades) | média δ∈{0,25;0,5;1,0} ±, variância ↑{1,5;2,5}/↓0,5, sinal ±, excedência q₉₅/q₉₉, dependência ± | recursão `max(0, S + incremento LLR)`; idade = passos desde o último zero |
| Filtro bayesiano ×3 hazards | K=48 candidatos com (n_j, mean_j, M2_j, logw_j) | §4.3 |
| Martingales conformais | logM abs/direita/sinal, acumulado e com reset, mistura de ε ∈ {0,05…0,4} | p_t por bisect O(log n_h); `L_ε += ln ε + (ε−1)ln p_t` |
| Famílias posteriores | rank two-sample, MMD/RFF, Haar multiescala, dependência não-linear, variância localizada, saltos, BOCPD | §5.2 |
| Calibração | nulo por série de cada estatística registrada | emite `*_cal` (z contra o nulo da própria série) |

Warm-up: features indefinidas emitem **NaN** (LightGBM trata NaN nativamente — nunca sentinelas
mágicas). `features.warmup_min_n = 5`.

### §4.3 Filtro bayesiano de troca única

Sob H0, `e_t ~ N(0,1)`; pós-mudança, `e_t ~ N(μ,σ²)` com prior conjugado Normal-Inv-χ²
(μ₀=0, κ₀=0,5, ν₀=2, σ₀²=1,5). Hazard constante, sem morte de regime. Por passo: nasce o candidato
k=t (a quebra pode ocorrer já no 1º ponto observado — confirmado pelo changelog W23/2026 da
organização); todos os candidatos vivos são atualizados pela preditiva t de Student do NIχ²; **poda**
mantém K=48 protegendo sempre os **8 mais recentes** (candidatos jovens têm pouca evidência e seriam
podados injustamente); **renormalização em log-espaço** quando o máximo excede 600 em módulo.
Saídas: `LO_h = logsumexp(logw_j) − logw0` (log-odds de "quebra já ocorreu"), `τ̂_MAP`, `age_MAP` e as
estatísticas do candidato MAP (duas amostras com janela auto-selecionada, de graça).

---

## §5. Taxonomia de features

### §5.1 Banco original (~78 features, agrupado por família)

| Família | Representantes | Ref | Sensível a |
|---|---|---|---|
| **Média** | z-Welford global; z-EWMA ×3λ; z de janela ×5w; CUSUM média ±×3δ | G/E/W | shift de nível (persistente ou recente) |
| **Variância** | ln v_λ ×3; ln(Q_w/w) ×5; CUSUM var ↑1,5/↑2,5/↓0,5; ln(M2/n) global | G/E/W | mudança de σ² |
| **Cauda** | frações de excedência (janela/EWMA/global); Bernoulli-CUSUM q₉₅/q₉₉; máx \|e_raw\| | G/W + H | engrossamento de cauda, extremos |
| **Média robusta** | Bernoulli-CUSUM de sinal ±; z de proporção de positivos (w=50, 250) | G/W | shift de mediana sob cauda pesada |
| **Dependência** | CUSUM de dependência ±; ρ̂₁ online e de janela (Fisher-z) | G/W | mudança de φ |
| **Forma** | quantile-crossing vs. quantis do histórico; assimetria incremental de janela | W + H | achatamento/assimetria |
| **Bayes** | LO por hazard; age_MAP; stats do candidato MAP | G + H | qualquer mudança de (μ,σ²) com τ desconhecido |
| **Conformal** | log-martingales abs/direita/sinal, acumulado e com reset | G + H | violação de exchangeability vs. histórico |
| **Localização** | idades dos CUSUMs; concordância entre localizadores | G | quebra real (concordam) vs. ruído |
| **Hedge** | EWMA de (x−μ̂₀) e ln var de janela **sem** whitening | E/W | seguro contra AR mal ajustado |
| **Meta** | `t`, `ln(1+t)`; descritores do H0 (n_h, ν̂, ρ̂₁, ar_r2, q₉₉, …) | — / H | condicionamento e comparabilidade transversal |

Por que não dispara com ruído normal: (i) tudo roda sobre inovações whitened padronizadas pela escala
do histórico; (ii) famílias robustas duplicam as paramétricas para que cauda pesada legítima não vire
quebra; (iii) as meta-features permitem aprender "desconte o CUSUM de variância quando ρ̂₁(|e|) do
histórico é alto"; (iv) nenhuma feature usa o futuro, o conjunto de teste ou `T`.

### §5.2 Famílias adicionadas depois (justificativa empírica em `HISTORICO.md`)

| Bloco | Arquivo | Ideia | Custo |
|---|---|---|---|
| **Calibração F1** | `state/calibration.py` | z de cada estatística contra o **nulo da própria série** medido no histórico (`*_cal`). Torna séries heterogêneas comparáveis na seção transversal | 0 µs/passo |
| **Digital H0 F2** | `state/fingerprint.py` | Hurst, Hill, persistência GARCH, perfil de ACF, inclinação espectral, vol-of-vol | 0 µs/passo |
| **MMD/RFF F3** | `state/mmd.py` | MMD de kernel via Random Fourier Features (D=64, `W,b` sorteados **uma vez com seed fixa e compartilhados por todas as séries** — obrigatório para comparabilidade). EWMAs em 3 velocidades + estatístico NEWMA. Versão **conjunta** sobre (e_t, e_{t−1}) = detector não-paramétrico de mudança de **dependência** | ~9 µs |
| **Haar multiescala F4** | `state/multiscale.py` | Cascata diádica causal; energia por escala e **contrastes entre escalas**. Discrimina "patamar novo de variância" (sobe em todas as escalas) de "burst GARCH" (sobe nas finas) | ~4 µs |
| **Rank two-sample R4** | `state/rank_twosample.py` | Wilcoxon/dispersão/χ²-de-forma de janela contra o histórico, sobre os p-values conformais | baixo |
| **Dependência P1** | `state/dependence.py` | ρ₁ de \|e\| e e² (persistência de vol como *estrutura*, não nível), massa multi-lag Σρ_k² | baixo |
| **Variância localizada P3** | `state/varloc.py` | max/min do z de variância sobre escalas + contraste recente-vs-defasado — ataca a **diluição por τ desconhecido** (janela fixa mistura pré e pós-quebra) | baixo |
| **Saltos P4** | `state/jumps.py` | bipower/RV, semivariância, leverage — ganho de *precisão* (separa GARCH de quebra real), não de recall | baixo |
| **BOCPD** (estacionado) | `state/bocpd.py` | Posterior completo sobre run-length (Adams–MacKay, R_max=256): variância do regime **atual**, prob. de changepoint recente, entropia da localização. Versão principiada de `varloc` | ~30 µs |
| **L-momentos P2** | `state/lmoments.py` | L-skewness/L-kurtosis de janela vs. histórico — forma de cauda robusta em amostra pequena, ortogonal ao nível de variância | ~65 µs |

---

## §6. Estratégia para posição de quebra desconhecida

| Mecanismo | Como varre τ | Custo | Emite prob.? | Usa os τ do treino? |
|---|---|---|---|---|
| CUSUM (recursão max) | implícita e exata p/ alternativa simples | O(1) | não | não |
| GLR exato | max explícito sobre k | O(t) — **bloqueado** | não | não |
| Bayes de troca única | soma ponderada sobre k ≤ t (podada) | O(K) | **sim** | só via hazard |
| Supervisionado por passo | aprende `P(τ≤t \| features)` | O(predict) | sim | **sim — única linha que usa** |

**Decisão: híbrido em camadas.** CUSUMs e janelas dão a varredura O(1) por família; o filtro bayesiano
dá a integração probabilística sobre τ e magnitude no núcleo (μ,σ²) mais a localização τ̂; o
supervisionado funde tudo e injeta a única informação que nenhum método clássico tem — a distribuição
real de τ, de tipos e de magnitudes do gerador, presente nos 10.000 rótulos.

---

## §7. Monotonicidade do score — decisão: **score livre**

Sem retenção (max-hold). Fundamentos:

1. O alvo ótimo é o posterior (C2), **não-monótono por natureza**: evidência transitória sobe e é
   corretamente revertida.
2. **CE1 (contraexemplo do max-hold):** série sem quebra, T=1000, outlier de 6σ em t=15. Com
   `s_t = max(s_{t−1}, p_t)`, a série fica travada em ~0,6–0,7 pelos 985 passos restantes — e ela é
   **negativa em todos eles**. Como 50% do universo é sem quebra, o custo esperado é de primeira ordem.
3. O argumento "quebra é permanente, a confiança não deveria cair" já é capturado sem retenção: pós
   quebra real, a evidência é recorrente e os acumuladores crescem sozinhos.
4. Por C1, qualquer piso uniforme em t é neutro; só a retenção **por série** muda o ranking — e ela
   retém tanto sinal quanto ruído.

Variantes implementadas em `postprocess/monotonicity.py`: `free` (default), `hold`, `soft`, `ema`.
`V-ema` é a única com hipótese plausível (redução de variância do predict) — testável de graça quando
sobrar uma sonda oficial, nunca antes. Assunto **encerrado** pela auditoria.

---

## §8. Camada supervisionada

### §8.1 Dataset (motor único)
Para cada série de treino: fase-histórico + **o mesmo `StreamScorer` da submissão**, passo a passo,
coletando features e `y_t = 1{τ ≤ t}`. Não existe implementação vetorizada paralela — isso elimina
por construção a classe de bug "backtest vetorizado ≠ execução causal" (§13.2).
**Thinning com correção de peso:** todos os passos t ≤ 100; 101–400 a cada 2 (peso ×2); >400 a cada 4
(peso ×4). Volume real: **2.541.134 linhas**. Paralelizável **entre séries** (`n_jobs`) — nunca dentro
de uma série.

### §8.2 Pesos de linha
`w_row(i,t) = ŵ(t)/n_alive(t) × fator_thinning`, com `ŵ(t) = n_pos(t)·n_neg(t)` medidos
empiricamente no treino, normalizado para média 1.
Variante **pareado-consistente** (R1, implementada): `w_pos(t) ∝ n_neg(t)`, `w_neg(t) ∝ n_pos(t)`,
suavizada e capada — deriva da reformulação "fração de pares concordantes" (§1.2). Testada: efeito
estatisticamente nulo (`HISTORICO.md` §4).

### §8.3 Modelo
LightGBM binário, 1 modelo por fold de **GroupKFold(5) por `id` da série** — obrigatório: linhas da
mesma série são fortemente autocorrelacionadas e qualquer split não agrupado infla o CV
catastroficamente (§13.6). Predição final = média das probabilidades dos folds.

Hiperparâmetros (todos em `configs/default.yaml`): `learning_rate 0,05`, `num_leaves 63`,
`min_data_in_leaf 200`, `lambda_l2 5,0`, `feature_fraction/bagging 0,8`, `max_bin 255`,
early stopping paciência 100, `deterministic=true`, `force_row_wise=true`, `predict num_threads=1`,
`seed=42`.

**`init_score = logit(p̂(t))`** (`model/base_rate.py`) e métrica de parada `binary_logloss`: a
taxa-base de `y_t` cresce fortemente com t (neutra para TS-AUC por C1) e dominava o early stopping.
A curva de taxa-base é persistida no artefato; `predict_one` **não** readiciona o offset (C1-neutro e
melhor para a suíte de robustez).

> **Fato importante de capacidade:** o **n efetivo é ~10⁴ séries, não 2,5M linhas** — o rótulo inteiro
> de uma série é determinado por um único τ_i. Toda intuição de hiperparâmetro calibrada para "2,5M
> linhas i.i.d." deve ser relida. A saturação medida em 60–90 árvores é exatamente o esperado.

### §8.4 Custo de inferência
Escada de mitigação se a latência estourar: reduzir folds → teto de árvores → compilar com
lleaves/treelite. **Nunca** reduzir a frequência de emissão do score (a métrica exige um score por
passo).

### §8.5 Fallback puro-estatístico
`score = σ(0,9·LO_{h=1/400} + 0,4·max_banco_CUSUM_z + 0,3·logM_abs_reset − b)`. Caminho de emergência
determinístico e o único modo **calibrado em [0,1] por construção** — por isso os gates *absolutos* da
suíte de robustez só se aplicam a ele (§10). Não é mais baseline científico (foi superado pelo
protocolo OOF).

---

## §9. Estratégia de validação

### §9.0 A regra original e sua revisão

**Regra original:** nenhuma ferramenta do repositório calcula ou reporta uma estimativa de TS-AUC como
substituto do score oficial. Motivo (real, não hipotético): tentativas anteriores de replicar a
métrica oficial localmente produziram números sistematicamente **otimistas**, porque o pipeline local
acaba sendo ajustado até "parecer bom" — overfitting ao próprio harness.

**Revisão adotada (auditoria D4), em três cláusulas:**

1. **Mantida:** nenhuma réplica do scoring oficial como **estimador absoluto** de leaderboard.
2. **Adicionada:** **TS-AUC out-of-fold** (GroupKFold, 10.000 séries) com **bootstrap pareado por
   série** é o juiz **relativo** padrão. Instrumento diferente, modos de falha diferentes: σ do nível
   é ~0,005–0,008 (vs. 0,054 com 100 séries), e muito menor ainda na *diferença pareada*.
   Regra de decisão: adotar a mudança se o IC 95% do Δ excluir 0 no agregado ou no bucket-alvo
   declarado **a priori**.
3. **Adicionada:** a submissão oficial é a **âncora periódica** que valida o instrumento. Se o Δ
   oficial discordar sistematicamente do Δ OOF em sinal, o instrumento é rebaixado.

O erro que a revisão corrige: a rodada de intervenções de julho foi revertida por um ΔOOF de −0,0035
**sem intervalo de confiança** — o erro simétrico ao que a regra original queria evitar.

### §9.1 O que é medido localmente
Curvas de treino por fold; **importância transversal (XS-SHAP)** — ver §9.5; distribuição do score
por fatia de t; trajetórias na suíte sintética; testes de causalidade e determinismo (prioridade
máxima: um vazamento não detectado é o mecanismo mais provável por trás de um número local "bom
demais").

### §9.2 Harness causal
Instancia `fit_h0` + `StreamScorer` por série e alimenta o online **um ponto por vez**. Serve para
(a) construir o dataset (motor único) e (b) os testes de integridade. Não agrega scores em métrica
de desempenho.

### §9.4 Esquema de divisão
`GroupKFold(k=5)` por `id`, com estratificação aproximada por (rótulo da série, bucket de T, terço de
τ).

### §9.5 A medida certa de importância de feature (correção metodológica)

`mean|SHAP|` **é a medida errada nesta competição**: mistura variação *entre passos* com variação
*entre séries*. Pela invariância C1, o que é constante dentro de um passo é **exatamente neutro** para
a TS-AUC. A medida correta é a **dispersão da contribuição dentro do passo**, agregada com o peso
oficial `w_t` — coluna `xs_shap` em `scripts/shap_report.py` (TreeSHAP exato, versionado).

A discordância é dramática e sistemática: `meta_ln1p_t` cai de 11,9% → 2,7%; `conformal` cai de 36,6%
→ 15,0% (log-martingales crescem com t, e isso inflava a medida convencional); `meta_h0_*` sobe de
17,8% → 30,5%. **Qualquer priorização feita com a coluna convencional estava enviesada para features
que apenas acompanham o relógio.**

---

## §10. Suíte de robustez sintética (T1–T13)

Cenários com verdade conhecida, seeds fixas, "controle" = gêmeo sem quebra com as mesmas seeds.

| ID | Cenário | O que expõe |
|---|---|---|
| T1 | quebra muito cedo (τ=3, +0,8σ) | regime de baixa informação inicial |
| T2 | quebra no fim (τ=T−5) | **não-antecipação** |
| T3 | shift sutil (+0,15σ, τ=200) | piso de sensibilidade |
| T4 | shift abrupto (+1,5σ, τ=200) | canal rápido |
| T5 / T5b | variância 1→1,5 abrupta / em rampa | família variância; quebra suave |
| T6 | GARCH(1,1) **sem quebra** | falso positivo por vol-clustering; testa a trava §3.4 |
| T7 | dependência pura (φ 0,2→0,6, var. constante) | ponto cego do Bayes gaussiano |
| T8 | forma pura (N(0,1)→t₄/√2) | famílias de cauda/forma |
| T9 | outliers isolados sem quebra | travamento de alarme falso (CE1) |
| T10 | sazonalidade forte sem quebra | qualidade do whitening sazonal |
| T11 | T=10 (mínimo), τ=5 | fronteira curta, warm-up/NaN |
| T12 / T12b | maratona numérica (série ~constante / \|x\|~10 alternando) | estabilidade das recursões, determinismo |
| T13 | excursão transitória (+1σ por 60 passos, sem quebra) | **permanência**: o score deve decair depois |

**Gates:** comportamentais (diferença de mediana cenário-vs-controle, decaimento, inclinação),
deliberadamente **não** uma AUC.

**Gates relativos (R5, obrigatório para o modo supervisionado):** T2/T6/T9/T10/T13 comparam contra um
**painel de referência** (séries i.i.d. N(0,1) sem quebra, mesmas seeds/T), não contra um nível
absoluto. Motivo: o calibrador supervisionado devolve o resíduo sem offset e **não é calibrado em
[0,1]** — 9 de 15 gates absolutos falhavam por incompatibilidade de escala, medindo a régua e não o
detector. Os gates **absolutos** permanecem apenas para o modo `fallback`, que é calibrado por
construção.

---

## §11. Orçamento computacional

| Componente | Ordem |
|---|---|
| Whitening + escalares + 15 CUSUMs | 3–6 µs |
| Ring buffers e janelas | 1–2 µs |
| Filtro bayesiano (K=48 × hazards) | 8–15 µs |
| Conformal (bisect + acumuladores) | 1–2 µs |
| MMD/RFF · Haar · BOCPD | ~9 · ~4 · ~30 µs |
| Montagem do vetor de features | 3–5 µs |
| Predict LightGBM single-row (5 folds) | dominante |
| **Total medido (V4, o empacotado)** | **973,8 µs/passo** — gate 1500 → PASS |

Nenhum componente é super-linear em T por série: tudo é O(1), O(K) ou O(log n_h). `fit_h0`:
32–68 ms/série. Build do dataset completo: ~9 min paralelizado; treino: ~10–15 min; suíte de
robustez com 200 seeds: perto de uma hora (CI usa 40).

---

## §12. Auditoria adversarial

- **§12.1 Vazamento de causalidade.** *Teste de prefixo:* o score no passo t da execução completa deve
  ser **bit a bit igual** ao último score obtido processando apenas `x_{1:t}`. *Canário:*
  `LeakyStreamScorer` espia `x_{t+1}` e **deve** ser reprovado — é a prova de que o detector morde.
  **CE5 (lema do t/T):** `P(τ ≤ t | quebra, T) = t/T` é uma feature "perfeita" que exige conhecer T,
  que é futuro (A3). Corolário: qualquer quantidade correlacionada com T é vazamento.
  **Lista proibida:** T e derivados; estatísticas do segmento online completo; qualquer normalização
  reajustada no online.
- **§12.2 Comparabilidade transversal.** **CE6:** classificador só-histórico tentando prever o rótulo
  da série. Medido: **AUC 0,5067** (taxa-base 0,4967) e 0,5030 com a digital estendida de F2 — o
  gerador **não** vaza o rótulo pelo histórico. Política registrada: não construir features para
  explorar vazamento do gerador; a organização os corrige.
- **§12.4 Determinismo.** `num_threads=1` no predict; `deterministic`+`force_row_wise` no treino;
  apenas arrays/listas no caminho de inferência (nunca iteração sobre dict/set); Welford + log-space +
  renormalização; **zero RNG** na inferência (grep automatizado); versões pinadas; re-execução de 30%
  das séries comparada **bit a bit**.
- **§12.5 Comportamento sem quebra** (metade do universo): CE1 (max-hold), CE2 (baseline adaptativo),
  CE4 (por que whitening) — todos com teste dedicado.
- **§12.6 Envelope de poder.** Com m pontos pós-quebra e shift δ, o poder da estatística ótima é
  ≈ Φ(δ√m − z_α). O plano **não** promete detecção instantânea; promete subida consistente com esse
  envelope, e C3 mostra que a métrica cobra pouco pelos primeiros passos pós-τ.

---

## §13. Armadilhas do cenário online e defesas

1. **Normalização global do segmento online** → todas as escalas vêm do H0 congelado.
2. **Backtest vetorizado ≠ execução causal** → princípio do motor único (§8.1).
3. **Descontinuidade histórico→online** → buffers de lags atravessam a fronteira (`lag_seed`).
4. **Instabilidade numérica em recursões longas** → Welford, log-sum-exp, renormalização, clipping,
   float64; nunca "média dos quadrados menos quadrado da média".
5. **Baseline adaptativo absorve a quebra** → B2/CE2; variância/cauda em escala congelada.
6. **Autocorrelação das linhas de treino** → GroupKFold por série; split aleatório por linha é
   proibido.
7. **Usar T ou t/T** → proibido (A3, CE5); nenhuma assinatura recebe T.
8. **Treinar no rótulo da série em vez de `y_t`** → ensinaria o modelo a inflar score pré-quebra.
9. **Contaminação de estado entre séries** → um `StreamScorer` novo por série.
10. **Latência do predict single-row** → medição obrigatória + escada de mitigação.
11. **Arredondar/quantizar o score** → empates artificiais alteram a AUC; emitir float64 cru.
12. **Hazard/priors errados no Bayes** → múltiplos hazards como features + recalibração supervisionada;
    o filtro é feature, não juiz.
13. **NaN estrutural em famílias novas** → uma família que só acorda em t alto **dilui** o sorteio de
    `feature_fraction` em t baixo e piora aquele bucket. Toda família nova precisa de uma variante de
    janela curta ou de transporte de escala do nulo (ver `HISTORICO.md` §6).

---

## §14. Riscos de generalização

| Risco | Status |
|---|---|
| τ muito cedo (poucos pontos pós-quebra) | mitigado parcialmente — envelope declarado; candidatos jovens protegidos na poda; peso pequeno da métrica |
| τ muito tarde | mitigado — não-antecipação verificada (T2) |
| Tipo de quebra fora das famílias cobertas | em aberto parcial — cobertura genérica via conformal/MMD/rank |
| Distribuição de T difere treino→teste | mitigado — nenhuma feature usa T |
| Deriva do gerador treino→privado | mitigado parcialmente — só estatísticas com significado sob H0 |
| Interpretação exata do rótulo por passo (A1) | resolvido na prática pelo formato do `tau_index` do adaptador |
| Falso positivo GARCH irredutível (T6) | trade-off consciente (§3.4); tratado no calibrador |

---

## §15. Esqueleto, determinismo e fases

### §15.1 Contratos de código
Ver [`NOTAS_AGENTES.md`](NOTAS_AGENTES.md) §2 (assinaturas congeladas, incluindo as correções feitas
sobre o contrato original). Em resumo:

```python
h0 = fit_h0(hist, cfg)                       # §3.1 — puro, determinístico, imutável
scorer = StreamScorer(h0, blocks, ensemble, cfg)
score = scorer.update(x)                      # UMA observação → UM score em [0,1]
#   update() = whiten_step → for b in blocks: b.update(...) → assembly → predict → postprocess
```

### §15.2 Checklist de determinismo (pré-submissão)
grep anti-RNG limpo no módulo de inferência · `num_threads=1` no predict · `deterministic=true` +
`force_row_wise=true` no treino · todos os seeds = 42 · re-execução de 30% bit a bit idêntica · teste
de prefixo aprovado e canário reprovado · T12 sem NaN/Inf · nenhuma iteração sobre set/dict na
inferência · versões pinadas · score em float64 sem arredondamento.

### §15.3 Fases
**P0** fundação (adaptador + harness + fallback) · **P1** motor de estado completo + microbenchmark +
suíte · **P2** camada supervisionada · **P3** ablações registradas · **P4** congelamento. Todas
concluídas; o projeto opera hoje em ciclo de iteração de features julgado por R0 (`HISTORICO.md`).

---

## §16. Fontes

**Competição:** writeup do 1º lugar 2025 (stacking de árvores sobre features estatísticas, ~0,9014
AUC privado) · 2º lugar 2025 (LightGBM sobre ~2400 features com seleção SHAP) · documentação oficial
CrunchDAO · evidência de fracasso de redes puras (CNN+RNN ≈ 0,5; transformer hierárquico 0,49–0,54) ·
changelog W23/2026 (29 séries com quebra no primeiro passo, mal rotuladas — corrigidas) · baseline
oficial da Real-Time Edition.

**Método:** Page (1954) CUSUM · Roberts (1959) EWMA · Shiryaev (1963) detecção bayesiana · Welford
(1962) · Hinkley (1971) · Lorden (1971) · Willsky & Jones (1976) GLR · Pollak (1985) Shiryaev–Roberts
· Moustakides (1986) · Basseville & Nikiforov (1993) · Lai (1998) GLR window-limited · Vovk et al.
(2005) martingales conformais · Adams & MacKay (2007) BOCPD · Fearnhead & Liu (2007) · Murphy (2007)
NIχ² · Tartakovsky et al. (2014) · Volkhonskiy et al. (2017) · Ke et al. (2017) LightGBM ·
Barndorff-Nielsen & Shephard (bipower variation) · NEWMA (arXiv:1805.08061) · RFF para change
detection (arXiv:2505.17789) · L-momentos para caudas pesadas (arXiv:2306.09548).
