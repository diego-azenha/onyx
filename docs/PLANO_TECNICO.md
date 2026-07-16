# Plano de Construção — ADIA Lab Structural Break Challenge: Real-Time Edition (2026)

**Entregável:** plano de construção completo e implementável de um algoritmo de detecção *online* de quebra estrutural com saída de probabilidade acumulada por passo, sob causalidade estrita, posição de quebra desconhecida e métrica TS-AUC.
**Data do documento:** 2026-07-15. **Escopo:** cobre todos os itens da Seção 2 (critérios de aceite) e da Seção 6 (formato de saída) do prompt de tarefa.
**Nota de método:** o "protocolo multiagente" da Seção 3 do prompt foi emulado por um único agente conduzindo linhas de investigação independentes (L1–L6), com registro de família, veredicto, bloqueios e polinização cruzada (ver §2), seguido de uma passada de auditoria adversarial dedicada (§12). Onde afirmações dependem de fontes públicas, elas estão citadas (§16); onde dependem de estimativas próprias, estão marcadas como estimativas com premissas explícitas.

**Nota de revisão (2026-07-15):** com base em experiência relatada pelo usuário — tentativas anteriores de replicar localmente a métrica oficial produziram sistematicamente números otimistas demais, que não se confirmavam na engine oficial da CrunchDAO —, este plano foi revisado para **não calcular TS-AUC localmente como estimador de desempenho**. Toda referência a "gate" numérico de TS-AUC calculado em CV local (G-0, G-mono, G-peso) foi redefinida para ser decidida por **submissão à engine oficial**, não por um harness local. A métrica continua sendo estudada matematicamente (§1, essencial para o design) — o que muda é que ela deixa de ser *calculada* localmente como proxy de score. Detalhes e justificativa completa em §9.

---

## 0. Sumário executivo (≤1 página)

**Abordagem recomendada:** uma **máquina de estados causal por série** (custo O(1) a O(K) por passo, K=48) que transforma cada nova observação em um vetor de ~80 estatísticas sequenciais suficientes — todas atualizadas incrementalmente — e um **calibrador supervisionado LightGBM** treinado sobre o rótulo por passo `y_t = 1{τ ≤ t}` com pesos de linha alinhados ao perfil de pesos da TS-AUC. O vetor de estatísticas combina quatro mecanismos clássicos de detecção sequencial, cada um cobrindo uma fraqueza dos demais:

1. **Whitening causal pelo modelo H0** ajustado uma única vez no segmento histórico (AR(p) + escala robusta + opcionalmente termo sazonal), convertendo "detectar qualquer mudança de DGP" em "detectar mudança na distribuição das inovações padronizadas", e neutralizando o falso-positivo clássico por autocorrelação.
2. **Banco de CUSUMs** (média em 3 magnitudes ×2 lados, variância ↑/↓, sinal, excedência de cauda, dependência lag-1) — acumuladores O(1) minimax-ótimos para alternativas simples (Page 1954; Moustakides 1986), aproximando o GLR por grade de alternativas.
3. **Filtro bayesiano de troca única** (Shiryaev/BOCPD restrito a no máximo uma mudança, posterior sobre o tempo de quebra com pós-mudança Normal de parâmetros desconhecidos e poda de candidatos, K=48; 2 hazards) — produz diretamente `P(τ ≤ t | dados)`, o ranqueador teoricamente ótimo da AUC transversal por passo, e exporta de graça estatísticas de duas amostras "desde o τ̂ mais provável".
4. **Martingales conformais** sobre p-values causais das inovações contra a distribuição do histórico — evidência livre de distribuição, robusta a caudas pesadas, O(1)/passo.

Janelas rodantes curtas (10–250) e EWMAs completam a cobertura de quebras suaves e recentes. O LightGBM aprende, a partir dos ~10.000 τ rotulados do treino, (i) a combinação ótima entre famílias de quebra, (ii) a recalibração dos priors (hazard, mistura de tipos e magnitudes do gerador real) e (iii) o condicionamento por *nuisances* da série (caudas, clustering de volatilidade, qualidade do ajuste AR) que garante **comparabilidade transversal do score entre séries no mesmo passo** — exatamente o que a TS-AUC mede.

**Como a proposta resolve os três desafios centrais da edição:**

- **Causalidade estrita:** por construção — não existe caminho de código que veja t' > t. Um único motor sequencial (`StreamScorer.update(x_t) → score`) é usado tanto na inferência quanto na geração das features de treino (princípio do motor único), e o harness de validação repõe as observações uma a uma, com teste-canário que injeta deliberadamente uma dependência do futuro e verifica que o detector de vazamento a captura (§12.1). Features estruturalmente impossíveis (ex.: `t/T`, normalização pelo desvio do segmento online completo) constam de uma lista proibida verificada em revisão de código.
- **Posição de quebra desconhecida:** tratada três vezes, por mecanismos complementares — recursão max do CUSUM (varre implicitamente todos os τ candidatos a custo O(1)), soma bayesiana explícita sobre candidatos podados (posterior sobre τ), e o supervisionado que aprende a distribuição real de τ do gerador a partir dos rótulos de treino. A comparação formal entre os três e a justificativa do híbrido estão em §6.
- **Saída por passo compatível com TS-AUC:** o alvo de treino é o rótulo por passo `1{τ ≤ t}` (não o rótulo da série), com pesos de linha proporcionais ao peso `n_pos(t)·n_neg(t)` da métrica; a validação usa diagnósticos locais (curvas de treino, importância de features, comportamento em cenários sintéticos — §9.1) para orientar o desenvolvimento, mas a TS-AUC em si **não é calculada localmente como estimador de score** — decisões de desempenho são tomadas por submissão à engine oficial (§9). A análise de invariância da métrica (§1) mostra que componentes de score que dependem só de t são neutros — o risco real é drift heterogêneo entre séries, e é isso que os diagnósticos comportamentais testam.

**Monotonicidade (decisão explícita):** o score é o posterior/probabilidade do modelo a cada passo, **sem retenção máxima (max-hold)**. O posterior de `{τ ≤ t}` dado os dados é não-monótono por natureza (evidência transitória deve decair), e o max-hold trava alarmes falsos nas 50% de séries sem quebra — que são negativas em todos os passos. Variantes com retenção (max-hold, soft-hold) só seriam adotadas mediante evidência de comportamento indesejado do score livre nos cenários sintéticos de robustez (§10) e confirmação por submissão oficial — nunca por um número de validação calculado localmente (§9).

**Orçamento:** custo estimado de 50–150 µs por passo (dominado pelo predict single-row do LightGBM), ou ≈ 8–25 min para 10 milhões de passos no pior caso — folga >15× sobre as 15 h semanais mesmo com overhead de plataforma de 1 ms/passo (§11). Determinismo garantido por ausência total de aleatoriedade na inferência, operações em ordem fixa e thread única no predict (§15).

**Por que vence as alternativas exploradas:** Bayes puro depende de premissas gaussianas e de hazard corretos que o gerador não garante; CUSUM puro não emite probabilidade e não cobre alternativas compostas sem calibração; supervisionado puro sobre janelas cruas teria de redescobrir estatísticas sequenciais suficientes com risco maior de overfitting ao gerador; GLR exato e modelos recorrentes/foundation estouram o orçamento ou o determinismo (registro de bloqueios em §2). A evidência pública da edição 2025 (batch) — vencedor com stacking de modelos de árvore sobre features estatísticas, ~0,90 AUC privado, e fracasso documentado de arquiteturas neurais puras — corrobora o paradigma "estatísticas suficientes + árvores supervisionadas", aqui adaptado ao regime sequencial (§16, fontes F1–F6).

---

## 1. Formulação probabilística e leitura estrutural da TS-AUC

### 1.1 Definições e notação

Por série i: histórico H_i = (x_{−n_h+1}, …, x_0), com n_h ∈ [1000, 5000], **livre de quebra por definição** e já z-scorado; segmento online x_1, x_2, …, x_{T_i}, com T_i ∈ [10, 1000], revelado um ponto por vez. Com probabilidade 0,5 existe um único τ_i ∈ {1..T_i} a partir do qual o DGP muda de forma permanente (abrupta ou suave); com probabilidade 0,5 não existe quebra. A quebra pode ocorrer já em τ=1 (confirmado publicamente: o changelog W23/2026 da organização corrigiu séries com quebra "no primeiríssimo passo" que estavam mal rotuladas — fonte F7). Após cada x_t o algoritmo emite s_{i,t} ∈ [0,1].

**Rótulo por passo:** y_{i,t} = 1{τ_i ≤ t}. Uma série com quebra futura (τ_i > t) é **negativa** no passo t; vira positiva a partir de τ_i. Séries sem quebra são negativas em todos os passos. Essa é a leitura consistente com a definição da tarefa ("confiança acumulada de que a quebra **já** ocorreu") e com a observação do enunciado de que a proporção de positivos por passo cresce com t.

**Métrica:** TS-AUC = Σ_t w_t · AUC_t / Σ_t w_t, onde AUC_t é a AUC transversal no passo t sobre as séries vivas (T_i ≥ t), comparando {s_{i,t}} contra {y_{i,t}}, e w_t = n_pos(t) · n_neg(t). Passos com uma única classe têm w_t = 0 automaticamente.

**Registro de premissas (a validar no harness, §9.5):**
- **A1.** Séries com quebra ainda não ocorrida contam como negativas no passo t (leitura acima). O avaliador local calcula também a variante que as exclui, como análise de sensibilidade — se a implementação oficial divergir da A1, apenas os pesos de treino mudam, não a arquitetura.
- **A2.** As séries são alinhadas pelo índice do passo online: no "passo t" comparam-se séries que viram exatamente t observações online.
- **A3.** T_i é desconhecido durante a inferência (o fluxo simplesmente termina); nenhuma feature pode usar T_i.

### 1.2 Três consequências estruturais da métrica

**(C1) Invariância por passo.** AUC_t é invariante a qualquer transformação estritamente monótona aplicada uniformemente a todos os scores do passo t. Corolário 1: um componente de score que é função determinística **apenas de t** (ex.: um prior crescente P(τ≤t)) desloca todas as séries vivas igualmente e é **neutro** para a métrica. Corolário 2: o perigo real do "score que sobe só porque o tempo passa" não é a subida uniforme — é a **subida heterogênea entre séries** (ex.: travamento de ruído via max-hold, ou drift dependente de n_h), que reordena séries dentro do passo. Toda a discussão de monotonicidade (§7) e o gate de drift (§12.5) derivam deste corolário.

**(C2) Alvo ótimo = posterior.** Para cada passo t, o ranqueador que maximiza AUC_t é qualquer transformação monótona da razão de verossimilhança entre as populações {τ ≤ t} e {τ > t ∨ sem quebra} condicionais a x_{1:t} (propriedade padrão da ROC via Neyman–Pearson). Sob o modelo gerador, isso é o posterior P(τ ≤ t | x_{1:t}, H). Logo: (i) o alvo de treino correto é o rótulo por passo, não o rótulo da série; (ii) o posterior é **não-monótono em t por natureza** — os eventos {τ≤t} são aninhados, então o *prior* é não-decrescente, mas a evidência pode reverter (um outlier em t=10 eleva o posterior; 50 pontos normais em seguida o derrubam, porque uma quebra real em t=10 teria deixado rastro sustentado). Isso fundamenta a decisão de monotonicidade em §7.

**(C3) Perfil de pesos w_t e o que ele diz sobre onde investir.** Sob T ~ U{10..1000}, quebra com prob. ½ e τ | quebra, T ~ U{1..T}: n_alive(t) ∝ P(T ≥ t) decresce ~linearmente; n_pos(t) cresce de ~0 (poucas séries já quebraram em t pequeno) até um pico e depois cai junto com n_alive; n_neg(t) = n_alive − n_pos. O produto w_t = n_pos·n_neg é **pequeno nos primeiros passos** (quase não há positivos), **pequeno nos últimos** (quase não há séries vivas) e **máximo no horizonte intermediário** (t na faixa de dezenas a poucas centenas). Implicações operacionais: (i) detecção "instantânea" em 1–3 passos vale pouco peso — o regime que paga é o de **10–200 pontos pós-quebra**, e as janelas/half-lives do design (§5) foram escolhidas para esse regime; (ii) os pesos de linha do treino supervisionado replicam w_t medido **empiricamente no próprio treino** (τ e T conhecidos), não a fórmula idealizada (§8.2). O harness plota o w_t empírico como primeiro diagnóstico do dataset.

### 1.3 Decomposição do problema

O pipeline fatora em três blocos com contratos claros:

1. **Fase-histórico (uma vez por série, antes do primeiro passo online):** ajustar o modelo H0 e pré-computar constantes (coeficientes AR, escalas, quantis, array ordenado de inovações do histórico, σ_u). Custo O(n_h·p + n_h log n_h). Não é caminho crítico.
2. **Fase-online (uma vez por observação):** atualizar o estado (Welford, ring buffers, CUSUMs, filtro bayesiano podado, martingales, EWMAs), montar o vetor de features e prever. Custo O(K) dominado pelo predict.
3. **Fase-treino (offline, uma vez):** repassar as 10.000 séries de treino pelo MESMO motor sequencial, coletar (features, y_t, peso) por passo, treinar o LightGBM com GroupKFold por série, congelar o ensemble.

---

## 2. Registro do processo de exploração multiagente (emulado)

Protocolo seguido: seis linhas desenvolvidas de forma independente (sem "favorita" declarada) até cada uma expor pontos fortes e fracos concretos; só então houve polinização cruzada (§2.3). Linhas agrupadas por mecanismo estatístico, não por redação. Bloqueios exigiram estimativa concreta (tempo, contraexemplo ou premissa não garantida), não impressão.

### 2.1 Linhas exploradas e veredictos

| Linha | Mecanismo | Pontos fortes constatados | Limitações constatadas | Veredicto |
|---|---|---|---|---|
| **L1 — Bayes de troca única** (Shiryaev 1963; Adams & MacKay 2007 restrito a 1 mudança; Fearnhead & Liu 2007) | Posterior P(τ≤t) por soma sobre candidatos k≤t com pós-mudança conjugada (Normal com μ,σ² desconhecidos) | Emite exatamente o alvo (C2); integra sobre τ e sobre magnitude; sufficient stats por candidato dão de graça "duas amostras desde τ̂"; O(K)/passo com poda padrão | Premissas: inovações ~gaussianas (mitigado por clipping + L5), hazard correto (mitigado por 2 hazards + recalibração L4); custo K×lgamma por passo (medível, cabe) | **Núcleo** (como família de features, não como score final) |
| **L2 — CUSUM/Page-Hinkley/GLR** (Page 1954; Hinkley 1971; Willsky & Jones 1976; Lorden 1971; Moustakides 1986; Lai 1998) | Acumuladores O(1); ótimo minimax para alternativa simples; recursão max varre todos os τ implícitos; robusto e barato | Não emite probabilidade (precisa calibração); alternativa composta exige grade/banco; GLR exato é O(t)/passo (bloqueio §2.2-B1) | **Núcleo** (banco de 15 CUSUMs + janelas estilo GLR window-limited de Lai 1998) |
| **L3 — Cartas de controle / janelas rodantes** (Roberts 1959 EWMA; Shewhart) | Cobertura de mudanças suaves; interpretável; O(1) | Sozinhas, fracas para forma/dependência; escolha de λ/w arbitrária → resolver com conjunto de escalas + seleção pelo modelo | **Suporte** (EWMAs multi-λ + 5 janelas) |
| **L4 — Supervisionado sequencial** (LightGBM sobre features incrementais, rótulo por passo; alternativa: GRU/TCN causal) | Única linha que usa os τ do treino; aprende prior real do gerador (mistura de tipos, magnitudes, distribuição de τ) e o condicionamento por nuisance que a TS-AUC exige; árvores + features estatísticas foi o paradigma vencedor comprovado da edição batch (F1–F4) | Sobre janelas cruas teria de redescobrir suficiência sequencial (amostra insuficiente, risco de artefato); RNN por passo: viável em custo só com célula numpy própria, e a evidência pública 2025 mostra redes puras ≈0,5 AUC (F5, F6) | **Núcleo como camada de fusão**; RNN **estacionada** (P2 opcional, §2.2-P1) |
| **L5 — Martingales conformais** (Vovk et al. 2005; Volkhonskiy et al. 2017) | Livre de distribuição (só troca-de-exchangeability); p-values causais contra o histórico a O(log n_h); robusto a caudas; evidência acumulada com garantia de Ville sob H0 | Menos potente que LR bem-especificada quando o modelo H0 é bom; betting function fixa (mitigado por mistura de ε) | **Suporte** (4 features de log-martingale) |
| **L6 — Foundation models / deep forecasting por passo** (sugeridos como permitidos pela organização, F8) | Potencial cobertura de padrões não lineares | Custo por passo proibitivo (forward de transformer × ~10⁷ passos), risco de determinismo, sem evidência de ganho no domínio (F5, F6) | **Bloqueada** (§2.2-B3) |

### 2.2 Bloqueios e estacionamentos (com números)

- **B1 — GLR exato sobre todos os candidatos.** Custo O(t)/passo ⇒ Σ_{t≤T} t = T(T+1)/2 ≈ 5·10⁵ atualizações de candidato por série de T=1000; para 10⁴ séries, ≈ 5·10⁹ atualizações por conjunto de teste. Em Python puro (~0,1–0,3 µs/op efetivo em laço quente) isso é 8–25 min *se* cada atualização fosse 1 op — na prática são ~10 ops ⇒ 1,5–4 h; em numpy por passo, o overhead de chamada (~1–2 µs por op vetorial) com vetores médios de 500 elementos dá ~1 ms/passo ⇒ 10⁷ passos ≈ 2,8 h **só neste componente**, sem contar o resto. Viável apenas com numba — dependência que decidimos não assumir no caminho crítico. **Substituído** por: (i) janelas rodantes = GLR *window-limited* (Lai 1998 mostra perda de eficiência assintótica pequena com janela adequada); (ii) filtro bayesiano podado (L1), que é a versão "soma" do GLR com priors.
- **B2 — Refit adaptativo do baseline no online** (ex.: re-estimar AR ou EWMA-vol e usar para *tudo*): contraexemplo concreto no §12-CE2 — o baseline adaptativo **absorve a própria quebra** (vol-EWMA com λ=0,06 converge à nova variância em ~17 passos e o detector de variância enxerga nada). Bloqueada como política geral; whitening adaptativo permitido **apenas** para as famílias média/dependência/forma, nunca para variância/cauda (§3.4).
- **B3 — Foundation models / transformers por passo.** Um forward de modelo pequeno (~10⁷ FLOPs) por passo × 10⁷ passos = 10¹⁴ FLOPs ⇒ horas de CPU só de inferência, mais overhead de framework (≥50–200 µs/chamada ⇒ 0,5–2 h só de overhead), mais risco de não-determinismo de kernels. Sem evidência pública de ganho no domínio (F5: CNN+RNN ≈ 0,5 AUC; F6: transformer hierárquico 0,49–0,54 AUC, abaixo de ensembles de árvore). **Bloqueada**; reabertura exigiria demonstração de custo O(1)/passo real e ganho em CV.
- **P1 — GRU causal em numpy puro (estacionada, não bloqueada).** Célula GRU d=32 escrita em numpy: ~10 ops vetoriais/passo ≈ 20 µs ⇒ 200 s por 10⁷ passos — **cabe no orçamento**. Estacionada por relação custo/benefício de desenvolvimento (treino BPTT próprio, determinismo de treino, evidência pública desfavorável a redes puras neste gerador). Critério de reabertura: platô do plano principal em CV com folga de cronograma; entraria como membro diverso do ensemble via rank-average.
- **P2 — p-values por permutação/bootstrap em janela.** 200 permutações × custo da estatística por passo multiplica o custo por ≥200× ⇒ estoura qualquer folga. Substituído por: estatísticas padronizadas analiticamente + recalibração supervisionada + martingales conformais (que já entregam p-values causais exatos sob H0 sem reamostragem).

### 2.3 Polinizações cruzadas (após desenvolvimento independente)

- **L1→L2/L3:** os *sufficient statistics* dos candidatos do filtro bayesiano (n_j, média_j, M2_j) são expostos como features "duas amostras desde o τ̂ MAP" — um GLR adaptativo de janela variável obtido **de graça**, sem custo adicional.
- **L2→L1:** as idades dos CUSUMs (passos desde o último zero) servem de localizador barato de τ e de feature de consistência cruzada com o τ̂ bayesiano (concordância entre localizadores é evidência de quebra real; discordância sugere ruído).
- **L5→L1:** a estimativa de ν (caudas) do histórico decide o clipping das inovações que protege a verossimilhança gaussiana do filtro bayesiano, enquanto os indicadores de excedência crus preservam o sinal de cauda para as famílias que precisam dele.
- **L4→todas:** a camada supervisionada recalibra hazards, magnitudes de grade do CUSUM e a mistura entre famílias — por isso as grades (§5) podem ser grosseiras sem perda: o modelo interpola.

### 2.4 Guarda contra convergência prematura

Durante a validação, três baselines de linha única são reportados a cada iteração ao lado do híbrido: (i) melhor CUSUM isolado (logístico do máximo do banco), (ii) posterior bayesiano isolado, (iii) LightGBM só com janelas/EWMAs (sem CUSUM/Bayes/martingale). O híbrido precisa superar claramente os três nos diagnósticos locais de §9.1 (curvas de treino, importância de features, comportamento nos cenários sintéticos) para ser levado a uma sonda de submissão; a decisão final sobre qual arquitetura vai para produção é tomada por comparação de submissões na engine oficial (gate G-0, §9.3) — nunca por um número de validação calculado localmente.

---

## 3. Fase-histórico: caracterização do regime H0 e whitening causal

Executada uma vez por série, ao receber o histórico completo (permitido: o histórico é dado de uma vez e é, por definição, livre de quebra — ele é referência, nunca objeto de teste). Tudo aqui é determinístico e O(n_h·p + n_h log n_h).

### 3.1 Estimativas do H0

1. **Momentos:** μ̂₀ = média(H), ŝ₀ = dp(H), ŝ_rob = 1,4826·MAD(H). Não assumir μ=0, σ=1 exatos — o z-score do organizador pode ter sido feito de forma que deixe resíduos de escala; estimar sempre e usar as estimativas, nunca as constantes nominais.
2. **AR(p) por Yule–Walker/Levinson–Durbin com p = 10** sobre H centrado. Aceitar o AR se var(resíduo)/var(x) ≤ 0,98 (redução ≥ 2%); caso contrário φ := 0 (série já ~branca; inovação = x centrado).
3. **Checagem sazonal parcimoniosa:** ACF dos resíduos até lag 128; se existir |ρ(L)| > 0,25 com L ∈ [6, 128], reajustar por mínimos quadrados com lags {1..10, L}. No máximo um termo sazonal (evita sobreajuste do H0 a ruído).
4. **Escala de inovação:** σ̂_e = dp(resíduos) e σ̂_e,rob = 1,4826·MAD(resíduos). Inovações do histórico: e = resid/σ̂_e.
5. **Caudas:** curtose excedente κ̂_ex de e → ν̂ = clip(4 + 6/κ̂_ex, 5, 50) (método dos momentos da t de Student; κ̂_ex ≤ 0 ⇒ ν̂ = 50 ≈ gaussiana). Quantis de e: q_{0,01;0,05;0,10;0,25;0,75;0,90;0,95;0,99}.
6. **Dependência residual:** ρ̂₁(e) (qualidade do whitening) e ρ̂₁(|e|) (clustering de volatilidade). σ̂_u = dp(u) com u_t = e_t·e_{t−1} no histórico (escala do detector de dependência).
7. **Array ordenado** das inovações do histórico (e ordenado; e |e| ordenado) para p-values conformais por busca binária O(log n_h).
8. **Buffers iniciais:** ring de lags com os últimos 10 valores de H; last_e com a última inovação do histórico. Isso garante **continuidade exata** na fronteira histórico→online (armadilha §13.3): a inovação e₁ do primeiro passo online é prevista com os mesmos lags que teriam sido usados se o histórico continuasse.

### 3.2 Whitening causal no online

A cada passo t: x̂_t = ĉ + Σ_j φ̂_j·x_{t−j} (lags do ring, que atravessa a fronteira), e_t^raw = (x_t − x̂_t)/σ̂_e, e_t = clip(e_t^raw, −8, +8). O clipping protege verossimilhanças e acumuladores de outliers extremos; os **indicadores de excedência crus** (|e^raw| > q₉₅, q₉₉, 6) preservam o sinal de cauda antes do clip. Parâmetros do H0 **nunca são reestimados no online** (bloqueio B2): um baseline adaptativo absorveria a própria quebra.

### 3.3 Por que whitening resolve a armadilha nº 1 da detecção sequencial

Sob H0 verdadeiro, e_t ≈ i.i.d.(0,1)-ish; qualquer estatística calibrada para i.i.d. fica aproximadamente válida. Sem whitening, um AR(1) com φ=0,9 infla a variância de médias amostrais por (1+φ)/(1−φ) = 19×, ou seja, estatísticas-z de média ficam ~4,4× maiores que o nominal e todo detector de média dispara em série sem quebra (contraexemplo quantificado em §12-CE4). As mudanças de DGP aparecem nas inovações assim: mudança de média de x → média de e desloca (transiente + sustentado); mudança de variância de x → variância de e muda; mudança de dependência (φ) → autocorrelação reaparece em e; mudança de forma/cauda → forma/cauda de e muda. Ou seja, o whitening **preserva todas as famílias de quebra** e remove o principal gerador de falso positivo.

### 3.4 Segundo fluxo de inovações (condicional) — com trava anti-absorção

Se ρ̂₁(|e|) > 0,15 no histórico (série tipo GARCH), mantém-se um segundo fluxo ẽ_t = e_t/√v_t com v_t = (1−λ_v)·v_{t−1} + λ_v·e_t², λ_v = 0,06, v₀ = 1. **Regra rígida:** ẽ alimenta apenas as famílias média/dependência/forma (onde vol-clustering gera falso positivo); as famílias **variância e cauda usam sempre e com escala congelada do histórico** — caso contrário o EWMA-vol absorve a quebra de variância em ~1/λ_v ≈ 17 passos e a cega (contraexemplo §12-CE2). Se ρ̂₁(|e|) ≤ 0,15, ẽ ≡ e (fluxo único, features duplicadas colapsam e o LightGBM as ignora).

---

## 4. Modelo de estado e atualização incremental (critério de aceite §2.1)

Estado por série (todos float64, salvo inteiros indicados). Nenhuma atualização recomputa sobre o histórico do online; tudo é recursivo.

### 4.1 Campos fixos (pós fase-histórico)
`phi[10], c, sigma_e, sigma_e_rob, nu_hat, q[8], sorted_e_hist[n_h], sorted_abs_e_hist[n_h], sigma_u, rho1_e, rho1_abs_e, ar_r2, seasonal_lag, seasonal_coef, flags`.

### 4.2 Campos dinâmicos e regras de atualização (por passo, ao chegar x_t)

| Bloco | Estado | Atualização (O(1) salvo indicado) |
|---|---|---|
| Lags | ring `xlag[10]`, `last_e`, `last_e2` | push x_t; guardar e_t ao final do passo |
| Contador | `t` (int) | t += 1 |
| Welford global (desde t=1) | `n, mean_e, M2` | d = e − mean; mean += d/n; M2 += d·(e − mean) (Welford 1962 — estável para 10³ passos) |
| Acumuladores globais | `S_abs, C_2s, C_3s, C_q95, C_q99, C_pos, S_u` (u = e·e_prev) | somas/contagens O(1) |
| EWMA média | m_λ, λ ∈ {0,05; 0,10; 0,30} | m ← (1−λ)m + λe; z-EWMA = m/√(λ/(2−λ)) |
| EWMA variância | v_λ, mesmo conjunto de λ | v ← (1−λ)v + λe² (v₀=1); estat = ln v |
| EWMA sinal / excedência | s₀.₁ sobre 1{e>0}; x₀.₁ sobre 1{\|e^raw\|>q₉₅} | idem; estat = (s−0,5)/√(0,25·λ/(2−λ)) |
| Ring de janelas | buffer `ebuf[256]`, `e2buf[256]` (máscara &255); para w ∈ {10,25,50,100,250}: (S_w, Q_w, C2_w, Cq95_w, Cpos_w) | S_w += e_t − e_{t−w} (elemento antigo lido do ring); análogo para os demais; w efetivo = min(w, t) no warm-up |
| Banco CUSUM (15 estat. + 15 idades int) | W⁺_δ, W⁻_δ (δ∈{0,25;0,5;1,0}); V↑_ρ (ρ∈{1,5;2,5}), V↓_{0,5}; B_sign±; B_exc95, B_exc99; D⁺, D⁻ (dependência) | média: W⁺ ← max(0, W⁺ + δ·ê − δ²/2), ê = fluxo ẽ; variância (sobre e congelado): V↑ ← max(0, V↑ + ½[(1−1/ρ)e² − ln ρ]); Bernoulli: B ← max(0, B + b·ln(p₁/p₀) + (1−b)·ln((1−p₁)/(1−p₀))) com (p₀,p₁) = (0,5, 0,65)/(0,5, 0,35) para sinal, (0,05, 0,15) e (0,01, 0,05) para excedências; dependência: incremento gaussiano com δ_u = 0,3 sobre u_t/σ̂_u; idade: 0 se acumulador zerou, senão +1 |
| Filtro bayesiano ×2 (hazards h ∈ {1/100, 1/400}) | por filtro: arrays K=48 de (n_j, mean_j, M2_j, logw_j), `logw0`, contadores | ver §4.3 |
| Martingales conformais | logM_abs, logM_right, logM_sign — cada um com 4 acumuladores por ε ∈ {0,05; 0,1; 0,2; 0,4} + variantes com reset SR | p_t = (rank_meio de \|e_t\| em sorted_abs_e_hist)/(n_h+1) via bisect O(log n_h); por ε: L_ε += ln ε + (ε−1)·ln p_t; logM = logmeanexp_ε(L_ε); versão reset: L ← max(0, L + inc) |
| Escala bruta (hedge) | m^x_{0,1} (EWMA de x−μ̂₀), janela-100 de x e x² | O(1), mesmas recursões |
| Saída | `prev_score` | guardado só para variantes de suavização (§7); default não usa |

Warm-up: para t < w ou t < 5, estatísticas usam n efetivo com padronização correspondente; features indefinidas emitem NaN (o LightGBM trata NaN nativamente — nunca sentinelas mágicas).

### 4.3 Filtro bayesiano de troca única (recursão exata, log-espaço)

Modelo: sob H0, e_t ~ N(0, 1) (com clipping ±8 e ν̂ apenas informando o clip/condicionamento); pós-mudança, e_t ~ N(μ, σ²) com prior conjugado Normal-Inv-χ² (μ₀=0, κ₀=0,5, ν₀=2, σ₀²=1,5). Hazard constante h; sem morte de regime (quebra é permanente). Por passo, para cada filtro:

1. `logw_new = logw0 + ln h + logpred_prior(e_t)` — nasce o candidato k = t (a quebra pode ocorrer já no primeiro ponto observado, cf. F7).
2. `logw0 ← logw0 + ln(1−h) + ℓ₀(e_t)`, com ℓ₀ = −½ln(2π) − e²/2.
3. Para cada candidato j vivo: `logw_j ← logw_j + logpred_j(e_t)`, onde logpred_j é a densidade preditiva t de Student do NIχ² com (n_j, mean_j, M2_j): κ_n = κ₀+n_j; μ_n = (κ₀μ₀ + n_j·mean_j)/κ_n; ν_n = ν₀+n_j; σ_n² = [ν₀σ₀² + M2_j + κ₀n_j(mean_j−μ₀)²/κ_n]/ν_n; predizer t_{ν_n}(μ_n, σ_n²(κ_n+1)/κ_n). Depois atualizar (n_j, mean_j, M2_j) com e_t (Welford).
4. Inserir o candidato novo; **poda:** se K>48, descartar os menores logw, protegendo sempre os 8 mais recentes (candidatos jovens têm pouca evidência acumulada e seriam podados injustamente).
5. **Renormalização em log-espaço:** subtrair max(logw*, logw0) de todos quando o máximo exceder 600 em módulo (estabilidade em 1000 passos).
6. Saídas: `LO_h = logsumexp(logw_j) − logw0` (log-odds de "quebra já ocorreu"); τ̂_MAP = candidato de maior logw; `age_MAP = t − τ̂_MAP`; e as estatísticas do MAP: (n_MAP, mean_MAP, ln((M2_MAP/n_MAP)+ε)).

Custo: 2 filtros × 48 candidatos × O(1) com 2 chamadas lgamma por candidato (caching de lgamma por ν_n inteiro-deslocado reduz a ~1 lookup); medição-alvo ≤ 15 µs/passo (§11).

---

## 5. Taxonomia de features/estatísticas de detecção (critério de aceite §2.2)

Colunas: **Ref** = janela de referência (H = histórico congelado; G = global desde t=1; W_w = janela rodante w; E_λ = EWMA). **Conversão** = como a saída bruta vira evidência em [0,1]: por padrão, **entra crua no LightGBM** (a conversão final é o predict_proba do modelo, §8); a coluna indica a padronização aplicada antes e, quando existe, o mapeamento probabilístico analítico usado no fallback puro-estatístico (§8.5). Todas as atualizações são as recursões de §4.2. Fluxo: ẽ = fluxo vol-ajustado (§3.4) para média/dependência/forma; e = fluxo congelado para variância/cauda.

| # | Feature (contagem) | Família | Atualização / fórmula | Ref | Custo/passo | Sensível a | Conversão |
|---|---|---|---|---|---|---|---|
| 1 | z-Welford global: √n·mean_e (1) | média | Welford | G | O(1) | shift de média desde o início | z→Φ(z) no fallback |
| 2 | z-EWMA média ×3 λ (3) | média | m_λ/√(λ/(2−λ)) | E_λ | O(1) | shift/drift recente de média | z→Φ |
| 3 | z de janela: √w·(S_w/w) ×5 (5) | média | ring | W_w | O(1) | shift de média nos últimos w | z→Φ |
| 4 | CUSUM média W⁺,W⁻ ×3 δ (6) | média | recursão max | G (reset implícito) | O(1) | shift persistente de qualquer início, magnitude ~δ | logístico(a·W+b) |
| 5 | Page-Hinkley ≡ CUSUM δ=0,25 (—) | média | coberto por #4 | — | — | drift lento | — |
| 6 | ln v_λ EWMA ×3 (3) | variância | v_λ | E_λ | O(1) | mudança recente de variância | \|z\| via var assint. |
| 7 | ln(Q_w/w) janela ×5 (5) | variância | ring | W_w | O(1) | mudança de variância nos últimos w | idem |
| 8 | CUSUM var ↑1,5, ↑2,5, ↓0,5 (3) | variância | recursão max sobre e² | G | O(1) | mudança persistente de σ² | logístico |
| 9 | ln(M2/n) global (1) | variância | Welford | G | O(1) | mudança de σ² desde o início | — |
| 10 | frac(\|e\|>2) janela ×2 (w=50,250) + EWMA exced. (3) | variância/cauda | contadores | W/E | O(1) | vol e caudas | binomial z |
| 11 | Bernoulli-CUSUM exced. q₉₅, q₉₉ (2) | cauda | recursão max | G | O(1) | engrossamento de cauda | logístico |
| 12 | frac(\|e^raw\|>q₉₉) global + máx \|e^raw\| até t (2) | cauda | contador/max | G | O(1) | eventos extremos | binomial z |
| 13 | Bernoulli-CUSUM de sinal ± (2) | média (robusta) | recursão max sobre 1{ẽ>0} | G | O(1) | shift de mediana sob cauda pesada | logístico |
| 14 | z de proporção de positivos, janela w=50, 250 (2) | média (robusta) | contadores | W | O(1) | shift de mediana recente | binomial z |
| 15 | CUSUM dependência D⁺,D⁻ (2) | dependência | max sobre u_t/σ̂_u | G | O(1) | surgimento/troca de autocorrelação | logístico |
| 16 | ρ̂₁ online: (S_u/n)/(M2/n) (1) | dependência | acumuladores | G | O(1) | mudança de φ | Fisher-z |
| 17 | ρ̂₁ janela w=100 (1) | dependência | ring de u | W | O(1) | mudança recente de φ | Fisher-z |
| 18 | quantile-crossing: frac em (q₂₅,q₇₅) e frac<q₁₀, janela 100 (2×) (4) | forma | contadores ring | W+H | O(1) | mudança de forma marginal (achatamento, assimetria) | binomial z |
| 19 | assimetria incremental de janela 250: (Σe³ ring) padronizada (1) | forma | ring de e³ | W | O(1) | mudança de skew | z |
| 20 | log-odds bayesiano LO_h ×2 hazards (2) | todas (média+var) | filtro §4.3 | G+H | O(K) | qualquer mudança de (μ,σ²) das inovações, τ desconhecido | σ(LO) — já é probabilidade |
| 21 | age_MAP, ln(1+age_MAP) ×2 hazards (4) | localização | filtro | G | O(K) | tempo desde a quebra mais provável | — |
| 22 | stats do candidato MAP: √n·mean_MAP, ln(M2_MAP/n_MAP) ×2 hazards (4) | média+var "desde τ̂" | filtro (grátis) | adaptativa | O(1) | duas-amostras com janela auto-selecionada | z |
| 23 | logM conformal: abs, cauda-direita, sinal; versões acumulada e reset (6→4 usadas) | forma/cauda/média, livre de distribuição | mistura de ε | G+H | O(log n_h) | violação de exchangeability vs. histórico | e^{−logM} é p-value (Ville) |
| 24 | idades dos CUSUMs (6 selecionadas: média δ=0,5 ±, var↑1,5, sinal±, exc95) | localização | contadores | G | O(1) | consistência de localização entre famílias | — |
| 25 | concordância de localizadores: \|age_MAP − idade_CUSUM_média\|, min das idades (2) | meta-localização | derivada | — | O(1) | quebra real (localizadores concordam) vs ruído | — |
| 26 | hedge bruto: EWMA(x−μ̂₀)/escala, ln var janela-100 de x (2) | média/var (sem whitening) | recursões | E/W | O(1) | robustez a AR mal ajustado | z |
| 27 | meta: t, ln(1+t) (2) | condicionamento | — | — | O(1) | neutro por C1; entra p/ interações | — |
| 28 | meta H0: n_h, ν̂, ρ̂₁(e), ρ̂₁(\|e\|), ar_r2, seasonal_flag, q₉₉, σ̂_e,rob/σ̂_e (8) | condicionamento | fixos | H | 0 | calibração transversal por nuisance | — |

**Total ≈ 78 features.** Justificativa de "não dispara com ruído normal": (i) tudo roda sobre inovações whitened e padronizadas pela escala do histórico — sob H0 as estatísticas têm distribuição aproximadamente conhecida e estável entre séries; (ii) famílias robustas (sinal, quantile-crossing, conformal) duplicam as paramétricas justamente para que cauda pesada legítima não seja lida como quebra; (iii) as features de condicionamento (#27–28) permitem ao modelo aprender "desconte o CUSUM de variância quando ρ̂₁(|e|) do histórico é alto", que é a correção empírica da heterocedasticidade condicional sem quebra; (iv) nenhuma feature usa estatística do conjunto de teste, do futuro da série ou de T.


---

## 6. Estratégia para posição de quebra desconhecida (critério de aceite §2.3)

Quatro mecanismos candidatos, avaliados nos eixos que importam para esta métrica e este orçamento:

| Mecanismo | Como varre τ | Custo/passo | Emite prob.? | Cobre alternativa composta (μ,σ,forma,dep.)? | Usa os τ do treino? | Risco principal |
|---|---|---|---|---|---|---|
| CUSUM (recursão max) | implícita e exata p/ alternativa simples: W_t = max_{k≤t} Σ_{i=k}^t LLR(e_i), computado por recursão O(1) (Page 1954) | O(1) por alternativa | não (estatística) | só via banco/grade de alternativas | não | calibração; grade discreta de magnitudes |
| GLR / GLR-janelado | max explícito sobre k (e MLE dos params pós-quebra) | O(t) exato → **bloqueado** (§2.2-B1); O(#janelas) na versão window-limited (Lai 1998) | não | sim, dentro da janela | não | perda de potência p/ quebras mais antigas que a janela — compensada pelos acumuladores globais |
| Bayes de troca única (Shiryaev; BOCPD-1) | soma ponderada sobre todos os k≤t (podada) | O(K) | **sim: P(τ≤t\|dados)** | (μ,σ²) sim; forma/dependência não diretamente | só via escolha de hazard | premissa gaussiana; hazard errado |
| Supervisionado por passo | aprende P(τ≤t\|features) direto dos rótulos | O(predict) | sim | sim, se as features cobrirem | **sim — única linha que usa** | overfitting ao gerador; precisa de features suficientes |

**Decisão: híbrido em camadas.** CUSUMs e janelas fornecem a varredura O(1) de τ para cada família de quebra; o filtro bayesiano fornece a integração probabilística sobre τ e magnitude no núcleo (μ,σ²) — que a evidência da edição 2025 indica ser o grosso da massa de tipos de quebra do gerador — mais a localização τ̂; o supervisionado funde tudo e injeta a única informação que nenhum método clássico tem: **a distribuição real de τ, de tipos e de magnitudes do gerador**, presente nos 10.000 rótulos de treino. Formalmente, cada mecanismo clássico entrega uma estatística ~suficiente da sua família; o LightGBM aprende a verossimilhança relativa entre elas sob a mistura verdadeira — que é exatamente o que o posterior exato exigiria e nenhuma forma fechada oferece.

Justificativa comparativa contra as alternativas puras: (i) Bayes puro perde quando as inovações têm cauda pesada ou a quebra é de dependência/forma (fora do seu modelo) — cenários T7/T8 da suíte (§10) foram desenhados para expor isso; (ii) CUSUM puro exige mapear estatística→probabilidade por série de forma comparável transversalmente, o que é precisamente um problema de calibração supervisionada — então é natural dar o passo completo; (iii) supervisionado puro sobre janelas cruas (sem CUSUM/Bayes) é o baseline (iii) comparado ao híbrido por submissão oficial (gate G-0, §9.3): se tiver desempenho comparável, o plano simplifica — mas a expectativa, apoiada na teoria (as recursões max/soma são estatísticas que uma árvore não consegue reconstruir de janelas fixas), é que não empate.

---

## 7. Decisão sobre monotonicidade do score (critério de aceite §2.4)

**Decisão: score livre (sem imposição de monotonicidade), com variantes de retenção testadas sob gate pré-registrado.**

Fundamentação (usa C1 e C2 de §1.2):

1. O alvo ótimo por passo é o posterior P(τ≤t | x_{1:t}), que é **não-monótono por natureza**: evidência transitória (outlier, rajada curta) eleva o posterior e depois é corretamente revertida por observações subsequentes consistentes com H0. No filtro de §4.3 isso é explícito: o candidato criado no outlier perde massa a cada nova observação normal, numa razão média e^{−KL(H0‖pós)} por passo.
2. **Contraexemplo contra o max-hold (CE1, §12):** série sem quebra, T=1000, um outlier de 6σ em t=15. O score livre sobe (digamos a ~0,6–0,7) e decai em ~10–30 passos. Com s_t = max(s_{t−1}, p_t), a série fica travada ≥0,6 pelos 985 passos restantes — e ela é **negativa em todos eles**, ranqueando acima de quebras fracas verdadeiras em quase todos os AUC_t subsequentes. Como 50% do universo é sem quebra, o custo esperado do travamento é de primeira ordem.
3. O argumento a favor da retenção ("uma quebra é permanente, a confiança não deveria cair") é capturado **sem** retenção: pós-quebra real, a evidência é recorrente, e os acumuladores (CUSUM, posterior) crescem sozinhos; quedas do score pós-quebra indicam evidência fraca — e rebaixar essas séries em favor de quebras mais claras é o comportamento que maximiza AUC_t.
4. Pela invariância C1, qualquer "piso" uniforme em t não muda nada; só a retenção **por série** muda o ranking — e ela retém tanto sinal quanto ruído.

**Variantes registradas (adoção decidida por submissão oficial, nunca por métrica local — §9):**
- V-hold: s_t = max(s_{t−1}, p_t) (retenção dura);
- V-soft: s_t = max(p_t, s_{t−1} − 0,02) (decaimento linear teto);
- V-ema: s_t = 0,7·p_t + 0,3·s_{t−1} (suavização, reduz variância do predict);
- V-livre (default): s_t = p_t.

Antes de gastar uma sonda em qualquer variante de retenção, ela precisa passar no diagnóstico local de §9.1: rodar a suíte sintética (§10) e confirmar que não reproduz o contraexemplo CE1 (travamento em série sem quebra). A hipótese de trabalho é que V-ema pode ajudar por redução de ruído do modelo e V-hold perde por CE1; só a submissão oficial decide. Implementação: pós-processamento de uma linha sobre `prev_score`, idêntico no harness e na submissão.

---

## 8. Camada supervisionada e conversão para [0,1] (formato de saída, itens 2 e 4)

### 8.1 Construção do dataset de treino (motor único)

Para cada série de treino: rodar a fase-histórico e depois **o mesmo `StreamScorer` da submissão** passo a passo (sem o predict), coletando o vetor de features a cada passo, com rótulo y_t = 1{τ ≤ t}. Não existe implementação vetorizada paralela — princípio do motor único: o código que gera features de treino é o código de inferência, eliminando por construção a classe de bugs "backtest vetorizado ≠ execução causal" (armadilha §13.2). Custo: 10⁴ séries × ~500 passos × ~25 µs ≈ 2–5 min (§11).

**Thinning com correção de peso** (controla o volume de linhas sem viesar): manter todos os passos t ≤ 100; passos 101–400 a cada 2 (peso ×2); passos >400 a cada 4 (peso ×4). Volume esperado ≈ 2,4 M linhas × ~78 features (float32 no dataset de treino) ≈ 750 MB.

### 8.2 Pesos de linha alinhados à métrica

Peso da linha (i,t): `w_row(i,t) = ŵ(t)/n_alive(t)`, normalizado para média 1, onde ŵ(t) = n_pos(t)·n_neg(t) e n_alive(t) são medidos **empiricamente no conjunto de treino** (τ e T conhecidos), multiplicado pelo fator de thinning. Racional: a log-loss ponderada por passo é uma proxy suave da AUC_t ponderada; a divisão por n_alive evita que passos com muitas séries dominem só por contagem. Ablação registrada: treino não-ponderado; decisão de manter os pesos tomada por comparação de submissões oficiais (§9.3), não por métrica calculada localmente.

### 8.3 Modelo e hiperparâmetros de partida

LightGBM binário, um modelo por fold do GroupKFold (5 folds agrupados por `id` da série — **obrigatório**: linhas da mesma série são fortemente autocorrelacionadas e qualquer split não agrupado infla o CV de forma catastrófica, armadilha §13.6). Predição final = média das probabilidades dos 5 modelos.

| Parâmetro | Valor | Nota |
|---|---|---|
| objective / metric | binary / auc | AUC de linha é métrica de treino (early stopping), calculada sobre o rótulo por passo dentro do próprio fold — **não** é usada como estimativa da TS-AUC oficial (ver §9) |
| learning_rate | 0,05 | |
| num_leaves / max_depth | 63 / −1 | |
| min_data_in_leaf | 200 | dataset grande e autocorrelacionado |
| feature_fraction / bagging | 0,8 / 0,8 (freq=1, seed fixa) | |
| lambda_l2 | 5,0 | |
| n_estimators | early stopping (paciência 100) no fold, teto 1500; esperado 400–800 | |
| max_bin | 255 | |
| deterministic / force_row_wise | true / true | determinismo |
| num_threads | treino: 8; **predict: 1** | ordem de operações fixa na inferência |
| seed | 42 em todos os campos de seed | |

### 8.4 Custo de inferência e mitigação de latência

Predict single-row com 5 modelos × ~600 árvores: alvo medido ≤ 120 µs/passo somados. Mitigações em ordem, se a medição exceder: (1) reduzir para 3 modelos (refit em 3 folds maiores); (2) teto de 400 árvores; (3) compilar com lleaves/treelite (dependência extra — só se necessário); (4) último recurso: prever a cada passo com modelo, mas features caras (Bayes) a cada passo e modelo idem — **não** reduzir frequência de emissão do score (a métrica exige score a cada passo).

### 8.5 Fallback puro-estatístico (caminho de emergência, determinístico)

Se a plataforma impuser restrição inesperada ao artefato do modelo, submeter: `score = σ(0,9·LO_{1/400} + 0,4·max_banco_CUSUM_z + 0,3·logM_abs_reset − b)` com b calibrado no treino para mediana 0,5 em séries sem quebra. Serve também de baseline (ii) do gate G-0. Não é o plano A; está aqui para que exista um caminho de submissão em qualquer cenário.

---

## 9. Estratégia de validação local — o que medir e o que não medir (critério de aceite §2.5; formato item 5)

### 9.0 Decisão: a TS-AUC não é replicada localmente como estimador de score

Base desta decisão: experiência relatada pelo usuário em tentativas anteriores de construir uma ferramenta de validação local para este tipo de desafio — o harness local produzia sistematicamente números otimistas demais, indicando um modelo "muito melhor" do que ele se revelava ao ser avaliado pela engine oficial da CrunchDAO. Esse é um padrão de falha conhecido em métricas com harness de scoring complexo (alinhamento exato de passos, definição de "série viva", tratamento de empates, ordem de agregação): pequenas divergências entre a réplica e o motor oficial não geram erro aleatório — geram viés sistemático otimista, porque o pipeline local inevitavelmente é ajustado até "parecer bom", um overfitting ao próprio harness, não aos dados. A causalidade estrita exigida por esta edição (§1.1) torna esse risco ainda maior do que numa métrica estática: qualquer vazamento sutil de t'>t no harness infla exatamente o tipo de score que mede detecção precoce, sem deixar rastro óbvio.

**Decisão adotada por este plano:** nenhuma ferramenta do repositório calcula ou reporta uma estimativa de TS-AUC como substituto do score oficial. Isso não significa abrir mão de validação — significa redistribuir o que é validado localmente (código, aprendizado, comportamento) do que só a engine oficial pode responder (desempenho competitivo real). Consequências práticas:

- Não existe um `ts_auc()` de repositório usado como critério de decisão. Os gates que antes eram expressos como "ΔTS-AUC ≥ x no CV" (G-0, G-mono, G-peso, referenciados em §2.4, §7, §8.2) passam a ser decididos por **comparação de submissões na engine oficial**, dentro do orçamento de sondas (§9.3).
- O **harness causal** (replay ponto a ponto) é mantido — mas seu único papel passa a ser **verificação de corretude de código**: causalidade (teste de prefixo, §12.1) e determinismo (§12.4), nunca estimativa de desempenho.
- Métricas locais continuam a existir, com um papel deliberadamente mais modesto: responder "o modelo está aprendendo algo coerente, as features fazem sentido, o código não vaza nem é instável" — não "este modelo vai tirar 0,87 ou 0,91 no leaderboard". A segunda pergunta só a engine oficial responde com confiança.

### 9.1 O que é medido localmente (diagnóstico, não estimativa de score)

| O quê | Onde | Papel |
|---|---|---|
| Curvas de treino do LightGBM (logloss/AUC de linha por fold, por rodada de boosting) | saída nativa de `train()` | detectar underfit/overfit grosseiro e decidir parada antecipada — é a métrica de treino padrão de um classificador binário por linha, **não** a métrica da competição |
| Importância de features (gain, split count; SHAP se o custo compensar) | relatório de `train()` | entender quais famílias (§5) o modelo está de fato usando; direciona investimento (ex.: se o conformal nunca aparece no topo, investigar por quê) |
| Distribuição do score bruto por fatia de t (histogramas de p para linhas y=0 vs y=1 do treino) | script de diagnóstico | inspeção visual de separação — fica como gráfico para leitura humana, **não** é reduzido a um número de "AUC estimado" |
| Trajetória de score nos cenários sintéticos (T1–T13, §10) | suíte de robustez | comportamento qualitativo (sobe/desce/satura como esperado?) — gates de §10 reescritos para usar diferenças de mediana entre cenário e controle, não AUC |
| Testes de causalidade e determinismo | harness + testes dedicados | corretude de código — ganham prioridade máxima: um vazamento de causalidade não detectado é o mecanismo mais provável por trás de um número local "bom demais" |

### 9.2 Harness causal (mantido, escopo revisado)

O harness continua instanciando `H0Model.fit(hist)`/`StreamScorer` por série e alimentando o online **um ponto por vez** — necessário tanto para construir o dataset de treino (motor único, §8.1) quanto para os dois testes de integridade abaixo. O que muda é que o harness **não agrega scores numa métrica de desempenho**; ele só produz a sequência de scores para (a) virar features de treino ou (b) ser inspecionada pelos testes de causalidade/determinismo/robustez.

- **Canário de vazamento + teste de prefixo:** inalterados (§12.1) — permanecem a defesa de primeira linha contra o tipo de bug que infla métricas locais de forma enganosa.
- **Teste de determinismo:** inalterado (§12.4).

### 9.3 Papel da engine oficial e orçamento de sondas

Toda pergunta do tipo "esta mudança melhora o modelo?" é respondida por submissão à engine oficial, não por validação local:

- Cada submissão é um experimento caro: hipótese registrada por escrito antes de submeter (o que mudou, o que se espera que aconteça).
- Mudanças agrupadas em lotes testáveis, não uma sonda por micro-ajuste; os diagnósticos de §9.1 filtram candidatos obviamente ruins (não aprendeu nada, quebrou determinismo, comportamento sintético absurdo) **antes** de gastar uma sonda.
- Registro de toda submissão (config usada, hipótese, resultado oficial) num log de submissões — esse log é a fonte de verdade histórica do projeto sobre desempenho, substituindo qualquer noção de "CV local".
- Sem alegação de frequência de sondas neste documento — depende do limite real da plataforma, a confirmar operacionalmente.

### 9.4 Esquema de divisão (mantido, propósito revisado)

GroupKFold k=5 por `id`, com estratificação aproximada por (rótulo da série, bucket de T, terço de τ) — mantido, mas agora serve só para (a) treinar o ensemble de 5 modelos (§8.3) e (b) gerar as curvas de treino/importância de §9.1. Não produz mais um número de "validação" a ser comparado com gate algum.

---

## 10. Suíte de robustez — casos sintéticos com gates (critério de aceite §2.7)

Todos os cenários: 200 séries por configuração (seeds 0–199 fixas), n_h = 2000, inovações-base N(0,1) salvo indicado, avaliados **pelo harness causal** com o scorer congelado. "Controle" = gêmeo sem quebra com as mesmas seeds. Gates quantitativos indicados usam **diferença de mediana entre o score do cenário e o do controle**, deliberadamente não uma AUC — são testes comportamentais internos com verdade sintética conhecida, não uma tentativa de estimar a TS-AUC oficial (§9.0). Falha em gate bloqueia a submissão até mitigação.

| ID | Cenário (spec de geração) | O que expõe | Comportamento esperado | Gate |
|---|---|---|---|---|
| T1 | quebra bem no início: τ=3, shift μ=+0,8σ, T=600 | regime de baixa informação inicial + peso quase nulo dos primeiros AUC_t | score sobe ao longo de ~10–30 passos, não instantâneo | mediana(s) ≥ 0,65 e mediana(s_controle) ≤ 0,25 para t≥50 |
| T2 | quebra no fim: τ=T−5, shift +0,8σ, T=600 | não-antecipação (série é negativa até T−5) | score baixo e estável até τ; leve subida depois | média de s em t<τ ≤ 0,35; sem tendência |
| T3 | shift sutil +0,15σ em τ=200, T=800 | piso de sensibilidade; potência ~Φ(δ√m−z): p/ δ=0,15, m=178 dá δ√m≈2 | subida lenta; detectável ~150–250 passos pós-τ | mediana(s) − mediana(s_controle) ≥ 0,15 em t=τ+200 |
| T4 | shift abrupto +1,5σ em τ=200 | canal rápido | subida em <10 passos | mediana(s) ≥ 0,75 e mediana(s_controle) ≤ 0,20 em t=τ+15 |
| T5 | variância 1→1,5 em τ=200 (abrupta) e T5b: rampa linear de σ 1→1,5 ao longo de 200 passos a partir de τ | família variância; quebra suave | subida gradual; T5b mais lenta que T5 | gap de mediana ≥ 0,35 em τ+100 (T5); ≥ 0,20 em τ+200 (T5b) |
| T6 | GARCH(1,1) (ω=0,05, α=0,10, β=0,85) **sem quebra**, T=1000, histórico GARCH idem | falso positivo por vol-clustering; testa a trava de §3.4 e o condicionamento ρ̂₁(\|e\|) | score baixo, com oscilação mas sem drift | média final de s ≤ 0,40; inclinação ~0 |
| T7 | dependência pura: AR φ 0,2→0,6 em τ=200, inovação reescalada p/ manter var incondicional (σ₂²=σ₁²(1−0,6²)/(1−0,2²)) | famílias de dependência (#15–17); ponto cego do Bayes gaussiano | detecção pelos CUSUMs de dependência | gap de mediana ≥ 0,20 em τ+150 |
| T8 | forma pura: N(0,1)→t₄/√2 (var=1) em τ=200 | famílias de cauda/forma (#10–12, #18, #23) | subida por excedências e conformal | gap de mediana ≥ 0,15 em τ+200 |
| T9 | outliers isolados sem quebra: 4 pontos de 6σ em t∈{50,180,420,700}, T=1000 | travamento de alarme falso (CE1); decisão de monotonicidade | picos de score com decaimento em ≤30 passos | s médio em t=999 ≤ 0,35; s(t=210)−s(t=250) ≥ 0,1 (decaiu) |
| T10 | sazonalidade forte no histórico e no online (sen, período 50, amplitude 1) **sem quebra** | qualidade do whitening sazonal (§3.1-3) | sem falso positivo sistemático | média final ≤ 0,40 |
| T11 | T=10 (mínimo), τ=5, shift +1,0σ | fronteira curta; warm-up de features (NaNs) | scores válidos em todos os 10 passos, subida a partir de t=5 | sem NaN/exceção; s(10)>s(4) na média |
| T12 | maratona numérica: T=1000, inovações ~0 (série quase constante) e T12b: \|x\| até 10 alternando | estabilidade das recursões (Welford, log-space, clip) em 1000 passos | sem NaN/Inf/drift; determinismo bit a bit em re-execução | zero NaN; re-execução idêntica |
| T13 | excursão transitória: nível +1σ entre t=200–260, volta ao normal, **sem quebra permanente** | "permanência" da definição de quebra; decaimento pós-excursão | score sobe durante a excursão e decai depois | s(600) ≤ s(260) − 0,15 |

Diagnóstico agregado da suíte: além dos gates, plota-se a trajetória média de s alinhada em τ por cenário — o "retrato de resposta ao degrau" do detector, comparável entre versões do modelo.


---

## 11. Orçamento computacional (critério de aceite §2.6; formato item 6)

Premissas: CPython 3.11, 1 núcleo na inferência, numpy para o filtro bayesiano, LightGBM com `num_threads=1` no predict. Valores são **estimativas de engenharia com microbenchmark obrigatório na fase P1** (gate: total medido ≤ 300 µs/passo).

### 11.1 Custo por passo (inferência)

| Componente | Estimativa/passo | Base da estimativa |
|---|---|---|
| Whitening + escalares (Welford, EWMAs, contadores, 15 CUSUMs) | 3–6 µs | ~50–70 ops float em Python puro a 0,05–0,1 µs/op |
| Ring buffers e 5 janelas | 1–2 µs | ~15 ops + indexação com máscara |
| Filtro bayesiano ×2 hazards (K=48, vetorizado) | 8–15 µs | ~8 chamadas numpy × 1–2 µs de overhead + lgamma cacheado por ν inteiro |
| Martingales conformais (bisect + 12 acumuladores) | 1–2 µs | bisect O(log 5000) ≈ 12 comparações |
| Montagem do vetor de 78 features (array pré-alocado) | 3–5 µs | escrita direta em float64 |
| Predict LightGBM single-row (5 modelos × ~600 árvores) | 30–120 µs | **dominante**; medição obrigatória |
| **Total** | **≈ 50–150 µs/passo** | |

### 11.2 Extrapolação para o volume total

- Passos esperados: 10⁴ séries × E[T]≈500 = 5·10⁶ → **4–13 min**; pior caso 10⁷ passos → **8–25 min**. Público + privado → **< 1 h** de scoring puro.
- Fase-histórico: O(n_h·p + n_h log n_h) por série ⇒ ~0,5–1 ms/série × 10⁴ ≈ **< 15 s**.
- Cenário pessimista de overhead da plataforma (1 ms por chamada de callback): 10⁷ × 1 ms = **2,8 h** — ainda ~5× de folga sobre as 15 h semanais.
- **Nenhum componente é super-linear em T por série**: tudo é O(1), O(K=48) ou O(log n_h) por passo; não há recomputação sobre o prefixo. Memória por série ≈ 50 KB (dominada pelo array ordenado do histórico, ≤ 5000 floats); 10⁴ séries simultâneas ⇒ ~500 MB no teto (mitigação, se preciso: subamostrar o array ordenado para 1024 pontos — degradação negligível do p-value conformal).

### 11.3 Custo do train()

Replay do motor (sem predict) em 10⁴ séries × ~500 passos × ~25 µs ≈ **2–5 min**; dataset ~2,4 M linhas × 78 features float32 ≈ 750 MB; LightGBM 5 folds ≈ **5–15 min** (8 threads); total train() **≤ 30 min**. Suíte de robustez (13 cenários × 200 séries) ≈ 1,6 M passos ≈ **2–4 min** por versão do modelo — barata o bastante para rodar a cada mudança.

### 11.4 Riscos de orçamento e planos B (escada pré-definida)

1. Predict acima do alvo → 3 modelos em vez de 5 → teto de 400 árvores → compilação lleaves/treelite (só se necessário).
2. Overhead de plataforma > 2 ms/passo (medido com contador interno na primeira sonda) → desligar o filtro de hazard 1/100 (corta metade do custo Bayes) e reduzir janelas para {25, 100, 250}.
3. Limite de memória → subamostragem determinística do array ordenado (item 11.2).

---

## 12. Auditoria adversarial (com construções e contraexemplos concretos)

### 12.1 Vazamento de causalidade

- **Teste de prefixo (verificação operacional de causalidade):** para 100 séries × 10 pontos de corte determinísticos: o score no passo t da execução completa deve ser **bit a bit igual** ao último score obtido processando apenas x_{1:t}. Roda no CI a cada commit.
- **Canário:** o repositório mantém `StreamScorerLeaky` (usa x_{t+1} numa média). O teste de prefixo deve REPROVÁ-lo e aprovar o scorer real — prova de que o detector de vazamento detecta.
- **CE5 (lema do t/T):** sob o gerador, P(τ ≤ t | quebra, T) = t/T; ou seja, `t/T` é uma feature "perfeita" do prior condicional — e **exige conhecer T, que é futuro** (A3). Corolário: qualquer quantidade correlacionada com T (metadados, buffers dimensionados por T) é vazamento. Verificação: o objeto de estado não recebe T em nenhuma assinatura; o harness não o expõe.
- **Lista proibida** (revisão de código): T e derivados; estatísticas do segmento online completo; qualquer normalização reajustada no online; reindexações globais.

### 12.2 Comparabilidade transversal do score (consistência com a TS-AUC)

Risco: a escala do score depender de artefatos por série (n_h, caudas) e quebrar o ranking dentro do passo — isso só é observável de fato na engine oficial (§9.0), então a auditoria local foca em identificar e eliminar os artefatos antes de gastar uma sonda, sem fingir medir o efeito em TS-AUC. Construções: (i) ablação sem as meta-features #27–28, inspecionada por importância de features e pelas curvas de treino (§9.1) — se o modelo aprende a usá-las de forma coerente com a teoria (histórico maior reduz falso positivo nos cenários sintéticos de robustez, §10), o condicionamento é mantido; a decisão final de incluí-las ou não vem de submissão oficial, nunca de um ganho de métrica calculado localmente; (ii) por passo t fixo, correlação entre s_t e n_h **dentro das séries negativas** do próprio conjunto de treino, reportada como diagnóstico visual — correlação sem justificativa causal (histórico maior ⇒ H0 mais bem estimado ⇒ menos falso positivo) é artefato a investigar, independente de qualquer número de desempenho; (iii) **CE6 (teste de artefato do gerador):** classificador só-histórico (features do H0, nenhum ponto online) tentando prever o rótulo da série via CV padrão sobre o conjunto de treino; log-loss sistematicamente melhor que a entropia da taxa-base indica que o gerador vaza o rótulo pelo histórico — diagnóstico interno sobre o dataset de treino, não uma tentativa de estimar a TS-AUC oficial. Política registrada: **não** construir features dedicadas a explorar esse vazamento — a organização corrige vazamentos quando os encontra (o changelog W23 é evidência disso, F7) e a exploração ficaria frágil; as meta-features #28 já capturam o que houver de estável e legítimo.

### 12.3 Custo computacional não verificado

Lema B1 (§2.2) quantifica o bloqueio do GLR exato: 5·10⁹ atualizações de candidato por conjunto de teste; em numpy por passo ≈ 1 ms/passo ⇒ 2,8 h só nesse componente. Microbenchmark obrigatório (P1): 100 séries T=1000 pelo scorer completo, `perf_counter` por bloco; gate ≤ 300 µs/passo; escada de mitigação em §11.4.

### 12.4 Determinismo ao longo de 1000 passos recursivos

Fontes de risco e defesas: (i) reduções paralelas → `num_threads=1` no predict, `force_row_wise`+`deterministic` no treino; (ii) ordem de iteração de dict/set → apenas arrays e listas no caminho de inferência (verificado em revisão); (iii) acúmulo de erro float → Welford + log-space + renormalização (§4.3-5); erro relativo esperado ~n·ε_machine ≈ 10³·2⁻⁵² — irrelevante; (iv) RNG → **nenhuma** chamada de aleatoriedade na inferência, com grep automatizado (`random|np\.random|default_rng`) no módulo; (v) versões de bibliotecas pinadas. Teste-espelho do protocolo da plataforma: re-execução de 30% das séries comparada **bit a bit** (mais estrito que a tolerância 1e-8 usada na família de competições, F9).

### 12.5 Comportamento sem quebra (metade do universo)

- **CE1 (contraexemplo do max-hold), formalizado:** série sem quebra, T=1000, outlier 6σ em t=15. Com retenção dura, um pico de ~0,6–0,7 fica travado pelos 985 passos restantes numa série que é negativa em todos eles. No score livre, o candidato bayesiano nascido no outlier perde massa a cada observação normal (razão média e^{−KL} por passo) e o score decai em ~10–30 passos. Este CE motivou a decisão de §7.
- Gates: inclinação média de s_t em negativos ≤ 1e-4/passo em módulo; em 500 séries N(0,1) i.i.d. T=1000, q95 de max_t s_t ≤ 0,7 e fração de séries com s_t > 0,6 por > 100 passos consecutivos ≤ 1%.
- **CE2 (baseline adaptativo):** quebra de σ:1→2 em τ; vol-EWMA com λ_v=0,06 converge à nova variância em ~1/λ_v ≈ 17 passos ⇒ ẽ volta a parecer N(0,1) e um detector de variância sobre ẽ **não vê a quebra**. Defesa estrutural: família variância/cauda roda sempre sobre e com escala congelada (§3.4). Teste dedicado: T5 com e sem a trava — a versão sem trava deve falhar o gate (prova de que o teste morde).
- **CE4 (por que whitening):** AR(1) com φ=0,9 sem quebra infla Var(x̄) por (1+φ)/(1−φ) = 19 ⇒ estatísticas-z de média ~4,4× o nominal ⇒ qualquer CUSUM de média calibrado para i.i.d. dispara. O whitening AR reduz o fator a ~1; a suíte T10 (sazonal) e o bucket "ar_r2 alto" do CV cobrem o residual.

### 12.6 Sensibilidade à posição da quebra (regime de baixa informação)

Declaração explícita do comportamento esperado: com m pontos pós-quebra e shift δ, o poder da estatística ótima é ≈ Φ(δ√m − z_α). Para δ=0,5σ: m=4 ⇒ δ√m=1 (indetectável com confiança), m=16 ⇒ 2 (detectável), m=64 ⇒ 4 (quase certo). O plano **não** promete detecção instantânea; promete subida consistente com esse envelope — e a análise C3 mostra que a métrica cobra pouco pelos primeiros passos pós-τ (peso w_t pequeno quando n_pos(t) é pequeno). Gate: curvas de atraso de T1/T4 dentro do envelope teórico ± 0,15. A poda do filtro bayesiano protege os 8 candidatos mais recentes justamente para não matar hipóteses jovens de quebra recém-nascida (e a quebra pode nascer no primeiro passo — F7).

---

## 13. Armadilhas do cenário online e defesas (critério de aceite §2.8)

1. **Normalização global do segmento online** (vazamento indireto) → todas as escalas vêm do H0 congelado; teste de prefixo (§12.1) pega qualquer regressão.
2. **"Mais tempo ⇒ mais confiança" sem evidência** → pela invariância C1, subida uniforme em t é neutra para AUC_t; a forma danosa é a **heterogênea** (travamento de ruído, drift dependente de artefato) → sem max-hold (§7), gate de drift (§12.5), t presente apenas como condicionador de interações.
3. **Descontinuidade histórico→online** → buffers de lags e last_e atravessam a fronteira (§3.1-8); teste: em séries sem quebra, média de s nos passos 1–5 ≤ média nos passos 6–50 + 0,05.
4. **Instabilidade numérica em recursões longas** → Welford, log-sum-exp, renormalização, clipping, float64; cenário T12; sem subtrações catastróficas (nenhuma variância por "média dos quadrados menos quadrado da média").
5. **Baseline adaptativo absorve a quebra** → B2/CE2: adaptação de vol só para média/dependência/forma; variância/cauda em escala congelada (§3.4).
6. **Autocorrelação das linhas de treino** → GroupKFold por série (§8.3); split aleatório por linha é proibido (inflaria o CV de forma catastrófica).
7. **Usar T ou t/T** → proibido (A3, CE5); o harness não expõe T; nenhuma assinatura recebe T.
8. **Treinar no rótulo da série em vez de y_t = 1{τ≤t}** → ensinaria o modelo a inflar score pré-quebra (a série é negativa ali) e degradaria exatamente os AUC_t que a métrica pesa; o alvo por passo é imposto pela construção do dataset (§8.1) e é o ótimo teórico (C2).
9. **Contaminação de estado entre séries** → um `StreamScorer` novo por série (factory); teste: processar as mesmas séries em ordens distintas produz scores idênticos.
10. **Latência do predict single-row** → medição obrigatória + escada de mitigação (§8.4, §11.4).
11. **Arredondar/quantizar o score** → empates artificiais alteram a AUC (correção de empates); emitir float64 cru (§9.1).
12. **Hazard/priors errados no filtro bayesiano** → dois hazards como features + recalibração supervisionada + ablação de sensibilidade; o filtro é feature, não juiz final.

---

## 14. Riscos de generalização — mitigados vs. em aberto (formato item 7)

| Risco | Status | Mitigação / o que permanece em aberto |
|---|---|---|
| τ muito cedo no online (pouquíssimos pontos pós-quebra) | mitigado parcialmente | envelope de poder declarado (§12.6); candidatos jovens protegidos na poda; peso pequeno da métrica nos primeiros AUC_t (C3). Em aberto: gerador concentrando τ cedo **e** T curto reduziria a folga — monitorar a fatia "τ no 1º terço" do CV |
| τ muito tarde (série negativa quase o tempo todo) | mitigado | não-antecipação verificada (T2); score livre não trava ruído |
| Tipo de quebra fora das famílias cobertas (ex.: estrutura espectral fina, não-linearidade sutil) | em aberto parcial | cobertura genérica fraca via conformal + quantile-crossing; slot reservado para 1 feature dedicada (ex.: energia de \|Δe\|) se as fatias do CV mostrarem buraco |
| Distribuição de T difere treino→teste | mitigado | nenhuma feature usa T; sensibilidade re-ponderando o CV com T~U{10,1000} vs. T empírico do treino |
| Deriva do gerador treino→privado | mitigado parcialmente | só estatísticas com significado sob H0 (nada de memorizar assinaturas do gerador); decisão de complexidade mediada pelo gate G-0 via submissão oficial (§9.3), não por métrica local; suíte sintética independente do treino como segundo eixo de validação comportamental |
| Interpretação exata do rótulo por passo (A1) | em aberto até confirmação | avaliador local calcula as duas leituras (§9.4); confirmar na documentação/fórum oficial antes da 1ª sonda de LB |
| Artefato de histórico que vaza o rótulo (CE6) | em aberto, decisão registrada | não explorar deliberadamente; a organização corrige vazamentos (F7) — construir sobre exploit é frágil |
| Assinatura exata do template/callback oficial | em aberto até P0 | design encapsulado (H0Model/StreamScorer) + adaptador fino ao quickstarter oficial (F8, F10) como primeira tarefa |

---

## 15. Esqueleto de implementação, checklist de determinismo e fases

### 15.1 Contratos de código

```python
class H0Model:
    def fit(self, hist: np.ndarray) -> "H0Model":
        """§3.1: AR(10) Yule-Walker, sazonal opcional, escalas, quantis,
        arrays ordenados, sigma_u, nu_hat. Determinístico, O(n_h·p + n_h log n_h)."""

class StreamScorer:
    def __init__(self, h0: H0Model, ensemble):   # aloca TODO o estado de §4
        ...
    def update(self, x: float) -> float:         # UMA observação → UM score
        e = self._whiten(x)                      # §3.2 (lags atravessam a fronteira)
        self._update_blocks(e)                   # §4.2–4.3: O(1)/O(K)
        f = self._assemble()                     # 78 floats, buffer pré-alocado
        p = self._predict(f)                     # média dos 5 LightGBM (ou fallback §8.5)
        return self._postprocess(p)              # default V-livre: identidade

# train(X_train, y_train, model_dir):
#   para cada série: h0 = H0Model().fit(hist)
#       replay do online com StreamScorer(h0, ensemble=None) coletando
#       (features, y_t = 1{tau <= t}, peso §8.2, id)
#   GroupKFold(5, groups=id) -> 5 LightGBM (§8.3) -> salvar modelos + versão
#
# infer(...):  # adaptar à assinatura EXATA do template oficial (F8/F10) na fase P0
#   h0 = H0Model().fit(historico_recebido)
#   scorer = StreamScorer(h0, ensemble)
#   para cada observação do stream:  emitir scorer.update(x)   # um score por ponto
```

O adaptador ao template é a única parte não especificável hoje (assinatura do callback oficial); todo o resto independe dela por construção.

### 15.2 Checklist de determinismo (pré-submissão)

[ ] grep anti-RNG limpo no módulo de inferência • [ ] `num_threads=1` no predict • [ ] `deterministic=true` + `force_row_wise=true` no treino • [ ] todos os seeds = 42 • [ ] re-execução de 30% bit a bit idêntica • [ ] teste de prefixo aprovado (e canário reprovado) • [ ] T12 sem NaN/Inf • [ ] nenhuma iteração sobre set/dict no caminho de inferência • [ ] versões de bibliotecas pinadas • [ ] score emitido em float64 sem arredondamento.

### 15.3 Fases de execução com gates de saída

- **P0 — fundação:** adaptador ao template oficial + harness causal (verificação de corretude, não estimador de score — §9) + fallback estatístico (§8.5) funcionando ponta a ponta. Sonda 1 de submissão com o fallback (valida o pipeline e mede overhead de plataforma; não é usada para comparar arquiteturas). Gate: teste de prefixo + determinismo verdes.
- **P1 — motor de estado:** implementação completa de §3–§5 + microbenchmark (≤ 300 µs/passo) + suíte T1–T13 (gates de mediana, §10) rodando sobre o fallback. Gate: todos os gates comportamentais da suíte para o fallback documentados (mesmo os reprovados — são a régua do ganho do modelo).
- **P2 — camada supervisionada:** dataset por passo + GroupKFold + LightGBM + diagnósticos de §9.1 (curvas de treino, importância de features). Gate G-0 (híbrido vs. melhor linha única) decidido por submissão oficial dedicada, não por CV local.
- **P3 — ablações registradas:** monotonicidade (G-mono), pesos (G-peso), hazards, meta-features (§12.2), fatias de diagnóstico por τ/T/nuisance sobre o treino. Gate: cada decisão congelada por resultado de submissão oficial, com os diagnósticos locais usados só para pré-filtrar candidatos antes de gastar uma sonda.
- **P4 — congelamento:** checklist §15.2, re-execução da suíte, submissões adicionais apenas confirmatórias. Nenhuma decisão de arquitetura após P3.

---

## 16. Fontes

### Competição e ecossistema (base factual das escolhas)

- **F1.** Brandão, H. — writeup do 1º lugar da edição 2025 (stacking de árvores sobre 4 blocos de features estatísticas; ~0,9014 AUC privado): https://humbertobrandao.medium.com/how-far-can-we-push-the-winning-model-of-the-adia-lab-structural-break-challenge-87ebf3d0ff67
- **F2.** Sağlam, F. — 2º lugar 2025 (LightGBM sobre ~2400 features estatísticas com seleção SHAP/gain): https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution
- **F3.** Soisson, G. — solução pública 2025 (views z/Δz/Δ²z, padronização robusta pelo segmento de referência, GBMs empilhados): https://github.com/gsoisson/adia-structural-break
- **F4.** Documentação oficial da edição 2025 (estrutura de dados, ROC AUC, tipos de quebra: média, variância, forma, dependência, cauda): https://docs.crunchdao.com/competitions/competitions/adia-lab-structural-break-challenge e https://hub.crunchdao.com/competitions/structural-break
- **F5.** Evidência pública de fracasso de redes puras no gerador (CNN+RNN ≈ 0,5 AUC): https://github.com/hawuxparrot/CrunchDao_structural_break_09_25
- **F6.** Benchmark público de 25 métodos (transformer hierárquico 0,49–0,54 AUC; árvores dominam): https://github.com/waddadaa/structural_break_detection
- **F7.** Changelog oficial W23/2026 da Real-Time Edition (correção de 29 séries com quebra no primeiríssimo passo mal rotuladas; correção de vazamento de valores completos no ambiente de nuvem): https://forum.crunchdao.com/t/2026-w23-structural-break-fixes/1156
- **F8.** Páginas oficiais da Real-Time Edition (mecânica de score por observação, quebra em ponto desconhecido ou ausente, TS-AUC por passo entre séries vivas, premiação no ADIA Lab Symposium out/2026): https://www.adialab.ae/adia-lab-x-crunch-the-structural-break-challenge-2026 , https://hub.crunchdao.com/competitions/structural-break-real-time e resumo público em https://internshala.com/competitions/adia-lab-structural-break-challenge-real-time-edition/
- **F9.** Open Benchmark da mesma família (protocolo de re-execução para verificação de determinismo com tolerância 1e-8; restrição de originalidade): https://hub.crunchdao.com/competitions/structural-break-open-benchmark
- **F10.** Baseline oficial da Real-Time Edition (referência do template de submissão): https://www.kaggle.com/code/crunchdao/structural-break-real-time-baseline

### Literatura de método

- Page, E.S. (1954). "Continuous Inspection Schemes". *Biometrika* 41 — CUSUM.
- Roberts, S.W. (1959). "Control Chart Tests Based on Geometric Moving Averages". *Technometrics* — EWMA.
- Shiryaev, A.N. (1963). "On Optimum Methods in Quickest Detection Problems". *Theory Probab. Appl.* — detecção bayesiana de desordem (base do filtro §4.3).
- Welford, B.P. (1962). "Note on a Method for Calculating Corrected Sums of Squares and Products". *Technometrics* — recursão estável de variância.
- Hinkley, D.V. (1971). "Inference about the change-point from cumulative sum tests". *Biometrika* — Page–Hinkley.
- Lorden, G. (1971). "Procedures for Reacting to a Change in Distribution". *Ann. Math. Statist.* — otimalidade minimax assintótica.
- Willsky, A. & Jones, H. (1976). "A generalized likelihood ratio approach to the detection and estimation of jumps in linear systems". *IEEE TAC* — GLR sequencial.
- Pollak, M. (1985). "Optimal Detection of a Change in Distribution". *Ann. Statist.* — Shiryaev–Roberts.
- Moustakides, G.V. (1986). "Optimal Stopping Times for Detecting Changes in Distributions". *Ann. Statist.* — otimalidade exata do CUSUM.
- Basseville, M. & Nikiforov, I. (1993). *Detection of Abrupt Changes: Theory and Application*. Prentice-Hall.
- Lai, T.L. (1998). "Information Bounds and Quick Detection of Parameter Changes in Stochastic Systems". *IEEE Trans. Inf. Theory* — GLR window-limited (justifica o banco de janelas).
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer — p-values conformais e martingales de exchangeability.
- Adams, R.P. & MacKay, D.J.C. (2007). "Bayesian Online Changepoint Detection". arXiv:0710.3742 — BOCPD (recursão de run-length; aqui restrita a uma mudança).
- Fearnhead, P. & Liu, Z. (2007). "On-line inference for multiple changepoint problems". *JRSS-B* — filtragem exata/partículas com poda.
- Murphy, K. (2007). "Conjugate Bayesian analysis of the Gaussian distribution" (notas técnicas) — preditivas Normal-Inv-χ² usadas em §4.3.
- Tartakovsky, A., Nikiforov, I. & Basseville, M. (2014). *Sequential Analysis: Hypothesis Testing and Changepoint Detection*. CRC.
- Volkhonskiy, D. et al. (2017). "Inductive Conformal Martingales for Change-Point Detection". *COPA/PMLR* — martingales conformais para CPD online.
- Ke, G. et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree". *NeurIPS*.

— Fim do plano —
