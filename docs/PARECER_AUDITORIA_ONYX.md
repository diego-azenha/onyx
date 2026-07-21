# Parecer técnico — Auditoria do `onyx` / structural-break-rt

**Papel assumido:** revisor sênior em detecção sequencial de mudanças e aprendizado estatístico online, sem envolvimento prévio no projeto.
**Material auditado:** `docs/PLANO_TECNICO.md`, `docs/PLANO_REPOSITORIO.md`, `docs/CONTRACTS.md`, `docs/DIAGNOSTICO_TS_AUC.md`, `plano_acao_v1_para_v2.md`, todo o código de `src/sbrt/`, `scripts/`, `tests/`, `configs/default.yaml` e os artefatos versionados em `resources/` (schema de 80 features, `fold_evals.json`, `base_rate_curve.json`).
**Verificações executadas nesta auditoria:** suíte `tests/unit` + `tests/causality` (53/53 PASS no ambiente da auditoria); inspeção dos `fold_evals` do modelo commitado; verificação aritmética independente da equivalência de atenuação whitening-vs-cru e da estrutura de pesos da métrica (detalhes em §4.2 e §6).
**Data:** 2026-07-20.

---

## 1. Parecer em uma página

O projeto é, em engenharia e em leitura matemática da métrica, um dos trabalhos mais disciplinados que já revisei nesse formato de competição. O rótulo por passo, a análise de invariância C1–C3, o motor único, o teste de prefixo com canário, o determinismo bit-a-bit e a decisão contra o max-hold estão todos corretos e bem fundamentados — nada disso deve ser revisitado. O pipeline não está quebrado; ele está **extraindo menos do que os dados provavelmente contêm**, e a evidência disponível aponta com razoável precisão onde.

Minha opinião central, em quatro afirmações:

**(1) O achado "de maior potencial" do DIAGNOSTICO — o descasamento de escala do `predict_one` — não pode, por construção, mover a TS-AUC.** Qualquer recalibração pós-hoc (Platt, isotônica, readição do offset) é uma transformação monótona aplicada uniformemente a todas as séries em cada passo; pela própria invariância C1 que o plano técnico demonstra, a AUC transversal é idêntica antes e depois. O descasamento de escala explica por que os gates *absolutos* da suíte falham — e a metade (b) da recomendação, redesenhar os gates para serem relativos, está certa e deve ser feita — mas ele **não explica nem um milésimo do 0,60**. O 0,60 é fraqueza de *ranking*, e ponto. (§5, D1–D2.)

**(2) A previsão falsificável nº 1 do plano de ação falhou — e isso é o dado mais informativo do projeto hoje.** O `plano_acao` previu que, corrigido o objetivo (init_score + logloss), a contagem de árvores subiria de ~90 para 300–800, e registrou: "se continuar em ~90, meu diagnóstico está errado e o problema é falta de sinal, não de objetivo". Os `fold_evals.json` do modelo commitado mostram best-iterations de **69, 89, 61, 84, 85** — pós-A2. Pelo critério do próprio autor, a tese "o objetivo era o gargalo" está, ao menos parcialmente, falsificada. Minha leitura refinada: o offset de taxa-base era necessário mas não suficiente, porque a família de objetivos *pontuais agrupados* satura rápido num problema cujo n efetivo é ~10.000 séries, não 2,5M linhas — e o que resta desalinhado não é o *offset* do objetivo, é a sua *estrutura* (pesos por classe dentro de cada t, e o fato de a métrica ser um ranking por grupo). (§4.5, D3.)

**(3) Existe um desalinhamento concreto, derivável e ainda não testado nos pesos de linha.** A TS-AUC se reescreve exatamente como *fração de pares concordantes (positivo, negativo) do mesmo passo, agregada sobre todos os passos*. O surrogate pontual consistente com isso dá a cada positivo do passo t peso ∝ n_neg(t) e a cada negativo peso ∝ n_pos(t). O esquema atual (`weights.py`) dá o mesmo peso às duas classes — o que, em t≤50, deixa a massa de gradiente dos positivos em ~8% da dos negativos, exatamente no bucket onde a AUC é 0,53 e que carrega, junto com 50–150, ~35% do peso da métrica. É uma mudança de ~5 linhas, com mecanismo limpo, previsão falsificável e que ataca diretamente a fraqueza documentada em t precoce. (§4.4, R1.)

**(4) A regra §9.0, na sua forma absoluta, está custando mais do que protege.** O trauma que a originou (réplicas locais da engine oficial sistematicamente otimistas, na edição 2025) era real — mas o instrumento que o projeto usa hoje, TS-AUC *out-of-fold* com GroupKFold sobre 10.000 séries, é um instrumento diferente, com modos de falha diferentes, e o próprio DIAGNOSTICO já o usa de facto como juiz de todas as decisões (0,5996 vs 0,5961). O problema real da rodada de intervenções não foi usar o OOF — foi usá-lo **sem barra de erro pareada**: uma diferença de −0,0035 foi tratada como conclusiva sem intervalo de confiança. Proponho formalizar o que já acontece: OOF pareado com bootstrap por série como juiz *relativo*, submissão oficial como âncora *absoluta*. (§5, D4; R0.)

O roadmap (§6) tem seis itens, ordenados por informação-por-hora: **R0** (metrologia: comparador OOF pareado com IC), **R1** (pesos pareados por classe dentro de t), **R2** (early stopping e hiperparâmetros julgados por AUC-por-t no fold), **R3** (objetivo de ranking por grupo t, como modelo paralelo), **R4** (a família de features que falta: estatísticas de duas amostras *rank-based* contra o histórico — o análogo causal do que venceu 2025), **R5** (gates da suíte convertidos para relativos). Tudo validado pelo protocolo completo da seção 3.8 do DIAGNOSTICO — que é, junto com a análise de ruído H1, a melhor contribuição metodológica daquele documento.

---

## 2. O que a evidência disponível estabelece (base factual do parecer)

Antes de opinar, o que considero *estabelecido* pelos documentos, pelo código e pelas verificações desta auditoria:

**E1 — Habilidade real ~0,60, com perfil conhecido.** TS-AUC OOF 0,5996–0,601 sobre 10.000 séries; por bucket: 0,529–0,530 (t≤50, 8,1% do peso), 0,566–0,568 (50–150, 26,6%), 0,615–0,617 (150–400, 48,7%), 0,641–0,643 (>400, 16,5%). O reamostrado de 100 séries tem σ≈0,054 — logo o 0,5244 do teste reduzido era percentil ~5 da distribuição amostral do próprio modelo, não um sinal. (DIAGNOSTICO §4.1–4.2; confirmo a análise integralmente.)

**E2 — O modelo pós-A2 satura em 61–89 árvores.** Verificado nesta auditoria em `resources/fold_evals.json` (rounds gravados 161–189 = best + 100 de paciência). A logloss de validação genuinamente atinge o mínimo ali e piora por 100 rodadas — não é artefato de paciência curta.

**E3 — O sinal dominante é de variância/cauda; o de média é estruturalmente fraco.** Correlações |mean_z|-com-y de 0,003–0,037 em qualquer janela vs. 0,06–0,09 para variância; famílias de média mortas em quatro implementações independentes (janelas, EWMAs, CUSUMs, hedge cru). A refutação de H3 no DIAGNOSTICO é sólida, e a explicação teórica do `plano_acao` está **correta e foi verificada aritmeticamente nesta auditoria**: para AR(1) com marginal unitária, o z por δ√m é √((1−φ)/(1+φ)) *tanto* nas inovações quanto na média crua — a atenuação é limite de informação do problema, não culpa do whitening.

**E4 — CE6 nulo.** AUC 0,5067 do classificador só-histórico: o gerador não vaza o rótulo pelo H0. Uso de `meta_h0_*` como efeito principal é ruído a priori; como interação (condicionamento), é o papel que o plano previu — e o padrão ganho/split (0,57–0,62 por split, ~600 splits) é consistente com interação, não com offset.

**E5 — Três intervenções pontuais bem motivadas, zero ganho líquido, uma regressão; revertidas.** E o achado colateral decisivo: 9/15 gates da suíte já falhavam com o calibrador supervisionado *antes* de qualquer intervenção, por incompatibilidade de escala absoluta — enquanto a discriminação relativa em T4 é visivelmente correta. (DIAGNOSTICO §4.6–4.7.)

**E6 — O mecanismo de T6 (GARCH → falso positivo de variância) está corretamente identificado** no código: a trava `vol_adjust` protege média/dependência/forma, e a família de variância consome deliberadamente `e` congelado (defesa CE2). O falso positivo em vol-clustering é o preço estrutural dessa defesa; a feature `volvol` foi adicionada como discriminador e a tentativa de canal adicional vol-ajustado não gerou ganho líquido.

**E7 — Engenharia verificada.** 53/53 testes unit+causality passam; o canário de vazamento é de fato reprovado pelo teste de prefixo; determinismo por re-execução bit-a-bit implementado; o motor único é real (dataset.py chama `update_features`, nenhuma reimplementação vetorizada existe); orçamento de latência com ~10–16× de folga.

**Um fato que reformula a discussão de capacidade: o n efetivo é ~10⁴, não 2,5M.** O rótulo inteiro de uma série é determinado por um único τ_i; as linhas da mesma série são quase perfeitamente dependentes; a generalização (GroupKFold) é entre séries. Toda intuição de hiperparâmetro calibrada para "2,5M linhas iid" (min_data_in_leaf=200, folga para 63 folhas) deve ser relida para um problema de ~10.000 exemplos independentes com features riquíssimas. A saturação em 60–90 árvores (E2) é exatamente o que se espera quando o juiz é uma logloss pontual sobre um n efetivo dessa ordem.

---

## 3. Auditoria por componente

Formato de cada item: motivação → a teoria sustenta? → funciona na prática? → hipóteses implícitas → "eu tomaria essa decisão hoje?".

### 3.1 Modelagem do problema e formulação probabilística — **excelente**

O rótulo por passo `y_t = 1{τ≤t}` é o alvo correto (C2 é a aplicação padrão de Neyman–Pearson por passo) e a análise C1/C3 é o tipo de leitura de métrica que a maioria dos competidores nunca faz. Acrescento uma reformulação que torna C3 operacional e fundamenta §4.4: como AUC_t = (concordâncias no passo t)/(n_pos·n_neg) e w_t = n_pos·n_neg, os pesos cancelam e

> **TS-AUC = fração de pares (positivo, negativo) do mesmo passo corretamente ordenados, agregada sobre todos os passos.**

Ou seja, a métrica é uma única probabilidade de concordância sobre o pool de pares intra-passo. Tudo que o treino deveria fazer é maximizar essa fração — e essa forma fechada diz exatamente qual peso cada *linha* deveria carregar num surrogate pontual (§4.4). Eu tomaria todas as decisões desta seção de novo, e adotaria a reformulação acima como lente padrão.

### 3.2 Whitening causal e construção do H0 — **boa; hipóteses implícitas administradas, com um resíduo de comparabilidade**

Motivação e teoria sólidas (CE4 quantifica o falso positivo por autocorrelação; o whitening preserva todas as famílias de quebra). Verifiquei a implementação: AR(10) por OLS com aceitação por ar_r2 ≥ 2%, sazonal parcimonioso (limiar 0,25 ≈ 11 desvios do ACF nulo com n_h≈2000 — praticamente nunca dispara por ruído, bom), continuidade exata na fronteira via `lag_seed`, clipping ±8 com indicadores crus preservando cauda. Três observações:

- **A equivalência de atenuação (E3) absolve o whitening** da acusação implícita em H3 — mas deixa uma consequência prática: a grade de δ do CUSUM de média ({0,25; 0,5; 1,0}) está em *unidades de inovação*; para uma série com Σφ̂→1, um shift de 0,5σ_x vira ~0,1 em unidade de inovação — abaixo da grade. Isso não é um bug a corrigir (δ=0,15 já foi testado e regrediu T13); é um limite a *documentar* e uma razão para as janelas longas (`welford_mean_z`) serem o canal de média que resta para séries persistentes.
- **Efeito de estimação do H0** (literatura de "Phase I estimation" em SPC): com n_h∈[1000,5000], σ̂_e tem erro relativo ~1–2%, o que desloca `ln var` por ±0,02–0,04 por série — pequeno, mas não nulo num regime em que as margens de discriminação são finas. As `meta_h0_*` dão ao modelo o material para descontar isso; a solução mais direta é a família rank-based de R4, que é invariante a erro de escala por construção.
- Hoje eu manteria AR(10) fixo (seleção de ordem adicionaria variância e não-determinismo de decisão por quase nada).

### 3.3 Roteamento `e`/`e_vol` e o dilema CE2×T6 — **decisão certa; o resíduo se trata no modelo, não no baseline**

A trava anti-absorção é a decisão assimétrica correta: um baseline de volatilidade adaptativo cegaria quebras de variância em ~17 passos (CE2), e isso é pior do que o falso positivo de T6, porque quebras de variância são o sinal dominante do gerador (E3). O custo — clusters GARCH lidos como evidência pela família de variância *e pelo filtro bayesiano* (que também consome `e` congelado) — é real, e a tentativa de canal duplo não pagou. O caminho restante é dar ao calibrador discriminadores explícitos de "burst vs. patamar": `volvol` já existe; contrastes entre escalas de janela de variância (nível recente vs. nível defasado) são a extensão natural, mas eu os classificaria como refinamento de baixa prioridade dado o histórico de intervenções pontuais nulas — não como gargalo. Eu tomaria a mesma decisão hoje.

### 3.4 Banco de CUSUM — **bom; redundância barata; não investir mais**

Minimax-ótimo por alternativa simples, O(1), recursão max varrendo τ implícito — teoria impecável, e o SHAP (21,6%, crescendo com t) confirma que é um dos dois cavalos de trabalho. A redundância interna (thresholds altos subsumidos pelo δ=0,25; Bernoulli de sinal e excedência sobrepondo janelas de fração) custa quase nada em árvores e nada relevante em latência. As idades como localizadores baratos são uma ideia elegante que o modelo usa pouco — manter, não expandir. Veredicto: manter como está; a rodada revertida já demonstrou que mexer na grade não é onde está o ganho.

### 3.5 Filtro bayesiano de troca única — **bom no papel, retorno modesto na prática; congelar, não estender**

A motivação (emite exatamente o alvo C2; duas-amostras adaptativas de graça) é a mais bonita do projeto, a implementação está correta (verifiquei a recursão NIχ², a poda com proteção dos 8 recentes, a renormalização em log-espaço, o cache de lgamma), e o custo cabe. Mas a contribuição real medida é 3,4% de |SHAP|, estável em todos os buckets, e o terceiro hazard (1/50) teve resultado nulo. Duas hipóteses implícitas merecem registro: (i) H0 gaussiano exato para `e` — em séries de cauda pesada legítima, ℓ₀ pune outliers e o LO infla sob H0, um artefato de comparabilidade transversal que o clipping limita e as `meta_h0_*` descontam imperfeitamente; (ii) pós-mudança Normal — quebras de dependência/forma ficam fora do modelo (coberto por design pelas outras famílias). Se eu começasse hoje, implementaria o filtro do mesmo jeito — mas com a expectativa calibrada de que ele é um membro do coro, não o solista, e **não** gastaria mais nenhuma iteração nele (t-likelihood com ν̂ fica no estacionamento, reabrível se R4 mostrar que caudas pesadas são uma fatia grande do gerador).

### 3.6 Martingales conformais — **os p-values valem ouro; a agregação em martingale é o elo fraco**

Aqui discordo parcialmente do design. Os *p-values* conformais (rank de e_t contra o histórico ordenado, O(log n_h)) são exatamente a moeda certa para este problema: livres de distribuição e, crucialmente, **comparáveis entre séries por construção** — a propriedade que a TS-AUC premia e que as estatísticas paramétricas só têm aproximadamente. O que rende pouco (2,2% de SHAP) é o *agregador*: o martingale de apostas de Vovk com mistura de ε é otimizado para controle de erro tipo Ville sob H0, não para potência de ranking — a evidência acumula devagar e a mistura de ε dilui. A correção não é remover o bloco; é **reaproveitar os mesmos p-values num agregador de potência**: médias de janela de (p−½) são estatísticas de Wilcoxon/Mann–Whitney da janela contra o histórico — a família de duas amostras rank-based que é o núcleo de R4. Hoje eu teria começado por aí e deixado o martingale como feature secundária.

### 3.7 Acumuladores, EWMAs e janelas — **o cavalo de trabalho; um upgrade de comparabilidade disponível**

33% do |SHAP|; implementação O(1) correta (Welford, rings com soma incremental, `volvol`). Uma assimetria digna de nota: as estatísticas de *média* saem padronizadas (z), mas as de *variância* saem como `ln(médias de e²)` cruas — cuja variância amostral sob H0 é ≈(κ−1)/w, dependente da curtose da série. Duas séries com a mesma quebra de variância e caudas diferentes produzem `accum_window_var_ln` em escalas diferentes; o modelo precisa aprender a correção via interação com `meta_h0_nu_hat`. Padronizar explicitamente (z_var = (mean_e²_w − 1)/√((κ̂_hist−1)/w), com κ̂ do histórico) é barato e ataca comparabilidade transversal — mas o caminho rank-based de R4 resolve o mesmo problema de forma mais radical, então classifico esta padronização como variante interna de R4, não item próprio.

### 3.8 Meta-features (t, H0, localizadores) — **manter; a questão já foi julgada empiricamente**

Pós-A2, `meta_t`/`meta_ln1p_t` perdem a função de raiz (o offset já modela a taxa-base) e sobram como condicionadores — inócuo por C1. As `meta_h0_*` foram alvo da Intervenção 2, que não sobreviveu ao ciclo completo; com CE6 nulo e o padrão de splits indicando interação, o veredicto empírico é "deixar quieto". Concordo e não reabriria — com uma exceção condicionada: se R3 (ranking por grupo t) for adotado, `meta_t` vira literalmente constante dentro de cada grupo e pode ser removida de graça.

### 3.9 Dataset, thinning e motor único — **excelente engenharia**

O motor único é a decisão de engenharia mais valiosa do repositório — elimina por construção a classe de bug que destrói projetos desse tipo — e o teste de prefixo + canário provam que a defesa morde (verifiquei: o canário é reprovado). Thinning com correção de peso está correto (o corte é por t, então cada t mantido tem a seção transversal completa — os pesos por t medidos ali são fiéis). O fix de memória por-série está bem documentado. Nada a mudar.

### 3.10 Pesos de linha — **o desalinhamento estrutural mais concreto e barato do projeto** (→ R1)

O esquema atual dá a toda linha do passo t o peso n_pos(t)·n_neg(t)/n_alive(t): a *massa por passo* está certa (∝ w_t), mas a *partição entre classes* está errada. Pela reformulação de §3.1, cada positivo do passo t participa de n_neg(t) pares e cada negativo de n_pos(t); o surrogate pontual consistente com a fração de pares concordantes é, portanto,

> w(linha positiva em t) ∝ n_neg(t)  e  w(linha negativa em t) ∝ n_pos(t)  (× fator de thinning),

que equaliza a massa de perda das duas classes dentro de cada passo (ambas = n_pos·n_neg). Hoje, em t≤50, a razão de massa positivos:negativos é ~0,08 — o gradiente que ensinaria o modelo a subir os poucos positivos precoces acima dos milhares de negativos está diluído por um fator ~12, precisamente no bucket com AUC 0,53 e ~8% do peso, adjacente ao bucket 50–150 (26,6% do peso, AUC 0,57) com o mesmo problema em grau menor. O `init_score` corrige o *offset* por t; não corrige esta *alocação de gradiente* — são mecanismos ortogonais. É a mudança com melhor razão (mecanismo derivável)/(custo) de todo o meu parecer, e tem previsões falsificáveis limpas (§6-R1). Cautela honesta, aprendida do próprio projeto: mecanismo limpo não garante ganho — por isso R1 vem *depois* de R0 (o juiz com barra de erro).

### 3.11 Função objetivo, LightGBM e treinamento — **A2 estava certa e é insuficiente; o próximo passo é estrutura, não offset** (→ R2, R3)

O `init_score = logit(p̂(t))` e a troca para logloss no early stopping são exatamente o que eu teria prescrito — e o fato de as árvores continuarem em 61–89 (E2) é informação, não fracasso: significa que a logloss pontual do resíduo esgota rápido o que consegue ver com n efetivo ~10⁴. Restam três alavancas na mesma direção, em ordem de fragilidade crescente: (i) os pesos de R1 (mudam o que o gradiente enfatiza); (ii) trocar o *juiz da parada e da seleção de hiperparâmetros* para a coisa certa — AUC ponderada por passo, calculada dentro do fold de validação (um `feval` custom; isso é critério interno de treino, não estimador de leaderboard — ver D4); sob esse juiz, revisitar lr (0,05→0,02) e a dupla min_data_in_leaf/lambda_l2 vale a pena, agora com a régua certa, absorvendo a recomendação 2 do DIAGNOSTICO; (iii) o objetivo de ranking por grupo t (lambdarank/rank_xendcg com query = passo), que torna estruturalmente impossível gastar capacidade em qualquer coisa constante-dentro-do-passo. O `plano_acao` deixou (iii) condicional a "A2 melhorar e não bastar"; dado E2, a condição está satisfeita — promovo A6 a candidato de primeira linha, com as ressalvas técnicas registradas em §6-R3 (truncation level, grupos grandes, rótulo binário). Detalhe correto que confirmo no código: o offset é somado de volta no OOF (`train.py`) mas não no `predict_one` — ambos C1-neutros; sem inconsistência.

### 3.12 Ensemble (5 folds, média de probabilidades) — **adequado; diversidade real fica para depois**

Média de sigmoides do resíduo entre folds: leve não-linearidade vs. média de logits, irrelevante para ranking na prática. A diversidade que valeria algo não é entre folds do mesmo modelo, é entre *objetivos* (binário-ponderado × ranking, R3, combinados por rank-average) — essa é a versão de "stacking" que faz sentido aqui, e só depois de cada membro justificar sua existência sob R0. GRU/redes permanecem corretamente estacionadas (evidência pública F5/F6 + custo de desenvolvimento).

### 3.13 Pós-processamento (score livre) — **correto; encerrado**

CE1 é um contraexemplo genuíno, C2 fundamenta a não-monotonicidade do posterior, e 50% do universo sem quebra torna o custo do travamento de primeira ordem. V-ema segue como única variante com hipótese plausível (redução de variância do predict) — testável de graça quando houver uma sonda sobrando, nunca antes.

### 3.14 Fallback estatístico — **cumpre o papel; não é mais baseline de comparação**

Como caminho de emergência determinístico, está bem construído (a transformação √(2·LLR) para escala z é um truque legítimo de combinação). Como baseline científico, foi superado pelo protocolo OOF — as comparações fallback-vs-modelo em 100 séries são ruído (a "manchete" 0,5249 vs 0,5385 que o `plano_acao` corretamente desmontou).

### 3.15 Validação: harness, suíte de robustez e a regra §9.0 — **a melhor e a mais datada parte do projeto, no mesmo lugar**

O harness causal, o canário e o determinismo são exemplares. A suíte T1–T13 é uma ideia excelente com uma incompatibilidade agora comprovada: gates *absolutos* pressupõem um score calibrado em [0,1], e o design deliberado do calibrador (resíduo sem offset) não entrega isso — 9/15 falhas pré-existentes são a suíte medindo a régua errada, não o detector errando (E5). Concordo integralmente com a recomendação 1(b) do DIAGNOSTICO: converter T2/T6/T9/T10/T13 para gates relativos (cenário vs. painel de referência com as mesmas seeds), mantendo os absolutos apenas para o modo fallback (§6-R5). Sobre a §9.0, meu dissenso está em D4: a regra acertou o diagnóstico histórico (réplica de harness ⇒ viés otimista sistemático) e errou a generalização (proibir *qualquer* número local de desempenho) — o projeto já vive fora dela, e o que falta é o rigor estatístico (IC pareado), não a proibição.

---

## 4. Análise dirigida dos quatro nós técnicos

### 4.1 Por que 0,60 — e por que o teto provável é mais alto

Comparações diretas com o 0,90 da edição batch são injustas (lá o método via o segmento pós-quebra inteiro; aqui, em t pequeno, ninguém vê quase nada). Mas o bucket 150<t≤400 — 48,7% do peso — permite uma comparação honesta: uma série positiva típica nesse bucket já acumulou dezenas a centenas de pontos pós-quebra, contra um histórico de 1000–5000. Isso é *quase o regime batch* para uma fração substancial dos positivos, e um teste de duas amostras rico (o paradigma vencedor de 2025) opera nesse regime muito acima de 0,617 — a menos que o gerador de 2026 tenha magnitudes drasticamente menores, o que o censo A1 mediria. Minha leitura: o bucket intermediário está deixando sinal na mesa por limitação de *representação e objetivo*, não de informação; já o bucket t≤50 pode estar perto do teto de informação (poucos pontos pós-quebra, envelope Φ(δ√m−z)) — e distinguir os dois casos é exatamente o que R0+R6 permitem fazer com números em vez de opinião.

### 4.2 A equivalência de atenuação (verificada) e o que ela implica

Verifiquei numericamente: para AR(1) com variância marginal 1, o z por unidade de δ√m é √((1−φ)/(1+φ)) idêntico nas inovações e na média crua (φ=0,9 ⇒ 0,229; φ=0,99 ⇒ 0,071). Implicações: (a) não há bug de whitening a caçar no canal de média — encerrar essa linha; (b) o hedge cru não-branqueado é redundante em teoria para média (e o gain zero confirma), útil só como seguro contra AR mal ajustado; (c) a única melhora possível para média em séries persistentes é *janela mais longa* — que já existe (`welford_mean_z`). O canal de média está no seu limite de informação; o orçamento de pesquisa deve ir para variância/cauda/forma/dependência, onde o sinal comprovadamente vive.

### 4.3 O dilema variância×GARCH não tem solução no detector — tem no calibrador

CE2 e T6 são as duas faces de uma indecidibilidade local: um patamar novo de variância e um cluster GARCH longo são indistinguíveis numa janela curta. A defesa estrutural (escala congelada) está do lado certo do trade-off dado E3. O que resta é dar ao calibrador o material de *persistência* (volvol, contrastes multi-escala) e de *contexto* (rho1_abs_e do histórico — já entre as features mais usadas) — e aceitar que uma fração de falso positivo GARCH é irredutível. Não classifiquei isso como gargalo de primeira ordem porque a intervenção dedicada já foi tentada e não pagou; a suíte relativa (R5) medirá se as mudanças de R1–R4 o agravam.

### 4.4 A estrutura fina dos pesos (derivação completa de R1)

Da reformulação de §3.1: maximizar TS-AUC = maximizar a fração de pares concordantes no pool {(i,j): i∈P_t, j∈N_t, todos os t}. Um surrogate pareado decomponível (logístico em s_i−s_j) tem gradiente marginal por linha proporcional ao número de pares em que a linha participa: n_neg(t) para positivos, n_pos(t) para negativos. O peso pontual pareado-consistente é, portanto, o enunciado em §3.10; sob ele, a massa por classe se equaliza dentro de cada t, e o minimizador populacional da BCE ponderada em cada t é uma transformação monótona da razão de verossimilhança com odds a priori 1 — i.e., um ranqueador ótimo de AUC_t por passo, com a massa entre passos ∝ w_t. O esquema atual difere exatamente na partição intra-passo (1:1 em vez de n_neg:n_pos). Interação com o `init_score`: sob pesos balanceados o offset ótimo por t tende a 0; manter o init_score atual é inócuo para ranking (constante por t) mas muda os números de logloss — recomendo mantê-lo na primeira rodada (diff mínimo) e registrar. Cautela numérica: em t muito pequeno, positivos raros recebem pesos enormes (n_neg≈5000) — suavizar contagens (pseudo-contagem ~5) e/ou capar a razão de pesos em ~50 para não trocar viés por variância.

### 4.5 O que a falsificação da previsão nº 1 realmente significa

Duas hipóteses permanecem vivas e o projeto ainda não as separou:

- **H-extração:** o sinal existe além de 0,60, mas a *forma* do objetivo (pontual, classes desbalanceadas intra-passo, juiz de parada desalinhado) não o extrai. Testes: R1, R2, R3 — baratos, sequenciais, cada um com previsão própria.
- **H-informação:** para estas famílias de features, o gerador simplesmente não dá mais que ~0,60–0,65. Teste: R4 introduz uma família *nova* (não uma variação das existentes, que foi o que a rodada revertida tentou); se R1–R4 todos falharem sob R0, H-informação ganha força e o jogo passa a ser eficiência de sonda e robustez, não busca de ganho.

Essa é a bifurcação que o roadmap resolve — deliberadamente, com um experimento por hipótese, e não com dez micro-ajustes.

---

## 5. Discordâncias e concordâncias explícitas

**D1 — DIAGNOSTICO, recomendação 1(a) (recalibração pós-hoc do resíduo): rejeitada por impossibilidade matemática.** Platt é estritamente monótona e uniforme entre séries ⇒ AUC_t idêntica em todo t ⇒ TS-AUC idêntica. Isotônica é fracamente monótona ⇒ só pode criar empates ⇒ TS-AUC igual ou pior. A recalibração é higiene para a suíte de robustez e para leitura humana do score — nunca uma alavanca de desempenho. A metade (b) da mesma recomendação (gates relativos) está certa e vira R5.

**D2 — DIAGNOSTICO, conclusão ("o descasamento de escala… é o candidato de maior potencial"): rejeitada como enunciada.** O próprio documento reconhece a invariância C1 duas linhas antes; a frase "discriminando corretamente em ranking mas de forma fraca em magnitude" descreve um não-problema (magnitude é invisível à métrica) colado a o problema real (ranking fraco). O candidato de maior potencial não é a escala; é a estrutura do objetivo (R1–R3) e a família de features ausente (R4).

**D3 — plano_acao, tese central ("objetivo ≠ métrica era o gargalo"): parcialmente falsificada pelos próprios critérios.** Previsão 1 (árvores 300–800) falhou — best-iters 61–89 verificados em `fold_evals.json`; previsão 4 (média continua morta) confirmou. Pelo texto do próprio plano, isso desloca o peso para "falta de sinal" — mas a formulação correta é "falta de sinal *extraível por objetivo pontual com estes pesos*", porque as variantes estruturais (pesos pareados, ranking por grupo) nunca foram testadas. A2 fica como correção necessária e correta; a tese precisa do adendo acima. Consequência prática: a condição de A6 ("só se A2 não bastar") está satisfeita — A6 sobe de condicional para candidato ativo (R3).

**D4 — Planos técnico/repositório, regra §9.0 na forma absoluta: revisar.** A regra nasceu do padrão de falha certo (réplica caseira do harness oficial ⇒ viés otimista sistemático) e o DIAGNOSTICO até o quantificou de outro ângulo (n=100 ⇒ σ≈0,054). Mas o instrumento OOF-GroupKFold sobre 10⁴ séries tem σ≈0,005–0,008 no nível, e *muito* menos na diferença pareada entre dois modelos avaliados nas mesmas séries — e é o que o projeto já usa para decidir (a reversão das três intervenções foi decidida por ΔOOF=−0,0035 *sem* IC, o que é o erro simétrico ao que a regra queria evitar). Proposta: reescrever §9.0 como três cláusulas — (i) nenhuma réplica do scoring oficial como estimador absoluto de leaderboard (mantida); (ii) OOF pareado com bootstrap por série como juiz *relativo* padrão, com hipótese registrada antes de cada comparação (anti-garden-of-forking-paths); (iii) submissão oficial como âncora periódica que valida o instrumento (se o Δ oficial discordar sistematicamente do Δ OOF em sinal, o instrumento é rebaixado). Isso formaliza a prática real e devolve ao projeto um ciclo de iteração de horas em vez de sondas.

**Concordâncias que faço questão de registrar:** a análise H1 (ruído amostral) é exemplar e deveria ser citada em qualquer discussão futura de número local; o protocolo 3.8 (baseline controlado + suíte + OOF por bucket) é a espinha dorsal de validação correta e todos os R0–R5 o pressupõem; a lição "ablação de fold único não é evidência" está certa e cara — foi paga uma vez, não precisa ser paga de novo; rodar CE6 e a suíte a cada iteração (rec. 4) é barato e correto; a identificação do mecanismo de T6 é trabalho de primeira; e a decisão de *não* readicionar o offset no `predict_one` está certa (C1-neutra e melhor para a suíte).

---

## 6. Roadmap de reestruturação

Poucos itens, cada um com potencial de alterar o comportamento do sistema, cada um com validação pré-registrada pelo protocolo 3.8 + R0. Nenhum repete as intervenções revertidas (canal var vol-ajustado, remoção de meta_h0, δ=0,15, hazard extra, readição de offset).

### R0 — Metrologia: comparador OOF pareado com intervalo de confiança
*Problema:* decisões de manter/reverter estão sendo tomadas sobre Δs de 0,003 sem barra de erro; sondas oficiais são escassas demais para arbitrar micro-diferenças.
*Mudança:* `scripts/compare_oof.py` — recebe dois parquets de OOF (mesmas séries), reporta ΔTS-AUC geral e por bucket com IC 95% por bootstrap pareado (reamostrar `id`s com reposição, ~500 réplicas; a AUC ponderada por t roda em segundos por réplica). Regra de decisão: adotar se IC do Δ exclui 0 no agregado ou no bucket-alvo declarado a priori; sondar oficialmente só o que passa.
*Por que ganho:* não move o score — move a *taxa de aprendizado do projeto*: transforma cada retreino de 7 min num experimento com veredicto estatístico.
*Evidência:* σ(nível, n=100)=0,054 medido; pareamento reduz a variância do Δ tipicamente por fator grande (predições altamente correlacionadas entre variantes).
*Risco:* overfitting por seleção se muitas comparações forem feitas sem registro — mitigado pela hipótese registrada por comparação e pela âncora oficial (D4-iii).
*Custo:* ~2 h. **Pré-requisito de tudo abaixo.**

### R1 — Pesos de linha pareado-consistentes
*Problema:* partição intra-passo 1:1 entre classes dilui o gradiente dos positivos precoces por ~12× em t≤50 (§3.10, §4.4), no território de 35% do peso da métrica onde a AUC é 0,53–0,57.
*Mudança:* em `model/weights.py`: w_pos(t) ∝ n_neg(t), w_neg(t) ∝ n_pos(t) (contagens empíricas suavizadas com pseudo-contagem ~5; razão de pesos capada em ~50; × thin_weight; normalizar média 1). `init_score` mantido. ~10 linhas; retreino ~7 min.
*Por que ganho:* alinha o surrogate pontual exatamente à fração de pares concordantes que a métrica é (§3.1); é a metade de "objetivo ≠ métrica" que A2 não tocou.
*Evidência:* derivação fechada em §4.4 + a assinatura observada (fraqueza concentrada nos buckets onde a distorção de massa é máxima).
*Previsões falsificáveis:* (1) ΔTS-AUC OOF > 0 com IC excluindo 0, concentrado em t≤150; (2) best-iters sobem (o gradiente novo demora mais a esgotar); (3) buckets tardios ~estáveis (lá n_pos≈n_neg e os esquemas quase coincidem).
*Risco:* variância de gradiente em t muito pequeno (mitigada por cap/suavização); possível piora de T2 (não-antecipação) se o modelo ficar agressivo cedo — a suíte relativa (R5) vigia.

### R2 — Juiz de treino alinhado: `feval` de AUC-por-passo no fold + mini-sweep sob a régua nova
*Problema:* early stopping e toda escolha de hiperparâmetro são julgados por logloss pontual — que E2 mostra saturar em 61–89 árvores; qualquer sweep sob esse juiz herda o desalinhamento (é por isso que rebaixei, sem rejeitar, a rec. 2 do DIAGNOSTICO: primeiro a régua, depois a regularização).
*Mudança:* em `model/train.py`, `feval` custom = AUC ponderada por grupo t sobre o fold de validação (subamostra determinística de ~150k linhas se o custo por rodada incomodar; `first_metric_only=True` para a parada). Em seguida, um sweep pequeno e registrado: lr∈{0,05; 0,02}, min_data_in_leaf∈{200; 50}, lambda_l2∈{5; 1} — 8 células, ~1 h, julgadas por R0.
*Por que ganho:* a parada passa a ocorrer quando o *ranking por passo* para de melhorar — que é a única coisa que importa; destrava o regime "mais árvores fracas" se ele existir.
*Nota de conformidade:* isto é critério interno de fold, não estimador de leaderboard — compatível com a §9.0 revisada (D4) e, honestamente, também com seu espírito original.
*Risco:* custo por rodada do feval (mitigado por subamostra); nenhum risco de regressão de inferência (nada muda no predict).

### R3 — Objetivo de ranking por grupo t (lambdarank/xendcg), como membro paralelo do ensemble
*Problema:* mesmo com R1+R2, o objetivo continua pontual; a métrica é ranking por grupo. Um objetivo de ranking com query = passo t torna estruturalmente impossível gastar capacidade em qualquer função constante-dentro-do-passo e otimiza diretamente a ordenação intra-t.
*Mudança:* modo `rank` em `model/train.py`: dataset ordenado por t dentro do fold, `group` = tamanhos por passo; `objective=lambdarank` com `label_gain=[0,1]` e — **armadilha crítica** — `lambdarank_truncation_level` ≥ tamanho máximo de grupo (o default ~30 daria gradiente só ao topo do grupo, péssimo para AUC); alternativa `rank_xendcg`. Predição = raw score (ranking) mapeado por sigmoide fixa para [0,1]. Combinação com o modelo binário por rank-average, avaliada por R0 em três braços: binário-R1, rank, média.
*Por que ganho:* é a versão exata do alinhamento; NDCG com rótulo binário e truncation total se comporta como um surrogate razoável de AUC por grupo.
*Evidência:* estrutura da métrica (§3.1); precedente geral de lambdarank em métricas de ordenação por grupo; nenhum precedente interno — por isso é braço paralelo, não substituição.
*Riscos:* grupos grandes (custo de treino ↑), sensibilidade de hiperparâmetros do lambdarank, fragilidade maior que R1 (o próprio plano_acao listou os riscos corretamente); determinismo preservado (`deterministic=true` cobre o objetivo de ranking).
*Custo:* ~1 dia incluindo a plumbing de grupos.

### R4 — A família ausente: duas amostras rank-based contra o histórico (o análogo causal de 2025)
*Problema:* o banco atual é forte em detectores sequenciais paramétricos e fraco na comparação distribucional direta janela-vs-histórico — exatamente a família que dominou a edição batch (F1–F3). Os únicos representantes atuais são 4 log-martingales (2,2% de SHAP, agregador fraco — §3.6) e o quantile-crossing de 2 bins. Além disso, estatísticas rank-based são invariantes a erro de escala do H0 e a caudas — a comparabilidade transversal de graça (§3.2, §3.7).
*Mudança:* novo `state/rank_twosample.py` (StateBlock), reusando os arrays ordenados e o bisect já existentes: por passo, p_right e p_abs (já computados no ConformalBlock — compartilhar); features: z de Wilcoxon de janela = média_w(p_right−½)·√(12w) para w∈{25,100} (localização, robusta); análogo sobre p_abs para w∈{25,100} (dispersão/cauda rank-based, tipo Ansari–Bradley); um χ²-de-forma de janela 100 sobre 4 bins de quantis do histórico (frações observadas vs. nominais — generaliza o quantile-crossing). ~6–7 features, O(log n_h + 1)/passo, ~30 µs — cabe com folga. Variante interna opcional: z de variância padronizado por curtose (§3.7).
*Por que ganho:* injeta informação *nova* (ordem completa contra o histórico, não momentos), no regime onde §4.1 argumenta que há sinal na mesa (bucket 150–400), com a propriedade de comparabilidade que a métrica premia.
*Evidência:* paradigma vencedor 2025 (duas amostras ricas + árvores); SHAP mostrando que as famílias baratas de rank (conformal) contribuem *apesar* do agregador ruim.
*Riscos:* o histórico do projeto (intervenções de feature com ganho nulo) — mitigado por ser família nova e não variação; inflar contagem de features (~87) é inócuo para árvores; validar com o ciclo completo + R0, e com a previsão específica de ganho concentrado em T8-like (forma/cauda) na suíte e nos buckets ≥150 no OOF.
*Custo:* ~1 dia (bloco + testes unitários + rebuild ~28 min + retreino).

### R5 — Suíte de robustez: gates relativos para o calibrador supervisionado
*Problema:* 9/15 falhas pré-existentes medem a escala, não o detector (E5); no estado atual, a suíte não consegue exercer seu papel de pré-filtro de sondas para o modo supervised.
*Mudança:* em `robustness/` + `configs`: para T2/T6/T9/T10/T13, gate = gap de mediana contra um *painel de referência* (séries N(0,1) i.i.d. sem quebra, mesmas seeds, mesmo T) em vez de limiar absoluto — T6 relativo vira "mediana(T6) − mediana(painel) ≤ x", que mede o falso positivo GARCH *acima do piso do próprio calibrador*; manter os gates absolutos atuais apenas para o modo fallback. É a rec. 1(b) do DIAGNOSTICO, que endosso.
*Por que ganho:* devolve à suíte a capacidade de detectar regressões comportamentais reais (como a de T13 na rodada revertida — que ela pegou) sem afogar o sinal em 9 falhas estruturais.
*Custo:* ~meio dia; risco ~zero.

### R6 — Fechar as medições que decidem H-extração vs. H-informação
*Mudança:* (i) rodar e **versionar em docs/** o censo A1 (`break_type_census.py` — já escrito, resultado ausente dos documentos): distribuição de Δmédia/Δlogvar/Δρ₁/Δcauda e de Σφ̂ — é o mapa do gerador que calibra as expectativas de todos os R acima; (ii) resposta ao degrau OOF alinhada em τ (A4) como artefato padrão por versão de modelo; (iii) comparação por bucket entre AUC observada e o envelope de potência Φ(δ√m−z) usando as magnitudes do censo — o teste direto de "t≤50 está no teto de informação?".
*Custo:* ~meio dia; é o que transforma a bifurcação de §4.5 em números.

**Estacionados (não fazer agora, critérios de reabertura explícitos):** t-likelihood no filtro bayesiano (reabrir se o censo mostrar fatia grande de caudas pesadas com ν̂<8); GRU numpy (reabrir só em platô pós-R4 com folga de cronograma — inalterado do plano); contrastes multi-escala de variância anti-GARCH (reabrir se R5 mostrar T6 piorando sob R1–R4); qualquer nova mexida em grades de CUSUM, hazards ou meta_h0 (já julgadas).

**Sequência sugerida:** R0 → R5 (destravam o julgamento) em 1 dia; R1 + R2 em 1–2 dias, sonda oficial do melhor braço; R3 e R4 em paralelo na semana seguinte, cada um julgado por R0 antes de sonda; R6 corre em paralelo desde o dia 1. Com o simpósio em outubro, isso deixa ~2 meses de margem para consolidação e congelamento (P4) — o cronograma não é o gargalo; a taxa de aprendizado por experimento é, e R0 é o multiplicador dela.

---

## 7. Tabela-síntese de veredictos

| Categoria | Itens |
|---|---|
| **Decisões excelentes** | rótulo por passo + análise C1–C3; motor único; teste de prefixo + canário; determinismo bit-a-bit; GroupKFold por série; score livre (CE1); trava CE2 (escala congelada p/ variância); init_score de taxa-base (A2); continuidade na fronteira (lag_seed); fix de memória por série; protocolo 3.8 do DIAGNOSTICO; análise H1 de ruído amostral |
| **Boas, mas discutíveis** | AR(10) fixo com aceitação a 2% de R²; filtro bayesiano gaussiano (ótimo em teoria, 3,4% de SHAP na prática); 3 hazards; grade de δ do CUSUM em unidades de inovação; média de probabilidades no ensemble; suíte T1–T13 com os gates *atuais*; thinning (correto, mas acopla-se aos pesos e merece re-checagem sob R1) |
| **Excessivamente complexas (para o retorno)** | agregação por martingale de Vovk com mistura de ε (os p-values ficam; o agregador troca por Wilcoxon de janela — R4); trio de mecanismos de excedência sobrepostos (Bernoulli-CUSUM + frações de janela + EWMA); features de concordância de localizadores (meta_locator_*) |
| **Provavelmente ineficientes** | recalibração pós-hoc do resíduo como alavanca de desempenho (C1-neutra — D1); gates absolutos aplicados ao calibrador supervisionado; qualquer nova iteração no canal de média para séries persistentes (limite de informação verificado — §4.2); comparações fallback-vs-modelo em 100 séries |
| **Hipóteses ainda não testadas** | pesos pareado-consistentes (R1); parada/seleção por AUC-por-t (R2); objetivo de ranking por grupo (R3); família rank-based de duas amostras (R4); distribuição real de tipos/magnitudes do gerador (censo A1 — escrito, não publicado); teto de informação em t≤50 (R6-iii); concordância Δoficial × ΔOOF (D4-iii) |
| **Oportunidades realmente promissoras** | R1 (melhor razão mecanismo/custo do projeto); R4 (única injeção de informação nova alinhada ao precedente 2025); R0+D4 (multiplicador da velocidade de pesquisa); R3 (alinhamento estrutural definitivo, maior variância) |

---

## 8. Fecho

Se eu começasse este projeto do zero hoje, reproduziria ~85% dele: a formulação, o motor único, a disciplina de causalidade/determinismo e a decisão anti-max-hold são o esqueleto certo, e a evidência de 2025 continua apontando para "estatísticas suficientes + árvores". As três coisas que eu faria diferente desde o dia 1 — pesos pareados, juiz de treino por grupo t, e a família rank-based de duas amostras no lugar do martingale como cidadão de primeira classe — são exatamente R1/R2–R3/R4, todas compatíveis com a arquitetura existente e nenhuma exigindo demolição. A quarta diferença é de processo: eu não teria proibido o instrumento OOF; teria lhe dado barras de erro. O projeto pagou caro para aprender que intervenções bem motivadas podem render zero — a resposta certa a essa lição não é parar de intervir, é intervir menos vezes, com hipóteses maiores, julgadas por um instrumento com resolução suficiente. É isso que este roadmap tenta ser.

— Fim do parecer —
