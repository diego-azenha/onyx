# Reflexão pós-campanha e portfólio v2 — o que ainda não foi tentado

**Data:** 2026-07-24 · **Insumos:** `RELATORIO_CAMPANHA_X0_X4.md`, repo `onyx` (lido na sessão anterior), as 7 propostas do usuário. **Incumbente novo:** V4 + X1-soft, OOF K=4 ≈ **0,6058**. Toda comparação futura é contra ele.

---

## 0. Três clarificações antes de qualquer ideia

**As duas fontes de erro não podem ser confundidas.** A barra ~0,0014 (2 EP de semente, K=4) responde "quantos sorteios bastam para medir". O IC pareado por séries (±0,003–0,004 no Δ, `compare_oof`) responde "isso generaliza para outro painel" — e continua sendo **o juiz de adoção** (o X1 passou nele, não só na barra de semente). E o board acrescenta uma terceira fonte: o painel de teste é outro sorteio de séries, então um Δ OOF de +0,0040 com IC [+0,0005; +0,0073] pode aparecer no placar como qualquer coisa nesse intervalo. Consequência estratégica: **vitórias pequenas se acumulam localmente e se submetem em pacote**, nunca uma a uma.

**"+0,0040 não é competitivo" merece um reparo.** É o segundo maior efeito verificado da história do projeto (atrás só do bagging), com IC excluindo zero no agregado e nos dois buckets de maior peso. O problema não é qualitativo, é aritmético: precisa de 3–5 desses empilhados. O portfólio abaixo é desenhado para maximizar *taxa de acerto × baratez*, não para caçar um único +0,02 — a história do projeto diz que ele não existe como um golpe só.

**O filtro do usuário está certo e eu o adoto como axioma:** feature nova está saturada; os eixos vivos são supervisão (1 vitória em 1 tentativa), variância do estimador (a maior alavanca já medida) e, condicionalmente, comparabilidade autorreferente (2 mortes, mas ambas pelo mesmo defeito de nulo — ver §2.2/§2.7).

---

## 1. Debate das sete propostas, uma a uma

### 1.1 Agregação do ensemble em espaço de rank — **veredito: rodar o comando que já existe, esperar ~0; o valor real está na versão heterogênea**

O raciocínio ("a métrica é rank de Mann-Whitney intra-passo, o bag funde em probabilidade") está correto, mas o efeito depende de os membros discordarem. As 4 sementes têm o mesmo dataset, os mesmos folds e as mesmas features — as distribuições marginais de score por passo são quase idênticas, então o percentil intra-passo é quase a **mesma** transformação monótona para todas, e média-após-transformação-comum ≈ transformação-da-média (o desvio é um efeito de Jensen sobre discordâncias que valem no total +0,0046). Esperado: 0 a +0,001. Detalhe que barateia o teste a minutos: **`scripts/combine_oof.py` já faz rank-average via percentil OOF** — foi escrito para a comparação do R3. Rode sobre as 4 sementes, registre, siga em frente.

Onde a ideia ganha corpo: agregação em espaço de percentil-OOF entre membros **heterogêneos** — o bag binário, o membro de ranking (R3, retreinado com K=4 sob a barra nova: o +0,0010 [−0,0035; +0,0058] de um sorteio virou questão aberta), e o **score de fallback** (bayes + CUSUM + martingale conformal), que não tem efeitos-fixos aprendidos e pode ser relativamente mais forte em t≤50, onde o supervisionado é mais fraco. Com pesos convexos **por bucket de t** ajustados em OOF aninhado (ajusta em 4 folds, avalia no 5º — 4–8 parâmetros, sem vazamento material). Implantação causal: mapas de quantis OOF por membro × bin de t, congelados e determinísticos. Isso absorve a proposta 1, reabre o R3 da forma certa e dá ao fallback um papel que ele nunca teve. Custo: baixo-médio, quase todo offline. Previsão: +0,001 a +0,003.

### 1.2 Score como p-valor/e-valor conformal por série — **veredito: como formulado, é o X3 renomeado; discordância declarada**

Preciso ser direto aqui. Um p-valor conformal **é** a CDF empírica do conjunto de calibração com correção de amostra finita ((1+#{calib ≥ teste})/(n+1)). A "garantia de cobertura" vem da **trocabilidade** entre calibração e teste — exatamente a hipótese que o viés de regime do X3 viola. Trocar o nome da transformação não conserta o estimador do nulo; a validade prometida não se materializa. O mesmo vale para o e-valor: a validade *anytime* é sobre erro tipo-I **ao longo do tempo da mesma série**; a comparabilidade **transversal** entre séries exige que o nulo de cada uma esteja certo — a mesma trocabilidade de novo. Sem consertar o nulo, 1.2 herda o −0,0294 do X3 com juros de complexidade.

O conteúdo aproveitável: a proposta 1.2 fixa o **alvo certo** (saída comparável por construção). O caminho até ele é a proposta 1.4 + 1.7 combinadas — ver §2.7, onde está o único desenho que considero legítimo para a terceira (e última) tentativa desse eixo.

### 1.3 Mineração de pares confusáveis — **veredito: sim, prioridade máxima; e ampliar em três direções que mudam o retorno**

Concordo integralmente com o espírito ("medir antes de construir") e acrescento estrutura que multiplica o valor:

**(a) Negativos quentes — a perda é concentrada, não difusa.** A perda da TS-AUC é uma soma sobre pares invertidos; um único negativo que fica acima da mediana dos positivos em todos os passos de um online longo destrói **milhares** de pares sozinho (a contribuição por negativo é ~linear no número de positivos × passos vivos). Ninguém nunca rankeou os negativos por massa de pares invertidos ponderada por w_t. Entregável: curva de Lorenz da concentração da perda. Se os top-50 negativos carregam 10–20% da massa, o jogo muda — passa a existir um alvo pequeno e caracterizável.

**(b) Suspeita de rótulo nos negativos quentes.** O changelog W23/2026 corrigiu 29 séries mal-rotuladas — é evidência de que o processo de rotulagem tem taxa de erro não-nula, e não há garantia de que 29 fossem todas. Um "negativo" que o modelo insiste em pontuar alto **e** cuja série exibe evidência anticausal forte de mudança (dois-amostras energia/MMD entre metades do online, calculado com o futuro — permitido em diagnóstico) é candidato a rótulo errado. Regra anti-circularidade obrigatória: a flag vem **só** da estatística anticausal independente, nunca do score do modelo (senão o modelo aprende a confirmar os próprios falsos alarmes). Uso: braço de pesos (só treino, avaliação intocada) rebaixando negativos flagados — o espelho exato do X1, no eixo que já venceu.

**(c) Positivos frios e a partição informação-vs-extração.** Para cada par invertido de alta massa: a evidência anticausal existia no passo t? Se não → gargalo de informação (irredutível, confirma o skyline). Se sim, alguma feature causal já tinha se movido e o score não → **gargalo de extração**, e a assinatura de quais features se moveram sem resposta do score diz qual braço construir (é o insumo que decide 2.6 e 1.5). Custo total do 1.3 ampliado: ~1–2 dias, zero risco. É o braço que des-arrisca todos os outros.

### 1.4 Consertar o viés de regime e re-testar a normalização — **veredito: sim, mas só na forma "emulador com alvo real", e com cláusula de fechamento definitivo**

O diagnóstico do X3 está certo: o CE6 prova que g_i não carrega rótulo, então removê-lo *bem estimado* não deveria piorar — o −0,0294 é assinatura de estimador viesado, não de mecanismo errado. Mas atenção ao que a saga da centragem já ensinou: mesmo o oráculo honesto de remoção de nível comprava ~+0,002. O teto realista deste eixo é modesto; o que justifica uma terceira tentativa é que ela é barata, offline, e **fecha o eixo para sempre se falhar**.

O desenho que muda o mecanismo (e não só o nome): o alvo do nulo passa a ser **o H0 do online real**. Linhas com t < τ (e séries sem quebra) são amostras verdadeiras do score-sob-H0 no regime certo — milhões de linhas no OOF. Ajustar um emulador quantílico (LGBM pinball, q10–q90) do score-nulo em função de (meta_h0, t, e opcionalmente os quantis do replay/as-of como covariáveis — o emulador aprende a des-viesar o replay). Studentizar o OOF contra o emulador (out-of-fold: emulador do fold k aplicado às séries do fold k). Uso de τ para selecionar linhas H0 = informação privilegiada **no treino do emulador**, mesma licença do X1. Custo: ~1 dia, sem retreino do modelo principal. Previsão: 0 a +0,003. **Cláusula pré-registrada: se falhar, o eixo de comparabilidade autorreferente fecha com prejuízo** — três estimadores de nulo progressivamente melhores, três mortes, é a definição empírica de eixo estéril, e a explicação teórica (mistura detectado/não-detectado com fronteira fina) já existe.

### 1.5 Surrogate diferenciável da AUC intra-passo no gradiente — **veredito: sim, mas não como soft-rank neural; como `fobj` custom no LightGBM, e com o diagnóstico do 0,5852 explicitado**

Por que o R3 colapsou a 0,5852 tem diagnóstico plausível nos próprios parâmetros: lambdarank com desconto NDCG **enfatiza o topo da lista** (a AUC é uniforme sobre pares) e `truncation_level=300` em grupos de ~8.000 significa que ~96% das linhas de um grupo não recebiam gradiente. Não é evidência contra "pressão de ranking intra-t"; é evidência contra *aquela* pressão. A versão alinhada, nunca tentada:

- **Objetivo custom único** (não um membro separado): `grad = (1−α)·grad_logloss + α·grad_pairwise`, com o termo pairwise = logística de pares (pos, neg) **do mesmo grupo t**, amostrados uniformemente (P pares por positivo por iteração, semente determinística por iteração), peso w_t, **sem** desconto de posição e **sem** truncation. Hessianas fechadas (σ(−m)σ(m)). α ∈ {0,1; 0,3}, um braço cada.
- Elegância técnica de graça: pares do mesmo t compartilham o `init_score` da taxa-base, então o termo pairwise é **automaticamente cego a f(t)** — a pressão vai inteira para a ordenação transversal, que é o que a métrica mede.
- Continua sendo o eixo supervisão — o de 1 vitória em 1 tentativa. Custo: 1–2 dias de implementação + K=4 (reusa o parquet). Previsão: +0,001 a +0,004. Kill: Δ≤0 nos dois α.

### 1.6 Variância por outros eixos — **veredito: sim; concretamente bagging de partição de folds, e acrescento a suspeita estrutural dos hiperparâmetros**

Ordem por retorno esperado dentro do eixo:

1. **Bagging de partição** (variar `cfg.seed` da partição, 2 partições × 4 sementes): diversidade maior que a de boost-seed (dados de treino diferentes por modelo), o análogo mais próximo do que já pagou +0,0046. Contabilidade OOF honesta necessária: para cada série, a predição OOF usa só os modelos que não a viram — o ensemble de avaliação fica menor que o de implantação (OOF levemente pessimista; registrar). Custo: 2× compute, zero código novo além da contabilidade. Previsão: +0,001 a +0,003.
2. **Sweep de hiperparâmetros sob a régua nova.** Ponto que acrescento ao 1.6: **todos os HPs atuais foram selecionados na era da semente 42** — comparações contaminadas por seleção. O incumbente é suspeito por construção. Grade grossa (num_leaves, min_data_in_leaf, lambda_l2, lr×cap), triagem K=2, confirmação K=4 nos top-3, julgada por TS-AUC OOF com IC. O script já existe (`sweep_hyperparams.py`). Previsão: 0 a +0,004.
3. **Snapshot averaging** (média de predições em contagens de árvores diferentes do mesmo booster): quase grátis via `num_iteration`, teste de minutos, esperado ~0 a +0,001.
4. Subamostra de séries por membro: redundante com `bagging_fraction=0,8`; pular.

### 1.7 Múltiplas verdades "as-of" da mesma série — **veredito: sim, e é maior do que parece — mas primeiro entender o 0,9998**

O probe do X4 a 0,9998 é sharp demais para ser "distribuição levemente diferente"; a assinatura provável é **suporte fora da distribuição**: truncamento uniforme de 200 pontos empurra `n_h` para valores que não existem nos dados reais (histórias reais têm n ≥ 1000; um portador de 1000−200=800 é impossível, e mesmo n−200 desloca a *distribuição conjunta* de comprimentos para fora do padrão). Se for isso, o conserto é barato: cortes `c` respeitando `len(hist′) ≥ 1000`, com k sorteado (determinístico por série) de modo que a distribuição de comprimentos resultante fique dentro do suporte real. Re-probar custa minutos.

Dito isso, discordo da ênfase: o valor principal do as-of **não é** multiplicar supervisão — réplicas da mesma série são altamente correlacionadas (mesmos valores!), então não aumentam o n efetivo de *séries*, que é o que limita padrões transversais; o ganho como augmentação é regularização de invariância à fronteira (modesto). O valor principal é ser **a fábrica de nulo sem artefato** que 1.2/1.4 precisam: pseudo-onlines 100% H0, gerados pelo pipeline idêntico ao de produção, com probe de falsificação embutido (discriminador linhas-H0-reais vs linhas-as-of ≈ 0,5 é o gate; se não passar nem assim, o §1.4 roda só com o emulador de alvo real). Sequência: consertar suporte → re-probe → usar como covariável/calibração do 1.4. Como augmentação de treino, só depois, com mix ≤20% e avaliação OOF-só-reais.

---

## 2. O que eu acrescento (não estava nas sete)

### 2.1 Dados de 2025 como portadores reais — a aposta estrutural que o X4 queria ser

O X4 morreu tentando **sintetizar** portadores; existe um estoque de portadores **reais** parado: as séries do desafio 2025 (edição batch). O formato converte quase diretamente: positivo de 2025 = (segmento pré, segmento pós, fronteira conhecida) → história = pré[:−m], pseudo-online = pré[−m:] ⊕ pós, **τ′ = m controlável** (dá para popular exatamente os τ pequenos onde o peso da métrica é alto); negativo de 2025 = série H0 inteira, corte livre. Sem síntese, sem gerador, sem o artefato que deu 0,9998.

Três gates, nessa ordem: (i) **regras** — confirmar que usar os dados públicos da edição anterior é permitido na 2026; (ii) **suporte** — comprimentos de 2025 compatíveis com história ≥ 1000 (senão, restringe ao subconjunto que cabe); (iii) **probe de domínio** — o discriminador do X4, reaproveitado: linhas H0 de portadores-2025 vs linhas H0 reais-2026. Separável → geradores diferentes → mix pequeno e pesado para baixo, ou aborta; ≈0,5 → é a única forma conhecida de **dobrar o n de séries reais**, o teto silencioso de tudo. Mix cap 20–30%, pesos ≤ reais, Δ reportado em OOF-2026-only. Previsão: 0 a +0,008 — a maior variância e o maior teto do portfólio.

### 2.2 V-ema assimétrico no pós-processamento — estacionado no repo, nunca medido, testável em uma tarde

O próprio HISTORICO estaciona "V-ema no pós-processamento (quando sobrar uma sonda)" — a sonda nunca sobrou, e agora ela custa uma tarde **offline** (transformação causal do próprio score: EWMA com λ_subida rápida e λ_descida lenta, aplicada às trajetórias OOF, `compare_oof` pareado). Mecanismo: o jitter do score é ruído transversal puro nos passos seguintes; a assimetria respeita a semântica "P(quebra já ocorreu)" sem o max-hold rígido que o CE1 matou. Disciplina de grade: 6 configurações, **seleção em 3 folds, confirmação nos 2 restantes** (com barra 0,0014, grid sem holdout fabrica falsos ganhos). Previsão: 0 a +0,003, concentrado em 150<t≤400 (49% do peso). Se ganhar, implantação é trivial e sem latência.

### 2.3 F3 — pesos por w_t: o item de fila mais antigo, no eixo que venceu

Está no backlog como P2 "[ ]" desde sempre: alinhar o peso das linhas de treino com o peso w_t da métrica (hoje o treino trata t=30 e t=700 quase igual; a métrica não). Mesma maquinaria do X1 (`weights.py`, sem build, K=4 em ~40 min), multiplicativo com o X1-soft — mas **braço separado** (regra anti-pacote). Interação a cuidar: reajustar a curva de `init_score` sob os pesos novos. Previsão: +0,001 a +0,004.

### 2.4 Completar a variante `ramp` do X1

Ficou inconclusa por hibernação da máquina, e é a versão (t−τ)-consciente — o degrau do gráfico 9 diz que a detectabilidade de uma linha cresce com passos-desde-quebra, e `ramp` é o único jeito de rebaixar as linhas iniciais de séries *tardiamente detectáveis* sem rebaixar a série inteira. É também a versão "destilação de professor para pobres": captura boa parte do que uma destilação anticausal completa daria, por fração do custo. 40 min de retreino ×4.

### 2.5 Restrições monótonas nos acumuladores de evidência — um braço de uma linha de config

P(quebra≤t) deveria ser não-decrescente em cada LLR acumulado (cusum_*_pos/neg, conformal_logm_abs_reset, bayes log-odds). `monotone_constraints` do LightGBM em ~10 features de evidência é regularização estrutural contra interações espúrias com efeitos-fixos — exatamente o tipo de restrição que ajuda quando n_eff é 10⁴ séries. Custo: config + K=4, sem build. Previsão: −0,001 a +0,002 (pode custar fit; por isso é braço, não fé).

### 2.6 `feature_contri` penalizando `meta_h0` — a versão barata dos espelhos

Os espelhos (neutralizar o canal de efeito-fixo das constantes, 34,3% do |SHAP| que o CE6 prova não carregar rótulo) morreram junto com o X4. A versão de uma linha: penalizar o ganho de split das colunas `meta_h0_*` (`feature_contri` 0,5–0,7), que empurra as árvores a usá-las como condicionadores (interações) e não como interceptos por série. Custo: config + K=4. Previsão: −0,001 a +0,003. Se o 1.3 mostrar negativos quentes com perfil "meta exótico", este braço sobe de prioridade.

### 2.7 Modelo em dois regimes de t — a invariância C1 torna isso seguro, e ninguém notou

A AUC de cada passo compara séries **no mesmo t**, logo todas usam o mesmo modelo do regime — descontinuidades entre regimes são *exatamente irrelevantes* para a métrica. Um modelo especializado em t≤~64 (onde metade das features está em warmup/NaN e os defaults de NaN do modelo único são ditados pelos dados de t grande) + um para o resto, cada um com seu `init_score` e HPs. Condicionado ao 1.3: só constrói se a partição informação-vs-extração mostrar gargalo de extração em t pequeno (se for piso informacional, não há o que especializar). Custo: médio. Previsão condicional: +0,001 a +0,004 em t≤150 (35% do peso).

### 2.8 F0.c — BOCPD isolado, a pendência que sobrevive a tudo

Continua sendo o único experimento que separa as metades do V5, agora sob a barra 0,0014 que torna o resultado interpretável. Build própria (~30 min) + K=4. Entra quando houver folga de build.

---

## 3. Portfólio consolidado

Comparações contra **V4+X1-soft**; adoção = IC pareado exclui 0 (agregado ou bucket-alvo declarado); grids offline com seleção/confirmação em folds disjuntos; braços de config/pesos nunca empacotados.

| # | braço | eixo | custo | previsão Δ | kill |
|---|---|---|---|---|---|
| **T0.1** | pares confusáveis + negativos quentes + flags de rótulo (1.3 ampliado) | diagnóstico | 1–2 dias | direciona tudo | — |
| **T0.2** | rank-avg das 4 sementes (`combine_oof`, já existe) | variância | minutos | ~0 (registrar) | — |
| **A1** | V-ema assimétrico (2.2) | pós-proc. | 1 tarde offline | 0 a +0,003 | confirmação falha |
| **A2** | stacking heterogêneo por bucket: bag + R3(K=4) + fallback em percentil-OOF (1.1) | variância/geom. | 2–3 dias | +0,001 a +0,003 | IC não exclui 0 |
| **A3** | emulador de nulo com alvo H0-real + studentização (1.4; as-of como covariável se 1.7 re-probar limpo) | comparabilidade | ~1 dia offline | 0 a +0,003 | **fecha o eixo c/ prejuízo** |
| **B1** | F3 pesos por w_t (2.3) | supervisão | 40 min ×4 | +0,001 a +0,004 | Δ≤0 |
| **B2** | `ramp` do X1 (2.4) | supervisão | 40 min ×4 | 0 a +0,003 | Δ≤0 |
| **B3** | pesos anti-rótulo-ruidoso nos negativos flagados (1.3b) | supervisão | 40 min ×4 | 0 a +0,003 | flags <0,5% ou Δ≤0 |
| **B4** | gradiente misto logloss+pairwise intra-t (1.5) | supervisão | 1–2 dias + K=4 | +0,001 a +0,004 | Δ≤0 em α={0,1;0,3} |
| **B5** | monotone_constraints nos LLRs (2.5) | estimador | config + K=4 | −0,001 a +0,002 | Δ≤0 |
| **B6** | feature_contri em meta_h0 (2.6) | estimador | config + K=4 | −0,001 a +0,003 | Δ≤0 |
| **C1** | dados 2025 como portadores reais (2.1) | n efetivo | 3–4 dias | 0 a +0,008 | regra/suporte/probe |
| **C2** | as-of: conserto de suporte + re-probe (1.7) | infra de nulo | 1 dia | habilita A3 | probe >0,55 |
| **C3** | bagging de partição 2×4 (1.6.1) | variância | 2× compute | +0,001 a +0,003 | Δ≤0 |
| **C4** | sweep de HPs sob régua nova (1.6.2) | estimador | compute | 0 a +0,004 | top-3 ≤ 0 |
| **C5** | dois regimes de t (2.7) — condicionado a T0.1 | arquitetura | médio | +0,001 a +0,004 | gargalo = informação |
| **C6** | F0.c BOCPD isolado (2.8) | fila | build + K=4 | ? | Δ≤0 |

**Sequência sugerida:** T0.1 e T0.2 imediatamente; em paralelo a esteira B (B1→B2→B3, 40 min cada, o eixo vencedor); A1 na primeira tarde livre; C2→A3 na semana; B4 e C1 como os dois investimentos da quinzena; C3/C4 como pano de fundo de compute; C5/C6 conforme T0.1 e folga.

**Disciplina de multiplicidade.** Com barra fina e ~15 braços, ~2,5% de falso-ganho por braço nulo é tolerável, **mas**: (i) grids offline sempre com seleção/confirmação separadas; (ii) no máximo um braço adotado por vez no incumbente; (iii) antes de qualquer submissão, o **pacote inteiro** re-medido de uma vez contra o V4+X1-soft com K=4 fresco — o número do pacote é o que se registra, não a soma das partes.

---

## 4. Estratégia de placar — o que "competitivo" exige em números

O mapa OOF↔board tem **um** ponto, e ruim: (OOF 0,6100 — o artefato da semente 42 — → board 0,6201). Sem um segundo ponto limpo, não há como traduzir Δ OOF em posições. Recomendação (mantida, com a justificativa completa): submeter o V4+X1-soft já verificado dá (a) o segundo ponto do mapa com um OOF honesto, (b) a confirmação externa do primeiro ganho da era pós-régua, (c) o denominador para a pergunta que governa a campanha — *quantos +0,003 separam a posição atual do alvo?* Sem esse número, "não é verdadeiramente competitivo" é um sentimento; com ele, vira um plano com contagem regressiva. A decisão é sua; o custo de informação de adiar é que todo o portfólio roda às cegas quanto ao câmbio OOF→board.

Aritmética honesta do portfólio: com taxa de acerto histórica (~1 em 3–4) sobre ~12 braços de medição, o resultado realista da quinzena é **+0,006 a +0,012 OOF** sobre o incumbente — empilhado com o X1, algo como 0,606→0,612–0,618 OOF. Se o câmbio do board for ~1:1 (o único ponto sugere até favorável), isso disputa o platô. O breakthrough único continua não existindo nos dados; o que existe é uma esteira com barra 4× mais fina do que se supunha e três eixos comprovadamente vivos.

---

## 5. Fechamentos com prejuízo (para não voltarmos)

- **Comparabilidade autorreferente**: fecha se A3 falhar (terceira morte com o nulo certo). A explicação teórica já está escrita — mistura detectado/não-detectado, fronteira fina.
- **Síntese de portadores** (X4 original): fechada; C1 (portadores reais de 2025) e C2 (as-of para nulo) são os sucessores, cada um com probe próprio.
- **Rank-avg de sementes homogêneas**: fecha com o registro do T0.2 (~0 esperado).
- **Canal de média como família de features**: permanece fechado (D1 + skyline |Δmean_e| mediano 0,0785); a única sobrevivente é a via indireta B3 (rótulos), que não é sobre média.

*Previsões registradas antes de qualquer R0. Um braço por mudança. Grids com confirmação. O pacote se mede inteiro antes de encostar no board.*
