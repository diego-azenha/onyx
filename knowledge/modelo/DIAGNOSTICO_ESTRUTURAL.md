# Diagnóstico estrutural — por que o platô, o que nos limita desde o dia 1, e qual porta nunca foi aberta

**Data:** 2026-07-24 · **Insumos:** `RELATORIO_CAMPANHA_X0_X4.md`, `RELATORIO_CAMPANHA_B_C.md`, HISTORICO §13–14, docs oficiais da competição (docs.crunchdao.com/competitions/competitions/structural-break-real-time), literatura 2024–26 de TSFMs e detecção. **Prazo real:** última quota **16/09/2026** — 7,5 semanas. Isso muda o que cabe no plano.

---

## 1. A tese, em um parágrafo

O sistema é uma **máquina de calibração sobre uma base de evidência fixa e artesanal**: fluxo cru → 183 funcionais escolhidos à mão → GBM. Nas duas campanhas (~25 braços), variamos *tudo que existe em volta da base* — pesos, objetivo, normalização, ensembling, restrições, parada, partições — e nunca a base. A taxa de rejeição altíssima não é azar: é a assinatura de que **a camada de calibração já estava perto do ótimo para essa base**, e o que sobra de erro mora no que os 183 funcionais não medem. `score = g(φ(x))`; otimizamos `g` até a exaustão; `φ` foi congelado no primeiro dia — por um veredito herdado de outra tarefa.

---

## 2. As evidências de que a calibração saturou (por que 25 braços morreram)

**A escadinha de mortes do A3 é um teorema disfarçado.** Centragem aditiva −0,002 → CDF do histórico −0,029 → nulo real emulado **−0,0457**: quanto melhor o nulo, pior o resultado. Isso é exatamente o que a teoria prevê quando o score já ordena por probabilidade a posteriori: ranquear por P(y|x) é o ranking ótimo para AUC, e qualquer "correção" por série só pode subtrair a parte do condicionamento que o modelo usava legitimamente. Tradução: **a calibração transversal do modelo já é quase-posterior. Não há mais nada a extrair dessa camada.**

**A perda é difusa (T0.1).** As 5% piores séries carregam 16% da massa de pares invertidos — sem alvo concentrado, sem bug escondido, sem série-vilã. Deficiência uniforme e pequena em toda parte é o retrato de um limite de base, não de um defeito de implementação.

**As duas únicas vitórias confirmam a regra.** X1 limpou a supervisão (movimento único, não repetível); B6 **restringiu o que a base pode expressar** (meta como condicionador, não intercepto). Ambas são meta-movimentos sobre a mesma base, ~+0,004 cada — e as campanhas varreram o espaço de movimentos desse tamanho.

**A impressão digital da base aparece no perfil de famílias.** Tudo downstream do branqueamento fala a língua do segundo momento: cauda alta 0,93, variância 0,75 — fluência nativa. Dependência 0,53–0,62 — a família cuja assinatura vive exatamente na estrutura que o filtro AR(10) foi ajustado para remover. Não é que o branqueamento esteja "errado" (o D1 refutou o mecanismo de atenuação específico); é que ele **molda** o que a base enxerga bem.

**E a régua retroativa:** variância de partição ~0,007 dissolve todos os vereditos pequenos do passado — nos dois sentidos. Nada de ±0,003 está de fato resolvido, e por isso mesmo nada de ±0,003 vale re-litigar. Com 7,5 semanas, só apostas com teto ≥0,005–0,01 são sequer mensuráveis.

---

## 3. As decisões fundadoras que nos limitam (respondendo "o que fizemos desde o começo")

**F1 — O axioma importado de 2025.** "Stacking de árvores sobre features estatísticas ganha (~0,9); neural colapsa (~0,5)" foi codificado como lei de arquitetura. Mas o veredito veio de **outra tarefa**: 2025 era batch, 10 mil rótulos únicos, fronteira conhecida — o território natal do GBM tabular e o pior caso possível de uma rede. 2026 é **sequência em streaming com 2,5 M rótulos por passo e métrica transversal** — regime diferente nos dois eixos que decidem quando representação aprendida paga. A rejeição nunca foi medida em casa. E o critério de reabertura que o próprio repo escreveu ("GRU: só em platô, com folga") **disparou hoje, pela boca do autor**.

**F2 — O portão de latência que a plataforma não impõe.** A arquitetura foi dimensionada para ~1,5 ms/passo, O(1) de estado. O limite real da plataforma é **15 horas por semana no total, com paralelismo oferecido no nível de séries** (`INFER_PARALLELISM`). Feita a conta: 20 mil séries × até 1.000 passos ≈ 2,7 ms/ponto num processo — vezes o número de workers. O orçamento verdadeiro é **uma ordem de grandeza acima** do que nos autoimpusemos. Esse portão silenciosamente excluiu: recomputação em janelas, scans multiescala, membros neurais pequenos, embeddings de foundation models. Nós projetamos para uma restrição que não existe.

**F3 — Monocultura de classe de função.** Um GBM, uma base, e todo membro alternativo (rank head, fallback) avaliado como *substituto* ou média trivial. O bagging — a maior alavanca já medida, +0,0046 — é a forma **mais fraca** de ensemble que existe: mesma classe, mesmas features, mesmos dados, RNG diferente. A forma forte (representações diferentes) nunca foi construída porque só existe uma representação. A lição do Netflix Prize em uma linha: os últimos pontos vieram de misturar *classes* de modelo, não de refinar uma.

**F4 — O teto nunca foi computado.** Três campanhas de "medir antes de construir", e o único número que decide tudo — **quanto TS-AUC a informação disponível permite** — não tem medição. Medimos tudo contra o incumbente; nunca contra o possível.

---

## 4. Os sinais que ignoramos (respondendo "qual sinal")

**S1 — A lista dos organizadores.** A página oficial de metodologia sugere seis caminhos. Os quatro primeiros — testes estatísticos vs. histórico, CPD online, features incrementais + classificador, modelos bayesianos sequenciais — são *literalmente* o onyx. Os dois últimos — **deep learning treinado em pares (série, passo)** e **foundation models de séries temporais como extratores de features** — nunca foram tocados. Os organizadores desenharam o mapa e nós exploramos exatamente a metade onde já estávamos.

**S2 — A literatura convergiu em 2025–26.** Embeddings congelados de TSFMs (Chronos/MOMENT) viraram extratores zero-shot competitivos para detecção de anomalia e OOD (THEMIS usa o encoder do Chronos congelado + detecção não supervisionada e bate SOTA em benchmarks; ChronosAD, 2026, reporta ganhos médios de AUC sobre 11 métodos), e há paper de 2026 estudando **não-estacionariedade no espaço de embedding de TSFMs sobre janelas AR(1) com shift de média/variância/tendência** — o nosso problema, descrito na língua deles. O campo está fazendo dois-amostras em espaço de representação pré-treinada; nós fazemos dois-amostras em 183 escalares artesanais.

**S3 — A resposta ao degrau de magnitude média: +0,05 após 200 observações pós-quebra.** Se "média" significa qualquer coisa, 200 observações saturam um bom teste dois-amostras. Um excesso de score de 0,05 com 200 pontos de evidência grita *extração*, não *informação* — mas isso é uma crença, e o Exp0 abaixo a converte em número.

**S4 — O mapa do board continua com um ponto, contaminado.** 0,6201 no placar contra o OOF 0,6100 da semente 42. Sem um segundo ponto limpo, nem sabemos o câmbio OOF→board, nem quantos +0,004 separam a posição atual do top. O número do topo do leaderboard (visível para você, não para mim) é a outra metade deste diagnóstico: se o #1 está a 0,63, o jogo é consistência; se está a 0,67, alguém abriu a porta da seção 6.

**S5 — O dataset é declaradamente sintético + mundo real** (docs oficiais). Heterogeneidade de portadores reais dentro do treino é mais um argumento contra uma base fixa de funcionais paramétricos e a favor de representação que generalize por pré-treino.

---

## 5. Exp0 — o teto, finalmente (1 semana; decide o resto do calendário)

Quatro medições, todas offline, nenhuma treina nada grande:

- **E0a · Skyline com τ-oráculo.** Para positivos, estatística dois-amostras (energia + AD + logvar) entre `online[τ:t]` e o histórico — localização perfeita, só a informação. Para negativos, o melhor scan. TS-AUC por bucket × magnitude. **É o teto informacional.**
- **E0b · Skyline causal sem oráculo.** Máximo sobre grade geométrica de candidatos s de dois-amostras(`online[s:t]`, histórico). O teto alcançável por estatística com localização honesta. A diferença E0a−E0b = custo de não saber τ.
- **E0c · Teste do toco.** GBM de profundidade 1 sobre as 183 features vs. incumbente. Se ≈ (dentro de ~0,003), interações não contribuem → o aprendiz já saturou a base → **a única porta é trocar a base**. Este é o teste mais barato e mais informativo da lista.
- **E0d · Fallback puro como TS-AUC** (`fallback_oof.py` já existe): quanto o GBM adiciona sobre as estatísticas cruas — o tamanho real da camada "aprendizado" hoje.

**Tabela de decisão, registrada agora:** E0a ≤ ~0,66 no agregado ponderado → o teto é real, o platô é da tarefa, e o jogo vira consistência (seção 7-B). E0a ≥ ~0,72 com E0b ≥ ~0,68 → gargalo de extração confirmado → as portas da seção 6 recebem as 6 semanas restantes. Entre os dois → decidir por bucket (o que importa é 150–400, 49% do peso).

---

## 6. A rejeição herdada incorretamente — e as três formas de reabri-la, por risco

A resposta direta à sua pergunta: **a família inteira de representação aprendida/aleatória** — rejeitada por veredito importado de uma tarefa diferente, nunca medida aqui, com as condições habilitadoras hoje todas presentes (2,5 M rótulos por passo; métrica que uma rede otimiza nativamente em batch por passo; orçamento de compute 10× o suposto; augmentação as-of que morreu para o GBM por artefato de `meta_h0` mas é limpa no espaço de resíduos crus; e o platô que o próprio critério do repo exigia).

**P1 — ROCKET-causal (a virada documentada da TSC, sem risco de determinismo).** A história de pivô mais parecida com a nossa: uma década de bases artesanais cada vez mais elaboradas (shapelets, BOSS, HIVE-COTE) e então ROCKET/MiniROCKET — milhares de kernels convolucionais **aleatórios fixos** + pooling + classificador linear — igualou ou bateu tudo por fração do custo. Nós somos o estágio HIVE-COTE. Versão para cá: K≈300–1000 kernels fixos (semente registrada) sobre o fluxo (cru **e** branqueado), pooling causal (PPV e máximo corrente) numa grade geométrica de janelas, cada pooled **padronizado pelo nulo do próprio histórico** (uma passada no init). Sai um vetor de z's por passo → ridge/GBM raso → **membro** do ensemble via stacking por bucket (a maquinaria do A2). 100% numpy, determinístico por construção, O(K) por passo — cabe com folga no orçamento real. Custo: 3–5 dias. Gate barato antes de integrar: AUC do contraste ROCKET sozinho por família em 500 séries (meio dia) — em particular na **dependência**, onde a base atual é analfabeta.

**P2 — Contraste em espaço de foundation model (a sugestão nº 6 dos organizadores).** Embedding congelado (Chronos-Bolt-tiny / MOMENT-small) da última janela W a cada k≈16–32 passos + banco de referência do histórico no init; features = distâncias/projeções discriminativas embedding-atual vs. banco-histórico = **dois-amostras em espaço de representação pré-treinada**. Conta de viabilidade: ~20 mil séries × (1.000/16) ≈ 1,25 M forwards de um encoder pequeno — horas dentro das 15 h/semana com paralelismo. Riscos honestos, em ordem: (i) determinismo 1e-8 no re-run de 30% com torch em processos paralelos — validar no molde local **antes** de qualquer coisa; (ii) domínio: séries z-scoradas quase-ruído são diferentes do corpus de pré-treino — por isso o probe offline primeiro (500 séries, AUC de contraste por família, meio dia); (iii) sobreposição com a base atual — exigir corr_xs < 0,9 com o score do GBM. Se o probe passar, é a aposta de maior teto do documento. Custo: ~1 semana.

**P3 — Membro sequencial pequeno com a loss que o GBM não tem.** TCN/GRU causal minúsculo (30–60 k parâmetros), input = resíduos + score/top-features do GBM (híbrido à la ES-RNN do M4: estrutura estatística + resíduo aprendido), treinado com **par-loss intra-t em batches agrupados por passo** — a otimização direta da métrica que o B4 só conseguiu imitar por gradiente colado, e que uma rede faz nativamente. Augmentação as-of no espaço de resíduos, com o probe de discriminador **rodado antes** (a falha 0,9993 era do pipeline de `meta_h0`; resíduos crus não têm esse canal — mas isso é hipótese até o probe dizer). Export numpy para inferência. Entra como membro, nunca como substituto. Custo: 1,5–2 semanas. **Condicionado ao Exp0** e ao aprendizado de P1/P2.

Nota sobre o que **não** estou propondo: substituir o GBM. A lição tabular (Grinsztajn et al.) continua válida onde a representação já é boa; a proposta é a do Netflix e do M4 — **adicionar uma classe ortogonal e misturar**, com o stacking por bucket que já existe e nunca teve um segundo membro digno.

---

## 7. O calendário das 7,5 semanas (última quota 16/09)

**Semana 1 (agora):** Exp0 completo · submissão-âncora do incumbente (X1+B6, notebook a regenerar/verificar) para o **segundo ponto do mapa** · leitura do topo do leaderboard · início do P1 em paralelo (é barato e independe do Exp0).

**Caminho A — Exp0 diz "gargalo de extração":** Semanas 2–3: P1 medido como membro (probe → integração → K=4 + réplica de partição). Probe do P2 no molde local; se determinismo e AUC de contraste passarem, semanas 3–5: P2. Senão, semanas 3–5: P3. Semana 6: C6 (BOCPD isolado, a pendência eterna) + consolidação. Semanas 7–7,5: bundle final, 2–3 submissões espaçadas, congelamento ~10/09.

**Caminho B — Exp0 diz "teto real":** aceitar publicamente que 0,61→0,62 é o jogo inteiro, e jogá-lo: ensemble robusto a partição (treinar o bag final sobre **as duas partições**), mapa do board com 2–3 âncoras, zero micro-braços, e as semanas restantes em confiabilidade (determinismo, latência, fallbacks) — em empate técnico, ganha quem não quebra. P1 ainda roda (barato, e o teto medido pelo E0b pode estar acima do modelo mesmo no caminho B).

**Fechamentos que este documento mantém:** studentização auto-referencial (3 mortes, teoria explicada), síntese de portadores via features (2 probes reprovados), micro-braços de ±0,003 (abaixo do ruído de partição), re-litigâncias do passado pequeno.

---

## 8. A contra-tese, com a mesma seriedade

É possível que o Exp0 diga que o teto é ~0,65 e que 40% dos positivos sejam indistinguíveis por desenho no passo em que pesam. Os indícios a favor: cauda 0,93 (quando há sinal, a máquina converte), perda difusa, magnitude-baixa colada no controle. Dou a esse cenário 35–45%. Se for ele, a conclusão não é derrota — é que as duas campanhas provaram que o modelo extrai quase tudo que existe, e a competição se decide em variância, robustez e leitura do board. Mas essa frase só pode ser dita **depois** do Exp0 — acreditá-la sem medir seria repetir, na direção oposta, o erro que este documento diagnostica: quatro meses guiados por um veredito que ninguém mediu.

*Registrado antes de qualquer execução: as previsões do Exp0 (§5), a tabela de decisão, e a ordem P1→P2→P3 condicionada a probes. Um braço por mudança. Réplica de partição antes de qualquer pacote. O board só vê bundles.*
