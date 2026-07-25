# Campanha de polimento — tudo que ainda é calibrável, tunável ou auditável no modelo atual

**Data:** 2026-07-25 · **Incumbente:** o que foi submetido (X1-soft + B6 + C6-blend), **board 0,6267**, OOF ~0,6125 (partição 42). **Mapa OOF→board:** dois pontos, câmbio ≈ +0,013 consistente. **Alvo declarado desta campanha:** a linha do **rank 10** (ler no leaderboard), não o topo. **Protocolo:** barra 0,0014 · réplica obrigatória na partição 43 · um braço por mudança · o pacote final medido inteiro · 8 quotas restantes (quartas), congelamento ~10/09.

Organizado em seis frentes, por tipo de intervenção. Dentro de cada frente, ordem de execução. Previsões somadas ao final.

---

## A. Auditorias forenses — o que pode ter sido feito errado (custo ~0, executar primeiro)

Nenhuma destas é "ideia nova"; são verificações de fidelidade. As campanhas já acharam quatro bugs reais (feval raw_to_prob, sweep confundido, baseline contaminado, vazamentos do E0a) — a taxa-base de achar mais um não é desprezível.

**A1 — Rótulos pós-W23.** O changelog corrigiu 29 séries com quebra no passo 1 rotuladas como sem-quebra. Pergunta binária: o snapshot local de `y_train` (e o parquet construído dele) é **anterior ou posterior a 8/jun**? Se anterior: 29 falsos-negativos envenenam o treino (cada um destrói pares em todos os passos em que vive) **e** o OOF local mede contra rótulos errados — parte do câmbio de +0,013 pode ser isso. Ação: diff do `y_train` local vs. cloud; se divergir, refresh + rebuild + retreino. Custo: 1h de verificação. Efeito se houver divergência: +0,001 a +0,003 e um OOF mais honesto.

**A2 — Réplica exata da métrica do board no OOF.** O OOF avalia na grade com **thinning** (t>100 passo 2, t>400 passo 4)? O board avalia **todos os passos**. Se a avaliação local herda a grade de treino, o modelo foi selecionado contra uma métrica levemente diferente da real — e o câmbio de +0,013 contém artefato de grade. Ação: recomputar a TS-AUC OOF do incumbente na grade cheia (todos os t, pesos w_t da população viva completa, tratamento de empates igual ao sklearn) e comparar. Custo: uma tarde. Se divergir ≥0,002, **toda seleção futura passa a usar a grade cheia**.

**A3 — O que exatamente é implantado.** Três verificações de 30 min cada: (i) o `infer` usa **todos os 5 folds × 4 sementes** (média dos 20 boosters) ou um subconjunto? Se um subconjunto, há ensemble grátis na mesa. (ii) O `init_score`/curva de taxa-base foi **reajustado sob os pesos do X1** quando o X1 foi adotado? Se não, há um descasamento de treino silencioso desde então. (iii) A emissão do score é float64 sem clip/arredondamento intermediário? (Empates artificiais valem 0,5 no AUC.) Bônus de 5 min: confirmar `zero_as_missing=false` no LightGBM — LLRs em zero são informação, não ausência.

**A4 — Pendências do Exp0.** E0c na partição 43 (~1h20) e a higiene do `v4_k4 = 0,6133 > incumbente 0,6125` — se esse número for real e não artefato de grade/semente, o incumbente está mal escolhido e o polimento inteiro parte da referência errada.

**A5 — Os primeiros passos.** O que o pipeline emite em t ∈ [1, 10], com metade das features em warmup? Quebras em τ=0/1 existem (as 29 do W23 eram exatamente isso). Verificar se o blend com o fallback nos passos frios está ativo e razoável — t≤50 pesa 8% e está em 0,53.

**A6 — Clip do resíduo.** Quais features veem `e` clipado vs. `e_raw`? A família cauda (a melhor, 0,93) vive nos extremos — um clip agressivo no lugar errado corta exatamente o sinal que mais paga.

## B. Polimento com curva já medida (quase grátis, efeito conhecido)

**B1 — Bag 4→7 sementes.** A curva medida na campanha X: K=4 captura ~83% do ganho; K=7 satura. Diferencial: **+0,0008 a +0,0010, só compute** (3 sementes novas × 5 folds ≈ 45 min). É o item de melhor razão efeito/custo/risco da lista inteira.

**B2 — Espaço de média do bag.** Hoje a média é em probabilidade. Testar média em **logit** (offline, minutos, mesmos OOFs): tipicamente ±0,0005 — se vier positivo nas duas partições, é de graça.

**B3 — O bundle de pós-processo, otimizado nas duas partições.** Peso do C6-blend (fixado em w=0,5 numa varredura de uma partição) + gate do A1 (V-EMA em t>150) reotimizados **em conjunto** nas partições 42 E 43, adotados como pacote único. Individualmente sub-barra; como bundle, +0,001 a +0,002.

## C. Vizinhanças dos dois vencedores (mecanismo comprovado, nunca varridas)

**C1 — A vizinhança do B6.** O `feature_contri=0,6` foi **um chute único** — o maior ganho recente do projeto nunca teve o valor varrido nem o conjunto penalizado questionado. Dois braços: (i) valor ∈ {0,4; 0,5; 0,7; 0,8} com o conjunto atual; (ii) conjunto estendido — penalizar também os rastreadores de t que o xs-SHAP expôs (`conformal_logm_abs`, `mmd_joint_slow_cal`) com contri próprio. Cada braço: triagem na 42, confirmação na 43. Previsão: +0,001 a +0,003.

**C2 — A vizinhança do X1.** `floor=0,3` e a normalização por q95 também foram chutes; `delta_mean` nunca entrou no d_i. Dois braços: floor ∈ {0,15; 0,5}; d_i com delta_mean incluído. Previsão: 0 a +0,002.

## D. Os knobs nunca tocados do LightGBM (as curiosidades pedidas)

**D1 — `linear_tree=true`.** O achado central do E0c é que o problema é **aditivo e suave**: o modelo é funcionalmente um GAM, e árvores de degraus aproximam curvas suaves com escadinhas. Folhas **lineares** ajustam exatamente o que o E0c diz que existe — é o knob obscuro com o melhor mecanismo da lista. Um braço, K=4, duas partições. Previsão: −0,001 a +0,004.

**D2 — `max_bin` 255→511/1023.** Resolução do histograma nas caudas dos LLRs (onde a evidência forte vive). Nota técnica: como o LightGBM binariza por quantis, transformar features monotonicamente não muda nada — **a resolução vem do número de bins, não da transformação**. Braço barato (rebuild de bins, treino normal). Previsão: 0 a +0,002.

**D3 — C4-mini: o sweep que nunca rodou, agora com protocolo.** O harness está corrigido desde a campanha B/C e os HPs atuais foram escolhidos na era da semente 42 — suspeitos por construção. Grade pequena e dirigida (learning_rate×cap, `min_data_in_leaf`, `lambda_l2`, `feature_fraction`, e uma célula `dart`): triagem K=2 na 42, top-3 confirmados K=4 na 43. Previsão: 0 a +0,004.

**D4 — `refit` sobre 100% dos dados.** Recurso pouco conhecido do LightGBM: manter a **estrutura** das árvores selecionada em CV e reajustar só os **valores das folhas** com todos os dados (o fold de validação incluído). Dá ao modelo implantado os 20% de dados que a CV lhe nega, sem tocar na seleção. Determinístico. Braço barato; medição contra o board exige uma quota (o OOF não enxerga esse ganho por construção — decidir pela âncora). Previsão: 0 a +0,002 no board.

**D5 — Micro-higienes:** `enable_bundle` (EFB com dezenas de colunas NaN-de-warmup pode agrupar mal — testar off), `first_metric_only`, `force_row_wise` documentado. Meia hora, provavelmente nada, custa quase zero.

## E. A distribuição de treino — mesmo modelo, mesmas features, dados melhores

**E1 — Des-thinning.** O thinning (t>100 passo 2, t>400 passo 4) foi uma decisão de **velocidade de build da fase P0, nunca medida** — e descarta metade das linhas exatamente na região que carrega 49% do peso da métrica. O E0c mostrou que os ajustes aditivos são famintos de dados (1.500–3.200 tocos até convergir). Rebuild sem thinning até t≤400 (parquet ~1,6×, build ~45 min), retreino K=4, duas partições. É o braço de treino com melhor mecanismo. Previsão: +0,001 a +0,004.

**E2 — Refresh do snapshot** (liga com A1): garantir que parquet e OOF nascem do dataset pós-correções.

**E3 — [rotulado: fora do escopo declarado, registrado por dever]** As linhas de 2025 como dados extras para o **mesmo** modelo não são feature nova nem modelo novo — são mais amostra para os mesmos ajustes aditivos. É o único item da lista com teto acima de +0,005. Fica registrado com os portões de sempre (regra, suporte, probe de domínio); a decisão de abrir é sua.

## F. Os knobs internos de φ — re-tunar canais existentes (substituição, não adição)

A lição "largura mata" era sobre **adicionar** funcionais; **afinar** os hiperparâmetros internos dos canais que já dominam o xs-SHAP é outra operação: melhora o SNR do mesmo canal, sem diluir sorteio de features.

**F1 — Três knobs, um braço cada, nos canais top:** a grade de deltas do CUSUM de variância (densidade/alcance — o canal da família campeã), o bandwidth/taxa do `mmd_joint_slow`, o ε do martingale conformal. Cada um exige rebuild — por isso só os três de maior importância. Previsão: 0 a +0,002 cada.

**F2 — Threshold do `vol_adjust`.** O gate liga/desliga por limiar duro de `rho1_abs_e`: séries na fronteira recebem tratamento descontínuo. Suavizar (rampa) ou varrer o limiar. Micro, um braço.

**F3 — Ordem do AR por série** (AIC em {5..15}) em vez de 10 fixo: mexe em tudo downstream, caro, efeito desconhecido — **último da fila**, só se sobrar semana.

---

## Sequência (8 semanas de quota, quartas-feiras)

| semana | execução | quota |
|---|---|---|
| 1 | **Frente A completa** (A1–A6) + B1 + B2 + A4 | — |
| 2 | D1 (linear_tree) ∥ E1 (des-thinning) — os dois braços de melhor mecanismo | âncora se a frente A achou bug |
| 3 | C1 (vizinhança B6) ∥ B3 (bundle de pós-processo) | — |
| 4 | D3 (C4-mini, triagem) ∥ C2 | âncora do melhor parcial |
| 5 | D3 (confirmações na 43) + D2 + D4 | — |
| 6 | F1 (os três knobs de φ) + D5 | âncora |
| 7 | montagem do pacote final, medido **inteiro** vs. incumbente nas duas partições | âncora do pacote |
| 8 | margem de emergência, verificação bit-a-bit, congelamento ~10/09 | quota final 16/09 |

**Regras mantidas:** braços de ±0,003 só existem porque a réplica na 43 é obrigatória antes de qualquer adoção; nada entra no incumbente sem IC pareado favorável nas duas partições; o board só vê bundles.

## Aritmética honesta e o contrato desta campanha

Somando as faixas com taxa de acerto histórica (~1 em 3): **+0,005 a +0,010 de OOF**, board ~**0,632–0,637**. Pré-registro: isso disputa a linha do rank 10 se ela estiver ≲0,638 — e **não alcança 0,65**. Se a leitura do leaderboard mostrar o rank 10 acima de ~0,64, esta campanha sozinha não paga prêmio, e a decisão sobre o E3 (ou sobre reabrir o pivô) volta à mesa com esse número na mão. O primeiro entregável da semana 1, antes de qualquer braço, é portanto uma leitura sua: **o score do 10º lugar**.
