# Plano de Ação — do run v1 para o v2

**Insumo:** relatório de diagnóstico do run v1 (`artifacts/models/v1`, 10.000 séries, 2,54M linhas, 28min).
**Documentos de referência:** `plano_structural_break_realtime.md` (plano técnico, §N) e `plano_repositorio_python.md` (engenharia).
**Conclusão em uma linha:** o pipeline está correto e o modelo aprende sinal real — mas está sendo **treinado e parado com critérios que a métrica não premia**. A correção é barata e mexe em ~10 linhas de `model/train.py`. Quase todas as recomendações do relatório são de prioridade baixa ou prematura, por razões desenvolvidas abaixo.

---

## 0. O que o v1 já provou (não subestimar)

Três coisas foram estabelecidas e não precisam ser revisitadas:

- **Escala:** 10.000 séries, 2,54M linhas, 337 MB, 28 min. O plano previu ~2,4M linhas (§8.1) — bateu. O problema de memória (8+ GB → 337 MB) está resolvido.
- **Determinismo:** PASS no re-run de 10% da plataforma. O gate mais caro de recuperar depois já está verde (§12.4).
- **Detecção real existe:** `cusum_var_up_r150` (ganho 278,8 em apenas 72,8 splits) e `accum_window_var_ln_w250` (180,2 em 70,8 splits) são features de alto valor usadas com parcimônia no topo das árvores. A família **variância/cauda funciona**. Há um lobo secundário em 0,85–0,9 para séries quebradas em t>400 que não existe nas não-quebradas.

Isso não é um pipeline quebrado. É um pipeline funcionando com o objetivo de treino errado.

---

## 1. O achado central que o relatório não faz: objetivo ≠ métrica

### 1.1 A taxa-base é o que o modelo está aprendendo, e ela é irrelevante para a métrica

Da própria finding 3 do relatório, a taxa-base de `y_t = 1{τ≤t}` por faixa de t:

| faixa | n(y=0) | n(y=1) | taxa-base | logit |
|---|---|---|---|---|
| t≤50 | 4.613 | 380 | 7,6% | −2,50 |
| 50–150 | 7.802 | 1.229 | 13,6% | −1,85 |
| 150–400 | 13.585 | 3.816 | 21,9% | −1,27 |
| t>400 | 11.795 | 7.763 | 39,7% | −0,42 |

**Uma amplitude de 2,08 em log-odds, previsível a partir de t sozinho.** Para uma logloss binária, isso é um banquete: é de longe a maior redução de perda disponível no dataset. E pela invariância C1 do plano técnico (§1.2), um componente do score que depende **apenas de t** desloca todas as séries vivas igualmente naquele passo e contribui **exatamente zero** para a AUC transversal — ou seja, para a TS-AUC.

O modelo está resolvendo, com competência, um problema que não é o nosso.

### 1.2 O ganho por split prova o mecanismo

Ganho total é uma métrica enganosa aqui; o que revela a estrutura é **ganho por split**:

| feature | ganho | splits | ganho/split | leitura |
|---|---|---|---|---|
| `meta_t` | 129,3 | 28,4 | **4,55** | poucos splits, valor enorme cada |
| `meta_ln1p_t` | 653,5 | 150,8 | **4,33** | idem — particionamento de taxa-base na raiz |
| `cusum_var_up_r150` | 278,8 | 72,8 | **3,83** | detector real, alto valor, uso parcimonioso |
| `accum_window_var_ln_w250` | 180,2 | 70,8 | 2,55 | detector real |
| `meta_h0_rho1_abs_e` | 381,3 | 610,2 | **0,62** | muitos splits, valor baixo cada |
| `meta_h0_n_h` | 363,9 | 642,2 | 0,57 | idem |
| `meta_h0_ar_r2` | 340,9 | 578,2 | 0,59 | idem |

Duas populações completamente distintas:

- **`meta_t`/`meta_ln1p_t`**: alto ganho/split, poucos splits → estão na **raiz das árvores**, particionando por taxa-base. É o padrão-livro-texto de um modelo modelando o prior.
- **`meta_h0_*`**: ~0,6 de ganho/split, ~600 splits cada, espalhados por ~470 árvores (94 × 5 folds) → ~1,3 split por árvore. Isso é **condicionamento fino**, não hack de taxa-base. É exatamente o comportamento que o plano previu como necessário (§5, features #27–28: "desconte o CUSUM de variância quando ρ̂₁(|e|) do histórico é alto").

Portanto a frase do relatório "dois terços do sinal aprendido vem de condicionamento, não de detecção" **soma duas coisas que não têm nada a ver uma com a outra**, e conclui errado sobre as duas. `meta_ln1p_t` é desperdício; `meta_h0_*` é provavelmente trabalho legítimo. Só um deles precisa de intervenção.

### 1.3 O early stopping está sendo dirigido pela mesma métrica errada

`metric: auc` no LightGBM é a **AUC de linha**, agregando todos os t num pool só. Ela é dominada pela taxa-base: ordenar todas as linhas só por t já entrega uma AUC de linha respeitável (~0,62–0,65, estimando pela tabela acima). Uma vez que o modelo aprendeu g(t), a AUC de linha **satura** — e melhorias reais de discriminação *dentro de cada t* mexem nela por milésimos.

Resultado previsto: parada antecipada muito cedo. Resultado observado: **89, 91, 70, 110, 109 árvores.** O plano técnico previa 400–800 (§8.3). Estamos a **1/5 do esperado**.

Esse é o dano concreto e mensurável: o treino está sendo interrompido antes de o modelo aprender a detectar, porque o juiz da parada não liga para detecção.

### 1.4 O orçamento não é restrição — é o oposto

| medida | valor |
|---|---|
| build de features (medido) | 1.178 s / ~5,05M passos = **233 µs/passo** |
| + predict estimado | ~330 µs/passo |
| orçamento real (15h / 10M passos) | **5.400 µs/passo** |
| **folga** | **~16×** |

O gate auto-imposto de 300 µs/passo (§11) era conservador demais e agora está atrapalhando: ele sugere "podar features", quando a leitura correta é que **temos folga para adicionar**. Ação: relaxar `latency_budget_us_per_step` de 300 para 1500 no config.

---

## 2. Por que a hipótese de "redundância" não se sustenta

O relatório explica as features de média mortas como "LightGBM escolhe um representante de um cluster correlacionado e mata o resto". Os dados do próprio relatório refutam isso:

**Contra-evidência 1 — o representante não engordou.** Se `accum_welford_mean_z` tivesse absorvido o sinal do cluster de média, teria ganho alto. Tem **43,4** (posição #17), com ganho/split de 0,51 — o mais baixo da tabela do topo. Ninguém absorveu nada. O cluster inteiro está morto, representante incluso.

**Contra-evidência 2 — a família de variância não sofre a mesma coisa.** Se o mecanismo fosse ganância-mata-correlacionados, a variância mostraria o mesmo padrão. Mostra o contrário: `cusum_var_up_r150` (278,8) **e** `accum_window_var_ln_w250` (180,2) **e** `accum_welford_var_ln` (73,6) **e** `accum_window_var_ln_w100` (28,1) — quatro features correlacionadas de variância, todas com peso. Mesma estrutura de correlação, resultado oposto.

**Contra-evidência 3 — quatro implementações independentes, todas zeradas.** Janelas (`w010/w025/w050`), EWMAs (`l050/l100/l300`), CUSUMs (`d025/d050/d100`, ±) e o hedge cru não-branqueado (`hedge_ewma_z` = **0**) morrem juntos. Bug de bloco produziria uma família morta, não quatro. Isso aponta para os **dados**, não para o código.

**A leitura correta:** o sinal de média é intrinsecamente fraco neste dataset, enquanto o de variância/cauda é forte. E há uma explicação teórica candidata: para uma série AR com Σφ̂ próximo de 1, um deslocamento de média μ em x produz um deslocamento sustentado de apenas μ(1−Σφ̂) nas inovações. Note que **isso não é culpa do whitening** — o teste sobre x cru sofre a mesma perda (a variância da média de m observações autocorrelacionadas infla por ~1/(1−φ)²; os dois z-scores coincidem). É um limite de informação real do problema, e é consistente com `hedge_ewma_z` = 0 e com `meta_h0_ar_r2`/`meta_h0_rho1_e` estarem no topo das features de condicionamento.

Mas "candidata" não é "confirmada", e as três explicações possíveis pedem respostas opostas:

| hipótese | como o v2 responde |
|---|---|
| o gerador quase não faz quebra de média | aceitar; realocar o orçamento das 9 features de média para variância/cauda/dependência |
| faz, mas Σφ̂→1 torna indetectável | aceitar parcialmente; manter só as features de janela longa (`welford_mean_z`), cortar as curtas |
| faz e é detectável, mas as features têm bug | **corrigir** — seria o maior ganho disponível no projeto |

Distinguir entre elas custa ~30 min de script sobre dados rotulados. É a ação **A1** abaixo.

---

## 3. Veredicto sobre cada recomendação do relatório

| # | Recomendação do relatório | Veredicto | Razão |
|---|---|---|---|
| 1 | Ablação do banco de média (remover 9 features do scorer vivo) | **Rejeitada por ora** | Prematura em três frentes: (a) ganho zero **sob um objetivo quebrado** não é evidência de ausência de informação — refaça a medição depois de A2; (b) o motivo declarado (economizar orçamento O(1)/passo) resolve um problema que não existe: temos 16× de folga; (c) remover do scorer vivo é uma decisão arquitetural quase irreversível baseada num único run de um modelo que ainda não funciona. |
| 2 | Correlação entre features "vencedoras" e "famintas" | **Absorvida em A1** | A hipótese de redundância que ela testaria já está enfraquecida pela contra-evidência da família de variância (§2). O censo de A1 responde a pergunta mais fundamental (existe sinal de média nos dados?) e torna a correlação um subproduto. |
| 3 | Investigar a parada precoce do fold 2 (70 rodadas) | **Rejeitada** | Com `early_stopping_rounds=100`, best-iter em 70 vs. 110 é ruído puro — o fold 2 treinou ~170 rodadas antes de parar. E o critério de parada em si está desalinhado (§1.3), então **nenhuma** contagem de rodadas carrega informação até A2 estar feito. Depois de A2, essa contagem vira um diagnóstico valioso (ver §5). |
| 4 | Rodar a suíte T1–T13 contra o modelo treinado | **Aceita**, reprioritizada | Boa ideia e barata. Mas ela grada um modelo que vamos reconfigurar — então roda **agora como foto "antes"**, não como insumo de decisão, e de novo depois de A2. É a única recomendação do relatório que sobrevive intacta. |
| 5 | Submeter oficialmente antes de decisões de arquitetura | **Aceita, promovida para primeira** | Certa no princípio, errada na posição. Está listada por último e condicionada a "go-ahead", quando na verdade **não temos nenhum número oficial** — todo o projeto está voando cego. Vira A0. |

**Observação de processo.** A manchete do relatório é "No local improvement over fallback yet" (0,5249 vs. 0,5385, 100 séries). Esse número tem erro-padrão de ~0,05–0,07: a diferença de 0,014 é indistinguível de zero, e **as duas leituras são indistinguíveis de 0,5**. Colocá-lo no topo do relatório convida exatamente ao comportamento que o projeto proibiu: otimizar contra um número local dominado por ruído. Ressalva importante: como ele vem de `python -m crunch test` (ferramenta **oficial**, não réplica caseira), o problema aqui não é o de sempre (implementação divergente) — é só tamanho de amostra. Vale checar se `crunch test` aceita rodar sobre mais séries; se aceitar 2.000+, o erro-padrão cai ~4,5× e passa a ser um sinal legítimo, **sem** violar a regra (não há réplica envolvida). Enquanto for 100 séries, o número não deve aparecer como manchete de nada.

---

## 4. Plano de ação priorizado

### A0 — Estabelecer a âncora oficial *(bloqueia todo o resto; começar hoje)*
Submeter o modelo v1 **como está** e o fallback à engine oficial. Custo marginal ≈ zero (já estão construídos). Descobrir, na mesma ação, **qual é o limite de submissões** (por dia/semana) — essa informação decide se o resto do plano é "submeta e itere" ou "acumule correções e submeta em lotes".
**Entrega:** duas linhas em `artifacts/reports/submission_log.md`. É o primeiro número real do projeto.
**Por que primeiro:** sem âncora, "melhorou" não tem significado. E o v1 já existe — adiar a submissão não compra informação nenhuma.

### A1 — Censo de tipos de quebra *(model-free, ~30 min, maior informação por minuto do plano inteiro)*
Script sobre `data/X_train.parquet` com os τ rotulados. Para cada série com quebra, medir o que **de fato** muda em τ:
- Δ média em x e em e (padronizado pelo desvio pré-τ do segmento online)
- Δ log-variância em x e em e
- Δ ρ̂₁, Δ curtose / taxa de excedência
- **distribuição de Σφ̂** entre séries (o fator de atenuação 1−Σφ̂)

**Entrega:** um histograma por eixo + a matriz de correlação entre eixos (absorve a rec. #2 do relatório).
**Decisão que ela destrava:** se Δ média tem massa longe de zero e as features de média continuam mortas depois de A2 → há bug e é o maior ganho do projeto. Se Δ média ≈ 0 → o gerador não faz quebra de média, aceite e realoque. Se Δ média ≠ 0 mas concentrado em séries com Σφ̂→1 → limite de informação, mantenha só janelas longas.
**Por que não foi feito antes:** deveria ter sido o passo 0 do projeto. É a única ação aqui que mede a verdade dos dados em vez de inferir do comportamento do modelo.

### A2 — Corrigir o desalinhamento objetivo/métrica *(a correção principal; ~1h + retreino)*
Duas mudanças em `model/train.py`:
1. **`init_score = logit(p̂_base(t))`**, com `p̂_base(t)` medido empiricamente no treino (suavizado; já temos τ e T). O modelo passa a aprender **só o resíduo** — que é precisamente a discriminação transversal que a TS-AUC mede.
2. **Trocar o early stopping de `auc` para `binary_logloss`.** Com `init_score`, a logloss mede apenas o ajuste residual (o LightGBM soma o `init_score` ao raw score antes da métrica), então ela deixa de ser dominada pela taxa-base — sem precisar de métrica customizada e sem recriar nada parecido com um estimador local de TS-AUC.

Na inferência: `score = sigmoid(logit(p̂_base(t)) + raw)`. Somar o offset de volta é opcional pela invariância C1 (é um deslocamento monótono uniforme em t → AUC_t idêntica), mas manter deixa o score calibrado em [0,1] e custa um lookup.
**Custo:** ~10 linhas + um retreino de 7 min.
**Falsificável:** ver §5.

### A3 — CE6: classificador só-histórico *(~30 min, decide o destino de 7 features de topo)*
Do plano técnico §12.2, nunca executado. Treinar um classificador usando **só** as features do H0 (nenhum ponto online) para prever o rótulo da série, via CV padrão no treino.
- **Se prevê melhor que a taxa-base** → o gerador vaza o rótulo pelo histórico. As `meta_h0_*` estão fazendo trabalho legítimo (mesmo como efeito principal), e a política registrada (§12.2: não explorar deliberadamente) precisa ser reavaliada explicitamente com o usuário.
- **Se não prevê nada** → então qualquer uso das `meta_h0_*` como **efeito principal** injeta um offset por série no ranking transversal sem informação sobre quebra — ativamente prejudicial. O padrão de ganho/split (§1.2) sugere que elas estão majoritariamente em interações (uso correto), mas ganho não distingue os dois casos. Nesse cenário, a ablação que vale rodar é **dropar as `meta_h0_*`** — não o banco de média.

### A4 — Resposta ao degrau alinhada em τ *(~15 min, sanidade global)*
Sobre predições **out-of-fold** do treino, plotar a média de `s_t` alinhada em (t−τ). Se o score sobe em τ → rótulos e features funcionam, é problema de potência. Se sobe em outro lugar ou em lugar nenhum → é problema de alinhamento do rótulo (a questão **A1 do plano técnico**, §14, listada como "em aberto até confirmação" e **ainda não confirmada**).
**Por que importa:** um τ indexado sobre a série completa em vez do segmento online explicaria simultaneamente todos os sintomas observados. O lobo em t>400 da finding 3 sugere que os rótulos estão *aproximadamente* certos, mas "aproximadamente" não basta e o teste é trivial.

### A5 — Suíte T1–T13 contra o modelo treinado *(rec. #4 do relatório, aceita)*
Rodar agora como foto "antes" e de novo depois de A2. T1 (quebra precoce) e T3 (shift sutil) são os pontos cegos conhecidos do fallback; se o modelo treinado repetir o mesmo padrão **depois** de A2, aí sim é evidência de limite de informação e não de configuração.

### A6 — *(condicional)* `lambdarank` com query = t
Se A2 melhorar mas não o suficiente: a métrica é AUC dentro de grupo t, e `lambdarank` com grupos = t otimiza exatamente isso, tornando estruturalmente impossível ao modelo usar t como discriminador dentro do grupo. Riscos reais: grupos enormes (~2.500 linhas), `lambdarank_truncation_level` distorce com rótulo binário, e é mais frágil que `init_score`. **Só depois de A2 ter sido medido oficialmente.**

### Ação transversal
Relaxar `gates.latency_budget_us_per_step` de 300 → 1500 no `configs/default.yaml`, com um comentário apontando para o cálculo de 16× de folga (§1.4). Sem isso, a próxima pessoa que ler o config vai propor podar features de novo.

---

## 5. Previsões falsificáveis (como saber se este diagnóstico está certo)

O diagnóstico de §1 não é uma opinião — ele faz previsões que o retreino de A2 confirma ou destrói em 7 minutos:

1. **Contagem de árvores sobe de ~90 para 300–800.** Se o early stopping estava sendo dirigido pela taxa-base, removê-la deve destravar centenas de rodadas de aprendizado residual. Se continuar em ~90, meu diagnóstico está errado e o problema é falta de sinal, não de objetivo.
2. **`meta_ln1p_t` e `meta_t` despencam no ranking de ganho.** Devem sobreviver como condicionadores (interações), mas perder o ganho/split de ~4,5 que hoje os coloca na raiz.
3. **A participação relativa das famílias de detecção sobe** — não porque ganharam sinal, mas porque o desperdício saiu do denominador.
4. **As features de média continuam mortas.** Esta é a previsão mais importante: se `init_score` não as ressuscita, o problema **não** era o objetivo, e A1 vira a única fonte de resposta.

Se (1) e (4) acontecerem juntos, o quadro fica limpo: objetivo corrigido, e o sinal de média genuinamente não está lá — aí a poda do banco de média (rec. #1 do relatório) passa a ser defensável, com a evidência que hoje falta.

---

## 6. O que explicitamente NÃO fazer agora

- **Não podar features.** Nem o banco de média, nem nada. Orçamento não é restrição (16× de folga) e o critério de "morta" está contaminado pelo objetivo errado.
- **Não perseguir o fold 2.** Ruído de early stopping sob um critério que vamos trocar.
- **Não tomar 0,5249 vs 0,5385 como informação.** Diferença de 0,014 com erro-padrão ~0,06. Se aparecer de novo como manchete de relatório, é bug de processo, não de código.
- **Não mexer em arquitetura** (hazards, K, janelas, features novas) antes de A0+A2. Toda mudança avaliada sobre um objetivo desalinhado gera conclusão inválida — inclusive as que "funcionarem".
- **Não construir estimador local de TS-AUC.** A regra do projeto segue de pé. `crunch test` com mais séries é a exceção legítima (ferramenta oficial, sem réplica), se a plataforma permitir.

---

## 7. Sequência recomendada

```
hoje        A0 (submeter v1 + fallback; descobrir cadência)   ─┐
            A1 (censo de tipos de quebra)                      ├─ paralelizáveis
            A5-antes (suíte T1–T13 no modelo v1)              ─┘
+1 dia      A2 (init_score + early stopping em logloss) → retreino → checar previsões §5
            A4 (resposta ao degrau OOF) sobre o modelo novo
+2 dias     A3 (CE6) → decide o destino das meta_h0_*
            A5-depois (suíte contra o modelo v2)
            submissão v2  ← primeiro número oficial comparável
depois      A6 (lambdarank) só se A2 melhorar e não bastar
```

— Fim do plano de ação —
