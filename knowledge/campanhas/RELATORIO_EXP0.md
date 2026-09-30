> ## ⚠️ ERRATA (2026-07-25) — este relatório mediu na GRADE ERRADA
>
> A frente A da campanha de polimento achou que a TS-AUC OOF era agregada sobre a grade com
> **thinning** (399 passos), enquanto o board avalia **todos** os ~999. `AUC_t` em cada passo retido
> é exato, mas o agregado não: `t>400` recebe **31,9%** do peso no board contra os 16,5% que se
> media. Ver `docs/RELATORIO_POLIMENTO.md` §1 e `NOTAS_AGENTES.md` §5.
>
> **O que muda neste documento:**
>
> - **§2.1 e §6 não sobrevivem.** O E0c, refeito no par limpo com a ponderação do board, dá
>   **+0,0059, IC [+0,0006; +0,0117] — EXCLUI 0** (contra +0,0039, IC incluindo 0, aqui reportado).
>   A conclusão *"`g` está saturado"* e a linha *"melhorias de `g` ficam formalmente fechadas"* caem:
>   as interações valem 4× a barra. O ganho sempre viveu em `t>400` (+0,0144), o bucket cujo peso
>   dobra.
> - **§2.2 (ensembling) e §4 (a aritmética do alvo) precisam ser refeitos.** O câmbio de +0,0021
>   sobre 16 membros e o deságio de 0,46 foram calibrados na mesma grade.
> - **§1 e §5 seguem válidos.** Os controles do E0a (vazamento de comprimento e de placebo), o
>   achado de que `-len(online)` vale 0,6506 e o veredito sobre a forma D não dependem da ponderação.
> - **O "câmbio OOF→placar" citado na abertura não existia** — era a grade. Reponderado, o OOF prevê
>   o placar com erro de ~0,002.
>
> A contra-tese do §7 (item 2, *"o E0c é uma semente e uma partição"*) apontava na direção certa por
> um motivo diferente do real: o problema não era a semente, era o instrumento.

# Relatório Exp0 — o beco sem saída desta construção, e as formas possíveis de sair dele

**Data:** 2026-07-24 · **Repo:** `onyx` (ADIA Lab Structural Break Challenge — Real-Time)
**Insumos:** `DIAGNOSTICO_ESTRUTURAL.md` §5 (o experimento aqui executado), `RELATORIO_CAMPANHA_B_C.md`, `RELATORIO_CAMPANHA_X0_X4.md`, HISTORICO §13–14.
**Posição de partida:** OOF `0,6125` (incumbente bag4, partição 42) · board `0,6201` · **topo do leaderboard `0,65`, 35 posições à frente**.
**Alvo implícito, e é ele que organiza este documento:** **+0,03**.

---

## 0. Sumário executivo

O Exp0 foi desenhado para medir o teto da *tarefa*. Ele não conseguiu — e a razão de não ter conseguido é, por si só, informativa (§6). O que ele mediu, com instrumento e intervalo de confiança, foi o teto da **construção**: quanto ainda existe para extrair sem trocar a base de evidência.

**Três camadas, três saturações, todas medidas hoje:**

| camada | instrumento | rendimento restante medido |
|---|---|---|
| **`g` — o aprendiz** | E0c: toco `max_depth=1` vs incumbente | **+0,0039, IC [−0,0015; +0,0100] — inclui 0** |
| **ensembling** | câmbio empírico sobre **16 membros** | **máximo +0,0021** (nenhum chega a +0,003) |
| **calibração / supervisão** | campanha X + B/C, **25 braços** | **+0,008 no total**, com uma adoção retratada |
| *(controle)* localização de τ | E0a com oráculo + placebo | +0,0014 |

Somando com generosidade e supondo que nada regrida: **~+0,01 em sete semanas, contra um alvo de +0,03.** A distância não é de esforço, é de ordem de grandeza.

**E a aritmética do combinar fecha a última porta barata (§4):** para chegar a 0,65 misturando, o segundo modelo precisa ser tão forte quanto o atual **e** quase descorrelacionado — e, sob a leitura calibrada pelos dados, nem isso basta. **A forma "membro auxiliar descorrelacionado" não entrega +0,03. O alvo de qualquer pivô tem de ser um modelo que sozinho chegue a ~0,63+.**

---

## 1. O que foi medido, e como

Todas as linhas abaixo estão na **mesma grade** (o OOF do incumbente: 2.541.134 linhas, 10.000 séries, partição de folds 42), porque comparar scores em grades diferentes mede a grade, não o score. Semente de boosting 777 onde aplicável.

| score | geral | t≤50 | 50<t≤150 | 150<t≤400 | t>400 |
|---|---|---|---|---|---|
| incumbente **bag4** (B6) | **0,6125** | — | — | — | — |
| incumbente 1 semente | 0,6067 | 0,5317 | 0,5698 | 0,6232 | 0,6548 |
| **E0c — toco `d=1` (aditivo puro)** | **0,6028** | 0,5289 | **0,5740** | 0,6182 | 0,6404 |
| E0a — skyline, oráculo de τ + scan | 0,5733 | 0,5285 | 0,5581 | 0,5810 | 0,5975 |
| E0a — skyline, janela de referência única | 0,5704 | 0,5322 | 0,5555 | 0,5791 | 0,5877 |
| E0b — skyline causal (scan, sem oráculo) | 0,5719 | 0,5271 | 0,5548 | 0,5785 | 0,6023 |
| E0d — fallback puro (sem aprendizado) | 0,5601 | 0,5258 | 0,5474 | 0,5653 | 0,5821 |

E por magnitude da quebra (positivos restritos por tercil de divergência do censo A1; negativos sempre inteiros, porque AUC é comparação par a par):

| score | baixa | média | alta |
|---|---|---|---|
| incumbente | 0,5352 | 0,5812 | 0,7030 |
| **toco `d=1`** | 0,5162 | **0,5838** | **0,7071** |
| skyline (E0b) | 0,5258 | 0,5640 | 0,6267 |
| fallback | 0,4485 | 0,5226 | 0,6974 |

---

## 2. O beco sem saída, camada por camada

### 2.1 `g` está saturado — o aprendiz não tem mais o que extrair da base (E0c)

**O teste.** Treinar o incumbente inteiro — mesmas 183 features, mesmos pesos X1-soft, mesmo `feature_contri` do B6, mesma partição, mesma semente — mudando **uma coisa**: `max_depth=1`, `num_leaves=2`. Isso transforma o LightGBM num **modelo aditivo puro**: uma soma de transformações univariadas, **zero interações** entre features. Teto de 5000 rodadas com parada em 300 para que o toco não perdesse por falta de capacidade (usou 1h22 de treino contra ~9 min do incumbente).

**O toco não foi truncado — e isto fecha a objeção óbvia.** Árvores efetivamente treinadas por fold: **1529, 1684, 2024, 3186, 1600** — a parada antecipada disparou em todos os cinco folds, nenhum encostou no teto de 5000. O toco convergiu no seu próprio ótimo. Para contraste, o **incumbente para em 79–117 árvores por fold**: ~90 árvores de 63 folhas (~5.700 folhas) contra ~2.000 árvores de 2 folhas (~4.000 folhas) — capacidade total da mesma ordem, gasta de dois jeitos opostos. Que o incumbente sature a logloss de validação em menos de 120 rodadas é, por si só, mais um sintoma da mesma coisa que o E0c mede.

**O resultado.** `0,6028` contra `0,6067`. O toco recupera **99,4%**. Bootstrap pareado por série (300 réplicas):

| bucket | Δ atribuível a **todas** as interações | IC 95% | exclui 0 |
|---|---|---|---|
| **geral** | **+0,0039** | **[−0,0015; +0,0100]** | **não** |
| t≤50 | +0,0029 | [−0,0136; +0,0186] | não |
| 50<t≤150 | **−0,0042** | [−0,0152; +0,0063] | não |
| 150<t≤400 | +0,0050 | [−0,0010; +0,0116] | não |
| t>400 | +0,0144 | [+0,0077; +0,0213] | **sim** |

**Por que isto é decisivo.** A estrutura de interação de um GBM de 63 folhas e ~1500 árvores é **estatisticamente indistinguível de zero no agregado**. E na decomposição por magnitude o toco *ganha* do incumbente exatamente onde existe sinal: média `0,5838` vs `0,5812`, **alta `0,7071` vs `0,7030`**. As interações só trabalham em `t>400` e nas séries quase-nulas — o que a profundidade compra é **calibração transversal entre séries fracas**, não detecção.

Tradução: **o modelo é, funcionalmente, um GAM sobre 183 canais artesanais, e está no limite do que esses canais expressam.** Não há um aprendiz melhor esperando ser encontrado. Trocar objetivo, profundidade, regularização, número de folhas — tudo isso opera dentro de uma capacidade que já não é o vínculo. É a explicação mecânica de por que 25 braços renderam +0,008.

### 2.2 O ensembling está saturado — e o teto dele é +0,002

**O teste.** Todo OOF que este repo já produziu (16 modelos: lambdarank, BOCPD, splice por regime, poda, spec/ord/mrep, wt-align, monotone, X1-hard, X0-tsauc, V4, fallback, skyline) misturado com o incumbente **bag4** por rank-average intra-`t`, varrendo o peso `w ∈ {0,1 … 0,5}` e ficando com o melhor. Correlação = Pearson dos **postos percentuais dentro de cada `t`** — a única correlação que a métrica enxerga.

| membro | solo | corr intra-`t` | melhor ganho | w* |
|---|---|---|---|---|
| `v4_k4` | 0,6133 | 0,858 | **+0,0021** | 0,5 |
| `mrep_bag3` | 0,6124 | 0,854 | +0,0017 | 0,5 |
| `c6_bocpd_bag4` | 0,6121 | 0,901 | +0,0011 | 0,5 |
| **skyline (E0b)** | 0,5719 | **0,321** | +0,0009 | 0,1 |
| `c5_splice_bag4` | 0,6134 | 0,946 | +0,0007 | 0,5 |
| `x1_hard_bag4` | 0,6098 | 0,875 | +0,0004 | 0,3 |
| `x0_tsauc_bag4` | 0,6033 | 0,720 | +0,0004 | 0,2 |
| `v1_rank` (lambdarank) | 0,5852 | 0,455 | +0,0003 | 0,1 |
| *(outros 7)* | 0,607–0,610 | 0,85–0,91 | ≤ +0,0003 | — |
| `b1_wtalign_bag4` | 0,6071 | 0,879 | −0,0000 | 0,1 |
| `fallback` | 0,5601 | 0,336 | −0,0003 | 0,1 |

**O melhor ganho de mistura já disponível neste projeto é +0,0021.** Os dois extremos do espaço — muito ortogonal e fraco (skyline, corr 0,32) e forte e correlacionado (`v4_k4`, corr 0,86) — rendem +0,0009 e +0,0021. Nada no meio faz melhor.

E o skyline, o membro mais ortogonal já construído, testado com o instrumento formal:

| baseline | Δ | IC 95% | exclui 0 |
|---|---|---|---|
| incumbente **1 semente** (0,6067) | +0,0016 | [+0,0002; +0,0027] | sim |
| incumbente **bag4** (0,6125) | **+0,0008** | **[−0,0004; +0,0020]** | **não** |

Quase todo o ganho aparente era ruído de semente que o bagging já colhe. **Ortogonalidade sozinha não paga.**

### 2.3 A calibração está saturada — o histórico já dizia, agora tem mecanismo

Duas campanhas, ~25 braços, variando peso, objetivo, normalização, restrição estrutural, parada, partição, pós-processamento: **2 adoções** (X1-soft +0,0040; B6 +0,0036), uma adoção **retratada no mesmo dia** (C5), e o resto morto. O A3 (emulador do nulo real + studentização) mediu **−0,0457**: quanto melhor o nulo, pior o resultado — assinatura de um score que já ordena por probabilidade a posteriori.

O E0c dá a esse histórico a explicação mecânica que faltava: se o aprendiz já esgotou a base **aditivamente**, então mudar como ele é supervisionado é reparametrizar uma função que já está no seu limite de informação. As mortes não foram azar de execução.

### 2.4 O controle: não é falta de saber *onde* a quebra está (E0a)

O E0a dá ao detector a **localização perfeita de τ** e mede quanto isso vale. Duas cautelas foram necessárias para a medição não mentir:

1. **Vazamento de comprimento.** A primeira versão limitava a grade de janelas por `len(online)` — informação do futuro. Nesta base, `-len(online)` sozinho vale **TS-AUC 0,6506**, *acima do incumbente* (§5). Corrigido: a grade depende só do histórico.
2. **Vazamento de placebo.** Se o positivo pontua com `max(oráculo, scan)` e o negativo só com `scan`, `max(a,b) ≥ b` produz AUC sozinho. Sem controle o E0a marcava **0,6634**; com controle, **0,5733**. **~60% do "teto" era o rótulo entrando pela porta dos fundos.** O controle: toda linha negativa também recebe uma janela de referência, com início sorteado da distribuição empírica de τ condicionada a `r<t`.

**Com os dois controles, o oráculo perfeito de τ vale +0,0014** (0,5733 contra 0,5719 do scan honesto). Qualquer proposta justificada por "localizar melhor a quebra" — scans multiescala, CPD online mais fino, refinamento de BOCPD — aponta para um problema que não existe.

*(Colateral, mas registrado: o branqueamento AR(10) vale **+0,0232** para esta família — skyline cru 0,5487 → branqueado 0,5719. A suspeita do diagnóstico de que o filtro estaria cegando a base não se sustenta.)*

---

## 3. Por que isto é um **beco sem saída**, e não "só mais difícil"

O argumento é estrutural, não estatístico. O sistema é

```
score = g( φ(x) )
```

e existem exatamente **três lugares** onde se pode trabalhar sem tocar em `φ`:

1. **melhorar `g`** — capacidade, objetivo, regularização, hiperparâmetros;
2. **combinar vários `g`** — ensembling, stacking, bagging;
3. **reparametrizar a supervisão de `g`** — pesos, rótulos, normalização, pós-processamento.

O E0c mede (1) em **+0,0039 com IC incluindo 0**. E (1) domina (2): se um único `g` já extrai aditivamente tudo o que `φ` carrega, uma média de vários `g` sobre o **mesmo `φ`** não pode extrair mais — é a mesma informação, filtrada pelo mesmo gargalo. A tabela do §2.2 confirma empiricamente o que o argumento prevê: **+0,0021 no melhor caso, e os membros fortes todos em corr ≥ 0,85 exatamente porque são funções das mesmas 183 colunas.** E (3) é uma reparametrização de (1), com 25 braços de evidência.

**Os dois caminhos que ainda tinham medição pendente — (1) e (2) — somam ~+0,006 no melhor cenário**, e o terceiro tem 25 braços de histórico dizendo que rende ~+0,003 por braço quando dá certo, com metade dos braços regredindo. Sendo generoso e supondo que nada regrida, a construção inteira entrega **~+0,01 em sete semanas, contra um alvo de +0,03.** Um fator de três a cinco. Não é uma questão de mais braços, mais sementes ou mais paciência: é que o objeto que produz o score já usa toda a informação que a base de evidência contém.

Há um segundo motivo, mais incômodo, e ele explica por que "insistir" é ativamente ruim: **a variância de partição de folds é ~0,007** — maior que qualquer efeito já medido em 25 braços, e 5× maior que a barra de decisão (~0,0014). Um programa de braços de ±0,003 dentro de um ruído de 0,007 não é lento; é **ruído sendo colhido como se fosse sinal**, e o C5 (adotado e retratado em 24 h) é a prova documentada de que isso já aconteceu.

### 3.1 Um corolário que reorganiza o cardápio

Correlação entre dois modelos **não é parâmetro livre**. Dois modelos bons sobre a *mesma informação* aproximam o mesmo posterior e convergem em correlação — foi isso que o A3 mediu ao ver a studentização só subtrair. **Correlação baixa com força alta exige informação diferente, não classe de função diferente.**

Isso reordena a prateleira de pivôs inteira: um método novo que seja apenas uma **nova função** dos mesmos resíduos branqueados (kernels aleatórios, outra arquitetura de árvore, outra rede sobre as mesmas features) vai **correlacionar assim que ficar forte**, e cairá na linha `ρ ≥ 0,85` da tabela do §4 — onde o ganho é ~zero. O skyline é a demonstração barata disso: é uma família de features fixas, multiescala, padronizadas pelo nulo, sem aprendizado — e entregou 0,572 sozinho, +0,0009 misturado.

---

## 4. A aritmética do alvo: quanto precisa vir de onde

Sob o modelo binormal, uma AUC de `0,6125` corresponde a uma separação `d = 0,404` em unidades de desvio do score. Misturando dois scores com pesos ótimos, ruídos de correlação `ρ` e forças `d₁, d₂`:

```
d' = [(1−w)·d₁ + w·d₂] / √[(1−w)² + w² + 2w(1−w)ρ]        AUC = Φ(d'/√2)
```

O modelo foi **calibrado no único ponto onde é verificável**: `v4_k4` (solo 0,6133, ρ=0,858) → previsto +0,0045, **medido +0,0021**. Deságio empírico ≈ **0,46**. As duas colunas abaixo são o otimista puro / o calibrado por esse deságio:

| **força do 2º modelo (solo)** | ρ=0,0 | ρ=0,3 | ρ=0,5 | ρ=0,7 | ρ=0,86 |
|---|---|---|---|---|---|
| 0,58 | 0,637 / 0,624 | 0,622 / 0,617 | 0,616 / 0,614 | 0,613 / 0,613 | 0,613 / 0,613 |
| 0,60 | 0,649 / 0,629 | 0,631 / 0,621 | 0,623 / 0,617 | 0,616 / 0,614 | 0,613 / 0,613 |
| **0,6125** *(gêmeo de força igual)* | **0,657 / 0,633** | 0,639 / 0,625 | 0,629 / 0,620 | 0,622 / 0,617 | 0,617 / 0,614 |
| 0,63 | 0,669 / 0,639 | 0,650 / 0,630 | 0,640 / 0,625 | 0,633 / 0,622 | 0,630 / 0,621 |
| 0,65 | 0,684 / 0,646 | 0,664 / 0,637 | 0,655 / 0,632 | 0,650 / 0,630 | 0,650 / 0,630 |

**Como ler esta tabela — é o núcleo do relatório:**

- Um **gêmeo perfeito** (mesma força, correlação **zero** — objeto que não existe na natureza) entrega **0,657 no otimista e 0,633 no calibrado**. Ou seja: o melhor caso concebível da forma "combinar" **empata ou fica aquém** de 0,65.
- Um membro a **0,58–0,60** — que é onde uma primeira implementação nova realisticamente aterrissa — entrega **+0,004 a +0,017**. Não move a agulha.
- Sob a leitura calibrada, **nenhuma célula chega a 0,65 a menos que o modelo novo já esteja em ~0,65 sozinho.**

O deságio de 0,46 é calibrado em **n=1**, e num membro (`v4_k4`) que é *irmão* do incumbente — modelos irmãos compartilham erro estruturado além da correlação de postos, então o deságio pode ser pessimista para um modelo genuinamente independente. A verdade provavelmente está entre as duas colunas. Mas mesmo na coluna otimista, a conclusão de forma se mantém.

> **Conclusão de engenharia:** o alvo de qualquer construção nova **não é "ser diferente"** — é **chegar a ~0,63+ sozinha**. Um auxiliar descorrelacionado de força moderada é uma forma de solução que a aritmética exclui.

---

## 5. As formas possíveis de pivô

Quatro formas estruturais. A pergunta não é "qual modelo", é "que **tipo de objeto** precisa ser construído".

### Forma A — segundo modelo descorrelacionado, **combinado** com o atual

**O que é.** Manter o GBM como está e somar um membro de outra classe via o stacking por bucket que já existe.

**Exigência medida.** Solo ≥ ~0,61 **e** ρ ≤ ~0,3 simultaneamente (§4). Abaixo disso o ganho é de ordem +0,002–0,01.

**Veredito.** **Insuficiente sozinha, e não confiável como plano.** Mesmo o gêmeo ideal para em 0,633–0,657. Todo o espaço já explorado (16 membros) satura em +0,0021, e o único membro de baixa correlação já construído rende +0,0008 com IC incluindo 0. A forma A **só funciona como consequência da forma B** — isto é, se o objeto novo já for forte por conta própria, aí sim vale misturar em vez de substituir.

### Forma B — modelo novo que **substitui** integralmente o atual

**O que é.** Uma segunda solução completa do problema, construída sobre outra representação, avaliada de ponta a ponta e adotada se vencer.

**Exigência medida.** Chegar a **≥0,65 sozinho** para bater o alvo por substituição pura — ou a **~0,63 sozinho** e, aí, a mistura com o incumbente completa o caminho (linha 0,63 da tabela: 0,621–0,669 conforme a correlação).

**Veredito.** **É a única forma que a aritmética admite** — e observe que **A e B não são alternativas: são o mesmo investimento.** Você constrói o objeto novo, mede `solo` e `ρ`, e só então decide misturar ou substituir. Essa decisão custa uma tarde e é tomada com número. O que **não** se pode fazer é planejar a forma A — sair para construir "um membro auxiliar" — porque isso define o alvo de engenharia no lugar errado desde o primeiro dia: um time que persegue ortogonalidade entrega 0,57/ρ=0,32, e um que persegue força entrega algo que pode virar as duas coisas.

**Consequência prática:** o critério de sucesso do pivô, declarado antes de começar, é **força solo**, medida em subamostra, com corte em ~0,60. Ortogonalidade é medida *depois*, e serve para escolher entre misturar e substituir — nunca como portão de entrada.

### Forma C — **nova informação** dentro da construção atual (novos canais em `φ`, mesmo `g`)

**O que é.** Não trocar o modelo: trocar o que ele observa. Acrescentar colunas a `φ` que carreguem informação que os 183 canais atuais não contêm.

**Por que é atraente.** Integra nativamente, herda todo o pipeline (determinismo, latência, folds, bagging), custa dias em vez de semanas, e não exige que nada novo seja "forte sozinho" — o GBM combina.

**Por que é armadilha, e a evidência disso.** A campanha já fez isso **sete vezes** (spectral, ordinal, multirep, jumps, varloc, lmoments, BOCPD) e o rendimento agregado foi ~0. O motivo é o corolário do §3.1: aquelas famílias eram **funções novas da mesma observação** — o fluxo de resíduos branqueados. O E0c explica por que isso não podia funcionar: um aprendiz aditivamente saturado sobre `φ` não melhora quando se acrescenta a `φ` uma coluna que é função das colunas que ele já tem. Pior: `largura sem ordenação nova dilui o sorteio de feature_fraction` — medido em três regressões (V5 −0,0042, F1 −0,0069).

**Quando a forma C é legítima.** Só se o canal novo for informação que **não está no fluxo de resíduos branqueados**. As categorias que sobram, sem citar implementação:
- (i) a **estrutura descartada pelo branqueamento** — o filtro AR(10) congelado joga fora a dinâmica que ele modela; *contra-indício medido: o branqueamento vale +0,023, então o que ele descarta não é obviamente útil*;
- (ii) o **histórico como sequência**, hoje comprimido em AR(10) + quantis + fingerprint — milhares de pontos reduzidos a algumas dezenas de escalares;
- (iii) **estrutura de população** aprendida no treino e aplicada por série na inferência (a que "família de gerador" esta série pertence); *contra-indício: o B6 ganhou justamente **penalizando** condicionadores por série*;
- (iv) **priors externos**, vindos de pré-treino em outros dados — a única categoria que traz informação genuinamente de fora do dataset.

**Veredito.** **Barata e vale um braço**, mas só nas categorias (ii) e (iv), e com a expectativa calibrada: a forma C não tem histórico de entregar +0,03 aqui. Ela é o caminho certo se o objetivo for +0,005–0,01; não é se o objetivo for o topo.

### Forma D — canal fora do modelo (estrutura do gerador / vazamento)

**O que é.** Explorar regularidade da *construção do dataset* em vez de do processo estocástico.

**O que foi encontrado.** `-len(online)` vale **TS-AUC 0,6506** na grade OOF (0,6312 na grade completa) — **acima do incumbente**. Em `t=50` o comprimento mediano do online é **227 nos positivos contra 548 nos negativos**: quase separação. O mecanismo é estrutural (um positivo em `t` tem τ<t, e a série de quebra tem o fluxo truncado em relação a uma sem quebra).

**Por que está fechado.** O contrato oficial tipa o online como **`Iterable[float]`**, não `List` — `infer(datasets: Iterable[Tuple[List[float], Iterable[float]]])`. O histórico é dimensionável; o fluxo online, por desenho, não. `len()` não existe do lado de lá. E o proxy causal foi testado: **`n_h` (comprimento do histórico, conhecido em `t=0`) vale 0,4934** — nada.

**Veredito.** **Fechado, e não recomendado mesmo se abrisse** — forçar a interface declarada é risco de regra, não estratégia. Mas a *categoria* merece **uma auditoria de um dia**: foi a única coisa medida nesta noite que sozinha supera o modelo inteiro, e se houver um segundo canal desse tipo que seja **causal**, ele vale mais que sete semanas de modelagem. É também a hipótese que melhor explicaria um 0,65 no topo do leaderboard sem nenhum avanço metodológico.

---

## 6. Como decidir, e em que ordem gastar

O erro a não repetir é o do próprio `DIAGNOSTICO_ESTRUTURAL.md` §6, que ordenou o cardápio **por risco** (P1 mais seguro primeiro). Com um alvo de +0,03, ordenar por risco seleciona sistematicamente as opções incapazes de atingi-lo. **Ordene por teto, e proteja-se com portões baratos.**

**Portões, em ordem de custo, todos antes de qualquer construção completa:**

| # | portão | custo | por que primeiro |
|---|---|---|---|
| 0 | auditoria de estrutura do gerador (forma D) | ~1 dia | é o único achado que sozinho supera o modelo; se houver versão causal, muda tudo |
| 1 | **determinismo** da pilha candidata (1e-8 no re-run de 30%, em processos paralelos) | ~½ dia | uma solução não determinística vale **zero** independentemente da AUC — reprovar aqui na segunda-feira é infinitamente melhor que descobrir em 10/09 |
| 2 | **força solo** em subamostra (500–2000 séries): o objeto novo chega a ~0,60? | ~½ dia | é o critério que a §4 identifica como o único que importa |
| 3 | correlação intra-`t` com o incumbente | minutos | **não é portão de entrada** — decide apenas misturar vs. substituir |
| 4 | réplica na partição 43 | ~40 min | ruído de partição 0,007 > qualquer efeito; sem isso nada de ±0,005 está decidido |

**Regra de decisão registrada antes de executar:** reprovou o portão 1 → a pilha morre, sem apelação. Reprovou o portão 2 → **não integrar como membro auxiliar** (a §4 mostra que não paga) — arquivar e passar para a próxima. Passou os dois → medir ρ e decidir misturar/substituir com a tabela do §4 na mão.

**O que fica formalmente fechado por este relatório:**

- **micro-braços de ±0,003** — abaixo do ruído de partição, e agora com mecanismo (§3);
- **membros auxiliares sobre a base atual** — teto medido +0,0021 sobre 16 candidatos;
- **melhorias de `g`** — capacidade, objetivo, profundidade, regularização (E0c: IC inclui 0);
- **eixo de localização de τ** — oráculo perfeito vale +0,0014;
- **largura nova em `φ` que seja função da mesma observação** — sete famílias, rendimento ~0, com mecanismo.

**Uma nota sobre "consistência e robustez" (o caminho B do diagnóstico original).** Reduzir variância de partição, ancorar o board, blindar determinismo e latência é a jogada correta de quem está empatado na frente. Estando **0,03 atrás — 4× o ruído de partição —** é a decisão de **travar a posição atual**. Continua valendo como *seguro* (o bundle final precisa não quebrar), mas não como **plano**: seguro não é caminho.

---

## 7. Contra-tese — o que poderia invalidar este relatório

Escrita com a mesma seriedade, porque um documento que só confirma a si mesmo não vale a leitura.

1. **O Exp0 não mediu o teto da tarefa.** E0a/E0b são um **piso**, não um teto: o incumbente bate os dois por +0,035, então eles não limitam nada por cima. A tabela de decisão do `DIAGNOSTICO_ESTRUTURAL.md` §5 ("E0a ≤ 0,66 ⇒ teto real") tinha uma pré-condição **não declarada** — que o skyline dominasse o modelo — e ela falhou. **Nada aqui prova que a tarefa está esgotada; prova que esta construção está.** A diferença importa: significa que existe espaço, e que ele não é alcançável por onde estamos cavando.
2. **O E0c é uma semente e uma partição.** O IC inclui 0 no agregado, mas exclui em `t>400`. Uma réplica na partição 43 custa ~1h20 e endureceria (ou trincaria) a conclusão mais importante do documento. **Recomendo rodá-la antes de qualquer decisão irreversível de calendário.**
3. **O câmbio da §2.2 é medido sobre membros que são todos filhos da mesma base.** Extrapolar dele para um modelo genuinamente independente é extrapolação — e o deságio de 0,46 pode ser pessimista justamente por isso (irmãos compartilham erro estruturado além da correlação de postos).
4. **O skyline pode ser fraco por construção, não por falta de informação.** Ele é um teste omnibus de janela contra histórico, com 6 componentes; o incumbente acumula evidência sequencialmente sobre 183 canais. Que ele perca não prova muito sobre a tarefa — só sobre ele.

Nenhuma dessas ressalvas toca o argumento central do §3, que não depende do skyline: **`g` saturado + ensembling saturado + calibração saturada = a construção acabou.** As ressalvas todas apontam para a mesma direção prática — o pivô é necessário; o que elas questionam é o quanto de teto existe do outro lado.

---

## 8. Reprodução

```bash
# E0a/E0b — skyline (branqueado; ~5 min para 10k séries). --repr raw para o contraste.
python scripts/e0_skyline.py --repr whitened --out artifacts/models/oof_e0_skyline.parquet

# E0c — o toco (~1h20). As flags --max-depth/--num-leaves foram acrescentadas a scripts/train.py.
python scripts/train.py --rows data/processed/train_rows_3eixos.parquet \
  --drop-prefix spec_ ord_ mrep_ --detectability-mode soft --feature-contri-meta 0.6 \
  --boost-seed 777 --max-depth 1 --num-leaves 2 --n-estimators-cap 5000 \
  --early-stopping-rounds 300 --out artifacts/models/e0c_stump_s777

# E0d — fallback puro
python scripts/fallback_oof.py --rows data/processed/train_rows_3eixos.parquet \
  --out artifacts/models/oof_e0d_fallback.parquet

# Tabela consolidada (buckets de t + magnitude do censo A1)
python scripts/e0_report.py

# ICs
python scripts/compare_oof.py --baseline artifacts/models/oof_e0c_stump_s777.parquet \
  --candidate artifacts/models/oof_b6_contri06_s777.parquet --n-boot 300
```

**Artefatos:** `artifacts/models/oof_e0_skyline{,_raw}.parquet`, `oof_e0c_stump_s777.parquet`, `oof_e0d_fallback.parquet`, `oof_e0_blend{,_bag4}_w02.parquet` · **ICs:** `artifacts/reports/compare_e0c_stump.json`, `compare_e0_blend{,_bag4}.json` · **logs:** `artifacts/reports/e0{_skyline,_skyline_raw,c_stump}.log`.

**Scripts novos:** `scripts/e0_skyline.py`, `scripts/e0_report.py`. **Modificado:** `scripts/train.py` (`--max-depth`, `--num-leaves`).

---

*Registrado em 2026-07-24. Os dois controles do §2.4 (comprimento e placebo) entraram **antes** dos números finais, não depois — sem eles o E0a teria sido reportado como 0,6634 e este relatório teria a conclusão oposta.*
