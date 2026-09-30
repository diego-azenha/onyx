# Que informação a série ainda esconde

**Detecção de quebra estrutural — Real-Time Edition 2026**
Análise das lacunas de informação sobre as 78 features / 28 grupos.

---

## Seção 0 — O filtro que precede tudo

Antes de responder "que features faltam", uma pergunta desconfortável: **o modelo erra por falta de informação, ou por outra coisa?**

Quando um detector erra, existem exatamente três causas, e só uma delas é resolvida com features:

1. **Lacuna de informação** — o sinal existe na série, mas nenhuma feature o projeta. A quebra é *ortogonal* à base que suas 78 features geram. Só esta causa é feature-solucionável.
2. **Falha de calibração/objetivo** — o sinal *está* nas features, mas o objetivo de treino (logloss dominado pela base rate) não o transforma em ranking cross-seccional útil. Adicionar features aqui não ajuda — pode piorar, injetando variância num objetivo já mal-condicionado.
3. **Indetectabilidade intrínseca** — a quebra é pequena demais para o número de observações disponíveis. Nenhum algoritmo, por mais features que tenha, separa melhor que o acaso. Perseguir esses casos com features novas é *adicionar variância sem sinal*.

Isto importa pra você especificamente porque a TS-AUC é **cross-seccional por passo**: o que compete não é a magnitude absoluta do seu score, é o *ranking* dele contra as outras séries no mesmo passo `t`. Uma feature que carrega sinal real mas é ruidosa piora seu ranking mesmo estando "correta". E porque seu problema ativo declarado é a base rate dominando o objetivo — parte das suas misses quase certamente é causa (2), não (1).

**Postura deste documento:** trato cada lacuna abaixo perguntando "isto é sinal *rankeável* que ainda não está na base, ou é ruído com cara de sinal?". Só o primeiro tipo entra. E o critério de valor não é "captura mais informação" — é "melhora a ordenação cross-seccional relativa a um controle pareado" (que é, aliás, seu único gate local confiável).

---

## 1. Propriedades ainda não capturadas

### O enquadramento: suas features geram um subespaço de "tipos de mudança"

Média, variância, autocorrelação e entropia são features de **momentos** e de **dependência linear de segunda ordem**. Elas detectam uma quebra quando a quebra *projeta* sobre esse subespaço. É por isso que "às vezes sim, às vezes não":

- Um *mean shift* projeta fortemente sobre features de média → detectado.
- Uma mudança *só na estrutura de dependência* (mesma marginal, autocorrelação diferente) projeta fracamente sobre features marginais → miss.
- Uma mudança *só na cauda* (centro idêntico, extremos diferentes) projeta ~zero sobre momentos centrais → miss.

**A miss acontece quando a quebra vive fora do subespaço que suas features geram.** Não é falha de quantidade (78 é bastante); é falha de *direção*.

### As direções que provavelmente faltam

**Forma distribucional além dos momentos.** Duas distribuições com média, variância e autocorrelação idênticas podem ser radicalmente diferentes em assimetria, curtose, multimodalidade. O instrumento certo não é "mais um momento" — é comparar as **funções de distribuição empíricas** diretamente: distância de Wasserstein, energy distance ou Kolmogorov–Smirnov entre a janela histórica e a janela online. Isso captura mudança de *forma* que nenhum momento isolado vê, e é naturalmente normalizável (útil pra TS-AUC).

**Reorganização espectral a variância constante.** Uma quebra pode mover energia de baixa pra alta frequência sem mexer na variância total. A série "parece igual" em variância, mas o espectro girou. Centroide espectral, entropia espectral, razão de potência entre bandas. Uma mudança nos coeficientes AR muda a *forma* do espectro mesmo com variância fixa — e isso conecta direto com sua pergunta de alta vs. baixa frequência (ver abaixo).

**Dependência não-linear e de cauda.** Autocorrelação vê *só* dependência linear. Uma série pode ser branca (ACF ≈ 0) e fortemente dependente de outra forma (volatility clustering tipo ARCH). Isto merece seção própria — Seção 5.

**Estrutura ordinal.** Padrões de ordem (permutation entropy, Bandt–Pompe) capturam a sequência de subidas/descidas, invariante a transformações monótonas. Uma quebra pode mudar o *padrão de ups e downs* sem mudar a distribuição dos valores. Barato e robusto.

### Por que aparece só em contextos específicos

Sua intuição está certa: certas propriedades só emergem sob regime específico, e a razão é **mascaramento por normalização**.

- **Sob tendência forte:** um mean shift fica escondido dentro da própria deriva da tendência. Você precisa das features calculadas sobre a série *destendenciada / diferenciada*, senão a tendência engole o sinal.
- **Sob alta frequência de amostragem:** dominam quebras na dependência de curto alcance (estrutura AR/volatilidade). Sob baixa frequência, dominam quebras de nível/tendência.

A consequência prática é forte: **não basta escolher features, é preciso escolher representações.** Calcular seu núcleo de detectores sobre três representações — bruta, diferenciada e rank-transformada (ou em espaço de retornos) — cobre um leque de força-de-tendência e captura mudanças invariantes a monotonia. Muitas das suas misses de contexto são features boas aplicadas na representação errada.

---

## 2. A história da série — a trajetória, não o estado final

Este é, na minha leitura, o ponto de maior alavancagem que você tem, justamente porque ataca seu problema declarado: *validação local absoluta é uma armadilha; você precisa de sinal relativo.*

### Você está jogando fora a ordem

O histórico é z-scored e livre de quebra. O procedimento padrão calcula features sobre o histórico *inteiro como uma amostra*. Isso descarta a informação de *como o histórico chegou até ali*.

Duas coisas moram nessa trajetória:

**A volatilidade interna das próprias features (o null personalizado).** Fatie o histórico em janelas deslizantes e meça *quanto as features flutuam dentro do próprio histórico livre de quebra*. Uma série cujas features já oscilam bastante no histórico tem uma "linha de base de surpresa" alta — uma flutuação online que assustaria um detector ingênuo é normal *pra ela*. Isto é uma análise de change-point aplicada ao histórico não pra achar quebra (não há), mas pra **estimar a distribuição nula das flutuações de feature daquela série específica**.

O payoff é exatamente o que você precisa: se você normaliza cada desvio online pela escala de flutuação do próprio histórico da série, seus scores viram quantidades tipo *quantil-dentro-da-própria-série*. E quantidades assim são **cross-seccionalmente comparáveis por construção** — que é a moeda da TS-AUC. Isso explica *por que* seu AUC local absoluto sempre foi otimista demais: ele mede separabilidade dentro de uma escala arbitrária por série, não a ordenação relativa que a plataforma avalia. Um score baseado em quantil-do-null-próprio é intrinsecamente mais rankeável.

**A dinâmica de aproximação (B→A ≠ C→A).** Duas séries podem terminar com estatísticas de janela-final idênticas mas ter chegado ali de formas diferentes: uma convergiu monotonicamente pra baixa volatilidade, outra oscilou até estabilizar. O segmento online é uma *continuação* — uma série que *vinha convergindo* se comporta diferente pós-histórico de uma que *vinha divergindo*. O que distingue isso é a **velocidade e aceleração do vetor de features ao longo do histórico**: o histórico está com trajetória de feature convergente, divergente ou cíclica? A derivada das features no *final* do histórico é um preditor da dinâmica online inicial — e é exatamente o que resolve o caso "quebra no passo 1", onde você não tem observações online mas tem a inclinação terminal do histórico.

---

## 3. Sinais de instabilidade — o que procurar antes da quebra

### O que a teoria diz que existe (critical slowing down)

Há uma literatura sólida de *early warning signals* pra transições críticas. Quando um sistema se aproxima de um ponto de bifurcação, aparecem assinaturas previsíveis:

- **Autocorrelação lag-1 crescente** — o sistema recupera mais devagar de perturbações ("lentidão crítica").
- **Variância crescente** — as flutuações incham.
- **Assimetria crescente** — o sistema passa mais tempo perto do estado alternativo.
- **Flickering** — excursões breves ao novo regime antes de comprometer-se.

Como capturar "o histórico começou a desestabilizar": estimativas *rolling* de AC(1) e de variância sobre a *porção final* do histórico, e — o que importa — as **inclinações** dessas estimativas. Inclinação positiva de AC(1) e de variância no fim do histórico é a assinatura de desestabilização. Sobre seus resíduos pós-branqueamento AR(10): variância residual crescente ou autocorrelação residual crescente no fim do histórico dizem que a estrutura estava começando a ceder.

### A ressalva honesta — e ela é importante

Esses precursores valem para transições **tipo bifurcação** (sistemas suaves se aproximando de um tipping point). Muitas quebras deste desafio são provavelmente **exógenas/abruptas**: um parâmetro salta sem qualquer aviso. Para essas, *não existe* precursor — a quebra é uma surpresa por definição.

Você já sabe que "nem toda quebra tem precursor". A formalização útil é: **features de precursor ajudam num subconjunto de quebras e são ruído puro no resto.** Injetadas cegamente, elas melhoram as bifurcações e *pioram* as abruptas (adicionam variância onde não há sinal). Isto é caso (1) e caso (3) do filtro convivendo. A forma de não se machucar é deixar o modelo *condicionar*: as features de precursor devem coexistir com um indicador do tipo de dinâmica (a série é suave/persistente ou branca?), pra que a árvore possa usá-las só onde fazem sentido. Não trate precursor como sinal universal.

---

## 4. A dinâmica da mudança online — a sequência de scores carrega informação que o score isolado não tem

Sim, e isto é subexplorado. A trajetória do estatístico ao longo dos passos online contém informação que o valor instantâneo não tem.

### Velocidade ≠ magnitude — são dois eixos, ambos importam

- **Magnitude** = quão diferente é o novo regime.
- **Velocidade** = quão rápido acontece a transição (degrau abrupto vs. rampa gradual).

Uma quebra de *pequena magnitude e rápida* e uma de *grande magnitude e lenta* podem produzir a mesma evidência instantânea num dado passo, mas **trajetórias de score completamente diferentes**. Se você só olha o valor instantâneo, colapsa esses dois eixos num só e perde metade da informação.

A implicação de design é que você precisa de **dois detectores complementares**:

- **Instantâneo (tipo Shewhart):** dispara em saltos grandes. Detecta magnitude imediata.
- **Integrador (tipo CUSUM / Page-Hinkley):** acumula desvios pequenos e *persistentes* na mesma direção. Detecta *deriva* mesmo quando a magnitude por passo é minúscula. Sensível à velocidade integrada, não à magnitude pontual.

Um pega o que o outro deixa passar. Só o instantâneo perde derivas lentas; só o integrador demora em saltos.

### Sustentado vs. transitório — o coração do seu problema de falso-positivo

Uma quebra estrutural produz evidência **sustentada**: a rampa do estatístico sobe e *fica*. Um outlier transitório (um draw raro de uma distribuição de cauda pesada, mas estacionária) produz um **spike que reverte**. A forma da rampa distingue os dois — e distinguir sustentado de transitório é *exatamente* o problema de você classificar como quebra o que não é.

Features da trajetória do estatístico que capturam isso: inclinação, curvatura, monotonicidade, **tempo-desde-a-primeira-excedência**, **persistência da excedência** (quantos passos consecutivos acima do limiar), área sob o estatístico. Um spike tem tempo-de-persistência curto; uma quebra tem persistência crescente. Essa família ataca o falso-positivo mais diretamente que qualquer feature marginal nova.

### Compatibilidade com suas restrições

Isto é barato. CUSUM, EWMA e Page-Hinkley são todos recursivos **O(1)** — encaixam perfeitamente no seu orçamento de estado por passo. As features de trajetória do estatístico são mantidas incrementalmente. Não há tensão com a causalidade estrita nem com O(1)/O(log n).

---

## 5. Estruturas de dependência que a autocorrelação não vê

Autocorrelação é *uma* forma de dependência — a linear, de segunda ordem, no bulk. Existem pelo menos quatro tipos que ela é cega:

- **Dependência de volatilidade (efeitos ARCH):** autocorrelação de `|x|` ou de `x²`. Uma série pode ser não-correlacionada em níveis e fortemente dependente em volatilidade. Uma quebra pode ser *puramente* na estrutura de dependência da volatilidade — invisível à ACF dos níveis.
- **Dependência de cauda / extremal:** os valores grandes se agrupam? Extremograma, runs de excedências acima de um limiar. A ACF é dominada pelo bulk e cega a co-ocorrência de extremos.
- **Dependência não-linear:** informação mútua em defasagens, ou o *gap* entre previsibilidade linear (R² de um ajuste AR) e não-linear. Se a série é imprevisível linearmente mas previsível não-linearmente, a ACF é zero e a estrutura existe.
- **Dependência assimétrica / que varia no tempo:** subidas e descidas com persistência diferente; autocorrelação que só se manifesta num sub-período. ACF sobre a janela inteira faz média sobre o tempo e apaga uma quebra que só existe num pedaço.

### "Autocorrelação alta mas ainda assim quebra" — como distinguir

Aqui está o ponto fino, e ele valida e estende sua arquitetura. Sob autocorrelação alta, o *tamanho de amostra efetivo* é pequeno, e detectores ingênuos **super-disparam** — confundem persistência com quebra. A saída é não testar o caminho realizado, e sim o **mecanismo gerador**.

Uma quebra estrutural é uma mudança nos *parâmetros geradores* (os coeficientes AR, a variância da inovação, a distribuição da inovação) — **não** na suavidade do caminho realizado. Uma série é suave porque é persistente (sem quebra); o que caracteriza a quebra é o *mecanismo* que gera a suavidade ter mudado.

Por isso seu branqueamento AR(10) está certo — mas provavelmente subexplorado. Monitore as **inovações**, não o caminho:

1. Variância das inovações.
2. Autocorrelação dos *quadrados* das inovações (dependência de volatilidade residual).
3. A *distribuição* das inovações (forma, não só variância — ver Seção 6).

E o detector mais direto de todos: **congele o filtro do histórico e veja se ele ainda ajusta online.** Ajuste o AR(10) no histórico, congele os coeficientes, aplique no online, monitore o erro de predição um-passo-à-frente. Se online é o mesmo regime, os resíduos ficam brancos e com a mesma variância; se houve quebra, os resíduos inflam ou correlacionam. Isto testa *diretamente* a hipótese "o modelo do histórico ainda vale", é O(10) = O(1), e é uma direção quase certamente ausente da sua base atual porque você provavelmente branqueia pra *calcular* features, não pra *testar mismatch de modelo* como sinal por si só.

---

## 6. Ambiguidade inerente — o que é detectável em princípio e o que não é

### A teoria: detectabilidade é razão sinal-ruído

A capacidade de distinguir "quebra" de "não-quebra" num dado passo é governada pela **distância estatística** (divergência KL, ou razão de verossimilhança) entre as distribuições pré e pós-quebra, *relativa a* quantas observações pós-quebra você tem e ao nível de ruído.

- **Quebra óbvia:** divergência grande entre pré/pós **e** amostras pós-quebra suficientes. A razão de verossimilhança cresce rápido; a evidência acumula; detectável com poucas amostras.
- **Quebra ambígua:** divergência pequena (regimes parecidos) **ou** pouquíssimas amostras pós-quebra (quebra perto do fim, ou no passo 1). A verossimilhança quase não se move; a evidência fica soterrada no ruído.

### A resposta direta à sua pergunta: sim, algumas são intrinsecamente indetectáveis

Se o regime pós-quebra difere do pré por uma quantidade *menor que o ruído amostral dado o número de amostras disponíveis*, **nenhum algoritmo separa melhor que o acaso.** Isso é um piso informacional, não uma falha de modelagem. Reconhecê-lo importa porque perseguir esses casos com mais features *adiciona variância sem sinal* — e sob TS-AUC cross-seccional, variância espúria machuca sua calibração relativa.

Dois corolários úteis:

**A quebra no passo 1 com mudança pequena é quase indetectável *naquele passo*.** Uma observação não distingue duas distribuições próximas. A detectabilidade *cresce* com os passos online. É por isso que a ponderação da TS-AUC por `n_pos(t)·n_neg(t)` faz sentido: passos iniciais são inerentemente mais difíceis, e a métrica já contabiliza a dificuldade cross-seccional. Você não vence o limite informacional — você só evita *deixar sinal detectável na mesa*. E no passo 1, o único sinal disponível não vem do online (você tem 1 obs), vem da *inclinação terminal do histórico* (Seção 2/3).

**Cauda pesada confunde-se com quebra de nível.** Um único draw grande de um processo de cauda pesada mas *estacionário* parece uma quebra de nível por um tempo. Distinguir "draw raro da mesma distribuição" de "draw de uma distribuição nova" exige evidência de **persistência** — que exige tempo. Ou seja: **parte da ambiguidade se resolve esperando**, o que troca contra a recompensa de detecção precoce da TS-AUC. Essa tensão é real e você deveria decidi-la explicitamente, não por acidente do modelo.

### O que fazer com isso operacionalmente

Você *pode estimar* a detectabilidade por série: a divergência janela-online-vs-histórico combinada com o tamanho de amostra disponível te dá uma medida de "quão detectável é este caso". Um detector bem-calibrado deve emitir score perto da base rate quando o caso é intrinsecamente ambíguo, e score confiante quando a divergência é grande. Sob TS-AUC, o que ranqueia é a *divergência-por-ruído* — deixe **ela** dirigir o score, não o ruído das features. Uma feature de "detectabilidade estimada" pode servir de *gate* pro modelo: onde ela é baixa, o modelo aprende a não se arriscar.

---

## 7. Lacunas específicas — o que está lá e ninguém está olhando

Priorizado por (valor esperado de ranking cross-seccional) × (baixo custo / compatibilidade com suas restrições). Cada item mapeado à sua arquitetura.

**Tier 1 — provavelmente o maior ganho, baixo custo, ataca seu problema declarado**

1. **Erro de predição do filtro congelado.** AR(10) do histórico, congelado, aplicado online; monitore crescimento do erro um-passo. Testa "o modelo do histórico ainda vale" diretamente. O(1). Quase certamente ausente hoje porque você branqueia pra extrair features, não pra medir mismatch como sinal. *(Seção 5)*
2. **Null personalizado por série + normalização.** Flutuação das features dentro do histórico deslizante → normalize todo desvio online por essa escala. Torna scores quantil-dentro-da-série, logo cross-seccionalmente comparáveis. Ataca diretamente a razão de o AUC local ser otimista demais. *(Seção 2)*
3. **Features de trajetória do estatístico online.** Tempo-desde-primeira-excedência, persistência da excedência, inclinação/curvatura da rampa. Separa spike transitório de quebra sustentada → ataca falso-positivo. O(1). *(Seção 4)*

**Tier 2 — direções ausentes da base, custo moderado**

4. **Distância distribucional histórico-vs-janela-online** (Wasserstein / energy / KS) em vez de diferença de momentos. Captura mudança de *forma*; naturalmente normalizável. *(Seção 1)*
5. **Forma da distribuição das inovações pós-branqueamento** (assimetria, curtose, quantis, ECDF-distance residual-histórico vs residual-online) — não só variância residual. *(Seção 5/1)*
6. **Detector integrador explícito** (CUSUM / Page-Hinkley) sobre média, variância e resíduo, *lado a lado* com o instantâneo. Captura deriva lenta que o Shewhart perde. O(1). *(Seção 4)*
7. **Dependência de volatilidade:** ACF dos quadrados dos resíduos, tipo ARCH-LM. Pega quebras puras em volatility clustering. *(Seção 5)*

**Tier 3 — cobertura de direções ortogonais, usar com condicionamento**

8. **Deslocamento espectral** (centroide/entropia espectral, razão de bandas) histórico vs online — pega reorganização de frequência a variância constante. Custo em janela limitada; O(janela). *(Seção 1)*
9. **Entropia de permutação** e sua mudança — ordinal, robusta a monotonia, barata. *(Seção 1)*
10. **Features de precursor** (inclinação de AC(1) e variância no fim do histórico) — **condicionadas** a um indicador de tipo-de-dinâmica, senão viram ruído nas quebras abruptas. *(Seção 3)*
11. **Multi-representação:** rode o núcleo de detectores sobre série bruta, diferenciada e rank-transformada. Resolve misses de contexto (força-de-tendência) sem features novas. *(Seção 1)*
12. **Detectabilidade estimada** (divergência × tamanho-de-amostra) como feature-gate — o modelo aprende a recuar onde o caso é intrinsecamente ambíguo. *(Seção 6)*

---

## Uma nota final, no espírito de discordar quando é o caso

Você chegou com a pergunta "que informação falta extrair". Boa parte deste documento responde isso. Mas o filtro da Seção 0 não é enfeite: **se suas misses forem majoritariamente causa (2) — base rate dominando o objetivo — nenhuma feature acima resolve.** Features melhoram a *separabilidade disponível*; elas não consertam um objetivo que não converte separabilidade em ranking cross-seccional.

Meu palpite, dado seu diagnóstico v1, é que o ganho está *repartido*: os itens Tier 1 (especialmente #1 e #2) provavelmente ajudam de verdade porque produzem sinal *rankeável e cross-seccionalmente comparável* — que é precisamente o que um objetivo mal-calibrado consegue aproveitar melhor do que features de magnitude bruta. Ou seja, as features certas aqui são as que *aliviam* o problema de calibração, não as que ignoram ele. Eu priorizaria #1 e #2 antes de qualquer expansão de largura da base, mediria contra controle pareado (seu único gate local confiável), e só então decidiria se o resto do Tier 2/3 vale a variância que adiciona.
