# Detecção sequencial de quebras estruturais sob avaliação transversal: formulação, estatísticas e agregação

**Resumo.** Estudamos o problema de detectar, em tempo real e sem revisão retrospectiva, a ocorrência
de uma quebra estrutural num processo estocástico do qual se observa previamente uma realização livre
de quebra. O problema difere da detecção clássica de pontos de mudança em dois aspectos que
determinam toda a solução: a decisão é sequencial e irrevogável, e a avaliação é *transversal* — o
desempenho é medido comparando, num mesmo instante, os escores atribuídos a um conjunto de processos
independentes. Demonstramos que esse funcional de avaliação é invariante a transformações monótonas
comuns a todos os processos, e derivamos as consequências: componentes do escore que dependam apenas
do tempo são exatamente neutros, ao passo que normalizações idiossincráticas por processo não o são.
Sobre essa base, construímos uma redução ao regime nulo por filtragem autorregressiva congelada e
apresentamos a matemática de nove famílias de estatísticas de detecção, organizadas pelo tipo de
alternativa que cada uma detecta e pelo funcional (supremo ou integral) que aplicam ao caminho
amostral. Discutimos a agregação por modelo aditivo em log-odds, incluindo um resultado de fusão
exata que permite promediar réplicas de treino sem custo de inferência. Por fim, tratamos da
metodologia de comparação: decompomos a variância do estimador de desempenho em componente de
amostragem e componente de treino, mostramos que ignorar a segunda invalida o teste usual, e
formalizamos o viés de seleção que surge quando uma configuração é escolhida sob a mesma realização
do gerador aleatório usada para avaliá-la.

---

## 1. Introdução

A detecção de quebras estruturais tem duas tradições. A **retrospectiva** pergunta, dada uma série
completa, onde estão os pontos de mudança; admite formulações por minimização de custo penalizado e é
resolvida por programação dinâmica ou segmentação. A **sequencial**, iniciada por Page e Shiryaev,
pergunta quando parar e declarar mudança, com critérios de atraso médio de detecção sujeito a uma
taxa de alarme falso.

O problema aqui pertence à segunda tradição, mas com uma diferença que se revela decisiva: não se
pede uma regra de parada, e sim, a cada instante, um **escore de evidência**; e a qualidade desse
escore não é medida por atraso ou taxa de falso alarme, mas por sua capacidade de **ordenar** um
conjunto de processos independentes observados simultaneamente. Essa mudança de funcional de
avaliação não é cosmética: ela torna irrelevantes propriedades usualmente centrais — calibração
absoluta, por exemplo — e torna centrais outras que a teoria sequencial clássica não enfatiza, como a
comparabilidade de escala entre processos com dinâmicas nulas distintas.

Organizamos o texto assim. A §2 formaliza o problema e demonstra a propriedade de invariância do
funcional de avaliação, da qual decorrem as decisões estruturais subsequentes. A §3 apresenta a
redução ao regime nulo. A §4, o núcleo técnico, desenvolve a matemática das famílias de estatísticas.
A §5 trata da calibração idiossincrática. A §6, da agregação por aprendizado supervisionado e de um
resultado de fusão exata. A §7 desenvolve a metodologia de comparação. A §8 reporta resultados
empíricos, a §9 os discute, e a §10 conclui.

---

## 2. Formulação

### 2.1 Modelo probabilístico

Considere $N$ processos indexados por $i$. Para cada $i$ observa-se uma realização histórica
$\{x^{(i)}_s\}_{s=1}^{n_i}$, garantidamente gerada por um regime estacionário $P_0^{(i)}$, seguida de
um fluxo $\{x^{(i)}_t\}_{t\ge 1}$ no qual pode existir um instante de quebra $\tau_i \in
\mathbb{N}\cup\{\infty\}$ tal que

$$
x^{(i)}_t \sim
\begin{cases}
P_0^{(i)}, & t < \tau_i \\
P_1^{(i)}, & t \ge \tau_i
\end{cases}
$$

com $P_1^{(i)} \neq P_0^{(i)}$ desconhecida. Note que os regimes nulos $P_0^{(i)}$ **diferem entre
processos** — alguns exibem heterocedasticidade condicional, outros dependência linear persistente,
outros caudas pesadas. Esta heterogeneidade é essencial: ela implica que uma mesma estatística
assume, sob a hipótese nula, distribuições distintas conforme o processo, e que comparar seus valores
brutos entre processos é comparar quantidades não comensuráveis.

Seja $\mathcal{F}_t^{(i)}$ a filtração gerada pelo histórico e por $x^{(i)}_{1:t}$. Um **detector** é
uma família de funções mensuráveis $S_t : \mathcal{F}_t^{(i)} \to \mathbb{R}$. A restrição de
causalidade é estrita: $S_t$ não pode depender de $x^{(i)}_{s}$ para $s>t$, nem do comprimento total
do fluxo, nem de qualquer estatística agregada sobre os outros processos no mesmo instante.

### 2.2 O funcional de avaliação

Defina o rótulo $y^{(i)}_t = \mathbf{1}\{\tau_i \le t\}$. Para cada instante $t$, seja
$\mathcal{P}_t = \{i : y^{(i)}_t = 1\}$ e $\mathcal{N}_t = \{i : y^{(i)}_t = 0\}$, com cardinalidades
$n^+_t$ e $n^-_t$. A área sob a curva ROC da seção transversal no instante $t$ é

$$
A_t \;=\; \frac{1}{n^+_t n^-_t} \sum_{i \in \mathcal{P}_t} \sum_{j \in \mathcal{N}_t}
\left[ \mathbf{1}\{S_t^{(i)} > S_t^{(j)}\} + \tfrac12 \mathbf{1}\{S_t^{(i)} = S_t^{(j)}\} \right],
$$

e o funcional de avaliação é a média ponderada

$$
\mathcal{A} \;=\; \frac{\sum_t w_t A_t}{\sum_t w_t}, \qquad w_t = n^+_t n^-_t .
$$

A ponderação por $w_t$ é a que faz de $\mathcal{A}$ um estimador da probabilidade de concordância
agregada sobre todos os pares discordantes disponíveis em todos os instantes.

### 2.3 Invariância e suas duas consequências

**Proposição 1 (invariância a transformações comuns).** Seja $\varphi_t:\mathbb{R}\to\mathbb{R}$
estritamente crescente, e considere o detector transformado $\tilde S^{(i)}_t = \varphi_t(S^{(i)}_t)$,
com a *mesma* $\varphi_t$ para todo $i$. Então $\tilde A_t = A_t$ para todo $t$, e portanto
$\tilde{\mathcal{A}} = \mathcal{A}$.

*Demonstração.* $A_t$ depende de $\{S_t^{(i)}\}$ apenas através dos indicadores
$\mathbf{1}\{S_t^{(i)} > S_t^{(j)}\}$ e $\mathbf{1}\{S_t^{(i)} = S_t^{(j)}\}$. Monotonicidade
estrita de $\varphi_t$ preserva ambas as relações. $\square$

Duas consequências, de sinais opostos, governam o desenho de todo o sistema.

**Corolário 1.1 (neutralidade do componente comum).** Qualquer parcela do escore que dependa apenas
de $t$ — em particular a taxa-base $\pi(t) = \Pr(\tau \le t)$ marginalizada sobre processos — é
neutra para $\mathcal{A}$. Formalmente, se $S^{(i)}_t = g(\mathcal{F}^{(i)}_t) + h(t)$, então
$\mathcal{A}$ não depende de $h$.

Isto elimina de saída a calibração absoluta como objetivo: o escore não precisa ser uma probabilidade,
e esforço dedicado a torná-lo uma não pode, por construção, melhorar o desempenho medido.

**Corolário 1.2 (não-neutralidade do componente idiossincrático).** Se
$\tilde S^{(i)}_t = \varphi^{(i)}(S^{(i)}_t)$ com $\varphi^{(i)}$ dependente do processo, então em
geral $\tilde A_t \ne A_t$.

Este é o resultado que motiva a §5: como os regimes nulos diferem entre processos, uma padronização
que use a distribuição nula *de cada processo* altera a ordenação e pode, em princípio, melhorá-la.
A distinção entre os Corolários 1.1 e 1.2 é sutil e sua confusão é uma fonte concreta de erro: uma
transformação que *parece* idiossincrática mas é, na prática, quase constante entre processos —
porque a quantidade que a define tem variabilidade desprezível entre eles — recai no Corolário 1.1 e
é inócua.

**Observação (variância induzida pela estimação).** Suponha $\varphi^{(i)}(s) = (s -
\hat\mu_i)/\hat\sigma_i$ com $\hat\mu_i$ estimado a partir de $m$ observações do regime nulo do
processo $i$. O erro $\hat\mu_i - \mu_i$ é **independente entre processos**. No instante $t$, esse
erro entra na comparação transversal como ruído aditivo puro, de variância $\mathrm{Var}(\hat\mu)$,
não compartilhado. Portanto uma padronização idiossincrática só melhora a ordenação se a
heterogeneidade que ela remove exceder o ruído de estimação que ela injeta — um balanço que depende
de $m$ e que deve ser verificado, não assumido.

---

## 3. Redução ao regime nulo

### 3.1 Filtragem autorregressiva congelada

Ajusta-se sobre o histórico um modelo autorregressivo de ordem $p$,

$$
x_s = c + \sum_{k=1}^{p} \phi_k x_{s-k} + \varepsilon_s ,
$$

por mínimos quadrados, aceitando-o apenas se a redução relativa da variância residual exceder um
limiar. Opcionalmente inclui-se um termo sazonal num atraso $\ell$ selecionado pela autocorrelação
amostral. A inovação padronizada é

$$
e_s \;=\; \frac{\varepsilon_s}{\sigma_\varepsilon},
$$

truncada a um intervalo simétrico para limitar a influência de observações extremas isoladas.

O ponto essencial não é o ajuste, e sim o **congelamento**: os coeficientes estimados no histórico são
aplicados sem reestimação a todo o fluxo online. Isto tem duas consequências.

Primeiro, sob $H_0$ a sequência $\{e_t\}$ é aproximadamente ruído branco de variância unitária, e
qualquer estatística com distribuição nula conhecida sob ruído branco torna-se utilizável.

Segundo, e mais importante, $e_t$ **é** o erro de predição um-passo do filtro do histórico. Logo, se
o regime muda de modo que a estrutura de dependência de segunda ordem se altere, o filtro deixa de
branquear e a perda de brancura de $\{e_t\}$ é ela própria a evidência de quebra — sem que nada
precise ser reestimado. Reestimar o filtro online seria contraproducente: ele absorveria a quebra que
se quer detectar. Esta é a razão pela qual o objeto que caracteriza o regime nulo é construído uma
única vez e mantido imutável.

### 3.2 Normalização de volatilidade e o problema da absorção

Para processos com dependência na volatilidade, a inovação padronizada por uma escala fixa não é
homocedástica, o que distorce estatísticas que pressupõem variância constante. A correção natural é
uma normalização adaptativa,

$$
v_t = (1-\lambda) v_{t-1} + \lambda\, e_t^2, \qquad
\tilde e_t = \frac{e_t}{\sqrt{v_t}},
$$

que produz uma sequência aproximadamente homocedástica.

Essa correção, porém, tem um custo que precisa ser explicitado. Suponha que em $\tau$ a variância
salte de $1$ para $\rho^2 > 1$. Então $\mathbb{E}[v_t] \to \rho^2$ com constante de tempo
$1/\lambda$, e portanto

$$
\mathbb{E}\big[\tilde e_t^2\big] \;\longrightarrow\; 1
\qquad \text{para } t - \tau \gg 1/\lambda .
$$

A normalização **absorve** a quebra de variância: após algumas dezenas de passos, a sequência
normalizada é indistinguível do regime nulo. Uma estatística de variância calculada sobre
$\tilde e_t$ é, portanto, cega precisamente ao alternativo mais frequente.

Daí a regra de roteamento adotada: estatísticas cujo alvo é **escala ou cauda** consomem a inovação
de escala congelada; estatísticas cujo alvo é **localização, dependência ou forma** consomem a versão
normalizada. Trata-se de um trade-off explícito entre robustez a heterocedasticidade nula e
sensibilidade a heterocedasticidade induzida pela quebra, resolvido separadamente por família.

---

## 4. Famílias de estatísticas de detecção

Organizamos as famílias por dois eixos: a **alternativa** que cada uma privilegia (localização,
escala, dependência, forma de cauda, distribuição inteira) e o **funcional** que aplicam ao caminho
amostral da evidência acumulada — de tipo supremo ou de tipo integral, distinção desenvolvida em §4.8.

### 4.1 Razão de verossimilhança sequencial: a estatística de Page

Para detectar um deslocamento de magnitude $\delta$ na média de inovações $\mathcal{N}(0,1)$, o
incremento da log-verossimilhança sob a alternativa contra a nula é

$$
\log \frac{f_\delta(e_t)}{f_0(e_t)} \;=\; \delta e_t - \frac{\delta^2}{2}.
$$

A estatística de soma cumulativa refletida,

$$
G_t \;=\; \max\!\left(0,\; G_{t-1} + \delta e_t - \frac{\delta^2}{2}\right), \qquad G_0 = 0,
$$

é o teste sequencial de razão de verossimilhança generalizado com reinício. Sob $H_0$ o incremento
tem média $-\delta^2/2 < 0$, o que faz de $G_t$ uma cadeia positiva recorrente com distribuição
estacionária; sob a alternativa a média torna-se $+\delta^2/2 > 0$ e $G_t$ cresce linearmente. É
ótimo no sentido de Lorden para $\delta$ conhecido.

A construção se estende a outras alternativas substituindo a razão de verossimilhança apropriada:
para uma mudança de escala de fator $\rho$,

$$
\log\frac{f_\rho(e_t)}{f_1(e_t)} \;=\; -\log\rho + \frac{e_t^2}{2}\left(1 - \frac{1}{\rho^2}\right),
$$

e analogamente para proporção de sinal (Bernoulli), taxa de excedência de um quantil de referência, e
autocovariância defasada, esta última usando o produto $e_t e_{t-1}$ como escore.

Duas quantidades são extraídas de cada detector: o valor corrente e a **idade desde o último
reinício**, isto é, $t - \max\{s \le t: G_s = 0\}$. Sob $H_0$ os reinícios são frequentes e a idade
permanece pequena; sob a alternativa o detector deixa de reiniciar e a idade cresce linearmente,
fornecendo um estimador implícito de $\tau$.

### 4.2 Posterior sobre o comprimento de execução

Uma alternativa bayesiana mantém a distribuição a posteriori do **comprimento de execução** $r_t$,
definido como o número de observações desde a última mudança. Com prior de risco constante $H$
(equivalente a uma geométrica sobre o intervalo entre mudanças), a recursão é

$$
p(r_t, x_{1:t}) \;=\; \sum_{r_{t-1}} p(r_t \mid r_{t-1})\; p(x_t \mid r_{t-1}, x^{(r)})\; p(r_{t-1}, x_{1:t-1}),
$$

com $p(r_t = r_{t-1}+1 \mid r_{t-1}) = 1-H$ e $p(r_t = 0 \mid r_{t-1}) = H$. Adotando prior conjugado
normal–gama-inversa para média e variância do regime, a preditiva $p(x_t \mid r_{t-1}, x^{(r)})$ é
uma $t$ de Student, o que confere robustez a caudas pesadas.

Extraem-se: a razão de verossimilhança logarítmica entre "houve mudança recente" e "não houve", o
modo da posterior de $r_t$, e a idade implícita $t - r_t^{\text{MAP}}$. O custo por passo é linear no
truncamento do suporte de $r_t$.

### 4.3 Martingales conformais

Esta família dispensa qualquer suposição paramétrica. Dado o conjunto de referência do histórico,
define-se para cada observação online um $p$-valor conformal por posto médio,

$$
p_t \;=\; \frac{\#\{s : \alpha_s > \alpha_t\} + \tfrac12 \#\{s : \alpha_s = \alpha_t\} + \tfrac12}{n+1},
$$

onde $\alpha$ é uma medida de não-conformidade (valor com sinal, ou módulo, conforme a alternativa
visada). Sob permutabilidade — que vale sob $H_0$ — os $p_t$ são aproximadamente i.i.d. uniformes.

Um **martingale de apostas** é então

$$
M_T \;=\; \prod_{t=1}^{T} \epsilon\, p_t^{\,\epsilon-1}, \qquad \epsilon \in (0,1),
$$

e, como $\int_0^1 \epsilon u^{\epsilon-1}\,du = 1$, tem-se $\mathbb{E}[M_T \mid \mathcal{F}_{T-1}] =
M_{T-1}$ sob $H_0$: $M$ é martingale não-negativa de valor inicial 1. Pela desigualdade de Ville,

$$
\Pr\left(\sup_{T} M_T \ge 1/\alpha\right) \le \alpha,
$$

o que fornece controle de erro tipo I **uniforme no tempo**, sem correção para múltiplas
comparações — propriedade que nenhuma das famílias anteriores possui. Uma mistura sobre $\epsilon$
evita a escolha de um único parâmetro de aposta. Sob a alternativa, os $p_t$ concentram-se perto de
zero e $\log M_T$ cresce linearmente.

### 4.4 Embeddings de kernel

Para detectar mudanças na distribuição inteira, e não apenas em momentos escolhidos, usa-se a
discrepância média máxima. Para um kernel característico $k$ com espaço de Hilbert associado
$\mathcal{H}$,

$$
\mathrm{MMD}^2(P,Q) \;=\; \big\| \mu_P - \mu_Q \big\|_{\mathcal{H}}^2,
\qquad \mu_P = \mathbb{E}_{X\sim P}[k(X,\cdot)],
$$

que se anula se e somente se $P=Q$. O cálculo direto é quadrático no número de amostras, inviável
sequencialmente. Pelo teorema de Bochner, um kernel invariante a translação e positivo definido é a
transformada de Fourier de uma medida de probabilidade, o que autoriza a aproximação por
características aleatórias:

$$
k(x,y) \;\approx\; z(x)^\top z(y), \qquad
z(x) = \sqrt{\tfrac{2}{D}}\left[\cos(\omega_1 x + b_1), \dots, \cos(\omega_D x + b_D)\right]^\top,
$$

com $\omega_j$ amostradas da medida espectral e $b_j$ uniformes. A média empírica de $z$ pode então
ser mantida recursivamente, e a discrepância contra a referência do histórico torna-se uma norma
euclidiana em dimensão $D$, com custo constante por passo.

Duas variantes são úteis: a marginal, sobre $e_t$, e a conjunta, sobre o par $(e_t, e_{t-1})$, esta
última sensível a mudanças na estrutura de dependência que preservam a marginal. Uma terceira
construção compara duas médias exponenciais de velocidades distintas, dispensando referência fixa e
tornando-se um detector de mudança *relativa* ao passado recente.

### 4.5 Decomposição multiescala

O nível de energia sozinho não distingue três alternativas qualitativamente distintas: um patamar
persistente de variância, uma explosão transitória de volatilidade, e uma deriva lenta. A separação
requer decompor a energia em bandas de frequência disjuntas.

A transformada de Haar fornece a decomposição mais barata compatível com causalidade. Em cada escala
$j$, pares consecutivos $(a,b)$ produzem

$$
d^{(j)} = \frac{a-b}{\sqrt2}\quad\text{(detalhe)}, \qquad s^{(j)} = \frac{a+b}{\sqrt2}\quad\text{(aproximação)},
$$

com a aproximação alimentando a escala $j+1$. A transformada é ortonormal, de modo que sob ruído
branco de variância unitária $\mathbb{E}[(d^{(j)})^2] = 1$ em toda escala — as energias já nascem
comparáveis entre processos, sem normalização adicional. Mantendo uma média exponencial de
$(d^{(j)})^2$ por escala e formando contrastes entre escalas finas e grossas, obtém-se um descritor
do *formato* da curva energia-versus-escala, que é o que discrimina as três alternativas acima.

### 4.6 Momentos de ordem e forma de cauda

Momentos convencionais de ordem elevada são inúteis quando a distribuição tem caudas pesadas, pois
podem não existir. Os **L-momentos**, definidos como combinações lineares de esperanças de estatísticas
de ordem,

$$
\lambda_1 = \mathbb{E}[X], \qquad
\lambda_2 = \tfrac12\,\mathbb{E}[X_{2:2} - X_{1:2}], \qquad
\lambda_3 = \tfrac13\,\mathbb{E}[X_{3:3} - 2X_{2:3} + X_{1:3}],
$$

existem sempre que o primeiro momento existe. As razões adimensionais
$\tau_3 = \lambda_3/\lambda_2$ (L-assimetria) e $\tau_4 = \lambda_4/\lambda_2$ (L-curtose) são
estimadores de forma robustos, calculados em janelas móveis.

### 4.7 Variação bipotência e separação de saltos

Para distinguir um aumento de volatilidade contínua de uma sequência de saltos, usa-se o contraste
entre a variância realizada e a variação bipotência,

$$
RV_n = \sum_{i=1}^n e_i^2, \qquad
BV_n = \frac{\pi}{2}\sum_{i=2}^n |e_i|\,|e_{i-1}| .
$$

Sob um processo de difusão sem saltos, ambas convergem para a variação quadrática integrada e a razão
$RV/BV \to 1$; na presença de saltos, $RV$ incorpora os quadrados dos saltos enquanto $BV$ permanece
robusto a eles, e a razão excede 1. A construção se completa com semivariâncias — as somas restritas
a incrementos de cada sinal — cuja assimetria detecta mudanças na direcionalidade, e com a correlação
entre sinal e magnitude subsequente, que capta efeito de alavancagem.

### 4.8 Funcionais de tipo supremo e de tipo integral

Todas as famílias anteriores extraem do caminho amostral da evidência acumulada um funcional do tipo
**supremo**: a soma cumulativa guarda seu máximo refletido, a posterior bayesiana seu modo, o
martingale é dominado por seu pico. A teoria clássica de testes de estabilidade distingue essa classe
de uma segunda, de tipo **integral**, com perfil de potência diferente.

Considere a ponte construída a partir das somas parciais centradas de uma janela de $n$ observações
$u_1,\dots,u_n$:

$$
B_i \;=\; \sum_{j\le i} \left(u_j - \bar u\right)
\;=\; P_i - \frac{i}{n} P_n, \qquad P_i = \sum_{j\le i} u_j .
$$

A estatística de tipo integral é

$$
\eta_n \;=\; \frac{1}{n^2 \hat\sigma^2} \sum_{i=1}^{n} B_i^2 ,
$$

que sob a hipótese de estabilidade converge em distribuição para $\int_0^1 \mathbb{B}^\circ(r)^2\,dr$,
onde $\mathbb{B}^\circ$ é a ponte browniana padrão. Como
$\mathbb{E}[\mathbb{B}^\circ(r)^2] = r(1-r)$, segue

$$
\mathbb{E}\left[\int_0^1 \mathbb{B}^\circ(r)^2 dr\right] = \int_0^1 r(1-r)\,dr = \frac{1}{6},
$$

um valor **universal**: independente da distribuição de $u$, de sua variância e de suas caudas. A
divisão por $\hat\sigma^2$ torna a estatística auto-normalizada, e a construção da ponte a torna
invariante a deslocamentos — de modo que a comparabilidade entre processos é obtida por construção,
sem estimar nenhuma constante idiossincrática, evitando o ruído descrito na Observação da §2.3.

A distinção de potência é a seguinte. Um funcional de supremo é sensível a um desvio grande e
localizado; um de integral, ao desvio acumulado ao longo da janela. Segue que o segundo domina quando
a alternativa é difusa — deriva gradual, múltiplas quebras pequenas — e, notavelmente, quando a
quebra ocorre **próximo ao fim da janela**: nesse regime o supremo ainda não acumulou evidência, mas a
ponte inteira já se deslocou.

A mesma ponte pode ser aplicada a diferentes **representações** do fluxo, cada uma sensível a um tipo
de alternativa: a inovação (localização), seu quadrado (escala), e sua transformação integral de
probabilidade contra a distribuição do histórico (a distribuição inteira, de forma livre de
distribuição e robusta a caudas).

Uma observação de completude: a representação por **diferenças primeiras** é degenerada nesta
construção. Se $u_j = e_j - e_{j-1}$, a soma parcial telescopa, $P_i = e_i - e_0$, a ponte torna-se
função dos valores brutos e $\eta_n$ colapsa numa variância de janela — quantidade já contemplada por
outras famílias. A verificação é algébrica e dispensa experimento.

### 4.9 Entropias espectral e ordinal

Duas construções merecem menção por serem **invariantes a escala por definição**, e portanto
estruturalmente incapazes de duplicar a informação das famílias de variância.

*Entropia espectral.* Estimando a densidade espectral de potência em $K$ bandas por transformada de
Fourier de tempo curto com janela exponencial, e formando as proporções $\hat p_k = \hat P_k / \sum_j
\hat P_j$, define-se o centroide $\sum_k \hat p_k \omega_k$ e a entropia $-\sum_k \hat p_k \log \hat
p_k$. Ambas são razões, logo invariantes a multiplicação do processo por constante. Sob ruído branco o
espectro é chato: entropia máxima e centroide central. Persistência positiva desloca massa para
baixas frequências; alternância, para altas — e o centroide, ao contrário de somas de quadrados de
autocorrelações, **tem sinal**, distinguindo as duas direções.

Um cuidado é essencial. O periodograma num único instante é assintoticamente exponencial: seu desvio
padrão iguala sua média, *qualquer que seja o comprimento da série*. As proporções $\hat p$ tornam-se
então aproximadamente $\mathrm{Dirichlet}(1,\dots,1)$, para a qual

$$
\mathbb{E}\left[-\sum_{k=1}^K p_k \log p_k\right] = \psi(K+1) - \psi(2),
$$

com $\psi$ a função digama. Para $K=6$ isto vale $\approx 1{,}45$, ou $0{,}81$ após normalização por
$\log K$ — bem abaixo do máximo teórico, e com dispersão entre realizações comparável ao efeito que se
deseja medir. A correção necessária é promediar periodogramas (estimador de Welch), o que reduz a
variância sem viés apreciável. Trata-se de um caso geral: **uma estatística construída como razão de
estimativas ruidosas não converge com o comprimento da série**; apenas promediação explícita a
estabiliza.

*Entropia de permutação.* Para cada bloco de $m$ observações consecutivas, registra-se apenas o padrão
de ordenação — um dos $m!$ símbolos. A entropia normalizada da distribuição empírica desses símbolos
numa janela,

$$
H_m \;=\; -\frac{1}{\log m!}\sum_{\pi} \hat p(\pi) \log \hat p(\pi),
$$

é máxima sob independência e diminui sob qualquer dependência serial, sem escolher atraso nem supor
linearidade. Sua propriedade distintiva é a invariância a **qualquer transformação monótona
ponto-a-ponto**: escala, cauda e normalização são invisíveis. Uma variante mede **irreversibilidade
temporal**, $\tfrac12\sum_\pi |\hat p(\pi) - \hat p(\pi^{R})|$ com $\pi^R$ o padrão do bloco lido em
ordem inversa; sob qualquer processo reversível no tempo — o que inclui todo processo linear
gaussiano — essa soma tem esperança nula, de modo que ela isola não-linearidade com direção temporal.

---

## 5. Calibração contra o regime nulo idiossincrático

Pelo Corolário 1.2, padronizar cada estatística pela sua distribuição nula *no processo em questão*
pode alterar — e potencialmente melhorar — a ordenação transversal. Para uma estatística de janela
$S$, estimam-se $\hat\mu_i$ e $\hat\sigma_i$ aplicando o mesmo estimador ao histórico do processo $i$,
que é nulo por construção, e define-se

$$
S^{\text{cal}} \;=\; \frac{S - \hat\mu_i}{\hat\sigma_i}.
$$

Duas ressalvas teóricas delimitam a aplicabilidade.

**(i) Estatísticas recursivas exigem tratamento distinto.** Para uma estatística de janela, o
histórico fornece muitas observações aproximadamente independentes do valor nulo. Para uma
estatística recursiva com reinício — como as somas cumulativas de §4.1 — a distribuição nula depende
de $t$ durante um regime transitório antes de atingir a estacionariedade, e uma passagem contínua
pelo histórico estima a distribuição estacionária, não a transitória. A estimação correta requer
réplicas com reinício, o que reduz drasticamente o número efetivo de amostras independentes.

**(ii) O balanço ruído-versus-heterogeneidade.** Pela Observação da §2.3, a padronização injeta ruído
transversal de variância $\mathrm{Var}(\hat\mu_i) + S^2\,\mathrm{Var}(\hat\sigma_i)/\sigma_i^2$,
independente entre processos. Ela só compensa se a heterogeneidade removida for maior. Em particular,
se a estatística crua já for aproximadamente comensurável entre processos — como ocorre, por
construção, para as energias de Haar de §4.5 e para os funcionais integrais de §4.8 —, a calibração
adiciona ruído sem remover viés, e o efeito líquido é negativo. Este é o cenário em que uma coluna
calibrada e sua versão crua exibem correlação transversal próxima de um: o resíduo não carrega
ordenação nova, apenas erro de estimação.

---

## 6. Agregação

### 6.1 Remoção do componente comum

O rótulo $y_t$ tem taxa-base $\pi(t)$ fortemente crescente. Pelo Corolário 1.1, esse componente é
neutro para o funcional de avaliação; contudo, ele domina qualquer perda pontual usada como
substituto de treino — a log-perda binária, por exemplo, é minimizada em primeira ordem por acertar
$\pi(t)$, e não por ordenar corretamente dentro do instante.

A solução é decompor o preditor como

$$
S_t^{(i)} \;=\; \sigma\!\left( g\big(\mathbf{z}^{(i)}_t\big) + \mathrm{logit}\,\hat\pi(t) \right),
$$

fixando o segundo termo como deslocamento conhecido e ajustando apenas $g$. O modelo aprende então o
**resíduo transversal** — exatamente a quantidade que o funcional mede — em vez de gastar capacidade
numa componente que o funcional ignora.

Na emissão do escore, o deslocamento pode ser omitido sem prejuízo, pelo Corolário 1.1; a escolha entre
incluí-lo ou não é indiferente para a avaliação e decidível por outros critérios, como o comportamento
fora do suporte de treino.

### 6.2 Modelo aditivo e validação por agrupamento

O agregador $g$ é um modelo aditivo em log-odds ajustado por *boosting* de gradiente sobre árvores,

$$
g(\mathbf{z}) \;=\; \sum_{b=1}^{B} f_b(\mathbf{z}), \qquad f_b \text{ árvore de regressão},
$$

escolha motivada por três propriedades: invariância a transformações monótonas de cada coordenada —
o que dispensa normalização das features e é coerente com a Proposição 1 —, tratamento nativo de
valores ausentes, que é essencial porque estatísticas de janela longa são genuinamente indefinidas
nos primeiros instantes, e capacidade de representar interações, necessária porque a interpretação de
um detector depende do regime nulo do processo (§2.1), o que é uma interação entre features
dinâmicas e descritores estáticos.

A validação cruzada deve agrupar por processo: todas as linhas de um mesmo processo pertencem à mesma
partição. Sem isso, o modelo memoriza processos individuais e a estimativa fora da amostra é
otimista de forma severa, já que linhas do mesmo processo em instantes vizinhos são fortemente
dependentes.

### 6.3 Agregação sobre réplicas de treino e fusão exata

O procedimento de ajuste é aleatório: a amostragem de colunas e de linhas em cada iteração de
*boosting* faz de $g$ uma variável aleatória mesmo com dados fixos. Seja $g^{(1)},\dots,g^{(K)}$ um
conjunto de réplicas independentes. O preditor médio tem variância reduzida por fator $1/K$ na
componente idiossincrática de treino, e portanto desempenho esperado superior ao de uma réplica
única — o argumento clássico de agregação bootstrap.

O obstáculo é o custo: a inferência é sequencial, uma observação por vez, e avaliar $K$ modelos
multiplica por $K$ um custo dominado por sobrecarga fixa de chamada, não por travessia de árvores.

Há, contudo, um resultado exato que elimina o obstáculo. Como cada $g^{(m)}$ é uma **soma** de
árvores, tem-se

$$
\frac{1}{K}\sum_{m=1}^{K} g^{(m)}(\mathbf{z})
\;=\; \frac{1}{K}\sum_{m=1}^{K}\sum_{b} f^{(m)}_b(\mathbf{z})
\;=\; \sum_{m,b} \frac{1}{K} f^{(m)}_b(\mathbf{z}),
$$

isto é, a média das $K$ réplicas é *ela própria* um modelo aditivo, obtido concatenando todas as
árvores e escalando cada valor de folha por $1/K$. A agregação passa a custar uma única avaliação.

Uma sutileza: a identidade vale no espaço de log-odds. A média das probabilidades,
$\frac1K\sum_m \sigma(g^{(m)})$, difere de $\sigma\!\left(\frac1K\sum_m g^{(m)}\right)$ por
desigualdade de Jensen, e as duas podem, em princípio, ordenar processos de modo distinto. A escolha
entre elas é uma questão empírica; a diferença é de segunda ordem quando os $g^{(m)}$ são próximos.

---

## 7. Metodologia de comparação

Esta seção trata de um problema de inferência estatística sobre o próprio procedimento experimental,
independente do domínio de aplicação.

### 7.1 Decomposição da variância do estimador

Seja $\hat{\mathcal{A}}(\theta, \xi)$ a estimativa do funcional de avaliação para uma configuração
$\theta$ (conjunto de estatísticas, hiperparâmetros), onde $\xi$ denota a realização do gerador
aleatório do procedimento de ajuste. A quantidade de interesse ao comparar $\theta_1$ e $\theta_0$ é

$$
\Delta = \mathbb{E}_\xi\!\left[\hat{\mathcal{A}}(\theta_1,\xi)\right] - \mathbb{E}_\xi\!\left[\hat{\mathcal{A}}(\theta_0,\xi)\right].
$$

O estimador usual toma uma realização de cada lado, $\hat\Delta = \hat{\mathcal{A}}(\theta_1,\xi_1) -
\hat{\mathcal{A}}(\theta_0,\xi_0)$, e constrói um intervalo de confiança por **bootstrap pareado sobre
processos**: reamostram-se índices $i$ com reposição e recomputa-se $\hat{\mathcal{A}}$ para ambos os
lados sobre a *mesma* reamostragem.

Esse procedimento estima corretamente a componente de variância devida à amostragem de processos, e o
pareamento cancela a maior parte da covariância entre os lados. Mas ele **condiciona em $\xi_1$ e
$\xi_0$**: trata os preditores como funções fixas. A variância total do estimador é

$$
\mathrm{Var}(\hat\Delta) \;=\; \underbrace{\mathrm{Var}_{\text{amostra}}(\hat\Delta)}_{\text{capturada pelo bootstrap}}
\;+\; \underbrace{\mathrm{Var}_\xi\!\left[\hat{\mathcal{A}}(\theta_1,\xi)\right] + \mathrm{Var}_\xi\!\left[\hat{\mathcal{A}}(\theta_0,\xi)\right]}_{\text{invisível ao bootstrap}},
$$

onde o segundo termo não é cancelado pelo pareamento porque $\xi_1$ e $\xi_0$ são realizações
independentes — e necessariamente distintas, já que alterar a configuração altera o próprio espaço
sobre o qual o gerador atua.

Se $\mathrm{Var}_\xi$ for da ordem dos efeitos procurados, o intervalo de confiança do bootstrap é
severamente anticonservador, e a taxa de erro tipo I do procedimento de decisão é muito maior que a
nominal. A correção é direta: estimar $\mathbb{E}_\xi$ por média sobre $K$ réplicas de cada lado,
reduzindo o segundo termo por fator $1/K$, e reportar as duas famílias como distribuições.

### 7.2 Viés de seleção sobre a realização do gerador

Um segundo efeito, mais insidioso, surge quando a configuração é escolhida por comparações conduzidas
sob uma **realização fixa** $\xi^\star$. Seja $\theta^\star = \arg\max_{\theta \in \Theta}
\hat{\mathcal{A}}(\theta, \xi^\star)$ o resultado de uma sequência de decisões desse tipo. Então

$$
\mathbb{E}\left[\hat{\mathcal{A}}(\theta^\star, \xi^\star)\right] \;>\; \mathbb{E}_\xi\left[\hat{\mathcal{A}}(\theta^\star, \xi)\right],
$$

isto é, o valor observado sob a realização de seleção é otimisticamente enviesado — a maldição do
vencedor aplicada não a uma escolha isolada, mas acumulada ao longo de todas as decisões do processo
de desenvolvimento.

O efeito tem uma assinatura empírica verificável e específica: a realização $\xi^\star$ deve ser
excepcionalmente favorável para $\theta^\star$ e **neutra para configurações vizinhas**, já que a
vantagem decorre de uma interação entre a configuração selecionada e aquela realização particular, e
não de uma qualidade intrínseca. Observar essa assinatura distingue o viés de seleção de uma flutuação
aleatória de cauda.

A consequência prática é grave: se o baseline de comparação é $\theta^\star$ avaliado em $\xi^\star$,
toda comparação subsequente é enviesada **contra** qualquer alternativa, e a taxa de falsos negativos
do procedimento de decisão é elevada por um mecanismo sistemático, não aleatório.

### 7.3 Protocolo corrigido

Decorre do exposto: (i) estimar cada lado por média sobre $K$ réplicas, com o mesmo $K$; (ii) excluir
das comparações a realização usada historicamente para seleção; (iii) declarar antes da medição o
subconjunto do domínio em que se espera o efeito, sob pena de a multiplicidade implícita inflar o erro
tipo I; (iv) precedendo qualquer ciclo caro, aplicar um teste de necessidade — uma estatística
candidata cuja correlação com uma já existente, *dentro do instante*, seja próxima de um não pode,
pela Proposição 1, alterar a ordenação, e portanto não pode mover o funcional, independentemente de
sua motivação teórica.

O ponto (iv) merece ênfase: trata-se de condição **necessária e não suficiente**. Uma direção nova no
espaço de estatísticas pode ser ortogonal ao banco existente e simultaneamente ortogonal ao sinal.

---

## 8. Resultados empíricos

Os experimentos usam $10^4$ processos com histórico e fluxo online, avaliados fora da amostra por
validação cruzada agrupada por processo. Todos os contrastes seguem o protocolo da §7.3, com $K$
réplicas por lado e exclusão da realização de seleção.

### 8.1 Estrutura do desempenho ao longo do tempo

O funcional decompõe-se por regime temporal com pesos $w_t$ muito desiguais:

| regime | peso relativo em $\mathcal{A}$ | desempenho |
|---|---|---|
| $t \le 50$ | 8,1% | 0,536 |
| $50 < t \le 150$ | 26,6% | 0,580 |
| $150 < t \le 400$ | 48,7% | 0,624 |
| $t > 400$ | 16,5% | 0,653 |

A concentração do peso no regime intermediário implica que melhorias restritas ao regime inicial —
onde o desempenho é mais fraco e a tentação de intervir é maior — são, por construção, quase
indetectáveis no agregado.

### 8.2 Magnitude da componente de variância de treino

Sete réplicas independentes de uma mesma configuração, com dados, estatísticas e partições idênticos,
diferindo apenas na realização do gerador do procedimento de ajuste:

$$
0{,}6006 \quad 0{,}6015 \quad 0{,}6016 \quad 0{,}6020 \quad 0{,}6022 \quad 0{,}6030 \quad 0{,}6032
$$

com média $0{,}6020$ e desvio-padrão $0{,}0009$. O erro-padrão da diferença entre duas configurações
avaliadas com **uma réplica cada** é portanto $\approx 0{,}0013$; incluindo a realização usada
historicamente para seleção, o desvio sobe para $0{,}0041$ e o erro-padrão para $0{,}0058$.

Três contrastes anteriormente julgados negativos, reavaliados contra esse referencial:

| contraste | $\hat\Delta$ | em unidades de erro-padrão | conclusão correta |
|---|---|---|---|
| calibração de estatísticas recursivas | $-0{,}0069$ | 1,2 | inconclusivo |
| detector bayesiano com poda | $-0{,}0042$ | 0,7 | inconclusivo |
| brancura multi-atraso | $-0{,}0024$ | 0,4 | inconclusivo |

### 8.3 Assinatura do viés de seleção

A realização historicamente usada produz, para a configuração selecionada sob ela, desempenho nove
desvios-padrão acima da média das demais realizações. O contraste com configurações vizinhas confirma
a assinatura prevista em §7.2:

| configuração | sob a realização de seleção | média das demais | excesso |
|---|---|---|---|
| a selecionada | 0,6100 | 0,6019 | $+0{,}0081$ |
| com família espectral | 0,6035 | 0,6039 | $-0{,}0004$ |
| com família ordinal | 0,6030 | 0,6031 | $-0{,}0001$ |
| com família integral | 0,6048 | 0,6080 | $-0{,}0032$ |

A vantagem é específica da configuração selecionada e não transfere para nenhuma modificação, o que é
incompatível com flutuação aleatória e consistente com interação seleção-realização.

### 8.4 Contribuição das famílias

Contra o referencial reavaliado de $0{,}6020$, com $K$ réplicas por lado:

| família acrescentada | $\hat\Delta$ | conclusão |
|---|---|---|
| funcionais de tipo integral (§4.8) | $+0{,}0036$ | adotada |
| detector bayesiano de run-length com remoção de estatísticas de baixa contribuição | $+0{,}0056$ | positiva, não combinável |
| — apenas a remoção | $+0{,}0027$ | componente |
| — apenas o detector | $+0{,}0029$ | componente |
| entropia espectral (§4.9) | $+0{,}0021$ | abaixo do critério |
| entropia de permutação (§4.9) | $+0{,}0012$ | abaixo do critério |
| brancura multi-atraso | $+0{,}0004$ | abaixo do critério |
| agregação sobre 4 réplicas de treino (§6.3) | $+0{,}0040$ | adotada |

Para a família adotada, o intervalo de confiança pareado exclui zero a favor no agregado
($+0{,}0042$, IC 95% $[+0{,}0022,\,+0{,}0063]$) e no regime declarado a priori. O maior efeito
relativo ocorre no regime inicial, coerente com a predição teórica de §4.8: é ali que a quebra está
tipicamente próxima ao fim da janela de observação.

### 8.5 Não-aditividade

A união das duas famílias com efeito positivo isolado mede $0{,}6054$, **abaixo de cada uma
separadamente**. Os detectores capturam informação parcialmente sobreposta; a união acrescenta
dimensão sem acrescentar poder de ordenação.

### 8.6 Saturação da agregação sobre réplicas

| $K$ | desempenho | ganho marginal |
|---|---|---|
| 1 | 0,6020 | — |
| 2 | 0,6047 | $+0{,}0026$ |
| 4 | 0,6060 | $+0{,}0005$ |
| 7 | 0,6068 | $+0{,}0002$ |

O comportamento é o esperado para redução de variância por promediação: o ganho decresce como a
componente idiossincrática remanescente, e satura quando o erro residual passa a ser comum às
réplicas.

---

## 9. Discussão

**Sobre o funcional de avaliação.** A Proposição 1 e seus corolários não são observações auxiliares:
elas determinam quais problemas vale a pena resolver. Calibração absoluta, monotonicidade temporal do
escore e suavização são todos exercícios neutros. Em contrapartida, a comparabilidade *entre
processos* torna-se o problema central, e é notável que ele admita duas soluções qualitativamente
distintas — padronização estimada (§5), que injeta ruído idiossincrático, e auto-normalização por
construção (§4.8), que não. A evidência empírica favorece a segunda.

**Sobre o valor de uma hipótese de falha.** Três famílias introduzidas nesta rodada eram igualmente
inéditas em relação ao banco existente, verificadas pelo teste de necessidade da §7.3. Apenas uma
produziu efeito. A que produziu não era a mais sofisticada, mas a única acompanhada de uma hipótese
sobre *em que regime* o conjunto existente falha — a saber, que funcionais de supremo perdem potência
quando a quebra está próxima ao fim da janela. As outras duas eram justificadas por ortogonalidade,
que é condição necessária e vazia isoladamente.

**Sobre a remoção de estatísticas.** A observação de que remover doze estatísticas de baixa
contribuição *aumenta* o desempenho merece atenção teórica. Num modelo aditivo com amostragem de
colunas, cada dimensão sem sinal reduz a probabilidade de que dimensões informativas sejam
consideradas numa dada divisão, um efeito de diluição. A magnitude observada sugere que o conjunto
está além do ponto ótimo dessa troca.

**Sobre a metodologia.** O resultado de §7.1–7.2 transcende esta aplicação. Sempre que um
procedimento de ajuste é aleatório e comparações são feitas com uma realização por lado, o intervalo
de confiança usual é anticonservador de forma não quantificada. Sempre que um referencial é o produto
de uma sequência de seleções sob uma realização fixa, comparações contra ele são enviesadas de forma
sistemática. Ambos os efeitos são baratos de diagnosticar — basta reajustar o referencial contra si
mesmo — e, quando presentes, invalidam retroativamente decisões tomadas com o instrumento defeituoso.

---

## 10. Conclusão

Formalizamos a detecção sequencial de quebras estruturais sob um funcional de avaliação transversal,
demonstrando que a invariância desse funcional a transformações monótonas comuns elimina uma classe
inteira de objetivos e eleva outra à centralidade. Sobre uma redução ao regime nulo por filtragem
congelada, organizamos nove famílias de estatísticas segundo a alternativa visada e o funcional
aplicado ao caminho amostral, destacando a distinção entre funcionais de supremo e de integral como
eixo pouco explorado e empiricamente produtivo. Apresentamos um resultado de fusão exata que torna
gratuita, em custo de inferência, a agregação sobre réplicas de ajuste.

A contribuição de maior alcance, contudo, é metodológica: a demonstração de que o procedimento de
comparação usualmente adotado ignora a componente de variância devida ao próprio ajuste e é enviesado
por seleção acumulada sobre uma realização fixa do gerador aleatório. Ambos os efeitos foram
quantificados e revelaram-se da ordem de grandeza dos efeitos que se pretendia medir, o que invalidou
três decisões anteriores e obrigou à revisão do protocolo experimental.

As direções abertas seguem dessa revisão. A remoção sistemática de estatísticas de baixa contribuição
tem justificativa teórica e evidência preliminar favorável. A não-aditividade documentada em §8.5
sugere que o conjunto de detectores está próximo de saturação informacional, e que ganhos adicionais
exigirão não novas estatísticas sobre as mesmas representações, mas hipóteses explícitas sobre
regimes de falha ainda não cobertos — dos quais o mais persistente é a quebra puramente de dependência
sem alteração de escala.
