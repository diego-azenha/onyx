# Proposta de pesquisa: novas famílias de features (V2)

**Data:** 2026-07-20
**Contexto:** após R0–R6 (`docs/RESULTADOS_ROADMAP_R0_R6.md`), mudanças de objetivo/peso/parada/
hiperparâmetro se mostraram estatisticamente nulas. A hipótese de trabalho passa a ser: **o gargalo
está no que o modelo consome, não em como ele consome.**
**Status:** proposta — nada implementado. Objetivo é decidir o que construir antes de gastar ciclos
de retreino (~9 min dataset + ~15 min treino + ~20-60 min robustez por braço).

---

## 1. A hipótese se sustenta? Evidência dos dois lados

**A favor (forte):**

1. Três intervenções estruturais independentes (pesos R1, parada R2, objetivo de ranking R3) deram
   Δ estatisticamente indistinguível de zero. O modelo não está sendo mal treinado.
2. O banco atual é, quase inteiramente, **momentos de baixa ordem + detectores sequenciais de
   limiar**. Nenhuma feature do banco calcula uma *distância distribucional* entre a janela recente e
   o histórico. As soluções vencedoras da edição batch 2025 usaram exatamente isso — divergências
   Jensen-Shannon, Hellinger e Wasserstein, diferenciais de entropia, testes de Fligner — famílias
   que o `onyx` **não possui**.
3. O censo A1 (real, 4552 séries) diz que o sinal vive em variância/cauda/forma (41,8% das séries
   com |Δlogvar_e|>0,3; média praticamente morta, 6,8%). Estatísticas de forma/divergência medem
   exatamente isso; momentos medem isso apenas de forma grosseira.

**Contra / cautela (deve ser levado a sério):**

1. **R4 já foi uma injeção de "família nova" e não rendeu nada.** O bloco rank-two-sample
   (Wilcoxon/dispersão/chi²-forma) entrou hoje no dataset e o modelo resultante ficou plano. Este é
   o dado mais relevante contra a versão ingênua da hipótese: "família nova" ≠ "ganho".
2. O envelope de potência (R6) mostrou que a AUC observada já está **muito acima** do que um detector
   de shift-de-média entregaria em todos os buckets — o modelo já extrai sinal de variância/forma.
3. Em t≤50 a mediana de pontos pós-quebra é **14**. Existe um piso de informação real ali que
   nenhuma feature remove.

**Conclusão:** a hipótese é plausível e há lacunas concretas e documentadas — mas o critério de
seleção não pode ser "o que os vencedores de 2025 usaram". Tem de ser **"o que é diferente em
espécie do que já existe"**, porque R4 provou que mais um funcional do mesmo contraste
janela-vs-histórico não move o ponteiro.

---

## 2. O achado que reorienta a prioridade: comparabilidade é o gargalo, não detecção

> **Correção (2026-07-20, após medição própria).** A primeira versão desta seção citava "34,3%" a
> partir de `artifacts/reports/shap_feature_importance.csv`, um arquivo pré-existente **sem script
> que o gerasse** e que descrevia o modelo *pré-auditoria*. Os números abaixo vêm de
> `scripts/shap_report.py` (TreeSHAP exato), escrito para esta análise e versionado. A conclusão
> qualitativa sobreviveu; o número mudou e a *medida* mudou — ver o parágrafo seguinte, que é o
> ponto metodológico mais importante desta seção.

**A medida convencional de SHAP engana nesta competição.** `mean|SHAP|` mistura variação *entre
passos* com variação *entre séries*. Mas a TS-AUC compara séries **dentro do mesmo passo**: pela
invariância C1, o que é constante dentro de um passo é *exatamente neutro* para a métrica. Portanto
a medida certa é a dispersão da contribuição **dentro do passo**, agregada com o peso oficial
w_t = n_pos·n_neg — abaixo, coluna `XS`.

Decomposição do modelo atual (91 features, pós-R4; `artifacts/reports/shap_preV2.csv`):

| Família | **XS% (move a TS-AUC)** | conv% (`mean\|SHAP\|`) |
|---|---|---|
| **`meta_h0_*` (constante por série)** | **30,5%** | 17,8% |
| `accum` | 21,3% | 11,7% |
| `conformal` | 15,0% | **36,6%** |
| `cusum` | 14,2% | 6,6% |
| `bayes` | 6,7% | 5,7% |
| `meta_tempo`/locator | 4,6% | **16,9%** |
| `ranktwo` (R4) | 4,2% | 2,4% |
| `hedge` | 3,5% | 2,3% |

As duas medidas discordam de forma dramática e sistemática, exatamente como a teoria previa:
`meta_ln1p_t` cai de 11,9% para 2,7% e `meta_t` de 4,5% para 1,1% (são constantes dentro do passo —
o resíduo não-nulo é a parte de *interação*, que de fato varia entre séries); `conformal` cai de
36,6% para 15,0% (os log-martingales crescem com t, e essa variação temporal inflava a medida
convencional). Qualquer decisão tomada com a coluna convencional teria sido enviesada para features
que apenas acompanham o relógio.

**Sob a medida correta, `meta_h0_*` é a maior família do modelo: 30,5%.** E sabemos por CE6 que essas
features **não carregam efeito principal** (classificador só-histórico: AUC 0,5067). Logo, quase um
terço da capacidade de ordenação transversal do modelo é gasto **aprendendo a calibrar**: "dada uma
série com esta cara, um CUSUM de variância deste tamanho é (ou não é) surpreendente".

**E a carga de calibração é maior exatamente onde o modelo é pior** (XS% de `meta_h0_*` por bucket,
contra a TS-AUC OOF do mesmo bucket):

| bucket | `meta_h0_*` XS% | TS-AUC |
|---|---|---|
| t≤50 | **40,1%** | 0,522 |
| 50<t≤150 | 35,4% | 0,567 |
| 150<t≤400 | 29,3% | 0,614 |
| t>400 | 27,0% | 0,640 |

A leitura é direta: com pouca informação na janela, o estatístico bruto é ruidoso e o modelo se apoia
ainda mais em "que tipo de série é esta". É precisamente o regime onde calibrar o estatístico na
origem deveria render mais.

**Evidência independente da premissa** (medida diretamente, não inferida do SHAP): a razão entre a
escala nula de uma série GARCH e de uma i.i.d. é de 2,0–2,35× nas estatísticas cruas e ~1,0 depois
da calibração F1 — ver `tests/unit/test_calibration.py::test_calibration_equalizes_null_scale_across_series`.

| estatística | cru (GARCH/i.i.d.) | calibrado |
|---|---|---|
| `ranktwo_dispersion_z_w100` | 1,95 | 1,14 |
| `accum_window_var_ln_w100` | 2,33 | 0,97 |
| `mmd_marginal_slow` | 2,35 | 0,86 |

Isso reformula o problema. A TS-AUC compara séries *diferentes* no mesmo passo. Uma estatística cujo
nível sob H0 depende das idiossincrasias da série (curtose, dependência, persistência de vol) é
**intrinsecamente mal-ordenada na seção transversal**, e o modelo tem de gastar árvores reconstruindo
a correção via interações. É provavelmente também **por que R4 falhou**: um z de Wilcoxon assume uma
escala nula que está errada para séries heterocedásticas/dependentes, então o ganho bruto foi
consumido pela recalibração.

> **Reorientação:** antes de adicionar mais detectores, tornar as estatísticas que já existem
> *auto-calibradas*. Isso ataca 34% do orçamento do modelo, custa **zero latência por passo**, e
> multiplica o valor de qualquer família futura.

---

## 3. A assimetria batch × tempo-real (por que não copiar a lista de 2025 direto)

Os vencedores de 2025 calculavam JS/Hellinger/Wasserstein **sobre o segmento pós-quebra inteiro**
(centenas de pontos). No nosso regime:

| Bucket | mediana de pontos pós-quebra | viabilidade de uma divergência de 16 bins |
|---|---|---|
| t≤50 | 14 | inviável (14 pontos, 16 bins → ruído puro) |
| 50<t≤150 | 45 | marginal |
| 150<t≤400 | 133 | viável |
| t>400 | 303 | confortável — **e é onde já somos melhores (0,64)** |

Portar a lista de 2025 literalmente tende a ajudar **só onde já vamos bem** e a não fazer nada onde
somos fracos. Features de alta dimensão efetiva (histogramas, divergências multi-bin) são
estruturalmente inadequadas para t pequeno. Onde t é pequeno o que paga é estatística de **alta
potência por observação** (baixa dimensão, bem normalizada) — o que reforça de novo §2.

---

## 4. Famílias propostas (ordenadas por leverage/custo)

Custos medidos em micro-benchmark real nesta máquina. Orçamento: gate atual 1500 µs/passo, consumo
atual medido **682 µs/passo** → folga ~800 µs sob o gate (e ~4700 µs sob o orçamento real de §11.4).

---

### F1 — Auto-calibração por nulo de histórico partido *(prioridade máxima)*

**Mecanismo.** No `fit_h0`, deslizar a MESMA estatística S (var_ln de janela, Wilcoxon z, MMD, o que
for) sobre o próprio histórico — que é H0 por definição — e guardar a distribuição nula de S *para
aquela série* (média, desvio, quantis). Online, além de S_t, emitir **o percentil / z de S_t contra
o nulo da própria série**.

**Por que é o item de maior leverage.** Converte uma interação cara que o modelo hoje aprende com
~600 splits (`meta_h0_*` × estatística) numa feature diretamente comparável entre séries. Uma série
com cauda pesada e uma gaussiana passam a emitir "0,99 do meu próprio nulo" com o mesmo significado
— que é exatamente o que a TS-AUC premia.

**Custo.** O(n_h) uma vez por série no `fit_h0` (que já é O(n_h log n_h)). **Zero µs/passo.**
**Features.** Não adiciona features novas necessariamente — *transforma* as existentes (ou duplica as
~10 mais importantes na versão calibrada). Sugestão inicial: calibrar as 8-10 de maior SHAP.
**Previsão falsificável.** (a) share de SHAP de `meta_h0_*` cai substancialmente (o modelo não
precisa mais reconstruir a calibração); (b) ganho concentrado nos buckets ≥150 (onde há janela
suficiente para a estatística se estabilizar); (c) se o share de `meta_h0_*` NÃO cair, a hipótese de
§2 está errada e o resto da proposta perde força.
**Risco.** Nulo estimado com poucas janelas independentes para w grande (n_h/w ≈ 4-20) → ruidoso;
mitigar com janelas sobrepostas + encolhimento para o nulo paramétrico. Histórico com drift lento
infla o nulo (conservador, não perigoso).

---

### F2 — Impressão digital de H0 mais rica *(prioridade alta, complementa F1)*

**Mecanismo.** Ampliar a caracterização por série calculada uma vez no `fit_h0`: expoente de Hurst /
memória longa, persistência tipo GARCH (α+β via proxy de ACF de e²), índice de cauda (Hill), perfil
de decaimento da ACF (não só ρ₁), inclinação espectral, Ljung-Box em |e|, força de sazonalidade,
vol-of-vol do histórico.

**Por quê.** As `meta_h0_*` já são a família mais usada do modelo (34,3%) — o modelo está *faminto*
por contexto de calibração. Dar-lhe um retrato melhor da série é barato e alinhado com o uso
comprovado.
**Custo.** Zero µs/passo (uma vez por série). **Features:** ~8.
**Previsão falsificável.** Ganho concentrado onde o falso-positivo é caro (T6/GARCH); as novas
`meta_h0_*` devem aparecer com padrão de *split* alto e *gain* por split baixo (assinatura de
interação, como as atuais). Se aparecerem como efeito principal forte, desconfiar de vazamento.
**Risco.** Baixo. CE6 deve continuar ≈0,5 (checagem obrigatória a cada iteração).

---

### F3 — MMD de kernel via Random Fourier Features (NEWMA / Online RFF-MMD) *(prioridade alta)*

**Mecanismo.** z(e) = √(2/D)·cos(W·e + b), D≈64, com W,b sorteados **uma vez com seed fixa e
compartilhados por todas as séries** (obrigatório para comparabilidade transversal). Referência
`h_ref` = média de z(e) sobre o histórico, congelada no `fit_h0`. Online, duas EWMAs de z(e_t) com
fatores de esquecimento diferentes (λ_rápido, λ_lento). Features: ‖m_rápida − h_ref‖²,
‖m_lenta − h_ref‖², e o estatístico NEWMA ‖m_rápida − m_lenta‖².

**Por que é diferente em espécie.** Com kernel característico, MMD² captura **todas** as diferenças
distribucionais (média, variância, assimetria, cauda) num único número — não é mais um momento nem
mais um funcional do contraste de CDFs empíricas (que é o que R4 já fez). E a extensão para o par
**(e_t, e_{t−1})** dá um MMD sobre a distribuição *conjunta*, isto é, um detector de mudança de
**dependência** que nenhuma feature atual cobre de forma não-paramétrica.
**Custo medido.** D=64 → **9,1 µs/passo**; D=128 → 10,7 µs/passo. Memória O(D).
**Features.** ~6 (3 escalares × {marginal, conjunta}).
**Previsão falsificável.** Ganho em buckets ≥150; a versão conjunta deve ajudar especificamente no
cenário T7 (quebra de dependência) da suíte de robustez. Se MMD marginal não superar
`accum_window_var_ln_w250` em SHAP, a família não está agregando informação nova.
**Risco.** Escolha de largura de banda σ (usar heurística da mediana sobre o histórico, fixa por
série — mas então perde comparabilidade; alternativa: σ=1 global já que e é padronizado → preferir
esta). Combinar com F1 para normalizar.

---

### F4 — Decomposição de energia multi-escala (Haar) *(prioridade média-alta, alvo específico)*

**Mecanismo.** Cascata diádica causal: em cada escala j, acumular pares → coeficiente de detalhe
d_j = (a−b)/√2; manter EWMA de d_j² por escala (J≈5). Features: ln-energia por escala e, o que
importa de verdade, **contrastes entre escalas** (ln E_fina − ln E_grossa).

**Por quê.** Um patamar novo e persistente de variância eleva a energia em **todas** as escalas
aproximadamente igual; um *burst* GARCH eleva desproporcionalmente as escalas finas; um drift lento,
as grossas. O *formato* da curva energia-vs-escala é o discriminador — e isso ataca frontalmente o
confundimento CE2×T6 que o parecer classificou como "irredutível no detector" (§4.3) e o item
estacionado "contrastes multi-escala anti-GARCH". As janelas atuais (`w10..w250`) são *suavizações*
do mesmo nível, não uma decomposição de escala — não é a mesma informação.
**Custo medido.** J=5 → **4,1 µs/passo** (o mais barato da lista).
**Features.** ~7 (5 energias + 2 contrastes).
**Previsão falsificável.** Melhora mensurável em T6/T9/T13 na suíte relativa (R5) *sem* perder gap em
T5/T5b (quebras reais de variância). Se T6 não melhorar, a hipótese do discriminador de escala está
errada.
**Risco.** Baixo. Escalas grossas demoram 2^J passos para "encher" → NaN em t pequeno (o pipeline já
trata NaN nativamente).

---

### F5 — Padrões ordinais / entropia de permutação *(prioridade média, eixo ortogonal)*

**Mecanismo.** Para m=3 (6 padrões) ou m=4 (24), histograma de padrões ordinais numa janela vs.
histograma do histórico (congelado no `fit_h0`). Features: entropia de permutação da janela,
divergência KL/JS entre janela e histórico, assimetria sobe/desce.

**Por que é ortogonal.** Padrões ordinais dependem **apenas da ordem** de valores consecutivos —
invariantes sob *qualquer* transformação monótona. Isso significa duas coisas: (i) são
completamente insensíveis a mudança de escala/variância, portanto **não competem** com o que já
domina o banco; (ii) são automaticamente comparáveis entre séries, sem calibração.
**Custo medido.** m=3, W=100 → **21,9 µs/passo**; m=4 → 23,7 µs/passo.
**Features.** ~5.
**Honestidade sobre o rendimento esperado.** Como são cegos a variância — que é onde está 41,8% do
sinal — **não espero que detectem mais quebras verdadeiras**. O valor provável está em **reduzir
falsos positivos**: clusters GARCH e excursões transitórias têm assinatura ordinal distinta de uma
quebra estrutural. Ganho de AUC via denominador, não numerador.
**Risco.** Pode render zero. É barato e verdadeiramente ortogonal, mas é o candidato com maior chance
de nulo entre F1-F5. Priorizar abaixo de F1-F4.

---

### F6 — Painel de divergências JS / Hellinger / Wasserstein-1 / KS *(prioridade baixa — ver ressalva)*

**Mecanismo.** Bins de quantis do histórico fixados no `fit_h0` (K=16); contagens rolantes por
janela; calcular JS, Hellinger, W1 (= Σ|F_jan − F_hist| sobre bins), KS (= sup|·|) e entropia.

**Por que está em último apesar de ser o que venceu em 2025.** Todas essas quantidades são
**funcionais do mesmo objeto** — o contraste entre a CDF empírica da janela e a do histórico — que é
precisamente o que o chi²-de-forma e o Wilcoxon de R4 já computam. A expectativa honesta é
correlação alta com R4 e ganho marginal pequeno. Somado à §3 (inviável para t pequeno), o custo de
oportunidade é ruim.
**Custo medido.** K=16, W=100 → **46,6 µs/passo** (o mais caro da lista, por 5 escalares).
**Quando reabrir.** Se o diagnóstico de §5 mostrar que as features de R4 **têm** SHAP relevante (isto
é, o eixo informa, só não bastou), então trocar chi² por JS/Hellinger (bem-comportados com contagens
pequenas, limitados em [0,1]) passa a fazer sentido como *substituição*, não adição.

---

## 5. Diagnóstico obrigatório antes de implementar — **EXECUTADO**

Critério pré-registrado (SHAP transversal das 6 features `ranktwo_*` de R4):

| Desfecho | Interpretação | Consequência |
|---|---|---|
| SHAP ≈ 0 | o eixo two-sample rank-based não informa aqui | **matar F6**; priorizar F3/F4 |
| SHAP relevante, AUC plana | informam, mas *substituem* features existentes | reforça §2 → **F1 mais prioritário** |
| SHAP relevante e concentrado em buckets tardios | confirma a assimetria de §3 | F6 só para t alto |

**Resultado medido** (`artifacts/reports/shap_preV2.log`, XS%, família total 4,24%):

| feature | XS% | rank (de 91) |
|---|---|---|
| `ranktwo_dispersion_z_w100` | 3,57% | **10** |
| `ranktwo_dispersion_z_w025` | 0,31% | 44 |
| `ranktwo_shape_chi2_w100` | 0,24% | 50 |
| `ranktwo_shape_chi2_w025` | 0,10% | 61 |
| `ranktwo_wilcoxon_z_w100` | 0,02% | 71 |
| `ranktwo_wilcoxon_z_w025` | 0,00% | **91 (último)** |

**Desfecho: o segundo caso, com um refinamento importante.** A família não é uniformemente morta —
ela é **uma feature viva e cinco mortas**. `ranktwo_dispersion_z_w100` entrou direto no top-10 do
modelo (à frente de `accum_window_var_ln_w100` e de todas as features de Bayes), e ainda assim a
TS-AUC ficou plana: ela **substituiu** capacidade que já existia em vez de somar. As de *localização*
(`wilcoxon`) são as duas piores do modelo inteiro — exatamente o que o censo A1 prevê (só 6,8% das
séries têm |Δmean_e|>0,3) e uma confirmação independente de que o canal de média está morto.

Consequências, todas incorporadas ao plano:
1. **F6 morre.** JS/Hellinger/W1/KS são majoritariamente funcionais de *localização e forma* do mesmo
   contraste de CDFs — o eixo que acabou de se mostrar inerte. O único funcional vivo é o de
   dispersão, e F3 (MMD) cobre esse eixo de forma mais rica.
2. **F1 sobe ainda mais.** "Feature boa que não move a métrica" é a assinatura exata do gargalo de
   comparabilidade descrito em §2.
3. **F5 desce.** Padrões ordinais são invariantes a escala — vivem no eixo que se mostrou morto, não
   no de dispersão. Fica para depois de F1–F4 e apenas se houver folga.

---

## 6. Sequenciamento recomendado

1. **Diagnóstico §5** (~15 min) → decide a ordem.
2. **F1 + F2** juntos (custo zero de latência; F2 alimenta F1). Um rebuild + treino + R0. É a aposta
   de maior leverage e a que tem a previsão falsificável mais nítida (share de `meta_h0_*` cai).
3. **F3** (MMD/RFF marginal + conjunta). Espécie genuinamente nova, 9 µs.
4. **F4** (multi-escala). Barato, alvo específico e mensurável na suíte de robustez.
5. **F5** só se 2-4 não fecharem a conta; **F6** só sob o desfecho apropriado de §5.

Cada etapa julgada por `scripts/compare_oof.py` (R0) com hipótese registrada *antes*, e com CE6 +
suíte de robustez a cada iteração, conforme o protocolo §3.8. Custo total se tudo for adiante:
~9+4+22+47 ≈ **82 µs/passo** → ~764 µs/passo, bem dentro do gate de 1500.

---

## 7. O que eu explicitamente NÃO faria

- **Força bruta de 2408 features** (abordagem do 2º lugar de 2025, com seleção por SHAP). O n efetivo
  é ~10⁴ séries e cada feature custa latência real no motor causal; a seleção pós-hoc sobre 2408
  candidatos num n desses é uma máquina de overfitting de seleção — o mesmo erro que R2 cometeu hoje
  em escala menor.
- **Mais variantes das famílias existentes** (mais deltas de CUSUM, mais janelas, mais hazards) — já
  julgadas empiricamente nulas, duas vezes.
- **Wavelet denoising como pré-processamento** (usado pelo 3º bloco de 2025): na forma padrão não é
  causal e conflitaria com o motor único. F4 usa a decomposição, não o denoising.
- **Features transversais** (rank da feature entre séries vivas no mesmo t): a API de submissão
  entrega uma série por vez; é estruturalmente impossível e seria vazamento.

---

## 8. Ressalva final honesta

Nada aqui garante ganho. O precedente de R4 — uma família nova, bem motivada, corretamente
implementada, com resultado nulo — é o cenário base que qualquer uma destas propostas precisa bater.
A diferença de aposta é que **F1 não adiciona um detector: corrige a comparabilidade de todos os
detectores que já existem**, e é a única proposta cuja motivação vem de uma medição direta do
comportamento do modelo atual (34,3% do SHAP gasto em calibração) em vez de analogia com outra
competição. Se F1 falhar com o share de `meta_h0_*` inalterado, a leitura de §2 estava errada e vale
reconsiderar seriamente a hipótese H-informação do parecer (§4.5).

---

## Fontes

- [Winning solution write-up — ADIA Lab Structural Break Challenge (Alphabot)](https://humbertobrandao.medium.com/how-far-can-we-push-the-winning-model-of-the-adia-lab-structural-break-challenge-87ebf3d0ff67)
- [2nd place solution — GitHub (aParsecFromFuture)](https://github.com/aParsecFromFuture/ADIA-Lab-Structural-Break-Challenge-Solution)
- [NEWMA: a new method for scalable model-free online change-point detection](https://arxiv.org/abs/1805.08061)
- [Optimal Online Change Detection via Random Fourier Features](https://arxiv.org/html/2505.17789v1)
- [Change-Point Detection Using the Conditional Entropy of Ordinal Patterns](https://doi.org/10.3390/e20090709)
- [Detecting Change-Points by Maximum Mean Discrepancy of Ordinal Pattern Distributions](https://arxiv.org/pdf/1210.4903)
- [Winning Solutions of Structural Break Challenge — CrunchDAO forum](https://forum.crunchdao.com/t/winning-solutions-of-structural-break-challenge/1070)
