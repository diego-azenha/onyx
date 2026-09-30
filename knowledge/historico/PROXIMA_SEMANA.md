# O que falta rodar — plano da semana de 27/07/2026

**Escrito em:** 2026-07-25, ao fim da campanha de polimento.
**Insumos:** `CAMPANHA_POLIMENTO.md` (a lista original), `RELATORIO_POLIMENTO.md` (o que foi medido).

---

## Como esta lista está organizada, e por que mudou

A campanha original numerava por **frente temática** (A = auditorias, B = polimento barato, C =
vizinhanças, D = knobs do LightGBM, E = dados, F = knobs de φ). Isso funcionava para *planejar* —
agrupava por tipo de intervenção — mas é péssimo para *executar*: os itens de uma mesma frente têm
custos que diferem em duas ordens de grandeza, e a ordem de execução real acabou saltando entre
letras (A6 → C2 → D3 → A6 de novo), que é exatamente o que ficou confuso.

**Esta lista é ordenada pela ordem de execução**, e agrupada pelo que de fato governa o custo: **se o
braço exige ou não reconstruir o dataset**. Um rebuild custa 45–60 min e bloqueia a máquina; um braço
sem rebuild custa dois treinos (~17 min) mais uma medição (~8 min). Essa é a distinção que decide
quantos braços cabem numa semana, e por isso é ela que organiza o documento.

Os códigos antigos (`A6`, `C1 (ii)`, `E1`…) aparecem **entre colchetes, só para rastreio** com os
documentos anteriores. Não os use para ordenar nada.

**Custos medidos nesta campanha, para planejar:**

| operação | custo real |
|---|---|
| treino, 1 semente × 5 folds | **8,5 min** (com `linear_tree`: 9–15 min) |
| medição com IC (bootstrap pareado, 2,5 M linhas) | **~8 min** |
| rebuild do dataset | **45–60 min** (`--n-jobs 4`; nunca `-1`) |
| um braço completo sem rebuild (2 sementes + medição) | **~25 min** |

---

## Passo 0 — a leitura de dois minutos que vem antes de qualquer braço

**Ler o score do 10º lugar no leaderboard.**

Não é um braço e não custa compute. É o primeiro entregável que o `CAMPANHA_POLIMENTO.md` pediu
("antes de qualquer braço") e o único item da campanha inteira que **nunca foi feito**.

Ele decide se o resto desta lista faz sentido. O pré-registro é explícito: somando todas as faixas com
a taxa de acerto histórica, a campanha valia **+0,005 a +0,010 de OOF**. Se a linha do rank 10
estiver acima de ~0,64, **nada nesta lista paga prêmio** — e a conversa correta passa a ser o Grupo 5
(dados de 2025) ou reabrir o pivô, não afinar knobs de φ.

Fazer isso primeiro pode economizar a semana inteira.

---

## Grupo 1 — minutos, sem rebuild (segunda de manhã)

São restos baratos. Somados, cabem numa manhã.

**1. Fechar a medição do clip do resíduo.** [A6, K=4]
O artefato está pronto no disco (`artifacts/models/oof_p8_a6_bag4.parquet`, 4 sementes) — falta
**só o bootstrap**, que foi derrubado em 25/07 para liberar a máquina para o gate do pacote.
Comando: `compare_oof.py --baseline oof_b6c6_joint_bag4 --candidate oof_p8_a6_bag4`.
**Custo: 8 min.** Decide dois defeitos já diagnosticados sobre a feature nº 1 (K=2 deu +0,0015).

**1-bis. Auditar o caminho de PRODUÇÃO contra o de MEDIÇÃO.** [nasceu do bug de fusão de 25/07]
**Este item não existia até 25/07, e é o mais importante do Grupo 1.**
O `linear_tree` foi medido, aprovado e adotado — e **quebrava a fusão de boosters**, que só existe no
caminho de produção (`adapter/platform.py:train`). O OOF nunca passa por lá: `scripts/train.py` faz
média das *predições* por semente. O defeito só apareceu ao regenerar o notebook, e teria feito o
`train()` na nuvem levantar `RuntimeError` — submissão inteira perdida.
A pergunta que fica: **quais outros flags do `default.yaml` divergem entre os dois caminhos?**
Candidato imediato: `dart` (o LightGBM **ignora** `linear_tree` em `dart`, então as duas flags juntas
significam coisas diferentes em cada caminho). Ação: rodar `verify_submission_notebook.py` **como
portão de adoção**, não só de empacotamento — ou seja, antes de adotar um flag, não depois.
**Custo: ~20 min por rodada de verificação.** É a terceira ocorrência da classe A3 ("o que é medido
não é o que é implantado") e a primeira com consequência fatal.

**2. Derrubar as 15 colunas 100% NaN em `t=50`.** [nasceu do A5 bis]
`haar_*`, `mmd_*_cal`, `jump_*_w100_cal`, `varloc_recent_vs_lagged*` estão 100% NaN em `t=50` e
**diluem o sorteio** do `feature_fraction=0,8`. É higiene de conjunto, não sweep de hiperparâmetro.
Alternativa mais fraca: subir `feature_fraction` para 0,9.
**Custo: ~25 min.** *Nota: o `p16_ff06` (0,6) foi pulado em 25/07 porque aponta para o lado errado —
menos features sorteadas fazem as colunas NaN ocuparem fração maior. Não refazer nessa direção.*

**3. Micro-higienes do LightGBM.** [D5]
`enable_bundle=false` (o EFB pode agrupar mal com dezenas de colunas NaN), `first_metric_only`,
`force_row_wise` documentado. **Custo: ~25 min o braço do `enable_bundle`**; os outros dois são
verificação de configuração, não braço.

**4. A célula `dart` do sweep.** [D3]
Única célula da grade original que nunca rodou. **Custo: ~25 min.**
*Atenção: o LightGBM ignora `linear_tree` em `dart`* — como o `linear_tree` agora está adotado, esta
célula testa uma configuração que **exclui** o D1. Medir contra o incumbente certo.

**5. Completar a grade do `feature_contri`: 0,5 e 0,7.** [C1 (i)]
Medidos: 0,2 (−0,0117 em `t≤50`, rejeitado), 0,4 (+0,0017), 0,6 (adotado), 0,8 (−0,0013). A curva
vira entre 0,2 e 0,4 e o 0,4 é o melhor ponto conhecido. **Custo: ~50 min os dois.**
Prioridade baixa: a curva já tem forma, e o ganho de refinar entre 0,4 e 0,6 é marginal.

---

## Grupo 2 — o braço grande da semana (exige rebuild)

**6. Des-thinning do dataset.** [E1]

**É o item de maior valor esperado que nunca foi medido.** Previsão da campanha: **+0,001 a +0,004**.

O thinning (passo 2 em `t>100`, passo 4 em `t>400`) foi uma decisão de *velocidade de build da fase
P0, nunca medida* — e descarta metade das linhas exatamente onde o peso da métrica se concentra. Dois
achados posteriores reforçaram o alvo:

- o **E0c** mostrou que os ajustes aditivos são famintos de dados (1.500–3.200 tocos até convergir);
- o **A2** mostrou que `t>400` vale **31,9%** do peso do board, não 16,5% — ou seja, a região mais
  subamostrada é a mais valiosa, e por quase o dobro do que se pensava.

**Estado:** `configs/e1.yaml` existe (`full_until: 100` → `400`). O parquet **não** existe — o build
de 25/07 morreu num bug de quebra de linha do driver v1, não por RAM. **Não está refutado, está por
fazer.**

**Custo:** build ~45–60 min (dataset ~1,4× maior) + 2 treinos + medição ≈ **1h45**.
**Risco:** é o único item que pode estourar os 16 GB. Rodar sozinho, nada mais na máquina.
**Cuidado de medição:** o OOF do E1 vive numa grade com mais linhas, então o `compare_oof.py` casaria
só o subconjunto comum. A leitura honesta é o **nível na grade do board**, não o Δ pareado.

---

## Grupo 3 — vizinhanças que a lista pedia e a execução pulou (sem rebuild)

Estes dois são os únicos braços da campanha original que foram **listados e nunca tocados** — não
foram refutados nem descartados, só ficaram para trás quando a fila virou.

**7. O conjunto penalizado, não o valor.** [C1 (ii)]
O `feature_contri_meta=0,6` teve o **valor** varrido (item 5 acima), mas o **conjunto** de colunas
penalizadas nunca foi questionado. A campanha pedia estender a penalização aos rastreadores de `t`
que o xs-SHAP expôs — `conformal_logm_abs` e `mmd_joint_slow_cal` — com `feature_contri` próprio.
**Este é o maior ganho recente do projeto (+0,0036) com metade da hipótese por testar.**
**Custo: ~25 min.**

**8. `delta_mean` no `d_i`.** [C2 (ii)]
A vizinhança do X1 foi varrida só no eixo do `floor` (0,15 e 0,50 — ambos fechados negativos ou
sub-barra). O segundo braço que a campanha listava — incluir `delta_mean` no cálculo de `d_i` —
nunca rodou. **Custo: ~25 min.**

---

## Grupo 4 — um rebuild cada, só se sobrar semana

Toda a frente F. **Nada dela foi tocado.** Cada item exige reconstruir o dataset, o que a torna cara
por construção: três braços = três rebuilds = ~3h só de build.

A lógica que a justifica: "largura mata" era sobre **adicionar** funcionais; **afinar** os
hiperparâmetros internos dos canais que já dominam o xs-SHAP é outra operação — melhora o SNR do
mesmo canal sem diluir o sorteio de features.

**9. Grade de deltas do CUSUM de variância** (densidade/alcance) — o canal da família campeã (0,93).
**10. Bandwidth/taxa do `mmd_joint_slow`.**
**11. ε do martingale conformal.**

Previsão: **0 a +0,002 cada**. Um rebuild + braço cada ≈ **1h30 por item**.

**12. Threshold do `vol_adjust`.** [F2] O gate liga/desliga por limiar duro de `rho1_abs_e`, então
séries na fronteira recebem tratamento descontínuo. Suavizar (rampa) ou varrer o limiar. Micro.

**13. Ordem do AR por série** (AIC em {5..15} em vez de 10 fixo). [F3] Mexe em tudo downstream, caro,
efeito desconhecido. **Último da fila por desenho** — só se sobrar semana de verdade.

---

## Dívidas de protocolo — não são braços novos

Não testam hipótese nenhuma; fecham medições que o protocolo exige. Baratas, e é o tipo de coisa que
volta a morder quando se esquece.

| o quê | custo | por quê |
|---|---|---|
| Réplica do bag K=7 na partição 43 | ~35 min | adotado com **uma** partição só, pelo argumento de que redução de variância é propriedade matemática — o argumento é bom, mas a réplica fecha a conta |
| O pacote medido na partição 43 | ~35 min | o gate de 25/07 rodou só na 42 |
| E0c na partição 43 | ~1h20 | pendência declarada do Exp0 [A4] |
| Reauditoria em lote, 15 pares | ~1h | `scripts/a2_reaudit.py` está pronto; só o recorte focado rodou. **Precisa rodar sozinho** — competindo com treino, causou o deadlock de OpenMP do `NOTAS_AGENTES.md` §7 |
| `refit` sobre 100% dos dados | 1 quota | [D4] Mantém a estrutura das árvores da CV e reajusta só os valores das folhas com todos os dados. **O OOF não enxerga esse ganho por construção** — só mede contra o board, então gasta quota |

---

## Grupo 5 — fora do escopo declarado

**As linhas de 2025 como dados extras.** [E3]

Não é feature nova nem modelo novo — é mais amostra para os mesmos ajustes aditivos. **É o único item
com teto acima de +0,005**, e portanto o único que muda a aritmética da campanha se a leitura do
Passo 0 vier ruim.

Fica registrado com os portões de sempre (regra, suporte, probe de domínio). **A decisão de abrir é
do usuário**, e ela deve ser tomada com o número do rank 10 na mão, não antes.

---

## Resumo em uma tela

| ordem | item | custo | rebuild? | previsão |
|---|---|---|---|---|
| **0** | **ler o rank 10 do leaderboard** | 2 min | — | *decide se o resto vale* |
| 1 | fechar a medição do clip do resíduo | 8 min | não | K=2 deu +0,0015 |
| **1-bis** | **auditar produção vs. medição** | 20 min | não | *evita perder uma submissão* |
| 2 | derrubar as 15 colunas NaN | 25 min | não | desconhecida |
| 3 | micro-higienes do LightGBM | 25 min | não | ~0 |
| 4 | célula `dart` | 25 min | não | 0 a +0,004 (faixa do D3) |
| 5 | `contri` 0,5 e 0,7 | 50 min | não | marginal |
| **6** | **des-thinning** | **1h45** | **sim** | **+0,001 a +0,004** |
| 7 | conjunto penalizado estendido | 25 min | não | +0,001 a +0,003 (faixa do C1) |
| 8 | `delta_mean` no `d_i` | 25 min | não | 0 a +0,002 |
| 9–11 | os três knobs de φ | 1h30 cada | sim | 0 a +0,002 cada |
| 12–13 | `vol_adjust`, ordem do AR | — | sim | desconhecida |

**Se a semana der para dois itens só: o Passo 0 e o item 6.** Um custa dois minutos e pode cancelar a
lista inteira; o outro é o único braço não medido com previsão acima de +0,003.

---

## Regras que continuam valendo

- **Barra de adoção: 0,0014.** Réplica na partição 43 obrigatória antes de adotar.
- **Um braço por mudança.** O pacote final medido **inteiro** — quatro não-aditividades documentadas
  neste repo já mostraram que duas mudanças positivas não somam por decreto.
- **K=2 ordena braços, não estima efeitos.** Nunca comparar K=4 contra K=2, e nunca reverter por
  evidência mais fraca do que a que fez adotar (`HISTORICO.md` §15.16).
- **Um processo pesado por vez**, `OMP_NUM_THREADS=4`, build com `--n-jobs 4`, bootstrap com
  `--n-jobs 2`. A v1 do driver travou a máquina por sobre-assinatura aritmética.
- **Nenhuma submissão sem uma linha em `artifacts/reports/submission_log.md`.**
