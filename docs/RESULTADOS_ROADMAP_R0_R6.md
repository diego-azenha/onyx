# Resultados da implementação do roadmap R0–R6 (docs/PARECER_AUDITORIA_ONYX.md)

**Data:** 2026-07-20
**Escopo:** implementação completa de R0–R6 do parecer de auditoria, com retreino real sobre o
dataset completo (10.000 séries, 2.541.134 linhas pós-thinning, 91 features) e validação via R0.

---

## 1. Resumo executivo

Todos os seis itens do roadmap foram implementados, testados (80 testes unit+causality+determinism
passando) e validados com dados reais. O resultado mais importante **não é um ganho de TS-AUC** — é
que o processo (R0) capturou uma regressão real antes que ela pudesse ser adotada silenciosamente, e
os três braços de modelo produzidos (binário-R1, rank-R3, combinado) são **estatisticamente
indistinguíveis do baseline pré-auditoria** (nenhum IC exclui 0 a favor de ganho). Isso é uma
resposta honesta e válida à bifurcação H-extração vs. H-informação do parecer (§4.5) — ver §5.

## 2. O que foi implementado

| Item | Entregável | Status |
|---|---|---|
| R0 | `scripts/compare_oof.py` — comparador OOF pareado, bootstrap por série, IC 95% | feito, usado extensivamente |
| R1 | `src/sbrt/model/weights.py` — pesos pareado-consistentes (w_pos∝n_neg, w_neg∝n_pos, suavizados, capados) | feito |
| R2 | `feval` custom (`ts_auc_by_t` + `binary_logloss_diag`) em `model/train.py`; `scripts/sweep_hyperparams.py` | feito, com achado empírico importante (§3) |
| R3 | `model/train.py:train_rank`, `RankModelEnsemble`, `CombinedModelEnsemble`, `scripts/train_rank.py`, `scripts/combine_oof.py` | feito, com bug de performance corrigido (§4) |
| R4 | `src/sbrt/state/rank_twosample.py` (6 features novas: Wilcoxon/dispersão/forma janela-vs-histórico) | feito, no dataset retreinado |
| R5 | Gates relativos (`RELATIVE_GATE_SCENARIOS`) em `robustness/gates.py` + `generators.py:generate_reference_panel` | feito, mecanismo validado (§6) |
| R6 | Censo A1 real, resposta ao degrau OOF real, comparação com envelope de potência | feito, números reais abaixo (§7) |

## 3. Achado R2: `ts_auc_by_t` sozinho como critério de parada REGRIDE a TS-AUC real

A especificação original do parecer (§6-R2) propunha usar uma subamostra de ~150k linhas do fold de
validação para o `feval`, com `first_metric_only=True` na parada. Medido em retreino real:

- **Com subamostra de 150k linhas:** ΔTS-AUC OOF vs. baseline = **-0,0119** (IC 95% exclui 0)
- **Sem subamostra (fold inteiro no feval):** ΔTS-AUC OOF vs. baseline = **-0,0099** (IC 95% exclui 0, ainda regressão real)
- **R1 sozinho, revertendo a parada para `binary_logloss` original:** ΔTS-AUC OOF vs. baseline = **-0,0014** (IC 95% inclui 0 — indistinguível do baseline)

Diagnóstico: o número de árvores subiu de ~61–89 (baseline) para 100–236 rodadas ao usar
`ts_auc_by_t` como critério de parada — a régua certa em teoria, mas com ruído entre rodadas dominado
pelo **n efetivo de ~10⁴ séries** (não pelo número de linhas, mesmo com o fold inteiro no feval). A
seleção do argmax ao longo de 100+ rodadas nessa métrica ruidosa produz um viés de seleção
("winner's curse") que não generaliza para o OOF completo. Isso reforça exatamente o argumento do
próprio parecer (§2, "n efetivo é ~10⁴, não 2,5M") — só que aplicado ao critério de PARADA, não apenas
à capacidade do modelo.

**Correção aplicada:** `configs/default.yaml:lightgbm.early_stopping_metric` (novo campo) default
`"logloss"` — reproduz o comportamento original, validado. `"ts_auc_by_t"` continua disponível como
opção para experimentação futura com estabilização adicional (ex.: `min_delta`, suavização por
janela), não para uso direto. Ambas as métricas são sempre computadas e registradas (`fold_evals`,
`training_curves`); só a ordem que `first_metric_only` usa muda.

## 4. Achado R3: `lambdarank_truncation_level` sem cap trava o treino

A recomendação literal do parecer ("truncation_level ≥ tamanho máximo de grupo") pressupõe grupos
moderados. Neste dataset, `t<=100` mantém **todas** as ~10.000 séries vivas (thinning só começa
depois, `configs/default.yaml:thinning`), então o maior grupo de um fold de treino chega a **~8.000
linhas**. Sem cap, o custo por grupo escala ~`group_size × truncation_level` — o treino real rodou
**mais de 4 horas sem terminar** (processo morto manualmente). Um benchmark controlado confirmou a
causa: 10 rodadas em 1,4s com `truncation_level=300` vs. 6,2s com `truncation_level=3000` (mesmo
grupo, ~4,5x mais lento já numa escala 10x menor que a real).

**Correção aplicada:** `configs/default.yaml:rank.truncation_level_cap` (novo campo, default 300) —
`lambdarank_truncation_level = min(maior_grupo_do_fold, cap)`. Grupos maiores que o cap ficam com
gradiente pleno só no topo — risco aceito por tratabilidade computacional, documentado no código.

## 5. Comparação dos três braços (R0, dados reais, 300 réplicas de bootstrap)

Baseline = OOF do modelo pré-auditoria (`artifacts/models_baseline_preaudit/`), TS-AUC OOF geral
≈ 0,5996–0,601 (medido antes desta sessão).

| Braço | TS-AUC OOF geral | Δ vs. baseline | IC 95% | Exclui 0? |
|---|---|---|---|---|
| Binário (R1 pesos + R4 features, parada `logloss`) | 0,5982 | -0,0014 | [-0,0069, 0,0045] | não |
| Rank (R3, lambdarank sozinho) | 0,5852 | (mais fraco, não testado formalmente vs. baseline) | — | — |
| Combinado (rank-average binário+rank) | 0,5993 | -0,0003 | [-0,0061, 0,0058] | não |
| Combinado vs. binário sozinho | +0,0010 | — | [-0,0035, 0,0058] | não |

Por bucket de t (binário vs. baseline): t≤50 Δ=-0,0067 [-0,0229, 0,0130]; 50<t≤150 Δ=+0,0007
[-0,0097, 0,0109]; 150<t≤400 Δ=-0,0011 [-0,0073, 0,0060]; t>400 Δ=-0,0026 [-0,0086, 0,0047]. Nenhum
bucket mostra ganho ou perda estatisticamente significativa.

**Leitura honesta:** a previsão central do parecer (R1 melhora t≤150 por realinhar o gradiente à
métrica) **não se confirmou** neste dataset real — nem para melhor, nem para pior. R3 (rank) sozinho
performa pior que o binário; combinado não supera o binário isoladamente de forma significativa. Isso
é uma falsificação legítima, no mesmo espírito da previsão nº1 do `plano_acao` que o próprio parecer
documenta como falsificada (§4.5, D3) — o mecanismo é limpo e a derivação é correta, mas "mecanismo
limpo não garante ganho" (parecer §3.10), e agora isso está medido, não presumido.

## 6. R5: gates relativos — mecanismo validado

Rodado contra o modelo binário retreinado (`artifacts/reports/robustness_v1.json`, 60 seeds). Os
gaps relativos são computados corretamente e produzem números sensatos onde antes haveria um nível
absoluto sem sentido para um score não calibrado:

- t2: gap=0,0000 (nível idêntico ao painel de referência)
- t6: gap=0,1205 (abaixo do limiar 0,40 — passaria pelo componente de nível)
- t9: gap=0,0613 (abaixo do limiar 0,40)
- t10: gap=0,0182 (abaixo do limiar 0,40)
- t13: gap=0,1135 (abaixo do limiar de decaimento 0,15 por pouco)

A maioria das reprovações remanescentes vem de um componente **separado e fora do escopo de R5**:
`drift_slope_abs_max=1e-4` é extremamente apertado e reprova t2/t6/t10/t12/t12b por um slope da
ordem de -0,0003 a -0,0008 — não uma falha do mecanismo de gate relativo. Não recalibrei esse limiar
nesta sessão (não é um dos itens R0–R6; recalibrá-lo é uma decisão de tuning que merece sua própria
validação). t9 e t13 reprovam por decaimento insuficiente, também não relacionado a R5.

## 7. R6: números reais

**Censo A1** (`artifacts/reports/break_type_census.csv`, 4552 séries analisadas de 4967 com quebra,
415 puladas por segmento curto): confirma E3 do parecer com dados reais —
mediana |Δmean_e| = 0,0033 (shift de média desprezível), apenas 6,8% das séries têm |Δmean_e|>0,3;
mediana |Δlogvar_e| = 0,0798, mas **41,8%** das séries têm |Δlogvar_e|>0,3 — o sinal dominante é
variância/cauda, não média, exatamente como o parecer previu.

**Resposta ao degrau OOF** (`artifacts/reports/oof_step_response.csv`): score médio sobe de ~0,43
perto de τ para ~0,53 em offset+250, uma curva de resposta genuína e monotonicamente crescente
(com oscilação de paridade par/ímpar — artefato do thinning em t>100, não sinal real).

**Envelope de potência** (mean-shift z-test, δ=mediana|Δmean_e| do censo=0,0785, α=0,05): a AUC
observada excede o envelope em TODOS os buckets (gaps de -0,25 a -0,44, i.e., AUC bem ACIMA do que
um detector de shift-de-média simples entregaria) — confirma que o modelo extrai sinal de
variância/forma que o envelope de média sozinho não modela, consistente com o achado do censo.

## 8. Notas operacionais (para quem retomar este trabalho)

- **`| tail -N` em comandos de background quebra o streaming de progresso**: `tail` sem `-f` não
  emite nada até o processo de origem fechar o pipe — mesmo com `python -u`/`flush=True`, um job
  real de dezenas de minutos aparenta estar "travado" até terminar. Rode sem o `| tail`, leia o
  arquivo de saída bruto incrementalmente.
- O treino do braço rank (R3) leva ~16 min no dataset completo (5 folds × ~3,3 min); a suíte de
  robustez com 200 seeds leva a maior parte de uma hora — considerar n_seeds menor para iteração
  rápida (a suíte de CI já usa 40).

## 9. Próximos passos sugeridos (não executados nesta sessão)

- R2, segunda metade: sweep de hiperparâmetros real (script pronto, `scripts/sweep_hyperparams.py`)
  julgado pelo `logloss` (critério validado) em vez do `ts_auc_by_t` ruidoso.
- Investigar por que R3 (rank) performa pior isoladamente — candidatos: `label_gain`/objective
  alternativo (`rank_xendcg`), ausência de pesos de linha (thin-only pode não bastar), ou o mesmo
  problema de ruído de seleção na parada (agora usando `logloss` sigmoid-mapeado, que pode não ser o
  critério certo para um objetivo de ranking — ver nota em `config.py:LightGBMConfig`).
- R4: as 6 features novas entraram no dataset mas seu SHAP/importância individual não foi medido
  nesta sessão — rodar `scripts/diagnose.py` e inspecionar `feature_importance.csv` /
  `shap_feature_importance.csv` para o modelo retreinado.
- Recalibrar `drift_slope_abs_max` (fora do escopo R0–R6) se as reprovações de slope em T2/T6/T10
  forem consideradas um problema real, não um artefato de limiar apertado demais.
