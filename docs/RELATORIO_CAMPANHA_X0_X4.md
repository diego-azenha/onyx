# Relatório — Campanha de ruptura TS-AUC (X0–X4)

**Data:** 2026-07-24 · **Repo:** `onyx` (ADIA Lab Structural Break Challenge — Real-Time) · **Fonte:** `BRAINSTORM_RUPTURA_TSAUC.md`

Documento autocontido para handoff a um agente externo. Cinco apostas ortogonais (X0–X4), cada uma com mecanismo, gate barato, previsão pré-registrada e critério de morte. **Resultado: 1 vitória (X1), 4 mortes, e uma recalibração da régua de decisão mais valiosa que qualquer braço.**

---

## 1. Sumário executivo

| braço | eixo | veredito | Δ TS-AUC (OOF, vs V4) | decisão |
|---|---|---|---|---|
| **X1** | supervisão ponderada por detectabilidade | **✓ VITÓRIA** | **soft +0,0040** [+0,0005; +0,0073] · hard +0,0036 [+0,0004; +0,0071] | **ADOTADO (soft)** |
| X0 | rodadas de boosting fixas | ✗ morto | logloss −0,0019 (IC exclui 0 *contra*) · tsauc −0,0028 | descartar |
| X2 | canal de nível/média | ✗ morto no gate D1 | — (build evitado) | descartar |
| X3 | studentização do score por série | ✗ morto | −0,0294 | descartar |
| X4 | injeção sintética (n efetivo) | ✗ bloqueado no probe | — (build evitado) | reabrir só após redesenho (~2 dias) |

**Baseline (incumbente):** V4 legítimo, 183 features, OOF K=4 (sementes 777/101/202/303) = **0,6018**. O histórico 0,6100 é a semente 42, contaminada por seleção (a 9 desvios das demais) — não usar.

---

## 2. Metodologia (os trilhos que todo braço seguiu)

- **K=4 sementes** `[777,101,202,303]`, semente 42 **excluída**. Folds fixos (`cfg.seed=42`) idênticos nos dois lados; OOF por linha casa por `(id,t)`.
- **Baseline honesto:** `data/processed/train_rows_3eixos.parquet` **com `--drop-prefix spec_ ord_ mrep_`** = as 183 features do V4 (bit-exato; `train_rows.parquet` é o dataset do V5, não do V4 — errata de 22/07).
- **R0 (juiz relativo):** `scripts/compare_oof.py` — Δ pareado por série com IC 95% (bootstrap de `id`s). **Adotar sse o IC exclui 0 a favor** no agregado ou no bucket-alvo declarado a priori. Comparações sempre sobre OOF médias de K sementes (`scripts/avg_oof.py`, média em espaço de probabilidade).
- **Barra de decisão:** ver §4 — a barra correta é **~0,0014**, não os 0,0058 do histórico.
- **Buckets de peso da métrica:** t≤50 = 8%, 50–150 = 27%, **150–400 = 49%**, >400 = 16%. Ganho útil concentra-se em 50–400.

---

## 3. Braço a braço

### X1 — Supervisão ponderada por detectabilidade · **VITÓRIA, ADOTADO**

**Mecanismo.** Parte dos positivos não carrega sinal no passo em que pesa (magnitude baixa; linhas logo após τ). Com logloss o booster paga caro por elas e distorce a fronteira do resto. Solução: baixar o peso desses positivos no **treino** — peso do professor, usa τ que o próprio alvo `y=1{τ≤t}` já usa; nenhuma causalidade de inferência tocada.

**Fórmula.** `w_positivo ×= clip(d_i / q95, 0,3, 1)`, com `d_i` = detectabilidade do censo A1 = norma L2 dos eixos padronizados `delta_logvar_e / delta_rho1 / delta_kurt / delta_exceed` × √m_bucket, `m_bucket = min(50−τ, n_post)`. Só séries de quebra precoce (0<τ<50, ~712 séries) recebem d_i; as demais ficam com peso 1. `q95` = 95º percentil de d_i.

**O que foi rodado.** 2 variantes × K=4 retreinos (ES-on, sem build — reusa o parquet), R0 vs `oof_x0_a_bag4` (V4 K=4):

| variante | overall | t≤50 | 50–150 | 150–400 | >400 |
|---|---|---|---|---|---|
| **soft** (clip d/q95) | **+0,0040** [+0,0005;+0,0073] ✓ | — | **+0,0081** [+0,0015;+0,0147] ✓ | +0,0038 [+0,0000;+0,0074] ✓ | — |
| hard (dropa quintil inferior) | +0,0036 [+0,0004;+0,0071] ✓ | −0,0048 (incl.) | +0,0074 [+0,0013;+0,0134] ✓ | +0,0042 [+0,0008;+0,0075] ✓ | −0,0000 |

**Veredito.** Vitória robusta e replicada (2/2 variantes, IC exclui 0 no agregado E nos dois buckets de maior peso). `soft` marginalmente melhor. (A 3ª variante `ramp` não foi concluída — ambiente matou os jobs de background; não altera a conclusão.)

### X0 — Rodadas de boosting fixas pela curva média entre sementes · **MORTO**

**Mecanismo (hipótese).** O early stopping por logloss escolhia de 51 a 103 árvores no mesmo fold conforme a semente; fixar rodadas pela curva média entre sementes eliminaria esse ruído (σ_seed) e possivelmente ajudaria (o modelo com mais árvores fora o melhor).

**O que foi rodado.** K=4 retreinos com ES **desligado** até o cap (200 árvores/fold, ~9 min/semente), curvas completas salvas; `scripts/x0_fixed_rounds.py` fatia os boosters no argmax da curva média por fold (logloss `[75,108,63,79,72]`; tsauc `[32,115,20,34,41]`). 3 políticas comparadas.

**Resultado.** σ_seed **não caiu** (ES: dp 0,0010, média 0,6018; fixas: dp 0,0011, média 0,6001). Δ: política logloss **−0,0019** [−0,0028; −0,0010] (IC exclui 0 *contra*, pior em todo bucket); política tsauc −0,0028 [−0,0082; +0,0027] (t>400 −0,0041 exclui 0 contra).

**Veredito.** Kill acionado. A premissa (parada errática = ruído grande) só valia por causa da semente 42; entre sementes limpas a parada já é estável. **Ver §4 — o subproduto é o que importa.**

### X2 — Canal de nível/média (2 defeitos de código) · **MORTO NO GATE D1**

**Mecanismo (hipótese).** Defeito 1: em séries persistentes (Σφ→1) um degrau de média vira um pulso curto no canal branqueado e satura em `e_vol` → detector cego. Defeito 2: `hedge_ewma_z` padroniza pelo desvio marginal, não pelo nulo sob H0.

**Gate D1 (minutos, antes de qualquer código).** Estratificar a AUC da família de média por tercil de persistência (`ar_r2`). Predição registrada: cegueira cresce com a persistência (AUC decrescente).

**Resultado.** AUC (subconjunto |Δmean_e|>0,3, n≈100/tercil): baixo **0,6356** / médio 0,5770 / alto **0,6383**. O tercil persistente é o **melhor**, não o cego → Defeito 1 **refutado**.

**Veredito.** Morto no gate, build de ≤3 colunas evitado. Defeito 2 (`hedge_ewma_z_cal`) tem racional independente mas prior enfraquecido; não perseguido.

### X3 — Studentização distribucional do score por série · **MORTO**

**Mecanismo (hipótese).** Transformar o score pela CDF nula da própria série (equaliza escala e forma, não só nível — diferente da centragem aditiva que já morrera). Por-série ⇒ reordena transversal (não é a recalibração global C1-neutra).

**O que foi rodado (offline, sem retreino).** Reusa `centering_baselines.parquet` (score cru do modelo sobre a cauda H0 de cada série, 10k séries × 200 passos). Studentiza o OOF pelo percentil na CDF nula da série. Gate: corr_xs(score, studentizado) < 0,95.

**Resultado.** Gate **passa** (corr_xs 0,326 — reordena de fato), mas TS-AUC cai **0,6062 → 0,5767 (Δ −0,0294)**. O nulo do histórico tem viés de regime e destrói sinal transversal legítimo.

**Veredito.** Morto. Mesmo destino da centragem aditiva, como o prior baixo previa.

### X4 — Injeção sintética de quebras em portadores reais · **BLOQUEADO NO PROBE**

**Mecanismo (hipótese).** Gargalo é n_eff ≈ 10⁴ séries. Caudas de histórico (≥400 obs) são portadores H0 reais: pseudo-história + pseudo-online com quebra injetada em τ′ conhecido → supervisão ilimitada.

**Probe de artefato (OBRIGATÓRIO antes do braço).** Treinar discriminador real-vs-injetado **sob H0** (m=0). AUC>0,6 ⇒ o gerador vaza assinatura ⇒ o modelo aprenderia a detectar "injeção", não "quebra".

**Resultado.** Gerador implementado (`src/sbrt/adversarial/inject.py`, famílias média/variância + controles-espelho). Probe: real (linhas H0 de `train_rows`) vs sintético (portador do gerador) = **AUC 0,9998**. Vaza: os portadores sintéticos têm `meta_h0` ajustado em histórico truncado (`hist[:-200]`), trivialmente separável das linhas reais.

**Veredito.** Bloqueado — o gate evitou um build + K=4 desperdiçado. Reabrir exige redesenhar a construção do portador (~2 dias) para o H0 sintético casar com o real, e re-probar.

---

## 4. Achado transversal — a régua estava 4× errada (subproduto do X0)

A tese σ_seed ≈ 0,0041 / barra 0,0058 que fundamentava o BACKLOG inteiro era **artefato de incluir a semente 42** no cálculo do σ. Entre as 4 sementes limpas:

- σ single-seed ≈ **0,0010** (não 0,0041)
- EP da diferença de duas médias K=4 ≈ **0,0007**
- **barra 2-EP ≈ 0,0014** (não 0,0058)

**Consequência:** a régua já estava ~4× mais fina do que a campanha supunha (com a semente 42 fora, como o protocolo já mandava). Sob 0,0058 a vitória do X1 (+0,0040) teria sido descartada como "inconclusiva"; contra 0,0014 ela é ~2,5× a barra. **Régua nova para todo braço futuro: ~0,0014 (K=4, semente 42 fora.** Reabre em tese as três rejeições históricas F1/V5/F2, agora contra a barra correta.

Diagnóstico complementar (skyline, `scripts/power_envelope_check.py`): mediana |Δmean_e| = **0,0785** — o canal de média puro tem teto baixo (confirma o "não fazer" do X2).

---

## 5. Implementação do X1 em produção

- **Config:** `configs/default.yaml → weights.detectability_mode: soft`, `detect_floor: 0.3` (+ `WeightsConfig` em `src/sbrt/config.py`).
- **Detectabilidade inline:** `src/sbrt/model/detectability.py` (novo) — a nuvem treina do zero e não tem o `detectability.csv` local, então `compute_detectability_map` reproduz o censo A1 + F0.b a partir dos próprios registros de treino. **Bit-idêntico ao artefato medido: max|Δ| = 3,6e-15, corr = 1,000000.**
- **Pesos:** `src/sbrt/model/weights.py` aplica o multiplicador só nos positivos; default `none` (chamadores nus — sweep, testes — intocados). `src/sbrt/adapter/platform.py` (caminho de submissão) opta pelo modo do config e passa o mapa.
- **Custo:** só treino (um passe extra de `fit_h0`/whiten por série). **Inferência e latência inalteradas.**
- **Notebook de submissão:** `scripts/build_submission_notebook.py` inclui o novo módulo; `submission_notebook.ipynb` regenerado e **verificado bit-a-bit** (`scripts/verify_submission_notebook.py`: max|score_notebook − score_pacote| = 0). `train()` roda o X1 inline sem erro no molde real.
- **Docs:** `docs/HISTORICO.md` §13 + nota no §1; `docs/NOTAS_AGENTES.md` §6.
- **Desligar/baseline:** `weights.detectability_mode: none`, ou `scripts/train.py --detectability-mode none`.

---

## 6. Artefatos e código produzidos

**Código novo/alterado:** `src/sbrt/model/detectability.py` (novo), `src/sbrt/adversarial/inject.py` (novo), `src/sbrt/model/weights.py`, `src/sbrt/adapter/platform.py`, `src/sbrt/config.py`, `configs/default.yaml`, `scripts/train.py` (+overrides `--early-stopping-rounds/--n-estimators-cap/--detectability-mode`), `scripts/x0_fixed_rounds.py`, `scripts/d1_persistence_stratify.py`, `scripts/score_studentize_probe.py`, `scripts/x4_artifact_probe.py`, `scripts/build_submission_notebook.py`.

**Resultados (R0 JSON):** `artifacts/reports/compare_x1_{hard,soft}.json`, `compare_x0_{logloss,tsauc}.json`, `compare_x3.json`. **Log de âncoras:** `artifacts/reports/submission_log.md` (criado, com a âncora `5e42ff5` board 0,6201 / OOF 0,6100). **Modelos OOF:** `artifacts/models/oof_x1_{hard,soft}_bag4.parquet`, `oof_x0_a_bag4.parquet`, `oof_x0_{logloss,tsauc}_bag4.parquet`.

---

## 7. Nota operacional

O ambiente (laptop) hiberna em ociosidade e **matou repetidamente os jobs de background** (retreinos longos). Mitigação usada: persistir cada resultado por variante/semente assim que termina, e rodar em cadeia sequencial (nunca 3 jobs pesados em paralelo — isso satura os 8 núcleos, já que cada `train` usa 8 threads). A variante `ramp` do X1 ficou por concluir por isso (não muda a conclusão). Para retreinos longos: manter a máquina acordada ou rodar em nuvem/box dedicado.

---

## 8. Pendências e recomendações

1. **Submeter oficialmente** o V4+X1-soft (notebook já pronto e verificado) e registrar o par (OOF, placar) em `submission_log.md`. É a única forma de confirmar o +0,0040 contra o leaderboard — o mapa OOF↔placar tem só um ponto. *(Não feito por decisão do usuário.)*
2. **Fixar a barra em ~0,0014** (K=4, semente 42 fora) para todo braço futuro.
3. **Não** perseguir X0/X2/X3 como estão (mortos por medição). **X4** só vale reabrir com o redesenho do portador para passar no probe de artefato.
4. Concluir a variante `ramp` do X1 é opcional (hard/soft já decidem).
