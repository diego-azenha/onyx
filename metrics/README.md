# metrics/ — painel de diagnóstico do modelo

Pasta dedicada aos gráficos e às tabelas de métrica que os sustentam. Tudo aqui é diagnóstico
**relativo**, na linguagem da própria TS-AUC; nada é um estimador do score oficial de leaderboard
(docs/MODELO.md §9.0). Cada gráfico tem um `.csv` par com os números por trás.

Regenerar tudo:

```bash
python scripts/metrics_report.py          # os 6 painéis abaixo (modelo oof_final_bag3)
python scripts/viz_break_detection.py     # as grades de séries-exemplo por faixa de tau
```

## Painéis (`scripts/metrics_report.py`)

| Arquivo | O que responde | Achado no modelo atual |
|---|---|---|
| `auc_by_step` | Anatomia da métrica: `AUC_t` contínua (bins de 10) + o peso `w_t=n_pos·n_neg`. Onde a ordenação se perde e se há peso ali. | AUC sobe de ~0,54 (t≤50) a ~0,65 (t>400); o peso concentra-se em 50–400. |
| `auc_by_break_axis` | TS-AUC por família de quebra dominante × tercil de magnitude (positivos do eixo vs. séries sem quebra). Qual família falha. Cobre 92% das quebras (as com linha no censo). | Dependência-baixa ≈ 0,53 (quase acaso); canal de **média chato ~0,55–0,58 mesmo em alta magnitude**; variância/cauda/curtose respondem forte à magnitude. |
| `step_response` | Excesso de score sobre a média do MESMO passo (desvio transversal — remove a taxa-base) vs. `(t−τ)`, por tercil de magnitude + controle. Detecção limpa por magnitude. | Alta magnitude atinge +0,11 de excesso; baixa mal descola do controle (~0). |
| `xs_base_level` | Causa (2): sinal da quebra **líquido do passo** (gap pós−pré no desvio transversal) vs. espalhamento de base ENTRE séries no mesmo passo. | O sinal líquido é ~0 em t≤50 (piso de informação, causa 3) e ≤0,007 depois — **sempre** dominado pelo espalhamento (0,04→0,13): gargalo de comparabilidade. O gap CRU (0,047 em t≤50) seria taxa-base, não sinal (BACKLOG §1). |
| `seed_spread` | TS-AUC por bucket do modelo empacotado contra a nuvem de sementes individuais. Efeito real vs. sorteio de semente (~0,004). | O bag fica no topo da nuvem; a dispersão entre sementes é da ordem dos efeitos procurados. |
| `feature_importance_xs` | xs-SHAP (dispersão dentro do passo — a medida certa sob C1, §9.5) vs. mean\|SHAP\| convencional. **Referência: V4.** | `conformal_logm_abs` e `mmd_joint_slow` são muito maiores no convencional (acompanham o relógio); as `meta_h0_*` sobem no xs-SHAP. |

## Grades de exemplo (`scripts/viz_break_detection.py`)

`break_detection_faixa_{1-50,51-150,151-400,401+}.png` — 3×3 de séries com quebra real por faixa de
τ: série (valor) + score + linha de τ, no mesmo eixo x. Inspeção qualitativa "o score sobe depois da
quebra?".

> Observação: os scripts de diagnóstico mais antigos (`diagnose.py`, `oof_ts_auc_by_bucket.py`,
> `oof_step_response.py`, `power_envelope_check.py`, `detectability_report.py`, `shap_report.py`)
> ainda gravam por padrão em `artifacts/reports/`. Aponte-os a `--out-dir metrics` / `--out metrics/…`
> se quiser tudo aqui.
