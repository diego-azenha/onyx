# Frente: roteiro de 30/09 (passos 0–5)

**Aberta em:** 2026-09-30 10:40 · **Origem:** revisão das premissas depois do sprint. O estado entre
séries é viável com um processo; o rótulo ruidoso é o gargalo (professor/aluno); a sonda dentro de cada
série decide se as quebras "invisíveis" são invisíveis. Plano comprimido para caber no dia (prazo da
competição: 1º/10). As compressões foram declaradas antes de medir, no plano da sessão.

Incumbente: **E1 = 0,6349** OOF (partição 42, K=4).

| Passo | O quê | Status |
|---|---|---|
| 0 | Premissa 2, empates em float32, fechar V4/V8 | **concluído**: premissa 2 corrigida; float32 sem efeito (Δ = 0,000000); V4 e V8 nulos |
| 1 | [Sonda dentro de cada série](P1-sonda.md) | **0,578 (faixa do meio)**; sanidades OK; o sinal fraco está na dependência de curto prazo → **passo 3 sai** |
| 2 | [Professor/aluno](P2-professor-aluno.md) | **descartado**: −0,0010 no bag (+ por semente) → **pivô para o [oráculo destilado](../oraculo-destilado/README.md)** |
| 3 | Eixo escondido como colunas (condicional à sonda) | **fora** (a sonda não atingiu 0,60) |
| 4 | [Especialistas com poucos pesos](P4-especialistas.md) | **nulo**: +0,0001; w_desce=0 nas duas metades |
| 5 | [Aprender com o teste (simulação)](P5-aprender-com-teste.md) | **descartado**: pseudo-rótulos −0,0026; até com rótulos verdadeiros −0,0062 |

## Passo 0

- **Premissa 2 corrigida** a partir do `runner.py` oficial ([premissa 2](../../premissas/02-estado-entre-series.md)).
- **Empates em float32:** no `oof_e1_vema_bag4`, `astype(float32)` reduz os valores distintos por passo
  de 2.541.134 para 2.539.518, e a TS-AUC muda **0,000000**. A produção emite `sigmoid(raw)` em
  [0,106; 1,0], sem colapso relevante. **Não se troca a saída para logit.**
- **V4** (poda top-65): −0,0019 [−0,0041; +0,0003]. **V8**: −0,0006. Nenhum empilhamento sobre o E1
  ([V-empilhamento](../teto-offline/V-empilhamento.md)).
- **Submissão do E1:** fica com o usuário (conta e token Crunch).

## Testes extras do dia (insights no caminho)

| Teste | Resultado |
|---|---|
| G1: calibração cruzada por grupo (perfil ACF × curtose) × faixa de t | **−0,0098**: a escala do Onyx entre tipos de série já é melhor que uma logística por célula |
| G2: auto-referência (score − α·linha de base da própria série) | mínimo corrente −0,025; média dos 10/25 primeiros −0,006/−0,005; média corrente 0: descartado |
| Sobreposição invariante a afim entre o teste reduzido e o treino | **zero** em 344k janelas: canal fechado |
| Leaderboard (API, crunch 22, release 234) | **151 posições.** rank 1: 68,7% · 10: 66,9% · 25: ~66,0% · 50: 65,0% · 100: 63,9% · 150: 63,2%. Scores ≥ 0,66: 25; ≥ 0,65: 50; ≥ 0,64: 100. **O E1 (~0,634 estimado) ficaria por volta do rank 125.** Top 50 exige +0,015; top 25, +0,025 |
| n_on previsível pelo conteúdo do histórico? (GBM com CV) | **não**: Spearman 0,015, R² < 0. O prior t/n_on (que vale 0,65 sozinho) é inacessível |
