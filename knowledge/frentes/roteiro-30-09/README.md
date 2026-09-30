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
| 2 | Professor/aluno (rótulos de detectabilidade por linha) | — |
| 3 | Eixo escondido como colunas (condicional à sonda) | **fora** (a sonda não atingiu 0,60) |
| 4 | Especialistas com poucos pesos | — |
| 5 | Simulação de aprender com o teste | — |

## Passo 0

- **Premissa 2 corrigida** a partir do `runner.py` oficial ([premissa 2](../../premissas/02-estado-entre-series.md)).
- **Empates em float32:** no `oof_e1_vema_bag4`, `astype(float32)` reduz os valores distintos por passo
  de 2.541.134 para 2.539.518, e a TS-AUC muda **0,000000**. A produção emite `sigmoid(raw)` em
  [0,106; 1,0], sem colapso relevante. **Não se troca a saída para logit.**
- **V4** (poda top-65): −0,0019 [−0,0041; +0,0003]. **V8**: −0,0006. Nenhum empilhamento sobre o E1
  ([V-empilhamento](../teto-offline/V-empilhamento.md)).
- **Submissão do E1:** fica com o usuário (conta e token Crunch).
