# Frente: oráculo destilado (Rao-Blackwellização do alvo)

**Aberta em:** 2026-09-30 12:05 · **Origem:** o leaderboard (API do hub, lida em 30/09) mostra **21
posições entre 65,5% e 68,7%**, todas determinísticas. O pelotão está em 0,66–0,675, e a nossa
estimativa (~0,635) não entraria no top 21. O líder aparece como "distilled-oracle"; o usuário lembrou
que o nome pode ser só um apelido gerado pela plataforma. A abordagem se sustenta pelos próprios
méritos, como mostra a seção abaixo.

## A ideia, e por que ela ataca exatamente o gargalo medido

O Onyx é **limitado por amostra de positivos** ([C1](../teto-offline/C1-curva-aprendizado-onyx.md),
[C2](../teto-offline/C2-positivos-ou-negativos.md)). O rótulo por linha, y_t = 1{τ ≤ t}, é muito
ruidoso: linhas logo depois de τ e quebras invisíveis são "positivas" sem nenhuma evidência, e
negativos com trechos esquisitos são "negativos" com cara de quebra.

Um **oráculo** que vê a série **inteira** (features do Onyx em t e em snapshots futuros, t+50, t+200 e
fim da série, mais o comprimento total T) estima q_t = P(y_t = 1 | série inteira). As features do aluno
estão contidas no que o oráculo vê, então E[q_t | x_t] = E[y_t | x_t]: **o alvo q tem a mesma média do
rótulo e variância menor** (Rao-Blackwell). Para um aprendiz limitado por amostra, isso equivale a mais
dados. O oráculo **não** conhece τ. Conhecer o comprimento T é legítimo para ele (sob τ uniforme,
P(τ ≤ t | T) = t/T) e é justamente parte da informação que o torna melhor.

Diferença para o [passo 2 do roteiro](../roteiro-30-09/P2-professor-aluno.md), que deu nulo: aquele
professor conhecia τ, os negativos ficavam presos em 0 e metade do rótulo continuava duro. Não era
uma Rao-Blackwellização.

## O1: sonda do oráculo (12:00)

`scripts/o1_oraculo.py --sonda`: top-65 features × {t, t+50, t+200, fim} + log t, log T, log(T−t+1).
GBM treinado nos folds 1–4, avaliado no fold 0 (partição 42):

| Modelo | TS-AUC no fold 0 |
|---|---|
| E1 (aluno atual) | 0,6302 |
| **oráculo** | **0,7063** |

Features mais importantes: `log_resto`, `o50_mmd_*`, `o50_meta_t`, `log_t`, `oT_mmd_*`, `o200_cusum_age_var_up`.
O oráculo tem muito mais informação, então há espaço para a destilação.

## O2: aluno destilado (hipótese escrita ANTES de medir)

- **Alvos:** `scripts/o1_oraculo.py --alvos`, **nested cross-fit**. Para cada fold f do aluno, o oráculo
  é ajustado só nas séries fora de f, com CV interna de 4 partes dando q para essas séries. As séries de
  f nunca passam pelo oráculo que rotula o treino de f. Saída: `data/processed/train_rows_oraculo_s42.parquet`.
- **Aluno:** receita do E1 + `soft_label=true soft_label_modo=destilacao soft_label_mix=1.0`
  (alvo = q em todas as linhas, `cross_entropy`). Pesos, curva de taxa-base, feval e OOF seguem o `y`
  verdadeiro. K=4, partição 42, R0 contra o E1.
- **Previsão:** **+0,005 a +0,020**. O C1 mediu +0,0144 para ×2 de dados, e a redução de variância do
  alvo deve valer na mesma ordem.
- **Adotar** se ≥ +0,005 com IC excluindo 0 nas partições 42 (K=4) e 43 (K=2). **Descartar** se ≤ +0,002.
- **Checagem de vazamento:** o aluno não pode passar do oráculo nas mesmas séries. Se a destilação
  pura empatar, testar `mix=0,5` como exploratório.
