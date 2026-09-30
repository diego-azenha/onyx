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

## O2 (v1): falhou, e o motivo é teórico (12:10)

Aluno destilado com o oráculo v1 (mix 1,0), semente 777: **0,6303 → 0,6248 (−0,0055)**. Interrompido
depois da 1ª semente.

**Diagnóstico:** a Rao-Blackwellização só garante E[q|x] = E[y|x] se a informação do oráculo **contém** a
do aluno. O oráculo v1 via só as **65** colunas principais em t, e o aluno vê **194**. Nas ~130 restantes,
q "esquece" o que o rótulo carregava e puxa o aluno para um alvo mais pobre. Sinal lateral: o aluno
destilado parou antes (best ≈ 30–48 árvores, contra ≈ 65–93 do E1), porque a parada antecipada mede a
logloss contra o `y` verdadeiro.

## O3: oráculo v2 (todas as 194 colunas em t + snapshots futuros das top-65)

Sonda no fold 0: **0,7117** (v1: 0,7063; E1: 0,6302). Alvos nested gerados por
`scripts/o1_oraculo.py --alvos --todas-em-t --tag _v2`. Antes de K=4, uma semente (777) com mix 1,0 e
outra com mix 0,5 (`scripts/_fila_oraculo2.sh`). Regra: só vai para K=4 a variante com Δ ≥ +0,003 na
semente 777 contra o E1 (+0,0000 é o E1 da mesma semente, 0,6303).

### Resultado O3 (12:55)

Oráculo v2, TS-AUC interno por fold: 0,7143 · 0,7052 · 0,7132 · 0,7064 · 0,7149.

| Aluno (semente 777, pareado com o E1 da mesma semente, 0,6303) | TS-AUC | Δ |
|---|---|---|
| destilação pura (mix 1,0), oráculo v1 (top-65 em t) | 0,6248 | **−0,0055** |
| destilação pura (mix 1,0), oráculo v2 (194 em t) | 0,6258 | **−0,0045** |
| mistura 0,5, oráculo v2 | 0,6330 | +0,0027 |

## Decisão

**A frente fecha sem K=4.** Pela regra registrada, só a variante com ≥ +0,003 na semente 777 iria para
K=4; a melhor deu +0,0027. Todo ganho por semente dessa ordem sumiu no bag até agora (V7, passo 2), e
a destilação pura **piora**.

**Por que a teoria falhou:** a Rao-Blackwellização supõe um q **exato**. Aqui o oráculo é estimado com
os **mesmos ~5 mil eventos escassos** que limitam o aluno. O erro de estimação dele se correlaciona com
as features do aluno, e o aluno aprende esse viés junto com a redução de variância. A condição de
conter a informação do aluno (v2) melhorou pouco (−0,0055 → −0,0045). **Com o mesmo dado, um professor
não é mais confiável que o rótulo.** O ganho de uma destilação de verdade exige um professor com
informação **externa** aos eventos (outro dado, um prior estrutural). Nenhum dos candidatos testados
serviu (2025, transplante, modelo de fundação).

O código fica pronto e **desligado**: `soft_label_modo="destilacao"`, `model/oraculo.py` e o gancho em
`adapter/platform.py`.

**Pendência de maior valor:** o E1 **não foi submetido**. Este clone não tem o token da Crunch, e a
submissão fica com o usuário. As âncoras OOF → placar são de julho; uma submissão do E1 no release atual
(234) diz se a distância ao pelotão (0,66–0,675) é de fato ~0,03.
