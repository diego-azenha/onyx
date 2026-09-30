# R1 — Refit com 100% das séries em produção

**Frente:** [teto offline](README.md) · **Status:** concluído: **inconclusivo, não adotado** · **Hipótese:** 2026-09-30 03:00

## Por quê

A produção (`adapter/platform.py:train`) funde os 5 boosters **de fold**, cada um treinado com 80% das
séries, × 7 sementes. O [C1](C1-curva-aprendizado-onyx.md) mostrou o Onyx limitado por amostra (×2
séries = +0,0144). Um modelo treinado com 100% das séries e o número médio de rodadas dos folds deveria
render algo como ×1,25 de dados. **O OOF nunca mede isso**, porque cada predição OOF vem de um modelo
de 80%.

## Desenho

`scripts/r1_refit.py`: holdout **externo** = séries com id % 5 == 0 (2.000). Nas outras 8.000, com a
receita do E1: (a) `train()` normal, média dos 5 boosters de fold, que é o que a produção faz; (b) refit
com todas as 8.000 séries, mesmos parâmetros (lidos do booster) e round(média das `best_iteration`)
rodadas; (c) a média de (a) e (b). TS-AUC no holdout, grade do board, sementes 777 e 101.

## Hipótese (escrita ANTES de medir)

- (b) − (a) entre **+0,002 e +0,006**, com o mesmo sinal nas duas sementes.
- Se confirmar: implementar `lightgbm.full_refit` em `train()` (off por default), ligar em produção e
  verificar a fusão (`verify_submission_notebook.py`).
- Risco: sem parada antecipada própria, o número de rodadas vira hiperparâmetro. A média das
  `best_iteration` foi calibrada com 80% dos dados e pode ser curta para 100%.

## Resultado

Holdout externo de 2.000 séries, grade do board, receita do E1:

| Semente | Média dos 5 folds (produção atual) | Refit 100% | Média dos dois | Rodadas do refit |
|---|---|---|---|---|
| 777 | 0,6269 | 0,6310 (+0,0041) | 0,6313 (+0,0044) | 71 |
| 101 | 0,6306 | 0,6253 (**−0,0053**) | 0,6295 (−0,0011) | 81 |

## Decisão

**Não adotado.** O sinal troca entre as sementes. Um booster único tem variância alta neste regime
(o mesmo fenômeno do [C1](C1-curva-aprendizado-onyx.md)): o ganho de ver 100% das séries fica menor
que a perda da média entre 5 modelos, e o saldo depende do sorteio. A variante 50/50 tem média de
+0,0016 nas duas sementes, o que não passa a barra.

O código fica pronto e **desligado**: `LightGBMConfig.full_refit` (default `False`) e
`model/train.py:refit_full`, chamado em `adapter/platform.py:train`. Para reabrir: K ≥ 4 no holdout
externo, ou refit bagged (vários refits com sementes diferentes), que ataca justamente a variância.
