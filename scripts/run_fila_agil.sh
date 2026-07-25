#!/usr/bin/env bash
# Fila reordenada por RETORNO ESPERADO POR HORA DE MAQUINA (2026-07-25).
#
# Nao e "barato primeiro" nem "maior teto primeiro". Neste projeto teto e custo estao
# CORRELACIONADOS -- os knobs baratos sao baratos por nao mudarem o que o modelo VE, e e ai que o
# E0c diz nao haver folga. A rodada de hoje confirmou de forma quase caricata:
#
#   max_bin 1023      -0,0004     media em logit    +0,00001
#   passos frios      +0,00004    capacidade (D3)   +0,0012
#
# Quatro itens baratos, quatro quase-zeros. O unico com sinal (D1, +0,0027 na particao 42) e de
# custo medio e o unico com historia mecanica. Ordenar por custo puro selecionaria justamente os
# quatro de cima -- e o `RELATORIO_EXP0` §6 ja tinha escrito a versao disso para risco: "ordenar por
# risco seleciona sistematicamente as opcoes incapazes de atingir o alvo".
#
# Criterio adotado: BARATO **E COM MECANISMO** primeiro; depois caro-com-teto; knobs de `g`
# rebaixados independentemente do custo.
#
# Rodar SOZINHO (um job pesado por vez -- NOTAS_AGENTES.md §7). RESUMIVEL.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE
LOGDIR="artifacts/reports/polimento"; mkdir -p "$LOGDIR"
ROWS="data/processed/train_rows_bocpd.parquet"
BASE="--detectability-mode soft --feature-contri-meta 0.6 --drop-prefix spec_ ord_ mrep_"

# ---------------------------------------------------------------- 1. B3 (zero treino, ~30 min)
# O gate do A1 (v-EMA em t>150) foi escolhido com a justificativa "+0,0009/+0,0012 em 150-400 / t>400,
# perde -0,0014 em t<=50" -- pesos que o A2 mostrou errados: t>400 vale 31,9% (nao 16,5%) e t<=50
# vale 3,9% (nao 8,1%). O otimo do gate MUDA por construcao. Unico item da lista sem treino nenhum.
if [ ! -f artifacts/reports/b3_bundle.csv ]; then
  echo "=== B3 bundle de pos-processo @ $(date +%H:%M:%S) ==="
  PYTHONPATH=scripts python -u scripts/b3_bundle_postproc.py \
    --oof-a artifacts/models/oof_b6c6_joint_bag4.parquet \
    --oof-b artifacts/models/oof_b6c6_partB_bag2.parquet \
    > "$LOGDIR/b3_bundle.log" 2>&1 || echo "FALHOU B3"
  echo "=== B3 PRONTO @ $(date +%H:%M:%S) ==="
fi

# ---------------------------------------------------------------- 1b. C1 estendido: contri=0,2
# A varredura do B6 fechou com um GRADIENTE MONOTONO na particao 42 (K=2):
#   contri 0,4  +0,0017   |   0,6 (atual)  --   |   0,8  -0,0013
# Monotono na direcao "penalizar `meta_h0` MAIS", que e exatamente o que o mecanismo do B6 preve:
# essas colunas descrevem a serie inteira e o modelo as usava como quase-intercepto por serie, o que
# nao ajuda uma AUC calculada DENTRO de cada passo. Tres pontos alinhados com o mecanismo e melhor
# evidencia que um braco isolado -- entao vale ver onde a curva vira.
for S in 777 101; do
  [ -d "artifacts/models/p11_contri02_s$S" ] && { echo "skip p11_contri02_s$S"; continue; }
  echo "=== INICIO C1 contri=0,2 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$ROWS" $BASE --boost-seed "$S" --feature-contri-meta 0.2     --out "artifacts/models/p11_contri02_s$S" > "$LOGDIR/p11_contri02_s$S.log" 2>&1     || { echo "FALHOU p11 s$S"; exit 1; }
  echo "=== FIM    C1 contri=0,2 semente $S @ $(date +%H:%M:%S) ==="
done
python scripts/avg_oof.py --inputs   artifacts/models/oof_p11_contri02_s777.parquet artifacts/models/oof_p11_contri02_s101.parquet   --out artifacts/models/oof_p11_contri02_bag2.parquet > "$LOGDIR/p11_avg.log" 2>&1
echo "=== C1 contri=0,2 PRONTO @ $(date +%H:%M:%S) ==="

# ---------------------------------------------------------------- 2. B1 (3 treinos, ~30 min)
# O UNICO item da lista com efeito CONHECIDO e nao apenas esperado: a curva de bagging foi medida na
# campanha X (K=4 capta 83% do ganho; satura em K=7). Diferencial previsto +0,0008 a +0,0010, so
# compute, sem rebuild e sem risco de conjunto de features.
for S in 404 505 606; do
  [ -d "artifacts/models/b6c6_joint_s$S" ] && { echo "skip b6c6_joint_s$S"; continue; }
  echo "=== INICIO B1 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$ROWS" $BASE --boost-seed "$S" \
    --out "artifacts/models/b6c6_joint_s$S" > "$LOGDIR/b1_bag7_s$S.log" 2>&1 \
    || { echo "FALHOU B1 s$S"; exit 1; }
  echo "=== FIM    B1 semente $S @ $(date +%H:%M:%S) ==="
done
python scripts/avg_oof.py --inputs \
  artifacts/models/oof_b6c6_joint_s777.parquet artifacts/models/oof_b6c6_joint_s101.parquet \
  artifacts/models/oof_b6c6_joint_s202.parquet artifacts/models/oof_b6c6_joint_s303.parquet \
  artifacts/models/oof_b6c6_joint_s404.parquet artifacts/models/oof_b6c6_joint_s505.parquet \
  artifacts/models/oof_b6c6_joint_s606.parquet \
  --out artifacts/models/oof_b6c6_joint_bag7.parquet > "$LOGDIR/b1_bag7_avg.log" 2>&1
echo "=== B1 bag7 PRONTO @ $(date +%H:%M:%S) ==="

echo "=== FILA AGIL COMPLETA @ $(date +%H:%M:%S) ==="
echo "proximo, por teto e nao por custo: A6 (scripts/run_a6.sh) e depois E1 (des-thinning)."
