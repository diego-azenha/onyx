#!/usr/bin/env bash
# CAMPANHA DE POLIMENTO, fase 4 -- promover o D1 e medir o PAR (docs/RELATORIO_POLIMENTO.md §4b).
#
# Estado que motiva esta fase (triagem K=2, particao 42, grade do board, baseline 0,62640):
#   p3_lintree (D1)  +0,0025 geral, 150-400 +0,0043 IC EXCLUI 0  <- passa a regra do R0
#   p2_brw     (A3b) +0,0024 geral, sinal consistente por bucket
#
# Tres coisas a medir, nesta ordem de valor:
#
# 1. `p9_lintree_pac` -- linear_tree COM MAIS PACIENCIA. O D1 entrega +0,0025 com ~35% MENOS arvores
#    (46-77 contra 73-120) e logloss igual ou PIOR em 3 de 5 folds: o ganho e de ordenacao por
#    arvore, e a parada, que le `binary_logloss`, corta o braco antes do ponto util. O teto medido
#    ate agora e um piso. Ver HISTORICO.md §15.8.
#
# 2. `p3_lintree` K=4 + `p3_lintree_partB` -- a replica de particao, sem a qual nada entra
#    (a variancia de particao e ~0,007, 5x a barra).
#
# 3. `p10_combo` -- os DOIS juntos. Nao se pode supor aditividade: o precedente literal do repo e
#    "V5+mrep: os ganhos NAO SOMAM" (`scorer.py`) e o B6+C6 so somou ~70% (§14.9). Se o par nao bater
#    o melhor individual, o par nao entra.
#
# Rodar SOZINHO. RESUMIVEL. Ver NOTAS_AGENTES.md §7.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE

ROWS="data/processed/train_rows_bocpd.parquet"
BASE="--detectability-mode soft --feature-contri-meta 0.6"
DROP="--drop-prefix spec_ ord_ mrep_"
LOGDIR="artifacts/reports/polimento"; mkdir -p "$LOGDIR"

run_arm () {           # run_arm <nome> <sementes> <args...>
  local name="$1"; local seeds="$2"; shift 2
  for S in $seeds; do
    if [ -d "artifacts/models/${name}_s$S" ]; then echo "skip ${name}_s$S"; continue; fi
    echo "=== INICIO ${name} semente $S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows "$ROWS" $BASE --boost-seed "$S" \
      --out "artifacts/models/${name}_s$S" "$@" > "$LOGDIR/${name}_s$S.log" 2>&1 \
      || { echo "FALHOU ${name} s$S"; return 1; }
    echo "=== FIM    ${name} semente $S @ $(date +%H:%M:%S) ==="
  done
  local inputs="" ; local k=0
  for S in $seeds; do inputs="$inputs artifacts/models/oof_${name}_s$S.parquet"; k=$((k+1)); done
  # O nome do bag leva as SEMENTES, nao so o K: chamar o mesmo braco duas vezes com 2 sementes
  # gravava `bag2` duas vezes e a segunda sobrescrevia a primeira (aconteceu com o p2_brw).
  local tag; tag=$(echo $seeds | tr -d ' ')
  python scripts/avg_oof.py --inputs $inputs --out "artifacts/models/oof_${name}_bag${k}_${tag}.parquet" \
    > "$LOGDIR/${name}_avg_${tag}.log" 2>&1
  echo "=== BAG${k} PRONTO ${name} (${tag}) @ $(date +%H:%M:%S) ==="
}

# 1. [REMOVIDO -- HIPOTESE REFUTADA, 2026-07-25] `p9_lintree_pac` (linear_tree com paciencia 300).
#    A hipotese era que a parada por logloss cortava o D1 cedo, porque ele entrega +0,0025 com ~35%
#    MENOS arvores. MEDIDO: com paciencia 300 o braco treina 200 rodadas a mais (373,376,366,377,346
#    contra 173,176,166,177,146) e a MELHOR ITERACAO nao muda -- arvores/fold identicas (73,76,66,
#    77,46) e logloss identica. O modelo resultante e bit-identico ao `p3_lintree`.
#    O otimo de logloss ocorre de fato ali; mais paciencia nao acha nada. O que NAO fica descartado e
#    a regua: a TS-AUC pode seguir melhorando alem do otimo de logloss -- mas testar isso exige
#    RODADAS FIXAS (o desenho do X0), nao mais paciencia.
# 2. promocao do D1
run_arm p3_lintree     "202 303"  $DROP --linear-tree                             || exit 1
run_arm p3_lintree_partB "777 101" $DROP --linear-tree --fold-seed 43              || exit 1
# 3. o par
run_arm p10_combo      "777 101"  $DROP --linear-tree --base-rate-weighted         || exit 1

echo "=== FASE 4 COMPLETA @ $(date +%H:%M:%S) ==="
