#!/usr/bin/env bash
# Resolve o D1 com a MESMA precisao nas duas particoes.
#
# O erro que isto corrige: a particao 42 foi medida em K=4 (+0,0027, IC [+0,0002; +0,0049]) e a 43
# em K=2 (-0,0019, SEM IC). Comparar precisoes diferentes e depois reverter pela mais fraca e
# assimetrico -- o ruido de semente que o bootstrap pareado NAO ve e ~0,0041 por modelo de semente
# unica (NOTAS_AGENTES.md §5), o que da ~0,004 na diferenca de dois bags K=2. Logo -0,0019 e
# compativel com o efeito verdadeiro ser +0,002: aquilo era INCONCLUSIVO, nao refutacao.
#
# Aqui a particao 43 vai a K=4 dos dois lados (candidato E baseline), que e a unica comparacao que
# decide alguma coisa.
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

run () {  # run <nome> <args...>
  local name="$1"; shift
  for S in 202 303; do
    [ -d "artifacts/models/${name}_s$S" ] && { echo "skip ${name}_s$S"; continue; }
    echo "=== INICIO ${name} s$S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows "$ROWS" $BASE --boost-seed "$S" --fold-seed 43 \
      --out "artifacts/models/${name}_s$S" "$@" > "$LOGDIR/${name}_s$S.log" 2>&1 \
      || { echo "FALHOU ${name} s$S"; return 1; }
    echo "=== FIM    ${name} s$S @ $(date +%H:%M:%S) ==="
  done
  python scripts/avg_oof.py --inputs \
    artifacts/models/oof_${name}_s777.parquet artifacts/models/oof_${name}_s101.parquet \
    artifacts/models/oof_${name}_s202.parquet artifacts/models/oof_${name}_s303.parquet \
    --out "artifacts/models/oof_${name}_bag4.parquet" > "$LOGDIR/${name}_avg4.log" 2>&1
  echo "=== BAG4 PRONTO ${name} @ $(date +%H:%M:%S) ==="
}

run b6c6_partB      $DROP                  || exit 1   # baseline da 43 a K=4
run p3_lintree_partB $DROP --linear-tree   || exit 1   # candidato da 43 a K=4

echo "=== D1 PARTICAO 43 EM K=4 PRONTO @ $(date +%H:%M:%S) ==="
