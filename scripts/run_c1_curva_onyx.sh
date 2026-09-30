#!/usr/bin/env bash
# C1 (knowledge/frentes/teto-offline/C1-curva-aprendizado-onyx.md): o Onyx é limitado por amostra?
# Treina a receita do B0 em METADE das séries (ids pares) e compara, nas mesmas séries, com o B0 (todas).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/c1
ROWS=data/processed/train_rows_metade_par.parquet
[ -f $ROWS ] || python -u -c "
import pandas as pd, pyarrow.parquet as pq
t=pq.read_table('data/processed/train_rows.parquet', filters=[('id','in',list(range(0,10000,2)))])
t.to_pandas().to_parquet('$ROWS'); print('ok')
"
for S in 777 101; do
  [ -f artifacts/models/oof_c1_metade_s$S.parquet ] && continue
  echo "=== c1 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows $ROWS --detectability-mode soft --linear-tree --boost-seed $S \
    --out artifacts/models/c1_metade_s$S > artifacts/reports/c1/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
echo "=== C1 COMPLETO @ $(date +%H:%M:%S) ==="
