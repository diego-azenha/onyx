#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
L=artifacts/reports/fila_i8.log
until grep -q "FILA I36 COMPLETA" artifacts/reports/fila_i36.log 2>/dev/null; do sleep 30; done
[ -f data/processed/train_rows_f3.parquet ] || python -u scripts/i8_faixas.py --split > artifacts/reports/i8_split.log 2>&1
for k in 0 1 2 3; do
  [ -f artifacts/models/oof_i8f${k}_s777.parquet ] && continue
  python -u scripts/train.py --rows data/processed/train_rows_f$k.parquet --detectability-mode soft --linear-tree \
    --boost-seed 777 --out artifacts/models/i8f${k}_s777 --set extra_trees=true > artifacts/reports/i8f${k}_s777.log 2>&1
  echo "f$k $(date +%H:%M) $?" >> $L
done
python -u scripts/i8_faixas.py --junta 777 >> $L 2>&1
echo "FILA I8 COMPLETA $(date +%H:%M)" >> $L
