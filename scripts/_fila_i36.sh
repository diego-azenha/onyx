#!/usr/bin/env bash
# Fila 01/10: I3 (nulo aprendido) e I6 (padronização por t). Um job pesado por vez.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
L=artifacts/reports/fila_i36.log
until grep -q "I2 FIM" artifacts/i2/log_K50.txt 2>/dev/null; do sleep 30; done
echo "inicio $(date +%H:%M)" >> $L
[ -f data/processed/train_rows_i3.parquet ] || python -u scripts/i3_nulo_aprendido.py > artifacts/reports/i3_build.log 2>&1
echo "i3 build $(date +%H:%M) $?" >> $L
[ -f data/processed/train_rows_i6.parquet ] || python -u scripts/i6_padroniza_por_t.py > artifacts/reports/i6_build.log 2>&1
echo "i6 build $(date +%H:%M) $?" >> $L
for A in i3 i6; do
  [ -f artifacts/models/oof_${A}_s777.parquet ] && continue
  python -u scripts/train.py --rows data/processed/train_rows_$A.parquet --detectability-mode soft --linear-tree \
    --boost-seed 777 --out artifacts/models/${A}_s777 --set extra_trees=true > artifacts/reports/${A}_s777.log 2>&1
  echo "$A s777 $(date +%H:%M) $?" >> $L
done
echo "FILA I36 COMPLETA $(date +%H:%M)" >> $L
