#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/s6
for T in desce sobe fixa; do
  for S in 777 101; do
    [ -f artifacts/models/oof_s6_${T}_s$S.parquet ] && continue
    echo "=== s6 $T $S @ $(date +%H:%M:%S) ==="
    S6_TIPO=$T python -u scripts/s6_especialistas.py --rows data/processed/train_rows.parquet --detectability-mode soft \
      --linear-tree --set extra_trees=true --boost-seed $S --out artifacts/models/s6_${T}_s$S > artifacts/reports/s6/${T}_s$S.log 2>&1 || { echo FALHOU $T $S; exit 1; }
  done
done
echo "=== S6 COMPLETO @ $(date +%H:%M:%S) ==="
