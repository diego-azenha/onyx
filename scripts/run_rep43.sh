#!/usr/bin/env bash
# Réplica do E1 na partição 43: B0 e E1, sementes 777 e 101, mesma partição. RESUMÍVEL.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/rep43
R=data/processed/train_rows.parquet
for S in 777 101; do
  for V in b0 e1; do
    [ -f artifacts/models/oof_p43_${V}_s$S.parquet ] && continue
    EXTRA="--set extra_trees=false"; [ $V = e1 ] && EXTRA="--set extra_trees=true"
    echo "=== p43 $V semente $S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows $R --detectability-mode soft --linear-tree --fold-seed 43 --boost-seed $S $EXTRA \
      --out artifacts/models/p43_${V}_s$S > artifacts/reports/rep43/${V}_s$S.log 2>&1 || { echo FALHOU; exit 1; }
  done
done
for V in b0 e1; do
  python -u scripts/avg_oof.py --inputs artifacts/models/oof_p43_${V}_s{777,101}.parquet --out artifacts/models/oof_p43_${V}_bag2.parquet > /dev/null
  python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_p43_${V}_bag2.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_p43_${V}_vema_bag2.parquet > /dev/null
done
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_p43_b0_vema_bag2.parquet \
  --candidate artifacts/models/oof_p43_e1_vema_bag2.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_e1_p43.json > artifacts/reports/rep43/r0.log 2>&1
echo "=== REP43 COMPLETO @ $(date +%H:%M:%S) ==="
