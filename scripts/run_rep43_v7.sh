#!/usr/bin/env bash
# Réplica do V7 (E1 + lr 0,025, cap 3000, parada 200) na partição 43, contra o E1 da partição 43 (rep43).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/rep43_v7
for S in 777 101; do
  [ -f artifacts/models/oof_p43_v7_s$S.parquet ] && continue
  echo "=== p43 v7 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows data/processed/train_rows.parquet --detectability-mode soft --linear-tree --fold-seed 43 \
    --set extra_trees=true learning_rate=0.025 n_estimators_cap=3000 early_stopping_rounds=200 --boost-seed $S \
    --out artifacts/models/p43_v7_s$S > artifacts/reports/rep43_v7/v7_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
python -u scripts/avg_oof.py --inputs artifacts/models/oof_p43_v7_s{777,101}.parquet --out artifacts/models/oof_p43_v7_bag2.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_p43_v7_bag2.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_p43_v7_vema_bag2.parquet > /dev/null
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_p43_e1_vema_bag2.parquet \
  --candidate artifacts/models/oof_p43_v7_vema_bag2.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_v7_p43.json > artifacts/reports/rep43_v7/r0.log 2>&1
echo "=== REP43 V7 COMPLETO @ $(date +%H:%M:%S) ==="
