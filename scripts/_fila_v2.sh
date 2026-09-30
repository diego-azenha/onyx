#!/usr/bin/env bash
# Depois da réplica na partição 43: R1-refit (produção) -> V5 -> V6. Um job pesado por vez.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
until grep -q 'REP43 COMPLETO\|FALHOU' artifacts/reports/rep43.log 2>/dev/null; do sleep 30; done
mkdir -p artifacts/reports/r1_refit
for S in 777 101; do
  [ -f artifacts/reports/r1_refit/r1_s$S.json ] || python -u scripts/r1_refit.py --seed $S > artifacts/reports/r1_refit/log_s$S.txt 2>&1
done
echo "=== R1 COMPLETO $(date +%H:%M:%S)" >> artifacts/reports/fila_v2.log
bash scripts/run_braco.sh v5_bag05 data/processed/train_rows.parquet train.py --set bagging_fraction=0.5 > artifacts/reports/v5.log 2>&1
bash scripts/run_braco_sem_lt.sh > artifacts/reports/v6.log 2>&1
echo "FILA V2 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_v2.log
