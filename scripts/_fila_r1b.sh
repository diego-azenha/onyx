#!/usr/bin/env bash
# R1b: mais duas sementes do refit (espera o V11 terminar; um job pesado por vez)
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
until grep -q "COMPLETA" artifacts/reports/fila_v11.log 2>/dev/null; do sleep 30; done
for S in 202 303; do
  [ -f artifacts/reports/r1_refit/r1_s$S.json ] && continue
  python -u scripts/r1_refit.py --seed $S > artifacts/reports/r1_refit/log_s$S.txt 2>&1
done
echo "FILA R1B COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_r1b.log
