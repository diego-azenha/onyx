#!/usr/bin/env bash
# B0h (knowledge/frentes/surpresa-acumulada/B0-incumbente.md, pendência): a receita da campanha que mediu
# 0,6312 -- sem as colunas mrep_ (187 features) e feature_contri_meta=0,6 -- nos mesmos dados e sementes do B0.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/b0h
for S in 777 101 202 303; do
  [ -f artifacts/models/oof_b0h_s$S.parquet ] && continue
  echo "=== b0h semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows data/processed/train_rows.parquet --detectability-mode soft --linear-tree \
    --drop-prefix mrep_ --feature-contri-meta 0.6 --boost-seed $S --out artifacts/models/b0h_s$S \
    > artifacts/reports/b0h/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
python -u scripts/avg_oof.py --inputs artifacts/models/oof_b0h_s{777,101,202,303}.parquet --out artifacts/models/oof_b0h_bag4.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_b0h_bag4.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_b0h_vema_bag4.parquet > /dev/null
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
  --candidate artifacts/models/oof_b0h_vema_bag4.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_b0h.json > artifacts/reports/b0h/r0.log 2>&1
echo "=== B0H COMPLETO @ $(date +%H:%M:%S) ==="
