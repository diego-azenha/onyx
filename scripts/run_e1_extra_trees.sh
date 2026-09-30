#!/usr/bin/env bash
# E1 (knowledge/frentes/teto-offline/E1-extra-trees.md): receita do B0 + extra_trees. RESUMÍVEL.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/e1
for S in 777 101 202 303; do
  [ -f artifacts/models/oof_e1_s$S.parquet ] && continue
  echo "=== e1 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows data/processed/train_rows.parquet --detectability-mode soft --linear-tree \
    --set extra_trees=true --boost-seed $S --out artifacts/models/e1_s$S > artifacts/reports/e1/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
python -u scripts/avg_oof.py --inputs artifacts/models/oof_e1_s{777,101,202,303}.parquet --out artifacts/models/oof_e1_bag4.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_e1_bag4.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_e1_vema_bag4.parquet > /dev/null
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
  --candidate artifacts/models/oof_e1_vema_bag4.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_e1.json > artifacts/reports/e1/r0.log 2>&1
echo "=== E1 COMPLETO @ $(date +%H:%M:%S) ==="
