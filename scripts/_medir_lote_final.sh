#!/usr/bin/env bash
# Mede com IC os tres braços que fecharam sem medicao formal: B1 (bag 4->7), C1 (contri=0,2) e
# B3 (o bundle de pos-processo escolhido pelo PIOR das duas particoes).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

# B3: materializa o OOF transformado nas duas particoes e mede.
for P in "b6c6_joint_bag4:42" "b6c6_partB_bag4:43"; do
  SRC="${P%%:*}"; TAG="${P##*:}"
  python -u scripts/apply_vema_oof.py --oof "artifacts/models/oof_${SRC}.parquet" \
    --a-up 1.0 --a-dn 0.2 --min-t 50 --out "artifacts/models/oof_b3_vema_p${TAG}.parquet"
  echo "=== B3 particao ${TAG} ==="
  python -u scripts/compare_oof.py --baseline "artifacts/models/oof_${SRC}.parquet" \
    --candidate "artifacts/models/oof_b3_vema_p${TAG}.parquet" \
    --n-boot 300 --n-jobs 4 --out "artifacts/reports/polimento_b3_p${TAG}.json"
done

echo "=== B1: bag 4 -> 7 sementes ==="
python -u scripts/compare_oof.py --baseline artifacts/models/oof_b6c6_joint_bag4.parquet \
  --candidate artifacts/models/oof_b6c6_joint_bag7.parquet \
  --n-boot 300 --n-jobs 4 --out artifacts/reports/polimento_b1_bag7.json

echo "=== C1: contri=0,2 ==="
python -u scripts/compare_oof.py --baseline artifacts/models/oof_b6c6_joint_bag2.parquet \
  --candidate artifacts/models/oof_p11_contri02_bag2.parquet \
  --n-boot 200 --n-jobs 4 --out artifacts/reports/polimento_p11_contri02.json
echo "=== LOTE FINAL COMPLETO @ $(date +%H:%M:%S) ==="
