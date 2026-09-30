#!/usr/bin/env bash
# N1: receita do B0 com negativos subamostrados no treino. Uso: bash scripts/run_n1.sh 0.5
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE N1_FRAC="$1"
NOME=n1_f$(echo $1 | tr -d .); mkdir -p artifacts/reports/$NOME
for S in 777 101 202 303; do
  [ -f artifacts/models/oof_${NOME}_s$S.parquet ] && continue
  echo "=== $NOME semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/n1_treino.py --rows data/processed/train_rows.parquet --detectability-mode soft --linear-tree \
    --boost-seed $S --out artifacts/models/${NOME}_s$S > artifacts/reports/$NOME/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
python -u scripts/avg_oof.py --inputs artifacts/models/oof_${NOME}_s{777,101,202,303}.parquet --out artifacts/models/oof_${NOME}_bag4.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_${NOME}_bag4.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_${NOME}_vema_bag4.parquet > /dev/null
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
  --candidate artifacts/models/oof_${NOME}_vema_bag4.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_$NOME.json > artifacts/reports/$NOME/r0.log 2>&1
echo "=== $NOME COMPLETO @ $(date +%H:%M:%S) ==="
