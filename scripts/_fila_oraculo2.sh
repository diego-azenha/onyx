#!/usr/bin/env bash
# Oráculo v2 (todas as features em t): alvos nested na partição 42 e depois uma semente (777) do aluno com
# mix 1,0 e com mix 0,5, para escolher antes de gastar K=4.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
[ -f data/processed/train_rows_oraculo_v2_s42.parquet ] || python -u scripts/o1_oraculo.py --alvos --todas-em-t --tag _v2 > artifacts/oraculo/alvos_v2_s42.log 2>&1 || { echo FALHOU alvos; exit 1; }
R=data/processed/train_rows_oraculo_v2_s42.parquet
for MIX in 1.0 0.5; do
  NOME=o3_mix$(echo $MIX | tr -d .)
  mkdir -p artifacts/reports/$NOME
  [ -f artifacts/models/oof_${NOME}_s777.parquet ] && continue
  echo "=== $NOME 777 @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows $R --detectability-mode soft --linear-tree --boost-seed 777 \
    --set extra_trees=true soft_label=true soft_label_modo=destilacao soft_label_mix=$MIX \
    --out artifacts/models/${NOME}_s777 > artifacts/reports/$NOME/train_s777.log 2>&1 || { echo FALHOU $NOME; exit 1; }
done
echo "FILA ORACULO2 COMPLETA $(date +%H:%M:%S)"
