#!/usr/bin/env bash
# Espera os alvos do oráculo (partição 42) e roda o aluno destilado (mix 1,0), K=4, R0 contra o E1.
cd "$(dirname "$0")/.."
until grep -q 'gravado\|Traceback' artifacts/oraculo/alvos_s42.log 2>/dev/null; do sleep 20; done
grep -q Traceback artifacts/oraculo/alvos_s42.log && { echo "FALHOU alvos"; exit 1; }
bash scripts/run_braco.sh o2_destilado data/processed/train_rows_oraculo_s42.parquet train.py \
  --set soft_label=true soft_label_modo=destilacao soft_label_mix=1.0 > artifacts/reports/o2.log 2>&1
echo "FILA ORACULO COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_oraculo.log
