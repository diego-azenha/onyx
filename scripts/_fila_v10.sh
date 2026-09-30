#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q 'S6 COMPLETO\|FALHOU' artifacts/reports/s6.log 2>/dev/null; do sleep 30; done
bash scripts/run_braco.sh v10_abs data/processed/train_rows_abs.parquet train.py > artifacts/reports/v10.log 2>&1
echo "FILA V10 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_v10.log
