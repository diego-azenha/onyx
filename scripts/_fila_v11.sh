#!/usr/bin/env bash
cd "$(dirname "$0")/.."
bash scripts/run_braco.sh v11_wtalign data/processed/train_rows.parquet train.py --wt-align > artifacts/reports/v11.log 2>&1
echo "FILA V11 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_v11.log
