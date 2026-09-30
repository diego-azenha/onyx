#!/usr/bin/env bash
# Depois da fila 2: V7 (lr 0,025), V8 (min_data_in_leaf 800) e V4 (top-65, refeito com thin_weight).
cd "$(dirname "$0")/.."
until grep -q 'FILA V2 COMPLETA' artifacts/reports/fila_v2.log 2>/dev/null; do sleep 30; done
R=data/processed/train_rows.parquet
bash scripts/run_braco.sh v7_lr025 $R train.py --set learning_rate=0.025 n_estimators_cap=3000 early_stopping_rounds=200 > artifacts/reports/v7.log 2>&1
bash scripts/run_braco.sh v8_leaf800 $R train.py --set min_data_in_leaf=800 > artifacts/reports/v8.log 2>&1
bash scripts/run_braco.sh v4_top65 data/processed/train_rows_p1_top65.parquet train.py > artifacts/reports/v4.log 2>&1
echo "FILA V3 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_v3.log
