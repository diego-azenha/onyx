#!/usr/bin/env bash
# Fila de variância sobre o E1 (2026-09-30), um job pesado por vez.
cd "$(dirname "$0")/.."
until grep -q 'FILA COMPLETA' artifacts/reports/fila.log 2>/dev/null; do sleep 30; done
R=data/processed/train_rows.parquet
bash scripts/run_braco.sh v1_ff05 $R train.py --set feature_fraction=0.5 > artifacts/reports/v1.log 2>&1
N1_FRAC=0.5 bash scripts/run_braco.sh v2_n1 $R n1_treino.py > artifacts/reports/v2.log 2>&1
bash scripts/run_braco.sh v3_hist $R train.py --drop-prefix mrep_ --feature-contri-meta 0.6 > artifacts/reports/v3.log 2>&1
bash scripts/run_braco.sh v4_top65 data/processed/train_rows_p1_top65.parquet train.py > artifacts/reports/v4.log 2>&1
echo "FILA V COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila_v.log
