#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q 'FILA3 COMPLETA' artifacts/reports/fila3.log 2>/dev/null; do sleep 30; done
bash scripts/run_n1.sh 0.5 > artifacts/reports/n1.log 2>&1
echo "FILA4 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila4.log
