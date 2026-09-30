#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q 'FILA COMPLETA' artifacts/reports/fila.log 2>/dev/null; do sleep 30; done
bash scripts/run_p1_poda.sh 65 > artifacts/reports/p1.log 2>&1
echo "FILA2 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila2.log
