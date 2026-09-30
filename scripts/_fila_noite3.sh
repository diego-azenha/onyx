#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q 'FILA2 COMPLETA' artifacts/reports/fila2.log 2>/dev/null; do sleep 30; done
bash scripts/run_b0h_receita_historica.sh > artifacts/reports/b0h.log 2>&1
echo "FILA3 COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila3.log
