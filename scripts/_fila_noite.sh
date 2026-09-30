#!/usr/bin/env bash
# Fila da madrugada de 2026-09-30, um job pesado por vez: A1-metade -> C2 -> E1.
cd "$(dirname "$0")/.."
bash scripts/run_a1_metade.sh > artifacts/reports/a1h.log 2>&1
bash scripts/run_c2_pos_neg.sh > artifacts/reports/c2.log 2>&1
bash scripts/run_e1_extra_trees.sh > artifacts/reports/e1.log 2>&1
echo "FILA COMPLETA $(date +%H:%M:%S)" >> artifacts/reports/fila.log
