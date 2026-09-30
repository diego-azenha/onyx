#!/usr/bin/env bash
cd "$(dirname "$0")/.."
until grep -q 'v1_ff05 COMPLETO\|FALHOU' artifacts/reports/v1.log 2>/dev/null; do sleep 30; done
echo "fila_v continua em paralelo? NAO: a rep43 so roda depois da fila_v inteira" > /dev/null
until grep -q 'FILA V COMPLETA' artifacts/reports/fila_v.log 2>/dev/null; do sleep 30; done
bash scripts/run_rep43.sh > artifacts/reports/rep43.log 2>&1
