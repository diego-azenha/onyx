#!/usr/bin/env bash
# Mede tudo que ficou pendente, em serie, com nomes de arquivo corretos.
#   1. o par D1+A3b contra o D1 sozinho (a pergunta e se ACRESCENTA)
#   2. D1 na particao 43 a K=4 dos DOIS lados -- a comparacao que decide o D1
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

echo "=== par D1+A3b vs D1 sozinho @ $(date +%H:%M:%S) ==="
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_p3_lintree_bag2.parquet \
  --candidate artifacts/models/oof_p10_combo_bag2_777101.parquet \
  --n-boot 200 --n-jobs 2 --out artifacts/reports/polimento_p10_combo.json

until [ -f artifacts/models/oof_p3_lintree_partB_bag4.parquet ] \
   && [ -f artifacts/models/oof_b6c6_partB_bag4.parquet ]; do sleep 60; done
sleep 15
echo "=== D1 na PARTICAO 43, K=4 dos dois lados @ $(date +%H:%M:%S) ==="
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_b6c6_partB_bag4.parquet \
  --candidate artifacts/models/oof_p3_lintree_partB_bag4.parquet \
  --n-boot 300 --n-jobs 2 --out artifacts/reports/polimento_p3_lintree_partB_k4.json
echo "=== MEDICOES PENDENTES COMPLETAS @ $(date +%H:%M:%S) ==="
