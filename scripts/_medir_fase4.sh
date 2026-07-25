#!/usr/bin/env bash
# Mede o que a fase 4 produz, na ordem em que fica pronto e SEM nunca abrir um terceiro job pesado.
#
#  1. D1 na particao 43 -- a replica sem a qual o D1 nao decide nada. A variancia de particao e
#     ~0,007, 2,6x o efeito medido (+0,0027), e este projeto ja adotou e retratou um braco em 24h
#     (a costura C5) por pular exatamente este passo.
#  2. `p10_combo` (D1 + A3b juntos) contra o D1 sozinho -- a pergunta e se o par ACRESCENTA, nao se
#     o par e positivo. A nao-aditividade ja apareceu tres vezes neste repo.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

# espera a medicao do D2/C1 sair, para nunca haver dois bootstraps + um treino
until [ -f artifacts/reports/polimento_triagem_p567.json ]; do sleep 60; done

until [ -f artifacts/models/oof_p3_lintree_partB_bag2.parquet ] \
   && [ -f artifacts/models/oof_b6c6_partB_bag2.parquet ]; do sleep 60; done
echo "=== D1 na particao 43 @ $(date +%H:%M:%S) ==="
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_b6c6_partB_bag2.parquet \
  --candidate artifacts/models/oof_p3_lintree_partB_bag2.parquet \
  --n-boot 300 --n-jobs 2 --out artifacts/reports/polimento_p3_lintree_partB.json

until [ -f artifacts/models/oof_p10_combo_bag2.parquet ]; do sleep 60; done
echo "=== par D1+A3b contra o D1 sozinho @ $(date +%H:%M:%S) ==="
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_p3_lintree_bag2.parquet \
  --candidate artifacts/models/oof_p10_combo_bag2.parquet \
  --n-boot 300 --n-jobs 2 --out artifacts/reports/polimento_p10_combo.json
echo "=== MEDICOES DA FASE 4 COMPLETAS @ $(date +%H:%M:%S) ==="
