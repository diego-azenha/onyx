#!/usr/bin/env bash
# D1 promovido a K=4 na particao 42 -- a medicao que decide o unico candidato vivo da campanha.
# Encadeado apos a medicao do p1_deploy193 para nunca haver tres jobs pesados ao mesmo tempo
# (a contencao ja deadlocou o LightGBM uma vez -- NOTAS_AGENTES.md §7).
#
# Contexto que torna esta medicao a que importa: o A3b estava em +0,0024 com K=2 e caiu para +0,0010
# com K=4. O D1 esta em +0,0025 com K=2. Ate este numero sair, o D1 nao e um resultado.
set -u
cd "$(dirname "$0")/.."
until [ -f artifacts/reports/polimento_p1_deploy193.json ]; do sleep 30; done
sleep 10
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_b6c6_joint_bag4.parquet \
  --candidate artifacts/models/oof_p3_lintree_bag4.parquet \
  --n-boot 300 --n-jobs 2 --out artifacts/reports/polimento_p3_lintree_k4.json
