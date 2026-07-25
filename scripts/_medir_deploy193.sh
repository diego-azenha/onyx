#!/usr/bin/env bash
# A3 (v): mede o conjunto REALMENTE IMPLANTADO (193 features = V4 183 + mrep 6 + bocpd 4, o que
# `default_blocks()` emite) contra o conjunto em que o incumbente foi MEDIDO (187, sem `mrep_`).
# Ver HISTORICO.md §15.3. Espera a reauditoria sair para nao virar o terceiro job pesado.
set -u
cd "$(dirname "$0")/.."
until [ -f artifacts/reports/a2_reaudit_focado.json ]; do sleep 30; done
sleep 10
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_b6c6_joint_bag2.parquet \
  --candidate artifacts/models/oof_p1_deploy193_bag2.parquet \
  --n-boot 200 --n-jobs 2 --out artifacts/reports/polimento_p1_deploy193.json
