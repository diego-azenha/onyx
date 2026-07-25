#!/usr/bin/env bash
# Espera o p10_combo terminar e leva a particao 43 a K=4 dos dois lados -- a unica comparacao que
# decide o D1 (ver HISTORICO.md §15.16: a "refutacao" anterior comparava K=4 contra K=2).
set -u
cd "$(dirname "$0")/.."
until ls artifacts/models/oof_p10_combo_bag2_*.parquet >/dev/null 2>&1; do sleep 45; done
sleep 15
exec bash scripts/run_d1_part43_k4.sh
