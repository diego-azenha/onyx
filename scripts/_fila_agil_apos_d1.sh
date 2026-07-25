#!/usr/bin/env bash
# Encadeia a fila agil apos o D1 na particao 43 terminar (um job pesado por vez).
set -u
cd "$(dirname "$0")/.."
until [ -f artifacts/models/oof_p3_lintree_partB_bag4.parquet ]; do sleep 60; done
sleep 20
exec bash scripts/run_fila_agil.sh
