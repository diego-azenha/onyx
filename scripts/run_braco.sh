#!/usr/bin/env bash
# Braço genérico da frente teto-offline: receita do B0 + extra_trees (E1) + flags extras, K=4, v-EMA,
# R0 contra o E1 (nova base) e contra o B0. RESUMÍVEL.
# Uso: bash scripts/run_braco.sh <nome> <rows> <treinador.py> [flags extras do train.py...]
#      variáveis de ambiente (ex.: N1_FRAC) passam para o treinador.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
NOME="$1"; ROWS="$2"; TREINADOR="$3"; shift 3
mkdir -p artifacts/reports/$NOME
# funde o --set do braço com extra_trees=true (um segundo --set sobrescreveria o primeiro: nargs="*")
ARGS=(); TEM_SET=0
for a in "$@"; do ARGS+=("$a"); if [ "$a" = "--set" ]; then ARGS+=("extra_trees=true"); TEM_SET=1; fi; done
[ $TEM_SET -eq 1 ] || ARGS=(--set extra_trees=true "${ARGS[@]}")
for S in 777 101 202 303; do
  [ -f artifacts/models/oof_${NOME}_s$S.parquet ] && continue
  echo "=== $NOME semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/$TREINADOR --rows $ROWS --detectability-mode soft --linear-tree \
    --boost-seed $S --out artifacts/models/${NOME}_s$S "${ARGS[@]}" > artifacts/reports/$NOME/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
python -u scripts/avg_oof.py --inputs artifacts/models/oof_${NOME}_s{777,101,202,303}.parquet --out artifacts/models/oof_${NOME}_bag4.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_${NOME}_bag4.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_${NOME}_vema_bag4.parquet > /dev/null
for B in e1 b0; do
  OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_${B}_vema_bag4.parquet \
    --candidate artifacts/models/oof_${NOME}_vema_bag4.parquet --n-boot 300 --n-jobs 4 --out artifacts/reports/r0_${NOME}_vs_$B.json > artifacts/reports/$NOME/r0_vs_$B.log 2>&1
done
echo "=== $NOME COMPLETO @ $(date +%H:%M:%S) ==="
