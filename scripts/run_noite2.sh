#!/usr/bin/env bash
# FILA NOTURNA -- PARTE 2 (2026-07-26): os dois bracos do Grupo 3 que exigiam CODIGO NOVO.
#
# `run_noite.sh` cobre tudo o que ja tinha suporte no `train.py`. Estes dois nao tinham:
#
#   C1(ii) conjunto penalizado estendido -> `--contri-extra-cols` (casamento por NOME EXATO; ver
#          config.py: o prefixo pegaria `conformal_logm_abs_reset`, que e acumulador de evidencia).
#   C2(ii) `delta_mean` no `d_i`         -> `--detectability-path` + CSV gerado com
#          `detectability_report.py --extra-axes delta_mean_e`.
#
# Ambas as adicoes sao NO-OP por default, provado em `tests/unit/test_feature_contri.py` -- condicao
# necessaria para nao contaminar os bracos que a parte 1 ainda esta medindo.
#
# ESPERA a parte 1 liberar o lock (nao mata nada, nao duplica carga). Teto de 8 h.
set -u
cd "$(dirname "$0")/.."

L="artifacts/reports/polimento"
LOCK="artifacts/reports/.run_resto.lock"
mkdir -p "$L"

esperou=0
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do
  [ "$esperou" -ge 86400 ] && { echo "timeout de 24 h esperando o lock; abortando."; exit 1; }
  sleep 60; esperou=$((esperou + 60))
done
echo "lock livre apos $((esperou / 60)) min"
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=scripts

ROWS="data/processed/train_rows_bocpd.parquet"
DROP="spec_ ord_ mrep_"
BL2="artifacts/models/oof_b6c6_joint_bag2.parquet"

banner () { echo; echo "########## $* @ $(date +%H:%M:%S) ##########"; }

treina_seed () {
  local nome="$1" S="$2"; shift 2
  [ -d "artifacts/models/${nome}_s$S" ] && { echo "skip ${nome}_s$S"; return 0; }
  echo "=== INICIO ${nome} s$S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$ROWS" --boost-seed "$S" \
    --out "artifacts/models/${nome}_s$S" "$@" > "$L/${nome}_s$S.log" 2>&1 \
    || { echo "FALHOU ${nome} s$S"; return 1; }
  echo "=== FIM ${nome} s$S @ $(date +%H:%M:%S) ==="
}

braco_k2 () {
  local nome="$1"; shift
  treina_seed "$nome" 777 "$@" || return 1
  treina_seed "$nome" 101 "$@" || return 1
  local out="artifacts/models/oof_${nome}_bag2.parquet"
  [ -f "$out" ] || python scripts/avg_oof.py --inputs \
    "artifacts/models/oof_${nome}_s777.parquet" "artifacts/models/oof_${nome}_s101.parquet" \
    --out "$out" > "$L/${nome}_avg2.log" 2>&1 || { echo "FALHOU media ${nome}"; return 1; }
  [ -f "artifacts/reports/polimento_${nome}.json" ] && { echo "skip medicao ${nome}"; return 0; }
  echo "=== MEDINDO ${nome} @ $(date +%H:%M:%S) ==="
  python -u scripts/compare_oof.py --baseline "$BL2" --candidate "$out" \
    --n-boot 200 --n-jobs 2 --out "artifacts/reports/polimento_${nome}.json" \
    > "$L/medir_${nome}.log" 2>&1 || echo "FALHOU medicao ${nome}"
}

banner "FILA NOTURNA PARTE 2 -- INICIO"

# ============================================================ 9. C1(ii) conjunto penalizado estendido
# "O maior ganho recente do projeto (+0,0036) com metade da hipotese por testar": o B6 varreu o VALOR
# do contri e nunca o CONJUNTO. Penaliza os dois rastreadores de `t` que o xs-SHAP expos, com o mesmo
# 0,6 que funcionou para as `meta_h0_*` -- e o valor com evidencia, nao um chute novo.
banner "9  C1(ii) conjunto penalizado estendido"
braco_k2 p22_contri_ext --detectability-mode soft --feature-contri-meta 0.6 --drop-prefix $DROP \
  --contri-extra-cols conformal_logm_abs mmd_joint_slow_cal --contri-extra 0.6 \
  --set linear_tree=false

# ============================================================ 10. C2(ii) delta_mean no d_i
# PRIOR NEGATIVO, registrado: o cabecalho de model/detectability.py diz que o canal de media foi
# "medido como fraco" e por isso ficou fora de AXES. O braco existe para MEDIR, nao para presumir.
# d_i novo correlaciona 0,95 com o do X1 e muda 100% dos valores -- perturbacao real e moderada.
banner "10  C2(ii) delta_mean no d_i"
braco_k2 p23_deltamean --detectability-mode soft --feature-contri-meta 0.6 --drop-prefix $DROP \
  --detectability-path artifacts/reports/detectability_deltamean.csv \
  --set linear_tree=false

banner "FILA NOTURNA PARTE 2 -- COMPLETA"
