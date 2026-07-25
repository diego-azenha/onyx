#!/usr/bin/env bash
# SPRINT ATE O FIM DO D3 (2026-07-25). Recorte do `run_resto.sh`, mesmas travas.
#
# ## O que este script faz, e onde ele PARA
#
#   1. A6   -- rebuild do dataset (`conformal.use_raw` + `h0.null_clip_match`) + treino + medicao
#   2. C2   -- `p13_floor015`, `p13_floor050`
#   3. D3   -- `p14_mdl100`, `p15_l2_20`, `p16_ff06`
#
# PARA AQUI, de proposito. NAO roda o D5 (`p17_nobundle`), NAO roda o E1 (des-thinning) e NAO
# monta o notebook -- o plano e verificar os resultados, montar o notebook e fazer a submissao
# parcial com decisao humana no meio. Para o resto da fila: `bash scripts/run_resto.sh`.
#
# ## Travas (identicas as do run_resto.sh -- a v1 dele travou a maquina por sobre-assinatura)
#
#   - UM processo pesado por vez, inclusive medicoes (o bootstrap pareado sobre 2,5 M linhas
#     e tao pesado quanto um treino).
#   - `OMP_NUM_THREADS=4` de 12 logicas / 6 fisicas -- folga para o SO e para o usuario.
#   - build com `--n-jobs 4`, NUNCA `-1`.  bootstrap com `--n-jobs 2`.
#   - lockfile COMPARTILHADO com o run_resto.sh: os dois nunca rodam juntos.
#   - RESUMIVEL: cada etapa pula se o artefato dela ja existe.
#   - uma etapa que falha NAO derruba as seguintes.
set -u
cd "$(dirname "$0")/.."

mkdir -p artifacts/reports artifacts/reports/polimento
LOCK="artifacts/reports/.run_resto.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ja existe execucao viva (pid $(cat "$LOCK")). Abortando para nao duplicar carga."; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=scripts

L="artifacts/reports/polimento"
ROWS="data/processed/train_rows_bocpd.parquet"
BASE="--detectability-mode soft --feature-contri-meta 0.6 --drop-prefix spec_ ord_ mrep_"
BL="artifacts/models/oof_b6c6_joint_bag2.parquet"

treinar () {  # treinar <nome> <rows> <cfg|-> <args...>
  local nome="$1" rows="$2" cfg="$3"; shift 3
  local cfgarg=""; [ "$cfg" != "-" ] && cfgarg="--config $cfg"
  for S in 777 101; do
    [ -d "artifacts/models/${nome}_s$S" ] && { echo "skip ${nome}_s$S"; continue; }
    echo "=== INICIO ${nome} s$S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py $cfgarg --rows "$rows" $BASE --boost-seed "$S" \
      --out "artifacts/models/${nome}_s$S" "$@" > "$L/${nome}_s$S.log" 2>&1 \
      || { echo "FALHOU ${nome} s$S -- pulando o braco"; return 1; }
  done
  [ -f "artifacts/models/oof_${nome}_bag2.parquet" ] || \
    python scripts/avg_oof.py --inputs "artifacts/models/oof_${nome}_s777.parquet" \
      "artifacts/models/oof_${nome}_s101.parquet" \
      --out "artifacts/models/oof_${nome}_bag2.parquet" > "$L/${nome}_avg.log" 2>&1
  echo "=== ${nome} TREINADO @ $(date +%H:%M:%S) ==="
}

medir () {   # medir <rotulo> <candidato.parquet> [baseline.parquet]
  local rot="$1" cand="$2"; local bl="${3:-$BL}"
  [ -f "artifacts/reports/polimento_${rot}.json" ] && { echo "skip medicao ${rot}"; return 0; }
  [ -f "$cand" ] || { echo "sem candidato para ${rot}"; return 0; }
  echo "=== MEDINDO ${rot} @ $(date +%H:%M:%S) ==="
  python -u scripts/compare_oof.py --baseline "$bl" --candidate "$cand" \
    --n-boot 200 --n-jobs 2 --out "artifacts/reports/polimento_${rot}.json" \
    > "$L/medir_${rot}.log" 2>&1 || echo "FALHOU medicao ${rot}"
}

echo "########## SPRINT ATE O D3 -- INICIO @ $(date +%H:%M:%S) ##########"

# ==================================== 1. A6 (maior teto: dois defeitos de escala sobre a feature n1)
# `configs/a6.yaml` ja existe e difere do default APENAS nas duas flags (verificado). So regenera
# se sumir -- nao sobrescreve um arquivo bom.
if [ ! -f configs/a6.yaml ]; then
  python - <<'PY'
from pathlib import Path
s = Path("configs/default.yaml").read_text(encoding="utf-8")
s = s.replace("  null_clip_match: false", "  null_clip_match: true", 1)
s = s.replace("  use_raw: false", "  use_raw: true", 1)
assert "null_clip_match: true" in s and "use_raw: true" in s
Path("configs/a6.yaml").write_text(s, encoding="utf-8"); print("configs/a6.yaml regenerado")
PY
fi

if [ ! -f data/processed/train_rows_a6.parquet ]; then
  echo "=== BUILD A6 (n_jobs=4, ~45-60 min) @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --config configs/a6.yaml --n-jobs 4 \
    --out data/processed/train_rows_a6.parquet > "$L/a6_build.log" 2>&1 \
    || echo "FALHOU build A6 -- ver $L/a6_build.log"
fi
# Portao: sem o parquet, o treino so produziria FileNotFoundError (foi o que a v1 do run_resto fez).
if [ -f data/processed/train_rows_a6.parquet ]; then
  echo "=== build A6 OK ($(du -h data/processed/train_rows_a6.parquet | cut -f1)) @ $(date +%H:%M:%S) ==="
  treinar p8_a6 data/processed/train_rows_a6.parquet configs/a6.yaml
  medir   p8_a6 artifacts/models/oof_p8_a6_bag2.parquet
else
  echo "!!! A6 PULADO: o parquet nao existe. Os demais bracos NAO dependem dele, seguindo."
fi

# ==================================== 2. C2 -- vizinhanca do X1 (sem rebuild)
# `detect_floor` nunca foi alcancavel offline: `scripts/train.py` chamava `compute_row_weights` sem
# passar o parametro, entao valia sempre o 0,3 da assinatura. Corrigido com `--detect-floor`.
treinar p13_floor015 "$ROWS" - --detect-floor 0.15
medir   p13_floor015 artifacts/models/oof_p13_floor015_bag2.parquet
treinar p13_floor050 "$ROWS" - --detect-floor 0.50
medir   p13_floor050 artifacts/models/oof_p13_floor050_bag2.parquet

# ==================================== 3. D3 -- sweep de HPs (sem rebuild)
treinar p14_mdl100   "$ROWS" - --set min_data_in_leaf=100
medir   p14_mdl100   artifacts/models/oof_p14_mdl100_bag2.parquet
treinar p15_l2_20    "$ROWS" - --set lambda_l2=20.0
medir   p15_l2_20    artifacts/models/oof_p15_l2_20_bag2.parquet
treinar p16_ff06     "$ROWS" - --set feature_fraction=0.6
medir   p16_ff06     artifacts/models/oof_p16_ff06_bag2.parquet

# ==================================== FIM -- parada deliberada
echo "########## D3 CONCLUIDO @ $(date +%H:%M:%S) ##########"
echo
echo "PARADA DELIBERADA. NAO rodados de proposito: D5 (p17_nobundle), E1 (des-thinning), notebook."
echo "Resumo dos JSONs deste sprint:"
for r in p8_a6 p13_floor015 p13_floor050 p14_mdl100 p15_l2_20 p16_ff06; do
  f="artifacts/reports/polimento_${r}.json"
  [ -f "$f" ] && echo "  OK    $r" || echo "  AUSENTE $r"
done
