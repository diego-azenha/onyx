#!/usr/bin/env bash
# TUDO O QUE FALTA DA CAMPANHA -- ESTRITAMENTE SERIAL, COM TETO DE RECURSOS (2026-07-25, v2).
#
# ## Por que esta versao existe
#
# A v1 travou a maquina. Nao foi "paralelismo demais" no sentido vago -- foi sobre-assinatura
# aritmetica: `build_dataset.py` com `n_jobs=-1` (12 nucleos) + um lote de medicoes com `n_jobs=4`
# + `OMP_NUM_THREADS=6` por worker = ~40 threads disputando 12 CPUs logicas. O erro de julgamento foi
# tratar "medicao" como leve: o bootstrap pareado sobre 2,5 M linhas e tao pesado quanto um treino.
#
# ## Travas
#
#   - UM processo pesado por vez, sem excecao (inclusive medicoes). Nada em background aqui dentro.
#   - `OMP_NUM_THREADS=4` de 12 -- folga para o SO e para o usuario.
#   - build do dataset com `--n-jobs 4`, NUNCA `-1`.
#   - bootstrap com `--n-jobs 2`.
#   - lockfile: relancar enquanto roda nao duplica carga.
#   - RESUMIVEL: cada etapa pula se o artefato dela ja existe.
#   - uma etapa que falha NAO derruba as seguintes.
#
# ## Ordem: valor primeiro, risco de RAM por ultimo
#
#   0. medir B1/C1/B3-p43 -- ja treinados, so falta o IC. Barato e comeca dando resultado.
#   1. A6                 -- maior teto: os dois defeitos de escala sobre a feature nº 1 (rebuild)
#   2. C2 / D3 / D5       -- treinos baratos, sem rebuild
#   3. E1                 -- dataset ~1,4x maior; unico item que pode estourar RAM => por ultimo
#   4. notebook           -- regenerar + verificar bit-a-bit
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

# ==================================== 0. o que ja esta treinado (barato, comeca dando resultado)
medir b1_bag7      artifacts/models/oof_b6c6_joint_bag7.parquet  artifacts/models/oof_b6c6_joint_bag4.parquet
medir p11_contri02 artifacts/models/oof_p11_contri02_bag2.parquet

if [ ! -f artifacts/models/oof_b3_vema_p43.parquet ]; then
  python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_b6c6_partB_bag4.parquet \
    --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_b3_vema_p43.parquet \
    > "$L/b3_p43_apply.log" 2>&1 || echo "FALHOU aplicar v-ema na particao 43"
fi
medir b3_p43 artifacts/models/oof_b3_vema_p43.parquet artifacts/models/oof_b6c6_partB_bag4.parquet

# ==================================== 1. A6 (maior teto; rebuild com n_jobs CAPADO)
python - <<'PY'
from pathlib import Path
s = Path("configs/default.yaml").read_text(encoding="utf-8")
s = s.replace("  null_clip_match: false", "  null_clip_match: true", 1)
s = s.replace("  use_raw: false", "  use_raw: true", 1)
assert "null_clip_match: true" in s and "use_raw: true" in s
Path("configs/a6.yaml").write_text(s, encoding="utf-8"); print("configs/a6.yaml ok")
PY
if [ ! -f data/processed/train_rows_a6.parquet ]; then
  echo "=== BUILD A6 (n_jobs=4) @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --config configs/a6.yaml --n-jobs 4 \
    --out data/processed/train_rows_a6.parquet > "$L/a6_build.log" 2>&1 || echo "FALHOU build A6"
fi
treinar p8_a6 data/processed/train_rows_a6.parquet configs/a6.yaml
medir p8_a6 artifacts/models/oof_p8_a6_bag2.parquet

# ==================================== 2. treinos baratos, sem rebuild
# `detect_floor` nunca foi alcancavel offline: `scripts/train.py` chamava `compute_row_weights` sem
# passar o parametro, entao valia sempre o 0,3 da assinatura. Corrigido com `--detect-floor`.
treinar p13_floor015 "$ROWS" - --detect-floor 0.15
medir   p13_floor015 artifacts/models/oof_p13_floor015_bag2.parquet
treinar p13_floor050 "$ROWS" - --detect-floor 0.50
medir   p13_floor050 artifacts/models/oof_p13_floor050_bag2.parquet
treinar p14_mdl100   "$ROWS" - --set min_data_in_leaf=100
medir   p14_mdl100   artifacts/models/oof_p14_mdl100_bag2.parquet
treinar p15_l2_20    "$ROWS" - --set lambda_l2=20.0
medir   p15_l2_20    artifacts/models/oof_p15_l2_20_bag2.parquet
treinar p16_ff06     "$ROWS" - --set feature_fraction=0.6
medir   p16_ff06     artifacts/models/oof_p16_ff06_bag2.parquet
treinar p17_nobundle "$ROWS" - --set enable_bundle=false
medir   p17_nobundle artifacts/models/oof_p17_nobundle_bag2.parquet

# ==================================== 3. E1 -- maior risco de RAM, por ultimo
# Dataset ~1,4x maior (t<=400 sem thinning). Com 16 GB e nada mais rodando cabe, mas e o unico item
# que pode estourar memoria -- por isso vem depois de tudo o que ja rendeu.
python - <<'PY'
from pathlib import Path
s = Path("configs/default.yaml").read_text(encoding="utf-8")
s = s.replace("  full_until: 100", "  full_until: 400", 1)
assert "full_until: 400" in s
Path("configs/e1.yaml").write_text(s, encoding="utf-8"); print("configs/e1.yaml ok")
PY
if [ ! -f data/processed/train_rows_e1.parquet ]; then
  echo "=== BUILD E1 (n_jobs=4) @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --config configs/e1.yaml --n-jobs 4 \
    --out data/processed/train_rows_e1.parquet > "$L/e1_build.log" 2>&1 || echo "FALHOU build E1"
fi
# O OOF do E1 vive numa grade DIFERENTE (mais linhas), entao `compare_oof.py` casaria so o
# subconjunto comum -- a leitura honesta e o nivel na grade do board, feita no relatorio.
treinar p12_e1 data/processed/train_rows_e1.parquet configs/e1.yaml

# ==================================== 4. notebook
echo "=== NOTEBOOK @ $(date +%H:%M:%S) ==="
python -u scripts/build_submission_notebook.py > "$L/nb_build.log" 2>&1
python -u scripts/verify_submission_notebook.py > "$L/nb_verify.log" 2>&1 \
  && echo "notebook VERIFICADO bit-a-bit" || echo "notebook FALHOU verificacao"

echo "=== RESTO DA CAMPANHA COMPLETO @ $(date +%H:%M:%S) ==="
