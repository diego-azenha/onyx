#!/usr/bin/env bash
# S5 (knowledge/frentes/surpresa-acumulada/S5-familias-como-features.md): famílias de evidência direcional
# do detector de surpresa como features do Onyx. Receita idêntica ao B0 (run_b0_incumbente.sh); só
# muda o parquet de linhas. RESUMÍVEL: pula o que já existe.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE
SEEDS="777 101 202 303"
mkdir -p artifacts/reports/s5

monta_rows () {   # monta_rows <saida> <colunas...>
  local out="$1"; shift
  [ -f "$out" ] && return 0
  python -u - "$out" "$@" <<'EOF'
import sys
import pandas as pd
out, cols = sys.argv[1], sys.argv[2:]
rows = pd.read_parquet("data/processed/train_rows.parquet")
sc = pd.read_parquet("artifacts/surpresa/scores.parquet", columns=["id", "t"] + cols)
sc[cols] = sc[cols].astype("float32")
m = rows.merge(sc, on=["id", "t"], how="left", validate="one_to_one")
assert len(m) == len(rows) and m[cols].notna().all().all()
m.to_parquet(out)
print(out, m.shape)
EOF
}

roda_braco () {   # roda_braco <nome> <rows>
  local nome="$1" rows="$2"
  for S in $SEEDS; do
    if [ -f "artifacts/models/oof_${nome}_s$S.parquet" ]; then echo "skip ${nome}_s$S"; continue; fi
    echo "=== ${nome} semente $S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows "$rows" --detectability-mode soft --linear-tree --boost-seed "$S" \
      --out "artifacts/models/${nome}_s$S" > "artifacts/reports/s5/${nome}_s$S.log" 2>&1 \
      || { echo "FALHOU ${nome} $S"; return 1; }
  done
  local inputs=""; for S in $SEEDS; do inputs="$inputs artifacts/models/oof_${nome}_s$S.parquet"; done
  python -u scripts/avg_oof.py --inputs $inputs --out "artifacts/models/oof_${nome}_bag4.parquet" > /dev/null
  python -u scripts/apply_vema_oof.py --oof "artifacts/models/oof_${nome}_bag4.parquet" \
    --a-up 1.0 --a-dn 0.2 --min-t 50 --out "artifacts/models/oof_${nome}_vema_bag4.parquet" > /dev/null
  OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
    --candidate "artifacts/models/oof_${nome}_vema_bag4.parquet" --n-boot 300 --n-jobs 4 \
    --target-bucket "150<t<=400" --out "artifacts/surpresa/r0_${nome}.json" > "artifacts/reports/s5/r0_${nome}.log" 2>&1
  echo "=== ${nome} COMPLETO @ $(date +%H:%M:%S) ==="
}

monta_rows data/processed/train_rows_s5a.parquet fam_escala_desce fam_escala_sobe || exit 1
monta_rows data/processed/train_rows_s5b.parquet fam_escala_sobe fam_escala_desce fam_cauda fam_forma3 \
  fam_forma4 fam_media fam_correlacao fam_arch || exit 1
roda_braco s5a data/processed/train_rows_s5a.parquet || exit 1
roda_braco s5b data/processed/train_rows_s5b.parquet || exit 1
echo "=== S5 COMPLETO @ $(date +%H:%M:%S) ==="
