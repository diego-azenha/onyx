#!/usr/bin/env bash
# A1 (knowledge/frentes/teto-offline/A1-recorte.md): Onyx treinado com originais + re-cortes k=1..K.
# Uso: bash scripts/run_a1.sh "<ks>" <nome>     ex.: bash scripts/run_a1.sh "1" a1k1
# RESUMÍVEL. Um job pesado por vez.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
KS="$1"; NOME="$2"; SEEDS="${3:-777 101 202 303}"
mkdir -p artifacts/reports/$NOME
ROWS=data/processed/train_rows_$NOME.parquet
DET=artifacts/reports/detectability_$NOME.csv

for K in $KS; do
  [ -f data/processed/aug_recorte_k$K.parquet ] || \
    python -u scripts/a1_gerar_recortes.py --k $K --n-jobs 4 > artifacts/reports/$NOME/gerar_k$K.log 2>&1 || { echo FALHOU gerar $K; exit 1; }
done

if [ ! -f $DET ]; then
  python -u - "$DET" $KS <<'EOF' > artifacts/reports/$NOME/det.log 2>&1 || { echo FALHOU det; exit 1; }
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from surpresa_avaliar import _carregar_series
from a1_gerar_recortes import recortar
from sbrt.config import load_config, DEFAULT_CONFIG_PATH
from sbrt.model.detectability import compute_detectability_map
out, ks = sys.argv[1], [int(k) for k in sys.argv[2:]]
cfg = load_config(DEFAULT_CONFIG_PATH)
yi = pd.read_parquet("data/y_train_index.parquet")
series = _carregar_series(Path("data"))
dets = [pd.read_csv("artifacts/reports/detectability.csv")]
for k in ks:
    rng = np.random.default_rng(20260930 + k)          # mesma sequência do gerador => mesmos re-cortes
    recs = [r for sid, h, on in series if (r := recortar(sid, h, on, int(yi.loc[sid, "tau_index"]), rng, k)) is not None]
    dets.append(compute_detectability_map(recs, cfg, n_jobs=4))
pd.concat(dets, ignore_index=True).to_csv(out, index=False)
print(out, sum(len(d) for d in dets))
EOF
fi

if [ ! -f $ROWS ]; then
  python -u - "$ROWS" $KS <<'EOF' > artifacts/reports/$NOME/concat.log 2>&1 || { echo FALHOU concat; exit 1; }
import sys
import pandas as pd
out, ks = sys.argv[1], sys.argv[2:]
base = pd.read_parquet("data/processed/train_rows.parquet")
partes = [base] + [pd.read_parquet(f"data/processed/aug_recorte_k{k}.parquet")[base.columns] for k in ks]
pd.concat(partes, ignore_index=True).to_parquet(out)
print(out, sum(len(p) for p in partes))
EOF
fi

for S in $SEEDS; do
  [ -f artifacts/models/oof_${NOME}_s$S.parquet ] && { echo "skip $S"; continue; }
  echo "=== $NOME semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/a1_treino.py --rows $ROWS --detectability-mode soft --detectability-path $DET \
    --linear-tree --boost-seed $S --out artifacts/models/${NOME}_s$S > artifacts/reports/$NOME/train_s$S.log 2>&1 \
    || { echo "FALHOU treino $S"; exit 1; }
done
INPUTS=""; for S in $SEEDS; do INPUTS="$INPUTS artifacts/models/oof_${NOME}_s$S.parquet"; done
python -u scripts/avg_oof.py --inputs $INPUTS --out artifacts/models/oof_${NOME}_bag4.parquet > /dev/null
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_${NOME}_bag4.parquet --a-up 1.0 --a-dn 0.2 --min-t 50 \
  --out artifacts/models/oof_${NOME}_vema_bag4.parquet > /dev/null
OMP_NUM_THREADS=4 python -u scripts/compare_oof.py --baseline artifacts/models/oof_b0_vema_bag4.parquet \
  --candidate artifacts/models/oof_${NOME}_vema_bag4.parquet --n-boot 300 --n-jobs 4 \
  --out artifacts/reports/r0_${NOME}.json > artifacts/reports/$NOME/r0.log 2>&1
echo "=== $NOME COMPLETO @ $(date +%H:%M:%S) ==="
