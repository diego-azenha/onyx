#!/usr/bin/env bash
# C2 (knowledge/frentes/teto-offline/C2-positivos-ou-negativos.md): o limite de amostra do Onyx está
# nos positivos (eventos de quebra) ou nos negativos (variedade de H0)?
# meio_neg = todos os positivos + metade (id par) dos negativos; meio_pos = todos os negativos + metade
# (id par) dos positivos. Avaliação comum: as séries de id par (o mesmo conjunto do C1).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/c2
for V in meio_neg meio_pos; do
  ROWS=data/processed/train_rows_c2_$V.parquet
  [ -f $ROWS ] || python -u - $V $ROWS <<'EOF'
import sys
import pandas as pd
v, out = sys.argv[1], sys.argv[2]
yi = pd.read_parquet("data/y_train_index.parquet")
pos = set(yi.index[yi.tau_index >= 0])
ids = [i for i in yi.index if (i % 2 == 0) or ((i in pos) if v == "meio_neg" else (i not in pos))]
r = pd.read_parquet("data/processed/train_rows.parquet", filters=[("id", "in", ids)])
r.to_parquet(out); print(v, r["id"].nunique(), len(r))
EOF
  for S in 777 101; do
    [ -f artifacts/models/oof_c2_${V}_s$S.parquet ] && continue
    echo "=== c2 $V semente $S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows $ROWS --detectability-mode soft --linear-tree --boost-seed $S \
      --out artifacts/models/c2_${V}_s$S > artifacts/reports/c2/${V}_s$S.log 2>&1 || { echo FALHOU; exit 1; }
  done
done
echo "=== C2 COMPLETO @ $(date +%H:%M:%S) ==="
