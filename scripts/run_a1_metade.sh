#!/usr/bin/env bash
# A1-metade (knowledge/frentes/teto-offline/A1-recorte.md): 5k séries de id par + os re-cortes k=1 delas,
# contra o C1 (5k sozinhas = 0,6129; 10k = 0,6273, sementes 777/101, mesmas séries avaliadas).
# Cabe na RAM (~2,3M linhas, o tamanho do B0); a versão com 10k + re-cortes (4,7M linhas) trocou para disco.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=PASSIVE
mkdir -p artifacts/reports/a1h
ROWS=data/processed/train_rows_a1h.parquet
[ -f $ROWS ] || python -u - <<'EOF'
import pandas as pd
base = pd.read_parquet("data/processed/train_rows_metade_par.parquet")
aug = pd.read_parquet("data/processed/aug_recorte_k1.parquet")
aug = aug[(aug["id"] % 10000) % 2 == 0][base.columns]
pd.concat([base, aug], ignore_index=True).to_parquet("data/processed/train_rows_a1h.parquet")
print(len(base), len(aug))
EOF
for S in 777 101; do
  [ -f artifacts/models/oof_a1h_s$S.parquet ] && continue
  echo "=== a1h semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/a1_treino.py --rows $ROWS --detectability-mode soft --detectability-path artifacts/reports/detectability_a1k1.csv \
    --linear-tree --boost-seed $S --out artifacts/models/a1h_s$S > artifacts/reports/a1h/train_s$S.log 2>&1 || { echo FALHOU; exit 1; }
done
echo "=== A1H COMPLETO @ $(date +%H:%M:%S) ==="
