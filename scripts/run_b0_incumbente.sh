#!/usr/bin/env bash
# B0 (knowledge/frentes/surpresa-acumulada/B0-incumbente.md): reconstroi o OOF do incumbente neste clone,
# que nao tem data/processed nem artifacts/. E o baseline dos experimentos S2-S4 da frente de surpresa
# acumulada. Receita = o pacote aprovado em HISTORICO.md §15.18: config atual (default_blocks, com
# BOCPD e mrep), X1 soft, linear_tree (D1), K=4 sementes, e o v-EMA assimetrico do B3 por cima.
#
# RESUMIVEL: cada etapa pula se o artefato ja existe (NOTAS_AGENTES.md §7 -- o wrapper de background
# deste ambiente e morto a esmo). Rodar com: nohup bash scripts/run_b0_incumbente.sh > artifacts/reports/b0.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE

ROWS="data/processed/train_rows.parquet"
DET="artifacts/reports/detectability.csv"
SEEDS="777 101 202 303"
mkdir -p artifacts/reports/b0 artifacts/models

if [ ! -f "$ROWS" ]; then
  echo "=== dataset @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --n-jobs 4 --out "$ROWS" > artifacts/reports/b0/dataset.log 2>&1 \
    || { echo "FALHOU dataset"; exit 1; }
fi

if [ ! -f "$DET" ]; then
  echo "=== detectability.csv @ $(date +%H:%M:%S) ==="
  python -u - <<'EOF' > artifacts/reports/b0/detectability.log 2>&1 || { echo "FALHOU detectability"; exit 1; }
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from build_dataset import _load_series_records
from sbrt.config import load_config, DEFAULT_CONFIG_PATH
from sbrt.model.detectability import compute_detectability_map
cfg = load_config(DEFAULT_CONFIG_PATH)
det = compute_detectability_map(_load_series_records(Path("data")), cfg, n_jobs=4)
det.to_csv("artifacts/reports/detectability.csv", index=False)
print(len(det), "series no mapa")
EOF
fi

for S in $SEEDS; do
  if [ -f "artifacts/models/oof_b0_s$S.parquet" ]; then echo "skip b0_s$S"; continue; fi
  echo "=== treino semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$ROWS" --detectability-mode soft --linear-tree --boost-seed "$S" \
    --out "artifacts/models/b0_s$S" > "artifacts/reports/b0/train_s$S.log" 2>&1 \
    || { echo "FALHOU semente $S"; exit 1; }
done

INPUTS=""; for S in $SEEDS; do INPUTS="$INPUTS artifacts/models/oof_b0_s$S.parquet"; done
python -u scripts/avg_oof.py --inputs $INPUTS --out artifacts/models/oof_b0_bag4.parquet \
  > artifacts/reports/b0/avg.log 2>&1 || { echo "FALHOU avg"; exit 1; }
python -u scripts/apply_vema_oof.py --oof artifacts/models/oof_b0_bag4.parquet \
  --a-up 1.0 --a-dn 0.2 --min-t 50 --out artifacts/models/oof_b0_vema_bag4.parquet \
  > artifacts/reports/b0/vema.log 2>&1 || { echo "FALHOU vema"; exit 1; }
echo "=== B0 COMPLETO @ $(date +%H:%M:%S) ==="
