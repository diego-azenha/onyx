#!/usr/bin/env bash
# B6+C6 conjuntos: X1-soft + feature_contri=0,6 (B6) SOBRE o dataset com BOCPD (C6, 187 feat).
#
# A pergunta NAO e "o C6 replica?" e sim "o C6 acrescenta alguma coisa em cima do B6, que ja esta
# adotado?". Se nao acrescentar, o C6 e irrelevante independentemente de replicar.
#
# Precedente de fracasso, explicito: o V5 juntou poda + BOCPD, cada um positivo SOZINHO, e o pacote
# REGREDIU (-0,0042). Este braco e exatamente a mesma forma de aposta.
#
# Criterio: adotar o C6 no pacote so se o conjunto bater o B6 SOZINHO (`oof_b6_contri06_bag4`,
# 0,6125) por mais que a barra (~0,0014). Comparar tambem contra o incumbente para registro.
#
# RESUMIVEL: pula a semente cujo diretorio ja existe.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1          # o dataset ja tem as colunas; a variavel mantem o scorer coerente

ROWS="data/processed/train_rows_bocpd.parquet"
NAME="b6c6_joint"

if [ ! -f "$ROWS" ]; then echo "FALTA $ROWS (rode run_c6_bocpd.sh antes)"; exit 1; fi

for S in 777 101 202 303; do
  if [ -d "artifacts/models/${NAME}_s$S" ]; then echo "skip ${NAME}_s$S (ja existe)"; continue; fi
  echo "=== ${NAME} semente $S @ $(date +%H:%M) ==="
  python -u scripts/train.py --rows "$ROWS" --drop-prefix spec_ ord_ mrep_ \
    --detectability-mode soft --feature-contri-meta 0.6 \
    --boost-seed "$S" --out "artifacts/models/${NAME}_s$S" || { echo "FALHOU semente $S"; exit 1; }
done

echo "=== avg_oof @ $(date +%H:%M) ==="
python scripts/avg_oof.py \
  --inputs artifacts/models/oof_${NAME}_s777.parquet artifacts/models/oof_${NAME}_s101.parquet \
           artifacts/models/oof_${NAME}_s202.parquet artifacts/models/oof_${NAME}_s303.parquet \
  --out artifacts/models/oof_${NAME}_bag4.parquet || exit 1

echo "=== B6+C6 PRONTO @ $(date +%H:%M) ==="
