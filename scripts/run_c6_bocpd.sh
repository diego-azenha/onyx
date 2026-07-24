#!/usr/bin/env bash
# C6 (BRAINSTORM_RUPTURA_V2.md): BOCPD ISOLADO sobre o V4 -- o experimento que separa as duas
# mudancas empacotadas no V5 (poda + BOCPD) e que nunca foi rodado. Historico: o V5 regrediu como
# pacote, mas os dois componentes mediram positivo separados (poda +0,0027, BOCPD +0,0029), sempre
# na regua velha; aqui o BOCPD sozinho enfrenta a regua nova (~0,0014).
#
# `SBRT_ENABLE_BOCPD=1` acrescenta o BOCPDBlock ao `default_blocks()` (opt-in; sem a variavel o
# default de producao nao muda). O bloco emite 4 colunas `bocpd_*` => 183 + 4 = 187 features depois
# dos --drop-prefix de sempre.
#
# RESUMIVEL: build e cada semente sao pulados se a saida ja existir.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1          # vale para a build E para os treinos deste script

ROWS="data/processed/train_rows_bocpd.parquet"
NAME="c6_bocpd"

if [ ! -f "$ROWS" ]; then
  echo "=== build do dataset com BOCPD @ $(date +%H:%M) ==="
  python -u scripts/build_dataset.py --out "$ROWS" || { echo "FALHOU build"; exit 1; }
else
  echo "skip build ($ROWS ja existe)"
fi

for S in 777 101 202 303; do
  if [ -d "artifacts/models/${NAME}_s$S" ]; then
    echo "skip ${NAME}_s$S (ja existe)"
    continue
  fi
  echo "=== ${NAME} semente $S @ $(date +%H:%M) ==="
  python -u scripts/train.py --rows "$ROWS" --drop-prefix spec_ ord_ mrep_ \
    --detectability-mode soft --boost-seed "$S" \
    --out "artifacts/models/${NAME}_s$S" || { echo "FALHOU semente $S"; exit 1; }
done

echo "=== avg_oof @ $(date +%H:%M) ==="
python scripts/avg_oof.py \
  --inputs artifacts/models/oof_${NAME}_s777.parquet artifacts/models/oof_${NAME}_s101.parquet \
           artifacts/models/oof_${NAME}_s202.parquet artifacts/models/oof_${NAME}_s303.parquet \
  --out artifacts/models/oof_${NAME}_bag4.parquet || exit 1

echo "=== C6 PRONTO @ $(date +%H:%M) ==="
