#!/usr/bin/env bash
# Replicacao da costura C5 na PARTICAO DE FOLDS B (cfg.seed=43).
#
# Por que: o C3 mediu variancia de particao ~0,007 (7x a de semente de boosting) e mostrou que a
# particao A (42) e a SORTUDA. Todo o ganho da costura (+0,0032) foi medido em A. Se o efeito for
# real, ele reaparece em B; se for so a particao, some. `oof_c3_partB_bag4.parquet` ja tem o X1-soft
# em B -- falta o B6 em B, que e o que este script treina.
#
# RESUMIVEL de proposito: cada semente que ja tem diretorio e pulada, entao um kill do wrapper custa
# no maximo a semente em curso (o ambiente vem matando wrappers de job; use nohup para destacar).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8

RB="data/processed/train_rows_3eixos.parquet"
NAME="b6_partB"

for S in 777 101 202 303; do
  if [ -d "artifacts/models/${NAME}_s$S" ]; then
    echo "skip ${NAME}_s$S (ja existe)"
    continue
  fi
  echo "=== ${NAME} semente $S @ $(date +%H:%M) ==="
  python -u scripts/train.py --rows "$RB" --drop-prefix spec_ ord_ mrep_ \
    --detectability-mode soft --feature-contri-meta 0.6 --fold-seed 43 \
    --boost-seed "$S" --out "artifacts/models/${NAME}_s$S" || { echo "FALHOU semente $S"; exit 1; }
done

echo "=== avg_oof das 4 sementes @ $(date +%H:%M) ==="
python scripts/avg_oof.py \
  --inputs artifacts/models/oof_${NAME}_s777.parquet artifacts/models/oof_${NAME}_s101.parquet \
           artifacts/models/oof_${NAME}_s202.parquet artifacts/models/oof_${NAME}_s303.parquet \
  --out artifacts/models/oof_${NAME}_bag4.parquet || exit 1

echo "=== TUDO PRONTO @ $(date +%H:%M) ==="
