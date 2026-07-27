#!/usr/bin/env bash
# FILA NOTURNA -- PARTE 3: K=4 do unico sobrevivente da noite (`contri=0,7`).
#
# ## Por que este braco existe
#
# `p21_contri07` deu +0,0033 [+0,0004; +0,0061] em K=2, IC excluindo zero e acima da barra -- o melhor
# numero da campanha desde o B6. Mas TODA a tabela de contri desta noite cabe dentro do piso de ruido
# de semente que o proprio repo mediu: "a diferenca de dois bags K=2 e ~0,004, maior que o efeito que
# se pretendia medir, e INVISIVEL para o bootstrap pareado" (HISTORICO.md secao 15.16, a ERRATA que
# desfez a retratacao do D1). O IC do bootstrap trata as predicoes como fixas: ele nao ve semente.
#
# Alem disso, a "curva" 0,4/0,5/0,7 positivos e 0,2/0,8 negativos nao tem forma de curva -- tem forma
# de baseline azarado. So o K=4 separa as duas leituras.
#
# Acrescenta 202 e 303, faz a media das QUATRO e mede contra `oof_b6c6_joint_bag4` -- K=4 contra K=4,
# o padrao probatorio que a licao do D1 tornou obrigatorio.
#
# ESPERA o lock (parte 1 e 2 na frente). Teto de 8 h.
set -u
cd "$(dirname "$0")/.."

L="artifacts/reports/polimento"
LOCK="artifacts/reports/.run_resto.lock"
mkdir -p "$L"

esperou=0
while [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; do
  [ "$esperou" -ge 28800 ] && { echo "timeout de 8 h esperando o lock; abortando."; exit 1; }
  sleep 60; esperou=$((esperou + 60))
done
echo "lock livre apos $((esperou / 60)) min"
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=scripts

ROWS="data/processed/train_rows_bocpd.parquet"

echo "########## PARTE 3: contri=0,7 em K=4 @ $(date +%H:%M:%S) ##########"

for S in 202 303; do
  [ -d "artifacts/models/p21_contri07_s$S" ] && { echo "skip p21_contri07_s$S"; continue; }
  echo "=== INICIO p21_contri07 s$S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$ROWS" --boost-seed "$S" \
    --detectability-mode soft --feature-contri-meta 0.7 --drop-prefix spec_ ord_ mrep_ \
    --set linear_tree=false \
    --out "artifacts/models/p21_contri07_s$S" > "$L/p21_contri07_s$S.log" 2>&1 \
    || { echo "FALHOU p21_contri07 s$S"; exit 1; }
  echo "=== FIM p21_contri07 s$S @ $(date +%H:%M:%S) ==="
done

if [ ! -f artifacts/models/oof_p21_contri07_bag4.parquet ]; then
  python scripts/avg_oof.py --inputs \
    artifacts/models/oof_p21_contri07_s777.parquet artifacts/models/oof_p21_contri07_s101.parquet \
    artifacts/models/oof_p21_contri07_s202.parquet artifacts/models/oof_p21_contri07_s303.parquet \
    --out artifacts/models/oof_p21_contri07_bag4.parquet > "$L/p21_contri07_avg4.log" 2>&1 \
    || { echo "FALHOU media das 4"; exit 1; }
fi

if [ ! -f artifacts/reports/polimento_p21_contri07_k4.json ]; then
  echo "=== MEDINDO p21_contri07_k4 @ $(date +%H:%M:%S) ==="
  python -u scripts/compare_oof.py \
    --baseline artifacts/models/oof_b6c6_joint_bag4.parquet \
    --candidate artifacts/models/oof_p21_contri07_bag4.parquet \
    --n-boot 200 --n-jobs 2 --out artifacts/reports/polimento_p21_contri07_k4.json \
    > "$L/medir_p21_contri07_k4.log" 2>&1 || echo "FALHOU medicao p21_contri07_k4"
fi

echo "########## PARTE 3 COMPLETA @ $(date +%H:%M:%S) ##########"
