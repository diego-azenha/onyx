#!/usr/bin/env bash
# A6 (CAMPANHA_POLIMENTO.md) — a estatistica e o nulo dela na MESMA escala.
#
# Dois defeitos medidos na frente A (docs/RELATORIO_POLIMENTO.md §2, HISTORICO.md §15.2):
#
#   1. `state/conformal.py` ranqueia `|e|` CLIPADO em [-8,8] contra `h0.sorted_abs_e_hist`, que e
#      construido de `e_hist = resid/sigma_e` **sem clip**. Toda observacao com |e_raw| > 8 e
#      ranqueada como se valesse 8,0 => o p-value satura na massa de cauda do historico acima de 8
#      em vez de cair para o piso 1/(n_h+1). `conformal_logm_abs` e a feature nº 1 do modelo
#      (xs-SHAP 0,061; conv_share 0,147).
#
#   2. `state/calibration.py:compute_null_stats` faz o replay sobre `e_hist` NAO clipado, enquanto a
#      producao calcula as mesmas estatisticas sobre `e` clipado. O nulo de TODA coluna `_cal` e
#      estimado numa escala que a producao nunca ve.
#
# MEDIDO: o clip morde em 7,50% das series -- e 6,99% das linhas POSITIVAS contra 2,64% das
# negativas (2,6x). Comprime exatamente o que separa as classes.
#
# A correcao e DIRECIONAL, nao "unclip tudo": o maior |e_raw| da base e 15.774, e um ponto desses
# destroi `accum_welford_var_ln`. Conformal (rank-based, referencia crua) -> `e_raw`; replay do nulo
# (colunas `_cal` calculadas sobre `e` clipado) -> clipar.
#
# Exige REBUILD do dataset: as duas mudancas alteram as features, nao so o treino.
#
# RESUMIVEL. Rodar SOZINHO (ver NOTAS_AGENTES.md §7: treino concorrente com bootstrap deadlocou).
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE

ROWS="data/processed/train_rows_a6.parquet"
LOGDIR="artifacts/reports/polimento"; mkdir -p "$LOGDIR"

# As duas flags entram por YAML (unica fonte de numeros, NOTAS_AGENTES.md §1) -- o build le a config
# do disco, entao um override por CLI nao alcancaria `fit_h0`.
CFG="configs/a6.yaml"
python - <<'PY'
from pathlib import Path
src = Path("configs/default.yaml").read_text(encoding="utf-8")
src = src.replace("  null_clip_match: false", "  null_clip_match: true", 1)
src = src.replace("  use_raw: false", "  use_raw: true", 1)
assert "null_clip_match: true" in src and "use_raw: true" in src, "as flags do A6 nao foram ligadas"
Path("configs/a6.yaml").write_text(src, encoding="utf-8")
print("configs/a6.yaml gerado com conformal.use_raw=true e h0.null_clip_match=true")
PY

if [ ! -f "$ROWS" ]; then
  echo "=== BUILD do dataset A6 @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --config "$CFG" --out "$ROWS" > "$LOGDIR/a6_build.log" 2>&1 \
    || { echo "FALHOU o build (ver $LOGDIR/a6_build.log)"; exit 1; }
  echo "=== BUILD PRONTO @ $(date +%H:%M:%S) ==="
fi

# Mesma receita do incumbente (X1-soft + B6, 187 feat) -- so as features do A6 mudam.
for S in 777 101; do
  [ -d "artifacts/models/p8_a6_s$S" ] && { echo "skip p8_a6_s$S"; continue; }
  echo "=== INICIO p8_a6 semente $S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --config "$CFG" --rows "$ROWS" --drop-prefix spec_ ord_ mrep_ \
    --detectability-mode soft --feature-contri-meta 0.6 --boost-seed "$S" \
    --out "artifacts/models/p8_a6_s$S" > "$LOGDIR/p8_a6_s$S.log" 2>&1 \
    || { echo "FALHOU p8_a6 s$S"; exit 1; }
  echo "=== FIM    p8_a6 semente $S @ $(date +%H:%M:%S) ==="
done

python scripts/avg_oof.py \
  --inputs artifacts/models/oof_p8_a6_s777.parquet artifacts/models/oof_p8_a6_s101.parquet \
  --out artifacts/models/oof_p8_a6_bag2.parquet > "$LOGDIR/p8_a6_avg.log" 2>&1

echo "=== A6 PRONTO @ $(date +%H:%M:%S) ==="
echo "medir com: python -u scripts/compare_oof.py \\"
echo "  --baseline artifacts/models/oof_b6c6_joint_bag2.parquet \\"
echo "  --candidate artifacts/models/oof_p8_a6_bag2.parquet --n-boot 300"
