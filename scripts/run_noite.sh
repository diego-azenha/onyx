#!/usr/bin/env bash
# FILA NOTURNA (2026-07-26) -- o que restou da campanha, do mais barato ao mais caro.
#
# ## Travas (identicas as do run_resto.sh v2, que substituiu a v1 que travou a maquina)
#   - UM processo pesado por vez, sem excecao. Nada em background aqui dentro.
#   - OMP_NUM_THREADS=4 de 12; build --n-jobs 4; bootstrap --n-jobs 2.
#   - lockfile compartilhado com run_resto.sh: relancar nao duplica carga.
#   - RESUMIVEL: cada etapa pula se o artefato dela ja existe.
#   - uma etapa que falha NAO derruba as seguintes.
#
# ## A trava de COMPARABILIDADE (a mais importante deste script)
#
# Todos os bracos ja medidos da campanha (p13/p14/p15, triagem p567) foram treinados ANTES do D1, e
# os baselines `oof_b6c6_joint_bag2/bag4` tambem sao pre-D1. O `configs/default.yaml` hoje tem
# `linear_tree: true`. Treinar um braco novo com o YAML atual e medi-lo contra esses baselines
# conflacionaria o efeito do braco com o do D1 -- exatamente a classe de erro que produziu a
# retratacao do D1 (comparar medicoes de regimes diferentes).
# Por isso TODO braco aqui leva `--set linear_tree=false`: mede-se o braco, nao o D1.
#
# `--set` NAO e acumulativo no argparse (nargs="*", store): dois `--set` e o ultimo vence. Cada braco
# passa portanto UMA lista `--set` completa, com `linear_tree=false` incluido.
set -u
cd "$(dirname "$0")/.."

mkdir -p artifacts/reports artifacts/reports/polimento
LOCK="artifacts/reports/.run_resto.lock"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ja existe execucao viva (pid $(cat "$LOCK")). Abortando para nao duplicar carga."; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=scripts

L="artifacts/reports/polimento"
ROWS="data/processed/train_rows_bocpd.parquet"
DROP="spec_ ord_ mrep_"
BL2="artifacts/models/oof_b6c6_joint_bag2.parquet"
BL4="artifacts/models/oof_b6c6_joint_bag4.parquet"

# as 15 colunas 100% NaN em t=50 (verificadas no parquet, batem com PROXIMA_SEMANA.md item 2)
NAN15="dep_mass_abs_w100_cal haar_contrast_fine_coarse haar_energy_ln_s4 jump_leverage_w100_cal \
jump_ratio_w100_cal jump_rvbv_ln_w100_cal jump_semivar_asym_w100_cal mmd_joint_newma_cal \
mmd_joint_slow_cal mmd_marginal_newma_cal mmd_marginal_slow_cal varloc_recent_vs_lagged"

banner () { echo; echo "########## $* @ $(date +%H:%M:%S) ##########"; }

# treina_seed <nome> <semente> <rows> <args...>
treina_seed () {
  local nome="$1" S="$2" rows="$3"; shift 3
  [ -d "artifacts/models/${nome}_s$S" ] && { echo "skip ${nome}_s$S"; return 0; }
  echo "=== INICIO ${nome} s$S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --rows "$rows" --boost-seed "$S" \
    --out "artifacts/models/${nome}_s$S" "$@" > "$L/${nome}_s$S.log" 2>&1 \
    || { echo "FALHOU ${nome} s$S"; return 1; }
  echo "=== FIM ${nome} s$S @ $(date +%H:%M:%S) ==="
}

# media <nome> <bagN> <sementes...>
media () {
  local nome="$1" bag="$2"; shift 2
  local out="artifacts/models/oof_${nome}_bag${bag}.parquet"
  [ -f "$out" ] && { echo "skip media ${nome} bag${bag}"; return 0; }
  local ins=""; for S in "$@"; do ins="$ins artifacts/models/oof_${nome}_s$S.parquet"; done
  python scripts/avg_oof.py --inputs $ins --out "$out" > "$L/${nome}_avg${bag}.log" 2>&1 \
    || { echo "FALHOU media ${nome} bag${bag}"; return 1; }
}

# medir <rotulo> <candidato> <baseline>
medir () {
  local rot="$1" cand="$2" bl="$3"
  [ -f "artifacts/reports/polimento_${rot}.json" ] && { echo "skip medicao ${rot}"; return 0; }
  [ -f "$cand" ] || { echo "sem candidato para ${rot} ($cand)"; return 0; }
  echo "=== MEDINDO ${rot} @ $(date +%H:%M:%S) ==="
  python -u scripts/compare_oof.py --baseline "$bl" --candidate "$cand" \
    --n-boot 200 --n-jobs 2 --out "artifacts/reports/polimento_${rot}.json" \
    > "$L/medir_${rot}.log" 2>&1 || echo "FALHOU medicao ${rot}"
}

# braco_k2 <nome> <args...>  -- treina 777/101, media bag2, mede contra BL2
braco_k2 () {
  local nome="$1"; shift
  treina_seed "$nome" 777 "$ROWS" "$@" || return 1
  treina_seed "$nome" 101 "$ROWS" "$@" || return 1
  media "$nome" 2 777 101 || return 1
  medir "$nome" "artifacts/models/oof_${nome}_bag2.parquet" "$BL2"
}

banner "FILA NOTURNA -- INICIO"

# ============================================================ 1. C1 contri=0,4 em K=4
# O maior ponto isolado ainda vivo (+0,0017 em K=2, IC inconclusivo). s777/s101 JA EXISTEM e sao
# pre-D1 (o log nao tem a linha "D1: linear_tree=true"), entao as novas tem de ser pre-D1 tambem.
# Mede K=4 contra K=4 -- o padrao probatorio que a licao do D1 tornou obrigatorio.
banner "1/8  C1 contri=0,4 K=4"
for S in 202 303; do
  treina_seed p6_contri04 "$S" "$ROWS" --detectability-mode soft --feature-contri-meta 0.4 \
    --drop-prefix $DROP --set linear_tree=false || break
done
media p6_contri04 4 777 101 202 303
medir p6_contri04_k4 artifacts/models/oof_p6_contri04_bag4.parquet "$BL4"

# ============================================================ 2. D5 enable_bundle=false
banner "2/8  D5 enable_bundle=false"
braco_k2 p17_nobundle --detectability-mode soft --feature-contri-meta 0.6 \
  --drop-prefix $DROP --set linear_tree=false enable_bundle=false

# ============================================================ 3. D3 celula dart
# O LightGBM IGNORA linear_tree em dart -- esta celula testa uma config que EXCLUI o D1 por
# construcao, e por isso o baseline pre-D1 e o certo.
banner "3/8  D3 dart"
braco_k2 p18_dart --detectability-mode soft --feature-contri-meta 0.6 \
  --drop-prefix $DROP --set linear_tree=false dart=true

# ============================================================ 4. Derrubar as 15 colunas NaN
# Higiene de conjunto, nao sweep. --drop-prefix e uma lista so (o segundo sobrescreveria o primeiro).
banner "4/8  15 colunas 100% NaN em t=50"
braco_k2 p19_dropnan --detectability-mode soft --feature-contri-meta 0.6 \
  --drop-prefix $DROP $NAN15 --set linear_tree=false

# ============================================================ 5-6. contri 0,5 e 0,7
banner "5/8  C1 contri=0,5"
braco_k2 p20_contri05 --detectability-mode soft --feature-contri-meta 0.5 \
  --drop-prefix $DROP --set linear_tree=false
banner "6/8  C1 contri=0,7"
braco_k2 p21_contri07 --detectability-mode soft --feature-contri-meta 0.7 \
  --drop-prefix $DROP --set linear_tree=false

# ============================================================ 7. Divida: B1 K=7 na particao 43
# O bag K=7 foi ADOTADO com uma particao so. partB = folds com cfg.seed=43 (--fold-seed 43).
banner "7/8  divida B1: bag K=7 na particao 43"
for S in 404 505 606; do
  treina_seed b6c6_partB "$S" "$ROWS" --detectability-mode soft --feature-contri-meta 0.6 \
    --drop-prefix $DROP --fold-seed 43 --set linear_tree=false || break
done
media b6c6_partB 7 777 101 202 303 404 505 606
medir b1_bag7_p43 artifacts/models/oof_b6c6_partB_bag7.parquet \
  artifacts/models/oof_b6c6_partB_bag4.parquet

# ============================================================ 8. E1 des-thinning (rebuild; RAM)
# Ultimo por desenho: dataset ~1,4x maior, unico item que pode estourar os 16 GB.
banner "8/8  E1 des-thinning (rebuild)"
if [ ! -f data/processed/train_rows_e1.parquet ]; then
  echo "=== BUILD E1 (n_jobs=4) @ $(date +%H:%M:%S) ==="
  python -u scripts/build_dataset.py --config configs/e1.yaml --n-jobs 4 \
    --out data/processed/train_rows_e1.parquet > "$L/e1_build.log" 2>&1 \
    || echo "FALHOU build E1"
fi
if [ -f data/processed/train_rows_e1.parquet ]; then
  for S in 777 101; do
    treina_seed p12_e1 "$S" data/processed/train_rows_e1.parquet --config configs/e1.yaml \
      --detectability-mode soft --feature-contri-meta 0.6 --drop-prefix $DROP \
      --set linear_tree=false || break
  done
  media p12_e1 2 777 101
  # O OOF do E1 vive numa grade com MAIS linhas: compare_oof casaria so o subconjunto comum, entao o
  # Delta pareado subestima. Registrado aqui; a leitura honesta e o nivel na grade do board.
  medir p12_e1 artifacts/models/oof_p12_e1_bag2.parquet "$BL2"
fi

banner "FILA NOTURNA -- COMPLETA"
