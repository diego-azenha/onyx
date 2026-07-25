#!/usr/bin/env bash
# CAMPANHA DE POLIMENTO (CAMPANHA_POLIMENTO.md) -- triagem K=2 e promocao do lider.
#
# Ordem revisada pelo achado do A2: a TS-AUC OOF era agregada na grade com thinning, que da a `t>400`
# 16,5% do peso quando o board lhe da 31,9%. Sob a ponderacao correta o E0c passa de
# "+0,0039, IC [-0,0015; +0,0100] (inclui 0)" para "+0,0059, IC [+0,0006; +0,0117] (EXCLUI 0)",
# reproduzindo o numero antigo casa a casa na grade antiga.
#
# RESULTADOS DA FASE 1 (particao 42, K=2, baseline oof_b6c6_joint_bag2 = 0,62640):
#   p2_brw   +0,0024 [-0,0012, +0,0057]  <- lider, sinal consistente em 50-150/150-400/t>400
#   p4_cap   +0,0012 [-0,0011, +0,0036]     2,4x mais rodadas, ZERO em t>400
#
# OPERACIONAL: rodar UM job pesado por vez. Treino concorrendo com o bootstrap de `compare_oof.py`
# (n_jobs=-1) ficou com 1,2 nucleo de 12 e depois TRAVOU (51 threads em Wait, deadlock de OpenMP).
# Sem `| tee` no caminho do stderr do tqdm; saida por arquivo; OMP_NUM_THREADS fixado.
#
# RESUMIVEL: pula a semente cujo diretorio ja existe. O wrapper deste ambiente e morto a esmo
# (NOTAS_AGENTES.md §7) -- relancar o script continua de onde parou, custando no maximo uma semente.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=6
export OMP_WAIT_POLICY=PASSIVE

ROWS="data/processed/train_rows_bocpd.parquet"     # 193 feat = V4(183) + mrep(6) + bocpd(4)
BASE="--detectability-mode soft --feature-contri-meta 0.6"
DROP="--drop-prefix spec_ ord_ mrep_"              # -> 187 feat, o conjunto do incumbente MEDIDO
LOGDIR="artifacts/reports/polimento"
mkdir -p "$LOGDIR"

run_arm () {           # run_arm <nome> <sementes> <args...>
  local name="$1"; local seeds="$2"; shift 2
  for S in $seeds; do
    if [ -d "artifacts/models/${name}_s$S" ]; then echo "skip ${name}_s$S (ja existe)"; continue; fi
    echo "=== INICIO ${name} semente $S @ $(date +%H:%M:%S) ==="
    python -u scripts/train.py --rows "$ROWS" $BASE --boost-seed "$S" \
      --out "artifacts/models/${name}_s$S" "$@" > "$LOGDIR/${name}_s$S.log" 2>&1 \
      || { echo "FALHOU ${name} s$S (ver $LOGDIR/${name}_s$S.log)"; return 1; }
    echo "=== FIM    ${name} semente $S @ $(date +%H:%M:%S) ==="
  done
  local inputs=""
  for S in $seeds; do inputs="$inputs artifacts/models/oof_${name}_s$S.parquet"; done
  local k; k=$(echo $seeds | wc -w)
  python scripts/avg_oof.py --inputs $inputs --out "artifacts/models/oof_${name}_bag$k.parquet" \
    > "$LOGDIR/${name}_avg.log" 2>&1
  echo "=== BAG$k PRONTO ${name} @ $(date +%H:%M:%S) ==="
}

if [ ! -f artifacts/models/oof_b6c6_joint_bag2.parquet ]; then
  python scripts/avg_oof.py \
    --inputs artifacts/models/oof_b6c6_joint_s777.parquet artifacts/models/oof_b6c6_joint_s101.parquet \
    --out artifacts/models/oof_b6c6_joint_bag2.parquet > "$LOGDIR/baseline_avg.log" 2>&1
fi

# ============================ FASE 2 -- promover o lider ============================
# O p2_brw e o unico com Δ acima da barra e sinal consistente por bucket. A regra da campanha e
# explicita: nada entra sem replica na particao 43. Como a particao 43 nao tem baseline com ESTE
# conjunto de features (187 com BOCPD), o baseline dela tem de ser treinado junto -- comparar
# contra um baseline de outro conjunto mediria o conjunto, nao o braco.
run_arm p2_brw        "202 303"  $DROP --base-rate-weighted                       || exit 1
run_arm b6c6_partB    "777 101"  $DROP --fold-seed 43                             || exit 1
run_arm p2_brw_partB  "777 101"  $DROP --base-rate-weighted --fold-seed 43        || exit 1

# ============================ FASE 3 -- resto da triagem ============================
run_arm p3_lintree    "777 101"  $DROP --linear-tree                              || exit 1
run_arm p1_deploy193  "777 101"                                                   || exit 1
run_arm p5_maxbin1023 "777 101"  $DROP --max-bin 1023                             || exit 1
run_arm p6_contri04   "777 101"  $DROP --feature-contri-meta 0.4                  || exit 1
run_arm p7_contri08   "777 101"  $DROP --feature-contri-meta 0.8                  || exit 1

echo "=== CAMPANHA COMPLETA @ $(date +%H:%M:%S) ==="
