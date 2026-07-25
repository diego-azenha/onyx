#!/usr/bin/env bash
# A6 EM K=4 (2026-07-25). Handoff a partir do sprint D3, que e interrompido antes do `p16_ff06`.
#
# ## Por que este script existe
#
# O A6 foi medido em K=2 (sementes 777/101): +0,0015 [-0,0008; +0,0036]. O ponto esta ACIMA da barra
# (0,0014) e o IC inclui zero -- inconclusivo. Neste repo, K=2 ordena bracos mas NAO estima efeitos:
# o A3b caiu de +0,0024 (K=2) para +0,0010 (K=4), o D1 segurou (+0,0025 -> +0,0027). Sem o K=4, a
# deliberacao final sobre o A6 seria um palpite sobre um numero que o projeto ja mostrou nao ser
# confiavel nesse regime.
#
# Acrescenta as sementes 202 e 303, faz a media das QUATRO e mede contra o incumbente K=4 da mesma
# particao (`oof_b6c6_joint_bag4`) -- K=4 contra K=4, que e a comparacao que a licao do D1 exige.
#
# ## Por que ele PARA o sprint antes do `p16_ff06`
#
# `feature_fraction=0,6` aponta para o lado ERRADO do mecanismo que motivava o braco: o defeito
# documentado (NOTAS_AGENTES.md secao 7 / A5 bis) e que 15 colunas 100% NaN em t=50 DILUEM o sorteio
# de `feature_fraction=0,8`. Baixar para 0,6 sorteia MENOS features, entao as colunas NaN ocupam
# fracao MAIOR do sorteio -- piora a diluicao. O braco informativo seria `ff=0,9` ou derrubar as 15
# colunas NaN, nao 0,6. Decisao do usuario: pular.
#
# ## Travas: identicas as do run_sprint_d3.sh (serial, OMP=4, bootstrap n_jobs=2, lockfile)
set -u
cd "$(dirname "$0")/.."

L="artifacts/reports/polimento"
LOCK="artifacts/reports/.run_resto.lock"
mkdir -p "$L"

# ---------------------------------------------------------------- 1. esperar o p15 fechar
# Espera o JSON do `p15_l2_20` aparecer (o sprint grava ao fim da medicao) e SO ENTAO derruba o
# script -- assim o braco em curso nao se perde. Teto de 25 min para nao ficar preso se algo travar.
esperar=0
while [ ! -f artifacts/reports/polimento_p15_l2_20.json ] && [ "$esperar" -lt 1500 ]; do
  sleep 15; esperar=$((esperar + 15))
done
if [ -f artifacts/reports/polimento_p15_l2_20.json ]; then
  echo "=== p15_l2_20 fechou; interrompendo o sprint antes do p16 @ $(date +%H:%M:%S) ==="
else
  echo "!!! timeout esperando o p15 ($((esperar/60)) min). Interrompendo o sprint assim mesmo."
fi

# ---------------------------------------------------------------- 2. derrubar o sprint
# Mata os shells do run_sprint_d3.sh. Filhos python em curso terminam sozinhos (nao ha nenhum aqui:
# acabamos de esperar a medicao fechar). Depois libera o lock, que o trap pode nao ter removido.
python - <<'PY'
import subprocess, sys
try:
    out = subprocess.run(
        ["wmic", "process", "where", "name='bash.exe'", "get", "ProcessId,CommandLine", "/format:csv"],
        capture_output=True, text=True, timeout=30).stdout
except Exception as e:
    print("nao consegui listar processos:", e); sys.exit(0)
alvos = [l.split(",")[-1].strip() for l in out.splitlines() if "run_sprint_d3" in l]
for pid in alvos:
    if pid.isdigit():
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        print("morto shell do sprint:", pid)
if not alvos:
    print("nenhum shell do sprint vivo (ja havia terminado)")
PY
sleep 3
rm -f "$LOCK"

# ---------------------------------------------------------------- 3. tomar o lock
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

export PYTHONIOENCODING=utf-8
export SBRT_ENABLE_BOCPD=1
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONPATH=scripts

BASE="--detectability-mode soft --feature-contri-meta 0.6 --drop-prefix spec_ ord_ mrep_"
ROWS_A6="data/processed/train_rows_a6.parquet"

echo "########## A6 K=4 -- INICIO @ $(date +%H:%M:%S) ##########"

# ---------------------------------------------------------------- 4. sementes 202 e 303
for S in 202 303; do
  [ -d "artifacts/models/p8_a6_s$S" ] && { echo "skip p8_a6_s$S"; continue; }
  echo "=== INICIO p8_a6 s$S @ $(date +%H:%M:%S) ==="
  python -u scripts/train.py --config configs/a6.yaml --rows "$ROWS_A6" $BASE --boost-seed "$S" \
    --out "artifacts/models/p8_a6_s$S" > "$L/p8_a6_s$S.log" 2>&1 \
    || { echo "FALHOU p8_a6 s$S"; exit 1; }
done

# ---------------------------------------------------------------- 5. media das QUATRO
if [ ! -f artifacts/models/oof_p8_a6_bag4.parquet ]; then
  python scripts/avg_oof.py --inputs \
    artifacts/models/oof_p8_a6_s777.parquet artifacts/models/oof_p8_a6_s101.parquet \
    artifacts/models/oof_p8_a6_s202.parquet artifacts/models/oof_p8_a6_s303.parquet \
    --out artifacts/models/oof_p8_a6_bag4.parquet > "$L/p8_a6_avg4.log" 2>&1 \
    || { echo "FALHOU media das 4"; exit 1; }
fi
echo "=== p8_a6 K=4 PRONTO @ $(date +%H:%M:%S) ==="

# ---------------------------------------------------------------- 6. medir K=4 contra K=4
# BASELINE = `oof_b6c6_joint_bag4`, o incumbente K=4 da MESMA particao. Comparar K=4 contra o
# `bag2` seria repetir exatamente o erro que produziu a retratacao do D1.
if [ ! -f artifacts/reports/polimento_p8_a6_k4.json ]; then
  echo "=== MEDINDO p8_a6_k4 @ $(date +%H:%M:%S) ==="
  python -u scripts/compare_oof.py \
    --baseline artifacts/models/oof_b6c6_joint_bag4.parquet \
    --candidate artifacts/models/oof_p8_a6_bag4.parquet \
    --n-boot 200 --n-jobs 2 --out artifacts/reports/polimento_p8_a6_k4.json \
    > "$L/medir_p8_a6_k4.log" 2>&1 || echo "FALHOU medicao p8_a6_k4"
fi

echo "########## A6 K=4 COMPLETO @ $(date +%H:%M:%S) ##########"
tail -12 "$L/medir_p8_a6_k4.log" 2>/dev/null
