#!/usr/bin/env bash
# GATE DO PACOTE (2026-07-25, sessao de submissao). O unico item essencial antes do notebook.
#
# ## O que ele mede, e por que e este o teste
#
# `HISTORICO.md` §15.17 fecha dizendo que "o pacote final da campanha precisa ser medido INTEIRO",
# e o `CAMPANHA_POLIMENTO.md` poe isso na semana 7 ("medido inteiro vs. incumbente"). O motivo nao e
# formalidade: o repo tem QUATRO nao-aditividades documentadas, e uma delas (`p10_combo`, D1+A3b)
# transformou duas mudancas positivas num -0,0069 em `t>400` com IC excluindo zero.
#
# Os tres itens adotados no `default.yaml` foram medidos cada um SOZINHO contra o mesmo baseline.
# A combinacao que o notebook embutiria nunca existiu como artefato.
#
# ## Por que NAO precisa de treino
#
# O D1 em K=4 na particao 42 ja esta no disco (`oof_p3_lintree_bag4`), e o B3 e pos-processo: aplica
# offline sobre qualquer OOF (`apply_vema_oof.py`, que foi como o +0,0009 foi medido nas duas
# particoes). Entao D1+B3 juntos custam UM bootstrap, nao sete treinos.
#
# FICA DE FORA o B1 (K=7): o artefato do lintree e K=4. O mecanismo do B1 e reducao de variancia --
# propriedade matematica, nao hipotese que possa inverter de sinal -- e o proprio comentario do YAML
# usa esse argumento para adota-lo com uma particao so. O pacote medido aqui e portanto um PISO do
# que sera submetido, nao uma aproximacao de sinal incerto.
#
# ## O que ele faz com o A6 em curso
#
# Espera a semente 303 fechar (para nao jogar fora 8 min de treino), fecha a media K=4 do A6 -- que
# e barata e deixa o artefato pronto -- e SO ENTAO derruba o driver, antes do bootstrap dele. O A6
# nao entra nesta submissao (as duas flags estao `false` no YAML), entao a medicao dele pode esperar;
# o que nao pode esperar e o gate do pacote.
set -u
cd "$(dirname "$0")/.."

L="artifacts/reports/polimento"
export PYTHONIOENCODING=utf-8
export OMP_NUM_THREADS=4
export OMP_WAIT_POLICY=PASSIVE
export PYTHONPATH=scripts

# ---------------------------------------------------------------- 1. esperar a s303 fechar
esperar=0
while [ ! -f artifacts/models/oof_p8_a6_s303.parquet ] && [ "$esperar" -lt 900 ]; do
  sleep 10; esperar=$((esperar + 10))
done
echo "=== s303 fechou (ou timeout de $((esperar/60)) min) @ $(date +%H:%M:%S) ==="
sleep 5

# ---------------------------------------------------------------- 2. media K=4 do A6 (barata)
if [ -f artifacts/models/oof_p8_a6_s303.parquet ] && [ ! -f artifacts/models/oof_p8_a6_bag4.parquet ]; then
  python scripts/avg_oof.py --inputs \
    artifacts/models/oof_p8_a6_s777.parquet artifacts/models/oof_p8_a6_s101.parquet \
    artifacts/models/oof_p8_a6_s202.parquet artifacts/models/oof_p8_a6_s303.parquet \
    --out artifacts/models/oof_p8_a6_bag4.parquet > "$L/p8_a6_avg4.log" 2>&1 \
    && echo "media K=4 do A6 gravada (medicao dele fica para depois)"
fi

# ---------------------------------------------------------------- 3. derrubar o driver do A6
python - <<'PY'
import subprocess
try:
    out = subprocess.run(["wmic", "process", "where", "name='bash.exe'", "get",
                          "ProcessId,CommandLine", "/format:csv"],
                         capture_output=True, text=True, timeout=30).stdout
except Exception as e:
    print("nao consegui listar processos:", e); raise SystemExit(0)
alvos = [l.split(",")[-1].strip() for l in out.splitlines() if "run_a6_k4" in l]
for pid in alvos:
    if pid.isdigit():
        subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        print("derrubado driver do A6:", pid)
if not alvos:
    print("driver do A6 ja havia terminado")
PY
sleep 3
rm -f artifacts/reports/.run_resto.lock

# ---------------------------------------------------------------- 4. O GATE
# Candidato = D1 (linear_tree, K=4) + B3 (v-EMA assimetrico com gate min_t=50), na particao 42.
# Baseline = o incumbente K=4 da MESMA particao, que e o que esta no board com 0,6267.
echo "=== MEDINDO O PACOTE (D1+B3) vs INCUMBENTE @ $(date +%H:%M:%S) ==="
python -u scripts/compare_oof.py \
  --baseline artifacts/models/oof_b6c6_joint_bag4.parquet \
  --candidate artifacts/models/oof_PACOTE_d1b3_p42.parquet \
  --n-boot 200 --n-jobs 2 --out artifacts/reports/polimento_PACOTE_d1b3_p42.json \
  > "$L/medir_pacote.log" 2>&1 || echo "FALHOU a medicao do pacote"

echo "########## GATE DO PACOTE COMPLETO @ $(date +%H:%M:%S) ##########"
cat "$L/medir_pacote.log"
