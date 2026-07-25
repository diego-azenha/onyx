#!/usr/bin/env bash
# Ponte operacional: espera o `p1_deploy193` (a auditoria do A3 -- o conjunto REALMENTE implantado,
# 193 features, contra o conjunto MEDIDO, 187) terminar, encerra a fase 2 e inicia a fase 4.
#
# Existe como script destacado porque os wrappers de background deste ambiente sao mortos a esmo
# (NOTAS_AGENTES.md §7) e um `until ... done` inline ja se perdeu uma vez no meio da transicao.
#
# A fase 2 sai de cena aqui porque o que restava dela (p5_maxbin1023, p6/p7_contri) vale menos que
# promover o D1: o K=4 derrubou o A3b de +0,0024 para +0,0010, e o D1 esta exatamente no patamar de
# onde o A3b encolheu -- ou seja, o numero dele so significa alguma coisa depois do K=4.
set -u
cd "$(dirname "$0")/.."

until [ -f artifacts/models/oof_p1_deploy193_bag2.parquet ]; do sleep 30; done
echo "p1_deploy193 pronto @ $(date +%H:%M:%S) -- encerrando fase 2"

pkill -f "run_polimento.sh" 2>/dev/null || true
sleep 5
# o train.py sobrevive a morte do pai neste ambiente; encerra-lo explicitamente evita que ele
# continue disputando CPU com a fase 4 (a contencao ja deadlocou o LightGBM uma vez).
pkill -f "scripts/train.py" 2>/dev/null || true
sleep 5

echo "iniciando fase 4 @ $(date +%H:%M:%S)"
exec bash scripts/run_polimento_fase4.sh
