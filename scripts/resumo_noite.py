#!/usr/bin/env python
"""Consolida os bracos da fila noturna (run_noite.sh + run_noite2.sh) numa tabela unica.

Le os JSONs de `compare_oof.py` e imprime ponto, IC e veredito pelo protocolo da campanha:
barra de 0,0014 no agregado, IC 95% pareado, e o bucket `t>400` destacado porque o A2 mostrou que ele
vale 31,9% do peso do board (nao 16,5%). Nao decide nada -- so poe lado a lado.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BARRA = 0.0014

# rotulo -> (arquivo json, descricao curta, baseline usado)
ARMS = [
    ("p6_contri04_k4", "C1  contri=0,4 (K=4 vs K=4)", "b6c6_joint_bag4"),
    ("p17_nobundle", "D5  enable_bundle=false", "b6c6_joint_bag2"),
    ("p18_dart", "D3  boosting=dart", "b6c6_joint_bag2"),
    ("p19_dropnan", "A5b 15 colunas 100% NaN derrubadas", "b6c6_joint_bag2"),
    ("p20_contri05", "C1  contri=0,5", "b6c6_joint_bag2"),
    ("p21_contri07", "C1  contri=0,7", "b6c6_joint_bag2"),
    ("b1_bag7_p43", "B1  bag K=7 na particao 43 (divida)", "b6c6_partB_bag4"),
    ("p12_e1", "E1  des-thinning (grade diferente: ver nota)", "b6c6_joint_bag2"),
    ("p21_contri07_k4", "C1  contri=0,7 (K=4 vs K=4)", "b6c6_joint_bag4"),
    ("p22_contri_ext", "C1(ii) conjunto penalizado estendido", "b6c6_joint_bag2"),
    ("p23_deltamean", "C2(ii) delta_mean no d_i", "b6c6_joint_bag2"),
]


def veredito(pt: float, lo: float, hi: float) -> str:
    exclui = lo > 0 or hi < 0
    if not exclui:
        return "inconclusivo" if pt > 0 else "negativo/inconclusivo"
    if pt >= BARRA:
        return "PASSA (ponto>=barra, IC exclui 0)"
    if pt > 0:
        return "positivo mas SUB-BARRA"
    return "NEGATIVO (IC exclui 0)"


def main() -> None:
    print(f"\n{'braco':44s} {'geral':>9s} {'IC 95%':>22s}  {'t>400':>9s}  veredito")
    print("-" * 118)
    faltando = []
    for rot, desc, _bl in ARMS:
        p = ROOT / "artifacts" / "reports" / f"polimento_{rot}.json"
        if not p.exists():
            faltando.append(desc)
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        g = d["overall"]
        t4 = d.get("t>400", {})
        ic = f"[{g['ci_lo']:+.4f}; {g['ci_hi']:+.4f}]"
        t4s = f"{t4.get('point_delta', float('nan')):+.4f}" if t4 else "     -"
        print(f"{desc:44s} {g['point_delta']:+9.4f} {ic:>22s}  {t4s:>9s}  {veredito(g['point_delta'], g['ci_lo'], g['ci_hi'])}")

    if faltando:
        print("\nainda sem resultado:")
        for d in faltando:
            print(f"  - {d}")

    print(
        "\nnotas de leitura:\n"
        "  - barra da campanha = 0,0014 no agregado; adocao exige REPLICA na particao 43.\n"
        "  - todos os bracos foram treinados com `linear_tree=false` para casar com os baselines\n"
        "    pre-D1 -- mede-se o braco, nao o D1.\n"
        "  - E1 vive numa grade com MAIS linhas: o compare_oof casa so o subconjunto comum, entao o\n"
        "    Delta pareado SUBESTIMA. A leitura honesta e o nivel na grade do board.\n"
    )


if __name__ == "__main__":
    main()
