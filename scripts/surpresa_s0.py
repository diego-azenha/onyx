#!/usr/bin/env python
"""S0 (knowledge/frentes/surpresa-acumulada/S0-sanidade-sintetica.md): sanidade sintética do detector
de surpresa acumulada. Mede os itens 1-4 da hipótese registrada; o item 5 é `tests/surpresa`."""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from sbrt.robustness.generators import CONTROLLED_SCENARIOS, generate
from surpresa import carregar_config
from surpresa.evidencia import Alternativas
from surpresa.modelo import ajustar, fluxos
from surpresa.stream import pontuar_serie

import sys
sys.path.insert(0, "tests")
from surpresa_det.geradores import GERADORES  # noqa: E402  (mesmos geradores dos testes)


SCORES = ("sr", "sr_eta", "sr_cal", "ingenua_z")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="artifacts/surpresa/s0.json")
    args = ap.parse_args()
    cfg = carregar_config()
    alt = Alternativas(cfg)
    rng = np.random.default_rng(2026)
    res: dict = {"nulo": {}, "sr_t500": {}}

    # 1 e 2: nulo universal e deriva comum de sr
    for tipo, gen in GERADORES.items():
        ga, gb = [], []
        fim = {k: [] for k in SCORES}
        for _ in tqdm(range(args.n), desc=f"nulo {tipo}"):
            x = gen(rng, 2500)
            h, on = x[:2000], x[2000:]
            f = fluxos(ajustar(h, cfg), on)
            ga.append(f["g_a"]); gb.append(f["g_b"])
            o = pontuar_serie(h, on, cfg, alt)
            for k in SCORES:
                fim[k].append(o[k][-1])
        ga, gb = np.concatenate(ga), np.concatenate(gb)
        res["nulo"][tipo] = {"ga_media": ga.mean(), "ga_dp": ga.std(), "gb_media": gb.mean(), "gb_dp": gb.std()}
        res["sr_t500"][tipo] = {f"{k}_{nome}": float(fn(fim[k])) for k in SCORES
                                for nome, fn in (("mediana", np.median), ("q95", lambda v: np.quantile(v, 0.95)))}

    # 3: poder em sigma 1,0 -> 1,2 desde t=1, contra janela de 20 pontos
    T = 300
    fam_q, fam_c, jan_q, jan_c = [], [], [], []
    for _ in tqdm(range(args.n), desc="poder 1,2"):
        h = rng.standard_normal(2000)
        on = rng.standard_normal(T)
        for dest_f, dest_j, serie in ((fam_c, jan_c, on), (fam_q, jan_q, on * 1.2)):
            dest_f.append(pontuar_serie(h, serie, cfg, alt)["fam_escala_sobe"])
            jan = np.full(T, np.nan)
            c2 = np.concatenate([[0.0], np.cumsum(serie**2)])
            jan[19:] = np.log((c2[20:] - c2[:-20]) / 20.0)
            dest_j.append(jan)
    fam_q, fam_c, jan_q, jan_c = map(np.array, (fam_q, fam_c, jan_q, jan_c))

    def primeiro_t(q, c):
        ok = np.median(q, axis=0) > np.nanquantile(c, 0.95, axis=0)
        return int(np.argmax(ok) + 1) if ok.any() else None

    res["poder_1_2"] = {"surpresa_primeiro_t": primeiro_t(fam_q, fam_c),
                        "janela20_primeiro_t": primeiro_t(jan_q, jan_c),
                        "janela20_passa_em_150": bool(np.median(jan_q[:, 149]) > np.nanquantile(jan_c[:, 149], 0.95))}

    # 4: cenários do Onyx com controle, AUC cenário x controle 100 passos depois de tau
    res["cenarios"] = {}
    for sid in CONTROLLED_SCENARIOS:
        q_, c_ = {k: [] for k in SCORES}, {k: [] for k in SCORES}
        for seed in tqdm(range(100), desc=sid):
            h, on, tau = generate(sid, seed)
            hc, onc, _ = generate(sid + "_ctrl", seed)
            i = min(tau + 100, len(on) - 1)
            oq, oc = pontuar_serie(h, on, cfg, alt), pontuar_serie(hc, onc, cfg, alt)
            for k in SCORES:
                q_[k].append(oq[k][i]); c_[k].append(oc[k][i])
        y = np.r_[np.ones(100), np.zeros(100)]
        res["cenarios"][sid] = {f"auc_{k}": roc_auc_score(y, np.r_[q_[k], c_[k]]) for k in SCORES}

    txt = json.dumps(res, indent=2, default=float)
    print(txt)
    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
