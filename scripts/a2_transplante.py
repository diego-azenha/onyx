#!/usr/bin/env python
"""A2 (knowledge/frentes/teto-offline/A2-transplante.md): eventos de quebra NOVOS por transplante.

Doadora j (quebra real com >= 100 pontos pós-quebra): AR(5) ajustado no histórico (pré) e no trecho
pós-quebra (pós). A assinatura da quebra é (Δφ, razão de escala, mapa de quantis dos resíduos
padronizados pré -> pós, Δmédia em unidades do dp pré).

Hospedeira i: corte como no A1 (novo histórico + novo online) dentro da região H0 de uma série real.
Até τ', o online é a continuação REAL de i. A partir de τ', os resíduos AR reais de i passam pelo mapa
de quantis da doadora, são re-escalados e re-coloridos com φ_i + Δφ (encolhido até ficar estável),
partindo dos valores reais anteriores. Negativos: re-cortes sem quebra (o mesmo do A1), na mesma
proporção da competição (50%).

id = 10000*k + id_hospedeira; o fold segue a hospedeira (wrapper do A1). **Sem vazamento:** a doadora
é sorteada entre as séries do MESMO fold da hospedeira (folds do Onyx, `cfg.seed`). A série
transplantada só treina nos folds em que a hospedeira é treino, e ali a doadora também é treino; a
doadora só é validada no fold em que a transplantada não entra. Para outra partição (`--fold-seed`),
é preciso regenerar.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.model.dataset import SeriesRecord, build_training_rows  # noqa: E402

P = 5
MIN_H, MAX_H, MIN_ON, MAX_ON, MIN_POS = 1000, 5000, 10, 999, 100


def _ar(x: np.ndarray):
    lags = sliding_window_view(x[:-1], P)[:, ::-1]
    X = np.column_stack([np.ones(len(lags)), lags])
    coef, *_ = np.linalg.lstsq(X, x[P:], rcond=None)
    r = x[P:] - X @ coef
    return coef[1:], float(coef[0]), r


def _estavel(phi: np.ndarray) -> np.ndarray:
    for _ in range(30):
        raizes = np.roots(np.r_[1.0, -phi])
        if np.all(np.abs(raizes) < 0.98):
            return phi
        phi = phi * 0.9
    return phi * 0.0


def assinatura(h: np.ndarray, post: np.ndarray) -> dict:
    phi_a, _, ra = _ar(h)
    phi_b, _, rb = _ar(post)
    sa, sb = ra.std(), rb.std()
    return {"dphi": phi_b - phi_a, "escala": sb / sa, "za": np.sort(ra / sa), "zb": np.sort(rb / sb),
            "dmedia": (post.mean() - h.mean()) / h.std()}


def aplicar(assin: dict, hist: np.ndarray, pre: np.ndarray, cont: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Transforma a continuação real `cont` da hospedeira (depois de hist ++ pre). `lam` encolhe a
    assinatura (0 = sem quebra, 1 = assinatura estimada inteira): a estimada inclui ruído de estimação,
    que exagera a mudança (transferência real->sintético 0,736 contra 0,666 real->real com lam=1)."""
    phi_i, c_i, r_h = _ar(hist)
    s_i = r_h.std()
    base = np.concatenate([hist, pre, cont])
    k0 = len(hist) + len(pre)
    lags = sliding_window_view(base[k0 - P:-1], P)[:, ::-1]
    u = (cont - c_i - lags @ phi_i) / s_i                               # resíduos reais da hospedeira
    qa = (np.arange(len(assin["za"])) + 0.5) / len(assin["za"])
    qb = (np.arange(len(assin["zb"])) + 0.5) / len(assin["zb"])
    F = np.interp(u, assin["za"], qa)
    v = np.interp(F, qb, assin["zb"]) * assin["escala"]
    # caudas: fora do suporte da doadora, mantém o excesso da hospedeira (evita truncar extremos)
    v = np.where(u > assin["za"][-1], assin["zb"][-1] * assin["escala"] + (u - assin["za"][-1]) * assin["escala"], v)
    v = np.where(u < assin["za"][0], assin["zb"][0] * assin["escala"] + (u - assin["za"][0]) * assin["escala"], v)
    esc_lam = assin["escala"] ** lam
    v = ((1.0 - lam) * u + lam * v / assin["escala"]) * esc_lam
    phi_n = _estavel(phi_i + lam * assin["dphi"])
    x = list(base[k0 - P:k0])
    out = np.empty(len(cont))
    for t in range(len(cont)):
        val = c_i + float(np.dot(phi_n, x[::-1][:P])) + s_i * v[t]
        out[t] = val
        x.append(val); x.pop(0)
    return out + lam * assin["dmedia"] * hist.std()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0, help="encolhimento da assinatura")
    ap.add_argument("--fold-seed", type=int, default=None, help="partição de folds (default cfg.seed)")
    ap.add_argument("--apenas-registros", action="store_true", help="só gera e salva os registros (teste)")
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    yi = pd.read_parquet("data/y_train_index.parquet")
    series = _carregar_series(Path("data"))
    rng = np.random.default_rng(777000 + args.k)
    from sbrt.evaluation.splits import grouped_stratified_kfold
    meta = pd.read_parquet("data/processed/train_rows.parquet", columns=["id", "t", "y"])
    fold_de = {}
    for f, (_, va) in enumerate(grouped_stratified_kfold(meta, cfg.lightgbm.n_folds,
                                                         args.fold_seed if args.fold_seed is not None else cfg.seed)):
        for i in np.unique(meta["id"].to_numpy()[va]):
            fold_de[int(i)] = f
    doadoras: dict[int, list] = {}
    for sid, h, on in series:
        tau = int(yi.loc[sid, "tau_index"])
        if tau >= 0 and len(on) - tau >= MIN_POS:
            doadoras.setdefault(fold_de[sid], []).append(assinatura(h, on[tau:]))
    recs = []
    for sid, h, on in series:
        tau = int(yi.loc[sid, "tau_index"])
        S = np.concatenate([h, on]) if tau < 0 else np.concatenate([h, on[:tau]])   # só a região H0
        if len(S) < MIN_H + MIN_ON:
            continue
        n_on = int(rng.integers(MIN_ON, min(MAX_ON, len(S) - MIN_H) + 1))
        c = int(rng.integers(MIN_H, len(S) - n_on + 1))
        n_h = int(rng.integers(MIN_H, min(MAX_H, c) + 1))
        hist, online = S[c - n_h:c], S[c:c + n_on].copy()
        tau2 = None
        if rng.random() < 0.5:                                           # 50% com quebra, como na competição
            tau2 = int(rng.integers(0, n_on))
            pool = doadoras[fold_de[sid]]
            assin = pool[int(rng.integers(0, len(pool)))]
            online[tau2:] = aplicar(assin, hist, online[:tau2], online[tau2:], args.lam)
        mu, sd = hist.mean(), hist.std()
        recs.append(SeriesRecord(dataset_id=10000 * args.k + sid, x_hist=(hist - mu) / sd,
                                 x_online=(online - mu) / sd, tau_index=tau2))
    pos = sum(r.tau_index is not None for r in recs)
    print(f"{len(recs)} séries transplantadas ({pos} com quebra); doadoras por fold: "
          f"{[len(doadoras[f]) for f in sorted(doadoras)]}", flush=True)
    if args.apenas_registros:
        pd.to_pickle(recs, f"artifacts/offline/a2_registros_k{args.k}_lam{args.lam}.pkl")
        return
    rows = build_training_rows(recs, cfg, progress=True, n_jobs=args.n_jobs)
    out = Path(f"data/processed/aug_transplante_k{args.k}.parquet")
    rows.to_parquet(out)
    pd.to_pickle(recs, f"artifacts/offline/a2_registros_k{args.k}.pkl")
    print(f"gravado {len(rows)} linhas em {out}")


if __name__ == "__main__":
    main()
