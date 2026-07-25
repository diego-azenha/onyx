#!/usr/bin/env python
"""E0a/E0b (DIAGNOSTICO_ESTRUTURAL.md §5): o TETO INFORMACIONAL da tarefa, medido sem treinar nada.

A pergunta que quatro meses de campanha nunca respondeu: *quanta* TS-AUC a informação disponível
permite? Todo braço até aqui mediu contra o incumbente; este script mede contra o possível.

Construção (por série, tudo causal, tudo padronizado pelo nulo do PRÓPRIO histórico):

  1. PIT contra o histórico: u_i = posto de online_i dentro do histórico / (n_h+1). Se nada mudou,
     u ~ U(0,1) i.i.d. — o histórico É a hipótese nula, empiricamente.
  2. Seis componentes ortonormais em u (teste suave de Neyman + dependência):
     L1..L4 (Legendre: nível, escala, assimetria, curtose) e dois produtos defasados de L1
     (dependência lag-1 e lag-2). Cada um tem média 0 e variância 1 sob H0 i.i.d.
  3. Sobre uma janela [s,t): c_k = (soma de h_k) / sqrt(m). Padronizado pelo nulo da própria série
     (média/dp das MESMAS janelas deslizando pelo histórico — é isto que absorve a inflação de
     variância de uma série dependente, o problema que a studentização transversal do A3 não
     resolvia), vira z_k; S = soma z_k^2 é uma estatística dois-amostras janela-vs-histórico.
  4. log S é padronizado outra vez pelo nulo do histórico no mesmo m -> zs(m,t), escala comum entre
     séries. O máximo sobre a grade geométrica de m é padronizado pelo nulo do MÁXIMO no mesmo
     número de candidatos j -> z_scan(t), de novo escala comum (senão positivos com janela única e
     negativos com scan viveriam em distribuições diferentes e a AUC mediria o artefato).

Saídas (uma coluna de score por linha (id,t), comparável a qualquer OOF via compare_oof.py):

  e0b_scan    E0b — skyline CAUSAL: max sobre a grade de m, sem oráculo. Teto de uma estatística
              dois-amostras com localização honesta.
  e0a_oracle  E0a — janela de referência única [r,t), onde r = tau para as linhas positivas.
  e0a_aug     E0a — max(janela de referência, scan): "a melhor janela disponível".

CONTROLE DE PLACEBO (sem ele os dois E0a medem o rótulo, não a informação). Toda linha NEGATIVA
recebe também uma janela de referência, com r sorteado da distribuição EMPÍRICA de tau condicionada a
r<t — a mesma distribuição de comprimento de janela dos positivos. Sem isso, `e0a_aug` seria um
max sobre j+1 candidatos nos positivos contra j nos negativos, e max(a,b)>=b sozinho já produz AUC:
a primeira versão deste script marcava 0,66 com o placebo desligado e 0,60 com ele ligado — o
"teto" era, em 60% do seu excesso, o próprio rótulo entrando pela porta dos fundos.

NÃO é estimador de leaderboard (README, plano §9.0): é um TETO relativo, medido no mesmo OOF grid do
incumbente, para decidir se o platô é da tarefa ou da nossa base de features.

LEITURA (assimétrica, e a §5 não a registrou): isto é um PISO do teto, não um teto. Alto => existe
folga demonstrada. Baixo => só prova que ESTA família de estatísticas não bate o modelo.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

M_GRID = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024], dtype=np.int64)
LAGS = np.array([0, 0, 0, 0, 1, 2], dtype=np.int64)   # defasagem consumida por cada componente
K = len(LAGS)
_EPS = 1e-6


def _components(u: np.ndarray) -> np.ndarray:
    """(n,) PIT em (0,1) -> (n,K) componentes de média 0 e variância 1 sob H0 i.i.d."""
    x = 2.0 * u - 1.0
    x2 = x * x
    out = np.empty((len(u), K), dtype=np.float64)
    out[:, 0] = np.sqrt(3.0) * x
    out[:, 1] = np.sqrt(5.0) * 0.5 * (3.0 * x2 - 1.0)
    out[:, 2] = np.sqrt(7.0) * 0.5 * (5.0 * x2 * x - 3.0 * x)
    out[:, 3] = 3.0 * 0.125 * (35.0 * x2 * x2 - 30.0 * x2 + 3.0)
    # dependência: produto de L1 com sua defasada. Média 0 sob independência, variância 1 (L1 tem
    # variância 1 e os fatores são independentes). Posição i carrega o par (i-l, i), então a soma
    # sobre a janela [s,t) só pode começar em s+l -- é o que LAGS registra.
    d1 = np.empty(len(u)); d1[0] = 0.0; d1[1:] = out[1:, 0] * out[:-1, 0]
    d2 = np.empty(len(u)); d2[:2] = 0.0; d2[2:] = out[2:, 0] * out[:-2, 0]
    out[:, 4] = d1
    out[:, 5] = d2
    return out


def _prefix(H: np.ndarray) -> np.ndarray:
    P = np.zeros((H.shape[0] + 1, H.shape[1]), dtype=np.float64)
    np.cumsum(H, axis=0, out=P[1:])
    return P


def _window_c(P: np.ndarray, ends: np.ndarray, starts: np.ndarray, m: np.ndarray) -> np.ndarray:
    """c_k = soma de h_k sobre [start+lag_k, end) / sqrt(m - lag_k), vetorizado sobre janelas."""
    c = np.empty((len(ends), K), dtype=np.float64)
    for k in range(K):
        lk = LAGS[k]
        c[:, k] = (P[ends, k] - P[starts + lk, k]) / np.sqrt(np.maximum(m - lk, 1.0))
    return c


def _series_null(hist: np.ndarray, m_grid: np.ndarray) -> dict:
    """Nulo da própria série: desliza cada janela de m pelo histórico e mede o que H0 produz."""
    n = len(hist)
    order = np.argsort(hist, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    u = (ranks + 0.5) / n
    P = _prefix(_components(u))

    mean_c = np.zeros((len(m_grid), K))
    sd_c = np.ones((len(m_grid), K))
    mu_log = np.zeros(len(m_grid))
    sd_log = np.ones(len(m_grid))
    zs_hist = []                      # zs(m, e) sobre os fins e >= m, alinhados pelo fim
    for i, m in enumerate(m_grid):
        ends = np.arange(m, n + 1, dtype=np.int64)
        starts = ends - m
        c = _window_c(P, ends, starts, np.full(len(ends), float(m)))
        mean_c[i] = c.mean(axis=0)
        sd_c[i] = np.where(c.std(axis=0) > 1e-12, c.std(axis=0), 1.0)
        z = (c - mean_c[i]) / sd_c[i]
        s = np.log((z * z).sum(axis=1) + _EPS)
        mu_log[i] = s.mean()
        sd_log[i] = s.std() if s.std() > 1e-12 else 1.0
        zs_hist.append((s - mu_log[i]) / sd_log[i])

    # nulo do MÁXIMO com j candidatos: alinhado pelo fim, só os fins onde os j candidatos existem.
    mu_max = np.zeros(len(m_grid))
    sd_max = np.ones(len(m_grid))
    for j in range(1, len(m_grid) + 1):
        e0 = int(m_grid[j - 1])
        stack = np.stack([zs_hist[i][e0 - int(m_grid[i]):] for i in range(j)], axis=0)
        mx = stack.max(axis=0)
        mu_max[j - 1] = mx.mean()
        sd_max[j - 1] = mx.std() if mx.std() > 1e-12 else 1.0

    return {"sorted_hist": hist[order], "mean_c": mean_c, "sd_c": sd_c,
            "mu_log": mu_log, "sd_log": sd_log, "mu_max": mu_max, "sd_max": sd_max}


def _whiten(hist: np.ndarray, online: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Resíduos do MESMO H0 que a produção usa (state/h0.py: AR(p) + sazonal, congelado no init).
    `fit_h0` é replicado vetorialmente para o online — `whiten_step` é laço em Python e aqui são 5 M
    passos. A cauda do histórico semeia os lags, igual a `seed_lag_buffer` (continuidade em tau=0)."""
    from sbrt.state.h0 import fit_h0
    p = fit_h0(hist, cfg)
    lo, hi = cfg.h0.clip_e
    L, T = p.lag_capacity, len(online)
    z = np.concatenate([hist[-L:], online])
    xh = np.full(T, p.c, dtype=np.float64)
    for j, phi_j in enumerate(p.phi):
        xh += phi_j * z[L - 1 - j: L - 1 - j + T]
    if p.seasonal_lag is not None:
        s = p.seasonal_lag
        xh += p.seasonal_coef * z[L - s: L - s + T]
    e_on = np.clip((online - xh) / p.sigma_e, lo, hi)
    return np.clip(p.e_hist, lo, hi), e_on


def _ref_starts(T: int, tau: int, taus_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """r_t para t=1..T. Linha positiva (tau<t): r=tau. Linha negativa: r sorteado da distribuição
    empírica de tau condicionada a r<t — o placebo que iguala os dois lados (ver docstring)."""
    t_ax = np.arange(1, T + 1, dtype=np.int64)
    k = np.searchsorted(taus_pos, t_ax, side="left")          # quantos taus reais são < t
    idx = np.minimum((rng.random(T) * np.maximum(k, 1)).astype(np.int64), np.maximum(k - 1, 0))
    r = np.where(k > 0, taus_pos[idx], 0)
    if tau >= 0:
        r = np.where(t_ax > tau, tau, r)
    return np.minimum(r, t_ax - 1)


def _score_series(hist: np.ndarray, online: np.ndarray, tau: int, min_hist: int,
                  taus_pos: np.ndarray, rng: np.random.Generator) -> tuple:
    """-> (e0b_scan, e0a_oracle, e0a_aug) com um valor por passo t=1..T (t = posição online + 1)."""
    T = len(online)
    zeros = np.zeros(T, dtype=np.float64)
    n = len(hist)
    if n < min_hist or T < 1:
        return zeros, zeros.copy(), zeros.copy()

    # Grade de m viável. Depende SÓ do histórico (m <= n/4 => >= 4 janelas efetivamente
    # descorrelacionadas para estimar o nulo), nunca de T: o comprimento total do fluxo online é
    # informação do FUTURO em qualquer passo t. Não é escrúpulo teórico -- medido nesta base,
    # `-len(online)` sozinho vale TS-AUC 0,65, acima do incumbente (0,607), porque um positivo em t
    # tem tau < t e portanto fluxo mais curto que um negativo sobrevivente. Um teto contaminado por
    # esse canal não mediria informação nenhuma. O corte por t entra depois, via `j`.
    m_grid = M_GRID[M_GRID <= n // 4]
    if len(m_grid) == 0:
        return zeros, zeros.copy(), zeros.copy()
    nul = _series_null(hist, m_grid)

    sh = nul["sorted_hist"]
    rk = 0.5 * (np.searchsorted(sh, online, side="left") + np.searchsorted(sh, online, side="right"))
    u = np.clip((rk + 0.5) / (n + 1), 1e-9, 1 - 1e-9)
    P = _prefix(_components(u))

    # --- E0b: scan sobre a grade, padronizado duas vezes pelo nulo da série -------------------
    zs = np.full((len(m_grid), T + 1), -np.inf)          # indexado pelo FIM e = t
    for i, m in enumerate(m_grid):
        ends = np.arange(m, T + 1, dtype=np.int64)
        c = _window_c(P, ends, ends - m, np.full(len(ends), float(m)))
        z = (c - nul["mean_c"][i]) / nul["sd_c"][i]
        s = np.log((z * z).sum(axis=1) + _EPS)
        zs[i, ends] = (s - nul["mu_log"][i]) / nul["sd_log"][i]

    run_max = np.maximum.accumulate(zs, axis=0)          # linha j-1 = max sobre os j primeiros m
    t_ax = np.arange(T + 1)
    j = np.searchsorted(m_grid, t_ax, side="right")      # nº de candidatos disponíveis em t
    scan = np.zeros(T + 1)
    ok = j > 0
    idx = np.clip(j - 1, 0, len(m_grid) - 1)
    scan[ok] = (run_max[idx[ok], t_ax[ok]] - nul["mu_max"][idx[ok]]) / nul["sd_max"][idx[ok]]
    e0b = scan[1:]                                       # t = 1..T

    # --- E0a: janela de referência [r,t) — r = tau nas linhas positivas, placebo nas negativas ---
    m0 = int(m_grid[0])
    ta = np.arange(1, T + 1, dtype=np.int64)
    ref = _ref_starts(T, tau, taus_pos, rng)
    use = ta >= m0                       # abaixo de m0 não há janela avaliável (vale para os dois lados)
    e0a, e0a_aug = e0b.copy(), e0b.copy()
    if use.any():
        te = ta[use]
        m_or = np.maximum(te - ref[use], m0).astype(np.float64)
        c = _window_c(P, te, (te - m_or).astype(np.int64), m_or)
        lg, lgrid = np.log(m_or), np.log(m_grid.astype(np.float64))
        z = np.empty_like(c)
        for k in range(K):
            z[:, k] = ((c[:, k] - np.interp(lg, lgrid, nul["mean_c"][:, k]))
                       / np.interp(lg, lgrid, nul["sd_c"][:, k]))
        s = np.log((z * z).sum(axis=1) + _EPS)
        zo = (s - np.interp(lg, lgrid, nul["mu_log"])) / np.interp(lg, lgrid, nul["sd_log"])
        e0a[te - 1] = zo
        e0a_aug[te - 1] = np.maximum(zo, e0b[te - 1])
    return e0b, e0a, e0a_aug


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x-train", default="data/X_train.parquet")
    ap.add_argument("--y-index", default="data/y_train_index.parquet")
    ap.add_argument("--out", default="artifacts/models/oof_e0_skyline.parquet")
    ap.add_argument("--limit", type=int, default=None, help="só as N primeiras séries (smoke)")
    ap.add_argument("--min-hist", type=int, default=50)
    ap.add_argument("--repr", default="whitened", choices=["whitened", "raw"],
                    help="representação sobre a qual a estatística é computada. `whitened` usa o "
                         "MESMO H0 da produção (AR(p)+sazonal congelado no init, state/h0.py) — é a "
                         "comparação justa com o incumbente, que só vê resíduos. `raw` mede o que se "
                         "perde/ganha com o branqueamento (§2 do diagnóstico: o filtro MOLDA o que a "
                         "base enxerga).")
    ap.add_argument("--seed", type=int, default=42, help="RNG do placebo de r nas linhas negativas")
    args = ap.parse_args()

    t0 = time.time()
    X = pd.read_parquet(args.x_train).reset_index()
    yi = pd.read_parquet(args.y_index)
    ids = X["id"].to_numpy(np.int64)
    vals = X["value"].to_numpy(np.float64)
    per = X["period"].to_numpy(np.int8)
    del X
    uniq, starts = np.unique(ids, return_index=True)
    bounds = np.append(starts, len(ids))
    if args.limit:
        uniq, bounds = uniq[: args.limit], bounds[: args.limit + 1]
    print(f"{len(uniq)} séries carregadas em {time.time() - t0:.1f}s", flush=True)

    tau_map = yi["tau_index"].to_dict()
    taus_pool = np.sort(yi.loc[yi["tau_index"] >= 0, "tau_index"].to_numpy(np.int64))
    cfg = None
    if args.repr == "whitened":
        from dataclasses import replace as _replace

        from sbrt.config import DEFAULT_CONFIG_PATH, load_config
        cfg = load_config(DEFAULT_CONFIG_PATH)
        # F1 (null_stats por réplicas) é 97% do custo de fit_h0 e não entra aqui: o skyline calibra
        # seu próprio nulo. O ajuste AR/sazonal — a única coisa usada — é idêntico.
        cfg = _replace(cfg, calibration=_replace(cfg.calibration, enabled=False))

    out_id, out_t, out_y, out_b, out_a, out_g = [], [], [], [], [], []
    t1 = time.time()
    for i, sid in enumerate(uniq):
        a, b = bounds[i], bounds[i + 1]
        v, p = vals[a:b], per[a:b]
        hist, online = v[p == 1], v[p == 2]
        tau = int(tau_map.get(int(sid), -1))
        if cfg is not None and len(hist) >= args.min_hist and len(online) > 0:
            hist, online = _whiten(hist, online, cfg)
        rng = np.random.default_rng(args.seed * 1_000_003 + int(sid))
        e0b, e0a, e0ag = _score_series(hist, online, tau, args.min_hist, taus_pool, rng)
        T = len(online)
        tt = np.arange(1, T + 1, dtype=np.int32)
        out_id.append(np.full(T, sid, dtype=np.int32))
        out_t.append(tt)
        out_y.append(((tau >= 0) & (tau < tt)).astype(np.int8))
        out_b.append(e0b.astype(np.float32))
        out_a.append(e0a.astype(np.float32))
        out_g.append(e0ag.astype(np.float32))
        if (i + 1) % 1000 == 0:
            el = time.time() - t1
            print(f"  {i + 1}/{len(uniq)} séries · {el:.0f}s · eta {el / (i + 1) * (len(uniq) - i - 1):.0f}s",
                  flush=True)

    df = pd.DataFrame({"id": np.concatenate(out_id), "t": np.concatenate(out_t),
                       "y": np.concatenate(out_y), "e0b_scan": np.concatenate(out_b),
                       "e0a_oracle": np.concatenate(out_a), "e0a_aug": np.concatenate(out_g)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"skyline salvo em {args.out} ({len(df)} linhas) em {time.time() - t0:.0f}s")

    from sbrt.evaluation.ts_auc import weighted_ts_auc
    t, y = df["t"].to_numpy(), df["y"].to_numpy()
    for col in ("e0b_scan", "e0a_oracle", "e0a_aug"):
        print(f"  TS-AUC (todas as linhas) {col:12s} {weighted_ts_auc(t, y, df[col].to_numpy()):.4f}")


if __name__ == "__main__":
    main()
