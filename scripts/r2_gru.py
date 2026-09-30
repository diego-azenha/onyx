#!/usr/bin/env python
"""R2 (knowledge/frentes/rede-sequencial/README.md): rede sequencial de ponta a ponta nas séries brutas.

Roda no venv isolado com PyTorch (o Onyx não depende de torch). Uma GRU codifica os últimos 512 pontos
do histórico e inicializa uma segunda GRU que lê o online passo a passo (x, |x|, x², resíduo AR(5) do
histórico, log t) e emite P(tau <= t) em CADA passo. Perda: BCE com pesos pareados por passo (como o
R1 do Onyx). CV com os folds da partição 42 (artifacts/rnn/dados.npz). Época escolhida por uma
validação interna de 10% das séries de treino. Saída: OOF por passo em artifacts/rnn/oof_*.parquet.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from torch import nn


def ts_auc(t, y, s):
    d = pd.DataFrame({"t": t, "y": y, "s": s})
    d["r"] = d.groupby("t")["s"].rank(method="average")
    g = d.groupby("t")
    n = g["y"].size(); npos = g["y"].sum(); nneg = n - npos
    rpos = d[d.y == 1].groupby("t")["r"].sum().reindex(n.index, fill_value=0.0)
    ok = (npos > 0) & (nneg > 0)
    auc = (rpos[ok] - npos[ok] * (npos[ok] + 1) / 2) / (npos[ok] * nneg[ok])
    w = (npos[ok] * nneg[ok]).astype(float)
    return float((auc * w).sum() / w.sum())


def preparar(D):
    hist, online, off, tau = D["hist"], D["online"], D["off"], D["tau"]
    series = []
    for i in range(len(off) - 1):
        h = hist[i].astype(np.float64); on = online[off[i]:off[i + 1]].astype(np.float64)
        p = 5
        lags = np.lib.stride_tricks.sliding_window_view(h[:-1], p)[:, ::-1]
        X = np.column_stack([np.ones(len(lags)), lags])
        coef, *_ = np.linalg.lstsq(X, h[p:], rcond=None)
        r = h[p:] - X @ coef; sd = r.std() + 1e-6
        allx = np.concatenate([h[-p:], on])
        lo = np.lib.stride_tricks.sliding_window_view(allx[:-1], p)[:, ::-1]
        e = (on - coef[0] - lo @ coef[1:]) / sd
        T = len(on); t = np.arange(1, T + 1)
        fon = np.column_stack([on, np.abs(on), on * on, np.clip(e, -8, 8), np.log(t) / 7.0]).astype(np.float32)
        fh = np.column_stack([h, np.abs(h), h * h]).astype(np.float32)
        y = (t > tau[i]).astype(np.float32) if tau[i] >= 0 else np.zeros(T, np.float32)
        series.append((fh, fon, y))
    return series


class Modelo(nn.Module):
    def __init__(self, he=64, hd=96):
        super().__init__()
        self.enc = nn.GRU(3, he, batch_first=True)
        self.init = nn.Linear(he, hd)
        self.dec = nn.GRU(5, hd, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hd, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, fh, fon):
        _, h = self.enc(fh)
        h0 = torch.tanh(self.init(h))
        out, _ = self.dec(fon, h0)
        return self.head(out).squeeze(-1)


def lotes(idx, series, bs, rng=None):
    idx = np.array(idx)
    if rng is not None:
        idx = rng.permutation(idx)
    # agrupa por comprimento para reduzir padding: ordena dentro de blocos
    ordem = sorted(idx, key=lambda i: len(series[i][2]))
    grupos = [ordem[k:k + bs] for k in range(0, len(ordem), bs)]
    if rng is not None:
        grupos = [grupos[j] for j in rng.permutation(len(grupos))]
    for g in grupos:
        T = max(len(series[i][2]) for i in g)
        fh = torch.tensor(np.stack([series[i][0] for i in g]))
        fon = torch.zeros(len(g), T, 5); y = torch.zeros(len(g), T); m = torch.zeros(len(g), T)
        for j, i in enumerate(g):
            L = len(series[i][2])
            fon[j, :L] = torch.tensor(series[i][1]); y[j, :L] = torch.tensor(series[i][2]); m[j, :L] = 1
        yield g, fh, fon, y, m


def pesos_por_t(idx, series, tmax=1000):
    npos = np.zeros(tmax + 1); n = np.zeros(tmax + 1)
    for i in idx:
        y = series[i][2]; L = len(y)
        n[1:L + 1] += 1; npos[1:L + 1] += y
    nneg = n - npos
    wpos = np.where(n > 0, nneg / np.maximum(n, 1), 0); wneg = np.where(n > 0, npos / np.maximum(n, 1), 0)
    esc = (wpos * npos + wneg * nneg).sum() / max(n.sum(), 1)
    return torch.tensor(wpos / esc, dtype=torch.float32), torch.tensor(wneg / esc, dtype=torch.float32)


def prever(model, idx, series, bs=64):
    model.eval(); out = {}
    with torch.no_grad():
        for g, fh, fon, y, m in lotes(idx, series, bs):
            p = model(fh, fon).numpy()
            for j, i in enumerate(g):
                out[i] = p[j, :len(series[i][2])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epocas", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed)
    D = np.load("artifacts/rnn/dados.npz")
    series = preparar(D); fold = D["fold"]; ids = D["ids"]
    rel = {}
    for f in args.folds:
        rng = np.random.default_rng(args.seed + f)
        tr_all = np.flatnonzero(fold != f); va = np.flatnonzero(fold == f)
        perm = rng.permutation(tr_all); nv = len(perm) // 10
        inner, tr = perm[:nv], perm[nv:]
        wpos, wneg = pesos_por_t(tr, series)
        torch.manual_seed(args.seed * 100 + f)
        model = Modelo(); opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        passos = args.epocas * math.ceil(len(tr) / args.bs)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=passos)
        melhor, estado, hist_ep = -1, None, []
        for ep in range(args.epocas):
            model.train(); t0 = time.time(); perda = 0.0
            for g, fh, fon, y, m in lotes(tr, series, args.bs, rng):
                logit = model(fh, fon)
                T = y.shape[1]; tt = torch.arange(1, T + 1).clamp(max=1000)
                w = torch.where(y > 0.5, wpos[tt], wneg[tt]) * m
                loss = (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * w).sum() / m.sum()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
                perda += float(loss)
            pi = prever(model, inner, series)
            t_i = np.concatenate([np.arange(1, len(pi[i]) + 1) for i in inner])
            y_i = np.concatenate([series[i][2] for i in inner]); s_i = np.concatenate([pi[i] for i in inner])
            a = ts_auc(t_i, y_i, s_i); hist_ep.append(a)
            print(f"fold {f} época {ep}: perda {perda:.3f} TS-AUC interno {a:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if a > melhor:
                melhor, estado = a, {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(estado)
        pv = prever(model, va, series)
        rows = [pd.DataFrame({"id": ids[i], "t": np.arange(1, len(pv[i]) + 1), "y": series[i][2].astype(int), "pred": pv[i]}) for i in va]
        df = pd.concat(rows); df.to_parquet(f"artifacts/rnn/oof_{args.tag}_f{f}.parquet", index=False)
        rel[f] = {"interno_por_epoca": hist_ep, "tsauc_fold_full_grid": ts_auc(df.t.values, df.y.values, df.pred.values)}
        print(f"fold {f}: TS-AUC no fold (grade cheia) {rel[f]['tsauc_fold_full_grid']:.4f}", flush=True)
    json.dump(rel, open(f"artifacts/rnn/rel_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
