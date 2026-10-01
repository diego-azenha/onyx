#!/usr/bin/env python
"""I3 (knowledge/frentes/estrategias-01-10/README.md): NULO CONDICIONAL APRENDIDO nos históricos.

Cada série do Onyx modela o próprio nulo sozinha (AR + EWMA ajustados no histórico dela). Aqui um GBM
nosso aprende, com ~3M posições dos HISTÓRICOS de todas as séries (sem rótulo nenhum), a volatilidade
condicional do próximo ponto: alvo log(x² + ε), dado o passado imediato e o perfil fixo da série. Os
históricos não sofrem com a escassez de eventos de quebra (C1/C2).

No online: z_p = x_p / σ̂_p. Surpresas causais por passo t, todas padronizadas pela cauda do histórico
da própria série (os mesmos cálculos nos últimos 500 pontos do histórico):
variância (média de log z²), dependência (média de z_p·z_{p-1}), cauda (fração de |z| acima do q95 da
cauda), em janelas de 25, 100 e no prefixo, mais CUSUMs de log z² para cima e para baixo.

Cross-fit: a série do fold f é pontuada pelo modelo treinado nos históricos dos outros folds (como seria
uma série de teste nunca vista). Saída: data/processed/train_rows_i3.parquet (train_rows + colunas `nz_*`).
"""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import lfilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.evaluation.splits import grouped_stratified_kfold  # noqa: E402

L = 6                       # defasagens
LAMS = (0.02, 0.06, 0.2)
CAUDA = 500                 # pontos do fim do histórico usados como nulo da série
N_TREINO = 300              # posições amostradas por histórico para treinar o modelo do nulo
EPS = 1e-4
PARAMS = dict(objective="regression", learning_rate=0.08, num_leaves=63, min_data_in_leaf=500,
              feature_fraction=0.9, bagging_fraction=0.7, bagging_freq=1, lambda_l2=5.0, verbose=-1,
              seed=0, num_threads=6, deterministic=True, force_row_wise=True)


def _ewma_pred(x2: np.ndarray, lam: float) -> np.ndarray:
    """EWMA de x² até p−1 (preditor de p): e_p = (1−λ) e_{p−1} + λ x²_{p−1}."""
    e0 = float(np.mean(x2[:50]))
    e, _ = lfilter([lam], [1, -(1 - lam)], x2, zi=[(1 - lam) * e0])
    return np.r_[e0, e[:-1]]


def _estaticas(h: np.ndarray) -> np.ndarray:
    a = np.abs(h); c = h - h.mean()
    acf1 = float(np.dot(c[1:], c[:-1]) / (np.dot(c, c) + 1e-12))
    ca = a - a.mean()
    acf1a = float(np.dot(ca[1:], ca[:-1]) / (np.dot(ca, ca) + 1e-12))
    kurt = float(np.mean(c ** 4) / (np.mean(c ** 2) ** 2 + 1e-12))
    rv = pd.Series(h ** 2).rolling(50).mean().dropna().to_numpy()
    vov = float(np.std(np.log(rv + 1e-8)))
    return np.array([acf1, acf1a, np.log(kurt), vov], dtype=np.float32)


def _matriz(s: np.ndarray, est: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Features do preditor da posição p (só usa s[:p])."""
    x2 = s ** 2
    ew = [_ewma_pred(x2, lam) for lam in LAMS]
    cols = [s[pos - k] for k in range(1, L + 1)] + [np.abs(s[pos - k]) for k in range(1, L + 1)]
    cols += [np.log(e[pos] + 1e-8) for e in ew]
    cols += [np.full(len(pos), v, dtype=np.float64) for v in est]
    return np.column_stack(cols).astype(np.float32)


def _treino_serie(sid, h, on, seed):
    rng = np.random.default_rng(seed + sid)
    pos = np.sort(rng.choice(np.arange(60, len(h)), size=min(N_TREINO, len(h) - 60), replace=False))
    X = _matriz(h, _estaticas(h), pos)
    return X, np.log(h[pos] ** 2 + EPS).astype(np.float32)


def _feats_serie(sid, h, on, modelo_txt):
    m = lgb.Booster(model_str=modelo_txt)
    s = np.concatenate([h, on]); nh = len(h)
    pos_c = np.arange(max(60, nh - CAUDA), nh); pos_o = np.arange(nh, nh + len(on))
    X = _matriz(s, _estaticas(h), np.r_[pos_c, pos_o])
    g = m.predict(X, num_threads=1)
    u = np.log(s[np.r_[pos_c, pos_o]] ** 2 + EPS) - g                   # log z² (com viés constante)
    zz = s[np.r_[pos_c, pos_o]] * np.exp(-0.5 * g)
    uc, uo = u[: len(pos_c)], u[len(pos_c):]
    zc, zo = zz[: len(pos_c)], zz[len(pos_c):]
    mu, sd = uc.mean(), uc.std() + 1e-9
    dc = zc[1:] * zc[:-1]; dmu, dsd = dc.mean(), dc.std() + 1e-9
    q95 = np.quantile(np.abs(zc), 0.95)
    us = (uo - mu) / sd
    dprev = np.r_[zc[-1], zo[:-1]]
    ds = (zo * dprev - dmu) / dsd
    cau = (np.abs(zo) > q95).astype(np.float64) - 0.05
    n = len(on); tt = np.arange(1, n + 1)
    out = {"id": np.full(n, sid, dtype=np.int64), "t": tt.astype(np.int64)}
    for nome, v in (("var", us), ("dep", ds), ("cauda", cau)):
        c = np.cumsum(v)
        out[f"nz_{nome}_pref"] = c / np.sqrt(tt)                         # soma padronizada (z) no prefixo
        for w in (25, 100):
            cw = c - np.r_[np.zeros(w), c[:-w]] if n > w else c
            out[f"nz_{nome}_w{w}"] = cw / np.sqrt(np.minimum(tt, w))
    for nome, sinal in (("up", 1.0), ("dn", -1.0)):
        S = np.empty(n); acc = 0.0
        for i in range(n):
            acc = max(0.0, acc + sinal * us[i] - 0.5); S[i] = acc
        out[f"nz_cusum_{nome}"] = S
    out["nz_u_atual"] = us
    return pd.DataFrame(out)


def main() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    series = _carregar_series(Path("data"))
    rows = pd.read_parquet("data/processed/train_rows.parquet")
    meta = rows[["id", "t", "y"]]
    fold_de = {}
    for f, (_, va) in enumerate(grouped_stratified_kfold(meta, cfg.lightgbm.n_folds, cfg.seed)):
        for i in np.unique(meta["id"].to_numpy()[va]):
            fold_de[int(i)] = f
    print("montando treino do nulo", flush=True)
    res = Parallel(n_jobs=4, batch_size=64)(delayed(_treino_serie)(sid, h, on, 7) for sid, h, on in series)
    Xs = [r[0] for r in res]; ys = [r[1] for r in res]
    fs = np.concatenate([np.full(len(r[1]), fold_de[s[0]]) for r, s in zip(res, series)])
    X = np.concatenate(Xs); y = np.concatenate(ys); del res, Xs, ys
    print("treino do nulo:", X.shape, flush=True)
    partes = []
    for f in range(cfg.lightgbm.n_folds):
        tr = fs != f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), 400)
        txt = m.model_to_string()
        alvo = [s for s in series if fold_de[s[0]] == f]
        print(f"fold {f}: modelo pronto, pontuando {len(alvo)} séries", flush=True)
        partes += Parallel(n_jobs=4, batch_size=16)(delayed(_feats_serie)(sid, h, on, txt) for sid, h, on in alvo)
    F = pd.concat(partes, ignore_index=True)
    F["t"] = F["t"].astype(rows["t"].dtype); F["id"] = F["id"].astype(rows["id"].dtype)
    out = rows.merge(F, on=["id", "t"], how="left")
    assert len(out) == len(rows) and out["nz_var_pref"].notna().mean() > 0.99
    out.to_parquet("data/processed/train_rows_i3.parquet", index=False)
    print("gravado data/processed/train_rows_i3.parquet", out.shape, flush=True)


if __name__ == "__main__":
    main()
