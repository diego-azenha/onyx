#!/usr/bin/env python
"""Visualiza a deteccao de quebras da ultima versao do modelo (OOF `oof_final_bag3`).

Semantica (src/sbrt/model/dataset.py:67): para cada serie a quebra estrutural ocorre no passo
`tau` do fluxo online (period==2 em X_train); o rotulo por passo e `y = 1{t >= tau}` e o modelo
emite `oof_pred(t)` = probabilidade online de que a quebra ja aconteceu. O "ponto de quebra" e
`tau`; a "faixa de tempo em que ocorre a quebra" e o bucket de `tau`, os mesmos da tabela do
projeto: 1-50 / 51-150 / 151-400 / 401+ (docs/BACKLOG_TSAUC.md).

Para cada faixa gera um 3x3 de series-exemplo com quebra real. Cada celula empilha, no MESMO eixo x
(passo online t): a serie (valor, em cima) e o score (0-1, embaixo), com uma linha vertical
tracejada no ponto de quebra. Empilhar em vez de eixo-duplo respeita a regra de nao sobrepor duas
escalas no mesmo par de eixos (skill dataviz)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

OOF_PATH = "artifacts/models/oof_final_bag3.parquet"
X_PATH = "data/X_train.parquet"
OUT_DIR = Path("metrics")

BINS = [0, 50, 150, 400, np.inf]
LABELS = ["1-50", "51-150", "151-400", "401+"]
SEED = 42
N_CELLS = 9  # 3x3

# paleta validada (dataviz/references/palette.md) — series e score vivem em paineis separados
C_SERIES = "#2a78d6"   # azul (fluxo online)
C_HIST = "#9aa4b2"     # cinza-azulado (segmento historico, pre-online)
C_SCORE = "#eb6834"    # laranja
C_BREAK = "#e34948"    # vermelho (linha de quebra)
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def load_series_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Por serie: tau (1o passo com y==1), tmax, comprimento pos-quebra e faixa de tau."""
    tau = oof.loc[oof["y"] == 1].groupby("id")["t"].min().rename("tau")
    tmax = oof.groupby("id")["t"].max().rename("tmax")
    df = pd.concat([tmax, tau], axis=1)
    df["has_break"] = df["tau"].notna()
    df["post_len"] = df["tmax"] - df["tau"]
    df["faixa"] = pd.cut(df["tau"], bins=BINS, labels=LABELS)
    return df


def pick_examples(tbl: pd.DataFrame) -> dict[str, list[int]]:
    """9 series por faixa: quebra real, algum contexto pre-quebra e pos-quebra observavel."""
    rng = np.random.default_rng(SEED)
    picks: dict[str, list[int]] = {}
    elig = tbl[tbl["has_break"] & (tbl["post_len"] >= 8) & (tbl["tau"] >= 3)]
    for lbl in LABELS:
        ids = elig.index[elig["faixa"] == lbl].to_numpy()
        ids = np.sort(ids)
        chosen = rng.choice(ids, size=min(N_CELLS, len(ids)), replace=False)
        picks[lbl] = sorted(int(i) for i in chosen)
    return picks


def split_values(X: pd.DataFrame, sid: int) -> tuple[np.ndarray, np.ndarray]:
    """Segmento historico (period==1) e fluxo online (period==2), em ordem de tempo."""
    g = X.loc[sid]
    v = g["value"].to_numpy()
    p = g["period"].to_numpy()
    return v[p == 1], v[p == 2]


def hist_tail_len(n_online: int, n_hist: int) -> int:
    """Quanto do fim do historico mostrar: da ordem do proprio online (contexto pre-online amplo),
    com piso e teto, limitado ao que o historico tem."""
    return int(min(n_hist, np.clip(n_online, 250, 750)))


def draw_cell(fig, sub, sid, hist_tail, online_vals, sc_t, sc_v, tau, final_score):
    """Eixo-x contínuo e único: a cauda do historico (cinza, x<=0) emenda direto na serie online
    (azul, x>=1); x=0 e o inicio do online. A quebra (tau) e o tracejado vermelho, dentro do online.
    O score (laranja) fica embaixo partilhando o mesmo x, entao tau alinha nos dois paineis."""
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=sub, height_ratios=[2.0, 1.0], hspace=0.08)
    ax_v = fig.add_subplot(inner[0])
    ax_s = fig.add_subplot(inner[1], sharex=ax_v)

    H = len(hist_tail)
    xh = np.arange(-H + 1, 1)               # termina em 0 (ultimo ponto historico)
    xo = np.arange(1, len(online_vals) + 1)  # online comeca em 1
    x_lo, x_hi = -H, len(online_vals) + 1

    ax_v.plot(xh, hist_tail, color=C_HIST, lw=0.7)
    ax_v.plot(xo, online_vals, color=C_SERIES, lw=0.8)
    ax_v.axvline(0, color=C_HIST, ls=":", lw=1.0)      # inicio do online
    ax_v.axvline(tau, color=C_BREAK, ls="--", lw=1.1)  # quebra
    ax_v.set_xlim(x_lo, x_hi)
    ax_v.set_title(f"id {sid}  ·  tau={int(tau)}  ·  score_final={final_score:.2f}",
                   fontsize=7.5, color=INK, pad=3)
    ax_v.tick_params(labelbottom=False, labelsize=6, colors=INK2, length=2)
    ax_v.set_ylabel("valor", fontsize=6.5, color=INK2)

    ax_s.plot(sc_t, sc_v, color=C_SCORE, lw=1.1)
    ax_s.axvline(0, color=C_HIST, ls=":", lw=1.0)
    ax_s.axvline(tau, color=C_BREAK, ls="--", lw=1.1)
    ax_s.set_xlim(x_lo, x_hi)
    ax_s.set_ylim(-0.03, 1.03)
    ax_s.set_yticks([0, 0.5, 1])
    ax_s.tick_params(labelsize=6, colors=INK2, length=2)
    ax_s.set_ylabel("score", fontsize=6.5, color=INK2)
    ax_s.set_xlabel("passo online t  (0 = início do online)", fontsize=6.5, color=INK2)

    for ax in (ax_v, ax_s):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.5)
        for sp in ax.spines.values():
            sp.set_color(GRID)


def build_figure(lbl, ids, X, oof_by_id, tbl):
    fig = plt.figure(figsize=(14.5, 11), facecolor=SURFACE)
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.24,
                  left=0.05, right=0.99, top=0.90, bottom=0.055)
    for k, sid in enumerate(ids):
        sub = gs[k // 3, k % 3]
        hist_vals, online_vals = split_values(X, sid)
        H = hist_tail_len(len(online_vals), len(hist_vals))
        o = oof_by_id.get_group(sid).sort_values("t")
        tau = float(tbl.loc[sid, "tau"])
        final_score = float(o["oof_pred"].iloc[-1])
        draw_cell(fig, sub, sid, hist_vals[-H:], online_vals,
                  o["t"].to_numpy(), o["oof_pred"].to_numpy(), tau, final_score)

    fig.suptitle(f"Deteccao de quebra  ·  faixa tau ∈ [{lbl}]  ·  modelo final_bag3",
                 fontsize=14, color=INK, x=0.05, ha="left", y=0.965, weight="bold")
    fig.text(0.05, 0.925,
             "cinza: cauda do histórico (pré-online)  →  azul: série online    "
             "pontilhado cinza: início do online    laranja: score P(quebra já ocorreu)    "
             "tracejado vermelho: quebra (tau)",
             fontsize=8.5, color=INK2, ha="left")
    out = OUT_DIR / f"break_detection_faixa_{lbl}.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("carregando OOF...")
    oof = pd.read_parquet(OOF_PATH)
    tbl = load_series_table(oof)
    picks = pick_examples(tbl)
    all_ids = sorted({i for ids in picks.values() for i in ids})
    print("series escolhidas por faixa:", {k: v for k, v in picks.items()})

    print(f"carregando {len(all_ids)} series de X_train (filtrado)...")
    try:
        X = pd.read_parquet(X_PATH, filters=[("id", "in", all_ids)])
        if X.index.names[0] != "id":
            X = X.set_index(["id", "time"])
    except Exception:
        X = pd.read_parquet(X_PATH).loc[all_ids]

    oof_by_id = oof[oof["id"].isin(all_ids)].groupby("id")
    outs = [build_figure(lbl, picks[lbl], X, oof_by_id, tbl) for lbl in LABELS]
    print("\nfiguras geradas:")
    for o in outs:
        print(" ", o)


if __name__ == "__main__":
    main()
