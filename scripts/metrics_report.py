#!/usr/bin/env python
"""Painel de diagnóstico do modelo final (`oof_final_bag3`) — grava gráficos E os CSVs que os
sustentam em `metrics/`. Todos são diagnósticos RELATIVOS na linguagem da própria TS-AUC; nenhum é um
estimador do score oficial de leaderboard (docs/MODELO.md §9.0).

Seis artefatos, cada um PNG + CSV do que está por trás:

  1. auc_by_step        — a anatomia da métrica: AUC_t contínua (em bins de t) com o peso w_t=n_pos·n_neg
                          embaixo. Mostra ONDE a ordenação se perde e se há peso ali (48,7% em 151–400).
  2. auc_by_break_axis  — TS-AUC por eixo de quebra dominante × tercil de magnitude. Mostra QUAL família
                          o modelo ranqueia bem/mal (dependência pura ~0,49, abaixo do acaso).
  3. step_response      — score vs (t−τ) por tercil de magnitude + linha de controle (séries sem quebra):
                          latência de detecção e separação do falso alarme.
  4. xs_base_level      — causa (2): por bucket, o sinal DENTRO da série (gap pós−pré) contra o
                          espalhamento ENTRE séries no mesmo passo. Espalhamento ≫ gap = gargalo de
                          comparabilidade transversal, não de feature.
  5. seed_spread        — TS-AUC por bucket do modelo empacotado contra a nuvem de sementes individuais:
                          torna visível "efeito real vs sorteio de semente" (~0,004).
  6. feature_importance_xs — xs-SHAP (a medida certa sob C1, §9.5) contra o mean|SHAP| convencional,
                          expondo as features que só acompanham o relógio. (Referência: V4.)

Uso:
    python scripts/metrics_report.py
    python scripts/metrics_report.py --oof artifacts/models/oof_final_bag3.parquet --out-dir metrics
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- paleta de referência (dataviz/references/palette.md), ordem documentada ---------------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
RED = "#e34948"
SEQ_LO, SEQ_MID, SEQ_HI = "#86b6ef", "#3987e5", "#184f95"  # rampa azul ordinal (magnitude)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

# buckets de t do projeto (docs/MODELO.md, docs/BACKLOG_TSAUC.md)
T_EDGES = [0, 50, 150, 400, np.inf]
T_LABELS = ["t≤50", "50<t≤150", "150<t≤400", "t>400"]
# eixos de quebra do censo A1 (o de média fica de fora como no detectability_report: canal morto)
AXES = {
    "variância": "delta_logvar_e",
    "dependência": "delta_rho1",
    "cauda(exc)": "delta_exceed",
    "curtose": "delta_kurt",
    "média": "delta_mean_e",
}


# ---- núcleo TS-AUC -------------------------------------------------------------------------------
def _auc_per_step(t: np.ndarray, y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """AUC_t, n_pos, n_neg e w_t por passo t (Mann-Whitney via rank médio intra-passo)."""
    df = pd.DataFrame({"t": t, "y": y, "s": s})
    df["rank"] = df.groupby("t")["s"].rank(method="average")
    g = df.groupby("t")
    n = g["y"].size()
    n_pos = g["y"].sum()
    n_neg = n - n_pos
    r_pos = df.loc[df["y"] == 1].groupby("t")["rank"].sum().reindex(n.index, fill_value=0.0)
    out = pd.DataFrame({"n_pos": n_pos, "n_neg": n_neg})
    valid = (n_pos > 0) & (n_neg > 0)
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg).where(valid)
    out["auc_t"] = auc.where(valid)
    out["w_t"] = (n_pos * n_neg).where(valid, 0.0).astype(np.float64)
    return out.reset_index()


def _weighted_ts_auc(t, y, s) -> float:
    per = _auc_per_step(np.asarray(t), np.asarray(y), np.asarray(s))
    w = per["w_t"].to_numpy()
    a = per["auc_t"].to_numpy()
    m = np.isfinite(a) & (w > 0)
    return float(np.sum(a[m] * w[m]) / np.sum(w[m])) if m.any() else float("nan")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8, length=2)
    for sp in ax.spines.values():
        sp.set_color(AXIS)
    ax.set_axisbelow(True)


def _bucket_of(t: np.ndarray) -> np.ndarray:
    idx = pd.cut(t, bins=T_EDGES, labels=False, right=True)
    return np.asarray(T_LABELS, dtype=object)[idx.astype(np.int64)]


# ================================================================================================
# 1. AUC_t contínua + perfil de peso w_t
# ================================================================================================
def fig_auc_by_step(oof: pd.DataFrame, out: Path) -> pd.DataFrame:
    per = _auc_per_step(oof["t"].to_numpy(), oof["y"].to_numpy(), oof["oof_pred"].to_numpy())
    per = per.sort_values("t")
    # xmax: onde 99% da massa de peso se acumulou (o resto é cauda de w_t≈0)
    cw = per["w_t"].cumsum() / per["w_t"].sum()
    xmax = int(per.loc[cw <= 0.99, "t"].max())

    # bins de largura 10 → TS-AUC ponderada por bin (suavização que fica em unidades da métrica)
    bw = 10
    per["tb"] = (per["t"] // bw) * bw + bw / 2
    grp = per.groupby("tb").apply(
        lambda g: pd.Series(
            {
                "auc": np.sum(g["auc_t"].fillna(0) * g["w_t"]) / g["w_t"].sum() if g["w_t"].sum() > 0 else np.nan,
                "w": g["w_t"].sum(),
            }
        ),
        include_groups=False,
    ).reset_index()
    grp = grp[grp["tb"] <= xmax]

    fig = plt.figure(figsize=(11, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.12, left=0.08, right=0.97, top=0.9, bottom=0.1)
    ax = fig.add_subplot(gs[0])
    axw = fig.add_subplot(gs[1], sharex=ax)

    ax.axhline(0.5, color=MUTED, ls="--", lw=1.0)
    ax.scatter(per.loc[per["t"] <= xmax, "t"], per.loc[per["t"] <= xmax, "auc_t"],
               s=5, color=BLUE, alpha=0.12, edgecolors="none", zorder=1)
    ax.plot(grp["tb"], grp["auc"], color=BLUE, lw=2.2, zorder=3)
    for x0, x1, lbl in zip(T_EDGES[:-1], T_EDGES[1:], T_LABELS):
        if x0 <= xmax:
            ax.axvline(x0, color=AXIS, lw=0.8, ls=":")
    # rótulo direto: TS-AUC por bucket
    ymin = np.nanmin(grp["auc"]) - 0.01
    for x0, x1, lbl in zip(T_EDGES[:-1], T_EDGES[1:], T_LABELS):
        sub = oof[(oof["t"] > x0) & (oof["t"] <= (x1 if np.isfinite(x1) else oof["t"].max()))]
        a = _weighted_ts_auc(sub["t"], sub["y"], sub["oof_pred"])
        xc = (x0 + min(x1, xmax)) / 2
        if x0 <= xmax:
            ax.text(xc, ymin, f"{lbl}\nAUC {a:.3f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK2)
    ax.set_ylabel("AUC no passo t  (bins de 10)", fontsize=9, color=INK2)
    ax.set_title("Anatomia da TS-AUC — onde a ordenação transversal se perde",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.tick_params(labelbottom=False)
    _style(ax)

    axw.fill_between(grp["tb"], 0, grp["w"], color=SEQ_LO, alpha=0.9, lw=0)
    axw.plot(grp["tb"], grp["w"], color=SEQ_MID, lw=1.2)
    for x0 in T_EDGES[:-1]:
        if x0 <= xmax:
            axw.axvline(x0, color=AXIS, lw=0.8, ls=":")
    axw.set_ylabel("peso w_t\n= n_pos·n_neg", fontsize=8.5, color=INK2)
    axw.set_xlabel("passo online t", fontsize=9, color=INK2)
    axw.set_xlim(0, xmax)
    _style(axw)

    fig.savefig(out / "auc_by_step.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    grp.rename(columns={"tb": "t_bin_center", "auc": "ts_auc_bin", "w": "weight_bin"}).to_csv(
        out / "auc_by_step.csv", index=False)
    return grp


# ================================================================================================
# 2. TS-AUC por eixo de quebra dominante × tercil de magnitude
# ================================================================================================
def _dominant_axis(census: pd.DataFrame) -> pd.DataFrame:
    c = census.copy()
    z = {}
    for name, col in AXES.items():
        v = c[col].to_numpy(dtype=np.float64)
        sd = np.nanstd(v)
        z[name] = np.abs(v) / sd if sd > 0 else np.zeros_like(v)
    Z = pd.DataFrame(z, index=c.index)
    c["axis"] = Z.idxmax(axis=1)
    c["axis_mag"] = Z.max(axis=1)  # magnitude padronizada no eixo dominante
    return c[["id", "axis", "axis_mag"]]


def fig_auc_by_break_axis(oof: pd.DataFrame, census: pd.DataFrame, out: Path) -> pd.DataFrame:
    dom = _dominant_axis(census)
    # tercil de magnitude DENTRO de cada eixo
    dom["mag_tercile"] = dom.groupby("axis")["axis_mag"].transform(
        lambda s: pd.qcut(s, 3, labels=["baixa", "média", "alta"], duplicates="drop") if s.nunique() >= 3 else "média")

    has_break = oof.groupby("id")["y"].transform("max") > 0
    neg = oof.loc[~has_break, ["t", "y", "oof_pred"]]           # séries sem quebra = negativos "limpos"
    pos = oof.loc[has_break & (oof["y"] == 1)].merge(dom, on="id", how="left")
    n_broken = int((oof.groupby("id")["y"].max() > 0).sum())
    coverage = dom["id"].isin(oof.loc[has_break, "id"]).sum() / n_broken

    order = ["variância", "dependência", "cauda(exc)", "curtose", "média"]
    terciles = ["baixa", "média", "alta"]
    rows = []
    for ax_name in order:
        for terc in terciles:
            ids = dom.loc[(dom["axis"] == ax_name) & (dom["mag_tercile"] == terc), "id"]
            p = pos[pos["id"].isin(ids)]
            if p.empty:
                rows.append({"axis": ax_name, "tercile": terc, "ts_auc": np.nan, "n_series": 0})
                continue
            frame = pd.concat([p[["t", "y", "oof_pred"]], neg], ignore_index=True)
            a = _weighted_ts_auc(frame["t"], frame["y"], frame["oof_pred"])
            rows.append({"axis": ax_name, "tercile": terc, "ts_auc": a, "n_series": p["id"].nunique()})
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    x = np.arange(len(order))
    w = 0.26
    colors = {"baixa": SEQ_LO, "média": SEQ_MID, "alta": SEQ_HI}
    for i, terc in enumerate(terciles):
        vals = [tab[(tab["axis"] == a) & (tab["tercile"] == terc)]["ts_auc"].values[0] for a in order]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=colors[terc], label=f"magnitude {terc}",
                      edgecolor=SURFACE, linewidth=1.2)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7.2, color=INK2)
    ax.axhline(0.5, color=RED, ls="--", lw=1.1)
    ax.text(len(order) - 0.5, 0.505, "acaso 0,50", color=RED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9.5, color=INK)
    ax.set_ylabel("TS-AUC (positivos do eixo vs. séries sem quebra)", fontsize=9, color=INK2)
    ax.set_ylim(0.45, min(1.0, np.nanmax(tab["ts_auc"]) + 0.06))
    ax.set_title("Por onde o modelo falha — TS-AUC por família de quebra e magnitude",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")
    _style(ax)
    fig.text(0.99, 0.01,
             f"eixo dominante pelo censo A1 (|Δ| padronizado); negativos = séries sem quebra. "
             f"Cobertura: {coverage:.0%} das {n_broken} séries com quebra têm linha no censo.",
             fontsize=7.5, color=MUTED, ha="right", va="bottom")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "auc_by_break_axis.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    tab.to_csv(out / "auc_by_break_axis.csv", index=False)
    return tab


# ================================================================================================
# 3. Resposta ao degrau por magnitude + controle sem-quebra
# ================================================================================================
def fig_step_response(oof: pd.DataFrame, census: pd.DataFrame, out: Path) -> pd.DataFrame:
    dom = _dominant_axis(census)
    has_break = oof.groupby("id")["y"].transform("max") > 0
    broken = oof.loc[has_break].copy()
    tau = broken.loc[broken["y"] == 1].groupby("id")["t"].min()
    broken["tau"] = broken["id"].map(tau)
    broken["offset"] = broken["t"] - broken["tau"]

    # divergência global (norma L2 dos eixos padronizados) para os terciles de magnitude
    zsum = np.zeros(len(census))
    for col in ("delta_logvar_e", "delta_rho1", "delta_kurt", "delta_exceed"):
        v = census[col].to_numpy(float)
        sd = np.nanstd(v)
        if sd > 0:
            zsum = zsum + np.nan_to_num((v / sd) ** 2)
    diverg = pd.Series(np.sqrt(zsum), index=census["id"]).rename("div")
    broken = broken.merge(diverg, left_on="id", right_index=True, how="left")
    broken["mag"] = pd.qcut(broken["div"], 3, labels=["baixa", "média", "alta"], duplicates="drop")

    lo, hi = -30, 200
    win = broken[(broken["offset"] >= lo) & (broken["offset"] <= hi)]
    curves, curves_raw = {}, {}
    for mag in ["baixa", "média", "alta"]:
        # xs_dev: excesso sobre a média do MESMO passo — remove a subida de taxa-base em t, que
        # infla igualmente quebra e controle. Sem isso, a "subida" pós-quebra mistura detecção com
        # relógio (BACKLOG §1). É também exatamente o que a TS-AUC pontua (ordenação no passo).
        c = win[win["mag"] == mag].groupby("offset")["xs_dev"].mean()
        curves_raw[mag] = c
        curves[mag] = c.rolling(5, center=True, min_periods=1).mean()  # composição muda por offset
    # controle: séries sem quebra — por construção ~0 no desvio transversal (referência do "acaso")
    control_level = float(oof.loc[~has_break, "xs_dev"].mean())

    fig, ax = plt.subplots(figsize=(10.5, 5.6), facecolor=SURFACE)
    cmap = {"baixa": SEQ_LO, "média": SEQ_MID, "alta": SEQ_HI}
    for mag in ["baixa", "média", "alta"]:
        c = curves[mag]
        ax.plot(c.index, c.values, color=cmap[mag], lw=2.0, label=f"magnitude {mag}")
        if len(c):
            ax.text(c.index[-1] + 2, c.values[-1], mag, color=cmap[mag], fontsize=8.5, va="center")
    ax.axhline(control_level, color=RED, ls="--", lw=1.2)
    ax.text(hi, control_level, "  controle (sem quebra)", color=RED, fontsize=8.5, va="bottom", ha="right")
    ax.axvline(0, color=INK2, lw=1.2, ls=":")
    ax.text(0, ax.get_ylim()[1], " quebra (τ)", color=INK2, fontsize=8.5, va="top")
    ax.set_xlabel("passos desde a quebra  (t − τ)", fontsize=9, color=INK2)
    ax.set_ylabel("excesso de score sobre o passo  (desvio transversal)", fontsize=9, color=INK2)
    ax.set_title("Resposta ao degrau — detecção por magnitude, líquida da taxa-base",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_xlim(lo, hi + 10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "step_response.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    tab = pd.DataFrame({mag: curves_raw[mag] for mag in curves_raw}).reset_index()
    tab["control_level"] = control_level
    tab.to_csv(out / "step_response.csv", index=False)
    return tab


# ================================================================================================
# 4. Nível-de-base transversal (causa 2): sinal dentro-da-série vs espalhamento entre-séries
# ================================================================================================
def fig_xs_base_level(oof: pd.DataFrame, out: Path) -> pd.DataFrame:
    o = oof.copy()
    o["bucket"] = _bucket_of(o["t"].to_numpy())
    rows = []
    for lbl in T_LABELS:
        sub = o[o["bucket"] == lbl]
        # espalhamento ENTRE séries no mesmo passo, ponderado por w_t
        per_t = sub.groupby("t").agg(sd=("oof_pred", "std"), npos=("y", "sum"), n=("y", "size"))
        per_t["w"] = per_t["npos"] * (per_t["n"] - per_t["npos"])
        spread = float((per_t["sd"] * per_t["w"]).sum() / per_t["w"].sum()) if per_t["w"].sum() > 0 else np.nan
        # sinal DENTRO da série, LÍQUIDO DO PASSO: gap pós−pré medido sobre o desvio transversal
        # (xs_dev), não sobre o score cru. O gap cru é dominado pela subida da taxa-base em t
        # (BACKLOG §1): em t≤50 o gap cru dá ~0,047 mas o controlado é ~0. Só o controlado responde
        # "o sinal da quebra existe acima do que a métrica já enxerga no passo?".
        gm = sub.groupby(["id", "y"])["xs_dev"].mean().unstack()
        if {0, 1}.issubset(gm.columns):
            gap = (gm[1] - gm[0]).dropna()
            signal = float(gap.median())
            n_series = int(gap.shape[0])
        else:
            signal, n_series = np.nan, 0
        rows.append({"bucket": lbl, "sinal_intra_serie": signal, "espalhamento_entre_series": spread,
                     "razao": signal / spread if spread else np.nan, "n_series": n_series})
    tab = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    x = np.arange(len(T_LABELS))
    w = 0.38
    b1 = ax.bar(x - w / 2, tab["sinal_intra_serie"], w, color=BLUE, edgecolor=SURFACE, linewidth=1.2,
                label="sinal da quebra (gap pós−pré, líquido do passo)")
    b2 = ax.bar(x + w / 2, tab["espalhamento_entre_series"], w, color=ORANGE, edgecolor=SURFACE, linewidth=1.2,
                label="espalhamento de base entre séries (mesmo passo)")
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                ax.text(b.get_x() + b.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(T_LABELS, fontsize=9.5, color=INK)
    ax.set_ylabel("desvio de score", fontsize=9, color=INK2)
    ax.set_ylim(0, np.nanmax(tab["espalhamento_entre_series"]) * 1.28)
    ax.set_title("Gargalo de comparabilidade transversal (causa 2)",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_axisbelow(True)
    _style(ax)
    fig.text(0.5, 0.015,
             "Sinal líquido do passo (controlado por t): em t≤50 é ~0 (piso de informação); depois é positivo mas "
             "sempre menor que o espalhamento de base\nentre séries — o sinal existe, mas não vira ordenação "
             "transversal (causa 2). Comparar contra o gap CRU seria medir a taxa-base (BACKLOG §1).",
             fontsize=8, color=INK2, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(out / "xs_base_level.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    tab.to_csv(out / "xs_base_level.csv", index=False)
    return tab


# ================================================================================================
# 5. Dispersão de semente
# ================================================================================================
def fig_seed_spread(bag_oof: pd.DataFrame, seed_paths: list[Path], out: Path) -> pd.DataFrame:
    def by_bucket(df):
        b = _bucket_of(df["t"].to_numpy())
        res = {"geral": _weighted_ts_auc(df["t"], df["y"], df["oof_pred"])}
        for lbl in T_LABELS:
            m = b == lbl
            res[lbl] = _weighted_ts_auc(df["t"][m], df["y"][m], df["oof_pred"][m])
        return res

    cats = ["geral"] + T_LABELS
    bag = by_bucket(bag_oof)
    seeds = []
    for p in seed_paths:
        try:
            seeds.append(by_bucket(pd.read_parquet(p)))
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    x = np.arange(len(cats))
    seed_mat = {c: [s[c] for s in seeds] for c in cats}
    for i, c in enumerate(cats):
        vals = seed_mat[c]
        if vals:
            ax.plot([x[i]] * len(vals), vals, "o", color=MUTED, ms=6, alpha=0.7,
                    label="sementes individuais" if i == 0 else None)
            ax.vlines(x[i], min(vals), max(vals), color=AXIS, lw=1.2)
    ax.plot(x, [bag[c] for c in cats], "D", color=BLUE, ms=9, zorder=5, label="modelo empacotado (bag)")
    for i, c in enumerate(cats):
        ax.text(x[i], bag[c] + 0.006, f"{bag[c]:.3f}", ha="center", va="bottom", fontsize=8, color=BLUE)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=9.5, color=INK)
    ax.axhline(0.5, color=MUTED, ls="--", lw=0.9)
    ax.set_ylabel("TS-AUC OOF", fontsize=9, color=INK2)
    ax.set_title("Efeito real vs. sorteio de semente — o bag contra a nuvem de sementes",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "seed_spread.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    tab = pd.DataFrame([{"grupo": "bag", **bag}] + [{"grupo": f"seed_{i}", **s} for i, s in enumerate(seeds)])
    tab.to_csv(out / "seed_spread.csv", index=False)
    return tab


# ================================================================================================
# 6. xs-SHAP vs mean|SHAP| convencional
# ================================================================================================
def fig_feature_importance_xs(shap_csv: Path, out: Path, top: int = 18) -> pd.DataFrame:
    if not shap_csv.exists():
        print(f"  (pulado — {shap_csv} não existe)")
        return pd.DataFrame()
    d = pd.read_csv(shap_csv)
    d["xs_share"] = d["xs_shap"] / d["xs_shap"].sum()
    d["conv_share"] = d["mean_abs_shap"] / d["mean_abs_shap"].sum()
    d = d.sort_values("xs_share", ascending=False).head(top).iloc[::-1]

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=SURFACE)
    y = np.arange(len(d))
    h = 0.4
    ax.barh(y + h / 2, d["xs_share"], h, color=BLUE, edgecolor=SURFACE, linewidth=1.0,
            label="xs-SHAP (dispersão dentro do passo — a medida certa, §9.5)")
    ax.barh(y - h / 2, d["conv_share"], h, color=ORANGE, edgecolor=SURFACE, linewidth=1.0,
            label="mean|SHAP| convencional")
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"], fontsize=8, color=INK)
    ax.set_xlabel("fração da importância total", fontsize=9, color=INK2)
    ax.set_title("Importância na métrica certa — xs-SHAP vs. convencional  (ref.: V4)",
                 fontsize=12.5, color=INK, weight="bold", loc="left", pad=8)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "feature_importance_xs.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    d[["feature", "family", "xs_share", "conv_share"]].iloc[::-1].to_csv(
        out / "feature_importance_xs.csv", index=False)
    return d


# ================================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oof", default="artifacts/models/oof_final_bag3.parquet")
    ap.add_argument("--census", default="artifacts/reports/break_type_census.csv")
    ap.add_argument("--shap", default="artifacts/reports/shap_v4.csv")
    ap.add_argument("--seed-glob", default="artifacts/models/oof_final_s*.parquet")
    ap.add_argument("--out-dir", default="metrics")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"carregando OOF {args.oof} ...")
    oof = pd.read_parquet(args.oof)
    # desvio transversal: score menos a média do MESMO passo. É a quantidade que a TS-AUC de fato
    # pontua (C1) e a única que remove a tendência de base-rate em t — sem ela, qualquer contraste
    # dentro-da-série mede o relógio, não a quebra (docs/BACKLOG_TSAUC.md §1, "ARMADILHA").
    oof["xs_dev"] = oof["oof_pred"] - oof.groupby("t")["oof_pred"].transform("mean")
    census = pd.read_csv(args.census)
    seed_paths = sorted(Path().glob(args.seed_glob))

    print("1/6 auc_by_step ...")
    fig_auc_by_step(oof, out)
    print("2/6 auc_by_break_axis ...")
    fig_auc_by_break_axis(oof, census, out)
    print("3/6 step_response ...")
    fig_step_response(oof, census, out)
    print("4/6 xs_base_level ...")
    fig_xs_base_level(oof, out)
    print("5/6 seed_spread ...")
    fig_seed_spread(oof, seed_paths, out)
    print("6/6 feature_importance_xs ...")
    fig_feature_importance_xs(Path(args.shap), out)

    print(f"\npronto — {len(list(out.glob('*.png')))} PNG + {len(list(out.glob('*.csv')))} CSV em {out}/")


if __name__ == "__main__":
    main()
