#!/usr/bin/env python
"""F0.b (docs/BACKLOG_TSAUC.md): reparte o erro em `t<=50` entre as TRÊS causas do filtro da Seção 0.

Quando um detector erra, há exatamente três causas, e só uma é resolvível com features:

1. **Lacuna de informação** — o sinal existe na série mas nenhuma feature o projeta.
2. **Falha de calibração/objetivo** — o sinal ESTÁ nas features, mas o objetivo não o converte em
   ranking transversal.
3. **Indetectabilidade intrínseca** — a quebra é pequena demais para o número de observações
   disponíveis; nenhum algoritmo separa melhor que o acaso.

Perseguir (3) com features novas é *adicionar variância sem sinal*, e sob TS-AUC transversal isso
machuca a calibração relativa das demais séries. Este script mede quanto peso de erro cai em (3),
para decidir se vale gastar esforço nas frentes de largura (F6/F7) ou parar em P1.

**Como a detectabilidade é estimada.** A teoria diz que a separabilidade é governada pela distância
estatística entre os regimes pré e pós-quebra RELATIVA ao número de observações pós-quebra: o poder
da estatística ótima para um deslocamento delta com m pontos é ~Phi(delta*sqrt(m) - z_alpha)
(docs/MODELO.md §12.6). Usamos exatamente essa forma, com `delta` sendo a divergência multi-eixo do
censo A1 (`break_type_census.csv`, que já traz delta_logvar_e / delta_rho1 / delta_kurt / delta_exceed
e n_post por série) e `m = n_post`.

Não é um número absoluto calibrado — é um ORDENADOR de séries. O que decide o veredito é o contraste
entre os decis de detectabilidade, não o valor em si.

Saída: CSV por série + um resumo impresso. Diagnóstico, nunca um substituto do score oficial
(docs/MODELO.md §9.0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbrt.evaluation.ts_auc import weighted_ts_auc

# Eixos do censo A1 usados como divergência por série. O canal de média está de fora de propósito:
# está medido como morto (beta multivariado -0,005; só 6,8% das séries com |delta_mean|>0,3), então
# incluí-lo só adicionaria ruído ao ordenador.
AXES = ("delta_logvar_e", "delta_rho1", "delta_kurt", "delta_exceed")


def estimate_detectability(census: pd.DataFrame, t_max: int) -> pd.DataFrame:
    """delta_efetivo * sqrt(m), com delta_efetivo = norma L2 dos eixos padronizados.

    Padronizar cada eixo pelo seu próprio desvio entre séries é o que torna os eixos comparáveis
    entre si — sem isso, o eixo de maior escala bruta domina a norma sem carregar mais sinal.

    **`m` é o número de observações pós-quebra DENTRO do bucket**, não o da série inteira. A
    pergunta é "o erro em t<=t_max é piso informacional?", e nesse bucket o detector só viu
    `t_max - tau` pontos pós-quebra, por mais longo que o segmento pós-quebra venha a ser depois.
    Usar `n_post` da série inteira infla a detectabilidade justamente das séries de quebra precoce —
    que são todas as deste recorte — e embaralha os quintis."""
    out = census.copy()
    z = np.zeros(len(out), dtype=np.float64)
    for axis in AXES:
        v = out[axis].to_numpy(dtype=np.float64)
        sd = np.nanstd(v)
        if sd > 0:
            z = z + np.nan_to_num((v / sd) ** 2)
    out["divergence"] = np.sqrt(z)
    tau = out["tau_index"].to_numpy(dtype=np.float64)
    m_bucket = np.clip(t_max - tau, 0.0, None)
    m_bucket = np.minimum(m_bucket, out["n_post"].to_numpy(dtype=np.float64))
    out["m_bucket"] = m_bucket
    out["detectability"] = out["divergence"] * np.sqrt(m_bucket)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--census", default="artifacts/reports/break_type_census.csv")
    parser.add_argument("--oof", default="artifacts/models/oof_v4.parquet")
    parser.add_argument("--score-col", default="oof_pred")
    parser.add_argument("--t-max", type=int, default=50, help="bucket sob investigação")
    parser.add_argument("--n-deciles", type=int, default=5)
    parser.add_argument("--out", default="artifacts/reports/detectability.csv")
    args = parser.parse_args()

    census = estimate_detectability(pd.read_csv(args.census), args.t_max)
    oof = pd.read_parquet(args.oof).rename(columns={args.score_col: "score"})

    early = oof[oof["t"] <= args.t_max]
    if early.empty:
        raise SystemExit(f"nenhuma linha com t<={args.t_max} em {args.oof}")

    # TS-AUC por série no bucket precoce só faz sentido onde a série tem os dois rótulos; a maioria
    # não tem. O sinal por série usável é o GAP: score médio pós-quebra menos score médio pré-quebra,
    # dentro da própria série — o mesmo instrumento de gate local que o protocolo já usa, e imune à
    # escala arbitrária de cada série.
    g = early.groupby(["id", "y"])["score"].mean().unstack()
    gap = (g.get(1) - g.get(0)).dropna().rename("gap")

    df = census.set_index("id").join(gap, how="inner")
    df = df[np.isfinite(df["detectability"])]
    if df.empty:
        raise SystemExit("nenhuma série com quebra e ambos os rótulos em t<=t_max")

    df["decile"] = pd.qcut(df["detectability"], args.n_deciles, labels=False, duplicates="drop")

    # peso da métrica: o que a TS-AUC cobra de cada passo é n_pos*n_neg. Erro "pesado" = série que
    # erra onde a métrica paga.
    w_by_t = early.groupby("t")["y"].agg(["sum", "size"])
    w_by_t["w"] = w_by_t["sum"] * (w_by_t["size"] - w_by_t["sum"])
    total_w = float(w_by_t["w"].sum())

    print(f"\nDetectabilidade x acerto, bucket t<={args.t_max}")
    print(f"{len(df)} séries com quebra; TS-AUC global do bucket = "
          f"{weighted_ts_auc(early['t'].to_numpy(), early['y'].to_numpy(), early['score'].to_numpy()):.4f}")
    print(f"peso da métrica concentrado neste bucket: {total_w:.3g}\n")

    header = (f"{'quintil':>8} {'n':>6} {'detect_med':>11} {'divergencia':>12} {'m_bucket':>9} "
              f"{'gap_med':>9} {'frac_gap<=0':>12}")
    print(header)
    print("-" * len(header))
    for d, grp in df.groupby("decile"):
        print(f"{int(d):8d} {len(grp):6d} {grp['detectability'].median():11.2f} "
              f"{grp['divergence'].median():12.2f} {grp['m_bucket'].median():9.0f} "
              f"{grp['gap'].median():9.4f} {(grp['gap'] <= 0).mean():12.2f}")

    # --- causa (1) vs causa (2): o gap DENTRO da série contra o espalhamento ENTRE séries ---
    # A TS-AUC não pontua o gap interno; pontua a ordenação transversal. Um modelo pode acertar o
    # sinal do gap em quase toda série e ainda assim ficar em ~0,5 se o NÍVEL de base variar entre
    # séries mais do que o gap. É exatamente a distinção do filtro da Seção 0: se
    # espalhamento >> gap, o sinal existe nas features e o que falha é a comparabilidade
    # transversal -- causa (2), não (1).
    xs_sd = early.groupby("t")["score"].std()
    w = early.groupby("t")["y"].agg(["sum", "size"])
    w = (w["sum"] * (w["size"] - w["sum"])).astype(float)
    xs_sd_w = float((xs_sd.reindex(w.index) * w).sum() / w.sum())
    gap_med = float(df["gap"].median())
    print(f"\ngap mediano DENTRO da série (t<={args.t_max}):      {gap_med:.4f}")
    print(f"dp do score ENTRE séries no mesmo passo:        {xs_sd_w:.4f}   "
          f"(ponderado por n_pos*n_neg)")
    print(f"razão gap/espalhamento:                        {gap_med / xs_sd_w:.2f}")
    print(f"fração de séries com gap>0:                    {(df['gap'] > 0).mean():.0%}")

    lo = df[df["decile"] == df["decile"].min()]
    hi = df[df["decile"] == df["decile"].max()]
    frac_lo = float((lo["gap"] <= 0).mean())
    frac_hi = float((hi["gap"] <= 0).mean())
    share_lo = len(lo) / len(df)

    print(f"\nquintil MENOS detectável: {frac_lo:.0%} das séries com gap<=0 ({share_lo:.0%} das séries)")
    print(f"quintil MAIS  detectável: {frac_hi:.0%} das séries com gap<=0")
    print(
        "\nleitura: se o quintil menos detectável concentra os gaps<=0 E o mais detectável já vai bem,\n"
        "o erro restante em t<=50 é dominado pela causa (3) -- piso informacional, nao lacuna de\n"
        "feature. Nesse caso F6/F7 tem teto baixo e o esforco deve parar em P1.\n"
        "Se o quintil MAIS detectavel ainda erra muito, sobra sinal na mesa: causa (1) ou (2), e as\n"
        "frentes de feature/calibracao valem."
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index()[
        ["id", "tau_index", "n_post", "m_bucket", "divergence", "detectability", "gap", "decile"]
    ].to_csv(out_path, index=False)
    print(f"\nsalvo em {out_path}")


if __name__ == "__main__":
    main()
