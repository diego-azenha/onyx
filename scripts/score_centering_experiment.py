#!/usr/bin/env python
"""Centragem de score por série: o baseline H0 de cada série medido sobre a cauda do PRÓPRIO
histórico (docs/BACKLOG_TSAUC.md, frente de centragem).

## O que este experimento testa

A TS-AUC compara séries DIFERENTES no mesmo passo. Medido sobre `oof_v4` no bucket `t<=50`: o gap
que o modelo produz DENTRO de uma série é 0,0462, mas o efeito transversal implicado pela AUC é
0,0064 — **7,3x menor**. O modelo acerta a direção em 91% das séries; ~86% do poder discriminativo é
destruído porque cada série tem um NÍVEL de score próprio, e esse nível varia tanto quanto o efeito
da quebra.

Centrar cada série pelo próprio nível resolveria isso. Com oráculo (subtraindo a média pré-quebra
real, que usa o rótulo e portanto não é implementável) a TS-AUC vai de 0,5357 para 0,8026 em `t<=50`
e de 0,6100 para 0,6753 no agregado.

Este script implementa a versão CAUSAL: o baseline vem do histórico, que é H0 por definição.

## Por que da CAUDA do histórico, com índice `t` alinhado

O proxy óbvio — usar os primeiros passos online da própria série — foi medido e falha (+0,011).
O RMSE contra o baseline verdadeiro é 1,37x o desvio entre séries, contra os 0,32x que a teoria de
ruído previa para aquele tamanho de amostra. A causa é **viés de regime, não ruído**: nos primeiros
passos online as features ainda estão em warm-up (janelas não cheias, NaN), então aqueles scores não
são amostras do mesmo regime que t=30.

Daí o desenho: rodar o modelo sobre os últimos N pontos do histórico COMO SE fossem online, com
`t = 1..N`. Assim o baseline de cada passo é medido no mesmo índice `t` do passo online que vai
corrigir, e o viés de warm-up cancela por construção.

## Por que isto não é a recalibração pós-hoc que já foi descartada

Recalibração global (Platt/isotônica/offset) é **neutra** para a TS-AUC pela invariância C1: um
transform monotônico aplicado igual a todas as séries preserva a ordenação dentro do passo. A
centragem aqui é **por série** — cada uma subtrai um valor diferente — logo muda a ordenação
transversal e não é coberta por C1. É exatamente a distinção que `informacao_nao_capturada.md` §2
fazia.

## Espaço de trabalho: logit cru

`oof_pred` é `sigmoid(raw + init_score)`, onde `init_score = logit(p_hat(t))` é o offset de taxa-base
(model/base_rate.py). O baseline sai de `booster.predict(raw_score=True)`, sem offset. Subtrair nas
duas escalas misturaria transformações diferentes. Trabalhando em logit cru,

    centrado(t) = [logit(oof_pred(t)) - offset(t)] - raw_hist(t) = raw_online(t) - raw_hist(t)

o offset cancela exatamente. Como a TS-AUC só depende de ordem dentro do passo, trabalhar em logit é
livre.

## Honestidade do OOF

Cada série é pontuada pelo booster do fold que **não** a treinou — o mesmo `grouped_stratified_kfold`
do treino, reproduzido a partir de (id, t, y). Sem isso o baseline veria a série por dentro e o
experimento mediria vazamento em vez de sinal.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc
from sbrt.features.assembly import to_array
from sbrt.model.base_rate import predict_base_rate_logit
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks
from sbrt.utils.ring_buffer import RingBuffer

T_BUCKETS = ((50, "t<=50"), (150, "t<=150"), (400, "t<=400"), (10**9, "geral"))


def _baseline_for_series(
    hist: np.ndarray, booster, feature_order: tuple, n_tail: int, cfg, h0_mode: str = "holdout"
) -> np.ndarray:
    """Roda o modelo sobre `hist[-n_tail:]` como pseudo-online e devolve o score CRU (logit sem
    offset) em cada passo `t = 1..n_tail`.

    `t` recomeça em 1, para que o warm-up das features seja idêntico ao do segmento online que este
    baseline vai corrigir — é o que faz o viés de regime cancelar, e é a razão de o proxy "primeiros
    passos online" ter falhado.

    ## `h0_mode` — e por que `holdout` é o default

    - `full`: ajusta o H0 no histórico INTEIRO. Simples, mas a cauda pontuada participou do ajuste
      (coeficientes AR, sigma_e, quantis, `null_stats`), então os resíduos dela são bons demais e os
      scores saem **comprimidos**. Medido em 500 séries: dp do nível histórico 0,247 contra 0,458 do
      nível online — atenuação de ~46%, e a correlação com o nível online fica em 0,322.
    - `holdout`: ajusta o H0 em `hist[:-n_tail]`, deixando a cauda genuinamente fora da amostra —
      exatamente a relação que o online tem com o histórico. Bônus de arquitetura: `seed_lag_buffer`
      passa a semear certo sozinho, porque `h0.lag_seed` já é o fim de `hist[:-n_tail]`, que é onde o
      pseudo-online começa.

    O custo do `holdout` é o H0 ver `n_tail` pontos a menos (~7% do histórico mediano de 3.000).
    """
    if h0_mode == "holdout":
        h0 = fit_h0(hist[:-n_tail], cfg)
        scorer = StreamScorer(h0, default_blocks(), None, cfg)  # lag_seed já é o fim de hist[:-n_tail]
    else:
        h0 = fit_h0(hist, cfg)
        scorer = StreamScorer(h0, default_blocks(), None, cfg)
        cap = h0.lag_capacity
        start = len(hist) - n_tail
        buf = RingBuffer(cap)
        for x in hist[max(start - cap, 0): start]:
            buf.push(float(x))
        scorer.lags = buf  # o default semearia do FIM do histórico, n_tail pontos à frente daqui

    rows = np.empty((n_tail, len(feature_order)), dtype=np.float64)
    for j, x in enumerate(hist[len(hist) - n_tail:]):
        feats = scorer.update_features(float(x))
        rows[j] = to_array(feats, feature_order)
    # um único predict em lote: ~200 chamadas individuais dominariam o custo por série
    return booster.predict(rows, raw_score=True).astype(np.float64)


def _run_fold(ids, hist_by_id, booster, feature_order, n_tail, cfg, n_jobs, h0_mode):
    out = Parallel(n_jobs=n_jobs)(
        delayed(_baseline_for_series)(hist_by_id[i], booster, feature_order, n_tail, cfg, h0_mode)
        for i in ids
    )
    return dict(zip(ids, out))


def _load_histories(data_dir: Path, wanted: set) -> dict:
    X = pd.read_parquet(data_dir / "X_train.parquet")
    out = {}
    for dataset_id, group in X.groupby(level="id"):
        if int(dataset_id) not in wanted:
            continue
        g = group.reset_index()
        out[int(dataset_id)] = g.loc[g["period"] == 1, "value"].to_numpy(dtype="float64")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model", default="artifacts/models/v4")
    parser.add_argument("--oof", default="artifacts/models/oof_v4.parquet")
    parser.add_argument("--n-tail", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--h0-mode", choices=("holdout", "full"), default="holdout",
                        help="holdout: H0 ajustado em hist[:-n_tail], cauda fora da amostra (default); "
                             "full: H0 no historico inteiro, cauda in-sample (comprime o baseline)")
    parser.add_argument("--precheck", action="store_true",
                        help="fase 1a: só a amostra, e reporta a correlação com o baseline-oráculo")
    parser.add_argument("--n-series", type=int, default=None, help="limita o nº de séries (amostra determinística)")
    parser.add_argument("--baselines-out", default="artifacts/reports/centering_baselines.parquet")
    parser.add_argument("--out", default=None, help="parquet OOF centrado, para o compare_oof.py")
    parser.add_argument("--variant", default="centrado", help="qual variante gravar em --out")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_dir = Path(args.model)
    boosters = joblib.load(model_dir / "boosters.joblib")
    feature_order = tuple(json.loads((model_dir / "feature_schema.json").read_text(encoding="utf-8")))
    base_rate_curve = json.loads((model_dir / "base_rate_curve.json").read_text(encoding="utf-8"))

    oof = pd.read_parquet(args.oof)

    # mesmo split do treino, reproduzido de (id, t, y) — `grouped_stratified_kfold` não usa mais nada
    fold_of = {}
    for k, (_, valid_pos) in enumerate(grouped_stratified_kfold(oof, len(boosters), cfg.seed)):
        for i in oof.iloc[valid_pos]["id"].unique():
            fold_of[int(i)] = k
    assert len(fold_of) == oof["id"].nunique(), "split não cobriu todas as séries"

    ids_all = sorted(fold_of)
    if args.precheck and args.n_series is None:
        args.n_series = 500
    if args.n_series:
        step = max(len(ids_all) // args.n_series, 1)
        ids_all = ids_all[::step][: args.n_series]
    wanted = set(ids_all)
    print(f"[centering] {len(wanted)} séries, n_tail={args.n_tail}, h0_mode={args.h0_mode}, "
          f"modelo={model_dir}")

    hist_by_id = _load_histories(Path(args.data_dir), wanted)
    too_short = [i for i in ids_all if len(hist_by_id[i]) < args.n_tail + 64]
    if too_short:
        print(f"[centering] {len(too_short)} séries com histórico curto demais para n_tail — excluídas")
        ids_all = [i for i in ids_all if i not in set(too_short)]
        wanted = set(ids_all)

    baselines: dict = {}
    for k in tqdm(range(len(boosters)), desc="folds"):
        fold_ids = [i for i in ids_all if fold_of[i] == k]
        if not fold_ids:
            continue
        baselines.update(
            _run_fold(fold_ids, hist_by_id, boosters[k], feature_order, args.n_tail, cfg,
                      args.n_jobs, args.h0_mode)
        )

    bl = pd.DataFrame(
        {"id": np.repeat(list(baselines), args.n_tail),
         "t": np.tile(np.arange(1, args.n_tail + 1), len(baselines)),
         "baseline_raw": np.concatenate([baselines[i] for i in baselines])}
    )
    Path(args.baselines_out).parent.mkdir(parents=True, exist_ok=True)
    bl.to_parquet(args.baselines_out)
    print(f"[centering] baselines salvos em {args.baselines_out}")

    # --- score online em logit cru: remove o offset de taxa-base que o oof_pred carrega ---
    d = oof[oof["id"].isin(wanted)].copy()
    p = d["oof_pred"].to_numpy(dtype=np.float64).clip(1e-9, 1 - 1e-9)
    d["raw_online"] = np.log(p / (1 - p)) - predict_base_rate_logit(d["t"].to_numpy(), base_rate_curve)

    if args.precheck:
        # a premissa inteira: o nível H0 do histórico prevê o nível pré-quebra online?
        rng_t = d["t"] <= args.n_tail
        oracle = d[rng_t & (d["y"] == 0)].groupby("id")["raw_online"].mean().rename("oraculo")
        hist_lvl = bl.groupby("id")["baseline_raw"].mean().rename("historico")
        j = pd.concat([oracle, hist_lvl], axis=1).dropna()
        corr = float(j["oraculo"].corr(j["historico"]))
        print(f"\n=== PRÉ-RASTREIO (fase 1a) — {len(j)} séries ===")
        print(f"corr(nível H0 do histórico, nível pré-quebra online) = {corr:.3f}")
        print(f"dp do nível online entre séries: {j['oraculo'].std():.4f}")
        print(f"dp do nível histórico entre séries: {j['historico'].std():.4f}")
        print(f"RMSE do histórico como estimador: {np.sqrt(((j['oraculo']-j['historico'])**2).mean()):.4f}")

        # A correlação acima é ATENUADA pelo ruído do CRITÉRIO: o "nível online verdadeiro" de uma
        # série é estimado pela média de seus passos pré-quebra, e séries de quebra precoce têm
        # pouquíssimos. Estratificar por quantos passos existem separa "o histórico não prevê" de
        # "o alvo está mal medido" — que levam a decisões opostas.
        n_pre = d[rng_t & (d["y"] == 0)].groupby("id").size().rename("n_pre")
        js = pd.concat([j, n_pre], axis=1).dropna()
        print("\n{:>18s} {:>9s} {:>7s}".format("passos pre-quebra", "n series", "corr"))
        for lo, hi in ((1, 20), (21, 60), (61, 120), (121, 10**9)):
            sub = js[(js["n_pre"] >= lo) & (js["n_pre"] <= hi)]
            if len(sub) < 20:
                continue
            label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
            print("{:>18s} {:9d} {:7.3f}".format(
                label, len(sub), sub["oraculo"].corr(sub["historico"])))
        print(f"\nGATE: {'PASSA' if corr >= 0.3 else 'FALHA'} (limiar 0,30) — "
              f"{'seguir para a fase 1b' if corr >= 0.3 else 'parar; o histórico não prevê o nível online'}")
        return

    merged = d.merge(bl, on=["id", "t"], how="left")
    # para t > n_tail o baseline é o do último passo medido (o nível já estabilizou)
    tail_val = bl[bl["t"] == args.n_tail].set_index("id")["baseline_raw"]
    merged["baseline_raw"] = merged["baseline_raw"].fillna(merged["id"].map(tail_val))
    merged = merged.dropna(subset=["baseline_raw"])

    # --- variantes, todas sobre o MESMO passe de baseline (que é o custo real) ---
    #
    # `beta` encolhe a correção. O baseline do histórico é um estimador ATENUADO do nível online:
    # medido, dp 0,247 contra 0,458, com correlação 0,32 no agregado (0,65 onde o critério é bem
    # medido). Subtrair o baseline cru corrige demais nas séries em que ele exagera; o fator ótimo é
    # o coeficiente de regressão do nível online sobre o do histórico, estimado aqui nos próprios
    # dados. Sem isso a centragem injeta ruído em vez de removê-lo.
    b = merged["baseline_raw"]
    b_centered = b - b.mean()
    variants = {"centrado": merged["raw_online"] - b_centered}
    for beta in (0.25, 0.5, 0.75):
        variants[f"centrado_b{beta:.2f}"] = merged["raw_online"] - beta * b_centered
    # aplicar só onde o nível domina: em t alto o segmento online já tem amostra própria e a
    # correção vira ruído (padrão visto na amostra de 500)
    for t0 in (50, 100, 200):
        variants[f"centrado_ate_t{t0}"] = np.where(
            merged["t"] <= t0, merged["raw_online"] - b_centered, merged["raw_online"]
        )

    print(f"\n{'bucket':10s} {'base':>9s}" + "".join(f"{k:>18s}" for k in variants))
    print("-" * (20 + 18 * len(variants)))
    best_by_bucket = {}
    for tmax, label in T_BUCKETS:
        s = merged[merged["t"] <= tmax]
        if s.empty:
            continue
        t, y = s["t"].to_numpy(), s["y"].to_numpy()
        a0 = weighted_ts_auc(t, y, s["oof_pred"].to_numpy())
        cells, best = [], (None, -9)
        for name, v in variants.items():
            a1 = weighted_ts_auc(t, y, np.asarray(v)[merged["t"].to_numpy() <= tmax])
            cells.append(f"{a1:9.4f}({a1-a0:+.4f})")
            if a1 > best[1]:
                best = (name, a1)
        best_by_bucket[label] = best
        print(f"{label:10s} {a0:9.4f}" + "".join(f"{c:>18s}" for c in cells))

    print("\nmelhor variante por bucket:")
    for label, (name, a) in best_by_bucket.items():
        print(f"  {label:10s} {name}  ({a:.4f})")

    if args.out:
        pick = args.variant if args.variant in variants else "centrado"
        out = merged[["id", "t", "y"]].copy()
        out["oof_pred"] = np.asarray(variants[pick])  # nome esperado por compare_oof.py
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out)
        print(f"\nvariante '{pick}' salva em {args.out} — julgar por scripts/compare_oof.py")


if __name__ == "__main__":
    main()
