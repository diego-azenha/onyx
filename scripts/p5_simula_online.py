#!/usr/bin/env python
"""Passo 5 do roteiro de 30/09 (knowledge/frentes/roteiro-30-09/P5-aprender-com-teste.md): simulação offline de
"aprender com o teste" com estado entre séries (INFER_PARALLELISM=1).

Um fold do OOF do E1 vira o fluxo de teste, processado em ordem (ids, depois 3 embaralhamentos).
Rotulador retrospectivo: classificador SEM tau (features de duas amostras do online inteiro contra o
histórico), treinado nos outros folds. Das séries já concluídas, as 10% de nota mais alta viram
"quebra", com tau estimado pela varredura de mudança de variância, e as 10% mais baixas viram
"normais". A cada 500 séries concluídas, uma logística determinística sobre (logit do aluno, log t e
6 colunas) é reajustada com essas pseudo-linhas e aplicada às séries seguintes. Métrica: TS-AUC da 2ª
metade do fluxo, com e sem a correção.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

from offline.features import contexto_hist, features  # noqa: E402
from sbrt.config import DEFAULT_CONFIG_PATH, load_config  # noqa: E402
from sbrt.evaluation.splits import grouped_stratified_kfold  # noqa: E402
from sbrt.evaluation.ts_auc import board_grid_multipliers, weighted_ts_auc  # noqa: E402

COLS = ["conformal_logm_abs", "mmd_joint_slow_cal", "cusum_var_up_r150", "bayes_lo_h0100"]
FAM = ["fam_escala_desce", "fam_escala_sobe"]
BLOCO = 500


def _feat_serie(sid, h, on):
    f = features(contexto_hist(h), on, h[-5:])
    f["id"] = sid
    return f


def _tau_hat(on: np.ndarray) -> int:
    n = len(on)
    if n < 30:
        return n // 2
    cs = np.cumsum(on * on)
    melhor, arg = -np.inf, n // 2
    for c in range(10, n - 10, 5):
        va, vb = cs[c - 1] / c, (cs[-1] - cs[c - 1]) / (n - c)
        v = cs[-1] / n
        ll = n * np.log(v + 1e-12) - c * np.log(va + 1e-12) - (n - c) * np.log(vb + 1e-12)
        if ll > melhor:
            melhor, arg = ll, c
    return arg


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oraculo", action="store_true",
                    help="teto: usa os rótulos VERDADEIROS de todas as séries concluídas no lugar dos pseudo-rótulos")
    args = ap.parse_args()
    cfg = load_config(DEFAULT_CONFIG_PATH)
    art = Path("artifacts/roteiro/p5"); art.mkdir(parents=True, exist_ok=True)
    yi = pd.read_parquet("data/y_train_index.parquet")
    series = _carregar_series(Path("data"))
    on_por_id = {sid: on for sid, _, on in series}
    cache = art / "feats_online_inteiro.parquet"
    if cache.exists():
        F = pd.read_parquet(cache)
    else:
        F = pd.DataFrame(Parallel(n_jobs=4, batch_size=32)(delayed(_feat_serie)(s, h, on) for s, h, on in series))
        F.to_parquet(cache)
    F["y_serie"] = (F["id"].map(yi["tau_index"]) >= 0).astype(int)
    fcols = [c for c in F.columns if c not in ("id", "y_serie", "n", "h_n")]

    rows = pd.read_parquet("data/processed/train_rows.parquet", columns=["id", "t", "y"] + COLS)
    oof = pd.read_parquet("artifacts/models/oof_e1_vema_bag4.parquet")
    sc = pd.read_parquet("artifacts/surpresa/scores.parquet", columns=["id", "t"] + FAM)
    d = rows.merge(oof[["id", "t", "oof_pred"]], on=["id", "t"]).merge(sc, on=["id", "t"])
    p = np.clip(d["oof_pred"].to_numpy(), 1e-9, 1 - 1e-9)
    d["logit"] = np.log(p) - np.log1p(-p)
    d["log_t"] = np.log(d["t"].to_numpy(float))
    xcols = ["logit", "log_t"] + COLS + FAM
    d[xcols] = d[xcols].fillna(0.0)
    w = board_grid_multipliers("data/y_train.parquet", d["t"].to_numpy())

    folds = list(grouped_stratified_kfold(d[["id", "t", "y"]], cfg.lightgbm.n_folds, cfg.seed))
    va_ids = np.unique(d["id"].to_numpy()[folds[0][1]])            # fold 0 = fluxo de teste
    tr_ids = np.setdiff1d(np.unique(d["id"]), va_ids)
    Ftr = F[F["id"].isin(tr_ids)]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, verbose=-1,
                  seed=0, num_threads=4, deterministic=True)
    lab = lgb.train(params, lgb.Dataset(Ftr[fcols], Ftr["y_serie"]), 300)
    Fva = F[F["id"].isin(va_ids)].set_index("id")
    nota_lab = pd.Series(lab.predict(Fva[fcols]), index=Fva.index)
    from sklearn.metrics import roc_auc_score
    rel = {"auc_rotulador_serie": float(roc_auc_score(Fva["y_serie"], nota_lab))}
    tau_hat = {int(i): _tau_hat(on_por_id[int(i)]) for i in va_ids}
    dv = d[d["id"].isin(va_ids)]
    por_id = {int(i): g for i, g in dv.groupby("id")}

    resultados = {}
    for nome, ordem in [("ordem_ids", np.sort(va_ids))] + [
            (f"embaralhado_{s}", np.random.default_rng(s).permutation(va_ids)) for s in (1, 2, 3)]:
        ordem = [int(i) for i in ordem]
        metade = len(ordem) // 2
        score_corr = {}
        modelo = None
        for k, sid in enumerate(ordem):
            g = por_id[sid]
            if modelo is not None:
                score_corr[sid] = modelo.decision_function(g[xcols].to_numpy())
            else:
                score_corr[sid] = g["logit"].to_numpy()
            if (k + 1) % BLOCO == 0:                                 # reajuste com as séries já concluídas
                feitas = ordem[: k + 1]
                if args.oraculo:
                    P = pd.concat([por_id[i][xcols].assign(yy=por_id[i]["y"].to_numpy()) for i in feitas])
                    modelo = LogisticRegression(C=1.0, max_iter=500).fit(P[xcols].to_numpy(), P["yy"].to_numpy())
                    continue
                notas = nota_lab.loc[feitas]
                alto = notas[notas >= notas.quantile(0.9)].index
                baixo = notas[notas <= notas.quantile(0.1)].index
                partes = []
                for i in alto:
                    gi = por_id[int(i)]
                    partes.append(gi[xcols].assign(yy=(gi["t"].to_numpy() > tau_hat[int(i)]).astype(int)))
                for i in baixo:
                    partes.append(por_id[int(i)][xcols].assign(yy=0))
                P = pd.concat(partes)
                if P["yy"].nunique() == 2:
                    modelo = LogisticRegression(C=1.0, max_iter=500).fit(P[xcols].to_numpy(), P["yy"].to_numpy())
        seg = [sid for sid in ordem[metade:]]
        G2 = pd.concat([por_id[s] for s in seg])
        base = G2["logit"].to_numpy()
        corr = np.concatenate([score_corr[s] for s in seg])
        resultados[nome] = {"tsauc_2a_metade_base": weighted_ts_auc(G2.t.values, G2.y.values, base, w),
                            "tsauc_2a_metade_corrigido": weighted_ts_auc(G2.t.values, G2.y.values, corr, w)}
        resultados[nome]["delta"] = resultados[nome]["tsauc_2a_metade_corrigido"] - resultados[nome]["tsauc_2a_metade_base"]
        print(nome, json.dumps(resultados[nome]), flush=True)
    rel["resultados"] = resultados
    rel["delta_medio"] = float(np.mean([r["delta"] for r in resultados.values()]))
    (art / ("p5_oraculo.json" if args.oraculo else "p5.json")).write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
