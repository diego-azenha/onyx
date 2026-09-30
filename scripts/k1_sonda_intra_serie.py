#!/usr/bin/env python
"""Passo 1 do roteiro de 30/09 (knowledge/frentes/roteiro-30-09/P1-sonda.md): sonda DENTRO de cada série.

Para cada série, um classificador pequeno aprende a separar janelas de 20 pontos do fim do histórico
("antes") das janelas do trecho [tau, tau+400) ("depois"), com validação por blocos contínuos. A nota da
série é a AUC nas janelas de teste. Positivos: tau verdadeiro; negativos: pseudo-tau ~ U[0, len-400].
Sanidade: negativos com dependência AR(1) phi=0,3 injetada no trecho "depois" (variância preservada).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402

W, L_POS, N_PRE = 20, 400, 800
GRUPOS = {"ordenados": list(range(0, 20)), "momentos": [20, 21, 22], "defasagens": [23, 24, 25]}


def _janelas(x: np.ndarray, passo: int) -> np.ndarray:
    return sliding_window_view(x, W)[::passo]


def _feats(jan: np.ndarray) -> np.ndarray:
    srt = np.sort(jan, axis=1)
    mom = np.column_stack([jan.mean(1), np.abs(jan).mean(1), (jan ** 2).mean(1)])
    lag = np.column_stack([(jan[:, k:] * jan[:, :-k]).mean(1) for k in (1, 2, 3)])
    return np.hstack([srt, mom, lag])


def _blocos(n_jan: int, passo: int, k: int = 4):
    """k blocos contínuos no eixo do tempo; para cada bloco b, (treino, teste) em índices de janela.
    Janelas do treino a menos de W pontos do bloco de teste são descartadas (sem sobreposição)."""
    ini = np.arange(n_jan) * passo                      # ponto inicial de cada janela
    fim_total = ini[-1] + W
    cortes = np.linspace(0, fim_total, k + 1)
    for b in range(k):
        a, z = cortes[b], cortes[b + 1]
        te = np.flatnonzero((ini >= a) & (ini + W <= z))
        tr = np.flatnonzero((ini + W <= a - W) | (ini >= z + W))
        yield tr, te


def sonda(pre: np.ndarray, pos: np.ndarray, grupos: bool = False) -> dict:
    """CV de 4 blocos contínuos em cada trecho; AUC agregada sobre todas as janelas de teste."""
    jp, jq = _janelas(pre, 2), _janelas(pos, 1)
    Fp, Fq = _feats(jp), _feats(jq)
    chaves = ["lr", "hgb"] + ([f"{g}" for g in GRUPOS] if grupos else [])
    pred = {c: ([], []) for c in chaves}                 # (scores, rótulos)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (tr_p, te_p), (tr_q, te_q) in zip(_blocos(len(jp), 2), _blocos(len(jq), 1)):
            Xtr = np.vstack([Fp[tr_p], Fq[tr_q]]); ytr = np.r_[np.zeros(len(tr_p)), np.ones(len(tr_q))]
            Xte = np.vstack([Fp[te_p], Fq[te_q]]); yte = np.r_[np.zeros(len(te_p)), np.ones(len(te_q))]
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
            lr = LogisticRegression(C=1.0, max_iter=300).fit(Ztr, ytr)
            pred["lr"][0].append(lr.decision_function(Zte)); pred["lr"][1].append(yte)
            Rtr = np.vstack([jp[tr_p], jq[tr_q]]); Rte = np.vstack([jp[te_p], jq[te_q]])
            hgb = HistGradientBoostingClassifier(max_depth=3, max_iter=50, learning_rate=0.1,
                                                 early_stopping=False, random_state=0)
            hgb.fit(np.hstack([Ztr, Rtr]), ytr)
            pred["hgb"][0].append(hgb.predict_proba(np.hstack([Zte, Rte]))[:, 1]); pred["hgb"][1].append(yte)
            if grupos:
                for g, idx in GRUPOS.items():
                    m = LogisticRegression(C=1.0, max_iter=300).fit(Ztr[:, idx], ytr)
                    pred[g][0].append(m.decision_function(Zte[:, idx])); pred[g][1].append(yte)
    # AUC por bloco de teste e média (os scores de blocos diferentes vêm de modelos diferentes)
    out = {}
    for c, (ss, yy) in pred.items():
        aucs = [roc_auc_score(y, s) for s, y in zip(ss, yy) if 0 < y.mean() < 1]
        out[f"auc_{c}" if c in ("lr", "hgb") else f"auc_{c}"] = float(np.mean(aucs))
    out["nota"] = max(out["auc_lr"], out["auc_hgb"])
    return out


def _injeta_dependencia(x: np.ndarray, phi: float = 0.3) -> np.ndarray:
    y = np.empty_like(x)
    y[0] = x[0]
    c = np.sqrt(1 - phi * phi)
    for t in range(1, len(x)):
        y[t] = phi * y[t - 1] + c * x[t]
    return y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--n-injetados", type=int, default=500)
    ap.add_argument("--out", default="artifacts/roteiro/p1_sonda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    yi = pd.read_parquet("data/y_train_index.parquet")
    rng = np.random.default_rng(3011)
    tarefas = []
    for sid, h, on in _carregar_series(Path("data")):
        tau = int(yi.loc[sid, "tau_index"])
        pre = h[-N_PRE:]
        if tau >= 0:
            if len(on) - tau < L_POS:
                continue
            post_full = on[tau:]
            dv = np.log(post_full.var() / h.var())
            tipo = "sobe" if dv > 0.3 else ("desce" if dv < -0.3 else "var_fixa")
            tarefas.append((sid, 1, tipo, pre, on[tau:tau + L_POS]))
        else:
            if len(on) < L_POS:
                continue
            p = int(rng.integers(0, len(on) - L_POS + 1))
            tarefas.append((sid, 0, "neg", pre, on[p:p + L_POS]))
    negs = [t for t in tarefas if t[1] == 0]
    inj = [(sid, 1, "injetado", pre, _injeta_dependencia(pos)) for sid, _, _, pre, pos in negs[: args.n_injetados]]
    todas = tarefas + inj
    res = Parallel(n_jobs=args.n_jobs, batch_size=16)(
        delayed(sonda)(pre, pos, grupos=(tipo in ("var_fixa", "neg"))) for _, _, tipo, pre, pos in todas)
    df = pd.DataFrame([{"id": sid, "y": y, "tipo": tipo, **r} for (sid, y, tipo, _, _), r in zip(todas, res)])
    df.to_parquet(out / "notas.parquet")
    neg = df[df.tipo == "neg"]
    rel = {"n": df.tipo.value_counts().to_dict()}

    def auc_grupo(g, col="nota", base=neg):
        x = pd.concat([g[[col]].assign(y=1), base[[col]].assign(y=0)])
        return float(roc_auc_score(x.y, x[col]))

    var = df[df.tipo.isin(["sobe", "desce"])]
    inj_df = df[df.tipo == "injetado"]
    base_inj = neg[neg.id.isin(inj_df.id)]
    rel["sanidade_a_variancia_vs_neg"] = auc_grupo(var)
    rel["sanidade_b_injetado_vs_mesmos_neg"] = auc_grupo(inj_df, base=base_inj)
    fixa = df[df.tipo == "var_fixa"]
    rel["PRINCIPAL_var_fixa_vs_neg"] = auc_grupo(fixa)
    for col in ["auc_lr", "auc_hgb", "auc_ordenados", "auc_momentos", "auc_defasagens"]:
        rel[f"var_fixa_vs_neg_{col}"] = auc_grupo(fixa, col)
    for tp in ["sobe", "desce", "var_fixa", "neg", "injetado"]:
        rel[f"nota_media_{tp}"] = float(df[df.tipo == tp].nota.mean())
    (out / "p1.json").write_text(json.dumps(rel, indent=2), encoding="utf-8")
    print(json.dumps(rel, indent=2))


if __name__ == "__main__":
    main()
