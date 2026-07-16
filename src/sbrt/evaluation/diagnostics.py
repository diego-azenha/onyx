"""Diagnósticos locais (plano §9.1) — respondem "o modelo aprendeu algo coerente, as features
fazem sentido, o código não vaza nem é instável", NUNCA "que TS-AUC isto tira" (§9.0). Nenhuma
função aqui calcula ou reporta uma estimativa de TS-AUC."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def training_curves(booster_evals: list, out_path: str | Path) -> None:
    """Plota logloss/AUC de LINHA por fold e rodada de boosting (métrica de treino do LightGBM,
    plano §8.3) — eixo rotulado explicitamente para não ser confundido com TS-AUC oficial."""
    out_path = Path(out_path)
    if not booster_evals:
        return

    metrics = sorted({m for fold in booster_evals for ds in fold.values() for m in ds})
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)

    for col, metric in enumerate(metrics):
        ax = axes[0][col]
        for fold_idx, fold in enumerate(booster_evals):
            for ds_name, ds_metrics in fold.items():
                if metric in ds_metrics:
                    ax.plot(ds_metrics[metric], label=f"fold {fold_idx} ({ds_name})")
        ax.set_title(f"{metric} de linha por rodada de boosting (métrica de TREINO, não TS-AUC)")
        ax.set_xlabel("rodada de boosting")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def feature_importance_report(ensemble, out_path: str | Path) -> pd.DataFrame:
    """Gain + split count por feature, agregados nos folds do ensemble."""
    gains = np.zeros(len(ensemble.feature_order))
    splits = np.zeros(len(ensemble.feature_order))
    for booster in ensemble.boosters:
        gains += np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
        splits += np.asarray(booster.feature_importance(importance_type="split"), dtype=np.float64)

    n = max(len(ensemble.boosters), 1)
    df = pd.DataFrame(
        {"feature": list(ensemble.feature_order), "gain_mean": gains / n, "split_mean": splits / n}
    ).sort_values("gain_mean", ascending=False).reset_index(drop=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def score_distribution_report(rows: pd.DataFrame, scores: np.ndarray, out_path: str | Path) -> None:
    """Histogramas de score por fatia de t, y=0 vs y=1 — inspeção visual de separação, sem reduzir
    a um número de desempenho (plano §9.1)."""
    out_path = Path(out_path)
    df = rows[["t", "y"]].copy()
    df["score"] = np.asarray(scores, dtype=np.float64)

    t_bins = [0, 50, 150, 400, np.inf]
    t_labels = ["t<=50", "50<t<=150", "150<t<=400", "t>400"]
    df["t_bucket"] = pd.cut(df["t"], bins=t_bins, labels=t_labels, right=True)

    fig, axes = plt.subplots(1, len(t_labels), figsize=(5 * len(t_labels), 4), sharey=True)
    for ax, label in zip(axes, t_labels):
        sub = df[df["t_bucket"] == label]
        ax.hist(sub.loc[sub["y"] == 0, "score"], bins=30, alpha=0.6, label="y=0", density=True)
        ax.hist(sub.loc[sub["y"] == 1, "score"], bins=30, alpha=0.6, label="y=1", density=True)
        ax.set_title(label)
        ax.set_xlabel("score")
        ax.legend()

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
