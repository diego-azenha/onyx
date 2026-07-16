import dataclasses

import numpy as np
import pandas as pd

from sbrt.evaluation.diagnostics import feature_importance_report, score_distribution_report, training_curves
from sbrt.model.dataset import SeriesRecord, build_training_rows
from sbrt.model.train import train as train_ensemble
from sbrt.model.weights import compute_row_weights


def _tiny_supervised_run(cfg, tmp_path):
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, mode="supervised"))
    cfg = dataclasses.replace(
        cfg, lightgbm=dataclasses.replace(cfg.lightgbm, n_folds=3, n_estimators_cap=20, early_stopping_rounds=5)
    )
    rng = np.random.RandomState(21)
    records = []
    for i in range(30):
        hist = rng.randn(1100)
        online = rng.randn(120)
        tau = None
        if i % 2 == 0:
            tau = int(rng.randint(10, 110))
            online[tau:] += 1.2
        records.append(SeriesRecord(dataset_id=i, x_hist=hist, x_online=online, tau_index=tau))
    rows = build_training_rows(records, cfg, progress=False)
    weights = compute_row_weights(rows, cfg)
    ensemble, oof_pred = train_ensemble(rows, weights, cfg, progress=False)
    return rows, ensemble, oof_pred


def test_diagnostics_run_without_error_on_synthetic_data(cfg, tmp_path):
    """docs/PLANO_REPOSITORIO.md §6: as curvas/relatórios são gerados sem erro, com dados
    sintéticos pequenos — não é teste de "o modelo aprendeu bem o suficiente" (plano §9.0)."""
    rows, ensemble, oof_pred = _tiny_supervised_run(cfg, tmp_path)

    training_curves(ensemble.fold_evals, tmp_path / "curves.png")
    assert (tmp_path / "curves.png").exists()

    df = feature_importance_report(ensemble, tmp_path / "importance.csv")
    assert (tmp_path / "importance.csv").exists()
    assert set(df.columns) == {"feature", "gain_mean", "split_mean"}
    assert len(df) == len(ensemble.feature_order)

    assert np.all(np.isfinite(oof_pred))
    assert np.all((oof_pred >= 0) & (oof_pred <= 1))
    score_distribution_report(rows, oof_pred, tmp_path / "dist.png")
    assert (tmp_path / "dist.png").exists()
