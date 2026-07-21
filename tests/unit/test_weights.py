import numpy as np
import pandas as pd

from sbrt.model.weights import compute_row_weights


def _rows(n_pos: int, n_neg: int, t: int = 1, thin_weight: float = 1.0) -> pd.DataFrame:
    y = [1] * n_pos + [0] * n_neg
    return pd.DataFrame({"t": [t] * len(y), "y": y, "thin_weight": [thin_weight] * len(y)})


def test_weights_balance_class_mass_within_step(cfg):
    # sem suavização (pseudo_count=0) e sem cap ativo, a massa de perda das duas classes se
    # equaliza exatamente dentro do passo (parecer §3.10: ambas = n_pos*n_neg).
    rows = _rows(n_pos=20, n_neg=200)
    w = compute_row_weights(rows, cfg, pseudo_count=0.0, max_ratio=50.0)
    y = rows["y"].to_numpy()
    mass_pos = w[y == 1].sum()
    mass_neg = w[y == 0].sum()
    assert abs(mass_pos - mass_neg) / max(mass_pos, mass_neg) < 1e-6


def test_weights_per_row_ratio_matches_class_count_ratio(cfg):
    rows = _rows(n_pos=20, n_neg=200)
    w = compute_row_weights(rows, cfg, pseudo_count=5.0, max_ratio=50.0)
    y = rows["y"].to_numpy()
    w_pos_row = w[y == 1][0]
    w_neg_row = w[y == 0][0]
    expected_ratio = (200 + 5.0) / (20 + 5.0)  # n_neg_s / n_pos_s
    assert abs((w_pos_row / w_neg_row) - expected_ratio) / expected_ratio < 1e-6


def test_weights_ratio_capped_for_extreme_imbalance(cfg):
    rows = _rows(n_pos=1, n_neg=5000)
    w = compute_row_weights(rows, cfg, pseudo_count=5.0, max_ratio=50.0)
    y = rows["y"].to_numpy()
    w_pos_row = w[y == 1][0]
    w_neg_row = w[y == 0][0]
    assert abs((w_pos_row / w_neg_row) - 50.0) / 50.0 < 1e-6


def test_weights_mean_is_one(cfg):
    rows = pd.concat(
        [_rows(n_pos=5, n_neg=500, t=1), _rows(n_pos=100, n_neg=150, t=2, thin_weight=2.0)],
        ignore_index=True,
    )
    w = compute_row_weights(rows, cfg)
    assert abs(w.mean() - 1.0) < 1e-9


def test_weights_thin_weight_multiplies_through(cfg):
    rows = _rows(n_pos=5, n_neg=50, thin_weight=4.0)
    w = compute_row_weights(rows, cfg)
    # todas as linhas do mesmo t/classe têm o mesmo peso; thin_weight só escala uniformemente
    assert np.allclose(w[rows["y"].to_numpy() == 1], w[rows["y"].to_numpy() == 1][0])
