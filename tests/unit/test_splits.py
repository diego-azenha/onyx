import numpy as np
import pandas as pd

from sbrt.evaluation.splits import grouped_stratified_kfold


def _make_rows(n_series=40, seed=0):
    rng = np.random.RandomState(seed)
    rows = []
    for sid in range(n_series):
        t_max = rng.randint(20, 100)
        tau = rng.randint(1, t_max) if sid % 2 == 0 else None
        for t in range(1, t_max + 1):
            y = 1 if (tau is not None and t >= tau) else 0
            rows.append({"id": sid, "t": t, "y": y})
    return pd.DataFrame(rows)


def test_folds_partition_without_leaking_id_across_folds():
    rows = _make_rows()
    folds = list(grouped_stratified_kfold(rows, k=4, seed=0))
    assert len(folds) == 4

    for train_idx, valid_idx in folds:
        train_ids = set(rows.iloc[train_idx]["id"])
        valid_ids = set(rows.iloc[valid_idx]["id"])
        assert train_ids.isdisjoint(valid_ids)

    all_valid_positions = np.concatenate([valid_idx for _, valid_idx in folds])
    assert sorted(all_valid_positions.tolist()) == list(range(len(rows)))
