#!/usr/bin/env python
"""T2 (knowledge/frentes/teto-offline/T2-representacao-aprendida.md): as features de convolução
aleatória (estilo MiniRocket) encontram informação que as 118 features de duas amostras do T1 não
encontram? Mesmas amostras do oráculo do T1 (mesma semente), mesma CV."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surpresa_avaliar import _carregar_series  # noqa: E402
from t1_teto_offline import _amostras, _cv_auc  # noqa: E402

from offline.rocket import features_rocket, vieses_hist  # noqa: E402


def _uma(h, on, ini, L):
    v = vieses_hist(h)
    antes = np.concatenate([h, on[:ini]])
    return features_rocket(v, on[ini:ini + L], antes).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=4)
    args = ap.parse_args()
    series = _carregar_series(Path("data"))
    por_id = {sid: (h, on) for sid, h, on in series}
    yi = pd.read_parquet("data/y_train_index.parquet")
    am = _amostras(series, yi, args.L, np.random.default_rng(1000 + args.L))
    t1 = pd.read_parquet(f"artifacts/offline/t1_oraculo_L{args.L}.parquet")
    assert (t1["id"].to_numpy() == np.array([a[0] for a in am])).all(), "amostras diferentes do T1"
    R = np.stack(Parallel(n_jobs=args.n_jobs, batch_size=32)(
        delayed(_uma)(por_id[sid][0], por_id[sid][1], ini, args.L) for sid, _, ini in am))
    y = t1["y"].to_numpy()
    Xr = pd.DataFrame(R, columns=[f"rk{i}" for i in range(R.shape[1])])
    Xr.assign(id=t1["id"].to_numpy(), y=y).to_parquet(f"artifacts/offline/t2_rocket_L{args.L}.parquet")
    X1 = t1.drop(columns=["id", "y"])
    res = {"L": args.L, "n": int(len(y)),
           "auc_t1_maos": _cv_auc(X1, y)[0],
           "auc_rocket": _cv_auc(Xr, y)[0],
           "auc_t1_mais_rocket": _cv_auc(pd.concat([X1, Xr], axis=1), y)[0]}
    print(json.dumps(res), flush=True)
    Path(f"artifacts/offline/t2_L{args.L}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
