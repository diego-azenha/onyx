#!/usr/bin/env python
"""CLI fina: roda adapter/platform.py fim a fim (plano §15.1 P0) contra dados sintéticos pequenos,
sem depender do crunch CLI/rede — verifica o protocolo exato do `GeneratorWrapper` (1º yield None,
depois exatamente um score por ponto online, generator exaurido ao final). Para validação real
contra a plataforma, use `crunch test` diretamente (precisa de `crunch setup-notebook` autenticado)."""
from __future__ import annotations

import argparse
import tempfile

import numpy as np

from sbrt.adapter import platform


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-series", type=int, default=10)
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    train_series = []
    for i in range(args.n_series):
        hist = rng.randn(1200).tolist()
        online = rng.randn(300)
        tau = None
        if i % 2 == 0:
            tau = int(rng.randint(20, 250))
            online[tau:] += 1.2
        train_series.append((i, hist, online.tolist(), tau))

    with tempfile.TemporaryDirectory() as model_dir:
        platform.train(train_series, model_dir)
        print("train() OK")

        test_series = [(hist, online) for _, hist, online, _ in train_series]
        gen = platform.infer(test_series, model_dir)
        first = next(gen)
        assert first is None, "primeiro yield deve ser None (sinaliza prontidão ao runner)"

        total = 0
        for _, online in test_series:
            for _ in online:
                val = next(gen)
                assert isinstance(val, float) and 0.0 <= val <= 1.0
                total += 1
        try:
            next(gen)
            raise AssertionError("generator deveria estar exaurido após o último score")
        except StopIteration:
            pass

        print(f"infer() OK — {total} scores emitidos, generator exaurido corretamente")


if __name__ == "__main__":
    main()
