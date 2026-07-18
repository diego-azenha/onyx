#!/usr/bin/env python
"""CLI fina: roda o ciclo completo de validacao das alteracoes do plano de acao
(docs/DIAGNOSTICO_TS_AUC.md) -- testes -> build_dataset -> train -> diagnose -> robustness (com
modelo treinado) -> decomposicao da TS-AUC OOF por bucket de t. Cada etapa e um subprocesso do
script correspondente em scripts/, na mesma ordem recomendada no plano de acao. Para no primeiro
passo que falhar."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def run_step(name: str, cmd: list[str]) -> None:
    print(f"\n{'=' * 70}\n[run_all] {name}\n{'=' * 70}", flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"\n[run_all] FALHOU em '{name}' ({dt:.0f}s, exit={result.returncode}) -- interrompendo")
        sys.exit(result.returncode)
    print(f"[run_all] OK: {name} ({dt:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model-dir", default="artifacts/models/v1")
    parser.add_argument("--skip-tests", action="store_true", help="pula pytest (tests/unit, causality, determinism)")
    parser.add_argument(
        "--skip-dataset", action="store_true",
        help="pula o rebuild do dataset -- usa data/processed/train_rows.parquet existente",
    )
    parser.add_argument(
        "--skip-robustness", action="store_true",
        help="pula a suite de robustez T1-T13 (a etapa mais lenta, dezenas de minutos)",
    )
    args = parser.parse_args()

    py = sys.executable
    cfg = args.config
    t_start = time.time()

    if not args.skip_tests:
        run_step(
            "testes (unit + causality + determinism)",
            [py, "-m", "pytest", "tests/unit", "tests/causality", "tests/determinism", "-q"],
        )
    else:
        print("[run_all] pulando testes (--skip-tests)")

    if not args.skip_dataset:
        run_step("build_dataset", [py, "scripts/build_dataset.py", "--config", cfg])
    else:
        print("[run_all] pulando build_dataset (--skip-dataset)")

    run_step("train", [py, "scripts/train.py", "--config", cfg])
    run_step("diagnose", [py, "scripts/diagnose.py", "--config", cfg])

    if not args.skip_robustness:
        run_step(
            "robustness suite (com modelo treinado)",
            [py, "scripts/run_robustness_suite.py", "--config", cfg, "--model", args.model_dir],
        )
    else:
        print("[run_all] pulando robustness suite (--skip-robustness)")

    run_step("TS-AUC OOF por bucket de t", [py, "scripts/oof_ts_auc_by_bucket.py"])

    total_min = (time.time() - t_start) / 60.0
    print(f"\n[run_all] ciclo completo em {total_min:.1f} min")


if __name__ == "__main__":
    main()
