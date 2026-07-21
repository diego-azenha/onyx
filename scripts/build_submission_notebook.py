#!/usr/bin/env python
"""Gera `submission_notebook.ipynb` na raiz — notebook AUTOCONTIDO e LEGÍVEL para a plataforma Crunch.

Todo o pipeline (features de estado causais, whitening H0, calibração por série, ensemble LightGBM,
train/infer) é ACHATADO em células de código legíveis: cada módulo de `src/sbrt` vira uma célula, na
ordem de dependência, com os imports intra-pacote removidos. Um revisor lê o fluxo inteiro no
notebook e o código roda tal como no repositório. A única entrada externa são os dados raw do Crunch.

Colisões de nome entre módulos (medidas: só 4 reais) são resolvidas por renomeação cirúrgica:
`history_series`/`history_null_series` (ganham prefixo do módulo), `_acf` (difere entre h0 e
fingerprint), `apply`→`apply_monotonicity`, `train`→`train_ensemble`. As demais colisões têm corpo
idêntico (inofensivas ao serem redefinidas). A equivalência bit-a-bit com o pacote real é verificada
por `scripts/verify_submission_notebook.py`.

Re-gerável: rode após qualquer mudança de código.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Módulos inlinados, em ordem de dependência (nomes resolvem em tempo de chamada, mas mantemos uma
# ordem sensata para constantes/dataclasses/heranças).
MODULES = [
    "utils/numerics.py", "utils/ring_buffer.py", "config.py", "features/assembly.py",
    "state/accumulators.py", "state/cusum.py", "state/bayes_filter.py", "state/conformal.py",
    "state/rank_twosample.py", "state/mmd.py", "state/multiscale.py", "state/dependence.py",
    "state/varloc.py", "state/jumps.py", "state/bocpd.py", "state/fingerprint.py",
    "state/calibration.py", "state/h0.py", "postprocess/monotonicity.py", "model/fallback.py",
    "state/scorer.py", "model/base_rate.py", "model/predict.py", "evaluation/splits.py",
    "evaluation/ts_auc.py", "model/weights.py", "model/train.py", "model/dataset.py",
    "adapter/platform.py",
]

# Renomeações de identificadores DENTRO do módulo que os define (def + usos internos), por
# `\bnome\b`. Resolve colisões entre módulos e os aliases que os chamadores já usam.
RENAMES = {
    "state/mmd.py": [("history_reference", "mmd_history_reference"), ("history_series", "mmd_history_series")],
    "state/multiscale.py": [("history_series", "multiscale_history_series")],
    "state/dependence.py": [("history_null_series", "dependence_history_null_series")],
    "state/varloc.py": [("history_null_series", "varloc_history_null_series")],
    "state/jumps.py": [("history_null_series", "jumps_history_null_series")],
    "state/bocpd.py": [("history_null_series", "bocpd_history_null_series")],
    "state/fingerprint.py": [("_acf", "_acf_fp")],
    "state/h0.py": [("_acf", "_acf_h0")],
    "postprocess/monotonicity.py": [("apply", "apply_monotonicity")],
    # `train` NÃO entra aqui: `\btrain\b` pegaria `lgb.train`. Só a definição é renomeada (LITERAL).
}

# Substituições literais (chamadas qualificadas por alias de módulo, e a injeção do YAML embutido).
LITERAL = {
    "state/calibration.py": [
        ("mmd_mod.history_series", "mmd_history_series"),
        ("ms_mod.history_series", "multiscale_history_series"),
        ("dep_mod.history_null_series", "dependence_history_null_series"),
        ("varloc_mod.history_null_series", "varloc_history_null_series"),
        ("jump_mod.history_null_series", "jumps_history_null_series"),
        ("bocpd_mod.history_null_series", "bocpd_history_null_series"),
    ],
    "config.py": [
        ('DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"',
         "DEFAULT_CONFIG_PATH = None"),
        ('raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))',
         "raw = yaml.safe_load(_EMBEDDED_YAML)"),
    ],
    # só a DEFINIÇÃO do trainer é renomeada (evita colisão com o `train` adapter do platform.py);
    # `lgb.train`/`.train` ficam intactos. platform.py já o chama como `train_ensemble` (alias).
    "model/train.py": [("def train(", "def train_ensemble(")],
}


# OBRIGATÓRIO em toda célula do pipeline. O conversor do Crunch (`crunch_convert`, o passo
# notebook -> script que roda ANTES do treino na nuvem) opera em modo `keep:necessary` por padrão:
# só `def`/`class`/`import` sobrevivem — TODA atribuição de nível de módulo é COMENTADA. Sem este
# marcador, `_EMBEDDED_YAML` (a config inteira), `_NAN`, `_SQRT2`, `_CUSUM_BANK_KEYS`, `_MODEL_FILE`
# etc. somem e o submission quebra com NameError no primeiro passo — mesmo com o notebook rodando
# perfeitamente no Jupyter. `scripts/verify_submission_notebook.py` roda o conversor real e falha se
# alguma global for descartada.
KEEP_MARKER = "# @crunch/keep:on"


def _strip(src: str) -> str:
    """Remove `from __future__`, imports intra-pacote (`from sbrt.` / `import sbrt`) e blocos
    `if TYPE_CHECKING:` (que contêm imports do sbrt e não executam em runtime)."""
    lines = src.split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("from __future__ import"):
            i += 1
            continue
        if s.startswith("from sbrt.") or s.startswith("import sbrt"):
            i += 1
            continue
        if s == "if TYPE_CHECKING:":
            i += 1
            while i < len(lines) and (lines[i].strip() == "" or lines[i][:1] in (" ", "\t")):
                i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out).strip("\n")


def _process(rel: str) -> str:
    src = (SRC / "sbrt" / rel).read_text(encoding="utf-8")
    for old, new in RENAMES.get(rel, []):
        src = re.sub(rf"\b{re.escape(old)}\b", new, src)
    for old, new in LITERAL.get(rel, []):
        src = src.replace(old, new)
    header = (
        f"{KEEP_MARKER}\n"
        f"# ============================== sbrt/{rel} =============================="
    )
    return header + "\n" + _strip(src)


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build_notebook(token: str) -> dict:
    yaml_text = (ROOT / "configs" / "default.yaml").read_text(encoding="utf-8")

    intro = (
        "# ADIA Lab Structural Break Challenge — submissão `onyx` (autocontida)\n\n"
        "Este notebook contém **todo o pipeline inline e legível**: cada célula abaixo é um módulo do\n"
        "detector (features de estado causais O(1)/passo, whitening H0, calibração de nulo por série,\n"
        "ensemble LightGBM). Nada é importado de fora além das bibliotecas padrão e dos **dados raw**\n"
        "fornecidos pelo Crunch. `train()`/`infer()` (contrato Crunch) estão na última célula de código\n"
        "do pipeline (`adapter/platform.py`).\n\n"
        "> Latência ~0,95 ms/passo (medido); `train()` reconstrói o dataset de features pelo motor\n"
        "> único e ajusta o ensemble."
    )
    setup_env = (
        "%pip install crunch-cli lightgbm scikit-learn scipy joblib tqdm pyyaml --upgrade --quiet --progress-bar off\n"
        f"!crunch setup-notebook structural-break-real-time {token}"
    )
    load_tools = "import crunch\n\ncrunch_tools = crunch.load_notebook()"
    yaml_cell = KEEP_MARKER + "\n" \
                "# Configuração (configs/default.yaml embutida) — única fonte de números do pipeline.\n" \
                "_EMBEDDED_YAML = r'''\n" + yaml_text + "\n'''"

    cells = [
        _md(intro),
        _md("## Setup do ambiente (cole seu token)"),
        _code(setup_env),
        _code(load_tools),
        _md("## Pipeline `onyx` — módulos inline\n\nAs células abaixo definem o detector inteiro, na "
            "ordem de dependência. A última (`adapter/platform.py`) define `train()` e `infer()`."),
        _code(yaml_cell),
    ]
    for rel in MODULES:
        cells.append(_code(_process(rel)))

    cells += [
        _code("# @crunch/keep:on\n"
              "# infer() roda em N processos (fork); defina 1 se o ambiente usar spawn e algo falhar.\n"
              "INFER_PARALLELISM = 4\n# INFER_PARALLELISM = 1"),
        _md("## Teste local (mesmo fluxo da nuvem)"),
        _code("crunch_tools.test(\n    # force_first_train=False,\n    # no_determinism_check=True,\n)"),
        _md("### TS-AUC local"),
        _code(
            'import pandas as pd\nfrom sklearn.metrics import roc_auc_score\n\n'
            'y_test = pd.read_parquet("data/y_test.reduced.parquet")\n'
            'prediction = pd.read_parquet("prediction/prediction.parquet")\n'
            'merged = prediction.merge(y_test, how="left", left_index=True, right_index=True)\n'
            'merged["time_online"] = merged.groupby("id").cumcount()\n'
            "wsum = tot = 0.0\n"
            'for _, g in merged.groupby("time_online"):\n'
            '    lab = g["target"].values; n_pos = int(lab.sum()); n_neg = int((1 - lab).sum())\n'
            "    if n_pos == 0 or n_neg == 0:\n        continue\n"
            '    wsum += n_pos * n_neg * roc_auc_score(lab, g["prediction"].values); tot += n_pos * n_neg\n'
            'print(f"Local TS-AUC: {wsum / tot if tot else 0.5:.4f}")'
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", default="COLE_SEU_TOKEN_AQUI")
    parser.add_argument("--out", default=str(ROOT / "submission_notebook.ipynb"))
    args = parser.parse_args()

    nb = build_notebook(args.token)
    # ensure_ascii=True (não é cosmético): `crunch_convert.extract_from_file` abre o .ipynb SEM
    # `encoding=`, ou seja no encoding do locale — num Windows pt-BR (cp1252) o `crunch push` morre
    # com UnicodeDecodeError antes de converter qualquer coisa. Escapes \uXXXX são JSON válido e o
    # Jupyter os renderiza acentuados normalmente.
    Path(args.out).write_text(json.dumps(nb, ensure_ascii=True, indent=1), encoding="utf-8")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"gerado {args.out} ({len(nb['cells'])} células, {n_code} de código)")


if __name__ == "__main__":
    main()
