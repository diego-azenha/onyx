#!/usr/bin/env python
"""Portão de aceitação de `submission_notebook.ipynb`: prova que o pipeline ACHATADO no notebook é
bit-a-bit equivalente ao pacote `sbrt` testado do repositório -- nas TRÊS formas em que ele roda.

1. `células`: executa as células do pipeline (pulando pip/crunch/test) numa namespace isolada ->
   `train`/`infer` puramente do que está inline. É o que roda quando alguém abre o .ipynb.
2. `convertido`: passa o notebook pelo conversor OFICIAL (`crunch_convert`, o mesmo do `crunch push`)
   e executa o script resultante. É o que roda de fato na nuvem -- e o único estágio que pega globals
   comentadas pelo modo `keep:necessary` (ver `_converted_code`).
3. `subprocesso`: roda o script convertido como arquivo .py de verdade, com o `dataset_n_jobs: -1`
   REAL da config. Único estágio que exercita o `train()` paralelo do jeito que a nuvem exercita:
   loky/spawn precisa reserializar o pipeline inteiro a partir do `__main__` achatado, e objetos que
   se serializam por referência (ex.: `functools.lru_cache`) quebram só aqui.

Os três comparam scores bit-a-bit contra o pacote real sobre as MESMAS séries. Qualquer colisão de
nome silenciosa apareceria como divergência.

Isolamento: a namespace do notebook começa sem `sbrt` importável (removemos entradas do repo do
sys.path durante o exec), garantindo que os scores do notebook venham do código inline, não do pacote.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def _pipeline_code(nb_path: Path) -> str:
    """Concatena SÓ as células do pipeline: a do YAML embutido e as dos módulos (cujo cabeçalho
    `# === sbrt/...` vem logo após o marcador `# @crunch/keep:on`). Ignora pip/crunch/test — sem
    filtrar por substring de docstring."""
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        head = "\n".join(src.lstrip().split("\n", 2)[:2])
        is_pipeline = "# =" in head and "sbrt/" in head
        is_yaml = "_EMBEDDED_YAML =" in src
        if is_pipeline or is_yaml:
            chunks.append(src)
    return "\n\n".join(chunks)


def _converted_code(nb_path: Path) -> str:
    """O script que o Crunch REALMENTE executa: o notebook passado pelo conversor oficial
    (`crunch_convert`, o mesmo do `crunch push`/nuvem). Em modo `keep:necessary` ele COMENTA toda
    atribuição de nível de módulo que não esteja sob `# @crunch/keep:on` — foi assim que a config
    embutida inteira (`_EMBEDDED_YAML`) e constantes como `_NAN` sumiam do submission enquanto o
    notebook rodava perfeito no Jupyter. Aqui falhamos alto se qualquer global for descartada."""
    from crunch_convert.notebook import extract_from_file

    flat = extract_from_file(str(nb_path))
    dropped = [str(w) for w in flat.warnings if "global variable" in str(w)]
    if dropped:
        head = "\n".join(f"  {d}" for d in dropped[:10])
        more = f"\n  ... (+{len(dropped) - 10})" if len(dropped) > 10 else ""
        raise AssertionError(
            f"o conversor do Crunch descartaria {len(dropped)} global(is) do pipeline:\n{head}{more}\n"
            "-> a célula precisa começar com `# @crunch/keep:on` (ver build_submission_notebook.py)"
        )
    return flat.source_code


def _make_series(seed: int = 0, n: int = 12):
    rng = np.random.RandomState(seed)
    series = []
    for i in range(n):
        hist = rng.randn(500).tolist()
        online = rng.randn(160)
        tau = None
        if i % 2 == 0:
            tau = int(rng.randint(20, 120))
            kind = i % 6
            if kind == 0:
                online[tau:] *= 2.0            # quebra de variância
            elif kind == 2:
                online[tau:] += 1.5            # quebra de média
            else:
                online[tau:] = np.array([0.6 * online[tau + j - 1] + online[tau + j]
                                         for j in range(len(online) - tau)])  # dependência
        series.append((i, hist, online.tolist(), tau))
    return series


def _run(train_fn, infer_fn, series, model_dir):
    train_fn(series, model_dir)
    test = [(h, o) for _, h, o, _ in series]
    gen = infer_fn(test, model_dir)
    assert next(gen) is None, "1o yield deve ser None"
    scores = []
    for _, o in test:
        for _ in o:
            scores.append(float(next(gen)))
    try:
        next(gen)
        raise AssertionError("generator deveria exaurir")
    except StopIteration:
        pass
    return np.array(scores)


def _run_isolated(code: str, series):
    """Executa `code` numa namespace sem o pacote `sbrt` importável e roda train()+infer().

    n_jobs=1 só no teste: exec num módulo em memória impede o loky de reimportar as funções nos
    workers. Os valores das features são idênticos com/sem paralelismo (cada série é independente e
    determinística; a ordem de concatenação é preservada), então a equivalência bit-a-bit continua
    sendo um teste válido -- e o resultado confirma essa independência de n_jobs."""
    code = code.replace("dataset_n_jobs: -1", "dataset_n_jobs: 1")
    saved_path = list(sys.path)
    saved_modules = {k: v for k, v in sys.modules.items() if k == "sbrt" or k.startswith("sbrt.")}
    for k in saved_modules:
        del sys.modules[k]
    sys.path = [p for p in sys.path if os.path.abspath(p) != os.path.abspath(str(ROOT / "src"))]
    nb_mod = types.ModuleType("nb_pipeline")  # módulo real p/ as @dataclass resolverem via sys.modules
    sys.modules["nb_pipeline"] = nb_mod
    try:
        exec(compile(code, "<notebook-pipeline>", "exec"), nb_mod.__dict__)
        assert "sbrt" not in sys.modules, "notebook não pode depender do pacote sbrt"
        with tempfile.TemporaryDirectory() as md:
            return _run(nb_mod.train, nb_mod.infer, series, md)
    finally:
        sys.modules.pop("nb_pipeline", None)
        sys.path[:] = saved_path
        sys.modules.update(saved_modules)


_SUBPROCESS_HARNESS = '''

if __name__ == "__main__":
    import json as _json, sys as _sys
    import numpy as _np

    _series_path, _model_dir, _out_path = _sys.argv[1:4]
    _series = [(int(i), h, o, tau) for i, h, o, tau in _json.loads(open(_series_path).read())]
    train(_series, _model_dir)
    _gen = infer([(h, o) for _, h, o, _ in _series], _model_dir)
    assert next(_gen) is None, "1o yield deve ser None"
    _np.save(_out_path, _np.array([float(s) for s in _gen]))
'''


def _run_subprocess(code: str, series, tmp: Path):
    """Roda o script convertido como .py de verdade, com o `dataset_n_jobs` REAL (paralelo).

    Precisa ser um arquivo em processo próprio: o loky (backend padrão do joblib, spawn) reimporta o
    `__main__` do worker, então só aqui aparecem os objetos que o cloudpickle serializa por
    REFERÊNCIA e que somem no worker -- classe de bug que nem as células nem o exec em memória pegam.
    PYTHONPATH é limpo para o script não conseguir importar o pacote `sbrt` do repo."""
    script = tmp / "submission_flat.py"
    script.write_text(code + _SUBPROCESS_HARNESS, encoding="utf-8")
    series_path, out_path, model_dir = tmp / "series.json", tmp / "scores.npy", tmp / "model"
    model_dir.mkdir(exist_ok=True)
    series_path.write_text(json.dumps(series), encoding="utf-8")

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUTF8"] = "1"
    proc = subprocess.run(
        [sys.executable, str(script), str(series_path), str(model_dir), str(out_path)],
        cwd=str(tmp), env=env, capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().split("\n")[-25:])
        raise AssertionError(f"o script convertido falhou como .py (train paralelo real):\n{tail}")
    return np.load(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=str(ROOT / "submission_notebook.ipynb"))
    args = parser.parse_args()

    series = _make_series()

    # --- scores do PACOTE REAL ---
    sys.path.insert(0, str(ROOT / "src"))
    from sbrt.adapter import platform as real_platform  # noqa: E402
    with tempfile.TemporaryDirectory() as md:
        real_scores = _run(real_platform.train, real_platform.infer, series, md)
    print(f"pacote real: {len(real_scores)} scores")

    # --- scores dos TRÊS jeitos de rodar o notebook (ver docstring do módulo) ---
    nb_path = Path(args.notebook)
    converted = _converted_code(nb_path)
    with tempfile.TemporaryDirectory() as tmp:
        stages = (
            ("células", lambda: _run_isolated(_pipeline_code(nb_path), series)),
            ("convertido", lambda: _run_isolated(converted, series)),
            ("subprocesso", lambda: _run_subprocess(converted, series, Path(tmp))),
        )
        failed = _compare_stages(stages, real_scores)

    if failed:
        sys.exit(1)
    print("\n=== EQUIVALÊNCIA BIT-A-BIT (células, script convertido E subprocesso paralelo): VALIDADO ===")


def _compare_stages(stages, real_scores) -> bool:
    failed = False
    for label, run in stages:
        nb_scores = run()
        print(f"notebook ({label}): {len(nb_scores)} scores")

        assert nb_scores.shape == real_scores.shape, f"[{label}] contagem de scores difere"
        max_diff = float(np.max(np.abs(nb_scores - real_scores)))
        print(f"  max |score_notebook - score_pacote| = {max_diff:.2e}")
        if max_diff >= 1e-12:
            idx = int(np.argmax(np.abs(nb_scores - real_scores)))
            print(f"  !!! DIVERGÊNCIA no score {idx}: notebook={nb_scores[idx]} pacote={real_scores[idx]}")
            failed = True

    return failed


if __name__ == "__main__":
    main()
