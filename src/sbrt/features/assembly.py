"""Ordem canônica das features + schema (plano §5). A ordem canônica é `sorted(feats.keys())` —
determinística e estável sem precisar de um "primeiro run" especial; persistida junto do modelo
para que `model/predict.py` monte o vetor na mesma ordem usada em `model/dataset.py` (motor único,
docs/PLANO_REPOSITORIO.md §1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def build_feature_order(feats: dict) -> tuple:
    return tuple(sorted(feats.keys()))


def to_array(feats: dict, order: tuple) -> np.ndarray:
    """Materializa na ordem canônica; chave ausente (warm-up) -> NaN. LightGBM trata NaN
    nativamente — nunca usar sentinela numérica."""
    return np.array([feats.get(k, np.nan) for k in order], dtype=np.float64)


def save_schema(order: tuple, path: str | Path) -> None:
    Path(path).write_text(json.dumps(list(order), indent=2), encoding="utf-8")


def load_schema(path: str | Path) -> tuple:
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")))
