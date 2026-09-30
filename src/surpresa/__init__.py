"""Detector de surpresa acumulada — frente da premissa 1 (knowledge/premissas/01-surpresa-acumulada.md).

Pacote separado do `sbrt` de propósito: é construído e medido AO LADO do Onyx, sem tocá-lo, até que
o experimento S3 (knowledge/frentes/surpresa-acumulada/README.md) justifique ligá-lo como StateBlock.
"""
from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PADRAO = Path(__file__).resolve().parents[2] / "configs" / "surpresa.yaml"


def carregar_config(caminho: str | Path | None = None) -> dict:
    with open(caminho or CONFIG_PADRAO, encoding="utf-8") as f:
        return yaml.safe_load(f)
