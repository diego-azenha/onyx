"""Decisão sobre monotonicidade do score (plano §7). Default = V-livre (identidade): o posterior
P(tau<=t|dados) é não-monótono por natureza (evidência transitória deve decair) e o max-hold trava
alarmes falsos nas séries sem quebra (contraexemplo CE1, plano §12.5). Variantes com retenção só
seriam adotadas mediante confirmação por submissão oficial — nunca por métrica local (§9)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sbrt.config import Config

Mode = Literal["free", "hold", "soft", "ema"]


def apply(p: float, prev: float | None, mode: Mode, cfg: "Config") -> float:
    """mode='free' (default) = identidade. 'hold'/'soft'/'ema' só habilitados em
    configs/default.yaml se o gate G-mono (plano §9) tiver sido confirmado por submissão oficial."""
    if prev is None or mode == "free":
        return p
    if mode == "hold":
        return max(prev, p)
    if mode == "soft":
        return max(p, prev - cfg.postprocess.soft_decay)
    if mode == "ema":
        alpha = cfg.postprocess.ema_alpha
        return alpha * p + (1.0 - alpha) * prev
    raise ValueError(f"modo de monotonicidade desconhecido: {mode!r}")
