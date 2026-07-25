"""Decisão sobre monotonicidade do score (plano §7). Default = V-livre (identidade): o posterior
P(tau<=t|dados) é não-monótono por natureza (evidência transitória deve decair) e o max-hold trava
alarmes falsos nas séries sem quebra (contraexemplo CE1, plano §12.5). Variantes com retenção só
seriam adotadas mediante confirmação por submissão oficial — nunca por métrica local (§9)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from sbrt.config import Config

Mode = Literal["free", "hold", "soft", "ema", "ema_asym"]


def ema_asym_step(p: float, prev: float, alpha_up: float, alpha_down: float) -> float:
    """A1 (BRAINSTORM_RUPTURA_V2.md §2.2): EWMA ASSIMÉTRICO -- α rápido na subida, lento na descida.
    O jitter do score é ruído transversal puro nos passos seguintes; a assimetria respeita a semântica
    'P(quebra já ocorreu)' (a evidência sobe rápido e decai devagar) sem o max-hold rígido que o CE1
    matou. `alpha_up` >= `alpha_down`."""
    alpha = alpha_up if p >= prev else alpha_down
    return alpha * p + (1.0 - alpha) * prev


def apply(p: float, prev: float | None, mode: Mode, cfg: "Config", t: int | None = None) -> float:
    """mode='free' (default) = identidade. 'hold'/'soft'/'ema'/'ema_asym' só habilitados em
    configs/default.yaml se o gate G-mono (plano §9) tiver sido confirmado por submissão oficial.

    `t` (B3): passo atual (1-based), usado só pelo gate `cfg.postprocess.min_t`. Opcional para não
    quebrar chamadores antigos -- sem ele o gate é inerte e o comportamento é o de antes."""
    if prev is None or mode == "free":
        return p
    if mode == "hold":
        return max(prev, p)
    if mode == "soft":
        return max(p, prev - cfg.postprocess.soft_decay)
    if mode == "ema":
        alpha = cfg.postprocess.ema_alpha
        return alpha * p + (1.0 - alpha) * prev
    if mode == "ema_asym":
        # B3: gate de regime, DELIBERADAMENTE restrito a este modo -- é o único em que ele foi
        # medido. Aplicá-lo a 'hold'/'soft'/'ema' mudaria a semântica de modos que nunca foram
        # avaliados com gate e quebraria o contraexemplo CE1 (plano §12.5: o V-hold TEM de travar
        # desde o primeiro passo). 'hold' também não deve depender de `cfg.postprocess`.
        # Duas condições, e a segunda é a sutil:
        #   (a) t <= min_t          -> abaixo do corte, identidade.
        #   (b) t - 1 <= min_t < t  -> PRIMEIRA observação acima do corte: a recursão RECOMEÇA aqui,
        #       i.e. trata-se `prev` como inexistente. Isso reproduz `apply_vema_oof.py --min-t`, que
        #       fatia o OOF em `t > min_t` e roda o EWMA do zero dentro da fatia (por série). Sem (b)
        #       a produção herdaria o `prev` do regime frio e NÃO reproduziria o +0,0009 medido.
        #       `tests/unit/test_b3_vema_gate.py` prova a equivalência bit-a-bit.
        min_t = cfg.postprocess.min_t
        if min_t is not None and t is not None and (t <= min_t or t - 1 <= min_t):
            return p
        return ema_asym_step(p, prev, cfg.postprocess.ema_up_alpha, cfg.postprocess.ema_down_alpha)
    raise ValueError(f"modo de monotonicidade desconhecido: {mode!r}")
