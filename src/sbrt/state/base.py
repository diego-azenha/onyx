"""Contrato comum a todo bloco de state/* (docs/CONTRACTS.md). Um StreamScorer é, na prática,
uma lista de StateBlocks (docs/PLANO_REPOSITORIO.md §1)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params


class StateBlock(Protocol):
    def reset(self, h0: "H0Params", cfg: "Config") -> None:
        """Uma vez por série, logo após fit_h0."""
        ...

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        """Uma vez por passo. e = inovação whitened+clipada (escala congelada, usar para
        variância/cauda); e_vol = inovação whitened+vol-ajustada (plano §3.4, usar para
        média/dependência/forma); t = índice do passo, 1-based."""
        ...

    def features(self) -> dict[str, float]:
        """Features atuais do bloco, nomes estáveis (convenção `<bloco>_<estatistica>_<param>`)."""
        ...
