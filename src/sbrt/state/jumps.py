"""JumpBlock — decomposição salto/contínuo e assimetria de leverage
(docs/INVESTIGACAO_FALHAS_V3.md P4; Barndorff-Nielsen & Shephard 2004).

## Por que existe

Papel de PRECISÃO, não de recall (INVESTIGACAO §4.3): separa duas coisas que o banco confunde e que
são os falsos-positivos nomeados do projeto — cluster GARCH (T6) e outlier isolado (T9) — de uma
quebra genuína de variância.

**Bipower variation.** A variância realizada RV = Σe² capta variância contínua + saltos; a bipower
variation BV = (π/2)·Σ|eₜ||eₜ₋₁| é *robusta a saltos* (o produto de dois vizinhos é grande só se
AMBOS forem grandes, o que um salto isolado não garante). Logo:

- razão de salto (RV−BV)/RV isola a componente descontínua: **alta** num outlier/salto isolado (T9),
  **baixa** num cluster GARCH (volatilidade contínua) ou numa quebra de patamar de variância;
- BV dá um estimador de variância *robusto a saltos* — uma quebra de variância limpa move BV; um
  outlier isolado não. O contraste ln(RV) − ln(BV) é o discriminador.

**Semivariância / leverage.** RS⁺ = Σe²·1{e>0}, RS⁻ = Σe²·1{e<0}: a assimetria (RS⁺−RS⁻)/(RS⁺+RS⁻)
distingue uma quebra que empurra a cauda para um lado. E a correlação sinal-magnitude
corr(sign(eₜ₋₁), |eₜ|) capta o *efeito leverage* (volatilidade responde de forma assimétrica ao
sinal) — um eixo de dependência que cruza sinal×magnitude, ausente do banco (que tem CUSUMs de sinal
e de dependência separados, nunca cruzados).

Consome `e` (escala congelada) — família de variância/cauda, trava CE2 (plano §3.4). Calibrado por
F1: um GARCH tem razão de salto e leverage característicos no seu *próprio histórico*, então a versão
`_cal` só acende no *excesso* pós-quebra — é isso que o torna um discriminador de T6, não mais um
detector que dispara junto com o GARCH.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from sbrt.utils.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan
_MU1 = math.pi / 2.0  # fator de escala da bipower (E[|Z|]² para Z~N(0,1) é 2/pi -> inverso)


def _d0(ev: float | None) -> float:
    return ev if ev is not None else 0.0


class _RSum:
    __slots__ = ("ring", "total")

    def __init__(self, w: int):
        self.ring = RingBuffer(w)
        self.total = 0.0

    def push(self, v: float) -> None:
        ev = self.ring.push(v)
        self.total += v - _d0(ev)

    def __len__(self) -> int:
        return len(self.ring)


class JumpBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        jc = cfg.jumps
        self.warmup = cfg.features.warmup_min_n
        self.windows = list(jc.windows)
        self.t = 0
        self.prev_abs: float | None = None
        self.prev_sign: float | None = None
        # por janela: RV=Σe², BV=Σ|e||e_prev|, RS+=Σe²1{e>0}, RS-, LEV=Σ sign(e_prev)|e|
        self.rv = {w: _RSum(w) for w in self.windows}
        self.bv = {w: _RSum(w) for w in self.windows}
        self.rsp = {w: _RSum(w) for w in self.windows}
        self.rsm = {w: _RSum(w) for w in self.windows}
        self.lev = {w: _RSum(w) for w in self.windows}
        self.abs_sum = {w: _RSum(w) for w in self.windows}  # Σ|e| para a média usada no leverage

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        ae = abs(e)
        se = e * e
        sgn = 1.0 if e > 0 else (-1.0 if e < 0 else 0.0)
        bp = ae * self.prev_abs if self.prev_abs is not None else 0.0
        lv = self.prev_sign * ae if self.prev_sign is not None else 0.0
        for w in self.windows:
            self.rv[w].push(se)
            self.bv[w].push(bp)
            self.rsp[w].push(se if e > 0 else 0.0)
            self.rsm[w].push(se if e < 0 else 0.0)
            self.lev[w].push(lv)
            self.abs_sum[w].push(ae)
        self.prev_abs = ae
        self.prev_sign = sgn

    def features(self) -> dict[str, float]:
        warm = self.t >= self.warmup
        out: dict[str, float] = {}
        for w in self.windows:
            if not warm or len(self.rv[w]) < 3:
                out[f"jump_ratio_w{w:03d}"] = _NAN
                out[f"jump_rvbv_ln_w{w:03d}"] = _NAN
                out[f"jump_semivar_asym_w{w:03d}"] = _NAN
                out[f"jump_leverage_w{w:03d}"] = _NAN
                continue
            n = len(self.rv[w])
            rv = self.rv[w].total / n
            n_bv = len(self.bv[w])
            bv = _MU1 * self.bv[w].total / max(n_bv, 1)
            out[f"jump_ratio_w{w:03d}"] = max(rv - bv, 0.0) / (rv + 1e-12)
            out[f"jump_rvbv_ln_w{w:03d}"] = math.log(max(rv, 1e-12)) - math.log(max(bv, 1e-12))
            rsp = self.rsp[w].total
            rsm = self.rsm[w].total
            out[f"jump_semivar_asym_w{w:03d}"] = (rsp - rsm) / (rsp + rsm + 1e-12)
            out[f"jump_leverage_w{w:03d}"] = self.lev[w].total / max(n_bv, 1)
        return out


def history_null_series(e_hist, cfg) -> dict:
    """Roda o próprio JumpBlock sobre o histórico (H0), para a calibração F1."""
    blk = JumpBlock()
    blk.reset(None, cfg)
    acc: dict[str, list] = {}
    for i, ev in enumerate(e_hist, start=1):
        blk.update(float(ev), float(ev), float(ev), i)
        for name, val in blk.features().items():
            acc.setdefault(name, []).append(val)
    return acc
