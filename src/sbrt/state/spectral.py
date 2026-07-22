"""SpectralBlock — a *forma* do espectro de `e` mudou? (eixo novo, docs/BACKLOG_TSAUC.md)

## Por que este eixo não está coberto

O banco mede dependência de três formas, e todas são de **domínio do tempo**: ρ₁ em janelas
(`accum_*_rho1_fz`, `dep_absrho1/sqrho1`), massa multi-lag Σρ_k² (`dep_mass_*`, `mismatch_white_e_*`)
e energia por escala diádica (`haar_energy_ln_s*`). O Haar é o mais próximo de espectral, mas ele
mede **nível de energia por banda** — e nível de energia é exatamente o canal que a variância já
domina (o xs-SHAP põe 34,9% em `meta_h0` e 14,7% em acumuladores; a família de variância é o modelo).

O que ninguém mede é a **distribuição relativa** da energia entre frequências, normalizada. As duas
estatísticas abaixo são **invariantes a escala por construção** (dividem pela potência total):

- **centroide espectral** — o "centro de massa" em frequência. Sob H0 o `e` é branco (o AR(10) do
  histórico o branqueou), então o espectro é chato e o centroide fica em ½. Uma quebra que introduz
  persistência positiva empurra massa para baixa frequência (centroide cai); uma quebra que introduz
  alternância/overdifferencing empurra para alta (centroide sobe). Note o **sinal**: `dep_mass` e
  `mismatch_white` são somas de quadrados — cegas à direção. O centroide não é.
- **entropia espectral** — quão chato é o espectro. Sob H0 é máxima (=1 normalizada). Qualquer
  estrutura serial, em qualquer lag, a reduz. É a versão "sem escolher lag" da massa multi-lag.

Como as duas são razões, uma quebra **pura de variância** (o alternativo mais comum e o que o banco
já cobre bem) as deixa **exatamente inalteradas**. Isto é a definição de eixo ortogonal: elas só
podem contribuir onde o modelo hoje é cego — quebras de dependência, cuja detectabilidade medida é
0,492, *abaixo do acaso* (docs/INVESTIGACAO_FALHAS_V3.md §1).

## Implementação: DFT recursiva + média de Welch, O(K) por passo

Não há FFT nem janela materializada. Para cada frequência ω_k mantém-se um único acumulador complexo

    z_k(t) = e_t + d·e^{-iω_k}·z_k(t-1)   =>   z_k(t) = Σ_{j>=0} d^j e^{-iω_k j} e_{t-j}

que é a transformada de Fourier de tempo curto com janela exponencial (comprimento efetivo
~1/(1-d)) — dois floats por bin, uma multiplicação complexa por passo.

**`|z_k|²` cru não serve.** O periodograma num único ponto do tempo é exponencialmente distribuído:
desvio-padrão igual à média, por mais longa que seja a série. Medido: com K=6 as proporções
`p_k` viram um Dirichlet(1,...,1) e a entropia normalizada de H0 cai para ~0,81 **com dispersão de
±0,10 entre séries i.i.d.** — ruído maior que o efeito de um AR(1) com φ=0,6. Uma primeira versão
deste bloco fazia exatamente isso e o teste de sinal reprovou (a série AR ficou ACIMA do branco).

A correção é a média de Welch: uma segunda EWMA sobre a potência, com taxa `alpha` bem mais lenta
que a janela da DFT, de modo a promediar ~1/(alpha·(1-d)) periodogramas quase independentes. Custo:
mais K multiplicações-acumulações por passo.

Duas taxas de promediação: a rápida responde em ~10 periodogramas, a lenta em ~33. A **diferença**
entre as duas é a feature de reorganização propriamente dita — análogo espectral de um contraste
recente-vs-defasado, sem nenhum nulo estimado por série (a lição do F1: constante estimada por série
= ruído transversal puro).

Consome `e` (escala congelada), não `e_vol`: o ajuste de volatilidade é um reescalonamento
*variável no tempo*, ou seja, uma modulação — ele **altera o espectro** que este bloco mede. Mesmo
argumento de `state/mismatch.py`: a pergunta é se o filtro congelado do histórico ainda vale.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sbrt.config import Config
    from sbrt.state.h0 import H0Params

_NAN = math.nan


class _RecursiveSpectrum:
    """Banco de K bins de Fourier de tempo curto + duas médias de Welch. O(K) por passo, sem alocação."""

    __slots__ = ("K", "re", "im", "cos_w", "sin_w", "omega", "n_low", "n",
                 "p_fast", "p_slow", "a_fast", "a_slow", "min_fast", "min_slow")

    def __init__(self, n_bins: int, decay: float, alpha_fast: float, alpha_slow: float, low_bins: int):
        self.K = n_bins
        # ω_k no centro de K bandas iguais de (0, π) — nenhum bin cai em 0 (onde o AR congelado já
        # removeu a média) nem em π exatamente.
        self.omega = [math.pi * (k + 0.5) / n_bins for k in range(n_bins)]
        self.cos_w = [decay * math.cos(w) for w in self.omega]
        self.sin_w = [-decay * math.sin(w) for w in self.omega]  # e^{-iω} => imaginário negativo
        self.re = [0.0] * n_bins
        self.im = [0.0] * n_bins
        self.p_fast = [0.0] * n_bins
        self.p_slow = [0.0] * n_bins
        self.a_fast = alpha_fast
        self.a_slow = alpha_slow
        self.n_low = low_bins
        # aquecimento honesto: a DFT precisa encher sua janela E a média de Welch precisa promediar.
        # Antes disso os acumuladores ainda carregam o zero inicial.
        w_dft = 1.0 / (1.0 - decay)
        self.min_fast = int(math.ceil(w_dft + 1.0 / alpha_fast))
        self.min_slow = int(math.ceil(w_dft + 1.0 / alpha_slow))
        self.n = 0

    def update(self, x: float) -> None:
        self.n += 1
        re, im, c, s = self.re, self.im, self.cos_w, self.sin_w
        pf, ps, af, as_ = self.p_fast, self.p_slow, self.a_fast, self.a_slow
        for k in range(self.K):
            r, i = re[k], im[k]
            nr = x + r * c[k] - i * s[k]
            ni = r * s[k] + i * c[k]
            re[k] = nr
            im[k] = ni
            p = nr * nr + ni * ni
            pf[k] += af * (p - pf[k])
            ps[k] += as_ * (p - ps[k])

    def _stats(self, p: list, ready: bool) -> tuple[float, float, float]:
        """(centroide/π, entropia normalizada, fração de potência nos bins baixos)."""
        if not ready:
            return _NAN, _NAN, _NAN
        tot = 0.0
        for v in p:
            tot += v
        if tot <= 1e-300:
            return _NAN, _NAN, _NAN
        cent = 0.0
        ent = 0.0
        low = 0.0
        for k in range(self.K):
            pk = p[k] / tot
            cent += pk * self.omega[k]
            if pk > 1e-300:
                ent -= pk * math.log(pk)
            if k < self.n_low:
                low += pk
        return cent / math.pi, ent / math.log(self.K), low

    def fast_stats(self) -> tuple[float, float, float]:
        return self._stats(self.p_fast, self.n >= self.min_fast)

    def slow_stats(self) -> tuple[float, float, float]:
        return self._stats(self.p_slow, self.n >= self.min_slow)


class SpectralBlock:
    def reset(self, h0: "H0Params | None", cfg: "Config") -> None:
        sc = cfg.spectral
        self.warmup = cfg.features.warmup_min_n
        self.spec = _RecursiveSpectrum(sc.n_bins, sc.decay, sc.alpha_fast, sc.alpha_slow, sc.low_bins)
        self.t = 0

    def update(self, e: float, e_raw: float, e_vol: float, t: int) -> None:
        self.t = t
        self.spec.update(e)

    def features(self) -> dict[str, float]:
        if self.t < self.warmup:
            cf = ef = lf = cs = es = ls = _NAN
        else:
            cf, ef, lf = self.spec.fast_stats()
            cs, es, ls = self.spec.slow_stats()
        return {
            "spec_centroid_fast": cf,
            "spec_centroid_slow": cs,
            "spec_entropy_fast": ef,
            "spec_entropy_slow": es,
            "spec_lowratio_fast": lf,
            "spec_lowratio_slow": ls,
            # reorganização: o espectro recente contra o espectro de médio prazo da MESMA série.
            # Contraste interno, sem constante estimada — não injeta ruído transversal.
            "spec_dcentroid": cf - cs if not (math.isnan(cf) or math.isnan(cs)) else _NAN,
            "spec_dentropy": ef - es if not (math.isnan(ef) or math.isnan(es)) else _NAN,
        }
