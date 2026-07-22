"""Calibração de nulo por série (docs/PROPOSTA_FEATURES_V2.md F1) — o item de maior leverage da
proposta V2, e o único motivado por uma medição direta do modelo atual em vez de analogia externa.

## O problema que isto resolve

A TS-AUC ordena **séries diferentes no mesmo passo**. Uma estatística cujo nível sob H0 depende das
idiossincrasias da série (curtose, dependência, clustering de volatilidade) está intrinsecamente
mal-ordenada nessa seção transversal: um `ranktwo_wilcoxon_z` de 2,5 significa coisas diferentes numa
série i.i.d. e numa série com forte clustering, porque o z é normalizado por √(12·w) — uma fórmula
que **assume independência**. Com p-values consecutivos correlacionados, a variância verdadeira da
média de janela é muito maior, e o z fica sistematicamente inflado justamente nas séries mais
dependentes.

Hoje o modelo corrige isso sozinho, aprendendo interações `meta_h0_* × estatística` — e é por isso
que as `meta_h0_*` consomem **34,3% do |SHAP|** apesar de o CE6 mostrar que não carregam efeito
principal (AUC 0,5067). Um terço do orçamento do modelo é gasto reconstruindo uma calibração que
podemos simplesmente calcular.

## A ideia

O histórico **é H0 por definição** (livre de quebra, plano §3). Então basta deslizar a MESMA
estatística sobre o histórico da própria série para obter a distribuição nula dela *naquela série*,
e emitir, além do valor cru, o desvio padronizado contra esse nulo:

    S_cal(t) = (S(t) − μ_nulo) / σ_nulo

Custo: O(n_h) uma vez por série dentro do `fit_h0` (que já é O(n_h log n_h)); **zero µs por passo**
além de uma subtração e uma divisão.

## Decisões de implementação (e por quê)

- **Estatísticas baseadas em `e_vol` exigem `history_evol`.** Até o V4 a calibração cobria só o que
  é baseado em `e` (variância/cauda/rank), porque reproduzir `e_vol` pedia replicar a EWMA de
  volatilidade sobre o histórico. Para o canal de **média** esse corte segue valendo por mérito: o
  censo A1 mostra que ele é quase morto (6,8% das séries com |Δmean_e|>0,3). Mas ele também excluía,
  sem querer, o canal de **dependência** (`cusum_dep_*`, `accum_*_rho1_fz`, `dep_mass_evol_*`) — que
  não é morto por natureza, é morto por miscalibração (0,492, abaixo do acaso). `history_evol`
  remove o bloqueio; o que entra ou não vira decisão medida, não consequência acidental.
- **`_cal` só quando a janela está cheia** (`t >= min_t`). Para t < w o estatístico online usa
  n_eff = t, cuja distribuição nula é outra (σ de `ln E[e²]` escala com √(2/n_eff)); calibrar com o
  nulo de janela cheia daria um número errado. NaN é tratado nativamente pelo LightGBM e é a resposta
  honesta: ainda não há janela suficiente.
- **Encolhimento para o nulo teórico** onde ele é conhecido. Com janela w e histórico n_h há apenas
  ~n_h/w janelas *independentes* (4 a 20 para w=250), então σ empírico é ruidoso. Encolhemos para o
  σ teórico i.i.d. com peso n_eff/(n_eff+pseudo). Onde não há teoria (MMD, Haar), usa-se o empírico
  puro — lá o número de amostras efetivas é alto porque a escala de tempo do estatístico (1/λ) é
  muito menor que n_h.
- **Vetorização.** O cálculo sobre o histórico é vetorizado, enquanto o online é recursivo. Isto NÃO
  é a armadilha §13.2 do plano ("backtest vetorizado ≠ execução causal"): não produz features de
  treino nem scores, produz uma *constante por série* a partir de dados que já são H0. Ainda assim a
  equivalência é verificada por testes dedicados (`tests/unit/test_calibration.py`,
  `test_mmd.py`, `test_multiscale.py`), porque um desalinhamento aqui envenenaria silenciosamente
  todas as features calibradas.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from sbrt.state import accumulators as accum_mod
from sbrt.state import conformal as conf_mod
from sbrt.state import cusum as cusum_mod
from sbrt.state import lmoments as lmom_mod
from sbrt.state import dependence as dep_mod
from sbrt.state import jumps as jump_mod
from sbrt.state import mmd as mmd_mod
from sbrt.state import multiscale as ms_mod
from sbrt.state import varloc as varloc_mod
from sbrt.utils.numerics import vol_adjust_step

# P(|Z| > 2) para Z ~ N(0,1) — taxa nominal de excedência usada por `accum_window_exceed2_frac_*`.
_P0_EXCEED2 = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(2.0 / math.sqrt(2.0))))

# Piso de amostras para aplicar a extrapolação de escala: abaixo disto as aproximações assintóticas
# do nulo teórico (ex.: dp de ln(chi²_n/n) ≈ sqrt(2/n)) são ruins demais para valer a pena.
_MIN_N_FOR_SCALING = 10


class NullSpec(NamedTuple):
    """Nulo de uma estatística, medido na JANELA CHEIA, mais o necessário para transportá-lo a
    janelas parciais (t < w).

    `kind` diz como o nulo escala com o número efetivo de amostras n = min(t, window):
    - `"z"`      : já normalizado por sqrt(n) na origem (ranktwo) -> nulo não depende de n;
    - `"var_ln"` : ln(média de e²) -> mu_teo(n) = −1/n, dp_teo(n) = sqrt(2/n);
    - `"frac"`   : fração de excedência -> mu_teo(n) = p0, dp_teo(n) = sqrt(p0(1−p0)/n);
    - `"rho"`    : autocorrelação -> mu ~ const, dp_teo(n) ∝ 1/sqrt(n) (P1, dependência);
    - `"none"`   : sem lei de escala conhecida (MMD, Haar, massa multi-lag) -> só vale na janela cheia.

    `table` (F1.a) resolve o caso que nenhum `kind` acima cobre: estatísticas **recursivas** que
    partem de um estado inicial fixo (CUSUMs partem de 0; a autocorrelação global parte de uma janela
    expansiva). O nulo delas não é estacionário em t — é uma curva de transiente. `table` é
    `(mu_por_t, sd_por_t)` medida por réplicas com reinício sobre o histórico, usada para
    `t <= len(table)` e substituída por `(mu, sd)` depois. Sem ela a única saída honesta seria
    `min_t` = fim do transiente (~75 passos, medido), deixando a feature 100% NaN justo no bucket
    `t<=50` — a diluição de sorteio que docs/NOTAS_AGENTES.md §7 registra como pegadinha cara.

    A ideia do transporte: o que a série tem de idiossincrático é o *fator de inflação* em relação ao
    nulo i.i.d. (k = dp_medido / dp_teórico), não o nível absoluto. Esse fator é aproximadamente
    constante em n para uma série estacionária, então podemos aplicá-lo ao dp teórico de qualquer n.
    Isso libera a versão calibrada muito antes de a janela encher — exatamente no regime de t pequeno
    onde o modelo é mais fraco e onde antes essas colunas eram 100% NaN."""

    mu: float
    sd: float
    min_t: int
    kind: str = "none"
    window: int = 0
    aux: float = 0.0  # p0, para kind="frac"
    table: tuple = ()  # (mu_por_t, sd_por_t) do transiente; vazio = sem transiente medido


def history_evol(e_hist: np.ndarray, rho1_abs_e: float, cfg) -> np.ndarray:
    """Reproduz a série `e_vol` do laço online sobre o histórico (F1.0, docs/BACKLOG_TSAUC.md).

    Sem isto, a calibração de nulo só alcança estatísticas baseadas em `e` — o que deixou de fora
    justamente as features de dependência (`cusum_dep_*`, `accum_*_rho1_fz`, `dep_mass_evol_*`), que
    são `e_vol`-based e estão medidas como mortas (eixo de dependência em 0,492, abaixo do acaso).

    Recebe `rho1_abs_e` solto em vez de `H0Params` por dois motivos: `h0.py` importa este módulo (o
    inverso seria circular), e `compute_null_stats` roda DENTRO de `fit_h0`, antes de o `H0Params`
    existir.

    O laço é sequencial de propósito. Uma EWMA vetorizada em numpy não é bit-a-bit igual à recursão
    em float64, e a igualdade exata com o online é o requisito inteiro desta função. Custo: uma
    passada Python sobre o histórico, uma vez por série, dentro de um `fit_h0` que já custa 32–68 ms.
    """
    e = np.asarray(e_hist, dtype=np.float64)
    if rho1_abs_e <= cfg.state.vol_adjust["threshold_rho1_abs"]:
        return e.copy()  # ajuste desligado nesta série: e_vol É e (mesmo galho de scorer.py)

    lam = cfg.state.vol_adjust["lambda_v"]
    v = 1.0  # mesma semente do StreamScorer; e_hist já é padronizado por sigma_e, então v0=1 é a escala natural
    out = np.empty(len(e), dtype=np.float64)
    for i in range(len(e)):
        v, out[i] = vol_adjust_step(v, float(e[i]), lam)
    return out


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Média móvel de janela cheia (comprimento len(x)-w+1). Equivale ao que o bloco online emite
    quando o ring está cheio."""
    if len(x) < w or w < 1:
        return np.empty(0, dtype=np.float64)
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64))])
    return (c[w:] - c[:-w]) / w


def _upper_tail_p_vec(sorted_arr: np.ndarray, x: np.ndarray, n: int) -> np.ndarray:
    """Versão vetorizada de `conformal._upper_tail_p` (mid-rank, cauda superior)."""
    lo = np.searchsorted(sorted_arr, x, side="left")
    hi = np.searchsorted(sorted_arr, x, side="right")
    mid_rank = (lo + hi) / 2.0
    return (n - mid_rank + 0.5) / (n + 1.0)


def _add(
    out: dict,
    name: str,
    arr: np.ndarray,
    min_t: int,
    n_eff: float,
    theory: tuple | None,
    pseudo: float,
    kind: str = "none",
    window: int = 0,
    aux: float = 0.0,
) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 8:
        return
    mu_emp = float(arr.mean())
    sd_emp = float(arr.std(ddof=1))

    if theory is not None and pseudo > 0.0:
        mu_th, sd_th = theory
        wgt = n_eff / (n_eff + pseudo)
        mu = wgt * mu_emp + (1.0 - wgt) * mu_th
        sd = math.sqrt(max(wgt * sd_emp ** 2 + (1.0 - wgt) * sd_th ** 2, 1e-12))
    else:
        mu, sd = mu_emp, sd_emp

    if not (np.isfinite(mu) and np.isfinite(sd)):
        return
    # Com lei de escala conhecida a calibração vale desde cedo (ver NullSpec); sem ela, só na
    # janela cheia.
    effective_min_t = min(int(min_t), _MIN_N_FOR_SCALING) if kind != "none" else int(min_t)
    out[name] = NullSpec(mu, max(sd, 1e-6), effective_min_t, kind, int(window), float(aux))


def _smooth(a: np.ndarray, w: int) -> np.ndarray:
    """Média móvel centrada sobre o eixo do transiente. A curva verdadeira é suave em t por
    construção, então suavizar troca viés desprezível por menos ruído de estimação — e o ruído aqui
    entra como ruído POR SÉRIE dentro do passo, exatamente o que degrada o ranking transversal."""
    if w <= 1 or len(a) < w:
        return a
    pad = w // 2
    padded = np.concatenate([np.full(pad, a[0]), a, np.full(w - 1 - pad, a[-1])])
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def _add_from_replicates(
    out: dict, name: str, mat: np.ndarray, pseudo: float, smooth_w: int, kind: str = "none"
) -> None:
    """Registra o nulo de uma estatística RECURSIVA a partir de réplicas com reinício.

    `mat` tem forma (n_reps, K): a linha r é uma execução do bloco começando do zero sobre um trecho
    virgem do histórico, e a coluna j é o valor no passo online-equivalente j+1. Isso reproduz
    exatamente o que o online faz (todo bloco é resetado em `StreamScorer.__init__`), então a coluna j
    É a distribuição nula da feature no passo j+1.

    O transiente vira `NullSpec.table` (t <= K). Para t > K vale `kind`: `none` extrapola o nulo da
    última coluna como estacionário; `cumsum` guarda a deriva por passo e a escala por sqrt(passo),
    porque um acumulador sem reset nunca estaciona.

    O dp de cada coluna é encolhido para a referência com peso n_reps/(n_reps+pseudo) — mesma mecânica
    de `_add`, e o viés é para o lado seguro (nulo mais largo => calibrada menos agressiva)."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] < 4 or mat.shape[1] < 2:
        return
    # média/dp por coluna ignorando NaN, sem `nanmean`/`nanstd`: colunas inteiramente NaN (a feature
    # ainda em warm-up) fariam aquelas funções emitir RuntimeWarning, e `np.where` não evita isso
    # porque avalia os dois ramos.
    finite = np.isfinite(mat)
    valid = finite.sum(axis=0)
    ok = valid >= 2
    safe = np.where(finite, mat, 0.0)
    mu_j = np.where(ok, safe.sum(axis=0) / np.maximum(valid, 1), np.nan)
    dev = np.where(finite, mat - mu_j, 0.0)
    sd_j = np.where(ok, np.sqrt((dev ** 2).sum(axis=0) / np.maximum(valid - 1, 1)), np.nan)
    if not np.isfinite(mu_j[-1]) or not np.isfinite(sd_j[-1]):
        return

    # colunas ainda em warm-up (feature emite NaN) herdam o primeiro valor válido — apply_calibration
    # já devolve NaN nesses passos porque o CRU é NaN, então o valor aqui nunca chega a ser usado.
    first = int(np.argmax(np.isfinite(mu_j)))
    mu_j = np.where(np.isfinite(mu_j), mu_j, mu_j[first])
    sd_j = np.where(np.isfinite(sd_j), sd_j, sd_j[first])

    K = mat.shape[1]
    if kind == "cumsum":
        # o nulo não estaciona: extrai a deriva por passo e a escala por sqrt(passo) do fim da
        # janela de réplicas, que `_null_at` extrapola para t > K.
        mu_stat, sd_stat = float(mu_j[-1]) / K, float(max(sd_j[-1], 1e-6)) / math.sqrt(K)
    else:
        mu_stat, sd_stat = float(mu_j[-1]), float(max(sd_j[-1], 1e-6))
    # referência do encolhimento na escala de CADA passo: constante para um nulo estacionário,
    # crescendo com sqrt(t) para um acumulador. Encolher a coluna j de um `cumsum` para um escalar
    # seria comparar escalas diferentes e estragaria justamente as colunas mais tardias.
    steps = np.arange(1, mat.shape[1] + 1, dtype=np.float64)
    sd_ref = sd_stat * np.sqrt(steps) if kind == "cumsum" else np.full(mat.shape[1], sd_stat)
    n_reps = float(mat.shape[0])
    wgt = n_reps / (n_reps + pseudo)
    sd_j = np.sqrt(np.maximum(wgt * sd_j ** 2 + (1.0 - wgt) * sd_ref ** 2, 1e-12))

    mu_j = _smooth(mu_j, smooth_w)
    sd_j = np.maximum(_smooth(sd_j, smooth_w), 1e-6)

    out[name] = NullSpec(
        mu_stat, sd_stat, min_t=1, kind=kind,
        table=(tuple(float(v) for v in mu_j), tuple(float(v) for v in sd_j)),
    )


def compute_null_stats(e_hist: np.ndarray, h0, cfg) -> dict:
    """{nome_da_feature: NullSpec}. Chamado uma vez por série no fim de `fit_h0`.

    `h0` é o `H0Params` já montado, porém com `null_stats={}` — é ele que `fit_h0` completa depois
    via `dataclasses.replace`. Receber o objeto inteiro (em vez de meia dúzia de arrays soltos) é o
    que permite rodar os BLOCOS REAIS sobre o histórico: `CusumBlock.reset` e `AccumulatorBlock.reset`
    exigem um `h0` de verdade (sigma_u, quantis, sorted_abs_e_hist)."""
    cal_cfg = cfg.calibration
    if not cal_cfg.enabled:
        return {}

    e = np.asarray(e_hist, dtype=np.float64)
    n_h = len(e)
    if n_h < 64:
        return {}

    sorted_e_hist = h0.sorted_e_hist
    sorted_abs_e_hist = h0.sorted_abs_e_hist
    rff_href, rff_href_joint = h0.rff_href, h0.rff_href_joint

    out: dict = {}
    pseudo = cal_cfg.shrink_pseudo
    e2 = e * e
    exceed2 = (np.abs(e) > 2.0).astype(np.float64)


    # --- accum: variância de janela (ln) e fração de excedência ---
    for w in cfg.state.window_sizes:
        arr = np.log(np.maximum(_rolling_mean(e2, w), 1e-12))
        # teoria i.i.d. gaussiana: w·E[e²] ~ chi²_w  =>  ln(E[e²]) tem média ≈ -1/w e dp ≈ sqrt(2/w)
        _add(out, f"accum_window_var_ln_w{w:03d}", arr, min_t=w, n_eff=n_h / w,
             theory=(-1.0 / w, math.sqrt(2.0 / w)), pseudo=pseudo, kind="var_ln", window=w)

    for w in cfg.state.exceedance_windows:
        arr = _rolling_mean(exceed2, w)
        _add(out, f"accum_window_exceed2_frac_w{w:03d}", arr, min_t=w, n_eff=n_h / w,
             theory=(_P0_EXCEED2, math.sqrt(_P0_EXCEED2 * (1.0 - _P0_EXCEED2) / w)), pseudo=pseudo,
             kind="frac", window=w, aux=_P0_EXCEED2)

    # --- ranktwo (R4): z de Wilcoxon e de dispersão ---
    p_right = _upper_tail_p_vec(sorted_e_hist, e, n_h)
    p_abs = _upper_tail_p_vec(sorted_abs_e_hist, np.abs(e), n_h)
    for w in cfg.rank_twosample.windows:
        scale = math.sqrt(12.0 * w)
        _add(out, f"ranktwo_wilcoxon_z_w{w:03d}", _rolling_mean(p_right - 0.5, w) * scale,
             min_t=w, n_eff=n_h / w, theory=(0.0, 1.0), pseudo=pseudo, kind="z", window=w)
        _add(out, f"ranktwo_dispersion_z_w{w:03d}", _rolling_mean(p_abs - 0.5, w) * scale,
             min_t=w, n_eff=n_h / w, theory=(0.0, 1.0), pseudo=pseudo, kind="z", window=w)

    # --- MMD (F3): sem teoria fechada -> nulo empírico puro, só na janela cheia ---
    mmd_series = mmd_mod.history_series(e, rff_href, rff_href_joint, cfg)
    taus = {
        "_vfast": int(1.0 / max(cfg.mmd.lambda_vfast, 1e-9)),
        "_fast": int(1.0 / max(cfg.mmd.lambda_fast, 1e-9)),
        "_slow": int(1.0 / max(cfg.mmd.lambda_slow, 1e-9)),
    }
    tau_slow = taus["_slow"]
    for name, arr in mmd_series.items():
        min_t = next((v for suf, v in taus.items() if name.endswith(suf)), tau_slow)
        # descarta o transiente inicial da EWMA antes de medir o nulo
        _add(out, name, arr[min_t:], min_t=min_t, n_eff=len(arr), theory=None, pseudo=0.0)

    # --- dependência (P1): roda o próprio DependenceBlock sobre o histórico (garante equivalência
    # online/nulo por construção). ρ₁ de |e|/e² tem kind="rho" (escala 1/sqrt(n), disponível cedo);
    # a massa multi-lag não tem lei fechada -> kind="none" (janela cheia). ---
    # `e_vol_hist=None` mantém o conjunto de features do V4: passar o `e_vol` real destrava
    # `dep_mass_evol_w100_cal`, que entrou no braço F1.a e regrediu por R0 junto com o resto
    # (docs/BACKLOG_TSAUC.md). O replay continua disponível em `history_evol` para quando um braço
    # que precise dele for reaberto.
    dep_series = dep_mod.history_null_series(e, cfg)
    for name, series in dep_series.items():
        w = int(name.rsplit("_w", 1)[1])
        kind = "rho" if "rho1" in name else "none"
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=w, n_eff=n_h / w,
             theory=None, pseudo=0.0, kind=kind, window=w)

    # --- L-momentos (P2): forma de cauda robusta; nulo empírico da própria série (uma série de cauda
    # pesada tem L-kurtosis alta no seu próprio histórico -> a calibrada só acende no excesso). ---
    lmom_series = lmom_mod.history_null_series(e, cfg)
    for name, series in lmom_series.items():
        w = int(name.rsplit("_w", 1)[1])
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=w, n_eff=n_h / w,
             theory=None, pseudo=0.0, kind="rho", window=w)

    # --- variância localizada (P3): max/min_z já são z-scores; nulo empírico corrige a inflação por
    # curtose da série (D-10). recent_vs_lagged só existe com a janela cheia. ---
    varloc_series = varloc_mod.history_null_series(e, cfg)
    rl_min_t = cfg.varloc.recent + cfg.varloc.lagged
    for name, series in varloc_series.items():
        min_t = rl_min_t if name.endswith("recent_vs_lagged") else cfg.features.warmup_min_n
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=min_t, n_eff=len(series),
             theory=None, pseudo=0.0)

    # --- bipower/saltos/leverage (P4): nulo empírico da própria série (a razão de salto e o leverage
    # de um GARCH são altos no seu histórico -> a versão calibrada só acende no excesso pós-quebra) ---
    jump_series = jump_mod.history_null_series(e, cfg)
    for name, series in jump_series.items():
        w = int(name.rsplit("_w", 1)[1])
        _add(out, name, np.asarray(series, dtype=np.float64), min_t=w, n_eff=n_h / w,
             theory=None, pseudo=0.0)

    # --- Haar multi-escala (F4): idem ---
    haar_series = ms_mod.history_series(e, cfg)
    n_scales, min_coeffs = cfg.multiscale.n_scales, cfg.multiscale.warmup_min_coeffs
    for name, arr in haar_series.items():
        if name.startswith("haar_energy_ln_s"):
            j = int(name.rsplit("s", 1)[1])
        elif name == "haar_contrast_fine_mid":
            # o contraste fino-vs-médio depende da escala 2, NÃO da mais grossa: usar n_scales-1
            # aqui mantinha a feature em NaN até t=96 sem necessidade nenhuma.
            j = min(2, n_scales - 1)
        else:
            j = n_scales - 1
        _add(out, name, arr, min_t=(2 ** (j + 1)) * min_coeffs, n_eff=len(arr), theory=None, pseudo=0.0)

    # --- F1.a: estatísticas RECURSIVAS (CUSUMs, autocorrelação de janela expansiva). O nulo vem de
    # réplicas com reinício, não de uma passada contínua — ver `_add_from_replicates`. A whitelist
    # está em YAML para que abrir a cobertura de F1.b seja um diff de configuração. ---
    kind_of = dict(cal_cfg.recursive_features)
    if kind_of:
        # F1.0: a série `e_vol` do online reproduzida sobre o histórico — só é calculada quando
        # algum braço recursivo está ligado, porque custa uma passada sequencial pelo histórico
        e_vol = history_evol(e, h0.rho1_abs_e, cfg)
        K, smooth_w = cal_cfg.transient_restart_every, cal_cfg.transient_smooth_w
        for prefix, mod in (("cusum_", cusum_mod), ("accum_", accum_mod), ("conformal_", conf_mod)):
            sub = frozenset(n for n in kind_of if n.startswith(prefix))
            if not sub:
                continue  # não paga a passada de um bloco que a whitelist não pediu
            reps = mod.history_null_series(
                e, e_vol, h0, cfg, K, max_reps=cal_cfg.transient_max_reps, wanted=sub
            )
            for name in sorted(reps):  # ordem estável, independente da iteração do dict
                _add_from_replicates(out, name, reps[name], pseudo, smooth_w, kind=kind_of[name])

    return out


def _null_at(spec: NullSpec, t: int) -> tuple[float, float]:
    """(mu, sd) do nulo no número efetivo de amostras n = min(t, window), transportando o nulo
    medido na janela cheia pela lei de escala de `spec.kind` (ver NullSpec)."""
    # transiente medido tem precedência: é o nulo exato daquele passo, não uma extrapolação
    if spec.table and t <= len(spec.table[0]):
        return spec.table[0][t - 1], spec.table[1][t - 1]

    if spec.kind == "cumsum":
        # acumulador sem reset: mu e sd guardam a deriva POR PASSO e a escala POR sqrt(passo)
        return spec.mu * t, max(spec.sd * math.sqrt(t), 1e-9)

    if spec.kind == "none" or spec.kind == "z" or spec.window <= 0:
        return spec.mu, spec.sd

    n = max(min(t, spec.window), 1)
    if n >= spec.window:
        return spec.mu, spec.sd

    if spec.kind == "rho":
        # autocorrelação: média ~ const, dp ∝ 1/sqrt(n) -> transporta a dp da janela cheia por
        # sqrt(W/n), preservando o fator de inflação idiossincrático da série.
        return spec.mu, max(spec.sd * math.sqrt(spec.window / n), 1e-9)

    if spec.kind == "var_ln":
        mu_th_w, sd_th_w = -1.0 / spec.window, math.sqrt(2.0 / spec.window)
        mu_th_n, sd_th_n = -1.0 / n, math.sqrt(2.0 / n)
    elif spec.kind == "frac":
        p0 = spec.aux
        var_w = max(p0 * (1.0 - p0) / spec.window, 1e-18)
        var_n = max(p0 * (1.0 - p0) / n, 1e-18)
        mu_th_w, sd_th_w = p0, math.sqrt(var_w)
        mu_th_n, sd_th_n = p0, math.sqrt(var_n)
    else:
        return spec.mu, spec.sd

    # o que é idiossincrático da série é o fator de inflação sobre o nulo i.i.d., não o nível
    inflation = spec.sd / max(sd_th_w, 1e-12)
    mu = mu_th_n + (spec.mu - mu_th_w)
    sd = max(inflation * sd_th_n, 1e-9)
    return mu, sd


def apply_calibration(feats: dict, null_stats: dict, t: int) -> None:
    """Acrescenta `<nome>_cal` a `feats`, in-place. NaN quando ainda não há amostras suficientes
    (t < min_t) ou quando o valor cru é NaN — nunca inventa um número."""
    for name, spec in null_stats.items():
        if name not in feats:
            # O bloco que emitia esta coluna não está em `default_blocks()`. `compute_null_stats`
            # roda uma lista fixa de `history_null_series`, então desligar um bloco deixava aqui uma
            # coluna `_cal` ÓRFÃ — NaN em todas as linhas, largura pura sem informação. Medido ao
            # podar LMomentBlock (2026-07-22): 4 colunas fantasma. A calibração tem de seguir o
            # conjunto de blocos ativo, não uma lista paralela.
            continue
        raw = feats.get(name)
        if raw is None or t < spec.min_t or not math.isfinite(raw):
            feats[f"{name}_cal"] = math.nan
            continue
        mu, sd = _null_at(spec, t)
        feats[f"{name}_cal"] = (raw - mu) / sd
