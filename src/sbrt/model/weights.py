"""Pesos de linha pareado-consistentes com a TS-AUC (parecer de auditoria §3.10/§4.4, roadmap R1).

A TS-AUC se reescreve como a fração de pares (positivo, negativo) do MESMO passo t corretamente
ordenados, agregada sobre todos os t (parecer §3.1). Nesse pool de pares, cada linha positiva de t
participa de n_neg(t) pares e cada linha negativa de n_pos(t) pares — logo o surrogate pontual
pareado-consistente dá peso ∝ n_neg(t) aos positivos e ∝ n_pos(t) aos negativos, e não o mesmo peso
às duas classes como antes (w(t) = n_pos(t)*n_neg(t)/n_alive(t) para TODA linha de t, sem distinguir
classe). O esquema antigo acerta a massa agregada por passo (∝ n_pos*n_neg) mas erra a partição
intra-passo (1:1 em vez de n_neg:n_pos) — em t<=50 isso dilui o gradiente dos positivos por ~12x,
exatamente no bucket onde a AUC medida é mais fraca (parecer §3.10).

Contagens suavizadas por pseudo-contagem (t com poucos positivos não vira peso quase-infinito) e a
razão w_pos(t)/w_neg(t) capada em `max_ratio` (t muito pequeno tem n_pos raro, n_neg~5000 — sem cap
isso troca viés por variância de gradiente, parecer §4.4). Multiplicado pelo fator de thinning e
normalizado para média 1, como antes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _detectability_multiplier(
    rows: pd.DataFrame, mode: str, floor: float, ramp_len: float, det: pd.DataFrame
) -> np.ndarray:
    """X1 (BRAINSTORM_RUPTURA_TSAUC.md §3): multiplicador POR LINHA aplicado só aos POSITIVOS, para
    tirar peso de positivos que não carregam sinal no passo em que pesam. Pesos do professor -- usa τ
    só no treino (o alvo `y=1{τ≤t}` já usa), nenhuma causalidade de inferência tocada. `det` é o mapa
    de detectabilidade por série (colunas id/detectability/tau_index): na PRODUÇÃO vem de
    `model/detectability.py:compute_detectability_map` (inline); offline, do CSV do F0.b. Séries fora
    do mapa (quebra tardia, τ≥t_max) recebem multiplicador 1 (sem penalidade). Modos:
      hard  -- dropa (peso 0) positivos no quintil inferior de detectabilidade;
      soft  -- w_pos ×= clip(d_i/q95, floor, 1)  (variante ADOTADA);
      ramp  -- w_pos ×= clip((t−τ)/ramp_len, floor, 1): cresce com passos-desde-quebra."""
    is_pos = rows["y"].to_numpy(dtype=bool)
    mult = np.ones(len(rows), dtype=np.float64)
    if mode == "none" or det is None or len(det) == 0:
        return mult
    ids = rows["id"]
    if mode == "soft":
        q95 = float(det["detectability"].quantile(0.95))
        d = ids.map(det.set_index("id")["detectability"]).to_numpy(dtype=np.float64)
        m = np.clip(d / max(q95, 1e-12), floor, 1.0)
        m = np.where(np.isnan(d), 1.0, m)  # séries sem d_i: sem penalidade
        mult = np.where(is_pos, m, 1.0)
    elif mode == "hard":
        thr = float(det["detectability"].quantile(0.20))
        low = set(det.loc[det["detectability"] <= thr, "id"])
        drop = is_pos & ids.isin(low).to_numpy()
        mult[drop] = 0.0
    elif mode == "ramp":
        tau = ids.map(det.set_index("id")["tau_index"]).to_numpy(dtype=np.float64)
        age = rows["t"].to_numpy(dtype=np.float64) - tau
        r = np.clip(age / ramp_len, floor, 1.0)
        r = np.where(np.isnan(tau), 1.0, r)
        mult = np.where(is_pos, r, 1.0)
    else:
        raise ValueError(f"detectability_mode desconhecido: {mode!r}")
    return mult


def compute_row_weights(
    rows: pd.DataFrame, cfg, pseudo_count: float = 5.0, max_ratio: float = 50.0,
    detectability_mode: str = "none",
    detectability: "pd.DataFrame | None" = None,
    detectability_path: str = "artifacts/reports/detectability.csv",
    detect_floor: float = 0.3, ramp_len: float = 50.0,
    wt_align: bool = False,
    noisy_neg_ids: "set | None" = None, noisy_neg_factor: float = 0.2,
) -> np.ndarray:
    """X1: default `detectability_mode="none"` (comportamento V4 inalterado para todos os chamadores
    nus -- sweep, testes). O caminho de submissão (`adapter/platform.py`) opta explicitamente por
    `cfg.weights.detectability_mode` e passa `detectability` computado INLINE; offline,
    `scripts/train.py --detectability-mode` passa o modo e a função lê o CSV do F0.b."""
    counts = rows.groupby("t")["y"].agg(n_pos="sum", n_alive="count")
    counts["n_neg"] = counts["n_alive"] - counts["n_pos"]
    n_pos_s = counts["n_pos"] + pseudo_count
    n_neg_s = counts["n_neg"] + pseudo_count

    # w_pos(t) ~ n_neg_s (número de pares que cada positivo participa), w_neg(t) ~ n_pos_s,
    # capando a razão entre as duas em max_ratio (equivalente a clipar n_neg_s/n_pos_s em
    # [1/max_ratio, max_ratio] e manter a proporcionalidade exata dentro do cap).
    counts["w_t_pos"] = np.minimum(n_neg_s, max_ratio * n_pos_s)
    counts["w_t_neg"] = np.minimum(n_pos_s, max_ratio * n_neg_s)

    w_t_pos_map = counts["w_t_pos"].to_dict()
    w_t_neg_map = counts["w_t_neg"].to_dict()
    t_pos = rows["t"].map(w_t_pos_map).to_numpy(dtype=np.float64)
    t_neg = rows["t"].map(w_t_neg_map).to_numpy(dtype=np.float64)
    is_pos = rows["y"].to_numpy(dtype=bool)
    base_w = np.where(is_pos, t_pos, t_neg)

    w = base_w * rows["thin_weight"].to_numpy(dtype=np.float64)

    # X1: multiplicador de detectabilidade nos positivos (professor). A renormalização mean-1 abaixo
    # preserva a massa total automaticamente, então isto reordena a ênfase sem mudar a escala global.
    if detectability_mode != "none":
        det = detectability if detectability is not None else pd.read_csv(detectability_path)
        w = w * _detectability_multiplier(rows, detectability_mode, detect_floor, ramp_len, det)

    # B1/F3 (BRAINSTORM_RUPTURA_V2.md §2.3): alinhar a MASSA TOTAL de peso por passo t com o peso da
    # métrica w_t = n_pos(t)·n_neg(t). O esquema pareado acerta a partição pos:neg DENTRO de t, mas o
    # `max_ratio` e o thinning distorcem a massa ENTRE passos; este reescalonamento por t restaura a
    # proporção que a TS-AUC de fato cobra, preservando a razão intra-t. Braço separado do X1.
    # B3 (BRAINSTORM_RUPTURA_V2.md §1.3b): rebaixa os NEGATIVOS de séries flagadas como suspeita de
    # rótulo (T0.1, estatística anticausal independente do score). Só treino; espelho do X1 no eixo que
    # venceu. Multiplicador `noisy_neg_factor` nas linhas y=0 dessas séries.
    if noisy_neg_ids:
        is_neg = ~rows["y"].to_numpy(dtype=bool)
        flagged = rows["id"].isin(noisy_neg_ids).to_numpy()
        w = w * np.where(is_neg & flagged, noisy_neg_factor, 1.0)

    if wt_align:
        wt_target = (counts["n_pos"] * counts["n_neg"]).clip(lower=1.0)
        # Grade do board (2026-09-30, knowledge/frentes/teto-offline/V-empilhamento.md, V11): cada passo retido
        # representa `thin_weight` passos do board (1/2/4); sem isto, t>400 fica 4x sub-ponderado, o mesmo
        # erro de grade que o A2 achou na avaliação.
        if "thin_weight" in rows.columns:
            wt_target = wt_target * rows.groupby("t")["thin_weight"].first().reindex(counts.index).fillna(1.0)
        cur = pd.Series(w, index=rows["t"].to_numpy()).groupby(level=0).sum()
        scale = (wt_target / cur).reindex(counts.index).fillna(1.0)
        w = w * rows["t"].map(scale.to_dict()).to_numpy(dtype=np.float64)

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w
