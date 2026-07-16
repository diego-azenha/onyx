"""Motor único -> linhas de treino (plano §8.1). Para cada série: fit_h0 + o MESMO `StreamScorer`
da submissão (ensemble=None), passo a passo, coletando (features, y_t=1{tau<=t}, peso). NUNCA
vetorizar o laço *dentro* de uma série (substituir o loop incremental por uma reconstrução em lote é
exatamente a armadilha §13.2 — "backtest vetorizado != execução causal", motor único,
docs/PLANO_REPOSITORIO.md §1).

`n_jobs` paraleliza o laço *externo*, entre séries — cada série é 100% independente (nenhum estado
compartilhado) e continua rodando pelo idêntico laço serial passo a passo dentro do seu próprio
processo; não é a vetorização proibida acima, é só a mesma computação rodando em paralelo em vez de
em sequência. Existe porque o motor de estado é Python puro (~1ms/passo medido) e o dataset real tem
~5M passos — sem isso, construir o dataset de treino sozinho leva ~85 min (plano §11 assumia 25us/
passo vetorizado/numba; isto ainda não foi feito, ver §11.4)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from sbrt.features.assembly import build_feature_order
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks


@dataclass
class SeriesRecord:
    dataset_id: int
    x_hist: np.ndarray
    x_online: np.ndarray
    tau_index: Optional[int]  # 0-based índice em x_online onde a quebra ocorre, ou None


def _thinning_keep_and_weight(t: int, cfg) -> tuple:
    """plano §8.1: mantém todos os passos t<=100; 101-400 a cada 2 (peso x2); >400 a cada 4 (peso x4)."""
    th = cfg.thinning
    if t <= th.full_until:
        return True, 1.0
    if t <= 400:
        return (t % th.step_101_400 == 0), float(th.step_101_400)
    return (t % th.step_401_plus == 0), float(th.step_401_plus)


def _build_rows_for_series(rec: SeriesRecord, cfg) -> pd.DataFrame:
    """Uma série inteira, do jeito serial/causal de sempre — é isto que roda em cada worker quando
    `n_jobs != 1`. Definida em nível de módulo (não uma closure) para ser picklable pelo joblib.

    Devolve um DataFrame pequeno (uma série só, no máximo ~250 linhas após o thinning) já em
    float32, não uma lista de dicts crua — ver nota em `build_training_rows` sobre por quê."""
    h0 = fit_h0(rec.x_hist, cfg)
    scorer = StreamScorer(h0, default_blocks(), None, cfg)

    rows = []
    order = None
    for i, x in enumerate(rec.x_online):
        t = i + 1
        feats = scorer.update_features(float(x))
        if order is None:
            order = build_feature_order(feats)

        keep, thin_w = _thinning_keep_and_weight(t, cfg)
        if not keep:
            continue

        y = 1 if (rec.tau_index is not None and i >= rec.tau_index) else 0
        row = {k: feats.get(k, np.nan) for k in order}
        row["id"] = rec.dataset_id
        row["t"] = t
        row["y"] = y
        row["thin_weight"] = thin_w
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        feature_cols = list(order)  # `order` é tuple — df[tuple] é lida como 1 chave multi-index, não uma lista de colunas
        df[feature_cols] = df[feature_cols].astype(np.float32)
        df["thin_weight"] = df["thin_weight"].astype(np.float32)
        df["id"] = df["id"].astype(np.int32)
        df["t"] = df["t"].astype(np.int32)
        df["y"] = df["y"].astype(np.int8)
    return df


def build_training_rows(
    train_series: Iterable[SeriesRecord], cfg, progress: bool = True, n_jobs: int = 1
) -> pd.DataFrame:
    """Laço externo (por série) envolvido em tqdm quando progress=True — é laço de desenvolvimento
    local, não caminho de submissão (plano §8, regra tqdm). `n_jobs=1` (default) = serial, idêntico
    ao comportamento original; `n_jobs=-1`/N>1 = paralelo entre séries via joblib (ver docstring do
    módulo) — resultado é o mesmo DataFrame, só mais rápido de construir.

    IMPORTANTE (bug real encontrado com o dataset completo, ~2.5M linhas): construir uma lista
    Python de milhões de dicts e só then chamar `pd.DataFrame(lista_gigante)` uma única vez no
    final é um anti-padrão conhecido do pandas — cada dict tem overhead de objeto Python real, e a
    lista intermediária de milhões de dicts consome ordens de magnitude mais memória que o
    DataFrame final (float32, ~78 features x 2.5M linhas ~= 750MB, plano §8.1). Isso causou um
    `numpy._core._exceptions._ArrayMemoryError` com >8GB consumidos na conversão final. Correção:
    cada série vira seu próprio DataFrame pequeno (já em float32) dentro de `_build_rows_for_series`
    logo depois de coletar sua própria lista curta de dicts (no máximo ~250 linhas após thinning);
    aqui só concatenamos os ~10.000 DataFrames pequenos, uma vez, no final."""
    if n_jobs == 1:
        iterator = tqdm(train_series, desc="construindo dataset de treino") if progress else train_series
        dfs = [_build_rows_for_series(rec, cfg) for rec in iterator]
    else:
        train_series = list(train_series)
        jobs = Parallel(n_jobs=n_jobs, return_as="generator")(
            delayed(_build_rows_for_series)(rec, cfg) for rec in train_series
        )
        iterator = (
            tqdm(jobs, total=len(train_series), desc=f"construindo dataset de treino (n_jobs={n_jobs})")
            if progress
            else jobs
        )
        dfs = list(iterator)

    dfs = [df for df in dfs if not df.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
