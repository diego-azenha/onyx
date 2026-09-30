"""Shim para o callback oficial da plataforma (plano §15.1 P0).

Contrato CONFIRMADO via `quickstarter_notebook.ipynb` (célula `def train`/`def infer`) — não é mais
best-guess: `train(datasets, model_directory_path)` recebe uma lista de
`(dataset_id, x_historical, x_online, tau_index)`; `infer(datasets, model_directory_path)` é um
generator que primeiro dá um `yield` vazio (sinaliza prontidão ao runner) e depois, para cada
`(x_historical, x_online)`, itera `x_online` emitindo exatamente um `float` por ponto. Casa
exatamente com `crunch.container.GeneratorWrapper` (`ERROR_FIRST_YIELD_MUST_BE_NONE`).

`train()` em modo `fallback` (padrão em configs/default.yaml) não precisa de dado de treino — o
score é 100% determinístico a partir do histórico de cada série (plano §8.5) — mas ainda grava um
artefato placeholder para que `infer()` tenha algo a carregar, no mesmo espírito do baseline oficial.
Em modo `supervised`, treina o pipeline completo (Frente H) e persiste o `ModelEnsemble`.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Iterable, List, Optional, Tuple

import joblib
import numpy as np

from sbrt.config import load_config
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks

_MODEL_FILE = "model.joblib"


def train(
    datasets: List[Tuple[int, List[float], List[float], Optional[int]]],
    model_directory_path: str,
) -> None:
    cfg = load_config()
    os.makedirs(model_directory_path, exist_ok=True)

    if cfg.model.mode == "supervised":
        from sbrt.model.dataset import SeriesRecord, build_training_rows
        from sbrt.model.weights import compute_row_weights
        from sbrt.model.detectability import compute_detectability_map
        from sbrt.model.fuse import fuse_boosters
        from sbrt.model.train import refit_full
        from sbrt.model.train import train as train_ensemble

        records = [
            SeriesRecord(
                dataset_id=dataset_id,
                x_hist=np.asarray(x_hist, dtype=np.float64),
                x_online=np.asarray(x_online, dtype=np.float64),
                tau_index=tau_index,
            )
            for dataset_id, x_hist, x_online, tau_index in datasets
        ]
        rows = build_training_rows(records, cfg, n_jobs=cfg.model.dataset_n_jobs)

        # X1 (adotado 2026-07-24): supervisão ponderada por detectabilidade. A nuvem treina do zero e
        # NÃO tem o detectability.csv local, então o mapa d_i é computado INLINE dos próprios registros
        # (model/detectability.py, mesma matemática do censo A1 + F0.b). Só afeta o TREINO -- inferência
        # inalterada. Ver docs/HISTORICO.md e configs/default.yaml:weights.
        det_map = None
        if cfg.weights.detectability_mode != "none":
            det_map = compute_detectability_map(records, cfg, n_jobs=cfg.model.dataset_n_jobs)
        weights = compute_row_weights(
            rows, cfg,
            detectability_mode=cfg.weights.detectability_mode,
            detectability=det_map,
            detect_floor=cfg.weights.detect_floor,
        )

        # BAGGING DE SEMENTES (2026-07-22). A nuvem TREINA DO ZERO -- `resources/` nao e usado aqui --
        # entao o bagging tem de acontecer neste laco, ou a submissao perde os +0,0048 medidos.
        # Cada semente muda so o sorteio interno do LightGBM (`boost_seed`), NAO os folds
        # (`cfg.seed` intocado), e os K boosters sao FUNDIDOS num so para nao multiplicar por K o
        # custo de inferencia por passo. Ver docs/BACKLOG_TSAUC.md, "Bagging de sementes".
        # Oráculo destilado (knowledge/frentes/oraculo-destilado): o alvo q = P(y_t | série inteira) é
        # calculado UMA vez, com cross-fit por série, e reaproveitado por todas as sementes e folds.
        if cfg.lightgbm.soft_label and cfg.lightgbm.soft_label_modo == "destilacao":
            from sbrt.model.oraculo import alvos_oraculo
            q = alvos_oraculo(rows, cfg, cfg.seed)
            for k in range(cfg.lightgbm.n_folds):
                rows[f"y_soft_f{k}"] = q

        seeds = list(cfg.lightgbm.bag_seeds) or [cfg.lightgbm.boost_seed or cfg.seed]
        boosters, ensemble = [], None
        for s in seeds:
            cfg_s = replace(cfg, lightgbm=replace(cfg.lightgbm, boost_seed=int(s)))
            ensemble, _oof_pred = train_ensemble(rows, weights, cfg_s)
            fundido = fuse_boosters(ensemble.boosters)
            if cfg.lightgbm.full_refit:
                # R1 (knowledge/frentes/teto-offline/R1-refit-completo.md): cada booster de fold viu 80%
                # das séries; o refit vê 100%. Fusão 50/50 entre a média dos folds e o refit (a melhor
                # variante no holdout externo).
                fundido = fuse_boosters([fundido, refit_full(rows, weights, ensemble, cfg_s)])
            boosters.append(fundido)
        ensemble = replace(ensemble, boosters=[fuse_boosters(boosters)])
        ensemble.save(model_directory_path)
        joblib.dump({"mode": "supervised"}, os.path.join(model_directory_path, _MODEL_FILE))
    else:
        joblib.dump({"mode": "fallback"}, os.path.join(model_directory_path, _MODEL_FILE))


def infer(
    datasets: Iterable[Tuple[List[float], Iterable[float]]],
    model_directory_path: str,
):
    cfg = load_config()
    model_path = os.path.join(model_directory_path, _MODEL_FILE)
    payload = joblib.load(model_path) if os.path.exists(model_path) else {"mode": "fallback"}

    ensemble = None
    if payload.get("mode") == "supervised":
        from sbrt.model.predict import ModelEnsemble

        ensemble = ModelEnsemble.load(model_directory_path)

    yield  # sinaliza prontidão ao runner (GeneratorWrapper.ERROR_FIRST_YIELD_MUST_BE_NONE)

    for x_historical, x_online in datasets:
        hist = np.asarray(x_historical, dtype=np.float64)
        h0 = fit_h0(hist, cfg)
        scorer = StreamScorer(h0, default_blocks(), ensemble, cfg)
        for point in x_online:
            yield float(scorer.update(float(point)))
