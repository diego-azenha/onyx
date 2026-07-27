"""GroupKFold(5, groups=id) + 1 LightGBM por fold (plano §8.3). Predição final = média das
probabilidades dos 5 modelos (model/predict.py).

plano_acao_v1_para_v2.md A2: `y_t = 1{tau<=t}` tem uma taxa-base fortemente crescente com t (~7.6%
a ~39.7%), neutra para TS-AUC por invariância C1 mas dominante para logloss/AUC de linha. Por isso:
(1) a curva de taxa-base vira `init_score`, deixando o LightGBM aprender só o resíduo transversal;
(2) o early stopping usa `binary_logloss` (agora medindo só o resíduo, já que init_score desloca a
métrica), não mais `auc` (que saturava cedo dominada pela taxa-base — 89-110 árvores medidas contra
400-800 esperadas, plano §8.3).

R2 (docs/PARECER_AUDITORIA_ONYX.md §6-R2): mesmo com init_score corrigindo o offset, a parada e a
seleção de hiperparâmetros continuavam julgadas por `binary_logloss` pontual -- uma régua diferente
da métrica oficial (TS-AUC = fração de pares concordantes por passo, parecer §3.1). O juiz agora é
um `feval` custom: AUC ponderada por passo t sobre o PRÓPRIO fold de validação (`ts_auc_by_t`,
`first_metric_only=True` na parada); `binary_logloss` continua computado à mão dentro do mesmo feval
só para diagnóstico/plot (training_curves), sem influenciar a parada. Isto é critério interno de
fold, não estimador de leaderboard -- compatível com a §9.0 (nunca substitui a submissão oficial nem
o comparador pareado de scripts/compare_oof.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

import lightgbm as lgb

from sbrt.evaluation.splits import grouped_stratified_kfold
from sbrt.evaluation.ts_auc import weighted_ts_auc
from sbrt.model.base_rate import fit_base_rate_curve, predict_base_rate_logit
from sbrt.model.predict import ModelEnsemble, RankModelEnsemble

_NON_FEATURE_COLS = {"id", "t", "y", "thin_weight"}


def _feature_contri(lgb_cfg, feature_cols) -> list[float] | None:
    """B6 + C1(ii): vetor `feature_contri` do LightGBM, alinhado a `feature_cols`.

    `meta_h0_*` levam `feature_contri_meta` (B6, adotado) -- casamento por PREFIXO. As colunas de
    `feature_contri_extra_cols` levam `feature_contri_extra` (C1(ii)) -- multiplicador SEPARADO e
    casamento por NOME EXATO (ver a justificativa em config.py: o prefixo pegaria
    `conformal_logm_abs_reset`, que é acumulador de evidência, não rastreador de `t`).

    Devolve `None` quando nada está penalizado, para não injetar um vetor de 1.0 nos params (que
    seria inócuo mas mudaria o dicionário passado ao LightGBM e, com ele, a reprodutibilidade
    bit-a-bit das medições anteriores).
    """
    meta = getattr(lgb_cfg, "feature_contri_meta", 1.0)
    extra = getattr(lgb_cfg, "feature_contri_extra", 1.0)
    cols = frozenset(getattr(lgb_cfg, "feature_contri_extra_cols", ()) or ())
    if meta == 1.0 and (extra == 1.0 or not cols):
        return None
    out = []
    for c in feature_cols:
        if c.startswith("meta_h0"):
            out.append(meta)
        elif c in cols:
            out.append(extra)
        else:
            out.append(1.0)
    return out


def _make_fold_feval(
    t_valid: np.ndarray, max_rows: int | None, seed: int, raw_to_prob: bool = False, stopping_metric: str = "logloss"
):
    """Fecha sobre os valores de `t` do fold de validação (posicionalmente alinhados com as linhas
    do `lgb.Dataset` de validação, que preserva a ordem original -- LightGBM não embaralha dados
    internamente). `max_rows`: subamostra FIXA (sorteada uma vez, não a cada rodada -- resortear a
    cada rodada injetaria ruído extra no critério de parada) para manter o custo por rodada baixo em
    folds grandes (parecer §6-R2). `ts_auc_by_t` (rank-based, invariante a qualquer transformação
    monótona de `preds`) funciona idêntico para objetivo binário ou de ranking (R3, train_rank);
    `raw_to_prob=True` só afeta o diagnóstico `binary_logloss_diag`, aplicando a MESMA sigmoide fixa
    usada por `RankModelEnsemble.predict_one` antes de computar a logloss -- sem isso, `preds` de um
    objetivo de ranking (escala arbitrária, pode ser negativo) tornaria essa logloss sem sentido.

    `stopping_metric` controla qual das duas entra PRIMEIRO na lista retornada -- é essa ordem que
    `first_metric_only=True` usa para decidir a parada (cfg.lightgbm.early_stopping_metric). AMBAS
    são sempre computadas e registradas (fold_evals, training_curves); só a ordem muda. Ver a nota
    empírica em config.py:LightGBMConfig.early_stopping_metric -- "ts_auc_by_t" sozinho regrediu a
    TS-AUC OOF real por ruído de seleção (n efetivo ~10^4 séries, não o número de linhas)."""
    n = len(t_valid)
    if max_rows is not None and n > max_rows:
        rng = np.random.default_rng(seed)
        sub_idx = np.sort(rng.choice(n, size=max_rows, replace=False))
    else:
        sub_idx = None

    def _feval(preds: np.ndarray, dataset: "lgb.Dataset"):
        y_full = dataset.get_label()
        w_full = dataset.get_weight()
        if sub_idx is not None:
            preds_s, y_s, t_s = preds[sub_idx], y_full[sub_idx], t_valid[sub_idx]
            w_s = w_full[sub_idx] if w_full is not None else None
        else:
            preds_s, y_s, t_s = preds, y_full, t_valid
            w_s = w_full

        auc = weighted_ts_auc(t_s, y_s, preds_s)
        if not np.isfinite(auc):
            auc = 0.5  # nunca deixar a parada ver NaN (subamostra sem par completo em algum t)

        p_for_diag = 1.0 / (1.0 + np.exp(-preds_s)) if raw_to_prob else preds_s
        p = np.clip(p_for_diag, 1e-7, 1.0 - 1e-7)
        terms = y_s * np.log(p) + (1.0 - y_s) * np.log(1.0 - p)
        logloss = float(-np.average(terms, weights=w_s))

        entries = [
            ("ts_auc_by_t", float(auc), True),
            ("binary_logloss_diag", logloss, False),
        ]
        return entries if stopping_metric == "ts_auc_by_t" else list(reversed(entries))

    return _feval


def _make_pairwise_fobj(t_valid_full, y_full, w_row, w_t, alpha, n_pairs, seed):
    """B4 (BRAINSTORM_RUPTURA_V2.md §1.5): objetivo custom = (1−α)·logloss + α·pairwise intra-t.
    Pares (pos,neg) do MESMO grupo t, pré-amostrados uma vez (P por positivo, t-casados), peso w_t, SEM
    desconto de posição e SEM truncation. Vetorizado por `bincount` (sem laço por t por iteração).
    Pares do mesmo t compartilham o init_score da taxa-base ⇒ o termo pairwise é cego a f(t) de graça.
    Retorna grad/hess já com todos os pesos embutidos (o Dataset entra com weight=1)."""
    rng = np.random.default_rng(seed)
    n = len(y_full)
    # Amostragem t-a-t (nao positivo-a-positivo): um unico groupby resolve os indices de cada
    # (t, y), e cada grupo sorteia de uma vez os len(pos_t)*n_pairs negativos. Equivalente ao
    # sorteio por positivo, mas ~399 iteracoes em vez de ~500k (o pre-sorteio roda por fold).
    idx_by = pd.DataFrame({"t": t_valid_full, "y": y_full}).groupby(["t", "y"]).indices
    pos_arr, neg_arr, wt_pair = [], [], []
    for t in np.unique(t_valid_full):
        p_idx = idx_by.get((t, 1))
        n_idx = idx_by.get((t, 0))
        if p_idx is None or n_idx is None or len(p_idx) == 0 or len(n_idx) == 0:
            continue
        reps = np.repeat(p_idx, n_pairs)
        picks = n_idx[rng.integers(0, len(n_idx), size=len(reps))]
        pos_arr.append(reps); neg_arr.append(picks)
        wt_pair.append(np.full(len(reps), w_t.get(int(t), 1.0)))
    pa = np.concatenate(pos_arr); na = np.concatenate(neg_arr); wp = np.concatenate(wt_pair)
    wp = wp / max(wp.mean(), 1e-12)

    def _fobj(preds, _dataset):
        p = 1.0 / (1.0 + np.exp(-preds))
        g = (1.0 - alpha) * w_row * (p - y_full)
        h = (1.0 - alpha) * w_row * np.maximum(p * (1.0 - p), 1e-6)
        d = preds[pa] - preds[na]
        sig = 1.0 / (1.0 + np.exp(-d))
        c = (1.0 - sig) * wp        # dL/ds_pos = −(1−σ); dL/ds_neg = +(1−σ)
        hh = sig * (1.0 - sig) * wp
        g += alpha * (np.bincount(pa, weights=-c, minlength=n) + np.bincount(na, weights=c, minlength=n))
        h += alpha * (np.bincount(pa, weights=hh, minlength=n) + np.bincount(na, weights=hh, minlength=n))
        return g, h
    return _fobj


def train(rows: pd.DataFrame, weights: np.ndarray, cfg, progress: bool = True) -> tuple:
    """tqdm sobre os 5 folds; dentro de cada fold, LightGBM usa seu próprio log verbose (não
    duplicar barra de progresso, plano §8 regra tqdm). Retorna (ModelEnsemble, oof_pred) — oof_pred
    é a probabilidade calibrada out-of-fold por linha, alinhada a `rows` (diagnóstico A4, não faz
    parte do artefato salvo)."""
    feature_cols = sorted(c for c in rows.columns if c not in _NON_FEATURE_COLS)
    X = rows[feature_cols].to_numpy(dtype=np.float32)  # plano §8.1: float32 no dataset de treino
    y = rows["y"].to_numpy(dtype=np.int32)
    t_values = rows["t"].to_numpy(dtype=np.float64)

    lgb_cfg = cfg.lightgbm
    # A3b: sob `base_rate_weighted`, a curva é ajustada com os mesmos pesos que o LightGBM usa, e não
    # na contagem crua -- ver config.py:LightGBMConfig.base_rate_weighted para o descasamento medido
    # (até +3,81 em log-odds) que isto corrige.
    base_rate_curve = fit_base_rate_curve(
        t_values, y.astype(np.float64),
        weights=weights if getattr(lgb_cfg, "base_rate_weighted", False) else None,
    )
    init_score_full = predict_base_rate_logit(t_values, base_rate_curve)
    params = dict(
        objective="binary",
        metric="None",  # R2: métricas internas desligadas -- o feval custom cobre AUC-por-t
        # (parada, first_metric_only) e binary_logloss (diagnóstico), ambos vistos por valid_sets.
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        max_depth=lgb_cfg.max_depth,
        min_data_in_leaf=lgb_cfg.min_data_in_leaf,
        feature_fraction=lgb_cfg.feature_fraction,
        bagging_fraction=lgb_cfg.bagging_fraction,
        bagging_freq=lgb_cfg.bagging_freq,
        lambda_l2=lgb_cfg.lambda_l2,
        max_bin=lgb_cfg.max_bin,
        deterministic=lgb_cfg.deterministic,
        force_row_wise=lgb_cfg.force_row_wise,
        num_threads=lgb_cfg.train_num_threads,
        # `boost_seed` existe SO para calibrar o nulo da propria regra de decisao de R0. Ele muda o
        # sorteio de `bagging_fraction`/`feature_fraction` **sem** mexer nos folds (que continuam
        # vindo de `cfg.seed`), isolando a unica fonte de variancia que NAO se cancela no bootstrap
        # pareado: o booster e um sorteio aleatorio, e o bootstrap por serie trata as predicoes como
        # se fossem fixas. Ver docs/BACKLOG_TSAUC.md, seccao "O nulo da regra de decisao".
        seed=lgb_cfg.boost_seed if lgb_cfg.boost_seed is not None else cfg.seed,
        verbose=-1,
    )

    # B5/B6 (BRAINSTORM_RUPTURA_V2.md §2.5/§2.6): regularização estrutural alinhada a `feature_cols`.
    # Gated por cfg (off por default); só o braço binário os liga. Ver scripts/train.py.
    if getattr(lgb_cfg, "monotone_llr", False):
        _MONO = ("cusum_mean_pos", "cusum_var_up", "conformal_logm_abs_reset", "bayes_lo")
        params["monotone_constraints"] = [1 if c.startswith(_MONO) else 0 for c in feature_cols]
    _contri = _feature_contri(lgb_cfg, feature_cols)
    if _contri is not None:
        params["feature_contri"] = _contri

    # D1/D3 (CAMPANHA_POLIMENTO.md): knobs estruturais do aprendiz, off por default.
    if getattr(lgb_cfg, "linear_tree", False):
        params["linear_tree"] = True
    if getattr(lgb_cfg, "dart", False):
        params["boosting"] = "dart"
    if not getattr(lgb_cfg, "enable_bundle", True):
        params["enable_bundle"] = False

    boosters = []
    fold_evals = []
    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    folds = list(grouped_stratified_kfold(rows, lgb_cfg.n_folds, cfg.seed))
    fold_iter = tqdm(folds, desc="treinando folds") if progress else folds

    use_pairwise = getattr(lgb_cfg, "pairwise_alpha", 0.0) > 0.0
    for train_idx, valid_idx in fold_iter:
        params_use = params
        if use_pairwise:
            ytr = y[train_idx]; ttr = t_values[train_idx].astype(np.int64)
            gt = pd.DataFrame({"t": ttr, "y": ytr}).groupby("t")["y"]
            w_t = {int(t): float((s == 1).sum() * (s == 0).sum()) for t, s in gt}
            fobj = _make_pairwise_fobj(ttr, ytr, weights[train_idx].astype(np.float64), w_t,
                                       lgb_cfg.pairwise_alpha, lgb_cfg.pairwise_pairs, cfg.seed)
            dtrain = lgb.Dataset(X[train_idx], label=ytr, init_score=init_score_full[train_idx])  # weight embutido no fobj
            params_use = {**params, "objective": fobj}  # objetivo custom substitui "binary"
        else:
            dtrain = lgb.Dataset(
                X[train_idx], label=y[train_idx], weight=weights[train_idx], init_score=init_score_full[train_idx]
            )
        dvalid = lgb.Dataset(
            X[valid_idx],
            label=y[valid_idx],
            weight=weights[valid_idx],
            init_score=init_score_full[valid_idx],
            reference=dtrain,
        )
        feval = _make_fold_feval(
            t_values[valid_idx], lgb_cfg.feval_max_valid_rows, cfg.seed,
            # Com o objetivo BUILT-IN "binary" o LightGBM ja entrega `preds` como probabilidade ao
            # feval; com objetivo CUSTOM (B4) ele entrega o SCORE BRUTO. Sem esta sigmoide o
            # `binary_logloss_diag` do braco pairwise trata score bruto como probabilidade (medido:
            # 5,3 contra 0,66 do incumbente) e, como a parada le justamente essa metrica
            # (early_stopping_metric=logloss), a parada antecipada rodaria num sinal sem sentido.
            raw_to_prob=use_pairwise,
            stopping_metric=lgb_cfg.early_stopping_metric,
        )
        evals_result: dict = {}
        booster = lgb.train(
            params_use,
            dtrain,
            num_boost_round=lgb_cfg.n_estimators_cap,
            valid_sets=[dvalid],
            feval=feval,
            callbacks=[
                # `early_stopping_rounds <= 0` desliga a parada antecipada e treina exatamente
                # `n_estimators_cap` rodadas. Existe porque a parada antecipada e uma FONTE DE
                # VARIANCIA medida: o numero de arvores do mesmo fold varia 51->103 so trocando a
                # semente, e o dp de 0,0041 na TS-AUC que isso produz e maior que qualquer efeito de
                # feature ja medido no projeto (docs/BACKLOG_TSAUC.md, "O nulo da regra de decisao").
                # Rodadas fixas trocam um pouco de vies por muito menos ruido de sorteio.
                *([lgb.early_stopping(lgb_cfg.early_stopping_rounds, first_metric_only=True, verbose=False)]
                  if lgb_cfg.early_stopping_rounds > 0 else []),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        boosters.append(booster)
        fold_evals.append(evals_result)

        # raw_score=True nunca inclui init_score (é um construto do Dataset, não do modelo salvo) —
        # por isso somamos o mesmo offset usado no treino antes do sigmoid (plano A2, model/predict.py).
        raw_valid = booster.predict(X[valid_idx], raw_score=True)
        full_logit_valid = raw_valid + init_score_full[valid_idx]
        oof_pred[valid_idx] = 1.0 / (1.0 + np.exp(-full_logit_valid))

    ensemble = ModelEnsemble(
        boosters=boosters,
        feature_order=tuple(feature_cols),
        predict_num_threads=lgb_cfg.predict_num_threads,
        fold_evals=fold_evals,
        base_rate_curve=base_rate_curve,
    )
    return ensemble, oof_pred


def train_rank(rows: pd.DataFrame, cfg, progress: bool = True) -> tuple:
    """R3 (docs/PARECER_AUDITORIA_ONYX.md §6-R3): mesmo split por série (`grouped_stratified_kfold`)
    do modo binário (`train`), mas objetivo de RANKING por grupo t -- otimiza diretamente a
    concordância de pares intra-t, a forma fechada exata da TS-AUC (parecer §3.1). Membro PARALELO
    do ensemble binário, não substituto (nenhum precedente interno, parecer §6-R3) -- retorna um
    `RankModelEnsemble` (model/predict.py), combinável com o binário via `CombinedModelEnsemble` ou
    comparável via `scripts/combine_oof.py` (rank-average offline).

    SEM `init_score`: dentro de um grupo (linhas do mesmo t), um deslocamento constante não muda a
    ordem relativa dos itens -- matematicamente neutro para uma perda de ranking, ao contrário do
    modo binário (onde a taxa-base domina a logloss pontual, plano_acao A2). Peso de linha = só
    `thin_weight` normalizado: o desbalanceamento de classe intra-t já é tratado estruturalmente pela
    perda pareada (cada par (pos,neg) do grupo contribui um termo de gradiente) -- aplicar também os
    pesos classe-balanceados de R1 (model/weights.py) duplicaria esse efeito."""
    feature_cols = sorted(c for c in rows.columns if c not in _NON_FEATURE_COLS)
    X_full = rows[feature_cols].to_numpy(dtype=np.float32)
    y_full = rows["y"].to_numpy(dtype=np.int32)
    t_full = rows["t"].to_numpy(dtype=np.int64)

    thin_w = rows["thin_weight"].to_numpy(dtype=np.float64)
    thin_w = thin_w / thin_w.mean()

    lgb_cfg = cfg.lightgbm
    rank_cfg = cfg.rank
    base_params = dict(
        objective=rank_cfg.objective,
        label_gain=list(rank_cfg.label_gain),
        metric="None",
        learning_rate=lgb_cfg.learning_rate,
        num_leaves=lgb_cfg.num_leaves,
        max_depth=lgb_cfg.max_depth,
        min_data_in_leaf=lgb_cfg.min_data_in_leaf,
        feature_fraction=lgb_cfg.feature_fraction,
        bagging_fraction=lgb_cfg.bagging_fraction,
        bagging_freq=lgb_cfg.bagging_freq,
        lambda_l2=lgb_cfg.lambda_l2,
        max_bin=lgb_cfg.max_bin,
        deterministic=lgb_cfg.deterministic,
        force_row_wise=lgb_cfg.force_row_wise,
        num_threads=lgb_cfg.train_num_threads,
        # `boost_seed` existe SO para calibrar o nulo da propria regra de decisao de R0. Ele muda o
        # sorteio de `bagging_fraction`/`feature_fraction` **sem** mexer nos folds (que continuam
        # vindo de `cfg.seed`), isolando a unica fonte de variancia que NAO se cancela no bootstrap
        # pareado: o booster e um sorteio aleatorio, e o bootstrap por serie trata as predicoes como
        # se fossem fixas. Ver docs/BACKLOG_TSAUC.md, seccao "O nulo da regra de decisao".
        seed=lgb_cfg.boost_seed if lgb_cfg.boost_seed is not None else cfg.seed,
        verbose=-1,
    )

    # B5/B6 (BRAINSTORM_RUPTURA_V2.md §2.5/§2.6): regularização estrutural alinhada a `feature_cols`.
    # Gated por cfg (off por default); só o braço binário os liga. Ver scripts/train.py.
    #
    # Escreve em `base_params`, não em `params`: aqui `params` só passa a existir DENTRO do laço de
    # folds (linha ~343, onde recebe `lambdarank_truncation_level`). Escrever nele antes disso levanta
    # `UnboundLocalError` e derruba `train_rank` inteiro. Ficou latente enquanto os dois knobs eram
    # off por default, e passou a disparar SEMPRE quando o B6 adotou `feature_contri_meta: 0.6` no
    # YAML (2026-07-24) — o braço de ranking está quebrado desde então. Achado ao rodar a suíte
    # completa na campanha de polimento (2026-07-25).
    if getattr(lgb_cfg, "monotone_llr", False):
        _MONO = ("cusum_mean_pos", "cusum_var_up", "conformal_logm_abs_reset", "bayes_lo")
        base_params["monotone_constraints"] = [1 if c.startswith(_MONO) else 0 for c in feature_cols]
    _contri = _feature_contri(lgb_cfg, feature_cols)
    if _contri is not None:
        base_params["feature_contri"] = _contri

    boosters = []
    fold_evals = []
    oof_pred = np.full(len(rows), np.nan, dtype=np.float64)
    folds = list(grouped_stratified_kfold(rows, lgb_cfg.n_folds, cfg.seed))
    fold_iter = tqdm(folds, desc="treinando folds (rank)") if progress else folds

    for train_idx, valid_idx in fold_iter:
        # contiguidade por grupo: lambdarank exige que as linhas do mesmo t estejam adjacentes,
        # com `group` = contagens por t NESSA ordem (armadilha do objetivo de ranking do LightGBM).
        train_sorted = train_idx[np.argsort(t_full[train_idx], kind="stable")]
        valid_sorted = valid_idx[np.argsort(t_full[valid_idx], kind="stable")]

        _, train_group = np.unique(t_full[train_sorted], return_counts=True)
        _, valid_group = np.unique(t_full[valid_sorted], return_counts=True)

        # truncation_level idealmente cobriria o maior grupo por inteiro (o default do LightGBM, 30,
        # daria gradiente só ao topo de cada grupo, péssimo para uma AUC que depende de TODOS os
        # pares, parecer §6-R3) -- mas t<=100 mantém ~10000 séries vivas (thinning só começa depois,
        # configs/default.yaml:thinning), então o maior grupo de um fold chega a ~8000 linhas: sem
        # cap, o custo por grupo (~group_size*truncation_level) trava o treino (medido: >4h sem
        # terminar). `truncation_level_cap` (rank.truncation_level_cap) limita isso a um valor
        # tratável -- grupos maiores que o cap ficam com gradiente pleno só no topo, risco aceito
        # por tratabilidade computacional.
        truncation_level = min(int(train_group.max()), rank_cfg.truncation_level_cap)
        params = dict(base_params, lambdarank_truncation_level=truncation_level)
        params_use = params  # o braço de ranking não usa o objetivo pairwise (B4 é só binário)

        dtrain = lgb.Dataset(
            X_full[train_sorted], label=y_full[train_sorted], weight=thin_w[train_sorted], group=train_group
        )
        dvalid = lgb.Dataset(
            X_full[valid_sorted],
            label=y_full[valid_sorted],
            weight=thin_w[valid_sorted],
            group=valid_group,
            reference=dtrain,
        )
        feval = _make_fold_feval(
            t_full[valid_sorted], lgb_cfg.feval_max_valid_rows, cfg.seed, raw_to_prob=True,
            stopping_metric=lgb_cfg.early_stopping_metric,
        )
        evals_result: dict = {}
        booster = lgb.train(
            params_use,
            dtrain,
            num_boost_round=lgb_cfg.n_estimators_cap,
            valid_sets=[dvalid],
            feval=feval,
            callbacks=[
                # `early_stopping_rounds <= 0` desliga a parada antecipada e treina exatamente
                # `n_estimators_cap` rodadas. Existe porque a parada antecipada e uma FONTE DE
                # VARIANCIA medida: o numero de arvores do mesmo fold varia 51->103 so trocando a
                # semente, e o dp de 0,0041 na TS-AUC que isso produz e maior que qualquer efeito de
                # feature ja medido no projeto (docs/BACKLOG_TSAUC.md, "O nulo da regra de decisao").
                # Rodadas fixas trocam um pouco de vies por muito menos ruido de sorteio.
                *([lgb.early_stopping(lgb_cfg.early_stopping_rounds, first_metric_only=True, verbose=False)]
                  if lgb_cfg.early_stopping_rounds > 0 else []),
                lgb.log_evaluation(period=0),
                lgb.record_evaluation(evals_result),
            ],
        )
        boosters.append(booster)
        fold_evals.append(evals_result)

        raw_valid = booster.predict(X_full[valid_sorted], raw_score=True)
        oof_pred[valid_sorted] = 1.0 / (1.0 + np.exp(-raw_valid))

    ensemble = RankModelEnsemble(
        boosters=boosters,
        feature_order=tuple(feature_cols),
        predict_num_threads=lgb_cfg.predict_num_threads,
        fold_evals=fold_evals,
    )
    return ensemble, oof_pred
