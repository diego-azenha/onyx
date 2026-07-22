#!/usr/bin/env python
"""Funde os boosters de N sementes num ÚNICO modelo LightGBM — bagging de sementes sem pagar N vezes
o overhead de chamada.

## Por que existe

Bagging de sementes vale ~+0,005 de TS-AUC (docs/BACKLOG_TSAUC.md, "Bagging de sementes"), mas
empacotar 4 sementes × 5 folds = 20 boosters custa caro na inferência causal, que prediz UMA linha
por passo. Medido: cada chamada de `Booster.predict` sobre uma linha custa ~60 µs **quase
independentemente do tamanho do modelo** (1 chamada 56 µs, 20 chamadas 68,8 µs cada) — é overhead
fixo da API, não travessia de árvore. Resultado: 20 chamadas = 1362 µs, mais da metade do orçamento.

## Como funciona

A predição raw do LightGBM é uma **soma sobre árvores**. Logo, concatenar as árvores de K modelos e
dividir os valores de folha por K produz um modelo cuja saída raw é exatamente a **média dos raws**
dos K originais. Medido: `max |raw_fundido − média_raws| = 1,0e-15` (arredondamento de float64).

Ganho medido: 1362 µs -> 238 µs por passo (5,7×), com o modelo de 4 sementes ficando MAIS BARATO
que o de 1 semente sem fusão (1405 µs -> ~1310 µs total).

## A mudança de semântica, e por que ela é aceitável

`ModelEnsemble.predict_one` faz média de **probabilidades**: `mean(sigmoid(raw_i))`. O modelo fundido
dá `sigmoid(mean(raw_i))`. São funções diferentes e **podem** ordenar duas séries de forma diferente,
então isto é mudança de modelo, não otimização gratuita.

MEDIDO sobre as 4 sementes do mrep em OOF (2,5M linhas): TS-AUC **0,6118 nas duas**, correlação de
Spearman **0,999994**. Indistinguíveis para efeito da métrica. Refazer esta medição se o número de
sementes ou o objetivo mudar.

Nota de empacotamento: o resultado é salvo como um `ModelEnsemble` de **um** booster. Nenhuma mudança
em `model/predict.py` é necessária — `predict_one` sobre uma lista de um elemento já devolve
`sigmoid(raw)`, que é exatamente o alvo.
"""
from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np

from sbrt.model.predict import ModelEnsemble


def _split(model_str: str) -> tuple[str, str, str]:
    i = model_str.index("\nTree=0\n")
    j = model_str.index("\nend of trees\n")
    return model_str[: i + 1], model_str[i + 1 : j + 1], model_str[j + 1 :]


def _scale_leaves(block: str, k: int) -> str:
    def repl(m: re.Match) -> str:
        vals = [float(v) / k for v in m.group(1).split()]
        return "leaf_value=" + " ".join(repr(v) for v in vals)

    return re.sub(r"leaf_value=([^\n]+)", repl, block)


def fuse(boosters: list) -> lgb.Booster:
    """Concatena as árvores de todos os boosters, com folhas divididas por len(boosters)."""
    k = len(boosters)
    head0, _, tail0 = _split(boosters[0].model_to_string())

    trees: list[str] = []
    for b in boosters:
        _, body, _ = _split(b.model_to_string())
        for blk in re.split(r"(?=^Tree=\d+$)", body, flags=re.M):
            if not blk.strip():
                continue
            blk = re.sub(r"^Tree=\d+$", f"Tree={len(trees)}", blk, count=1, flags=re.M)
            trees.append(_scale_leaves(blk, k))

    # `tree_sizes` tem de bater com o tamanho em BYTES de cada bloco: o carregador do LightGBM o usa
    # para fatiar o arquivo em paralelo, e um valor errado corrompe a leitura silenciosamente.
    sizes = " ".join(str(len(t.encode("utf-8"))) for t in trees)
    head = re.sub(r"^tree_sizes=.*$", f"tree_sizes={sizes}", head0, count=1, flags=re.M)
    return lgb.Booster(model_str=head + "".join(trees) + tail0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="dirs de ensembles (uma semente cada)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--backup", default=None, help="copia o --out atual para cá antes de sobrescrever")
    parser.add_argument("--tol", type=float, default=1e-9,
                        help="tolerância da verificação raw_fundido vs média_raws")
    args = parser.parse_args()

    ens = [ModelEnsemble.load(p) for p in args.inputs]
    ordem = ens[0].feature_order
    for p, e in zip(args.inputs, ens):
        if e.feature_order != ordem:
            raise SystemExit(f"{p}: ordem de colunas diferente — as sementes têm de vir da MESMA build")
        if e.base_rate_curve != ens[0].base_rate_curve:
            raise SystemExit(f"{p}: curva de taxa-base diferente")

    boosters = [b for e in ens for b in e.boosters]
    print(f"{len(ens)} sementes × {len(ens[0].boosters)} folds = {len(boosters)} boosters, {len(ordem)} features")
    fused = fuse(boosters)
    print(f"fundido: {fused.num_trees()} árvores")

    # Verificação numérica — é ela, não o raciocínio acima, que autoriza empacotar isto.
    rng = np.random.RandomState(0)
    X = rng.randn(64, len(ordem))
    esperado = np.column_stack([b.predict(X, raw_score=True, num_threads=1) for b in boosters]).mean(axis=1)
    obtido = fused.predict(X, raw_score=True, num_threads=1)
    err = float(np.abs(esperado - obtido).max())
    # sem sinal unicode de menos nem acento: o console do Windows usa cp1252 e quebraria aqui
    print(f"max |raw_fundido - media_raws| = {err:.3e}  (tolerancia {args.tol:.0e})")
    if err > args.tol:
        raise SystemExit("FUSÃO INVÁLIDA — não empacotar")

    if args.backup:
        bak = Path(args.backup)
        if not bak.exists() and Path(args.out).exists():
            shutil.copytree(args.out, bak)
            print(f"backup do artefato anterior em {bak}")

    ModelEnsemble(
        boosters=[fused],
        feature_order=ordem,
        predict_num_threads=ens[0].predict_num_threads,
        fold_evals=[fe for e in ens for fe in e.fold_evals],
        base_rate_curve=ens[0].base_rate_curve,
    ).save(args.out)
    print(f"salvo em {args.out} (1 booster de {fused.num_trees()} árvores)")


if __name__ == "__main__":
    main()
