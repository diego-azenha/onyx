"""Fusão de K boosters LightGBM num único modelo — bagging de sementes sem pagar K vezes o
overhead de chamada.

Vive em `src/sbrt/` (e não só em `scripts/`) porque o caminho de treino da PLATAFORMA precisa dela:
`adapter/platform.py:train()` roda na nuvem e treina do zero, então o bagging tem de acontecer lá —
o artefato local em `resources/` não é usado pela submissão.

## Por que fundir em vez de guardar K boosters

A predição raw do LightGBM é uma **soma sobre árvores**. Concatenar as árvores de K modelos e dividir
os valores de folha por K produz um modelo cuja saída raw é exatamente a **média dos raws** dos K
originais. Medido: `max |raw_fundido - media_raws| = 3,6e-15`.

Isso importa porque a inferência é causal, uma linha por passo, e cada chamada de `Booster.predict`
custa ~60 µs **quase independentemente do tamanho do modelo** (medido: 1 chamada 56 µs, 20 chamadas
68,8 µs cada — é overhead de API, não travessia de árvore). Guardar 20 boosters custaria 1362 µs/passo;
o fundido custa 238 µs.

## A mudança de semântica

`ModelEnsemble.predict_one` faz média de **probabilidades**: `mean(sigmoid(raw_i))`. O fundido dá
`sigmoid(mean(raw_i))`. São funções diferentes e podem ordenar duas séries de forma diferente.
MEDIDO sobre 4 sementes em OOF (2,5M linhas): TS-AUC **0,6118 nas duas**, Spearman **0,999994**.
Indistinguíveis para a métrica. Refazer a medição se K ou o objetivo mudarem.
"""
from __future__ import annotations

import re

import lightgbm as lgb


def _split(model_str: str) -> tuple[str, str, str]:
    i = model_str.index("\nTree=0\n")
    j = model_str.index("\nend of trees\n")
    return model_str[: i + 1], model_str[i + 1 : j + 1], model_str[j + 1 :]


def _scale_leaves(block: str, k: int) -> str:
    def repl(m: "re.Match") -> str:
        vals = [float(v) / k for v in m.group(1).split()]
        return "leaf_value=" + " ".join(repr(v) for v in vals)

    return re.sub(r"leaf_value=([^\n]+)", repl, block)


def fuse_boosters(boosters: list) -> lgb.Booster:
    """Concatena as árvores de todos os boosters, com folhas divididas por len(boosters)."""
    if len(boosters) == 1:
        return boosters[0]
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
    # para fatiar o arquivo, e um valor errado corrompe a leitura silenciosamente.
    sizes = " ".join(str(len(t.encode("utf-8"))) for t in trees)
    head = re.sub(r"^tree_sizes=.*$", f"tree_sizes={sizes}", head0, count=1, flags=re.M)
    return lgb.Booster(model_str=head + "".join(trees) + tail0)
