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


def fuse_boosters(boosters: list, verify: bool = True, tol: float = 1e-9) -> lgb.Booster:
    """Concatena as árvores de todos os boosters, com folhas divididas por len(boosters).

    `verify=True` (padrão) confere numericamente que o raw do fundido é a média dos raws dos
    originais, e levanta `RuntimeError` se não for. **Isto não é zelo excessivo:** a fusão reescreve
    a linha `tree_sizes` por regex, e se o padrão não casar (outra versão do LightGBM, outro formato)
    o `re.sub` devolve a string INALTERADA, sem erro — o modelo resultante carrega e prediz coisa
    errada em silêncio. Este caminho roda na nuvem, onde ninguém está olhando; falhar alto é
    infinitamente melhor que submeter um modelo corrompido."""
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
    head, n_sub = re.subn(r"^tree_sizes=.*$", f"tree_sizes={sizes}", head0, count=1, flags=re.M)
    if n_sub != 1:
        raise RuntimeError(
            "fuse_boosters: linha `tree_sizes=` não encontrada no modelo do LightGBM. O formato "
            "mudou; a fusão não é segura. Trate os boosters sem fundir (K chamadas de predict)."
        )
    fused = lgb.Booster(model_str=head + "".join(trees) + tail0)

    if verify:
        import numpy as np

        n_feat = boosters[0].num_feature()
        x = np.linspace(-3.0, 3.0, 32 * n_feat, dtype=np.float64).reshape(32, n_feat)
        esperado = np.column_stack(
            [b.predict(x, raw_score=True, num_threads=1) for b in boosters]
        ).mean(axis=1)
        obtido = fused.predict(x, raw_score=True, num_threads=1)
        err = float(np.abs(esperado - obtido).max())
        if not (err <= tol):
            raise RuntimeError(
                f"fuse_boosters: fusão INVÁLIDA (max |raw_fundido - media_raws| = {err:.3e} > {tol:.0e}). "
                "O modelo fundido NÃO representa a média dos originais — não usar."
            )
    return fused
