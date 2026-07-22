#!/usr/bin/env python
"""Rastreio de redundância transversal: uma coluna candidata pode MOVER a TS-AUC?

## Por que este script existe

Custa ~30 min de build + ~20 min de treino descobrir por R0 que uma família nova é neutra. Este
rastreio custa minutos e responde antes, porque a TS-AUC tem uma propriedade que permite decidir sem
treinar: ela só enxerga a **ordenação dentro de cada passo t** (invariância C1). Logo:

- se a coluna nova tem correlação ~1 com uma coluna que já existe **dentro do passo**, ela não pode
  reordenar nada -> Delta esperado ~0, por mais bem motivada que seja a teoria por trás dela;
- e qualquer componente **comum a todas as séries** naquele passo (um deslocamento igual para todas)
  é exatamente neutro -- some na ordenação.

O segundo ponto é sutil e já custou uma previsão errada neste projeto: os log-martingales conformais
derivam linearmente em t, e a leitura inicial foi que o nível seria "escala idiossincrática por
série". Medindo: a deriva é -0,347/passo com desvio ENTRE séries de 0,004 (CV de 1,2%) -- ela é
praticamente universal, porque sob H0 o incremento vale log(eps)-eps+1, que depende só da grade de
eps. Deriva comum = deslocamento comum = neutro por C1. Calibrá-la não compra ordenação.

## Como ler a saída

`corr_xs` é a correlação média entre a coluna crua e a calibrada DENTRO do passo, sobre um painel de
séries heterogêneas. Perto de 1 = a calibrada reproduz a ordenação da crua e tende a ser redundante.

Isto é um RASTREIO, não um veredito: uma correlação de 0,95 ainda deixa 5% de variação residual, e
uma árvore pode usá-la se esse resíduo cair na região certa. O que o rastreio faz é ordenar o que
vale gastar um ciclo de R0 medindo -- e evitar gastar em quem não pode nem em princípio ajudar.

**O painel precisa ser heterogêneo.** Rodar isto sobre séries i.i.d. mede quase nada: se todas as
séries têm o mesmo nulo, calibrar vira uma transformação afim com as MESMAS constantes, a ordenação
não muda por construção, e a correlação dá ~1 mesmo para uma calibração excelente. O default usa os
controles (sem quebra) dos cenários de robustez, que cobrem GARCH, dependência AR, cauda pesada e
i.i.d.
"""
from __future__ import annotations

import argparse

import numpy as np

from sbrt.config import DEFAULT_CONFIG_PATH, load_config
from sbrt.robustness.generators import generate
from sbrt.state.h0 import fit_h0
from sbrt.state.scorer import StreamScorer, default_blocks

# Blocos que existem mas NÃO estão em `default_blocks()` — o rastreio precisa poder instanciá-los
# antes de qualquer decisão de ligar, que é justamente a ordem que a disciplina de R0 exige
# (rastrear primeiro, medir depois, ligar por último).
EXTRA_BLOCKS = {
    "spectral": ("sbrt.state.spectral", "SpectralBlock"),
    "ordinal": ("sbrt.state.ordinal", "OrdinalBlock"),
    "multirep": ("sbrt.state.multirep", "MultiRepBlock"),
    "bocpd": ("sbrt.state.bocpd", "BOCPDBlock"),
}
# `TrajectoryBlock` fica de fora de propósito: ele usa o SEGUNDO contrato
# (`update_from_feats(feats, t)`, docs/NOTAS_AGENTES.md §2.1) e não roda neste laço.

# controles (sufixo _ctrl = mesmo gerador, SEM a quebra): o rastreio é sobre a escala do NULO, então
# o painel tem de ser H0. Escolhidos por cobrirem dinâmicas bem distintas entre si.
DEFAULT_PANEL = ("t6_ctrl", "t7_ctrl", "t8_ctrl", "t9_ctrl", "t1_ctrl", "t3_ctrl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--panel", nargs="*", default=list(DEFAULT_PANEL))
    parser.add_argument("--seeds", type=int, default=9)
    parser.add_argument("--t-max", type=int, default=120)
    parser.add_argument("--at-steps", type=int, nargs="*", default=[30, 60, 120])
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="pares 'cru:calibrado'; default = tudo em calibration.recursive_features")
    parser.add_argument("--vs-all", nargs="*", default=None, metavar="PREFIXO",
                        help="modo família nova: para cada coluna com estes prefixos, reporta a MAIOR "
                             "correlação transversal contra QUALQUER coluna já existente. É o rastreio "
                             "certo para um eixo novo — a pergunta não é 'a versão calibrada reordena?' "
                             "e sim 'esta direção é nova ou já está coberta?'")
    parser.add_argument("--extra-blocks", nargs="*", default=[], choices=sorted(EXTRA_BLOCKS),
                        help="blocos fora de default_blocks() a acrescentar só para este rastreio")
    args = parser.parse_args()

    cfg = load_config(args.config)
    extra = []
    for name in args.extra_blocks:
        mod, cls = EXTRA_BLOCKS[name]
        extra.append(getattr(__import__(mod, fromlist=[cls]), cls)())
    if args.pairs:
        pairs = [tuple(p.split(":", 1)) for p in args.pairs]
    else:
        pairs = [(n, f"{n}_cal") for n, _ in cfg.calibration.recursive_features]

    by_t: dict = {}
    n_series = 0
    for s in range(args.seeds):
        for scenario in args.panel:
            hist, online, _ = generate(scenario, seed=900 + s, cfg=cfg)
            blocks = default_blocks() + [type(b)() for b in extra]
            scorer = StreamScorer(fit_h0(hist, cfg), blocks, None, cfg)
            n_series += 1
            for t, x in enumerate(online[: args.t_max], start=1):
                by_t.setdefault(t, []).append(scorer.update_features(float(x)))

    print(f"\npainel: {n_series} séries H0 de {len(args.panel)} dinâmicas distintas "
          f"({', '.join(args.panel)})")
    print(f"passos avaliados: {args.at_steps}\n")

    if args.vs_all:
        names = sorted(by_t[args.at_steps[-1]][0])
        novos = [n for n in names if n.startswith(tuple(args.vs_all))]
        antigos = [n for n in names if n not in set(novos)]
        print(f"{'coluna nova':34s} {'max|corr| vs existentes':>24s}  {'mais parecida com':28s} veredito")
        print("-" * 110)
        for novo in novos:
            best_c, best_n = 0.0, "—"
            for t in args.at_steps:
                rows = by_t.get(t, [])
                a = np.array([r.get(novo, np.nan) for r in rows], dtype=np.float64)
                if np.isfinite(a).sum() < 10 or np.nanstd(a) < 1e-12:
                    continue
                for velho in antigos:
                    b = np.array([r.get(velho, np.nan) for r in rows], dtype=np.float64)
                    m = np.isfinite(a) & np.isfinite(b)
                    if m.sum() < 10 or np.std(b[m]) < 1e-12 or np.std(a[m]) < 1e-12:
                        continue
                    c = abs(float(np.corrcoef(a[m], b[m])[0, 1]))
                    if c > best_c:
                        best_c, best_n = c, velho
            verdict = ("EIXO NOVO — vale medir" if best_c < 0.90 else
                       "parcialmente coberta" if best_c < 0.97 else
                       "JÁ COBERTA — não adicionar")
            print(f"{novo:34s} {best_c:24.3f}  {best_n:28s} {verdict}")
        print("\nLembrete: correlação alta contra uma coluna existente significa que a direção já está\n"
              "no banco — adicionar só acrescenta largura. Baixa significa direção nova, não que vai\n"
              "ganhar. O juiz continua sendo compare_oof.py.")
        return
    header = f"{'coluna':34s} {'corr_xs':>9s} {'dp_xs cru':>11s} {'dp_xs cal':>11s}  veredito"
    print(header)
    print("-" * (len(header) + 20))

    for raw, cal in pairs:
        corrs, sd_raw, sd_cal = [], [], []
        for t in args.at_steps:
            rows = by_t.get(t, [])
            a = np.array([r.get(raw, np.nan) for r in rows], dtype=np.float64)
            b = np.array([r.get(cal, np.nan) for r in rows], dtype=np.float64)
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 10 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
                continue
            corrs.append(abs(float(np.corrcoef(a[m], b[m])[0, 1])))
            sd_raw.append(float(np.std(a[m])))
            sd_cal.append(float(np.std(b[m])))
        if not corrs:
            print(f"{raw:34s} {'—':>9s} {'—':>11s} {'—':>11s}  indisponível no painel")
            continue
        c = float(np.mean(corrs))
        verdict = ("REORDENA — vale medir" if c < 0.90 else
                   "marginal" if c < 0.97 else
                   "redundante — Delta esperado ~0")
        print(f"{raw:34s} {c:9.3f} {np.mean(sd_raw):11.3f} {np.mean(sd_cal):11.3f}  {verdict}")

    print("\nLembrete: rastreio, não veredito. Correlação alta diz que a coluna dificilmente move a\n"
          "ordenação transversal; correlação baixa diz que vale gastar um R0 medindo — não que vai\n"
          "ganhar. O juiz continua sendo compare_oof.py com IC 95% pareado.")


if __name__ == "__main__":
    main()
