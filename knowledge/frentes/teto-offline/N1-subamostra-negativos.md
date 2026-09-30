# N1 — Subamostrar negativos no treino

**Frente:** [teto offline](README.md) · **Status:** **refutado** (medido sobre o E1, como V2) · **Hipótese:** 2026-09-30 02:12

## Por quê

No [C2](C2-positivos-ou-negativos.md), treinar com todos os positivos e metade dos negativos deu 0,6299,
contra 0,6273 com tudo, nas mesmas séries avaliadas.

## Desenho

`scripts/n1_treino.py`: wrapper do `train.py` (monkeypatch de `grouped_stratified_kfold`). Os folds
são os do Onyx; em cada fold, o **treino** exclui as séries negativas sorteadas fora (fração `f`,
sorteio determinístico por hash do id), e a **validação** fica intacta. O OOF sai para as 10k séries.
Receita do B0, 4 sementes, R0 contra o B0.

## Hipótese (escrita ANTES de medir)

- Com f=0,5: Δ entre **0,000 e +0,004** contra o B0. O efeito do C2 (+0,0026) veio de 2 sementes, sem
  IC, e pode ser ruído.
- Mecanismo candidato: os pesos do R1 equalizam as classes por passo, mas com o dobro de negativos
  as árvores gastam splits modelando a variedade de H0, que é irrelevante para a ordenação.
- Regra: IC excluindo 0 → adotar e testar f=0,33.

## Resultado

Medido sobre o E1 ([V2](V-empilhamento.md)): **0,6262 contra 0,6349, Δ −0,0088 [−0,0121; −0,0055]**. Refutado. O +0,0026 do `meio_neg` no C2 veio de 2 sementes e de um desenho diferente, com os negativos removidos também da avaliação.
