# Passo 2 — Professor e aluno (rótulos de detectabilidade por linha)

**Frente:** [roteiro 30/09](README.md) · **Status:** hipótese registrada · **Hipótese:** 2026-09-30 11:30

## Por quê

O Onyx é limitado por quebras ([C1](../teto-offline/C1-curva-aprendizado-onyx.md)), e parte dessa
escassez vem de **rótulos ruidosos**. Uma linha "quebrou" dois passos depois de τ, ou numa quebra
invisível, não carrega informação, e o modelo gasta eventos tentando ajustá-la. O X1-soft (peso por
série segundo a detectabilidade) rendeu +0,004. O professor é a versão por **linha**.

## Desenho

**Professor** (`scripts/p2_professor.py`), com os folds do Onyx (`grouped_stratified_kfold`, `cfg.seed`):
- linhas numa grade m ∈ {2,3,5,8,12,20,30,50,80,120,200,300,500,800}, limitada ao pós disponível;
  negativos com pseudo-τ ~ U[0, len−1] e a mesma grade;
- features privilegiadas: as 118 de `src/offline/features.py` sobre [τ, τ+m) (prefixo `a_`), as 118
  sobre o pós **inteiro**, o futuro (prefixo `f_`), e log m;
- alvo: positivo × negativo. A probabilidade = chance de a quebra já ser perceptível em m;
- **nested cross-fit:** para cada fold f do aluno, o professor é ajustado só nas séries fora de f, com
  CV interna de 4 partes (por série) gerando a nota dessas séries. As séries de f nunca passam pelo
  professor que rotula o treino de f. Saída: `y_soft_f{f}` por linha do `train_rows`, interpolada em
  log m (m=1 recebe o valor de m=2). Linhas pré-quebra e negativos ficam com 0.

**Aluno:** `LightGBMConfig.soft_label` (default `False`). Ligado, o fold k treina com
`0,5·y + 0,5·y_soft_fk` e `objective="cross_entropy"`. Pesos, curva de taxa-base, feval e OOF continuam
no `y` verdadeiro. Receita do E1.

## Hipótese e regras (escritas ANTES de medir)

- Δ contra o E1: **+0,004 a +0,012**.
- **Adotar** se ≥ +0,005 com IC excluindo 0 na partição 42 (K=4) **e** na 43 (K=2). **Descartar** se
  ≤ +0,002.
- Uma mistura principal (0,5/0,5); outras seriam exploratórias.
- Sanidade: com `soft_label=False`, o treino é o E1 exato (no-op). Professor com AUC por linha bem
  acima do aluno (ele conhece τ e o futuro).
- Limite declarado: mesmo aprovado hoje, a produção precisa do professor dentro do `train()` da
  nuvem. É trabalho separado, não incluído neste passo.

## Resultado

(pendente)
